// src/buffer/linear/api.rs

use super::{AsGpuBuffer, GpuOnly, LocationInfo};
use crate::Result;
use crate::{Device, GpuBuffer};

use ash::vk;

use std::marker::PhantomData;
use std::sync::atomic::AtomicU64;

pub enum BufferSource<'a, T> {
    Data(&'a [T]),
    Count(usize),
}

pub trait BufferInit<'a, T> {
    fn into_source(self) -> BufferSource<'a, T>;
}

/// Builder thingy for `GpuBuffer`.
pub struct BufferBuilder<'a, T, Loc = GpuOnly> {
    source: BufferSource<'a, T>,
    name: &'a str,
    _marker: PhantomData<Loc>,
}

// ----------------------------------------------------------------
// Buffer Source Impl
// ----------------------------------------------------------------

impl<'a, T> BufferInit<'a, T> for usize {
    fn into_source(self) -> BufferSource<'a, T> {
        BufferSource::Count(self)
    }
}

impl<'a, T> BufferInit<'a, T> for &'a [T] {
    fn into_source(self) -> BufferSource<'a, T> {
        BufferSource::Data(self)
    }
}

impl<'a, T> BufferInit<'a, T> for &'a Vec<T> {
    fn into_source(self) -> BufferSource<'a, T> {
        BufferSource::Data(self.as_slice())
    }
}

// ----------------------------------------------------------------
// Entry Point Function
// ----------------------------------------------------------------

pub fn create_buffer<'a, T>(init: impl BufferInit<'a, T>) -> BufferBuilder<'a, T, GpuOnly> {
    BufferBuilder {
        source: init.into_source(),
        name: "GpuBuffer",
        _marker: PhantomData,
    }
}

pub fn read_buffer<'a, T, Loc: LocationInfo>(
    buffer: &'a GpuBuffer<T, Loc>,
) -> BufferReadBuilder<'a, T, Loc> {
    BufferReadBuilder { buffer }
}

// ----------------------------------------------------------------
// Builder Method
// ----------------------------------------------------------------

impl<'a, T, CurrentLoc: LocationInfo> BufferBuilder<'a, T, CurrentLoc> {
    pub fn named(mut self, name: &'a str) -> Self {
        self.name = name;
        self
    }

    pub fn on<NewLoc>(self, _location: NewLoc) -> BufferBuilder<'a, T, NewLoc> {
        BufferBuilder {
            source: self.source,
            name: self.name,
            _marker: PhantomData,
        }
    }

    pub fn with(self, device: &Device) -> Result<GpuBuffer<T, CurrentLoc>> {
        let (size, _initial_data) = match self.source {
            BufferSource::Data(slice) => {
                ((slice.len() * std::mem::size_of::<T>()) as u64, Some(slice))
            }
            BufferSource::Count(count) => ((count * std::mem::size_of::<T>()) as u64, None),
        };

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS // BDA
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            device
                .shared
                .core
                .logical_device
                .create_buffer(&buffer_info, None)?
        };

        let requirements = unsafe {
            device
                .shared
                .core
                .logical_device
                .get_buffer_memory_requirements(buffer)
        };

        let desc = CurrentLoc::allocation_desc(self.name, requirements);

        let allocation = device.shared.allocator.allocate(desc)?;

        unsafe {
            device.shared.core.logical_device.bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )?;
        }

        match self.source {
            BufferSource::Data(slice) => {
                if let Some(mapped_ptr) = allocation.mapped_ptr() {
                    // ReBAR/UMA/HostVisible: Direct memcpy
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            slice.as_ptr() as *const u8,
                            mapped_ptr.as_ptr() as *mut u8,
                            size as usize,
                        );
                    }
                } else {
                    // GPU-Only: Staging Buffer + vkCmdCopyBuffer
                    device.immediate_upload(buffer, slice)?;
                }
            }
            BufferSource::Count(_) => {}
        }

        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe {
            device
                .shared
                .core
                .logical_device
                .get_buffer_device_address(&addr_info)
        };

        Ok(GpuBuffer::new(
            buffer,
            Some(allocation),
            AtomicU64::new(0),
            device_address,
            size,
            device.shared.clone(),
            PhantomData,
        ))
    }
}

// ----------------------------------------------------------------
// Builder Method - Read
// ----------------------------------------------------------------

pub struct BufferReadBuilder<'a, T, Loc: LocationInfo> {
    buffer: &'a GpuBuffer<T, Loc>,
}

pub struct BufferReadIntoBuilder<'a, 'd, T, Loc: LocationInfo> {
    buffer: &'a GpuBuffer<T, Loc>,
    dest: &'d mut [T],
}

fn read_buffer_internal<T: Copy, Loc: LocationInfo>(
    buffer: &GpuBuffer<T, Loc>,
    dest: &mut [T],
    device: &Device,
) -> Result<()> {
    let size = buffer.size() as usize;

    if let Some(alloc) = buffer.allocation.as_ref() {
        if let Some(mapped_ptr) = alloc.mapped_ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped_ptr.as_ptr() as *const u8,
                    dest.as_mut_ptr() as *mut u8,
                    size,
                );
            }
            return Ok(());
        }
    }

    device.immediate_download(buffer.vk_buffer(), dest)?;

    Ok(())
}

impl<'a, T: Copy, Loc: LocationInfo> BufferReadBuilder<'a, T, Loc> {
    pub fn into<'d>(self, dest: &'d mut [T]) -> BufferReadIntoBuilder<'a, 'd, T, Loc> {
        BufferReadIntoBuilder {
            buffer: self.buffer,
            dest,
        }
    }

    pub fn with(self, device: &Device) -> Result<Vec<T>> {
        let size = self.buffer.size() as usize;
        let count = size / std::mem::size_of::<T>();

        let mut vec = Vec::with_capacity(count);
        unsafe {
            vec.set_len(count);
        }

        read_buffer_internal(self.buffer, &mut vec, device)?;

        Ok(vec)
    }
}

impl<'a, 'd, T: Copy, Loc: LocationInfo> BufferReadIntoBuilder<'a, 'd, T, Loc> {
    pub fn with(self, device: &Device) -> Result<()> {
        let dest_size = self.dest.len() * std::mem::size_of::<T>();
        let src_size = self.buffer.size() as usize;

        assert!(
            dest_size >= src_size,
            "Destination slice is too small! (Src: {} bytes, Dest: {} bytes)",
            src_size,
            dest_size
        );

        // Exactly Gpu buffer size
        let exact_dest = &mut self.dest[0..(src_size / std::mem::size_of::<T>())];
        read_buffer_internal(self.buffer, exact_dest, device)
    }
}

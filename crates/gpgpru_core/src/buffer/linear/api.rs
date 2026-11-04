use super::*;
use std::marker::PhantomData;

// ----------------------------------------------------------------
// Buffer Source & Init
// ----------------------------------------------------------------

pub enum BufferSource<'a, T> {
    Data(&'a [T]),
    Count(usize),
}

pub trait BufferInit<'a, T> {
    fn into_source(self) -> BufferSource<'a, T>;
}

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
// Builder Definition
// ----------------------------------------------------------------

pub struct BufferBuilder<'a, T, const LOC: u8 = { GpuOnly.to_u8() }> {
    source: BufferSource<'a, T>,
    name: &'a str,
    _marker: PhantomData<T>,
}

// ----------------------------------------------------------------
// Entry Point Functions
// ----------------------------------------------------------------

pub fn create_buffer<'a, T>(
    init: impl BufferInit<'a, T>,
) -> BufferBuilder<'a, T, { GpuOnly.to_u8() }> {
    BufferBuilder {
        source: init.into_source(),
        name: "GpuBuffer",
        _marker: PhantomData,
    }
}

pub fn read_buffer<'a, T>(slice: &'a GpuSlice<T>) -> BufferReadBuilder<'a, T> {
    BufferReadBuilder { slice }
}

pub fn write_buffer<'a, 'b, T>(slice: &'a mut GpuSlice<'b, T>) -> BufferWriteBuilder<'a, 'b, T> {
    BufferWriteBuilder { slice }
}

// ----------------------------------------------------------------
// BufferBuilder Implementation
// ----------------------------------------------------------------

impl<'a, T: Copy, const LOC: u8> BufferBuilder<'a, T, LOC> {
    pub fn named(mut self, name: &'a str) -> Self {
        self.name = name;
        self
    }

    pub fn on<const NEW_LOC: u8>(
        self,
        _location: LocationTag<NEW_LOC>,
    ) -> BufferBuilder<'a, T, NEW_LOC> {
        BufferBuilder {
            source: self.source,
            name: self.name,
            _marker: PhantomData,
        }
    }

    pub fn with(self, device: &Device) -> Result<GpuBuffer<T, LOC>> {
        let size = match self.source {
            BufferSource::Data(slice) => (slice.len() * std::mem::size_of::<T>()) as u64,
            BufferSource::Count(count) => (count * std::mem::size_of::<T>()) as u64,
        };

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
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

        let desc = MemoryLocation::from_u8(LOC).allocation_desc(self.name, requirements);

        let allocation = device.shared.allocator.allocate(desc)?;

        unsafe {
            device.shared.core.logical_device.bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )?;
        }

        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe {
            device
                .shared
                .core
                .logical_device
                .get_buffer_device_address(&addr_info)
        };

        let mut gpu_buf = GpuBuffer::new(
            buffer,
            Some(allocation),
            device_address,
            size,
            device.shared.clone(),
        );

        if let BufferSource::Data(slice) = self.source {
            write_buffer_internal(&mut gpu_buf, slice, device)?;
        }

        Ok(gpu_buf)
    }
}

// ----------------------------------------------------------------
// BufferReadBuilder Implementation
// ----------------------------------------------------------------

pub struct BufferReadBuilder<'a, T> {
    slice: &'a GpuSlice<'a, T>,
}

pub struct BufferReadIntoBuilder<'a, 'd, T> {
    slice: &'a GpuSlice<'a, T>,
    dest: &'d mut [T],
}

fn read_buffer_internal<T: Copy>(
    slice: &GpuSlice<T>,
    dest: &mut [T],
    device: &Device,
) -> Result<()> {
    let last_written_epoch = device
        .shared
        .page_tracker
        .get_last_write_epoch(slice.bda(), slice.size());

    if last_written_epoch > 0 {
        device.wait_timeline(last_written_epoch)?;
    }

    // 1. Host-Visible: Mapped Pointer Direct Copy
    if let Some(mapped) = slice.try_as_mapped_slice() {
        let count = dest.len().min(mapped.len());
        dest[..count].copy_from_slice(&mapped[..count]);
        return Ok(());
    }

    // 2. GPU-Only: Staging Buffer -> DMA Copy Task
    let mut staging_buffer = create_buffer(dest.len()).on(GpuToCpu).with(device)?;
    let copy_epoch = dma::copy(slice, &mut staging_buffer).with(device)?;

    device.wait_timeline(copy_epoch)?;

    if let Some(mapped) = staging_buffer.try_as_mapped_slice() {
        let count = dest.len().min(mapped.len());
        dest[..count].copy_from_slice(&mapped[..count]);
        return Ok(());
    }

    Err(GpuError::MemoryMappingFailed)
}

impl<'a, T: Copy> BufferReadBuilder<'a, T> {
    pub fn into<'d>(self, dest: &'d mut [T]) -> BufferReadIntoBuilder<'a, 'd, T> {
        BufferReadIntoBuilder {
            slice: self.slice,
            dest,
        }
    }

    pub fn with(self, device: &Device) -> Result<Vec<T>> {
        let size = self.slice.size() as usize;
        let count = size / std::mem::size_of::<T>();

        let mut vec = Vec::with_capacity(count);
        unsafe {
            vec.set_len(count);
        }

        read_buffer_internal(self.slice, &mut vec, device)?;

        Ok(vec)
    }
}

impl<'a, 'd, T: Copy> BufferReadIntoBuilder<'a, 'd, T> {
    pub fn with(self, device: &Device) -> Result<()> {
        let dest_size = self.dest.len() * std::mem::size_of::<T>();
        let src_size = self.slice.size() as usize;

        assert!(
            dest_size >= src_size,
            "Destination slice is too small! (Src: {} bytes, Dest: {} bytes)",
            src_size,
            dest_size
        );

        let exact_dest = &mut self.dest[0..(src_size / std::mem::size_of::<T>())];
        read_buffer_internal(self.slice, exact_dest, device)
    }
}

// ----------------------------------------------------------------
// BufferWriteBuilder Implementation
// ----------------------------------------------------------------

pub struct BufferWriteBuilder<'a, 'b, T> {
    slice: &'a mut GpuSlice<'b, T>,
}

pub struct BufferWriteFromBuilder<'a, 'b, 's, T> {
    slice: &'a mut GpuSlice<'b, T>,
    src: &'s [T],
}

fn write_buffer_internal<T: Copy>(
    slice: &mut GpuSlice<T>,
    src: &[T],
    device: &Device,
) -> Result<()> {
    let size = src.len() * std::mem::size_of::<T>();

    // 1. ReBAR / UMA / HostVisible: Mapped Pointer Direct Copy
    if let Some(mapped) = slice.try_as_mut_slice() {
        let count = src.len().min(mapped.len());
        mapped[..count].copy_from_slice(&src[..count]);
        return Ok(());
    }

    // 2. GPU-Only & <= 64KB & 4-byte aligned: vkCmdUpdateBuffer (dma::update)
    if size <= 65536 && slice.offset() % 4 == 0 && size % 4 == 0 {
        dma::update(slice, src).with(device)?;
        return Ok(());
    }

    // 3. GPU-Only: Staging Buffer + DMA Copy (dma::copy)
    let staging_buffer = create_buffer(src).on(CpuToGpu).with(device)?;
    let copy_epoch = dma::copy(&staging_buffer.as_slice(), slice).with(device)?;

    device.wait_timeline(copy_epoch)?;

    Ok(())
}

impl<'a, 'b, T: Copy> BufferWriteBuilder<'a, 'b, T> {
    pub fn from<'s>(self, src: &'s [T]) -> BufferWriteFromBuilder<'a, 'b, 's, T> {
        BufferWriteFromBuilder {
            slice: self.slice,
            src,
        }
    }
}

impl<'a, 'b, 's, T: Copy> BufferWriteFromBuilder<'a, 'b, 's, T> {
    pub fn with(self, device: &Device) -> Result<()> {
        let src_size = self.src.len() * std::mem::size_of::<T>();
        let dst_size = self.slice.size() as usize;

        assert!(
            dst_size >= src_size,
            "Source slice is too large for GpuSlice! (Src: {} bytes, Dest: {} bytes)",
            src_size,
            dst_size
        );

        write_buffer_internal(self.slice, self.src, device)
    }
}

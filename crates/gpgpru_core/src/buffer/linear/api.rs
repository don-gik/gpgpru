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

/// A Swiss-army-knife utility function for creating and initializing a GPU buffer.
///
/// Yes, its generic type signature looks like a crime scene—please don't stare directly at it,
/// just read this docstring instead.
///
/// It handles both initial data and memory placement in one go.
/// By default, your data gets dropped into `GpuOnly` memory with zero extra fuss.
///
/// # Builder Options
///
/// - `.on(location)` — Tells the buffer where to live ([`GpuOnly`], [`Shared`], [`CpuToGpu`], [`GpuToCpu`], they are in prelude!).
///   Defaults to [`GpuOnly`] (fast GPU VRAM, strictly no CPU peeking).
/// - `.with(&device)` — Actually summons the buffer into existence on the given [`Device`]
///   and uploads your data if you brought any.
///
/// # Returns & Errors
///
/// Calling `.with(&device)` yields a [`Result<GpuBuffer<T>>`](crate::Result).
/// If all goes well, you get a shiny new [`GpuBuffer`]. If your GPU runs out of VRAM,
/// Vulkan throws a tantrum, or the DMA upload fails, you'll get a [`GpuError`] instead.
///
/// # Examples
///
/// ```no_run
/// # fn main() -> gpgpru::Result<()> {
/// use gpgpru::{create_buffer, Device, MemoryLocation::Shared};
///
/// let device = Device::new();
///
/// // Feed it a Vec or slice, and it quietly handles `dma::copy` behind the scenes.
/// let host_data = vec![1.0f32; 100];
/// let buf_a = create_buffer(&host_data).with(&device)?;
///
/// // Or just give it a number if you want a blank slate of 100 `u32`s.
/// let buf_b = create_buffer::<u32>(100).with(&device)?;
///
/// // Need the CPU and GPU to share custody? Use `.on(Shared)`.
/// let mut shared_buf = create_buffer(&host_data)
///     .on(Shared)
///     .with(&device)?;
/// # Ok(())
/// # }
/// ```
pub fn create_buffer<'a, T>(
    init: impl BufferInit<'a, T>,
) -> BufferBuilder<'a, T, { GpuOnly.to_u8() }> {
    BufferBuilder {
        source: init.into_source(),
        name: "GpuBuffer",
        _marker: PhantomData,
    }
}

/// Pulls data out of a [`GpuSlice`] and brings it back to CPU civilization.
///
/// Since you can't just dereference VRAM like a normal slice (unless you enjoy segfaults),
/// this builder sets up a DMA transfer from the GPU into a host slice you provide.
///
/// # Builder Options
///
/// - `.into(dst)` — Specifies the destination host slice (`&mut [T]`) where the GPU bytes will land.
/// - `.with(&device)` — Actually executes the copy on the given [`Device`].
///
/// # Returns & Errors
///
/// Calling `.with(&device)` yields a [`Result<()>`](crate::Result).
/// If the slice lengths don't match or the GPU decides not to cooperate during DMA,
/// you'll get a [`GpuError`] instead.
///
/// # Examples
///
/// ```no_run
/// # fn main() -> gpgpru::Result<()> {
/// use gpgpru::prelude::*;
/// use gpgpru::{create_buffer, read_buffer, Device};
///
/// let device = Device::new()?;
/// let gpu_buf = create_buffer(&vec![1.0f32, 2.0, 3.0, 4.0]).with(&device)?;
///
/// let mut host_dst = vec![0.0f32; 4];
/// read_buffer(&gpu_buf).into(&mut host_dst).with(&device)?;
///
/// assert_eq!(host_dst, vec![1.0, 2.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub fn read_buffer<'a, T>(slice: &'a GpuSlice<T>) -> BufferReadBuilder<'a, T> {
    BufferReadBuilder { slice }
}

/// Ships fresh data from CPU land straight into an existing [`GpuSlice`].
///
/// Use this when you already have an allocated GPU buffer and want to overwrite its
/// contents without spinning up a brand-new buffer every frame.
///
/// # Builder Options
///
/// - `.from(src)` — Specifies the source host slice (`&[T]`) to upload from.
/// - `.with(&device)` — Fires off the transfer on the given [`Device`].
///
/// # Returns & Errors
///
/// Calling `.with(&device)` yields a [`Result<()>`](crate::Result).
/// Returns a [`GpuError`] if the source and destination sizes don't line up
/// or if the underlying Vulkan transfer fails.
///
/// # Examples
///
/// ```no_run
/// # fn main() -> gpgpru::Result<()> {
/// use gpgpru::prelude::*;
/// use gpgpru::{create_buffer, write_buffer, Device};
///
/// let device = Device::new()?;
/// let mut gpu_buf = create_buffer::<f32>(4).with(&device)?;
///
/// let new_data = [10.0f32, 20.0, 30.0, 40.0];
/// write_buffer(&mut gpu_buf).from(&new_data).with(&device)?;
/// # Ok(())
/// # }
/// ```
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
        device.wait_timeline(SemaphoreValue(last_written_epoch))?;
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

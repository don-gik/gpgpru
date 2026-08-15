use ash::vk;
use gpu_allocator::vulkan::AllocationScheme;
use gpu_allocator::{MemoryLocation, vulkan::AllocationCreateDesc};

use crate::{Device, GpuBuffer, GpuError, Result};

pub fn create_buffer<T: Copy>(data: &[T]) -> BufferBuilder<'_, T> {
    BufferBuilder {
        data: Some(data),
        size: std::mem::size_of_val(data) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        location: MemoryLocation::CpuToGpu,
        _marker: std::marker::PhantomData,
    }
}

pub fn create_buffer_empty<T: Copy>(element_count: usize) -> BufferBuilder<'static, T> {
    let size = (element_count * std::mem::size_of::<T>()) as u64;
    BufferBuilder {
        data: None,
        size,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        location: MemoryLocation::CpuToGpu,
        _marker: std::marker::PhantomData,
    }
}

pub struct BufferBuilder<'a, T> {
    data: Option<&'a [T]>,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: Copy> BufferBuilder<'a, T> {
    pub fn usage(mut self, usage: vk::BufferUsageFlags) -> Self {
        self.usage |= usage;
        self
    }

    pub fn location(mut self, location: MemoryLocation) -> Self {
        self.location = location;
        self
    }

    pub fn with(self, device: &Device) -> Result<GpuBuffer> {
        if self.size == 0 {
            return Err(GpuError::InvalidBufferSize);
        }

        // 1. Vulkan Buffer Wrapper
        let buffer_info = vk::BufferCreateInfo::default()
            .size(self.size)
            .usage(self.usage);

        let buffer = unsafe { device.logical_device.create_buffer(&buffer_info, None) }?;

        // 2. Check Requirements
        let requirements = unsafe { device.logical_device.get_buffer_memory_requirements(buffer) };

        // 3. Allocate VRAM
        let allocation = device.allocate(&AllocationCreateDesc {
            name: "GpuBuffer",
            requirements,
            location: self.location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        // 4. Bind Memory
        unsafe {
            device.logical_device.bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )?;
        }

        let mut gpu_buf = GpuBuffer::new(buffer, allocation, self.size, device.get_free_sender());

        if let Some(data) = self.data {
            gpu_buf.write_data(data)?;
        }

        Ok(gpu_buf)
    }
}

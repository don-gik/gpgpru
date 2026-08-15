// src/error.rs

use ash::vk;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("Failed to load Vulkan lib.")]
    LoadFailed(#[from] ash::LoadingError),

    #[error("Vulkan Error : {0}")]
    Vulkan(#[from] vk::Result),

    #[error("No suitable physical device found for vulkan.")]
    NoSuitableDevice,

    #[error("Device doesn't support compute queue.")]
    NoComputeQueue,

    #[error("Device Out of Memory: {0:?}")]
    GpuAllocation(#[from] gpu_allocator::AllocationError),

    #[error("Allocation info is missing.")]
    NoAllocation,

    #[error("Memory is not mapped to CPU address space.")]
    NotMapped,

    #[error("Data size ({requested} bytes) exceeds allocated buffer size ({available} bytes).")]
    BufferOverflow { requested: usize, available: usize },

    #[error("The size of the buffer is invalid")]
    InvalidBufferSize,
}

pub type Result<T> = std::result::Result<T, GpuError>;

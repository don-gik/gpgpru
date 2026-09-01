// src/error.rs

use ash::vk;
use thiserror::Error;

/// Error enum for `gpgpru` crate.
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

    #[error("GPU does not support Vulkan 1.3 or higher.")]
    VulkanVersionNotSupported,

    #[error("No GPU matched the specified DeviceSelector criteria.")]
    HardwareNotSupported,

    #[error("Device Out of Memory: {0:?}")]
    GpuAllocation(#[from] gpu_allocator::AllocationError),

    #[error("Dynamic Buffer Out of Memory.")]
    OutOfDynamicMemory,

    #[error("Failed to Map Memory into a Pointer.")]
    MemoryMappingFailed,

    #[error("Allocation info is missing.")]
    NoAllocation,

    #[error("Memory is not mapped to CPU address space.")]
    NotMapped,

    #[error("Data size ({requested} bytes) exceeds allocated buffer size ({available} bytes).")]
    BufferOverflow { requested: usize, available: usize },

    #[error("The size of the buffer is invalid")]
    InvalidBufferSize,

    #[error("SPIR-V code is not aligned.")]
    InvalidShader,

    #[error("Invalid Compute Task.")]
    InvalidComputeTask,
}

/// `Result` for `gpgpru` crate.
/// Maps [`Result<T>`] with [`std::result::Result<T, GpuError>`]
pub type Result<T> = std::result::Result<T, GpuError>;

use ash::vk;
use thiserror::Error;

/// Everything that can go wrong when asking a GPU to do math.
///
/// Whether your drivers are missing, VRAM is full, or a shader has too many bindings,
/// this enum delivers the bad news.
///
/// The error messages themselves are already pretty self-explanatory, so we'll spare you the wall of text.
#[derive(Error, Debug)]
pub enum GpuError {
    /// Failed to dynamically load the Vulkan library (`ash::LoadingError`).
    /// Usually means Vulkan drivers or the Vulkan loader aren't installed on the system.
    #[error("Failed to load Vulkan lib.")]
    LoadFailed(#[from] ash::LoadingError),

    /// Raw error code returned directly by the Vulkan API (`vk::Result`).
    #[error("Vulkan Error : {0}")]
    Vulkan(#[from] vk::Result),

    /// No Vulkan-capable GPU was found on this machine.
    ///
    /// This crate only supports Vulkan SDK version over 1.3,
    /// so you might want to update your drivers.
    #[error("No suitable physical device found for vulkan.")]
    NoSuitableDevice,

    /// Found a GPU, but it refuses to expose a compute queue.
    #[error("Device doesn't support compute queue.")]
    NoComputeQueue,

    /// The GPU or driver is stuck on an older Vulkan version (Vulkan 1.3+ is required for dynamic rendering/BDA/push descriptors).
    #[error("GPU does not support Vulkan 1.3 or higher.")]
    VulkanVersionNotSupported,

    /// No available GPU satisfied the filters configured in `DeviceSelector`.
    #[error("No GPU matched the specified DeviceSelector criteria.")]
    HardwareNotSupported,

    /// The underlying memory allocator (`gpu_allocator`) failed—most likely out of VRAM.
    #[error("Device Out of Memory: {0:?}")]
    GpuAllocation(#[from] gpu_allocator::AllocationError),

    /// Ran out of space in the internal dynamic staging/uniform buffer pool.
    #[error("Dynamic Buffer Out of Memory.")]
    OutOfDynamicMemory,

    /// Failed to map a host-visible GPU buffer into a CPU-accessible pointer.
    #[error("Failed to Map Memory into a Pointer.")]
    MemoryMappingFailed,

    /// Internal allocation metadata is missing for a buffer operation.
    ///
    /// This is not supposed to happen, though.
    #[error("Allocation info is missing.")]
    NoAllocation,

    /// Tried to access a CPU pointer on a buffer that isn't host-mapped (e.g., a `GpuOnly` buffer).
    #[error("Memory is not mapped to CPU address space.")]
    NotMapped,

    /// Tried to read or write more bytes than the target GPU buffer can actually hold.
    #[error("Data size ({requested} bytes) exceeds allocated buffer size ({available} bytes).")]
    BufferOverflow { requested: usize, available: usize },

    /// Attempted to create or operate on a buffer with an invalid size (e.g., zero bytes).
    #[error("The size of the buffer is invalid")]
    InvalidBufferSize,

    /// Provided SPIR-V binary bytecode is corrupted or not 4-byte (`u32`) aligned.
    #[error("SPIR-V code is not aligned.")]
    InvalidShader,

    /// Encountered a malformed or unexecutable compute dispatch task in the queue.
    #[error("Invalid Compute Task.")]
    InvalidComputeTask,

    /// Timed out or failed while waiting for the GPU device to finish all pending work (`vkDeviceWaitIdle`).
    #[error("Failed to wait for device idle")]
    FailedWaitIdle,

    /// Shader uses more descriptor bindings than the push descriptor limit allows.
    #[error("Too many bindings in a shader: provided {provided}, maximum {max}.")]
    TooManyBindings { provided: usize, max: usize },

    /// Compiled SPIR-V shader module contained zero instructions.
    #[error("Shader code is empty.")]
    EmptyShaderCode,

    /// Specified entry point name could not be found in the compiled shader module.
    #[error("Invalid entry point in shader.")]
    InvalidEntryPoint,
}

/// A specialized [`Result`](std::result::Result) type for the `gpgpru` crate.
///
/// Maps [`Result`] to [`std::result::Result<T, GpuError>`].
///
/// This is mainly a convenience alias used across the core crate—in most cases,
/// you won't need to inspect this page closely.
pub type Result<T> = std::result::Result<T, GpuError>;

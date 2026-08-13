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

    #[error("Device Out of Memory")]
    OutOfMemory,
}

pub type Result<T> = std::result::Result<T, GpuError>;

// src/lib.rs

mod buffer;
mod compute;
mod device;
mod error;

pub use buffer::{GpuBuffer, create_buffer, create_buffer_empty, read_buffer};
pub use compute::ComputeTask;
pub use device::Device;
pub use error::{GpuError, Result};

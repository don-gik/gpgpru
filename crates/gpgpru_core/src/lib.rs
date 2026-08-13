// src/lib.rs

pub mod device;
pub mod error;

pub use device::Device;
pub use error::{GpuError, Result};

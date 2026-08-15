// src/lib.rs

pub use gpgpru_core::{Device, GpuError, Result};
pub use gpgpru_core::{GpuBuffer, create_buffer, create_buffer_empty, read_buffer};

pub mod ffi {
    pub use gpgpru_macro::compute_hlsl;
}

#[doc(hidden)]
pub mod __private {
    pub use gpgpru_core::ComputeTask;
}

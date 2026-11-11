pub use gpgpru_core::dma;
pub use gpgpru_core::utils;
pub use gpgpru_core::{Device, GpuError, Result};
pub use gpgpru_core::{GpuBuffer, GpuSlice, create_buffer, read_buffer, write_buffer};
pub use gpgpru_macro::kernel;

pub mod prelude {
    pub use gpgpru_core::IntoWorkgroup;
    pub use gpgpru_core::{CpuToGpu, GpuOnly, GpuToCpu, Shared};
}

pub mod config {
    pub use gpgpru_core::{DeviceConfig, DeviceSelector};
}

#[doc(hidden)]
pub mod __private {
    pub use bytemuck;
    pub use gpgpru_core::__private::{AccessType, CommandBuilder, HazardTrace, Task};
}

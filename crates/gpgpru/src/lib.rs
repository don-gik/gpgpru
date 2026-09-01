// src/lib.rs

pub use gpgpru_core::{Device, GpuError, Result};
pub use gpgpru_core::{GpuBuffer, create_buffer, read_buffer};
pub use gpgpru_macro::compute;

pub mod prelude {
    pub use gpgpru_core::IntoWorkgroup;
    pub use gpgpru_core::prelude::{GlobalId, GlobalIdX, GlobalIdY, GlobalIdZ};
}

#[doc(hidden)]
pub mod __private {
    pub use gpgpru_core::__private::{AsGpuBuffer, CommandBuilder, LocationInfo, Task};
}

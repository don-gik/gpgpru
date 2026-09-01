// src/lib.rs

// #![warn(missing_docs)]
// #![warn(clippy::missing_panics_doc)]
// #![warn(clippy::missing_errors_doc)]
// #![warn(clippy::missing_safety_doc)]

mod buffer;
mod device;
mod error;
mod kernel;

pub mod prelude;

pub use buffer::linear::{BufferSlice, GpuBuffer, create_buffer, read_buffer};
pub use buffer::{CpuToGpu, GpuOnly, GpuToCpu, Shared};
pub use device::{Device, IntoWorkgroup};
pub use error::{GpuError, Result};

pub(crate) use device::{AllocDesc, Dispose};

#[doc(hidden)]
pub mod __private {
    pub trait UpdateTimeline {
        fn __update_timeline(&self, signal_value: u64);
    }

    pub use crate::buffer::bda::Bda;
    pub use crate::buffer::linear::{
        BufferBuilder, BufferInit, BufferReadBuilder, BufferReadIntoBuilder, BufferSource,
    };
    pub use crate::buffer::{AsGpuBuffer, LocationInfo};
    pub use crate::device::{CommandBuilder, Task};
}

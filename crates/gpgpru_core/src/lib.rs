// #![warn(missing_docs)]
// #![warn(clippy::missing_panics_doc)]
// #![warn(clippy::missing_errors_doc)]
// #![warn(clippy::missing_safety_doc)]

mod buffer;
mod device;
mod error;
mod page;

use ash::vk;

pub use buffer::linear::{GpuBuffer, GpuSlice, create_buffer, read_buffer, write_buffer};
pub use buffer::{CpuToGpu, GpuOnly, GpuToCpu, Shared};
pub use device::dma;
pub use device::{Device, DeviceConfig, DeviceSelector, IntoWorkgroup};
pub use error::{GpuError, Result};

pub(crate) use device::{AllocDesc, Collectable, Dispose, SharedContext};
pub(crate) use page::*;

#[doc(hidden)]
pub mod __private {
    pub use crate::buffer::linear::{
        BufferBuilder, BufferInit, BufferReadBuilder, BufferReadIntoBuilder, BufferSource,
    };
    pub use crate::buffer::*;
    pub use crate::device::{CommandBuilder, Task};

    pub use crate::page::{AccessType, HazardTrace};
}

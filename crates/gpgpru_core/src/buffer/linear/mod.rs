// src/buffer/linear/mod.rs

mod api;
mod gpubuffer;
mod trash;

pub use api::{BufferBuilder, BufferInit, BufferReadBuilder, BufferReadIntoBuilder, BufferSource};
pub use api::{create_buffer, read_buffer};
pub use gpubuffer::{BufferSlice, GpuBuffer};

pub(crate) use trash::BufferTrash;

use super::{AsGpuBuffer, CpuToGpu, GpuOnly, GpuToCpu, LocationInfo, Shared};

// src/buffer/mod.rs

mod builder;
mod gpubuffer;
mod read;

pub use builder::{create_buffer, create_buffer_empty};
pub use gpubuffer::GpuBuffer;
pub use read::read_buffer;

pub(crate) use gpubuffer::TrashItem;

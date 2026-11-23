// #![warn(missing_docs)]
// #![warn(clippy::missing_panics_doc)]
// #![warn(clippy::missing_errors_doc)]
// #![warn(clippy::missing_safety_doc)]

//! A zero-boilerplate GPGPU compute crate powered by Vulkan 1.3+ and Slang.
//!
//! Tired of writing 800 lines of Vulkan boilerplate just to add two arrays on a GPU?
//! `gpgpru` lets you write your shaders in [Slang](https://shader-slang.com), slap a `#[kernel]`
//! attribute onto a Rust function signature, and let compile-time reflection handle descriptor
//! layouts, push constants, and type validation automatically.
//!
//! # Key Features
//!
//! - **Compile-Time Shader Reflection (`#[kernel]`)**: Automatically compiles `.slang` shaders to
//!   SPIR-V and binds [`GpuSlice`] arguments and scalar push constants with zero manual descriptor sets.
//! - **Automatic Hazard Tracking**: Queue up multiple compute kernels and [`dma`] transfers—`gpgpru`
//!   tracks memory accesses across pages and inserts the exact RAW/WAR/WAW pipeline barriers needed.
//! - **Safe Asynchronous Memory Management**: Dropping a [`GpuBuffer`] won't cause a use-after-free
//!   while the GPU is still chewing on it; an internal timeline-aware garbage collector frees VRAM
//!   only after the GPU finishes the batch.
//! - **Ergonomic Buffer Builders**: Create, upload, and read back buffers in a single fluent chain
//!   via [`create_buffer`], [`write_buffer`], and [`read_buffer`].
//!
//! # Quick Start
//!
//! ```no_run
//! # fn main() -> gpgpru::Result<()> {
//! use gpgpru::prelude::*;
//! use gpgpru::{create_buffer, kernel, read_buffer, Device};
//!
//! // Bind a Slang entry point directly to a Rust function signature
//! #[kernel("shaders/saxpy.slang", "saxpy")]
//! fn saxpy(a: f32, x: &GpuSlice<f32>, y: &mut GpuSlice<f32>, len: u32) {}
//!
//! let device = Device::new()?;
//!
//! let x = create_buffer(&vec![1.0f32; 1024]).with(&device)?;
//! let mut y = create_buffer(&vec![2.0f32; 1024]).with(&device)?;
//!
//! // Dispatch on the GPU (1D, 2D, or 3D workgroups all work)
//! let ticket = saxpy(3.0, &x, &mut y, 1024).with(&device, 4)?;
//! device.wait_timeline(ticket)?;
//!
//! // Bring the results back to CPU land
//! let mut out = vec![0.0f32; 1024];
//! read_buffer(&y).into(&mut out).with(&device)?;
//! # Ok(())
//! # }
//! ```

pub use gpgpru_core::utils;
pub use gpgpru_core::{Device, GpuError, Result};
pub use gpgpru_core::{GpuBuffer, GpuSlice, create_buffer, read_buffer, write_buffer};
pub use gpgpru_macro::{kernel, raw};

/// The `gpgpru` prelude—a convenience bundle of the traits, tags, and tokens you'll reach for constantly.
///
/// Importing `use gpgpru::prelude::*;` saves you from playing whack-a-mole with imports
/// every time you want to specify a buffer's memory location, pass a workgroup size, or
/// hold onto a timeline ticket:
///
/// - **Memory location tags**: `GpuOnly`, `CpuToGpu`, `GpuToCpu`, `Shared` (for `create_buffer(...).on(...)`)
/// - **Workgroup conversions**: `IntoWorkgroup` (so `.with(&device, 64)` and `.queue(&device, (8, 8))` work seamlessly)
/// - **Synchronization tickets**: `SemaphoreValue` (returned by [`Device::dispatch`] and `CommandBuilder::with`)
///
/// # Examples
///
/// ```no_run
/// use gpgpru::prelude::*;
/// ```
pub mod prelude {
    pub use gpgpru_core::IntoWorkgroup;
    pub use gpgpru_core::SemaphoreValue;
    pub use gpgpru_core::{CpuToGpu, GpuOnly, GpuToCpu, Shared};
}

/// Configuration types for initializing and customizing a `Device`.
pub mod config {
    pub use gpgpru_core::{DeviceConfig, DeviceSelector};
}

/// Direct Memory Access (DMA) operations for moving and initializing bytes on the GPU.
///
/// Instead of writing a whole compute shader just to zero out a buffer or copy arrays around,
/// use the hardware transfer engine directly via `copy`, `fill`, and `update`.
///
/// Every operation returns a `DmaTask` builder that automatically plugs into `gpgpru`'s
/// hazard tracker—so you don't have to worry about race conditions with queued compute kernels.
pub mod dma {
    pub use gpgpru_core::dma::{copy, fill, update};
}

#[doc(hidden)]
pub mod __private {
    pub use bytemuck;
    pub use gpgpru_core::__private::vk;
    pub use gpgpru_core::__private::{
        AccessType, CommandBuilder, ComputeArgs, DescriptorPayload, PushDescriptor, Task,
    };
}

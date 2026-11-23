pub mod linear;

use super::*;

// pub use MemoryLocation::*;

/// Describes where a GPU buffer's memory lives and how the CPU and GPU can access it.
///
/// Under the hood, this maps directly to Vulkan memory heap properties (via `gpu-allocator`).
/// When creating buffers with [`create_buffer`], you'll usually pass the matching tag constants
/// ([`GpuOnly`], [`CpuToGpu`], [`GpuToCpu`], or [`Shared`]) to `.on(...)` so the compiler can
/// encode the location straight into [`GpuBuffer`]'s const-generic parameter.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// Device-local VRAM (`DEVICE_LOCAL`). Fastest for GPU shaders, completely invisible to the CPU.
    ///
    /// Use [`read_buffer`] / [`write_buffer`] (DMA) to move data in and out.
    GpuOnly = 0,

    /// Host-visible staging/upload memory (`HOST_VISIBLE | HOST_COHERENT`).
    ///
    /// Optimized for frequent CPU writes that the GPU reads from.
    CpuToGpu = 1,

    /// Host-visible readback memory (`HOST_VISIBLE | HOST_CACHED`).
    ///
    /// Optimized for the GPU writing results that the CPU wants to read back quickly.
    GpuToCpu = 2,

    /// Unified/shared memory (`DEVICE_LOCAL | HOST_VISIBLE` when supported, e.g., ReBAR / Apple Silicon / APUs).
    ///
    /// Gives both the CPU and GPU direct access without explicit staging copies—though
    /// remember you're still responsible for synchronizing reads and writes!
    ///
    /// **This will fallback to [`CpuToGpu`] when hardware is not supported.**
    Shared = 3,
}

impl MemoryLocation {
    /// Converts a raw `u8` tag back into a [`MemoryLocation`].
    ///
    /// Any value outside `0..=2` defaults to [`MemoryLocation::Shared`].
    #[inline(always)]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::GpuOnly,
            1 => Self::CpuToGpu,
            2 => Self::GpuToCpu,
            _ => Self::Shared,
        }
    }

    /// Returns `true` if the CPU can map and write directly to this memory location
    /// ([`MemoryLocation::CpuToGpu`] or [`MemoryLocation::Shared`]).
    #[inline(always)]
    pub const fn is_cpu_writable(self) -> bool {
        matches!(self, Self::CpuToGpu | Self::Shared)
    }

    /// Builds a `gpu-allocator` allocation descriptor matching this memory location.
    pub fn allocation_desc<'a>(
        &self,
        name: &'a str,
        requirements: vk::MemoryRequirements,
    ) -> AllocDesc<'a> {
        match self {
            MemoryLocation::CpuToGpu => AllocDesc::cpu_to_gpu(name, requirements),
            MemoryLocation::GpuOnly => AllocDesc::gpu_only(name, requirements),
            MemoryLocation::GpuToCpu => AllocDesc::gpu_to_cpu(name, requirements),
            MemoryLocation::Shared => AllocDesc::shared(name, requirements),
        }
    }
}

/// A zero-sized compile-time marker encoding memory location as a `const u8` generic.
///
/// This type exists purely so you can write `.on(Shared)` instead of `.on::<{ MemoryLocation::Shared as u8 }>()`
/// (because nobody wants to type that). You shouldn't need to construct `LocationTag` manually—just
/// use the pre-defined [`GpuOnly`], [`CpuToGpu`], [`GpuToCpu`], and [`Shared`] constants.
pub struct LocationTag<const LOC: u8>;

impl<const LOC: u8> LocationTag<LOC> {
    /// The underlying `u8` discriminant corresponding to memory location.
    pub const VALUE: u8 = LOC;

    /// Unwraps this compile-time tag into its raw `u8` value.
    pub const fn to_u8(self) -> u8 {
        LOC
    }
}

/// Compile-time tag for gpu only memory.
///
/// Pass this to [`create_buffer`] via `.on(GpuOnly)` (though it's already the default).
/// Fast VRAM for shaders; strictly off-limits to direct CPU pointers.
#[allow(non_upper_case_globals)]
pub const GpuOnly: LocationTag<{ MemoryLocation::GpuOnly as u8 }> = LocationTag;

/// Compile-time tag for cpu to gpu memory.
///
/// Pass this to [`create_buffer`] via `.on(CpuToGpu)` when you want a host-visible buffer
/// tailored for streaming data from the CPU to the GPU.
#[allow(non_upper_case_globals)]
pub const CpuToGpu: LocationTag<{ MemoryLocation::CpuToGpu as u8 }> = LocationTag;

/// Compile-time tag for shared memory.
///
/// Pass this to [`create_buffer`] via `.on(Shared)` when you want both the CPU and GPU
/// to share direct access to the buffer (great for APUs, Apple Silicon, or ReBAR).
#[allow(non_upper_case_globals)]
pub const Shared: LocationTag<{ MemoryLocation::Shared as u8 }> = LocationTag;

/// Compile-time tag for gpu to cpu memory.
///
/// Pass this to [`create_buffer`] via `.on(GpuToCpu)` when the GPU produces results
/// that the CPU needs to read back efficiently.
#[allow(non_upper_case_globals)]
pub const GpuToCpu: LocationTag<{ MemoryLocation::GpuToCpu as u8 }> = LocationTag;

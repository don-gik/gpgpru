pub mod linear;

use super::*;

// pub use MemoryLocation::*;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    GpuOnly = 0,
    CpuToGpu = 1,
    GpuToCpu = 2,
    Shared = 3,
}

impl MemoryLocation {
    #[inline(always)]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::GpuOnly,
            1 => Self::CpuToGpu,
            2 => Self::GpuToCpu,
            _ => Self::Shared,
        }
    }

    #[inline(always)]
    pub const fn is_cpu_writable(self) -> bool {
        matches!(self, Self::CpuToGpu | Self::Shared)
    }

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

pub struct LocationTag<const LOC: u8>;

impl<const LOC: u8> LocationTag<LOC> {
    pub const VALUE: u8 = LOC;

    pub const fn to_u8(self) -> u8 {
        LOC
    }
}

#[allow(non_upper_case_globals)]
pub const GpuOnly: LocationTag<{ MemoryLocation::GpuOnly as u8 }> = LocationTag;
#[allow(non_upper_case_globals)]
pub const CpuToGpu: LocationTag<{ MemoryLocation::CpuToGpu as u8 }> = LocationTag;
#[allow(non_upper_case_globals)]
pub const Shared: LocationTag<{ MemoryLocation::Shared as u8 }> = LocationTag;
#[allow(non_upper_case_globals)]
pub const GpuToCpu: LocationTag<{ MemoryLocation::GpuToCpu as u8 }> = LocationTag;

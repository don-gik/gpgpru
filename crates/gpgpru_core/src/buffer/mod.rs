// src/buffer/mod.rs

pub mod bda;
pub mod linear;

use crate::AllocDesc;
use bda::Bda;

use ash::vk::{Buffer, MemoryRequirements};

pub struct GpuOnly;
pub struct CpuToGpu;
pub struct GpuToCpu;
pub struct Shared;

pub trait AsGpuBuffer<T> {
    fn base_device_address(&self) -> u64;
    fn vk_buffer(&self) -> Buffer;
    fn offset(&self) -> u64;
    fn size(&self) -> u64;
    #[inline]
    fn device_address(&self) -> u64 {
        self.base_device_address() + self.offset()
    }
    fn bda(&self) -> Bda<T> {
        Bda::new(self.device_address())
    }
    fn is_host_visible(&self) -> bool;
}

pub trait LocationInfo {
    const IS_HOST_VISIBLE: bool;
    fn allocation_desc<'a>(name: &'a str, requirements: MemoryRequirements) -> AllocDesc<'a>;
}

impl LocationInfo for GpuOnly {
    const IS_HOST_VISIBLE: bool = false;
    fn allocation_desc<'a>(name: &'a str, requirements: MemoryRequirements) -> AllocDesc<'a> {
        AllocDesc::gpu_only(name, requirements)
    }
}

impl LocationInfo for CpuToGpu {
    const IS_HOST_VISIBLE: bool = true;
    fn allocation_desc<'a>(name: &'a str, requirements: MemoryRequirements) -> AllocDesc<'a> {
        AllocDesc::cpu_to_gpu(name, requirements)
    }
}

impl LocationInfo for GpuToCpu {
    const IS_HOST_VISIBLE: bool = true;
    fn allocation_desc<'a>(name: &'a str, requirements: MemoryRequirements) -> AllocDesc<'a> {
        AllocDesc::gpu_to_cpu(name, requirements)
    }
}

impl LocationInfo for Shared {
    const IS_HOST_VISIBLE: bool = true;
    fn allocation_desc<'a>(name: &'a str, requirements: MemoryRequirements) -> AllocDesc<'a> {
        AllocDesc::shared(name, requirements)
    }
}

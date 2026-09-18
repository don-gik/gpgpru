// src/device/command/task.rs

use crate::__private::HazardTrace;
use crate::page::HazardResult;

use ash::vk::AccessFlags2;
use smallvec::SmallVec;

#[derive(Debug)]
pub struct Task {
    /// PipelineManager uses this hash.
    pub(crate) pipeline_id: u64,
    /// Pointer PushConstants
    pub(crate) push_constants: [u8; 128],
    pub push_constants_size: u32,
    pub(crate) dispatch_size: (u32, u32, u32),
    pub target_epoch: u64,

    // For Hazards
    pub hazards: HazardResult,
    pub has_write: bool,
}

impl Task {
    pub fn new<T: Copy + HazardTrace>(push_data: &T, dispatch_size: (u32, u32, u32)) -> Self {
        let size = std::mem::size_of::<T>();
        assert!(
            size <= 128,
            "Push constants size exceeds Vulkan standard limit (128 bytes)"
        );

        let mut push_constants = [0u8; 128];
        unsafe {
            std::ptr::copy_nonoverlapping(
                push_data as *const T as *const u8,
                push_constants.as_mut_ptr(),
                size,
            );
        }

        Self {
            pipeline_id: 0,
            push_constants,
            push_constants_size: size as u32,
            dispatch_size,
            target_epoch: 0,
            hazards: HazardResult {
                need_barrier: false,
                src_access: AccessFlags2::empty(),
                dst_access: AccessFlags2::empty(),
                deps: SmallVec::<[u16; 4]>::new(),
            },
            has_write: true,
        }
    }

    pub fn pipeline_id(mut self, pipeline_id: u64) -> Self {
        self.pipeline_id = pipeline_id;
        self
    }
}

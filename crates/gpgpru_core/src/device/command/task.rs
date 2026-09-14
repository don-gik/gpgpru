// src/device/command/task.rs

use crate::__private::{AccessType, HazardTrace};

pub struct Task {
    /// PipelineManager uses this hash.
    pub(crate) pipeline_id: u64,
    /// Pointer PushConstants
    pub(crate) push_constants: [u8; 128],
    pub push_constants_size: u32,
    pub(crate) dispatch_size: (u32, u32, u32),
    pub trace_fn: fn(&[u8], &mut dyn FnMut(u64, u64, AccessType)),
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
            trace_fn: |bytes, visitor| {
                let data = unsafe { &*(bytes.as_ptr() as *const T) };
                data.trace_hazards(visitor);
            },
        }
    }

    pub fn pipeline_id(mut self, pipeline_id: u64) -> Self {
        self.pipeline_id = pipeline_id;
        self
    }
}

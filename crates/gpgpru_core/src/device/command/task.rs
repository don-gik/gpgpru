use super::*;

#[derive(Debug, Clone)]
pub enum DmaOp {
    Copy {
        src: vk::Buffer,
        dst: vk::Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    },
    Fill {
        dst: vk::Buffer,
        dst_offset: u64,
        size: u64,
        pattern: u32,
    },
    Update {
        dst: vk::Buffer,
        dst_offset: u64,
        data: SmallVec<[u8; 128]>,
    },
}

#[derive(Debug, Clone)]
pub(crate) enum TaskPayload {
    Compute {
        pipeline_id: u64,
        push_constants: [u8; 128],
        push_constants_size: u32,
        dispatch_size: (u32, u32, u32),
    },
    Transfer(DmaOp),
}

#[derive(Debug)]
pub struct Task {
    pub(crate) payload: TaskPayload,
    pub target_epoch: u64,

    pub hazards: HazardResult,
    pub has_write: bool,
}

impl Task {
    /// Create compute task
    pub fn new_compute<T: Copy + HazardTrace>(
        push_data: &T,
        dispatch_size: (u32, u32, u32),
    ) -> Self {
        let size = std::mem::size_of::<T>();
        assert!(size <= 128, "Push constants size exceeds 128 bytes");

        let mut push_constants = [0u8; 128];
        unsafe {
            std::ptr::copy_nonoverlapping(
                push_data as *const T as *const u8,
                push_constants.as_mut_ptr(),
                size,
            );
        }

        Self {
            payload: TaskPayload::Compute {
                pipeline_id: 0,
                push_constants,
                push_constants_size: size as u32,
                dispatch_size,
            },
            target_epoch: 0,
            hazards: HazardResult {
                need_barrier: false,
                src_access: vk::AccessFlags2::empty(),
                dst_access: vk::AccessFlags2::empty(),
                deps: SmallVec::new(),
            },
            has_write: true,
        }
    }

    #[inline]
    pub fn with_pipeline_id(mut self, id: u64) -> Self {
        if let TaskPayload::Compute { pipeline_id, .. } = &mut self.payload {
            *pipeline_id = id;
        }
        self
    }

    #[inline]
    pub fn with_epoch(mut self, epoch: u64) -> Self {
        self.target_epoch = epoch;
        self
    }

    #[inline]
    pub(crate) fn kind(&self) -> BatchKind {
        match &self.payload {
            TaskPayload::Compute { pipeline_id, .. } => BatchKind::Compute(*pipeline_id),
            TaskPayload::Transfer(_) => BatchKind::Transfer,
        }
    }

    #[inline]
    pub fn push_constants_size(&self) -> u32 {
        match &self.payload {
            TaskPayload::Compute {
                push_constants_size,
                ..
            } => *push_constants_size,
            TaskPayload::Transfer(_) => 0,
        }
    }
}

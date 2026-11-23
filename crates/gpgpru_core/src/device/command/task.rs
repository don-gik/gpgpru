use super::*;
use smallvec::SmallVec;

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
        pipeline_id: PipelineId,
        push_constants: [u8; 128],
        push_constants_size: u32,
        descriptors: SmallVec<[PushDescriptor; 2]>,
        dispatch_size: (u32, u32, u32),
    },
    Transfer(DmaOp),
    Timestamp {
        query_pool: vk::QueryPool,
        query_index: u32,
    },
}

#[derive(Debug)]
pub struct Task {
    pub(crate) payload: TaskPayload,
    pub target_epoch: u64,
    pub hazards: HazardResult,
    pub has_write: bool,
}

impl Task {
    /// Create compute task from `ComputeArgs`
    pub(crate) fn new_compute<T: ComputeArgs>(
        pipeline_id: PipelineId,
        args: &T,
        dispatch_size: (u32, u32, u32),
    ) -> Self {
        const {
            assert!(
                std::mem::size_of::<T::PushConstants>() <= 128,
                "Push constants size exceeds 128 bytes"
            );
        }

        let pc = args.push_constants();
        let size = std::mem::size_of::<T::PushConstants>();

        let mut push_constants = [0u8; 128];
        if size > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const pc).cast::<u8>(),
                    push_constants.as_mut_ptr(),
                    size,
                );
            }
        }

        let descriptors = SmallVec::from_slice(args.descriptors().as_ref());

        Self {
            payload: TaskPayload::Compute {
                pipeline_id,
                push_constants,
                push_constants_size: size as u32,
                descriptors,
                dispatch_size,
            },
            target_epoch: 0,
            hazards: HazardResult::default(),
            has_write: false,
        }
    }

    /// Create timestamp task
    pub fn new_timestamp(query_pool: vk::QueryPool, query_index: u32) -> Self {
        Self {
            payload: TaskPayload::Timestamp {
                query_pool,
                query_index,
            },
            target_epoch: 0,
            hazards: HazardResult::default(),
            has_write: false,
        }
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
            TaskPayload::Timestamp { .. } => BatchKind::Timestamp,
        }
    }

    #[inline]
    pub fn push_constants_size(&self) -> u32 {
        match &self.payload {
            TaskPayload::Compute {
                push_constants_size,
                ..
            } => *push_constants_size,
            TaskPayload::Transfer(_) | TaskPayload::Timestamp { .. } => 0,
        }
    }
}

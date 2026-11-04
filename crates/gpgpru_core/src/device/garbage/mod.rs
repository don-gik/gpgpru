mod manager;

use super::*;
use crate::buffer::linear::BufferTrash;

pub(crate) trait Dispose {
    fn dispose(
        self,
        device: &LogicalDevice,
        allocator: &AllocationManager,
        page_tracker: &PageTracker,
    );
}

pub(crate) enum Collectable {
    Buffer(BufferTrash),
}

impl Collectable {
    #[inline(always)]
    pub(crate) fn dispose(
        self,
        device: &LogicalDevice,
        allocator: &AllocationManager,
        page_tracker: &PageTracker,
    ) {
        match self {
            Self::Buffer(trash) => trash.dispose(device, allocator, page_tracker),
        }
    }
}

pub(crate) struct Zombie {
    pub(crate) target_timeline: u64,
    pub(crate) payload: Option<Collectable>,
}

pub struct GarbageManager {
    pending: SegQueue<Zombie>,
    active: Mutex<Vec<Zombie>>,
}

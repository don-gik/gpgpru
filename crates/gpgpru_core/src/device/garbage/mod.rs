// src/device/garbage/mod.rs

mod manager;

use crate::buffer::linear::BufferTrash;
use crate::device::AllocationManager;
use crate::page::PageTracker;

use ash::Device;

use crossbeam_queue::SegQueue;
use parking_lot::Mutex;

pub(crate) trait Dispose {
    fn dispose(self, device: &Device, allocator: &AllocationManager, page_tracker: &PageTracker);
}

pub(crate) enum Collectable {
    Buffer(BufferTrash),
}

impl Collectable {
    #[inline(always)]
    pub(crate) fn dispose(
        self,
        device: &Device,
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

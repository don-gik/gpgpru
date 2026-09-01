// src/device/garbage/mod.rs

mod manager;

use crate::buffer::linear::BufferTrash;

use ash::Device;
use gpu_allocator::vulkan::Allocator;

use crossbeam_queue::SegQueue;
use parking_lot::Mutex;

pub(crate) trait Dispose {
    fn dispose(self, device: &Device, allocator: &mut Allocator);
}

pub(crate) enum Collectable {
    Buffer(BufferTrash),
}

impl Collectable {
    #[inline(always)]
    pub(crate) fn dispose(self, device: &Device, allocator: &mut Allocator) {
        match self {
            Self::Buffer(trash) => trash.dispose(device, allocator),
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

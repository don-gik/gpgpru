// src/device/garbage/manager.rs

use super::{Collectable, GarbageManager, Zombie};
use crate::device::AllocationManager;
use crate::page::PageTracker;

use crossbeam_queue::SegQueue;
use parking_lot::Mutex;

impl GarbageManager {
    pub(crate) fn new() -> Self {
        Self {
            pending: SegQueue::new(),
            active: Mutex::new(Vec::with_capacity(128)),
        }
    }

    pub(crate) fn push(&self, target_timeline: u64, payload: Collectable) {
        self.pending.push(Zombie {
            target_timeline,
            payload: Some(payload),
        });
    }

    pub(crate) fn cleanup(
        &self,
        device: &ash::Device,
        alloc_manager: &AllocationManager,
        page_tracker: &PageTracker,
        current_gpu_timeline: u64,
    ) {
        let mut active = self.active.lock();

        while let Some(zombie) = self.pending.pop() {
            active.push(zombie);
        }

        active.retain_mut(|zombie| {
            if zombie.target_timeline <= current_gpu_timeline {
                // Use option to take & destroy
                if let Some(payload) = zombie.payload.take() {
                    payload.dispose(device, alloc_manager, page_tracker);
                }
                false // Memory is Dropped.
            } else {
                true // GPU is still using the resource
            }
        });
    }

    pub(crate) unsafe fn destroy(
        &mut self,
        device: &ash::Device,
        alloc_manager: &AllocationManager,
        page_tracker: &PageTracker,
    ) {
        let mut active = self.active.lock();
        active.retain_mut(|zombie| {
            if let Some(payload) = zombie.payload.take() {
                payload.dispose(device, alloc_manager, page_tracker);
            }
            false
        });

        while let Some(mut zombie) = self.pending.pop() {
            if let Some(payload) = zombie.payload.take() {
                payload.dispose(device, alloc_manager, page_tracker);
            }
        }
    }
}

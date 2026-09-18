// src/page/mod.rs

mod pagetracker;
mod traits;

use portable_atomic::{AtomicU128, Ordering};

use ash::vk::AccessFlags2;
use parking_lot::RwLock;
use smallvec::SmallVec;

pub use traits::HazardTrace;

static PAGE_SHIFT: u64 = 12;
pub const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;

pub const L2_BITS: u32 = 16;
pub const L2_MASK: u64 = (1 << L2_BITS) - 1;

pub const TASK_BITS: u32 = 16;
pub const TASK_MASK: u64 = (1 << TASK_BITS) - 1;

// ------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
}

#[derive(Debug)]
pub struct HazardResult {
    pub need_barrier: bool,
    pub src_access: AccessFlags2,
    pub dst_access: AccessFlags2,
    pub deps: SmallVec<[u16; 4]>,
}

// ------------------------------------------------------------

#[derive(Copy, Clone, Debug, Default)]
pub struct PageState {
    pub last_read: u64,
    pub last_write: u64,
}

impl PageState {
    #[inline]
    fn pack(self) -> u128 {
        ((self.last_write as u128) << 64) | (self.last_read as u128)
    }

    #[inline]
    fn unpack(val: u128) -> Self {
        Self {
            last_write: (val >> 64) as u64,
            last_read: val as u64,
        }
    }
}

// ------------------------------------------------------------

pub struct PageMetadata {
    pub state: AtomicU128,
}

impl Default for PageMetadata {
    fn default() -> Self {
        Self {
            state: AtomicU128::new(0),
        }
    }
}

impl PageMetadata {
    pub fn update(
        &self,
        is_write: bool,
        current_encoded_epoch: u64,
    ) -> (bool, bool, bool, SmallVec<[u16; 4]>) {
        let current_dispatch = current_encoded_epoch >> TASK_BITS;
        let mut current_raw = self.state.load(Ordering::Acquire);

        loop {
            let current = PageState::unpack(current_raw);
            let mut next = current;

            let active_write = current.last_write;
            let active_read = current.last_read;

            let mut raw = false;
            let mut war = false;
            let mut waw = false;
            let mut deps = SmallVec::<[u16; 4]>::new();

            if !is_write {
                // Read Access
                if active_write > active_read {
                    raw = true;
                }
                if active_write > 0 && (active_write >> TASK_BITS) == current_dispatch {
                    let parent_idx = (active_write & TASK_MASK) as u16;
                    deps.push(parent_idx);
                }
                next.last_read = current.last_read.max(current_encoded_epoch);
            } else {
                // Write Access
                if active_read > 0 && active_read >= active_write {
                    war = true;
                }
                if active_write > 0 {
                    waw = true;
                }

                if active_read > 0 && (active_read >> TASK_BITS) == current_dispatch {
                    let parent_idx = (active_read & TASK_MASK) as u16;
                    deps.push(parent_idx);
                }

                if active_write > 0 && (active_write >> TASK_BITS) == current_dispatch {
                    let parent_idx = (active_write & TASK_MASK) as u16;
                    if !deps.contains(&parent_idx) {
                        deps.push(parent_idx);
                    }
                }
                next.last_write = current_encoded_epoch;
                next.last_read = 0;
            }

            match self.state.compare_exchange_weak(
                current_raw,
                next.pack(),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return (raw, war, waw, deps),
                Err(actual) => current_raw = actual,
            }
        }
    }

    pub fn reset(&self) {
        self.state.store(0, Ordering::Release);
    }

    pub fn get_last_write(&self) -> u64 {
        PageState::unpack(self.state.load(Ordering::Acquire)).last_write >> TASK_BITS
    }

    pub fn get_last_read(&self) -> u64 {
        PageState::unpack(self.state.load(Ordering::Acquire)).last_read >> TASK_BITS
    }
}

// ------------------------------------------------------------

pub struct L2Table {
    pub pages: Vec<PageMetadata>,
}

impl L2Table {
    pub fn new() -> Self {
        let count = 1 << L2_BITS;
        let mut pages = Vec::with_capacity(count);
        for _ in 0..count {
            pages.push(PageMetadata::default());
        }
        Self { pages }
    }
}

// ------------------------------------------------------------

pub struct PageTracker {
    pub directory: Vec<RwLock<Option<Box<L2Table>>>>,
}

impl PageTracker {
    pub fn new() -> Self {
        let mut directory = Vec::with_capacity(PAGE_SIZE as usize);
        for _ in 0..PAGE_SIZE {
            directory.push(RwLock::new(None)); // Empty page init
        }
        Self { directory }
    }
}

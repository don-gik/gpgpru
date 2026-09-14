// src/page/mod.rs

mod pagetracker;
mod traits;

use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk::AccessFlags2;
use parking_lot::RwLock;

pub use traits::HazardTrace;

static PAGE_SHIFT: u64 = 16;
static L2_MASK: u64 = (1 << 16) - 1;
static TABLE_SIZE: usize = 65_536;

// ------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
}

pub struct HazardResult {
    pub need_barrier: bool,
    pub src_access: AccessFlags2,
    pub dst_access: AccessFlags2,
}

// ------------------------------------------------------------

pub struct PageMetadata {
    pub last_read_epoch: AtomicU64,
    pub last_write_epoch: AtomicU64,
}

impl Default for PageMetadata {
    fn default() -> Self {
        Self {
            last_read_epoch: AtomicU64::new(0),
            last_write_epoch: AtomicU64::new(0),
        }
    }
}

// ------------------------------------------------------------

pub struct L2Table {
    pub pages: Vec<PageMetadata>,
}

// impl L2Table {
//     pub fn new() -> Self {
//         let mut pages = Vec::with_capacity(TABLE_SIZE);
//         for _ in 0..TABLE_SIZE {
//             pages.push(PageMetadata::default());
//         }
//         Self { pages }
//     }
// }

// ------------------------------------------------------------

pub struct PageTracker {
    pub directory: Vec<RwLock<Option<Box<L2Table>>>>,
}

impl PageTracker {
    pub fn new() -> Self {
        let mut directory = Vec::with_capacity(TABLE_SIZE);
        for _ in 0..TABLE_SIZE {
            directory.push(RwLock::new(None)); // Empty page init
        }
        Self { directory }
    }
}

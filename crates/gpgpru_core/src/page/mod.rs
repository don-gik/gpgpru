use super::*;

mod l1table;
mod traits;

use parking_lot::Mutex;
use smallvec::SmallVec;

pub use traits::HazardTrace;

static PAGE_SHIFT: u64 = 16;
// pub const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;

pub const L2_BITS: u32 = 10;
pub const L2_MASK: u64 = (1 << L2_BITS) - 1;

pub const L1_DIRECTORY_ENTRIES: usize = 1024;
pub const L2_SIZE: usize = 1 << L2_BITS;

pub const TASK_BITS: u32 = 16;
pub const TASK_MASK: u64 = (1 << TASK_BITS) - 1;

// Bit 63: DMA Access Flag, Bit 62: Shader Access Flag
pub const DMA_BIT: u64 = 1 << 63;
pub const SHADER_BIT: u64 = 1 << 62;
pub const FLAGS_MASK: u64 = DMA_BIT | SHADER_BIT;
pub const EPOCH_MASK: u64 = !FLAGS_MASK;
pub const EPOCH_SPLIT: u64 = u64::MAX;

// ------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
    DmaRead,
    DmaWrite,
}

impl AccessType {
    #[inline]
    pub fn is_write(&self) -> bool {
        matches!(self, AccessType::Write | AccessType::DmaWrite)
    }

    #[inline]
    pub fn is_dma(&self) -> bool {
        matches!(self, AccessType::DmaRead | AccessType::DmaWrite)
    }

    #[inline]
    pub fn as_access_flags(&self) -> vk::AccessFlags2 {
        match self {
            AccessType::Read => vk::AccessFlags2::SHADER_READ,
            AccessType::Write => vk::AccessFlags2::SHADER_WRITE,
            AccessType::DmaRead => vk::AccessFlags2::TRANSFER_READ,
            AccessType::DmaWrite => vk::AccessFlags2::TRANSFER_WRITE,
        }
    }

    #[inline]
    pub fn as_pipeline_stage(&self) -> vk::PipelineStageFlags2 {
        match self {
            AccessType::Read | AccessType::Write => vk::PipelineStageFlags2::COMPUTE_SHADER,
            AccessType::DmaRead | AccessType::DmaWrite => vk::PipelineStageFlags2::TRANSFER,
        }
    }
}

#[derive(Debug)]
pub struct HazardResult {
    pub need_barrier: bool,
    pub src_stage: vk::PipelineStageFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
    pub deps: SmallVec<[u16; 8]>,
}

impl Default for HazardResult {
    fn default() -> Self {
        HazardResult {
            need_barrier: false,
            src_stage: vk::PipelineStageFlags2::empty(),
            dst_stage: vk::PipelineStageFlags2::empty(),
            src_access: vk::AccessFlags2::empty(),
            dst_access: vk::AccessFlags2::empty(),
            deps: SmallVec::new(),
        }
    }
}

impl HazardResult {
    pub fn merge(&mut self, other: HazardResult) {
        self.need_barrier |= other.need_barrier;
        self.src_stage |= other.src_stage;
        self.dst_stage |= other.dst_stage;
        self.src_access |= other.src_access;
        self.dst_access |= other.dst_access;
        for dep in other.deps {
            if !self.deps.contains(&dep) {
                self.deps.push(dep);
            }
        }
    }

    pub fn merge_option(&mut self, other: Option<HazardResult>) {
        if let Some(other_res) = other {
            self.merge(other_res);
        }
    }
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

    #[inline]
    pub fn clean_read_epoch(&self) -> u64 {
        self.last_read & EPOCH_MASK
    }

    #[inline]
    pub fn clean_write_epoch(&self) -> u64 {
        self.last_write & EPOCH_MASK
    }

    #[inline]
    pub fn is_read_dma(&self) -> bool {
        (self.last_read & DMA_BIT) != 0
    }

    #[inline]
    pub fn is_read_shader(&self) -> bool {
        (self.last_read & SHADER_BIT) != 0
    }

    #[inline]
    pub fn is_write_dma(&self) -> bool {
        (self.last_write & DMA_BIT) != 0
    }
}

// ------------------------------------------------------------

pub struct PageMetadata {
    pub state: u128,
}

impl Default for PageMetadata {
    fn default() -> Self {
        Self { state: 0u128 }
    }
}

impl PageMetadata {
    pub fn update(
        &mut self,
        access: AccessType,
        current_encoded_epoch: u64,
    ) -> (
        bool,
        bool,
        bool,
        vk::AccessFlags2,
        vk::PipelineStageFlags2,
        SmallVec<[u16; 4]>,
    ) {
        let is_write = access.is_write();
        let access_flag = if access.is_dma() { DMA_BIT } else { SHADER_BIT };

        let current_dispatch = current_encoded_epoch >> TASK_BITS;

        let current = PageState::unpack(self.state);
        let mut next = current;

        let active_write_epoch = current.clean_write_epoch();
        let active_read_epoch = current.clean_read_epoch();

        let mut raw = false;
        let mut war = false;
        let mut waw = false;
        let mut src_access = vk::AccessFlags2::empty();
        let mut src_stage = vk::PipelineStageFlags2::empty();
        let mut deps = SmallVec::<[u16; 4]>::new();

        if !is_write {
            // --- Read Access ---
            if active_write_epoch > active_read_epoch {
                raw = true;
                if current.is_write_dma() {
                    src_access |= vk::AccessFlags2::TRANSFER_WRITE;
                    src_stage |= vk::PipelineStageFlags2::TRANSFER;
                } else {
                    src_access |= vk::AccessFlags2::SHADER_WRITE;
                    src_stage |= vk::PipelineStageFlags2::COMPUTE_SHADER;
                }
            }

            if active_write_epoch > 0 && (active_write_epoch >> TASK_BITS) == current_dispatch {
                let parent_idx = (active_write_epoch & TASK_MASK) as u16;
                deps.push(parent_idx);
            }

            if current_encoded_epoch > active_read_epoch {
                // Update to new epoch
                next.last_read = current_encoded_epoch | access_flag;
            } else if current_encoded_epoch == active_read_epoch {
                // Same epoch - cumulative
                next.last_read |= access_flag;
            }
        } else {
            // --- Write Access ---
            if active_read_epoch > 0 && active_read_epoch >= active_write_epoch {
                war = true;
                if current.is_read_dma() {
                    src_access |= vk::AccessFlags2::TRANSFER_READ;
                    src_stage |= vk::PipelineStageFlags2::TRANSFER;
                }
                if current.is_read_shader() {
                    src_access |= vk::AccessFlags2::SHADER_READ;
                    src_stage |= vk::PipelineStageFlags2::COMPUTE_SHADER;
                }
            }

            if active_write_epoch > 0 {
                waw = true;
                if current.is_write_dma() {
                    src_access |= vk::AccessFlags2::TRANSFER_WRITE;
                    src_stage |= vk::PipelineStageFlags2::TRANSFER;
                } else {
                    src_access |= vk::AccessFlags2::SHADER_WRITE;
                    src_stage |= vk::PipelineStageFlags2::COMPUTE_SHADER;
                }
            }

            if active_read_epoch > 0 && (active_read_epoch >> TASK_BITS) == current_dispatch {
                let parent_idx = (active_read_epoch & TASK_MASK) as u16;
                deps.push(parent_idx);
            }

            if active_write_epoch > 0 && (active_write_epoch >> TASK_BITS) == current_dispatch {
                let parent_idx = (active_write_epoch & TASK_MASK) as u16;
                if !deps.contains(&parent_idx) {
                    deps.push(parent_idx);
                }
            }

            next.last_write = current_encoded_epoch | access_flag;
            next.last_read = 0; // Write access - reset the flag
        }

        let next_state = next.pack();
        if self.state != next_state {
            self.state = next_state;
        }
        (raw, war, waw, src_access, src_stage, deps)
    }

    pub fn reset(&mut self) {
        self.state = 0;
    }

    #[inline]
    pub fn get_last_write(&self) -> u64 {
        let current = PageState::unpack(self.state);
        current.clean_write_epoch() >> TASK_BITS
    }

    #[inline]
    pub fn get_last_read(&self) -> u64 {
        let current = PageState::unpack(self.state);
        current.clean_read_epoch() >> TASK_BITS
    }
}

// ------------------------------------------------------------

pub struct L2Table {
    pub pages: Box<[PageMetadata; L2_SIZE]>,
}

impl L2Table {
    pub fn new() -> Self {
        fn alloc_box<T>() -> Box<T> {
            let layout = std::alloc::Layout::new::<T>();
            unsafe { Box::from_raw(std::alloc::alloc_zeroed(layout) as *mut T) }
        }

        Self { pages: alloc_box() }
    }
}

// ------------------------------------------------------------

pub struct L1Table {
    pub l1_write_epochs: Box<[u64; L1_DIRECTORY_ENTRIES]>,
    pub l1_read_epochs: Box<[u64; L1_DIRECTORY_ENTRIES]>,
    pub directory: Box<[Option<L2Table>; L1_DIRECTORY_ENTRIES]>,
}

impl L1Table {
    pub fn new() -> Self {
        fn alloc_box<T>() -> Box<T> {
            let layout = std::alloc::Layout::new::<T>();
            unsafe { Box::from_raw(std::alloc::alloc_zeroed(layout) as *mut T) }
        }

        Self {
            l1_read_epochs: alloc_box(),
            l1_write_epochs: alloc_box(),
            directory: alloc_box(),
        }
    }
}

// ------------------------------------------------------------

pub struct PageTracker {
    internal: Mutex<L1Table>,
}

impl PageTracker {
    pub fn new() -> Self {
        Self {
            internal: Mutex::new(L1Table::new()),
        }
    }

    pub fn unregister_range(&self, bda: u64, size: u64) {
        let mut guard = self.internal.lock();
        guard.unregister_range(bda, size);
    }

    pub fn check_and_update(
        &self,
        bda: u64,
        size: u64,
        access: AccessType,
        current_epoch: u64,
        current_index: u16,
    ) -> Option<HazardResult> {
        let mut guard = self.internal.lock();
        guard.check_and_update(bda, size, access, current_epoch, current_index)
    }

    pub fn get_last_write_epoch(&self, bda: u64, size: u64) -> u64 {
        let guard = self.internal.lock();
        guard.get_last_write_epoch(bda, size)
    }

    pub fn get_last_access_epoch(&self, bda: u64, size: u64) -> u64 {
        let guard = self.internal.lock();
        guard.get_last_access_epoch(bda, size)
    }
}

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

// Bit 63: DMA Access Flag, Bit 62: Shader Access Flag
pub const DMA_BIT: u64 = 1 << 63;
pub const SHADER_BIT: u64 = 1 << 62;
pub const FLAGS_MASK: u64 = DMA_BIT | SHADER_BIT;
pub const EPOCH_MASK: u64 = !FLAGS_MASK;

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
    pub fn as_access_flags(&self) -> AccessFlags2 {
        match self {
            AccessType::Read => AccessFlags2::SHADER_READ,
            AccessType::Write => AccessFlags2::SHADER_WRITE,
            AccessType::DmaRead => AccessFlags2::TRANSFER_READ,
            AccessType::DmaWrite => AccessFlags2::TRANSFER_WRITE,
        }
    }
}

#[derive(Debug)]
pub struct HazardResult {
    pub need_barrier: bool,
    pub src_access: AccessFlags2,
    pub dst_access: AccessFlags2,
    pub deps: SmallVec<[u16; 8]>,
}

impl Default for HazardResult {
    fn default() -> Self {
        HazardResult {
            need_barrier: false,
            src_access: AccessFlags2::empty(),
            dst_access: AccessFlags2::empty(),
            deps: SmallVec::new(),
        }
    }
}

impl HazardResult {
    pub fn merge(&mut self, other: HazardResult) {
        self.need_barrier |= other.need_barrier;
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
        access: AccessType,
        current_encoded_epoch: u64,
    ) -> (bool, bool, bool, AccessFlags2, SmallVec<[u16; 4]>) {
        let is_write = access.is_write();
        let access_flag = if access.is_dma() { DMA_BIT } else { SHADER_BIT };

        let current_dispatch = current_encoded_epoch >> TASK_BITS;
        let mut current_raw = self.state.load(Ordering::Acquire);

        loop {
            let current = PageState::unpack(current_raw);
            let mut next = current;

            let active_write_epoch = current.clean_write_epoch();
            let active_read_epoch = current.clean_read_epoch();

            let mut raw = false;
            let mut war = false;
            let mut waw = false;
            let mut src_access = AccessFlags2::empty();
            let mut deps = SmallVec::<[u16; 4]>::new();

            if !is_write {
                // --- Read Access ---
                if active_write_epoch > active_read_epoch {
                    raw = true;
                    src_access |= if current.is_write_dma() {
                        AccessFlags2::TRANSFER_WRITE
                    } else {
                        AccessFlags2::SHADER_WRITE
                    };
                }

                if active_write_epoch > 0 && (active_write_epoch >> TASK_BITS) == current_dispatch {
                    let parent_idx = (active_write_epoch & TASK_MASK) as u16;
                    deps.push(parent_idx);
                }

                if current_encoded_epoch > active_read_epoch {
                    // 더 최근의 Epoch로 갱신 (새로운 플래그로 설정)
                    next.last_read = current_encoded_epoch | access_flag;
                } else if current_encoded_epoch == active_read_epoch {
                    // 동일 Epoch 내 읽기: 기존 플래그에 누적 (비트 OR)
                    next.last_read |= access_flag;
                }
            } else {
                // --- Write Access ---
                if active_read_epoch > 0 && active_read_epoch >= active_write_epoch {
                    war = true;
                    if current.is_read_dma() {
                        src_access |= AccessFlags2::TRANSFER_READ;
                    }
                    if current.is_read_shader() {
                        src_access |= AccessFlags2::SHADER_READ;
                    }
                }

                if active_write_epoch > 0 {
                    waw = true;
                    src_access |= if current.is_write_dma() {
                        AccessFlags2::TRANSFER_WRITE
                    } else {
                        AccessFlags2::SHADER_WRITE
                    };
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
                next.last_read = 0; // 쓰기 발생 시 이전 읽기 플래그 초기화
            }

            match self.state.compare_exchange_weak(
                current_raw,
                next.pack(),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return (raw, war, waw, src_access, deps),
                Err(actual) => current_raw = actual,
            }
        }
    }

    pub fn reset(&self) {
        self.state.store(0, Ordering::Release);
    }

    #[inline]
    pub fn get_last_write(&self) -> u64 {
        let current = PageState::unpack(self.state.load(Ordering::Acquire));
        current.clean_write_epoch() >> TASK_BITS
    }

    #[inline]
    pub fn get_last_read(&self) -> u64 {
        let current = PageState::unpack(self.state.load(Ordering::Acquire));
        current.clean_read_epoch() >> TASK_BITS
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
            directory.push(RwLock::new(None));
        }
        Self { directory }
    }
}

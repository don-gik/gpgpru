mod allocator;

use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use super::*;

/// Descriptor that contains Allocation information.
#[derive(Debug, Clone)]
pub struct AllocDesc<'a> {
    /// Name for profiling
    pub(crate) name: &'a str,
    /// Memory Requirements
    pub(crate) requirements: vk::MemoryRequirements,
    /// Memory location
    pub(crate) location: MemoryLocation,
}

impl<'a> AllocDesc<'a> {
    /// Allocation on VRAM
    pub(crate) fn gpu_only(name: &'a str, requirements: vk::MemoryRequirements) -> Self {
        Self {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
        }
    }

    /// Allocation on Cpu -> Gpu (HostVisible)
    pub(crate) fn cpu_to_gpu(name: &'a str, requirements: vk::MemoryRequirements) -> Self {
        Self {
            name,
            requirements,
            location: MemoryLocation::CpuToGpu,
        }
    }

    /// Allocation on Gpu -> Cpu
    pub(crate) fn gpu_to_cpu(name: &'a str, requirements: vk::MemoryRequirements) -> Self {
        Self {
            name,
            requirements,
            location: MemoryLocation::GpuToCpu,
        }
    }

    /// Allocation on Shared
    pub(crate) fn shared(name: &'a str, requirements: vk::MemoryRequirements) -> Self {
        Self {
            name,
            requirements,
            location: MemoryLocation::CpuToGpu,
        }
    }

    #[allow(dead_code)]
    /// Memory alignment adjusting
    pub(crate) fn with_alignment(mut self, alignment: u64) -> Self {
        self.requirements.alignment = self.requirements.alignment.max(alignment);
        self
    }
}

/// Manager struct for Allocation managing.
///
/// Tracks allocation bytes and counts for debugging,
/// while providing `alloc()` and `free()` method.
pub(crate) struct AllocationManager {
    /// The actual static allocator that needs `Mutex`
    pub(crate) inner: Mutex<Allocator>,

    /// Profiling bytes
    allocated_bytes: AtomicU64,
    /// Profiling counts
    allocation_count: AtomicUsize,

    /// For preventing Drop of VulkanCore
    _core: Arc<VulkanCore>,
}

impl AllocationManager {
    /// Creates new `AllocationManager` struct.
    pub(crate) fn new(core: Arc<VulkanCore>) -> Result<Self> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: core.instance.clone(),
            device: core.logical_device.clone(),
            physical_device: core.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        Ok(Self {
            _core: core,
            inner: Mutex::new(allocator),
            allocated_bytes: AtomicU64::new(0),
            allocation_count: AtomicUsize::new(0),
        })
    }

    #[allow(dead_code)]
    /// Load current profiles.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    pub(crate) fn destroy(self) {
        drop(self);
    }
}

impl Drop for AllocationManager {
    fn drop(&mut self) {
        let remaining_bytes = self.allocated_bytes.load(Ordering::Relaxed);
        let remaining_count = self.allocation_count.load(Ordering::Relaxed);

        if remaining_bytes > 0 || remaining_count > 0 {
            tracing::warn!(
                target = "gpgpru",
                bytes = remaining_bytes,
                count = remaining_count,
                "GPU memory leak detected on AllocationManager drop"
            );
        }
    }
}

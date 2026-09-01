// src/device/allocator/allocator.rs

use super::AllocationManager;
use crate::{AllocDesc, Result};

use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use std::sync::atomic::Ordering;
use tracing::trace;

impl AllocationManager {
    /// Allocate memory using manager.
    pub(crate) fn allocate(&self, desc: AllocDesc) -> Result<Allocation> {
        let mut allocator = self.inner.lock();

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: desc.name,
            requirements: desc.requirements,
            location: desc.location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        let size = allocation.size();
        let total = self.allocated_bytes.fetch_add(size, Ordering::Relaxed) + size;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

        trace!(
            name = desc.name,
            bytes = size,
            total_bytes = total,
            "Buffer allocated"
        );
        Ok(allocation)
    }

    /// Free memory using manager.
    pub(crate) fn free(&self, allocation: Allocation) -> Result<()> {
        let size = allocation.size();

        let mut allocator = self.inner.lock();
        allocator.free(allocation)?;

        // Reduce profile counts
        self.allocated_bytes.fetch_sub(size, Ordering::Relaxed);
        self.allocation_count.fetch_sub(1, Ordering::Relaxed);

        Ok(())
    }
}

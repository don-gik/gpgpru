use crate::Dispose;
use crate::device::AllocationManager;
use crate::page::PageTracker;

use ash::Device;
use ash::vk::Buffer;
use gpu_allocator::vulkan::Allocation;

pub(crate) struct BufferTrash {
    pub buffer: Buffer,
    pub allocation: Allocation,
    pub bda: u64,
    pub size: u64,
}

impl Dispose for BufferTrash {
    fn dispose(
        self,
        device: &Device,
        alloc_manager: &AllocationManager,
        page_tracker: &PageTracker,
    ) {
        page_tracker.unregister_range(self.bda, self.size);
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
        let _ = alloc_manager.free(self.allocation);
    }
}

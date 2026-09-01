// src/buffer/linear/trash.rs

use crate::Dispose;

use ash::Device;
use ash::vk::Buffer;
use gpu_allocator::vulkan::{Allocation, Allocator};

#[allow(dead_code)]
pub(crate) struct BufferTrash {
    pub buffer: Buffer,
    pub allocation: Allocation,
}

impl Dispose for BufferTrash {
    fn dispose(self, device: &Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
        let _ = allocator.free(self.allocation);
    }
}

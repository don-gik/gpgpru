// src/buffer/gpubuffer.rs

use crate::device::Execution;
use crate::{GpuError, Result};
use ash::vk;
use crossbeam_channel::Sender;
use gpu_allocator::vulkan::Allocation;
use parking_lot::Mutex;

pub type TrashItem = (vk::Buffer, Option<Allocation>, Option<Execution>);

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: u64,
    free_sender: Sender<TrashItem>,
    pub(crate) pending_exec: Mutex<Option<Execution>>,
}

impl GpuBuffer {
    pub(crate) fn new(
        buffer: vk::Buffer,
        allocation: Allocation,
        size: u64,
        free_sender: Sender<TrashItem>,
    ) -> Self {
        Self {
            buffer,
            allocation: Some(allocation),
            size,
            free_sender,
            pending_exec: Mutex::new(None),
        }
    }

    pub fn write_data<T: Copy>(&mut self, data: &[T]) -> Result<()> {
        let allocation = self.allocation.as_ref().ok_or(GpuError::NoAllocation)?;

        let ptr = allocation.mapped_ptr().ok_or(GpuError::NotMapped)?;

        let bytes_len = data.len() * std::mem::size_of::<T>();
        let buffer_size = allocation.size() as usize;

        if bytes_len > buffer_size {
            return Err(GpuError::BufferOverflow {
                requested: bytes_len,
                available: buffer_size,
            });
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr.as_ptr() as *mut u8,
                bytes_len,
            );
        }

        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let Some(alloc) = self.allocation.take() {
            let exec = self.pending_exec.lock().take();
            let _ = self.free_sender.send((self.buffer, Some(alloc), exec));
        }
    }
}

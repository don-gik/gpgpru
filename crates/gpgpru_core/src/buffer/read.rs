// src/buffer/read.rs

use crate::GpuBuffer;
use crate::device::Device;
use crate::{GpuError, Result};

pub fn read_buffer(buffer: &GpuBuffer) -> BufferReadTask<'_> {
    BufferReadTask { buffer }
}

fn sync_pending_execution(buffer: &GpuBuffer, device: &Device) -> Result<()> {
    if let Some(exec) = buffer.pending_exec.lock().take() {
        device.wait_and_free(exec)?;
    }
    Ok(())
}

pub struct BufferReadInto<'a, T> {
    buffer: &'a GpuBuffer,
    destination: &'a mut [T],
}

impl<'a, T: Copy> BufferReadInto<'a, T> {
    pub fn with(self, device: &Device) -> Result<()> {
        // 1. Wait for GPU
        sync_pending_execution(self.buffer, device)?;

        // 2. Check Bounds
        let alloc = self
            .buffer
            .allocation
            .as_ref()
            .ok_or(GpuError::NoAllocation)?;
        let ptr = alloc.mapped_ptr().ok_or(GpuError::NotMapped)?;

        let read_bytes = self.destination.len() * std::mem::size_of::<T>();
        if read_bytes as u64 > self.buffer.size {
            return Err(GpuError::BufferOverflow {
                requested: read_bytes,
                available: self.buffer.size as usize,
            });
        }

        // 3. Mapped Memory -> Target
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr() as *const u8,
                self.destination.as_mut_ptr() as *mut u8,
                read_bytes,
            );
        }

        Ok(())
    }
}

pub struct BufferReadTask<'a> {
    buffer: &'a GpuBuffer,
}

impl<'a> BufferReadTask<'a> {
    pub fn into<T: Copy>(self, destination: &'a mut [T]) -> BufferReadInto<'a, T> {
        BufferReadInto {
            buffer: self.buffer,
            destination,
        }
    }

    pub fn with<T: Copy + Default>(&self, device: &Device) -> Result<Vec<T>> {
        sync_pending_execution(self.buffer, device)?;

        let alloc = self
            .buffer
            .allocation
            .as_ref()
            .ok_or(GpuError::NoAllocation)?;
        let ptr = alloc.mapped_ptr().ok_or(GpuError::NotMapped)?;

        let count = (self.buffer.size as usize) / std::mem::size_of::<T>();
        let mut result = vec![T::default(); count];

        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr() as *const u8,
                result.as_mut_ptr() as *mut u8,
                self.buffer.size as usize,
            );
        }

        Ok(result)
    }
}

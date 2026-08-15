// src/compute/mod.rs

use crate::device::Execution;
use crate::{Device, GpuBuffer, Result};

pub struct ComputeTask<'a> {
    spirv_code: &'a [u32],
    entry_point: &'a str,
    buffers: Vec<&'a GpuBuffer>,
    push_constants: Vec<u8>,
}

impl<'a> ComputeTask<'a> {
    pub fn new(spirv_code: &'a [u32], entry_point: &'a str) -> Self {
        Self {
            spirv_code,
            entry_point,
            buffers: Vec::new(),
            push_constants: Vec::new(),
        }
    }

    pub fn bind_buffer(mut self, buffer: &'a GpuBuffer) -> Self {
        self.buffers.push(buffer);
        self
    }

    pub fn push_param<T: Copy>(mut self, value: T) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(&value as *const T as *const u8, std::mem::size_of::<T>())
        };
        self.push_constants.extend_from_slice(bytes);
        self
    }

    pub fn with(self, device: &Device, groups: (u32, u32, u32)) -> Result<Execution> {
        let cmd = device.begin_single_time_command()?;

        let resources = device.dispatch_task_raw(
            cmd,
            self.spirv_code,
            self.entry_point,
            &self.buffers,
            &self.push_constants,
            groups,
        )?;

        let exec = device.submit_async(cmd, vec![resources])?;

        for buf in &self.buffers {
            *buf.pending_exec.lock() = Some(exec.clone());
        }

        Ok(exec)
    }
}

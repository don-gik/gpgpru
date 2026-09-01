// src/device/command/command.rs

use crate::__private::Task;
use crate::Device;
use crate::Result;

pub trait IntoWorkgroup {
    fn into_workgroup(self) -> (u32, u32, u32);
}

impl IntoWorkgroup for u32 {
    fn into_workgroup(self) -> (u32, u32, u32) {
        (self, 1, 1)
    }
}

impl IntoWorkgroup for (u32, u32) {
    fn into_workgroup(self) -> (u32, u32, u32) {
        (self.0, self.1, 1)
    }
}

impl IntoWorkgroup for (u32, u32, u32) {
    fn into_workgroup(self) -> (u32, u32, u32) {
        self
    }
}

pub struct CommandBuilder<'a, T: Copy> {
    spirv_code: &'a [u32],
    entry_point: &'a str,
    push_data: T,
}

impl<'a, T: Copy> CommandBuilder<'a, T> {
    pub fn new(spirv_code: &'a [u32], push_data: T) -> Self {
        let size = std::mem::size_of::<T>();
        assert!(size <= 128, "Push constants size exceeds 128 bytes");

        Self {
            spirv_code,
            entry_point: "main",
            push_data: push_data,
        }
    }

    pub fn entry_point(mut self, entry_point: &'a str) -> Self {
        self.entry_point = entry_point;
        self
    }

    pub fn with(&self, device: &Device, workgroup: impl IntoWorkgroup) -> Result<u64> {
        let dispatch_size = workgroup.into_workgroup();
        let task_prebake = Task::new(&self.push_data, dispatch_size);
        let pipeline_id = device.get_pipeline(
            self.spirv_code,
            task_prebake.push_constants_size,
            self.entry_point,
        )?;

        let task = task_prebake.pipeline_id(pipeline_id);

        device.push_command(task);

        let signal_value = device.dispatch()?;

        Ok(signal_value)
    }

    pub fn queue(&self, device: &Device, workgroup: impl IntoWorkgroup) -> Result<()> {
        let dispatch_size = workgroup.into_workgroup();
        let task_prebake = Task::new(&self.push_data, dispatch_size);
        let pipeline_id = device.get_pipeline(
            self.spirv_code,
            task_prebake.push_constants_size,
            self.entry_point,
        )?;

        let task = task_prebake.pipeline_id(pipeline_id);

        device.push_command(task);

        Ok(())
    }
}

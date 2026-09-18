// src/device/command/command.rs

use crate::__private::{AccessType, HazardTrace, Task};
use crate::Device;
use crate::Result;

pub trait IntoWorkgroup {
    fn into_workgroup(self) -> (u32, u32, u32);
}
pub trait IntoU32 {
    fn into_u32(self) -> u32;
}

macro_rules! impl_into_u32 {
    ($($t:ty),*) => {
        $(
            impl IntoU32 for $t {
                fn into_u32(self) -> u32 { self as u32 }
            }
        )*
    };
}
impl_into_u32!(u8, u16, u32, u64, usize, i32, i64, isize);

impl<T: IntoU32> IntoWorkgroup for T {
    fn into_workgroup(self) -> (u32, u32, u32) {
        (self.into_u32(), 1, 1)
    }
}

impl<A: IntoU32, B: IntoU32> IntoWorkgroup for (A, B) {
    fn into_workgroup(self) -> (u32, u32, u32) {
        (self.0.into_u32(), self.1.into_u32(), 1)
    }
}

impl<A: IntoU32, B: IntoU32, C: IntoU32> IntoWorkgroup for (A, B, C) {
    fn into_workgroup(self) -> (u32, u32, u32) {
        (self.0.into_u32(), self.1.into_u32(), self.2.into_u32())
    }
}

impl<T: IntoU32 + Copy> IntoWorkgroup for [T; 3] {
    fn into_workgroup(self) -> (u32, u32, u32) {
        (self[0].into_u32(), self[1].into_u32(), self[2].into_u32())
    }
}

pub struct CommandBuilder<'a, T: Copy> {
    spirv_code: &'a [u32],
    entry_point: &'a str,
    push_data: T,
}

impl<'a, T> CommandBuilder<'a, T>
where
    T: Copy + HazardTrace,
{
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

    fn prepare_task(&self, device: &Device, workgroup: impl IntoWorkgroup) -> Result<Task> {
        let mut task = Task::new(&self.push_data, workgroup.into_workgroup());
        let pipeline_id =
            device.get_pipeline(self.spirv_code, task.push_constants_size, self.entry_point)?;

        let target_epoch = device.pending_epoch();
        let task_idx = device.next_task_index();

        task.target_epoch = target_epoch;

        let mut has_write = false;
        self.push_data.trace_hazards(&mut |bda, size, access| {
            if let Some(hazard) = device.shared.page_tracker.check_and_update(
                bda,
                size,
                access,
                target_epoch,
                task_idx,
            ) {
                task.hazards.need_barrier = true;
                task.hazards.src_access |= hazard.src_access;
                task.hazards.dst_access |= hazard.dst_access;

                for &dep_idx in &hazard.deps {
                    if !task.hazards.deps.contains(&dep_idx) {
                        task.hazards.deps.push(dep_idx);
                    }
                }
            }
            if access == AccessType::Write {
                has_write = true;
            }
        });

        task.has_write = has_write;
        Ok(task.pipeline_id(pipeline_id))
    }

    pub fn with(&self, device: &Device, workgroup: impl IntoWorkgroup) -> Result<u64> {
        let task = self.prepare_task(device, workgroup)?;
        device.push_command(task);

        let signal_value = device.dispatch()?;
        Ok(signal_value)
    }

    pub fn queue(&self, device: &Device, workgroup: impl IntoWorkgroup) -> Result<()> {
        let task = self.prepare_task(device, workgroup)?;
        device.push_command(task);
        Ok(())
    }
}

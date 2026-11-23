use super::*;

/// Anything that can be converted into a 3D Vulkan compute workgroup dispatch grid `(x, y, z)`.
///
/// Nobody wants to write `(64, 1, 1)` every single time for a simple 1D array kernel—or cast
/// `usize` to `u32` twenty times a day. This trait lets you pass whatever dimension feels natural:
///
/// - **1D scalar** (`64`, `1024usize`, etc.) → expands to `(x, 1, 1)`
/// - **2D tuple** (`(16, 16)`) → expands to `(x, y, 1)`
/// - **3D tuple or array** (`(8, 8, 4)` or `[8, 8, 4]`) → maps directly to `(x, y, z)`
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
                #[inline]
                fn into_u32(self) -> u32 { self as u32 }
            }
        )*
    };
}
impl_into_u32!(u8, u16, u32, u64, usize, i32, i64, isize);

impl<T: IntoU32> IntoWorkgroup for T {
    #[inline]
    fn into_workgroup(self) -> (u32, u32, u32) {
        (self.into_u32(), 1, 1)
    }
}

impl<A: IntoU32, B: IntoU32> IntoWorkgroup for (A, B) {
    #[inline]
    fn into_workgroup(self) -> (u32, u32, u32) {
        (self.0.into_u32(), self.1.into_u32(), 1)
    }
}

impl<A: IntoU32, B: IntoU32, C: IntoU32> IntoWorkgroup for (A, B, C) {
    #[inline]
    fn into_workgroup(self) -> (u32, u32, u32) {
        (self.0.into_u32(), self.1.into_u32(), self.2.into_u32())
    }
}

impl<T: IntoU32 + Copy> IntoWorkgroup for [T; 3] {
    #[inline]
    fn into_workgroup(self) -> (u32, u32, u32) {
        (self[0].into_u32(), self[1].into_u32(), self[2].into_u32())
    }
}

#[must_use = "CommandBuilder does nothing unless `.with()` or `.queue()` is called"]
pub struct CommandBuilder<'a, T: ComputeArgs> {
    spirv_code: &'a [u32],
    entry_point: &'a str,
    args: T,
}

impl<'a, T: ComputeArgs> CommandBuilder<'a, T> {
    pub fn new(spirv_code: &'a [u32], args: T) -> Self {
        const {
            assert!(
                std::mem::size_of::<T::PushConstants>() <= 128,
                "Push constants size exceeds 128 bytes"
            );
        }

        Self {
            spirv_code,
            entry_point: "main",
            args,
        }
    }

    pub fn entry_point(mut self, entry_point: &'a str) -> Self {
        self.entry_point = entry_point;
        self
    }

    fn prepare_task(&self, device: &Device, workgroup: impl IntoWorkgroup) -> Result<Task> {
        let pc_size = std::mem::size_of::<T::PushConstants>() as u32;
        let pipeline_id = device.get_pipeline(
            self.spirv_code,
            pc_size,
            self.entry_point,
            T::DESCRIPTOR_TYPES,
        )?;

        let target_epoch = device.pending_epoch().raw();
        let task_idx = device.next_task_index();

        let mut task = Task::new_compute(pipeline_id, &self.args, workgroup.into_workgroup())
            .with_epoch(target_epoch);

        self.args.trace_hazards(&mut |bda, size, access| {
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
                task.has_write = true;
            }
        });

        Ok(task)
    }

    pub fn with(&self, device: &Device, workgroup: impl IntoWorkgroup) -> Result<SemaphoreValue> {
        let task = self.prepare_task(device, workgroup)?;
        device.push_command(task);
        device.dispatch()
    }

    pub fn queue(&self, device: &Device, workgroup: impl IntoWorkgroup) -> Result<()> {
        let task = self.prepare_task(device, workgroup)?;
        device.push_command(task);
        Ok(())
    }
}

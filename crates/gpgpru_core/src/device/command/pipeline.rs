// src/device/command/pipeline.rs

use crate::__private::Task;
use crate::Device;
use crate::Result;

use super::IntoWorkgroup;

pub struct PipelineRunner<F> {
    runner: F,
}

impl<F> PipelineRunner<F>
where
    F: FnOnce(&Device) -> Result<()>,
{
    pub fn new(runner: F) -> Self {
        Self { runner }
    }

    pub fn with(self, device: &Device) -> Result<u64> {
        (self.runner)(device)?;
        device.dispatch()
    }

    pub fn queue(self, device: &Device) -> Result<()> {
        (self.runner)(device)
    }
}

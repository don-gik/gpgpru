mod dispatch;

use super::*;

use std::ffi::CString;
use std::hash::{DefaultHasher, Hash, Hasher};

pub(crate) struct PipelineManager {
    pipelines: DashMap<u64, (vk::Pipeline, vk::PipelineLayout)>,
    core: Arc<VulkanCore>,
}

impl PipelineManager {
    pub(crate) fn new(core: Arc<VulkanCore>) -> Self {
        Self {
            pipelines: DashMap::new(),
            core: core.clone(),
        }
    }

    pub(crate) fn get(&self, id: u64) -> Option<(vk::Pipeline, vk::PipelineLayout)> {
        self.pipelines.get(&id).map(|entry| *entry.value())
    }

    pub(crate) unsafe fn destroy(&self) {
        self.pipelines.retain(|_, &mut (pipeline, layout)| {
            unsafe {
                self.core.logical_device.destroy_pipeline(pipeline, None);
                self.core
                    .logical_device
                    .destroy_pipeline_layout(layout, None);
            }
            false // Returning false removes element from the map
        });
    }
}

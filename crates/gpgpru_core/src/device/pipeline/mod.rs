mod dispatch;

use super::*;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use std::ffi::CString;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(u64);

#[derive(Clone, Copy)]
pub(crate) struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
}

impl ComputePipeline {
    unsafe fn destroy(&self, device: &LogicalDevice) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

pub(crate) struct PipelineManager {
    pipelines: DashMap<PipelineId, ComputePipeline>,
    pipeline_cache: vk::PipelineCache,
    core: Arc<VulkanCore>,
}

impl PipelineManager {
    pub(crate) fn new(core: Arc<VulkanCore>) -> Result<Self> {
        let cache_info = vk::PipelineCacheCreateInfo::default();
        let pipeline_cache = unsafe {
            core.logical_device
                .create_pipeline_cache(&cache_info, None)?
        };

        Ok(Self {
            pipelines: DashMap::new(),
            pipeline_cache,
            core,
        })
    }

    #[inline]
    pub(crate) fn get(&self, id: PipelineId) -> Option<ComputePipeline> {
        self.pipelines.get(&id).map(|entry| *entry.value())
    }

    fn compute_id(
        spirv_code: &[u32],
        entry_point: &str,
        push_constant_size: u32,
        descriptor_types: &[vk::DescriptorType],
    ) -> PipelineId {
        let mut hasher = DefaultHasher::new();
        spirv_code.hash(&mut hasher);
        entry_point.hash(&mut hasher);
        push_constant_size.hash(&mut hasher);
        descriptor_types.hash(&mut hasher);
        PipelineId(hasher.finish())
    }

    pub(crate) unsafe fn destroy(&mut self) {
        for entry in self.pipelines.iter() {
            unsafe { entry.value().destroy(&self.core.logical_device) };
        }
        self.pipelines.clear();
        unsafe {
            self.core
                .logical_device
                .destroy_pipeline_cache(self.pipeline_cache, None);
        }
    }
}

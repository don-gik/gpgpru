use super::*;

impl PipelineManager {
    pub(crate) fn get_or_create<'a>(
        &self,
        spirv_code: &'a [u32],
        entry_point: &'a str,
        push_constant_size: u32,
    ) -> Result<u64> {
        let mut hasher = DefaultHasher::new();
        spirv_code.hash(&mut hasher);
        entry_point.hash(&mut hasher);
        push_constant_size.hash(&mut hasher);
        let id = hasher.finish();

        if self.pipelines.contains_key(&id) {
            return Ok(id);
        }

        let (pipeline, layout) = Self::create_pipeline(
            &self.core.logical_device,
            spirv_code,
            entry_point,
            push_constant_size,
        )?;

        self.pipelines.insert(id, (pipeline, layout));

        Ok(id)
    }

    fn create_pipeline(
        logical_device: &LogicalDevice,
        spirv_code: &[u32],
        entry_point: &str,
        push_constant_size: u32,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
        let mut push_ranges = Vec::new();
        if push_constant_size > 0 {
            push_ranges.push(
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(push_constant_size),
            );
        }

        let layout_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges);
        let layout = unsafe { logical_device.create_pipeline_layout(&layout_info, None)? };

        let module = unsafe {
            logical_device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spirv_code),
                None,
            )?
        };
        let entry_c_str = CString::new(entry_point).unwrap_or_default();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry_c_str);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipeline = unsafe {
            let res = logical_device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            );
            logical_device.destroy_shader_module(module, None);

            match res {
                Ok(ps) => ps[0],
                Err((_, _)) => {
                    logical_device.destroy_pipeline_layout(layout, None);
                    return Err(GpuError::InvalidComputeTask);
                }
            }
        };

        Ok((pipeline, layout))
    }
}

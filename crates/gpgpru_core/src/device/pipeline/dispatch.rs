use super::*;

impl PipelineManager {
    pub(crate) fn get_or_create(
        &self,
        spirv_code: &[u32],
        entry_point: &str,
        push_constant_size: u32,
        descriptor_types: &[vk::DescriptorType],
    ) -> Result<PipelineId> {
        if spirv_code.is_empty() {
            return Err(GpuError::EmptyShaderCode);
        }
        let max_bindings = self.core.max_descriptors as usize;
        if descriptor_types.len() > max_bindings {
            return Err(GpuError::TooManyBindings {
                provided: descriptor_types.len(),
                max: max_bindings,
            });
        }

        let id = Self::compute_id(
            spirv_code,
            entry_point,
            push_constant_size,
            descriptor_types,
        );

        if self.pipelines.contains_key(&id) {
            return Ok(id);
        }

        let created = Self::create_pipeline(
            &self.core.logical_device,
            self.pipeline_cache,
            spirv_code,
            entry_point,
            push_constant_size,
            descriptor_types,
        )?;

        match self.pipelines.entry(id) {
            Entry::Occupied(_) => unsafe {
                created.destroy(&self.core.logical_device);
            },
            Entry::Vacant(entry) => {
                entry.insert(created);
            }
        }

        Ok(id)
    }

    fn create_pipeline(
        logical_device: &LogicalDevice,
        pipeline_cache: vk::PipelineCache,
        spirv_code: &[u32],
        entry_point: &str,
        push_constant_size: u32,
        descriptor_types: &[vk::DescriptorType],
    ) -> Result<ComputePipeline> {
        let entry_c_str = CString::new(entry_point).map_err(|_| GpuError::InvalidEntryPoint)?;

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = descriptor_types
            .iter()
            .enumerate()
            .map(|(i, &desc_type)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(desc_type)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&bindings);

        let set_layout =
            unsafe { logical_device.create_descriptor_set_layout(&set_layout_info, None)? };

        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_constant_size)];
        let push_ranges = if push_constant_size > 0 {
            &push_range[..]
        } else {
            &[]
        };

        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(push_ranges);

        let layout = match unsafe { logical_device.create_pipeline_layout(&layout_info, None) } {
            Ok(l) => l,
            Err(e) => {
                unsafe { logical_device.destroy_descriptor_set_layout(set_layout, None) };
                return Err(e.into());
            }
        };

        let module = match unsafe {
            logical_device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spirv_code),
                None,
            )
        } {
            Ok(m) => m,
            Err(e) => {
                unsafe {
                    logical_device.destroy_pipeline_layout(layout, None);
                    logical_device.destroy_descriptor_set_layout(set_layout, None);
                }
                return Err(e.into());
            }
        };

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry_c_str);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipeline = unsafe {
            let res = logical_device.create_compute_pipelines(
                pipeline_cache,
                std::slice::from_ref(&pipeline_info),
                None,
            );
            logical_device.destroy_shader_module(module, None);

            match res {
                Ok(ps) => ps[0],
                Err((_, vk_err)) => {
                    logical_device.destroy_pipeline_layout(layout, None);
                    logical_device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(GpuError::Vulkan(vk_err));
                }
            }
        };

        Ok(ComputePipeline {
            pipeline,
            layout,
            set_layout,
        })
    }
}

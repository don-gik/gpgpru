// src/device/mod.rs

mod debug;
mod execution;

use std::sync::Arc;

pub(crate) use execution::Execution;

use crate::buffer::TrashItem;
use crate::{GpuBuffer, GpuError, Result};

use ash::{Device as LogicalDevice, Entry, Instance, vk};
use crossbeam_channel::{Receiver, Sender, unbounded};
use crossbeam_queue::SegQueue;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use parking_lot::Mutex;

use debug::DebugContext;

pub struct DispatchResources {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
}

pub struct Device {
    pub entry: Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub logical_device: LogicalDevice,
    pub compute_queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,

    _debug_context: DebugContext,

    allocator: Mutex<Option<Allocator>>,

    available_cmds: SegQueue<vk::CommandBuffer>,
    inflight_executions: Mutex<Vec<Execution>>,
    pending_trash: Mutex<Vec<TrashItem>>,

    free_sender: Sender<TrashItem>,
    free_receiver: Receiver<TrashItem>,
}

impl Device {
    pub fn new() -> Result<Self> {
        // 1. Vulkan initialization logic
        let entry = unsafe { Entry::load()? };

        let instance = Device::create_instance(&entry)?;
        let _debug_context = DebugContext::new(&entry, &instance)?;

        let (physical_device, queue_family_index) = Device::pick_compute_device(&instance)?;

        let (logical_device, compute_queue, command_pool) =
            Device::create_logical_device(&instance, physical_device, queue_family_index)?;

        // 2. Allocator initialize
        let allocator = Mutex::new(Some(Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: logical_device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?));

        let (free_sender, free_receiver) = unbounded();

        Ok(Self {
            entry,
            instance,
            physical_device,
            logical_device,
            compute_queue,
            queue_family_index,
            command_pool,
            _debug_context,
            allocator,
            available_cmds: SegQueue::new(),
            inflight_executions: Mutex::new(Vec::new()),
            pending_trash: Mutex::new(Vec::new()),
            free_sender,
            free_receiver,
        })
    }

    fn auto_gc(&self) {
        while let Ok(item) = self.free_receiver.try_recv() {
            self.pending_trash.lock().push(item);
        }

        self.pending_trash
            .lock()
            .retain_mut(|(buffer, alloc_opt, exec_opt)| {
                let is_ready = match exec_opt {
                    Some(exec) => unsafe {
                        self.logical_device
                            .get_fence_status(exec.fence)
                            .unwrap_or(true)
                    },
                    None => true,
                };

                if is_ready {
                    unsafe {
                        self.logical_device.destroy_buffer(*buffer, None);
                    }
                    if let Some(alloc) = alloc_opt.take() {
                        let mut guard = self.allocator.lock();
                        if let Some(allocator) = guard.as_mut() {
                            let _ = allocator.free(alloc);
                        }
                    }
                    false
                } else {
                    true
                }
            });

        self.inflight_executions.lock().retain(|exec| {
            let is_ready = unsafe {
                self.logical_device
                    .get_fence_status(exec.fence)
                    .unwrap_or(true)
            };

            if is_ready && Arc::strong_count(&exec._tracker) == 1 {
                self.available_cmds.push(exec.cmd_buffer);
                unsafe {
                    self.logical_device.destroy_fence(exec.fence, None);
                }

                for res in exec.resources.iter() {
                    unsafe {
                        self.logical_device.destroy_pipeline(res.pipeline, None);
                        self.logical_device
                            .destroy_pipeline_layout(res.pipeline_layout, None);
                        self.logical_device
                            .destroy_descriptor_set_layout(res.descriptor_set_layout, None);
                        self.logical_device
                            .destroy_descriptor_pool(res.descriptor_pool, None);
                    }
                }

                false
            } else {
                true
            }
        });
    }

    pub(crate) fn allocate(&self, desc: &AllocationCreateDesc) -> Result<Allocation> {
        let mut guard = self.allocator.lock();
        let allocator = guard.as_mut().ok_or(GpuError::NoAllocation)?;
        let allocation = allocator.allocate(desc)?;
        Ok(allocation)
    }

    pub(crate) fn get_free_sender(&self) -> Sender<TrashItem> {
        self.free_sender.clone()
    }

    pub fn begin_single_time_command(&self) -> Result<vk::CommandBuffer> {
        self.auto_gc();

        let cmd = if let Some(cmd) = self.available_cmds.pop() {
            cmd
        } else {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            unsafe { self.logical_device.allocate_command_buffers(&alloc_info)?[0] }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { self.logical_device.begin_command_buffer(cmd, &begin_info)? };
        Ok(cmd)
    }

    pub fn submit_async(
        &self,
        cmd: vk::CommandBuffer,
        resources: Vec<DispatchResources>,
    ) -> Result<Execution> {
        self.auto_gc();

        unsafe {
            self.logical_device.end_command_buffer(cmd)?;
        }

        let fence = unsafe {
            self.logical_device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        unsafe {
            self.logical_device
                .queue_submit(self.compute_queue, &[submit_info], fence)?;
        }

        let exec = Execution {
            fence,
            cmd_buffer: cmd,
            _tracker: Arc::new(()),
            resources: Arc::new(resources),
        };

        self.inflight_executions.lock().push(exec.clone());

        Ok(exec)
    }

    pub fn wait_and_free(&self, exec: Execution) -> Result<()> {
        unsafe {
            self.logical_device
                .wait_for_fences(&[exec.fence], true, u64::MAX)?;
        }
        self.auto_gc();
        Ok(())
    }

    pub fn dispatch_task_raw(
        &self,
        cmd: vk::CommandBuffer,
        spirv_code: &[u32],
        entry_point: &str,
        buffers: &[&GpuBuffer],
        push_constants: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<DispatchResources> {
        unsafe {
            // 1. SPIR-V -> Shader Module
            let module_info = vk::ShaderModuleCreateInfo::default().code(spirv_code);
            let shader_module = self
                .logical_device
                .create_shader_module(&module_info, None)?;

            // 2. Create Descriptor Set Layout
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..buffers.len())
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let descriptor_set_layout = self
                .logical_device
                .create_descriptor_set_layout(&layout_info, None)?;

            // 3. Push Constant Range
            let mut push_constant_ranges = Vec::new();
            if !push_constants.is_empty() {
                push_constant_ranges.push(
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(0)
                        .size(push_constants.len() as u32),
                );
            }

            // 4. Create Pipeline Layout
            let set_layouts = [descriptor_set_layout];
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_constant_ranges);
            let pipeline_layout = self
                .logical_device
                .create_pipeline_layout(&pipeline_layout_info, None)?;

            // 5. Create Compute Pipeline
            let entry_name = std::ffi::CString::new(entry_point).unwrap();
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_name);

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);

            let pipeline = self
                .logical_device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| err)?[0];

            // 6. Create Descriptor Pool, Descriptor Set & Bind to Buffer
            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(buffers.len() as u32);
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(std::slice::from_ref(&pool_size));
            let descriptor_pool = self
                .logical_device
                .create_descriptor_pool(&pool_info, None)?;

            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            let descriptor_set = self.logical_device.allocate_descriptor_sets(&alloc_info)?[0];

            // Write Buffer Info
            let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|buf| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.buffer)
                        .offset(0)
                        .range(buf.size)
                })
                .collect();

            let write_sets: Vec<vk::WriteDescriptorSet> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            self.logical_device.update_descriptor_sets(&write_sets, &[]);

            // 7. Write Command Buffer
            self.logical_device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.logical_device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            if !push_constants.is_empty() {
                self.logical_device.cmd_push_constants(
                    cmd,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }

            self.logical_device
                .cmd_dispatch(cmd, groups.0, groups.1, groups.2);

            // 8. Can throw out ShaderModule
            self.logical_device
                .destroy_shader_module(shader_module, None);

            // 9. Return Resources for Execution
            Ok(DispatchResources {
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
            })
        }
    }

    // Utilities

    fn create_instance(entry: &Entry) -> Result<Instance> {
        let mut layers = Vec::new();
        let mut extensions = Vec::new();

        if cfg!(debug_assertions) {
            layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Gpgpru Engine")
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        unsafe { Ok(entry.create_instance(&create_info, None)?) }
    }

    fn pick_compute_device(instance: &Instance) -> Result<(vk::PhysicalDevice, u32)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        physical_devices
            .into_iter()
            .find_map(|pdev| {
                let props = unsafe { instance.get_physical_device_queue_family_properties(pdev) };

                // Find first index queue that supports compute queue
                props.iter().enumerate().find_map(|(idx, info)| {
                    if info.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                        Some((pdev, idx as u32))
                    } else {
                        None
                    }
                })
            })
            .ok_or(GpuError::NoComputeQueue)
    }

    fn create_logical_device(
        instance: &Instance,
        pdev: vk::PhysicalDevice,
        q_idx: u32,
    ) -> Result<(LogicalDevice, vk::Queue, vk::CommandPool)> {
        let queue_priorities = [1.0f32];
        let q_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(q_idx)
            .queue_priorities(&queue_priorities);

        let dev_info =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&q_info));

        let logical_device = unsafe { instance.create_device(pdev, &dev_info, None)? };

        let queue = unsafe { logical_device.get_device_queue(q_idx, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(q_idx)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { logical_device.create_command_pool(&pool_info, None)? };

        Ok((logical_device, queue, command_pool))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            // 1. Wait for remaining works
            if let Err(e) = self.logical_device.device_wait_idle() {
                eprintln!("[Vulkan WARN] Device wait idle failed: {}", e);
            }

            // 2. Delete trashes on channel
            while let Ok(item) = self.free_receiver.try_recv() {
                self.pending_trash.lock().push(item);
            }

            // 3. Free pending_trash buffer
            let mut alloc_guard = self.allocator.lock();
            for (buffer, alloc_opt, _exec_opt) in self.pending_trash.lock().drain(..) {
                self.logical_device.destroy_buffer(buffer, None);
                if let (Some(alloc), Some(allocator)) = (alloc_opt, alloc_guard.as_mut()) {
                    let _ = allocator.free(alloc);
                }
            }

            // 4. Free inflight_executions fence & pipeline resource
            for exec in self.inflight_executions.lock().drain(..) {
                self.logical_device.destroy_fence(exec.fence, None);
                for res in exec.resources.iter() {
                    self.logical_device.destroy_pipeline(res.pipeline, None);
                    self.logical_device
                        .destroy_pipeline_layout(res.pipeline_layout, None);
                    self.logical_device
                        .destroy_descriptor_set_layout(res.descriptor_set_layout, None);
                    self.logical_device
                        .destroy_descriptor_pool(res.descriptor_pool, None);
                }
            }

            // 5. Allocator  Explicit Drop (Before Logical Device)
            drop(alloc_guard.take());

            // 6. Destroy Command Pool
            self.logical_device
                .destroy_command_pool(self.command_pool, None);

            // 7. Destory Logical Device
            self.logical_device.destroy_device(None);

            // 8. Destroy Debug Context (Before Destroying Instance)
            self._debug_context.destroy();

            // 9. Destory Instance
            self.instance.destroy_instance(None);
        }
    }
}

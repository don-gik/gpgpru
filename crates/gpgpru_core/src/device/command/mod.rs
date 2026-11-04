mod command;
mod queue;
mod task;

pub use command::*;
use queue::*;
pub use task::*;

use super::*;

const MAX_FRAMES_IN_FLIGHT: usize = 3;

pub(crate) struct CommandManager {
    command_pool: vk::CommandPool,
    cmd_buffers: Vec<vk::CommandBuffer>,
    pub(crate) task_queue: TaskQueue,

    transfer_pool: vk::CommandPool,

    pub(crate) timeline_semaphore: vk::Semaphore,
    pub(crate) next_semaphore_value: AtomicU64,

    core: Arc<VulkanCore>,
}

impl CommandManager {
    pub(crate) fn new(core: Arc<VulkanCore>) -> Result<Self> {
        let (command_pool, transfer_pool) = Self::create_command_pool(core.clone())?;
        let task_queue = TaskQueue::new();

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);

        let cmd_buffers = unsafe {
            core.logical_device
                .allocate_command_buffers(&cmd_alloc_info)?
        };

        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
        let timeline_semaphore = unsafe { core.logical_device.create_semaphore(&sem_info, None)? };

        Ok(Self {
            command_pool,
            cmd_buffers,
            task_queue,
            transfer_pool,
            timeline_semaphore,
            next_semaphore_value: AtomicU64::new(1),
            core,
        })
    }

    fn create_command_pool(core: Arc<VulkanCore>) -> Result<(vk::CommandPool, vk::CommandPool)> {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(core.compute_queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe {
            core.logical_device
                .create_command_pool(&command_pool_info, None)?
        };

        let transient_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(core.transfer_queue_family_index)
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            );
        let transfer_pool = unsafe {
            core.logical_device
                .create_command_pool(&transient_pool_info, None)?
        };
        Ok((command_pool, transfer_pool))
    }

    pub(crate) fn push(&self, task: Task) {
        self.task_queue.push_task(task);
    }

    fn record_post_task_barriers(&self, cmd_buffer: ash::vk::CommandBuffer, task: &Task) {
        if task.has_write {
            let post_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ);

            let dependency_info =
                vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&post_barrier));

            unsafe {
                self.core
                    .logical_device
                    .cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
            }
        }
    }

    pub(crate) fn dispatch(
        &self,
        pipeline_manager: &PipelineManager,
        gc_manager: &GarbageManager,
        allocator: &AllocationManager,
        page_tracker: &PageTracker,
    ) -> Result<u64> {
        let current_value = self.next_semaphore_value.load(Ordering::Relaxed) - 1;

        gc_manager.cleanup(
            &self.core.logical_device,
            allocator,
            page_tracker,
            current_value,
        );

        let batches = self.task_queue.drain_sort_and_coalesce();
        if batches.is_empty() {
            return Ok(current_value);
        }

        // dbg!(&batches);

        let frame_slot = (current_value as usize) % MAX_FRAMES_IN_FLIGHT;
        let cmd_buffer = self.cmd_buffers[frame_slot];
        if current_value >= MAX_FRAMES_IN_FLIGHT as u64 {
            let wait_value = current_value - (MAX_FRAMES_IN_FLIGHT as u64) + 1;
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(std::slice::from_ref(&self.timeline_semaphore))
                .values(std::slice::from_ref(&wait_value));
            unsafe {
                self.core
                    .logical_device
                    .wait_semaphores(&wait_info, u64::MAX)?;
            }
        }

        unsafe {
            self.core
                .logical_device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())?;
        }

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.core
                .logical_device
                .begin_command_buffer(cmd_buffer, &begin_info)?;
        }

        let mut current_pipeline_id: Option<u64> = None;
        let mut current_layout: vk::PipelineLayout = vk::PipelineLayout::null();

        for batch in batches {
            if batch.need_barrier {
                // 1. src_access -> src_stage
                let mut src_stage = vk::PipelineStageFlags2::empty();
                if batch.src_access.contains(vk::AccessFlags2::SHADER_READ)
                    || batch.src_access.contains(vk::AccessFlags2::SHADER_WRITE)
                {
                    src_stage |= vk::PipelineStageFlags2::COMPUTE_SHADER;
                }
                if batch.src_access.contains(vk::AccessFlags2::TRANSFER_READ)
                    || batch.src_access.contains(vk::AccessFlags2::TRANSFER_WRITE)
                {
                    src_stage |= vk::PipelineStageFlags2::TRANSFER;
                }
                if src_stage.is_empty() {
                    src_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
                }

                // 2. dst_access -> dst_stage
                let mut dst_stage = vk::PipelineStageFlags2::empty();
                if batch.dst_access.contains(vk::AccessFlags2::SHADER_READ)
                    || batch.dst_access.contains(vk::AccessFlags2::SHADER_WRITE)
                {
                    dst_stage |= vk::PipelineStageFlags2::COMPUTE_SHADER;
                }
                if batch.dst_access.contains(vk::AccessFlags2::TRANSFER_READ)
                    || batch.dst_access.contains(vk::AccessFlags2::TRANSFER_WRITE)
                {
                    dst_stage |= vk::PipelineStageFlags2::TRANSFER;
                }
                if dst_stage.is_empty() {
                    dst_stage = match batch.kind {
                        BatchKind::Compute(_) => vk::PipelineStageFlags2::COMPUTE_SHADER,
                        BatchKind::Transfer => vk::PipelineStageFlags2::TRANSFER,
                    };
                }

                let memory_barrier = vk::MemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(batch.src_access)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(batch.dst_access);

                let dependency_info = vk::DependencyInfo::default()
                    .memory_barriers(std::slice::from_ref(&memory_barrier));

                unsafe {
                    self.core
                        .logical_device
                        .cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
                }
            }

            match batch.kind {
                BatchKind::Compute(pipeline_id) => {
                    if current_pipeline_id != Some(pipeline_id) {
                        let pipeline_info = pipeline_manager
                            .get(pipeline_id)
                            .ok_or(GpuError::InvalidComputeTask)?;

                        unsafe {
                            self.core.logical_device.cmd_bind_pipeline(
                                cmd_buffer,
                                vk::PipelineBindPoint::COMPUTE,
                                pipeline_info.0,
                            );
                        }

                        current_pipeline_id = Some(pipeline_id);
                        current_layout = pipeline_info.1;
                    }

                    for task in &batch.tasks {
                        if let TaskPayload::Compute {
                            push_constants,
                            push_constants_size,
                            dispatch_size,
                            ..
                        } = &task.payload
                        {
                            if *push_constants_size > 0 {
                                unsafe {
                                    self.core.logical_device.cmd_push_constants(
                                        cmd_buffer,
                                        current_layout,
                                        vk::ShaderStageFlags::COMPUTE,
                                        0,
                                        &push_constants[..*push_constants_size as usize],
                                    );
                                }
                            }

                            unsafe {
                                self.core.logical_device.cmd_dispatch(
                                    cmd_buffer,
                                    dispatch_size.0,
                                    dispatch_size.1,
                                    dispatch_size.2,
                                );
                            }
                        }

                        self.record_post_task_barriers(cmd_buffer, task);
                    }
                }

                BatchKind::Transfer => {
                    current_pipeline_id = None;

                    for task in &batch.tasks {
                        if let TaskPayload::Transfer(dma_op) = &task.payload {
                            match dma_op {
                                DmaOp::Copy {
                                    src,
                                    dst,
                                    src_offset,
                                    dst_offset,
                                    size,
                                } => unsafe {
                                    let region = vk::BufferCopy2::default()
                                        .src_offset(*src_offset)
                                        .dst_offset(*dst_offset)
                                        .size(*size);
                                    let copy_info = vk::CopyBufferInfo2::default()
                                        .src_buffer(*src)
                                        .dst_buffer(*dst)
                                        .regions(std::slice::from_ref(&region));

                                    self.core
                                        .logical_device
                                        .cmd_copy_buffer2(cmd_buffer, &copy_info);
                                },
                                DmaOp::Fill {
                                    dst,
                                    dst_offset,
                                    size,
                                    pattern,
                                } => unsafe {
                                    self.core.logical_device.cmd_fill_buffer(
                                        cmd_buffer,
                                        *dst,
                                        *dst_offset,
                                        *size,
                                        *pattern,
                                    );
                                },
                                DmaOp::Update {
                                    dst,
                                    dst_offset,
                                    data,
                                } => unsafe {
                                    self.core.logical_device.cmd_update_buffer(
                                        cmd_buffer,
                                        *dst,
                                        *dst_offset,
                                        data,
                                    );
                                },
                            }
                        }
                    }
                }
            }
        }

        unsafe {
            self.core.logical_device.end_command_buffer(cmd_buffer)?;
        }

        // Submit 2
        let signal_value = self.next_semaphore_value.fetch_add(1, Ordering::SeqCst);
        let cmd_buffer_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd_buffer);

        let wait_info = if current_value > 0 {
            vec![
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.timeline_semaphore)
                    .value(current_value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            ]
        } else {
            vec![]
        };

        // Semaphore
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline_semaphore)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);

        let submit_info2 = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_buffer_info))
            .wait_semaphore_infos(&wait_info)
            .signal_semaphore_infos(std::slice::from_ref(&signal_info));

        // Queue Submit2 (Non-blocking)
        let queue = self.core.compute_queue.lock();
        unsafe {
            self.core.logical_device.queue_submit2(
                *queue,
                std::slice::from_ref(&submit_info2),
                vk::Fence::null(),
            )?;
        }

        Ok(signal_value)
    }

    pub(crate) unsafe fn destroy(&self) {
        let device = &self.core.logical_device;

        unsafe {
            if !self.cmd_buffers.is_empty() {
                device.free_command_buffers(self.command_pool, &self.cmd_buffers);
            }

            device.destroy_command_pool(self.command_pool, None);
            device.destroy_command_pool(self.transfer_pool, None);

            device.destroy_semaphore(self.timeline_semaphore, None);
        }
    }
}

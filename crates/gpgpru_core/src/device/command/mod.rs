// src/device/command/mod.rs

mod command;
// mod pipeline;
mod queue;
mod staging;
mod task;

use super::{GarbageManager, PipelineManager, VulkanCore};
use crate::device::AllocationManager;
use crate::page::PageTracker;
use crate::{GpuError, Result};

use queue::TaskQueue;

pub use {command::CommandBuilder, command::IntoWorkgroup, task::Task};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk::{
    AccessFlags2, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo,
    CommandBufferLevel, CommandBufferSubmitInfo, CommandBufferUsageFlags, CommandPool,
    CommandPoolCreateFlags, CommandPoolCreateInfo, DependencyInfo, Fence, MemoryBarrier2,
    PipelineBindPoint, PipelineLayout, PipelineStageFlags2, Semaphore, SemaphoreCreateInfo,
    SemaphoreSubmitInfo, SemaphoreType, SemaphoreTypeCreateInfo, SemaphoreWaitInfo,
    ShaderStageFlags, SubmitInfo2,
};

// use parking_lot::Mutex;

// pub(crate) struct QueueContext {
//     pub queue: Mutex<Queue>,
//     pub family_index: u32,
// }

const MAX_FRAMES_IN_FLIGHT: usize = 3;

pub(crate) struct CommandManager {
    command_pool: CommandPool,
    cmd_buffers: Vec<CommandBuffer>,
    pub(crate) task_queue: TaskQueue,

    transient_pool: CommandPool,

    pub(crate) timeline_semaphore: Semaphore,
    pub(crate) next_semaphore_value: AtomicU64,

    core: Arc<VulkanCore>,
}

impl CommandManager {
    pub(crate) fn new(core: Arc<VulkanCore>) -> Result<Self> {
        let (command_pool, transient_pool) = Self::create_command_pool(core.clone())?;
        let task_queue = TaskQueue::new();

        let cmd_alloc_info = CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);

        let cmd_buffers = unsafe {
            core.logical_device
                .allocate_command_buffers(&cmd_alloc_info)?
        };

        let mut timeline_info = SemaphoreTypeCreateInfo::default()
            .semaphore_type(SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem_info = SemaphoreCreateInfo::default().push_next(&mut timeline_info);
        let timeline_semaphore = unsafe { core.logical_device.create_semaphore(&sem_info, None)? };

        Ok(Self {
            command_pool,
            cmd_buffers,
            task_queue,
            transient_pool,
            timeline_semaphore,
            next_semaphore_value: AtomicU64::new(1),
            core,
        })
    }

    fn create_command_pool(core: Arc<VulkanCore>) -> Result<(CommandPool, CommandPool)> {
        let command_pool_info = CommandPoolCreateInfo::default()
            .queue_family_index(core.queue_family_index)
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe {
            core.logical_device
                .create_command_pool(&command_pool_info, None)?
        };

        let transient_pool_info = CommandPoolCreateInfo::default()
            .queue_family_index(core.queue_family_index)
            .flags(CommandPoolCreateFlags::TRANSIENT);
        let transient_pool = unsafe {
            core.logical_device
                .create_command_pool(&transient_pool_info, None)?
        };
        Ok((command_pool, transient_pool))
    }

    pub(crate) fn push(&self, task: Task) {
        self.task_queue.push_task(task);
    }

    fn record_post_task_barriers(&self, cmd_buffer: ash::vk::CommandBuffer, task: &Task) {
        if task.has_write {
            let post_barrier = MemoryBarrier2::default()
                .src_stage_mask(PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(PipelineStageFlags2::HOST)
                .dst_access_mask(AccessFlags2::HOST_READ);

            let dependency_info =
                DependencyInfo::default().memory_barriers(std::slice::from_ref(&post_barrier));

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
            let wait_info = SemaphoreWaitInfo::default()
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
                .reset_command_buffer(cmd_buffer, ash::vk::CommandBufferResetFlags::empty())?;
        }

        let begin_info =
            CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.core
                .logical_device
                .begin_command_buffer(cmd_buffer, &begin_info)?;
        }

        let mut current_pipeline_id: Option<u64> = None;
        let mut current_layout: PipelineLayout = PipelineLayout::null();

        for batch in batches {
            if batch.need_barrier {
                let memory_barrier = ash::vk::MemoryBarrier2::default()
                    .src_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(batch.src_access)
                    .dst_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(batch.dst_access);

                let dependency_info = ash::vk::DependencyInfo::default()
                    .memory_barriers(std::slice::from_ref(&memory_barrier));

                unsafe {
                    self.core
                        .logical_device
                        .cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
                }
            }

            if current_pipeline_id != Some(batch.pipeline_id) {
                let pipeline_info = pipeline_manager
                    .get(batch.pipeline_id)
                    .ok_or(GpuError::InvalidComputeTask)?;

                unsafe {
                    self.core.logical_device.cmd_bind_pipeline(
                        cmd_buffer,
                        PipelineBindPoint::COMPUTE,
                        pipeline_info.0,
                    );
                }
                current_pipeline_id = Some(batch.pipeline_id);
                current_layout = pipeline_info.1;
            }

            for task in &batch.tasks {
                if task.push_constants_size > 0 {
                    unsafe {
                        self.core.logical_device.cmd_push_constants(
                            cmd_buffer,
                            current_layout,
                            ShaderStageFlags::COMPUTE,
                            0,
                            &task.push_constants[..task.push_constants_size as usize],
                        );
                    }
                }

                unsafe {
                    self.core.logical_device.cmd_dispatch(
                        cmd_buffer,
                        task.dispatch_size.0,
                        task.dispatch_size.1,
                        task.dispatch_size.2,
                    );
                }

                self.record_post_task_barriers(cmd_buffer, &task);
            }
        }

        unsafe {
            self.core.logical_device.end_command_buffer(cmd_buffer)?;
        }

        // Submit 2
        let signal_value = self.next_semaphore_value.fetch_add(1, Ordering::SeqCst);
        let cmd_buffer_info = CommandBufferSubmitInfo::default().command_buffer(cmd_buffer);

        let wait_info = if current_value > 0 {
            vec![
                SemaphoreSubmitInfo::default()
                    .semaphore(self.timeline_semaphore)
                    .value(current_value)
                    .stage_mask(PipelineStageFlags2::COMPUTE_SHADER),
            ]
        } else {
            vec![]
        };

        // Semaphore
        let signal_info = SemaphoreSubmitInfo::default()
            .semaphore(self.timeline_semaphore)
            .value(signal_value)
            .stage_mask(PipelineStageFlags2::COMPUTE_SHADER);

        let submit_info2 = SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_buffer_info))
            .wait_semaphore_infos(&wait_info)
            .signal_semaphore_infos(std::slice::from_ref(&signal_info));

        // Queue Submit2 (Non-blocking)
        let queue = self.core.compute_queue.lock();
        unsafe {
            self.core.logical_device.queue_submit2(
                *queue,
                std::slice::from_ref(&submit_info2),
                Fence::null(),
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
            device.destroy_command_pool(self.transient_pool, None);

            device.destroy_semaphore(self.timeline_semaphore, None);
        }
    }
}

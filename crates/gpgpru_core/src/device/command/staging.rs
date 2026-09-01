// src/device/command/staging.rs

use super::CommandManager;
use crate::Result;

use std::sync::atomic::Ordering;

use ash::vk::{
    Buffer, BufferCopy2, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
    CommandBufferSubmitInfo, CopyBufferInfo2, Fence, PipelineStageFlags2, SemaphoreSubmitInfo,
    SemaphoreWaitInfo, SubmitInfo2,
};

impl CommandManager {
    pub(crate) fn immediate_copy(
        &self,
        src_buffer: Buffer,
        dst_buffer: Buffer,
        size: u64,
    ) -> Result<()> {
        let cmd_alloc_info = CommandBufferAllocateInfo::default()
            .command_pool(self.transient_pool)
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe {
            self.core
                .logical_device
                .allocate_command_buffers(&cmd_alloc_info)?[0]
        };

        unsafe {
            self.core
                .logical_device
                .begin_command_buffer(cmd, &CommandBufferBeginInfo::default())?;

            let copy_region = BufferCopy2::default().size(size);
            let copy_info = CopyBufferInfo2::default()
                .src_buffer(src_buffer)
                .dst_buffer(dst_buffer)
                .regions(std::slice::from_ref(&copy_region));

            self.core.logical_device.cmd_copy_buffer2(cmd, &copy_info);

            self.core.logical_device.end_command_buffer(cmd)?;
        }

        // Semaphore
        let wait_value = self.next_semaphore_value.fetch_add(1, Ordering::Relaxed);

        // Submit 2
        let cmd_info = CommandBufferSubmitInfo::default().command_buffer(cmd);

        let signal_info = SemaphoreSubmitInfo::default()
            .semaphore(self.timeline_semaphore)
            .value(wait_value)
            .stage_mask(PipelineStageFlags2::COPY);

        let submit_info = SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info));

        let queue = self.core.compute_queue.lock();
        unsafe {
            self.core.logical_device.queue_submit2(
                *queue,
                std::slice::from_ref(&submit_info),
                Fence::null(),
            )?;

            let wait_info = SemaphoreWaitInfo::default()
                .semaphores(std::slice::from_ref(&self.timeline_semaphore))
                .values(std::slice::from_ref(&wait_value));

            self.core
                .logical_device
                .wait_semaphores(&wait_info, u64::MAX)?;

            self.core
                .logical_device
                .free_command_buffers(self.transient_pool, &[cmd]);
        }

        Ok(())
    }
}

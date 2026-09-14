// src/device/mod.rs

/// Crate modules
mod allocator;
mod command;
mod config;
mod debug;
mod garbage;
mod pipeline;
mod vulkancore;

pub use command::{CommandBuilder, IntoWorkgroup, Task};
pub use config::DeviceConfig;
pub use vulkancore::DeviceSelector;

// Pub some modules
pub(crate) use allocator::AllocDesc;
pub(crate) use allocator::AllocationManager;
pub(crate) use command::CommandManager;
pub(crate) use garbage::{Collectable, Dispose, GarbageManager};
pub(crate) use pipeline::PipelineManager;
pub(crate) use vulkancore::VulkanCore;

// Use crate modules
use crate::Result;
use crate::page::PageTracker;

// Use std, external crates
use ash::vk::{Buffer, BufferCreateInfo, BufferUsageFlags, SharingMode};
use std::mem::ManuallyDrop;
use std::sync::Arc;
use tracing::error;

pub(crate) struct SharedContext {
    pub(crate) core: Arc<VulkanCore>,
    pub(crate) gc: GarbageManager,
    pub(crate) cmd_manager: CommandManager,
    pub(crate) pipeline: PipelineManager,
    pub(crate) allocator: ManuallyDrop<AllocationManager>,
    pub(crate) page_tracker: PageTracker,
}

/// Represents the primary GPU device handle and core entry point for the engine.
///
/// `Device` encapsulates the low-level Vulkan abstractions and serves
/// as the central interface for dispatching compute workloads, allocating GPU resources,
/// and managing pipeline states.
///
/// # Examples
///
/// ```no_run
/// use gpgpru::{Device, DeviceSelector};
///
/// let device = Device::new(DeviceSelector::Default)?;
/// ```
pub struct Device {
    pub(crate) shared: Arc<SharedContext>,
}

impl Device {
    /// Creates new `Device` struct.
    ///
    /// # Examples
    /// ```no_run
    /// use gpgpru::{Device, Result};
    ///
    /// let device = Device::new()?;
    /// ```
    ///
    /// # Error
    /// The function fails when :
    /// - Vulkan fails to initialize
    /// - No device with certain criteria
    /// - Vulkan API issues, or Vulkan not found in system
    pub fn new() -> Result<Self> {
        Self::with(DeviceConfig::default())
    }

    /// Creates new `Device` struct, while providing `DeviceConfig` for selecting device.
    ///
    /// # Examples
    /// ```no_run
    /// # use gpgpru::{Device, Result};
    /// # use gpgpru::config::DeviceConfig;
    ///
    /// let device = Device::with(DeviceConfig::default())?;
    /// ```
    ///
    /// # Error
    ///
    /// The function fails when :
    /// - Vulkan fails to initialize
    /// - No device with certain criteria
    /// - Vulkan API issues, or Vulkan not found in system
    pub fn with(device_config: DeviceConfig) -> Result<Self> {
        let core = Arc::new(VulkanCore::init(device_config.device_selector)?);

        let allocator = AllocationManager::new(core.clone())?;
        let cmd_manager = CommandManager::new(core.clone())?;
        let pipeline = PipelineManager::new(core.clone());
        let gc = GarbageManager::new();

        let shared = Arc::new(SharedContext {
            core,
            allocator: ManuallyDrop::new(allocator),
            cmd_manager,
            pipeline,
            gc,
            page_tracker: PageTracker::new(),
        });

        Ok(Self { shared })
    }
}

impl Device {
    pub(crate) fn immediate_upload<T>(&self, dst_buffer: Buffer, data: &[T]) -> Result<()> {
        let size = (std::mem::size_of_val(data)) as u64;
        if size == 0 {
            return Ok(());
        }

        // 1. Staging Buffer (CpuToGpu / HostVisible)
        let staging_buffer_info = BufferCreateInfo::default()
            .size(size)
            .usage(BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe {
            self.shared
                .core
                .logical_device
                .create_buffer(&staging_buffer_info, None)?
        };

        let requirements = unsafe {
            self.shared
                .core
                .logical_device
                .get_buffer_memory_requirements(staging_buffer)
        };

        // Staging buffer allocation
        let staging_alloc = self.shared.allocator.allocate(AllocDesc {
            name: "StagingBuffer_Upload",
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
        })?;

        unsafe {
            self.shared.core.logical_device.bind_buffer_memory(
                staging_buffer,
                staging_alloc.memory(),
                staging_alloc.offset(),
            )?;
        }

        // 2. CPU -> Staging Buffer memcpy
        if let Some(mapped_ptr) = staging_alloc.mapped_ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    mapped_ptr.as_ptr() as *mut u8,
                    size as usize,
                );
            }
        }

        // 3. Command Buffer -> vkCopyBuffer
        self.shared
            .cmd_manager
            .immediate_copy(staging_buffer, dst_buffer, size)?;

        // Drop Staging Resources
        unsafe {
            self.shared
                .core
                .logical_device
                .destroy_buffer(staging_buffer, None);
        }
        self.shared.allocator.free(staging_alloc)?;

        Ok(())
    }

    pub(crate) fn immediate_download<T>(
        &self,
        src_buffer: Buffer,
        out_data: &mut [T],
    ) -> Result<()> {
        let size = (std::mem::size_of_val(out_data)) as u64;
        if size == 0 {
            return Ok(());
        }

        // 1. Staging Buffer (CpuToGpu / HostVisible)
        let staging_buffer_info = BufferCreateInfo::default()
            .size(size)
            .usage(BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe {
            self.shared
                .core
                .logical_device
                .create_buffer(&staging_buffer_info, None)?
        };

        let requirements = unsafe {
            self.shared
                .core
                .logical_device
                .get_buffer_memory_requirements(staging_buffer)
        };

        // Staging buffer allocation
        let staging_alloc = self.shared.allocator.allocate(AllocDesc {
            name: "StagingBuffer_Download",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuToCpu,
        })?;

        unsafe {
            self.shared.core.logical_device.bind_buffer_memory(
                staging_buffer,
                staging_alloc.memory(),
                staging_alloc.offset(),
            )?;
        }

        // 2. Command Buffer -> vkCopyBuffer
        self.shared
            .cmd_manager
            .immediate_copy(src_buffer, staging_buffer, size)?;

        // 3. Staging Buffer -> CPU memcpy
        if let Some(mapped_ptr) = staging_alloc.mapped_ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped_ptr.as_ptr() as *const u8,
                    out_data.as_mut_ptr() as *mut u8,
                    size as usize,
                );
            }
        }

        // Drop Staging Resources
        unsafe {
            self.shared
                .core
                .logical_device
                .destroy_buffer(staging_buffer, None);
        }
        self.shared.allocator.free(staging_alloc)?;

        Ok(())
    }

    pub fn get_pipeline<'a>(
        &self,
        spirv_code: &'a [u32],
        push_constants_size: u32,
        entry_point: &'a str,
    ) -> Result<u64> {
        self.shared
            .pipeline
            .get_or_create(spirv_code, entry_point, push_constants_size)
    }

    pub(crate) fn push_command(&self, task: Task) {
        self.shared.cmd_manager.push(task);
    }

    pub fn dispatch(&self) -> Result<u64> {
        self.shared.cmd_manager.dispatch(
            &self.shared.pipeline,
            &self.shared.gc,
            &self.shared.allocator,
            &self.shared.page_tracker,
        )
    }
}

// ------------------------------------------

impl Drop for SharedContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.cmd_manager.task_queue.drain_and_sort();

            if let Err(e) = self.core.logical_device.device_wait_idle() {
                error!(error = ?e, "Failed to wait for device idle");
            }
            self.pipeline.destroy();

            {
                self.gc.destroy(
                    &self.core.logical_device,
                    &self.allocator,
                    &self.page_tracker,
                );
            }

            self.cmd_manager.destroy();

            let allocator = std::mem::ManuallyDrop::take(&mut self.allocator);
            allocator.destroy();

            self.core.destroy();
        }
    }
}

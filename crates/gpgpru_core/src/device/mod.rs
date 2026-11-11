/// Crate modules
mod allocator;
mod command;
mod config;
mod debug;
pub mod dma;
mod garbage;
mod pipeline;
pub mod utils;
mod vulkancore;

pub use command::{CommandBuilder, IntoWorkgroup, Task};
pub use config::DeviceConfig;
pub use vulkancore::DeviceSelector;

// Pub some modules
pub(crate) use allocator::*;
pub(crate) use command::{CommandManager, DmaOp, TaskPayload};
pub(crate) use garbage::*;
pub(crate) use pipeline::*;
pub(crate) use vulkancore::VulkanCore;

// Use crate modules
use super::*;
use debug::DebugContext;

// Use std, external crates
use ash::Device as LogicalDevice;
use ash::{Entry, Instance};
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use parking_lot::Mutex;
use smallvec::{SmallVec, smallvec};
use std::cell::Cell;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

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
    #[inline]
    pub(crate) fn pending_epoch(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.shared
            .cmd_manager
            .next_semaphore_value
            .load(Ordering::Relaxed)
    }

    #[inline]
    pub(crate) fn next_task_index(&self) -> u16 {
        self.shared.cmd_manager.task_queue.len()
    }

    #[inline]
    pub fn pending_tasks(&self) -> usize {
        self.shared.cmd_manager.task_queue.len() as usize
    }

    pub fn wait_timeline(&self, value: u64) -> Result<()> {
        if value == 0 {
            return Ok(());
        }

        if self.pending_tasks() > 0 {
            self.dispatch()?;
        }

        let semaphore = self.shared.cmd_manager.timeline_semaphore;
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&semaphore))
            .values(std::slice::from_ref(&value));

        unsafe {
            self.shared
                .core
                .logical_device
                .wait_semaphores(&wait_info, u64::MAX)?;
        }

        Ok(())
    }

    pub fn wait_idle(&self) -> Result<()> {
        unsafe {
            self.shared.core.logical_device.device_wait_idle()?;
        }
        Ok(())
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

impl Device {
    pub(crate) fn get_pipeline<'a>(
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

    pub(crate) fn create_query_pool(&self, size: u32) -> Result<vk::QueryPool> {
        let create_info = &vk::QueryPoolCreateInfo::default()
            .query_count(size)
            .query_type(vk::QueryType::TIMESTAMP);
        let raw_query_pool = unsafe {
            self.shared
                .core
                .logical_device
                .create_query_pool(create_info, None)?
        };
        unsafe {
            self.shared
                .core
                .logical_device
                .reset_query_pool(raw_query_pool, 0, size);
        };
        Ok(raw_query_pool)
    }

    #[inline]
    pub(crate) fn timestamp_period(&self) -> f32 {
        self.shared
            .core
            .physical_device_props
            .limits
            .timestamp_period
    }

    pub(crate) fn get_query_pool_result(
        &self,
        query_pool: vk::QueryPool,
        index: u32,
    ) -> Result<u64> {
        let mut timestamp = 0u64;
        unsafe {
            self.shared.core.logical_device.get_query_pool_results(
                query_pool,
                index,
                std::slice::from_mut(&mut timestamp),
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        Ok(timestamp)
    }
}

// ------------------------------------------

impl Drop for SharedContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.cmd_manager.task_queue.drain();

            if let Err(e) = self.core.logical_device.device_wait_idle() {
                tracing::warn!(target: "gpgpru", error = ?e, "Device wait idle failed during cleanup");
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

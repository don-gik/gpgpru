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

pub use command::{
    CommandBuilder, ComputeArgs, DescriptorPayload, IntoWorkgroup, PushDescriptor, Task,
};
pub use config::{DeviceConfig, SlangProfile};
pub use vulkancore::DeviceSelector;

// Pub some modules
pub(crate) use allocator::*;
pub(crate) use command::{CommandManager, DmaOp, TaskPayload};
pub(crate) use garbage::*;
pub(crate) use pipeline::*;
pub(crate) use utils::*;
pub(crate) use vulkancore::VulkanCore;

// Use crate modules
use super::*;
use debug::DebugContext;

// Use std, external crates
use ash::Device as LogicalDevice;
use ash::{Entry, Instance};
use crossbeam_queue::SegQueue;
use parking_lot::Mutex;
use smallvec::{SmallVec, smallvec};
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

pub(crate) struct SharedContext {
    pub(crate) core: Arc<VulkanCore>,
    pub(crate) gc: GarbageManager,
    pub(crate) cmd_manager: CommandManager,
    pub(crate) pipeline: PipelineManager,
    pub(crate) allocator: ManuallyDrop<AllocationManager>,
    pub(crate) page_tracker: PageTracker,
}

/// An opaque handle representing a specific batch submission on the GPU timeline.
///
/// You can't construct or forge this yourself (nice try)—the only way to get one is by
/// calling [`Device::dispatch`] or [`CommandBuilder::with`]. Pass it to [`Device::wait_timeline`]
/// to wait for that exact batch of work to finish.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct SemaphoreValue(pub(crate) u64);

impl SemaphoreValue {
    /// Returns the raw underlying timeline counter value, in case you want to log or inspect it.
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Your diplomatic embassy to a physical GPU and its Vulkan context.
///
/// `Device` wraps all the gnarly low-level Vulkan machinery—command queues, pipeline caches,
/// memory allocators, and garbage collection—behind a single thread-safe handle.
///
/// You can cheaply `.clone()` it if needed (it's backed by an `Arc` internally), and when
/// the last handle drops, it politely waits for the GPU to finish before tearing everything down.
///
/// # Examples
///
/// ```no_run
/// # fn main() -> gpgpru::Result<()> {
/// use gpgpru::{Device, DeviceConfig};
///
/// // Grab the default GPU
/// let device = Device::new()?;
///
/// // Or pass a custom configuration
/// let custom_device = Device::with(DeviceConfig::default())?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Device {
    pub(crate) shared: Arc<SharedContext>,
}

impl Device {
    /// Initializes a new [`Device`] using [`DeviceConfig::default()`].
    ///
    /// Automatically picks a suitable Vulkan 1.3+ GPU with compute support so you can
    /// start crunching numbers immediately.
    ///
    /// # Errors
    ///
    /// Returns a [`GpuError`] if:
    /// - Vulkan fails to load or initialize on your system
    /// - No GPU supports Vulkan 1.3+ or exposes a compute queue
    /// - The [`DeviceSelector`] criteria rule out every GPU in your machine
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> gpgpru::Result<()> {
    /// use gpgpru::Device;
    ///
    /// let device = Device::new()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self> {
        Self::with(DeviceConfig::default())
    }

    /// Initializes a new [`Device`] with a custom [`DeviceConfig`].
    ///
    /// Use this when you have multiple GPUs (e.g., integrated vs. discrete) and want
    /// to be picky about which one gets the job.
    ///
    /// # Errors
    ///
    /// Returns a [`GpuError`] if:
    /// - Vulkan fails to load or initialize on your system
    /// - No GPU matches the criteria specified in `device_config`
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> gpgpru::Result<()> {
    /// use gpgpru::{Device, DeviceConfig};
    ///
    /// let device = Device::with(DeviceConfig::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with(device_config: DeviceConfig) -> Result<Self> {
        let core = Arc::new(VulkanCore::init(device_config.device_selector)?);

        let allocator = AllocationManager::new(core.clone())?;
        let cmd_manager = CommandManager::new(core.clone())?;
        let pipeline = PipelineManager::new(core.clone())?;
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
    /// Returns the timeline semaphore value that will be signaled by the next [`Device::dispatch`] call.
    ///
    /// Used internally to track which GPU batch a resource or operation belongs to
    /// before the commands are actually submitted.
    #[inline]
    pub(crate) fn pending_epoch(&self) -> SemaphoreValue {
        use std::sync::atomic::Ordering;
        SemaphoreValue(
            self.shared
                .cmd_manager
                .next_semaphore_value
                .load(Ordering::Relaxed),
        )
    }

    #[inline]
    pub(crate) fn next_task_index(&self) -> u16 {
        self.shared.cmd_manager.task_queue.len()
    }

    /// Returns the number of queued tasks waiting to be dispatched to the GPU.
    ///
    /// Calling `.queue(...)` on a kernel stages tasks here without actually poking
    /// the GPU yet—this tells you how big that backlog currently is.
    #[inline]
    pub fn pending_tasks(&self) -> usize {
        self.shared.cmd_manager.task_queue.len() as usize
    }

    /// Blocks the CPU thread until the GPU's timeline semaphore reaches `semaphore_value`.
    ///
    /// If there are still unsubmitted tasks sitting in the queue, this will automatically
    /// call [`Device::dispatch`] first so you don't accidentally deadlock yourself waiting
    /// for work that was never sent.
    ///
    /// # Errors
    ///
    /// Returns a [`GpuError`] if flushing pending tasks fails or the Vulkan semaphore wait errors out.
    pub fn wait_timeline(&self, semaphore_value: SemaphoreValue) -> Result<()> {
        if semaphore_value.0 == 0 {
            return Ok(());
        }

        if self.pending_tasks() > 0 {
            self.dispatch()?;
        }

        let semaphore = self.shared.cmd_manager.timeline_semaphore;
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&semaphore))
            .values(std::slice::from_ref(&semaphore_value.0));

        unsafe {
            self.shared
                .core
                .logical_device
                .wait_semaphores(&wait_info, u64::MAX)?;
        }

        Ok(())
    }

    /// Freezes the CPU until the GPU finishes literally everything in flight (`vkDeviceWaitIdle`).
    ///
    /// The brute-force synchronization hammer. Handy for debugging or shutdown, though
    /// [`Device::wait_timeline`] is usually nicer for fine-grained waiting.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Vulkan`] if the driver encounters an error while idling.
    pub fn wait_idle(&self) -> Result<()> {
        unsafe {
            self.shared.core.logical_device.device_wait_idle()?;
        }
        Ok(())
    }

    /// Flushes all queued compute tasks and DMA transfers to the GPU in a single batch.
    ///
    /// This resolves data hazards (RAW/WAR/WAW) across your queued commands, inserts the
    /// necessary pipeline barriers automatically, and submits the command buffer.
    ///
    /// # Returns & Errors
    ///
    /// Returns the timeline semaphore value ([`SemaphoreValue`]) representing this batch, which you can
    /// pass to [`Device::wait_timeline`] later—or returns a [`GpuError`] if submission fails.
    pub fn dispatch(&self) -> Result<SemaphoreValue> {
        Ok(SemaphoreValue(self.shared.cmd_manager.dispatch(
            &self.shared.pipeline,
            &self.shared.gc,
            &self.shared.allocator,
            &self.shared.page_tracker,
        )?))
    }
}

impl Device {
    pub(crate) fn get_pipeline<'a>(
        &self,
        spirv_code: &'a [u32],
        push_constants_size: u32,
        entry_point: &'a str,
        descriptor_types: &[vk::DescriptorType],
    ) -> Result<PipelineId> {
        self.shared.pipeline.get_or_create(
            spirv_code,
            entry_point,
            push_constants_size,
            descriptor_types,
        )
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

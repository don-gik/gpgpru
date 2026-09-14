// src/device/vulkancore.rs

use crate::{GpuError, Result};

use parking_lot::Mutex;

use ash::vk::{self, DeviceCreateInfo};
use ash::{Device as LogicalDevice, Entry, Instance};
use vk::API_VERSION_1_3;
use vk::{
    ApplicationInfo, DeviceQueueCreateInfo, InstanceCreateInfo, PhysicalDevice,
    PhysicalDeviceProperties, PhysicalDeviceType, PhysicalDeviceVulkan12Features,
    PhysicalDeviceVulkan13Features, Queue, QueueFlags,
};

use super::debug::DebugContext;

/// Strategy for selecting a physical device (GPU) for the Vulkan core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    /// Selects the best available device based on score (Discrete GPU > Integrated GPU > Others).
    #[default]
    Default,
    /// Forcefully selects a Discrete GPU. Fails if none are found.
    Discrete,
    /// Forcefully selects an Integrated GPU. Good for power saving.
    Integrated,
    /// Selects a specific device matching the exact vendor ID (e.g., `0x10DE` for NVIDIA).
    VendorId(u32),
}

/// A struct for handling core vulkan items. The name is mostly self-explanatory.
pub(crate) struct VulkanCore {
    pub(crate) _entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) physical_device: PhysicalDevice,
    pub(crate) logical_device: LogicalDevice,
    pub(crate) compute_queue: Mutex<Queue>,
    pub(crate) queue_family_index: u32,

    _debug_context: DebugContext,
}

impl VulkanCore {
    /// Initializes the context and returns [`Result<Self>`].
    ///
    /// Internal helper that loads the Vulkan entry points and creates the core
    /// instance along with the debug context.
    ///
    /// # Errors
    /// Fails if the Vulkan driver cannot be loaded or instance creation fails.
    pub fn init(device_selector: DeviceSelector) -> Result<Self> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry)?;

        let debug_context = DebugContext::new(&entry, &instance)?;

        // Get physical device
        let (physical_device, queue_family_index) =
            Self::get_physical_device(&instance, device_selector)?;

        // Get logical device
        let (logical_device, compute_queue) =
            Self::create_logical_device(&instance, physical_device, queue_family_index)?;

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            logical_device,
            compute_queue: Mutex::new(compute_queue),
            queue_family_index,
            _debug_context: debug_context,
        })
    }

    /// Creates instance from `&Entry` and returns [`Result<vk::Instance>`].
    ///
    /// # Errors
    /// Fails if the Vulkan driver cannot be loaded or instance creation fails.
    fn create_instance(entry: &Entry) -> Result<Instance> {
        let mut layers = Vec::new();
        let mut extensions = Vec::new();

        // If debug build -> then add validation layer
        if cfg!(debug_assertions) {
            layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let app_info = ApplicationInfo::default()
            .application_name(c"GPGPRU ENGINE")
            .api_version(API_VERSION_1_3);

        let create_info = InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        unsafe { Ok(entry.create_instance(&create_info, None)?) }
    }

    /// Fetches compute physical device from `&Instance` and returns [`Result<(PhysicalDevice, u32)>`].
    /// `u32` is for the queue family index for `PhysicalDevice`.
    /// The device selection score is based on `DeviceSelecter` enum.
    ///
    /// The device is guaranteed to have compute queue.
    ///
    /// # Error
    ///
    /// This function will return an error in the following cases:
    /// - When the Vulkan driver cannot be loaded or instance creation fails.
    /// - When there is no available device that supports compute queue.
    fn get_physical_device(
        instance: &Instance,
        device_selector: DeviceSelector,
    ) -> Result<(PhysicalDevice, u32)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        if physical_devices.is_empty() {
            return Err(GpuError::NoSuitableDevice);
        }

        let mut best_device = None;
        let mut max_score = 0;

        let mut found_compute_queue = false;
        let mut found_vulkan_1_3 = false;

        for pdevice in physical_devices {
            let props = unsafe { instance.get_physical_device_properties(pdevice) };
            let queue_props =
                unsafe { instance.get_physical_device_queue_family_properties(pdevice) };

            let mut compute_queue_idx = None;

            // 1. Compute-only-queue
            for (idx, info) in queue_props.iter().enumerate() {
                if info.queue_flags.contains(QueueFlags::COMPUTE)
                    && !info.queue_flags.contains(QueueFlags::GRAPHICS)
                {
                    compute_queue_idx = Some(idx as u32);
                    break;
                }
            }

            // 2. Fallback to general queue
            if compute_queue_idx.is_none() {
                for (idx, info) in queue_props.iter().enumerate() {
                    if info.queue_flags.contains(QueueFlags::COMPUTE) {
                        compute_queue_idx = Some(idx as u32);
                        break;
                    }
                }
            }

            // Skip when no compute queue
            let queue_idx = match compute_queue_idx {
                Some(idx) => {
                    found_compute_queue = true;
                    idx
                }
                None => continue,
            };

            // Checks if vulkan 1.3 is supported
            if props.api_version < API_VERSION_1_3 {
                continue; // Old hardware -> Skip
            }
            found_vulkan_1_3 = true;

            // 3. Now DeviceSelector Score
            let score = Self::calculate_device_score(device_selector, &props);
            if score > max_score {
                max_score = score;
                best_device = Some((pdevice, queue_idx));
            }
        }

        if let Some(device_info) = best_device {
            Ok(device_info)
        } else if !found_compute_queue {
            // No device that has compute queue
            Err(GpuError::NoComputeQueue)
        } else if !found_vulkan_1_3 {
            // There are compute queue but hardware doesn't support vulkan 1.3
            Err(GpuError::VulkanVersionNotSupported)
        } else {
            // No device that has property of DeviceSelector
            Err(GpuError::HardwareNotSupported)
        }
    }

    /// Creates logical device using `&Instance`, `PhysicalDevice`, and `u32`.
    ///
    /// Returns [`Result<(LogicalDevice, Queue)>`].
    ///
    /// # Error
    /// Fails when Vulkan device creation command fails to execute.
    fn create_logical_device(
        instance: &Instance,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
    ) -> Result<(LogicalDevice, Queue)> {
        let queue_priorities = [1.0f32];
        let queue_create_info = DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        // Vulkan Int64 feature
        let features = ash::vk::PhysicalDeviceFeatures::default().shader_int64(true);

        // Vulkan 1.2 feature
        let mut features_12 = PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .buffer_device_address(true)
            .descriptor_indexing(true);

        // Vulkan 1.3 feature
        let mut features_13 = PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true);

        let queue_create_infos = [queue_create_info];

        let device_create_info = DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&features)
            .push_next(&mut features_12)
            .push_next(&mut features_13);

        let logical_device =
            unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        let queue = unsafe { logical_device.get_device_queue(queue_family_index, 0) };

        Ok((logical_device, queue))
    }

    /// Gets physical device, a selector enum and returns a score.
    fn calculate_device_score(
        device_selector: DeviceSelector,
        props: &PhysicalDeviceProperties,
    ) -> u32 {
        match (device_selector, props.device_type) {
            (DeviceSelector::Discrete, PhysicalDeviceType::DISCRETE_GPU) => 1000,
            (DeviceSelector::Discrete, _) => 0,

            (DeviceSelector::Integrated, PhysicalDeviceType::INTEGRATED_GPU) => 1000,
            (DeviceSelector::Integrated, _) => 0,

            (DeviceSelector::VendorId(id), _) if props.vendor_id == id => 1000,
            (DeviceSelector::VendorId(_), _) => 0,

            (DeviceSelector::Default, PhysicalDeviceType::DISCRETE_GPU) => 500,
            (DeviceSelector::Default, PhysicalDeviceType::INTEGRATED_GPU) => 200,
            (DeviceSelector::Default, _) => 100,
        }
    }

    pub(crate) unsafe fn destroy(&self) {
        unsafe {
            self.logical_device.destroy_device(None);

            self._debug_context.destroy();

            self.instance.destroy_instance(None);
        }
    }
}

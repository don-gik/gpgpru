// src/device/mod.rs

mod debug;

use crate::error::{GpuError, Result};
use ash::{Device as LogicalDevice, Entry, Instance, vk};

use debug::DebugContext;

pub struct Device {
    pub entry: Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub logical_device: LogicalDevice,
    pub compute_queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,

    _debug_context: DebugContext,
}

impl Device {
    pub fn new() -> Result<Self> {
        let entry = unsafe { Entry::load()? };

        let instance = Device::create_instance(&entry)?;
        let _debug_context = DebugContext::new(&entry, &instance)?;

        let (physical_device, queue_family_index) = Device::pick_compute_device(&instance)?;

        let (logical_device, compute_queue, command_pool) =
            Device::create_logical_device(&instance, physical_device, queue_family_index)?;

        Ok(Self {
            entry,
            instance,
            physical_device,
            logical_device,
            compute_queue,
            queue_family_index,
            command_pool,
            _debug_context,
        })
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
            if let Err(e) = self.logical_device.device_wait_idle() {
                eprintln!("[Vulkan WARN] Waiting before dropping failed. {}", e);
            }

            self.logical_device
                .destroy_command_pool(self.command_pool, None);
            self.logical_device.destroy_device(None);

            self._debug_context.destroy();

            self.instance.destroy_instance(None);
        }
    }
}

// tests/common/mod.rs

use gpgpru_core::Device;
use std::sync::OnceLock;

static TEST_DEVICE: OnceLock<Device> = OnceLock::new();

pub fn setup_device() -> &'static Device {
    TEST_DEVICE.get_or_init(|| Device::new().expect("Failed Initializing Vulkan Device"))
}

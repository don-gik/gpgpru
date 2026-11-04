use super::DeviceSelector;

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub(crate) device_selector: DeviceSelector,
    // pub dynamic_buffer_capacity: u64,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_selector: DeviceSelector::Default,
            // dynamic_buffer_capacity: 64 * 1024 * 1024, // 64 MB
        }
    }
}

impl DeviceConfig {
    pub fn with_selector(mut self, device_selector: DeviceSelector) -> Self {
        self.device_selector = device_selector;
        self
    }

    // pub fn with_dynamic_buffer_capacity(mut self, dynamic_buffer_capacity: u64) -> Self {
    //     self.dynamic_buffer_capacity = dynamic_buffer_capacity;
    //     self
    // }
}

// src/device/execution.rs

use super::DispatchResources;
use ash::vk;
use std::sync::Arc;

#[derive(Clone)]
pub struct Execution {
    pub fence: vk::Fence,
    pub cmd_buffer: vk::CommandBuffer,
    pub(crate) _tracker: Arc<()>,
    pub(crate) resources: Arc<Vec<DispatchResources>>,
}

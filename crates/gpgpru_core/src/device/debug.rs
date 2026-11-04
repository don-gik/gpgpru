use super::*;

#[cfg(debug_assertions)]
use ash::vk;
#[cfg(debug_assertions)]
use std::ffi::CStr;

#[cfg(debug_assertions)]
use ash::ext::debug_utils::Instance as DebugUtils;

#[cfg(debug_assertions)]
pub(crate) struct DebugContext {
    loader: DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

#[cfg(not(debug_assertions))]
pub(crate) struct DebugContext;

impl DebugContext {
    #[cfg(debug_assertions)]
    pub(crate) fn new(entry: &Entry, instance: &Instance) -> Result<Self> {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let loader = DebugUtils::new(entry, instance);
        let messenger = unsafe { loader.create_debug_utils_messenger(&debug_info, None)? };

        Ok(Self { loader, messenger })
    }

    #[cfg(not(debug_assertions))]
    pub(crate) fn new(_entry: &Entry, _instance: &Instance) -> Result<Self> {
        Ok(Self)
    }

    #[cfg(debug_assertions)]
    pub(crate) unsafe fn destroy(&self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None);
        }
    }

    #[cfg(not(debug_assertions))]
    pub(crate) unsafe fn destroy(&self) {}
}

#[cfg(debug_assertions)]
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    if p_callback_data.is_null() {
        return vk::FALSE;
    }

    let callback_data = unsafe { *p_callback_data };
    let message = if callback_data.p_message.is_null() {
        std::borrow::Cow::Borrowed("Null message body")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message) }.to_string_lossy()
    };

    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        eprintln!("[Vulkan ERROR] {message}");
    } else {
        println!("[Vulkan WARN] {message}");
    }

    vk::FALSE
}

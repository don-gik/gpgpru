use super::DeviceSelector;

/// Configuration options for initializing a `Device`.
///
/// Start with [`DeviceConfig::default()`] and chain builder methods like
/// [`DeviceConfig::with_selector`] before passing it to `Device::with`.
///
/// # Examples
///
/// ```no_run
/// # fn main() -> gpgpru::Result<()> {
/// use gpgpru::{Device, DeviceConfig, DeviceSelector};
///
/// let config = DeviceConfig::default()
///     .with_selector(DeviceSelector::Default);
///
/// let device = Device::with(config)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub(crate) device_selector: DeviceSelector,
    // pub dynamic_buffer_capacity: u64,
}

impl Default for DeviceConfig {
    /// Creates a default configuration using [`DeviceSelector::Default`].
    fn default() -> Self {
        Self {
            device_selector: DeviceSelector::Default,
            // dynamic_buffer_capacity: 64 * 1024 * 1024, // 64 MB
        }
    }
}

impl DeviceConfig {
    /// Sets the [`DeviceSelector`] strategy used to pick which physical GPU gets the job.
    pub fn with_selector(mut self, device_selector: DeviceSelector) -> Self {
        self.device_selector = device_selector;
        self
    }

    // pub fn with_dynamic_buffer_capacity(mut self, dynamic_buffer_capacity: u64) -> Self {
    //     self.dynamic_buffer_capacity = dynamic_buffer_capacity;
    //     self
    // }
}

macro_rules! define_slang_profiles {
    ($( $(#[$meta:meta])* $variant:ident = $val:expr ),* $(,)?) => {
        #[repr(C)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum SlangProfile {
            $(
                $(#[$meta])*
                $variant = $val,
            )*
        }

        impl SlangProfile {
            pub const VARIANT_NAMES: &'static [&'static str] = &[
                $( stringify!($variant), )*
            ];

            pub fn from_variant_name(name: &str) -> Option<Self> {
                match name {
                    $( stringify!($variant) => Some(Self::$variant), )*
                    _ => None,
                }
            }
        }
    };
}

define_slang_profiles! {
    Spirv1_5 = 0,
    Spirv1_6 = 1,

    // Subgroup / Wave
    SpvGroupNonUniform = 100,
    SpvGroupNonUniformVote = 101,
    SpvGroupNonUniformArithmetic = 102,
    SpvGroupNonUniformBallot = 103,
    SpvGroupNonUniformShuffle = 104,
    SpvGroupNonUniformShuffleRelative = 105,

    // Matrix / Precision
    SpvCooperativeMatrixKHR = 200,
    SpvFloat16 = 201,
    SpvInt64 = 202,
}

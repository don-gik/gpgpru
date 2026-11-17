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

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlangProfile {
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

impl SlangProfile {
    pub fn from_variant_name(name: &str) -> Option<Self> {
        match name {
            "Spirv1_5" => Some(Self::Spirv1_5),
            "Spirv1_6" => Some(Self::Spirv1_6),
            "SpvGroupNonUniform" => Some(Self::SpvGroupNonUniform),
            "SpvGroupNonUniformVote" => Some(Self::SpvGroupNonUniformVote),
            "SpvGroupNonUniformArithmetic" => Some(Self::SpvGroupNonUniformArithmetic),
            "SpvGroupNonUniformBallot" => Some(Self::SpvGroupNonUniformBallot),
            "SpvGroupNonUniformShuffle" => Some(Self::SpvGroupNonUniformShuffle),
            "SpvGroupNonUniformShuffleRelative" => Some(Self::SpvGroupNonUniformShuffleRelative),
            "SpvCooperativeMatrixKHR" => Some(Self::SpvCooperativeMatrixKHR),
            "SpvFloat16" => Some(Self::SpvFloat16),
            "SpvInt64" => Some(Self::SpvInt64),
            _ => None,
        }
    }
}

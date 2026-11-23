use super::*;

#[derive(Debug, Clone, Copy)]
pub struct PushDescriptor {
    pub binding: u32,
    pub ty: vk::DescriptorType,
    pub payload: DescriptorPayload,
}

#[derive(Debug, Clone, Copy)]
pub enum DescriptorPayload {
    Buffer(vk::DescriptorBufferInfo),
    // Image(vk::DescriptorImageInfo),
}

pub unsafe trait ComputeArgs {
    type PushConstants: Copy + Send + Sync + 'static;
    type Descriptors: AsRef<[PushDescriptor]>;

    const DESCRIPTOR_TYPES: &'static [vk::DescriptorType];

    fn push_constants(&self) -> Self::PushConstants;
    fn descriptors(&self) -> Self::Descriptors;
    fn trace_hazards(&self, f: &mut dyn FnMut(u64, u64, AccessType));
}

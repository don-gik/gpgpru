use super::*;

pub trait HazardTrace {
    fn trace_hazards(&self, visitor: &mut dyn FnMut(u64, u64, AccessType));
}

// src/kernel/builtins.rs

pub mod builtins {
    pub type GlobalIdX = usize;
    pub type GlobalIdY = usize;
    pub type GlobalIdZ = usize;

    #[repr(transparent)]
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct GlobalId(pub glam::UVec3);
}

// src/ir/var.rs

use crate::SystemBinding;

#[derive(Debug, PartialEq)]
pub enum VarStorage {
    /// let x = 0;
    Local,
    /// Arguments
    PushConstant,
    Bda {
        elem_ty: String,
    },
    /// thread_idx
    Builtin {
        ty: SystemBinding,
    },
    /// Const,
    Const,
}

#[derive(Debug)]
pub struct VarInfo {
    pub name: String,
    pub kind: VarStorage,
}

impl VarInfo {
    pub fn new(name: impl Into<String>, kind: VarStorage) -> Self {
        Self {
            name: name.into(),
            kind,
        }
    }
}

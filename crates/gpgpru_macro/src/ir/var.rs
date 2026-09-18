// src/ir/var.rs

use crate::{SystemBinding, TypeIdx};

use std::borrow::Cow;

#[derive(Debug, PartialEq, Clone, Eq)]
pub enum VarStorage {
    /// let x = 0;
    Local,
    /// Arguments
    PushConstant,
    Bda {
        elem_ty: TypeIdx,
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
    pub name: Cow<'static, str>,
    pub kind: VarStorage,
    pub ty: TypeIdx,
}

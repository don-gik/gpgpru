// src/types/arguments.rs

use crate::TypeIdx;

use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SystemBinding {
    GlobalIdX,
    GlobalIdY,
    GlobalIdZ,
    GlobalId,
}

pub struct ParsedSignature {
    pub system_inputs: HashMap<Cow<'static, str>, SystemBinding>,
    pub pushed_fields: Vec<(Cow<'static, str>, TypeIdx)>,
}

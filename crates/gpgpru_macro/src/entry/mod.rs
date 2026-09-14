// src/entry/mod.rs

mod argument;
mod attribute;

pub use argument::{ParsedSignature, SystemBinding, parse_kernel_signature};
pub use attribute::ComputeAttr;

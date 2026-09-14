// src/types/mod.rs

mod gpu_type;
mod literal;

pub use gpu_type::GpuType;
pub use literal::LiteralKind;

pub trait ToSlangString {
    fn to_slang_string(&self) -> String;
}

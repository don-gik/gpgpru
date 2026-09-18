// src/lowering/mod.rs

mod builder;
mod expr;
mod stmt;

mod prelude;

mod codegen;
mod writer;

pub use builder::LoweringBuilder;
pub use prelude::{resolve_call, resolve_field, resolve_method};
pub use writer::CodeWriter;

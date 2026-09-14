// src/rust_emit/mod.rs

mod entry;
mod slang_code;

pub use entry::generate_entry;
pub use slang_code::compile_slang;

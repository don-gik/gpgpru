// src/ffi/mod.rs

pub mod hlsl;

pub use hlsl::{compile_hlsl_to_spirv, extract_raw_string};

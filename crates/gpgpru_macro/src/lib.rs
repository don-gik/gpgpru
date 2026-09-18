// src/lib.rs

mod ir;
mod types;

mod lowering;

mod macros;

pub(crate) use ir::*;
pub(crate) use lowering::*;
pub(crate) use types::*;

pub(crate) use macros::*;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn compute(attr: TokenStream, item: TokenStream) -> TokenStream {
    match compute_impl(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.into_compile_error().into(),
    }
}

mod ffi;
mod kernel;

use ffi::*;
use kernel::*;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, body: TokenStream) -> TokenStream {
    match kernel_impl(attr.into(), body.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn raw(_attr: TokenStream, body: TokenStream) -> TokenStream {
    body
}

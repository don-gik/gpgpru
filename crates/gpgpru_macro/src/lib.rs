// src/lib.rs

mod ffi;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{FnArg, ItemFn, Pat, parse_macro_input};

#[proc_macro_attribute]
pub fn compute_hlsl(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let vis = &input_fn.vis;

    // 1. Extract Raw String
    let hlsl_code = match ffi::extract_raw_string(&input_fn) {
        Ok(code) => code,
        Err(err) => return err.to_compile_error().into(),
    };

    // 2. Compile HLSL
    let spirv_u32 = match ffi::compile_hlsl_to_spirv(&hlsl_code, "main", Span::call_site()) {
        Ok(code) => code,
        Err(err) => return err.to_compile_error().into(),
    };

    // 3. For Rust-analyzer
    let mut fn_args = Vec::new();
    let mut task_binds = Vec::new();

    for arg in input_fn.sig.inputs.iter() {
        if let FnArg::Typed(pat_type) = arg {
            let var_name = match &*pat_type.pat {
                Pat::Ident(pat) => &pat.ident,
                _ => continue,
            };

            let ty = &pat_type.ty;
            let type_str = quote!(#ty).to_string();

            if type_str.contains("GpuBuffer") {
                // Buffer lifetime -> ComputeTask
                if let syn::Type::Reference(type_ref) = &**ty {
                    let mutability = &type_ref.mutability; // Some(Mut) or None
                    fn_args.push(quote! { #var_name: &'a #mutability gpgpru::GpuBuffer });
                    task_binds.push(quote! { .bind_buffer(#var_name) });
                }
            } else {
                fn_args.push(quote! { #var_name: #ty });
                task_binds.push(quote! { .push_param(#var_name) });
            }
        }
    }

    // 4. ComputeTask
    let expanded = quote! {
        #vis fn #fn_name<'a>(
            #(#fn_args),*
        ) -> gpgpru::__private::ComputeTask<'a> {
            static SPIRV: &[u32] = &[#(#spirv_u32),*];

            gpgpru::__private::ComputeTask::new(SPIRV, "main")
                #(#task_binds)*
        }
    };

    TokenStream::from(expanded)
}

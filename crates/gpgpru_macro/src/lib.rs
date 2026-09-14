// src/lib.rs

mod ir;
mod types;

mod entry;
mod lowering;

mod rust_emit;

pub(crate) use entry::{ComputeAttr, ParsedSignature, SystemBinding, parse_kernel_signature};
pub(crate) use ir::{ExprId, IrBlock, IrExpr, IrStmt, StmtId, VarId, VarInfo, VarStorage};
pub(crate) use types::{GpuType, LiteralKind};

pub(crate) use lowering::LoweringBuilder;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;

use rustc_hash::FxHasher;
use std::hash::Hasher;

use std::fs;
use std::path::PathBuf;

#[proc_macro_attribute]
pub fn compute(attr: TokenStream, item: TokenStream) -> TokenStream {
    match compute_impl(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.into_compile_error().into(),
    }
}

fn compute_impl(attr: TokenStream2, item_fn: TokenStream2) -> syn::Result<TokenStream2> {
    // Parse Attributes and Fn
    let compute_attr: ComputeAttr = syn::parse2(attr)?;
    let input_fn: syn::ItemFn = syn::parse2(item_fn.clone())?;

    // Fn Ident attributes
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_body = &input_fn.block;
    let fn_ident = &fn_sig.ident;

    // Signature Parsing
    let parsed_sig = parse_kernel_signature(&input_fn)?;

    // Lowering IR & CodeGen
    let mut builder = LoweringBuilder::new(&parsed_sig);
    let block_id: StmtId = builder.lower_block(fn_body)?;

    let slang_code: String = builder.emit_slang_code(
        fn_sig.ident.to_string().as_str(),
        &compute_attr,
        &parsed_sig,
        &block_id,
    );

    // Slang Code Hash
    let mut hasher = FxHasher::default();
    hasher.write("0.1.0".as_bytes());
    hasher.write("0.24.0".as_bytes());
    hasher.write(slang_code.as_bytes());

    let hash_u64 = hasher.finish();
    let hash_u32 = hash_u64 as u32; // Truncate last 32 bits

    let hash_str = format!("{:08x}", hash_u32);

    // Slang Code Cache Directory
    let is_rust_analyzer =
        std::env::var("RUST_ANALYZER").is_ok() || std::env::var("RA_RUST_ANALYZER").is_ok();

    let cache_dir = PathBuf::from("target/slang_cache");
    let _ = fs::create_dir_all(&cache_dir);

    let slang_path = cache_dir.join(format!(
        "{}_{}.slang",
        input_fn.sig.ident.to_string(),
        hash_str
    ));
    let spv_path = cache_dir.join(format!(
        "{}_{}.spv",
        input_fn.sig.ident.to_string(),
        hash_str
    ));

    // Slangc Compile
    if !spv_path.exists() && !is_rust_analyzer {
        let _ = fs::write(&slang_path, &slang_code);

        let spirv_code: Vec<u32> =
            rust_emit::compile_slang(&slang_code, &input_fn.sig.ident.to_string(), &hash_str)
                .map_err(|e| syn::Error::new(input_fn.sig.ident.span(), e.to_string()))?;

        let spirv_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                spirv_code.as_ptr() as *const u8,
                spirv_code.len() * std::mem::size_of::<u32>(),
            )
        };
        let _ = fs::write(&spv_path, spirv_bytes);
    }

    // Bytes Token
    let bytes_token = if spv_path.exists() {
        let spv_bytes = fs::read(&spv_path).map_err(|e| {
            syn::Error::new(
                input_fn.sig.ident.span(),
                format!("Failed to read SPV: {}", e),
            )
        })?;

        let spv_u32: Vec<u32> = spv_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();

        quote::quote! { &[#(#spv_u32),*] }
    } else {
        quote::quote! { &[] }
    };

    // Comment
    let orig_docs: Vec<_> = input_fn
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .collect();

    let mut param_doc_markdown = String::from("\n\n### Kernel Parameters\n");

    for (name, gpu_type) in &parsed_sig.pushed_fields {
        match gpu_type {
            GpuType::Slice { element, is_mut } => {
                let access_str = if *is_mut { "Read/Write" } else { "Read-Only" };
                param_doc_markdown.push_str(&format!(
                    "- **`{}`**: GPU Device Buffer (`{}`)\n  - **Access**: {}\n  - **Type**: `impl AsGpuBuffer<{}>` (BDA & Length auto-bound)\n",
                    name, element, access_str, element
                ));
            }
            _ => {
                param_doc_markdown.push_str(&format!(
                    "- **`{}`**: Uniform Push Constant (`{}`)\n",
                    name, gpu_type
                ));
            }
        }
    }

    // Generate Entry
    let (wrapper_args_token, struct_token) = rust_emit::generate_entry(&input_fn)?;

    let uppercase_name = fn_sig.ident.to_string().to_uppercase();
    let source_ident = quote::format_ident!("{}_SOURCE", uppercase_name);

    let args_ident = quote::quote!(args);

    let kernel_args_token = quote::quote!(KernelArgs);

    let generated_tokens = quote::quote! {
        #(#orig_docs)*
        #[doc = #param_doc_markdown]
        #fn_vis fn #fn_ident(#wrapper_args_token) -> gpgpru::__private::CommandBuilder<impl gpgpru::__private::HazardTrace + Copy> {
            #struct_token

            const #source_ident: &[u32] = #bytes_token;

            gpgpru::__private::CommandBuilder::<#kernel_args_token>::new(#source_ident, #args_ident).entry_point("main")
        }

        const _ : () = {
            #[allow(dead_code)]
            #item_fn
        };
    };

    // dbg!(builder);

    Ok(generated_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    use quote::ToTokens;

    #[test]
    fn test_slang_emit_success() {
        let attr_tokens: syn::Attribute = parse_quote! {
            #[threads(16, 16, 1)]
        };
        let input_fn: syn::ItemFn = parse_quote! {
            fn scale(data: &mut [f32], factor: f32, global_id_x: GlobalIdX) {
                if global_id_x < data.len() {
                    data[global_id_x] *= factor;
                }
            }
        };

        let attr = attr_tokens.meta.to_token_stream();
        let item_fn = input_fn.to_token_stream();

        let slang_code = compute_impl(attr, item_fn).expect("Oh no compile fail");

        println!("\n=== Generated Code ===\n{}", slang_code.to_string());
    }
}

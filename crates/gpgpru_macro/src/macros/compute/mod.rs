// src/macros/compute/mod.rs

mod argument;
mod attribute;
mod codegen;
mod entry;
mod slang_code;

pub use argument::*;
use attribute::*;
use entry::*;
use slang_code::*;

use crate::*;

use proc_macro2::TokenStream as TokenStream2;

use std::fs;
use std::path::PathBuf;

use rustc_hash::FxHasher;
use std::hash::Hasher;

pub fn compute_impl(attr: TokenStream2, item_fn: TokenStream2) -> syn::Result<TokenStream2> {
    // Parse Attributes and Fn
    let compute_attr: ComputeAttr = syn::parse2(attr)?;
    let input_fn: syn::ItemFn = syn::parse2(item_fn.clone())?;

    // Fn Ident attributes
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_body = &input_fn.block;
    let fn_ident = &fn_sig.ident;

    // Type Arena
    let mut arena = TypeArena::new();

    // Signature Parsing
    let parsed_sig = parse_kernel_signature(&mut arena, &input_fn)?;

    // Generate Entry
    let entry_info = generate_entry(&mut arena, &input_fn)?;

    let fn_generics_token = &entry_info.fn_generics_token;
    let where_clauses_token = &entry_info.where_clauses_token;
    let wrapper_args_token = &entry_info.wrapper_args_token;
    let struct_token = &entry_info.struct_token;
    let struct_info = &entry_info.struct_info;

    // Lowering IR & CodeGen
    let mut builder = LoweringBuilder::new(&mut arena, &parsed_sig);
    let block_id: StmtId = builder.lower_block(&mut arena, fn_body)?;

    let slang_code: String = builder.emit_slang_fn(
        fn_sig.ident.to_string().as_str(),
        &arena,
        &compute_attr,
        &struct_info,
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

    let cache_dir = PathBuf::from("target/gpgpru/shaders");
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
            compile_slang(&slang_code, &input_fn.sig.ident.to_string(), &hash_str)
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
        match arena.get(*gpu_type) {
            GpuType::Slice { element, is_mut } => {
                let access_str = if *is_mut { "Read/Write" } else { "Read-Only" };
                param_doc_markdown.push_str(&format!(
                    "- **`{}`**: GPU Device Buffer (`{}`)\n  - **Access**: {}\n  - **Type**: `impl AsGpuBuffer<{}>` (BDA & Length auto-bound)\n",
                    name, arena.to_rust_string(*element), access_str, arena.to_rust_string(*element)
                ));
            }
            _ => {
                param_doc_markdown.push_str(&format!(
                    "- **`{}`**: Uniform Push Constant (`{}`)\n",
                    name,
                    arena.to_rust_string(*gpu_type)
                ));
            }
        }
    }

    let uppercase_name = fn_sig.ident.to_string().to_uppercase();
    let source_ident = quote::format_ident!("{}_SOURCE", uppercase_name);

    let args_ident = quote::quote!(args);

    let kernel_args_token = quote::quote!(KernelArgs);

    let generated_tokens = quote::quote! {
        #(#orig_docs)*
        #[doc = #param_doc_markdown]
        #fn_vis fn #fn_ident<#fn_generics_token>(#wrapper_args_token) -> gpgpru::__private::CommandBuilder<
            'a,
            impl gpgpru::__private::HazardTrace + Copy + 'a,
        > where
            #where_clauses_token
        {
            #struct_token

            const #source_ident: &[u32] = #bytes_token;

            gpgpru::__private::CommandBuilder::<#kernel_args_token>::new(#source_ident, #args_ident).entry_point("main")
        }

        const _ : () = {
            #[allow(dead_code)]
            #item_fn
        };
    };

    Ok(generated_tokens)
}

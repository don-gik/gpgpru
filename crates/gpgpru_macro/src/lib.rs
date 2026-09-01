// src/lib.rs

mod arguments;
mod attribute;
mod slang_fields;
mod slang_transpile;

use arguments::parse_kernel_signature;
use attribute::ComputeAttr;
use slang_fields::{SlangField, get_field_bit_width, unpack_push_field};
use slang_transpile::build_slang_shader;

use std::collections::HashSet;
use std::ffi::CString;
use std::sync::atomic::{AtomicUsize, Ordering};

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use syn::{ItemFn, parse_macro_input};

use shader_slang::{
    Blob, CompileTarget, CompilerOptions, Downcast, GlobalSession, SessionDesc, TargetDesc,
};

static COMPUTE_MACRO_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[proc_macro_attribute]
pub fn compute(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    let unique_id = COMPUTE_MACRO_COUNTER.fetch_add(1, Ordering::SeqCst);

    match compute_impl(attr.into(), input_fn, unique_id) {
        Ok((tokens, _)) => tokens.into(),
        Err(err) => err.into_compile_error().into(),
    }
}

// #[proc_macro_attribute]
// pub fn pipeline(item: TokenStream) -> TokenStream {}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

fn blob_to_u32_slice(blob: &Blob) -> Vec<u32> {
    let byte_slice: &[u8] = blob.as_slice();
    let byte_len = byte_slice.len();

    assert!(
        byte_len % 4 == 0,
        "SPIR-V blob size must be a multiple of 4 bytes"
    );
    let u32_len = byte_len / 4;

    let raw_ptr = byte_slice.as_ptr() as *const u32;
    let spirv_slice: &[u32] = unsafe { std::slice::from_raw_parts(raw_ptr, u32_len) };

    spirv_slice.to_vec()
}

fn compile_slang(
    slang_code: &str,
    entry_point_name: &str,
    unique_id: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let metadata = cargo_metadata::MetadataCommand::new().exec()?;
    let target_dir = metadata.target_directory;

    let target_tmp_dir = std::path::Path::new(target_dir.as_str()).join("tmp");
    std::fs::create_dir_all(&target_tmp_dir)?;

    let file_name = format!("{}_{}.slang", entry_point_name, unique_id);
    let full_path = target_tmp_dir.join(&file_name);

    std::fs::write(&full_path, slang_code)?;

    let dir_path = full_path
        .parent()
        .ok_or("Parent directory cannot be found.")?;
    let c_dir_path = CString::new(dir_path.to_string_lossy().into_owned().into_bytes())?;
    let dir_ptr: *const std::os::raw::c_char = c_dir_path.as_ptr();

    let file_str = full_path
        .file_name()
        .ok_or("Failed to get file name")?
        .to_str()
        .ok_or("Invalid UTF-8 in file name")?;

    // Slang Compiler
    let global_session = match GlobalSession::new() {
        Some(session) => Ok(session),
        None => Err("Slang Shader Compiler Session could not be loaded."),
    }?;

    let session_options = CompilerOptions::default();

    let target_desc = TargetDesc::default()
        .format(CompileTarget::Spirv)
        .profile(global_session.find_profile("glsl_450"));

    let targets = [target_desc];
    let search_paths = [dir_ptr];

    let session_desc = SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let session = match global_session.create_session(&session_desc) {
        Some(session) => Ok(session),
        None => Err("Session could not be created."),
    }?;
    let module = session.load_module(file_str)?;
    let entry_point = match module.find_entry_point_by_name(entry_point_name) {
        Some(entry_point) => Ok(entry_point),
        None => Err("Entry Point not found"),
    }?;

    let program = session.create_composite_component_type(&[
        module.downcast().clone(),
        entry_point.downcast().clone(),
    ])?;

    let linked_program = program.link()?;

    let _reflection = linked_program.layout(0)?;
    let shader_bytecode = linked_program.entry_point_code(0, 0)?;

    Ok(blob_to_u32_slice(&shader_bytecode))
}

fn compute_impl(
    attr: TokenStream2,
    input_fn: ItemFn,
    unique_id: usize,
) -> syn::Result<(TokenStream2, Vec<SlangField>)> {
    let fn_entry = &input_fn.sig.ident.to_string();

    let attr_args: ComputeAttr = syn::parse2(attr)?;
    let (tx, ty, tz) = attr_args.threads;

    let signatures = parse_kernel_signature(&input_fn)?;

    let mut unpacked_fields = Vec::new();
    for (name, ty) in &signatures.pushed_fields {
        let fields = unpack_push_field(name, ty)?;
        unpacked_fields.extend(fields);
    }
    unpacked_fields.sort_by_key(|field| 256 - get_field_bit_width(field));
    let push_field_names: Vec<String> = unpacked_fields.iter().map(|f| f.name.clone()).collect();

    let struct_name = format!("{}Args", capitalize(&input_fn.sig.ident.to_string()));
    let slang_code = build_slang_shader(
        &struct_name,
        (tx, ty, tz),
        fn_entry,
        &unpacked_fields,
        &signatures.system_inputs,
        push_field_names,
        &input_fn.block,
    )?;

    let spirv_code = compile_slang(&slang_code, &fn_entry, unique_id).map_err(|err| {
        syn::Error::new_spanned(&input_fn, format!("Slang compilation failed: {}", err))
    })?;
    let spirv_bytes: Vec<u8> = spirv_code.iter().flat_map(|&x| x.to_ne_bytes()).collect();
    let byte_literal = syn::LitByteStr::new(&spirv_bytes, proc_macro2::Span::call_site());

    let system_input_names: HashSet<String> = signatures
        .system_inputs
        .iter()
        .map(|(name, _ty)| name.clone())
        .collect();

    let rust_struct_fields: Vec<&SlangField> = unpacked_fields
        .iter()
        .filter(|field| !system_input_names.contains(&field.name))
        .collect();

    // 1. Rewrite the inputs for the public wrapper signature
    let wrapper_inputs = input_fn.sig.inputs.iter().filter_map(|arg| {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = pat_ident.ident.to_string();

                if system_input_names.contains(&param_name) {
                    return None;
                }
            }

            let pat = &pat_type.pat;
            let ty = &pat_type.ty;

            let ty_str = quote::quote!(#ty).to_string().replace(' ', "");
            if ty_str.contains("[") {
                // Extract the primitive slice element type
                let elem_type = if ty_str.contains("f32") {
                    Some(quote::quote! { f32 })
                } else if ty_str.contains("u32") {
                    Some(quote::quote! { u32 })
                } else if ty_str.contains("i32") {
                    Some(quote::quote! { i32 })
                } else {
                    Some(quote::quote! { f32 })
                };

                // AsGpuBuffer Bound Trait
                Some(quote::quote! { #pat: &impl gpgpru::__private::AsGpuBuffer<#elem_type> })
            } else {
                // Keep regular types (e.g. f32)
                Some(quote::quote! { #arg })
            }
        } else {
            Some(quote::quote! { #arg })
        }
    });

    // 2. Build values for sorted assignments
    let struct_initializers = rust_struct_fields.iter().map(|field| {
        let field_ident = quote::format_ident!("{}", field.name);

        if field.name.ends_with("_____len") {
            // Find the original buffer (e.g., "data" from "data_len")
            let orig_buffer_name =
                quote::format_ident!("{}", field.name.trim_end_matches("_____len"));
            // Compute element count divide by item width (f32 = 4 bytes)
            quote::quote! { #field_ident: (#orig_buffer_name.size() / 4) as u32 }
        } else if field.slang_type.ends_with('*') {
            // Pass BDA
            let orig_ident = quote::format_ident!("{}", field.name);
            quote::quote! { #orig_ident: #orig_ident.base_device_address() }
        } else {
            // Map normal directly
            let orig_ident = quote::format_ident!("{}", field.name);
            quote::quote! { #orig_ident: #orig_ident }
        }
    });

    let struct_fields_tokens: Vec<proc_macro2::TokenStream> = unpacked_fields
        .iter()
        .map(|field| {
            let field_ident = quote::format_ident!("{}", field.name);
            let field_type = &field.rust_type;
            quote::quote! { pub #field_ident: #field_type }
        })
        .collect();

    let struct_name_str = format!("_____{}Args", capitalize(&input_fn.sig.ident.to_string()));
    let struct_ident = quote::format_ident!("{}", struct_name_str);
    let computetask_ident = quote::quote! { gpgpru::__private::CommandBuilder };

    let fn_vis = &input_fn.vis;
    let fn_name = &input_fn.sig.ident;

    let u32_len = spirv_code.len();
    let byte_len = spirv_bytes.len();
    let const_u8_name = quote::format_ident!("{}_SPIRV_U8_RAW", fn_name.to_string().to_uppercase());
    let const_spirv_name =
        quote::format_ident!("{}_SPIRV_BINARY", fn_name.to_string().to_uppercase());

    Ok((
        quote::quote! {
            #[repr(C)]
            #[derive(Copy, Clone, Debug)]
            struct #struct_ident {
                #(#struct_fields_tokens,)*
            }

            #fn_vis fn #fn_name(#(#wrapper_inputs),*) -> #computetask_ident<#struct_ident> {
                #[repr(C, align(4))]
                struct Aligned4<T: ?Sized>(T);

                static #const_u8_name: Aligned4<[u8; #byte_len]> = Aligned4(*#byte_literal);

                const #const_spirv_name: &[u32] = unsafe {
                    std::slice::from_raw_parts(
                        #const_u8_name.0.as_ptr() as *const u32,
                        #u32_len,
                    )
                };

                let argument = #struct_ident {
                    #(#struct_initializers,)*
                };

                gpgpru::__private::CommandBuilder::new(#const_spirv_name, argument)
                    .entry_point("main")
            }

            const _: () = {
                #[allow(dead_code)]
                #input_fn
                ()
            };
        },
        unpacked_fields,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::parse_quote;

    #[test]
    fn test_generate_scale_func() {
        let attr = quote! { threads(64, 1, 1) };
        let input_fn: syn::ItemFn = parse_quote! {
            fn scale(data: &mut [f32], factor: f32, global_id_x: GlobalIdX) {
                let idx = global_id_x;
                if idx < data.len()
                {
                    data[idx] *= factor;
                }
            }
        };

        let (tokens, unpacked_fields) = compute_impl(attr, input_fn, 0).unwrap();

        let tokens_str = tokens.to_string();
        let formatted = tokens_str.replace("\\n", "\n");

        dbg!(unpacked_fields);

        println!("Generated Tokens:\n{}", formatted);
    }

    #[test]
    fn test_generate_offset_func() {
        let attr = quote! { threads(64, 1, 1) };
        let input_fn: syn::ItemFn = parse_quote! {
            fn offset(data: &mut [f32], value: f32, global_id_x: GlobalIdX) {
               let idx = global_id_x;
               if idx < data.len() {
                  data[idx] += value;
               }
            }
        };

        let (tokens, unpacked_fields) = compute_impl(attr, input_fn, 0).unwrap();

        let tokens_str = tokens.to_string();
        let formatted = tokens_str.replace("\\n", "\n");

        dbg!(unpacked_fields);

        println!("Generated Tokens:\n{}", formatted);
    }
}

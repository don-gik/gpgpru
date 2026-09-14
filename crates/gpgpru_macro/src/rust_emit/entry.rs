// src/slang_emit/entry.rs

use crate::GpuType;
use crate::entry::parse_kernel_signature;

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};

struct FieldInfo {
    name: String,
    field_decl: TokenStream2,
    assignment: TokenStream2,
    align: usize,
}

pub fn generate_entry(item_fn: &syn::ItemFn) -> syn::Result<(TokenStream2, TokenStream2)> {
    let parsed = parse_kernel_signature(item_fn)?;

    let mut wrapper_args = Vec::new();
    let mut struct_fields_info = Vec::new();

    for (name, gpu_type) in &parsed.pushed_fields {
        let arg_ident = format_ident!("{}", name);

        match gpu_type {
            GpuType::Slice { element, is_mut } => {
                // &[T] -> &GpuBuffer<T>
                let elem_str = element.to_string();
                let elem_ty: syn::Type = syn::parse_str(&elem_str)?;

                let mut_ref = if *is_mut { quote!(&mut) } else { quote!(&) };
                wrapper_args.push(
                    quote! { #arg_ident: #mut_ref impl gpgpru::__private::AsGpuBuffer<#elem_ty> },
                );

                let ptr_field_name = format!("_gpu_{}_ptr", name);
                let ptr_field_ident = format_ident!("{}", ptr_field_name);
                let len_field_name = format!("_gpu_{}_len", name);
                let len_field_ident = format_ident!("{}", len_field_name);

                struct_fields_info.push(FieldInfo {
                    name: ptr_field_name,
                    field_decl: quote! { #ptr_field_ident: u64 },
                    assignment: quote! { #ptr_field_ident: #arg_ident.device_address() },
                    align: 8,
                });
                struct_fields_info.push(FieldInfo {
                    name: len_field_name,
                    field_decl: quote! { #len_field_ident: u32 },
                    assignment: quote! { #len_field_ident: #arg_ident.size() as u32 },
                    align: 4,
                });
            }

            _ => {
                let rust_ty_str = gpu_type.to_string();
                let rust_ty: syn::Type = syn::parse_str(&rust_ty_str)?;

                wrapper_args.push(quote! { #arg_ident: #rust_ty });
                struct_fields_info.push(FieldInfo {
                    name: name.clone(),
                    field_decl: quote! { #arg_ident: #rust_ty },
                    assignment: quote! { #arg_ident },
                    align: gpu_type.byte_size(),
                });
            }
        }
    }

    struct_fields_info.sort_by(|a, b| b.align.cmp(&a.align).then_with(|| a.name.cmp(&b.name)));

    let struct_fields = struct_fields_info.iter().map(|f| &f.field_decl);
    let field_assignments = struct_fields_info.iter().map(|f| &f.assignment);

    let fn_args_stream = quote! { #(#wrapper_args),* };

    let struct_stream = quote! {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct KernelArgs {
            #(#struct_fields),*
        }

        let args = KernelArgs {
            #(#field_assignments),*
        };
    };

    Ok((fn_args_stream, struct_stream))
}

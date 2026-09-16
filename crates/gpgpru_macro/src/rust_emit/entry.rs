// src/rust_emit/entry.rs

use crate::entry::parse_kernel_signature;
use crate::{GpuType, LiteralKind};

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};

pub struct FieldInfo {
    pub name: String,
    pub ty: GpuType,
    field_decl: TokenStream2,
    assignment: TokenStream2,
}

pub struct EntryInfo {
    pub fn_generics_token: TokenStream2,
    pub where_clauses_token: TokenStream2,
    pub wrapper_args_token: TokenStream2,
    pub struct_token: TokenStream2,
    pub struct_info: Vec<FieldInfo>,
}

pub fn generate_entry(item_fn: &syn::ItemFn) -> syn::Result<EntryInfo> {
    let parsed = parse_kernel_signature(item_fn)?;

    let mut wrapper_args = Vec::new();
    let mut struct_fields_info = Vec::new();
    let mut hazard_trace_calls = Vec::new();

    let mut fn_generics = vec![quote! { 'a }];
    let mut where_clauses = Vec::new();

    for (name, gpu_type) in &parsed.pushed_fields {
        let arg_ident = format_ident!("{}", name);

        match gpu_type {
            GpuType::Slice { element, is_mut } => {
                // &[T] -> &GpuBuffer<T>
                let elem_str = element.to_string();
                let elem_ty: syn::Type = syn::parse_str(&elem_str)?;

                let generic_ty = format_ident!("_T{}", fn_generics.len());
                fn_generics.push(quote! { #generic_ty });

                let mut_ref = if *is_mut {
                    quote!(&'a mut)
                } else {
                    quote!(&'a)
                };
                wrapper_args.push(quote! { #arg_ident: #mut_ref #generic_ty });

                where_clauses.push(quote! {
                    #generic_ty: gpgpru::__private::AsGpuBuffer<#elem_ty> + 'a
                });

                let ptr_field_name = format!("_gpu_{}_ptr", name);
                let ptr_field_ident = format_ident!("{}", ptr_field_name);
                let len_field_name = format!("_gpu_{}_len", name);
                let len_field_ident = format_ident!("{}", len_field_name);

                let access_enum = if *is_mut {
                    quote! { gpgpru::__private::AccessType::Write }
                } else {
                    quote! { gpgpru::__private::AccessType::Read }
                };
                hazard_trace_calls.push(quote! {
                    visitor(self.#ptr_field_ident, self.#len_field_ident as u64, #access_enum);
                });

                struct_fields_info.push(FieldInfo {
                    name: ptr_field_name,
                    ty: GpuType::Scalar(LiteralKind::UInt64),
                    field_decl: quote! { #ptr_field_ident: u64 },
                    assignment: quote! { #ptr_field_ident: #arg_ident.device_address() },
                });
                struct_fields_info.push(FieldInfo {
                    name: len_field_name,
                    ty: GpuType::Scalar(LiteralKind::UInt32),
                    field_decl: quote! { #len_field_ident: u32 },
                    assignment: quote! { #len_field_ident: #arg_ident.size() as u32 },
                });
            }

            _ => {
                let rust_ty_str = gpu_type.to_string();
                let rust_ty: syn::Type = syn::parse_str(&rust_ty_str)?;

                wrapper_args.push(quote! { #arg_ident: #rust_ty });
                struct_fields_info.push(FieldInfo {
                    name: name.clone(),
                    ty: gpu_type.clone(),
                    field_decl: quote! { #arg_ident: #rust_ty },
                    assignment: quote! { #arg_ident },
                });
            }
        }
    }

    struct_fields_info.sort_by(|a, b| {
        b.ty.byte_size()
            .cmp(&a.ty.byte_size())
            .then_with(|| a.name.cmp(&b.name))
    });

    let struct_fields = struct_fields_info.iter().map(|f| &f.field_decl);
    let field_assignments = struct_fields_info.iter().map(|f| &f.assignment);

    let wrapper_args_token = quote! { #(#wrapper_args),* };

    let struct_token = quote! {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct KernelArgs {
            #(#struct_fields),*
        }

        impl gpgpru::__private::HazardTrace for KernelArgs {
            fn trace_hazards(&self, visitor: &mut dyn FnMut(u64, u64, gpgpru::__private::AccessType)) {
                #(#hazard_trace_calls)*
            }
        }

        let args = KernelArgs {
            #(#field_assignments),*
        };
    };

    let fn_generics_token = quote! {
        #(#fn_generics),*
    };

    let where_clauses_token = quote! {
        #(#where_clauses)
        ,*
    };

    Ok(EntryInfo {
        fn_generics_token,
        where_clauses_token,
        wrapper_args_token,
        struct_token,
        struct_info: struct_fields_info,
    })
}

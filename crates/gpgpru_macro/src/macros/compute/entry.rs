use super::parse_kernel_signature;
use crate::LiteralKind;
use crate::{GpuType, TypeArena, TypeIdx};

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};

pub struct FieldInfo {
    pub name: String,
    pub alignment: usize,
    pub ty: TypeIdx,
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

fn type_idx_to_rust_tokens(arena: &TypeArena, idx: TypeIdx) -> TokenStream2 {
    match arena.get(idx) {
        GpuType::Scalar(LiteralKind::Float32) => quote! { f32 },
        GpuType::Scalar(LiteralKind::Int32) => quote! { i32 },
        GpuType::Scalar(LiteralKind::UInt32) => quote! { u32 },
        GpuType::Scalar(LiteralKind::Bool) => quote! { bool },

        GpuType::Vector { element, len } => match (element, len) {
            (LiteralKind::Float32, 2) => quote! { glam::Vec2 },
            (LiteralKind::Float32, 3) => quote! { glam::Vec3 },
            (LiteralKind::Float32, 4) => quote! { glam::Vec4 },
            (LiteralKind::Int32, 2) => quote! { glam::IVec2 },
            (LiteralKind::UInt32, 2) => quote! { glam::UVec2 },
            _ => panic!("Unsupported vector type"),
        },

        GpuType::Matrix { rows, cols, .. } => match (rows, cols) {
            (3, 3) => quote! { glam::Mat3 },
            (4, 4) => quote! { glam::Mat4 },
            _ => panic!("Unsupported matrix type"),
        },

        _ => panic!("Cannot directly convert this GPU type to a Rust host type."),
    }
}

pub fn generate_entry(arena: &mut TypeArena, item_fn: &syn::ItemFn) -> syn::Result<EntryInfo> {
    let parsed = parse_kernel_signature(arena, item_fn)?;

    let mut wrapper_args = Vec::new();
    let mut struct_fields_info = Vec::new();
    let mut hazard_trace_calls = Vec::new();

    let mut fn_generics = vec![quote! { 'a }];
    let mut where_clauses = Vec::new();

    for (name, type_idx) in &parsed.pushed_fields {
        let name_str = name.as_ref();
        let arg_ident = format_ident!("{}", name_str);

        match arena.get(*type_idx) {
            GpuType::Slice { element, is_mut } => {
                // &[T] -> &GpuBuffer<T>
                let elem_ty = type_idx_to_rust_tokens(arena, *element);

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

                let ptr_field_name = format!("_gpu_{}_ptr", name_str);
                let ptr_field_ident = format_ident!("{}", ptr_field_name);
                let len_field_name = format!("_gpu_{}_len", name_str);
                let len_field_ident = format_ident!("{}", len_field_name);

                let access_enum = if *is_mut {
                    quote! { gpgpru::__private::AccessType::Write }
                } else {
                    quote! { gpgpru::__private::AccessType::Read }
                };

                hazard_trace_calls.push(quote! {
                    visitor(self.#ptr_field_ident, self.#len_field_ident as u64, #access_enum);
                });

                // _ptr
                struct_fields_info.push(FieldInfo {
                    name: ptr_field_name,
                    alignment: 8,
                    ty: arena.alloc(GpuType::Ptr {
                        target: *element,
                        is_mut: *is_mut,
                    }),
                    field_decl: quote! { #ptr_field_ident: u64 },
                    assignment: quote! { #ptr_field_ident: #arg_ident.device_address() },
                });

                // _len
                struct_fields_info.push(FieldInfo {
                    name: len_field_name,
                    alignment: 4,
                    ty: arena.alloc(GpuType::Scalar(LiteralKind::UInt32)),
                    field_decl: quote! { #len_field_ident: u32 },
                    assignment: quote! { #len_field_ident: #arg_ident.size() as u32 },
                });
            }

            _ => {
                let rust_ty = type_idx_to_rust_tokens(arena, *type_idx);
                let alignment = arena.alignment(*type_idx);

                wrapper_args.push(quote! { #arg_ident: #rust_ty });
                struct_fields_info.push(FieldInfo {
                    name: name_str.to_string(),
                    alignment,
                    ty: *type_idx,
                    field_decl: quote! { #arg_ident: #rust_ty },
                    assignment: quote! { #arg_ident },
                });
            }
        }
    }

    struct_fields_info.sort_by(|a, b| {
        b.alignment
            .cmp(&a.alignment)
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

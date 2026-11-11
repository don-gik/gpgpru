use super::*;

use syn::visit_mut::VisitMut;

struct LifetimeChanger {
    lifetime: syn::Lifetime,
}

impl VisitMut for LifetimeChanger {
    fn visit_type_reference_mut(&mut self, i: &mut syn::TypeReference) {
        syn::visit_mut::visit_type_reference_mut(self, i);

        if i.mutability.is_some() {
            i.lifetime = Some(self.lifetime.clone());
        }
    }
}

fn apply_lifetime_to_mut_refs(ty: &syn::Type, lifetime_str: &str) -> syn::Type {
    let mut cloned_ty = ty.clone();
    let mut changer = LifetimeChanger {
        lifetime: syn::parse_str(lifetime_str).unwrap(),
    };
    changer.visit_type_mut(&mut cloned_ty);
    cloned_ty
}

fn parse_rust_math_type(ty: &syn::Type) -> Option<(raw::ScalarKind, u8, u8)> {
    match ty {
        // [T; N] Array (e.g. [f32; 3])
        syn::Type::Array(arr) => {
            let (scalar, r, c) = parse_rust_math_type(&arr.elem)?;
            if let syn::Expr::Lit(lit) = &arr.len {
                if let syn::Lit::Int(lit_int) = &lit.lit {
                    let len: u8 = lit_int.base10_parse().ok()?;
                    return Some((scalar, r, c * len));
                }
            }
            None
        }

        // Ident / Path (f32, glam::Vec3, Mat4 etc.)
        syn::Type::Path(type_path) => {
            let ident = type_path.path.segments.last()?.ident.to_string();
            match ident.as_str() {
                // Scalar
                "f32" => Some((raw::ScalarKind::Float32, 1, 1)),
                "u32" => Some((raw::ScalarKind::UInt32, 1, 1)),
                "i32" => Some((raw::ScalarKind::Int32, 1, 1)),
                "f64" => Some((raw::ScalarKind::Float64, 1, 1)),
                "u64" => Some((raw::ScalarKind::UInt64, 1, 1)),
                "i64" => Some((raw::ScalarKind::Int64, 1, 1)),
                "bool" => Some((raw::ScalarKind::Bool, 1, 1)),

                // Float Vectors (glam, nalgebra, cgmath, mint)
                "Vec2" | "Point2" => Some((raw::ScalarKind::Float32, 1, 2)),
                "Vec3" | "Point3" | "Vec3A" => Some((raw::ScalarKind::Float32, 1, 3)),
                "Vec4" | "Point4" | "Quat" => Some((raw::ScalarKind::Float32, 1, 4)),

                // Uint Vectors
                "UVec2" => Some((raw::ScalarKind::UInt32, 1, 2)),
                "UVec3" => Some((raw::ScalarKind::UInt32, 1, 3)),
                "UVec4" => Some((raw::ScalarKind::UInt32, 1, 4)),

                // Int Vectors
                "IVec2" => Some((raw::ScalarKind::Int32, 1, 2)),
                "IVec3" => Some((raw::ScalarKind::Int32, 1, 3)),
                "IVec4" => Some((raw::ScalarKind::Int32, 1, 4)),

                // Bool Vectors
                "BVec2" => Some((raw::ScalarKind::Bool, 1, 2)),
                "BVec3" => Some((raw::ScalarKind::Bool, 1, 3)),
                "BVec4" => Some((raw::ScalarKind::Bool, 1, 4)),

                // Matrices
                "Mat2" => Some((raw::ScalarKind::Float32, 2, 2)),
                "Mat3" => Some((raw::ScalarKind::Float32, 3, 3)),
                "Mat4" => Some((raw::ScalarKind::Float32, 4, 4)),

                _ => None, // Else, just pass em.
            }
        }
        _ => None,
    }
}

fn verify_primitive_or_math_type(
    ty: &syn::Type,
    member: &CachedMember,
    arg_span: proc_macro2::Span,
) -> syn::Result<()> {
    let Some((rust_scalar, rust_rows, rust_cols)) = parse_rust_math_type(ty) else {
        return Ok(()); // Custom (or complex slang structs) are passed (size_of)
    };

    let slang_scalar = member.scalar_type;
    let slang_rows = member.row_count.max(1);
    let slang_cols = member.col_count.max(1);

    // 1. Scalar element type (Float32 vs Uint32 vs Int32)
    if rust_scalar != slang_scalar {
        return Err(syn::Error::new(
            arg_span,
            format!(
                "Type scalar mismatch for field '{}': Rust uses '{:?}', but Slang shader expects '{:?}'.",
                member.name, rust_scalar, slang_scalar
            ),
        ));
    }

    // 2. Element dimension (Vector dim / Matrix size)
    let rust_total_components = rust_rows * rust_cols;
    let slang_total_components = slang_rows * slang_cols;

    if rust_total_components != slang_total_components {
        return Err(syn::Error::new(
            arg_span,
            format!(
                "Dimension mismatch for field '{}': Rust definition has {} components, but Slang shader expects {} ({}x{}).",
                member.name, rust_total_components, slang_total_components, slang_rows, slang_cols
            ),
        ));
    }

    Ok(())
}

fn extract_bda_element_type(ty: &syn::Type) -> syn::Result<syn::Type> {
    let syn::Type::Reference(type_ref) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            "BDA pointer must be in &, &mut reference type.",
        ));
    };

    let elem_ty = &*type_ref.elem;

    match elem_ty {
        // &impl AsGpuBuffer<T>
        syn::Type::ImplTrait(impl_trait) => {
            for bound in &impl_trait.bounds {
                if let syn::TypeParamBound::Trait(tb) = bound {
                    if let Some(segment) = tb.path.segments.last() {
                        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                            if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                                return Ok(inner.clone());
                            }
                        }
                    }
                }
            }
        }
        // &GpuBuffer<T>, &GpuBuffer<T, Loc>, &BufferSlice<'a, T, Loc> 등
        syn::Type::Path(type_path) => {
            if let Some(segment) = type_path.path.segments.last() {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(inner) = args.args.iter().find_map(|arg| match arg {
                        syn::GenericArgument::Type(t) => Some(t),
                        _ => None,
                    }) {
                        return Ok(inner.clone());
                    }
                }
            }
        }
        // &[T]
        syn::Type::Slice(slice) => return Ok(*slice.elem.clone()),
        _ => {}
    }

    Err(syn::Error::new_spanned(
        ty,
        "Cannot infer T from BDA Pointer (e.g. &GpuBuffer<Vertex> or &BufferSlice<Vertex>)",
    ))
}

pub fn kernel_impl(attr: TokenStream2, body: TokenStream2) -> syn::Result<TokenStream2> {
    let args: KernelArgs = syn::parse2(attr)?;
    let slang_path_str = args.slang_path.value();
    let entry_point_lit = args
        .entry_point
        .unwrap_or_else(|| syn::LitStr::new("main", args.slang_path.span()));
    let high_precision = args.high_precision;

    let input_fn: syn::ItemFn = syn::parse2(body)?;
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;

    // ----------------------------------------------------------------------------------
    // 1. Slang Compile + Reflection
    // ----------------------------------------------------------------------------------
    let entry_point = entry_point_lit.value();

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let absolute_slang_path = Utf8PathBuf::from(manifest_dir).join(&slang_path_str);

    if !absolute_slang_path.exists() {
        return Err(syn::Error::new(
            args.slang_path.span(),
            format!(
                "Shader file not found at '{}'. (Check relative path from crate root)",
                absolute_slang_path
            ),
        ));
    }

    let reflection = ShaderCache::get_or_compile(
        &fn_name.to_string(),
        &absolute_slang_path,
        &entry_point,
        high_precision,
    )
    .map_err(|err| syn::Error::new(input_fn.sig.ident.span(), err))?;

    let pc_resource = reflection.push_constant();
    let total_size = pc_resource.map(|r| r.total_size as usize).unwrap_or(0);
    let spirv_words = &reflection.spirv_bytes;

    // ----------------------------------------------------------------------------------
    // 2. Collect Members & Find offsets
    // ----------------------------------------------------------------------------------
    let mut fn_args = Vec::new();
    let mut param_copies = Vec::new();
    let mut hazard_traces = Vec::new();
    let mut const_asserts = Vec::new();
    let mut raw_arg_tys = Vec::new();

    let mut params_fields = Vec::new();
    let mut params_inits = Vec::new();
    let mut bda_count = 0usize;

    for arg in &input_fn.sig.inputs {
        let syn::FnArg::Typed(pat_type) = arg else {
            continue;
        };
        let syn::Pat::Ident(pat_ident) = &*pat_type.pat else {
            continue;
        };

        let arg_name = &pat_ident.ident;
        let arg_ty = &pat_type.ty;
        let arg_str = arg_name.to_string();

        raw_arg_tys.push(arg_ty);

        // Search for that member in slang reflection
        let (member_idx, member) = reflection.find_member(&arg_str).ok_or_else(|| {
            syn::Error::new(
                arg_name.span(),
                format!("Cannot find field '{arg_str}' in slang shader."),
            )
        })?;

        let offset = member.offset as usize;
        let slang_size = member.size as usize;

        match member.kind {
            // --------------------------------------------------------------------------
            // A. BDA Pointer (&impl AsGpuBuffer<T> / &mut impl AsGpuBuffer<T>)
            // --------------------------------------------------------------------------
            raw::MemberKind::Pointer => {
                let syn::Type::Reference(type_ref) = &**arg_ty else {
                    return Err(syn::Error::new_spanned(
                        arg_ty,
                        format!("BDA argument '{arg_str}' must be a reference (& or &mut)."),
                    ));
                };

                let inner_ty = extract_bda_element_type(arg_ty)?;
                let is_mut = type_ref.mutability.is_some();

                let bda_size_field = quote::format_ident!("bda_size_{}", bda_count);
                bda_count += 1;

                let transformed_arg_ty = apply_lifetime_to_mut_refs(arg_ty, "'a");

                fn_args.push(quote::quote! {
                    #arg_name: #transformed_arg_ty
                });

                params_fields.push(quote::quote! { #bda_size_field: u64 });
                params_inits.push(quote::quote! { #bda_size_field: #arg_name.size() });

                const_asserts.push(quote::quote! {
                    assert!(
                        8 == #slang_size,
                        concat!("Size of a BDA pointer '", #arg_str, "' must be 8 bytes.")
                    );
                });

                for child in reflection.child_members(member_idx as i32) {
                    let child_ident = quote::format_ident!("{}", child.name);
                    let expected_offset = child.offset as usize;

                    const_asserts.push(quote::quote! {
                        assert!(
                            ::core::mem::offset_of!(#inner_ty, #child_ident) == #expected_offset,
                            concat!(
                                "Field '", stringify!(#child_ident),
                                "' of BDA Struct '", stringify!(#inner_ty),
                                "' has different offset with slang shader (", #expected_offset, "B)."
                            )
                        );
                    });
                }

                let end = offset + 8;
                param_copies.push(quote::quote! {
                    let bda_val = #arg_name.bda();
                    ___push_constant_bytes[#offset..#end].copy_from_slice(&bda_val.to_ne_bytes());
                });

                let access = if is_mut {
                    quote::quote! { gpgpru::__private::AccessType::Write }
                } else {
                    quote::quote! { gpgpru::__private::AccessType::Read }
                };

                hazard_traces.push(quote::quote! {
                    let bda = u64::from_ne_bytes(self.raw[#offset..#end].try_into().unwrap());
                    visitor(bda, self.#bda_size_field, #access);
                });
            }

            // --------------------------------------------------------------------------
            // B. User-defined Struct (Direct Struct)
            // --------------------------------------------------------------------------
            raw::MemberKind::Struct => {
                fn_args.push(quote::quote! { #arg_name: #arg_ty });

                // Size validation
                const_asserts.push(quote::quote! {
                    assert!(
                        ::core::mem::size_of::<#arg_ty>() == #slang_size,
                        concat!(
                            "Struct '", stringify!(#arg_ty),
                            "' size mismatch with Slang shader (expected ", #slang_size, "B)."
                        )
                    );
                });

                // Field validation
                for child in reflection.child_members(member_idx as i32) {
                    let child_ident = quote::format_ident!("{}", child.name);
                    let expected_offset = child.offset as usize;

                    const_asserts.push(quote::quote! {
                        assert!(
                            ::core::mem::offset_of!(#arg_ty, #child_ident) == #expected_offset,
                            concat!(
                                "Field '", stringify!(#child_ident),
                                "' of Struct '", stringify!(#arg_ty),
                                "' has different offset with slang shader (", #expected_offset, "B)."
                            )
                        );
                    });
                }

                // 3) bytemuck Zero-Copy
                param_copies.push(quote::quote! {
                    let struct_bytes = ::bytemuck::bytes_of(&#arg_name);
                    let end = #offset + struct_bytes.len();
                    ___push_constant_bytes[#offset..end].copy_from_slice(struct_bytes);
                });
            }

            // --------------------------------------------------------------------------
            // C. Primitive Scalar / Matrix (f32, u32, Mat4, etc.)
            // --------------------------------------------------------------------------
            raw::MemberKind::Primitive => {
                fn_args.push(quote::quote! { #arg_name: #arg_ty });

                verify_primitive_or_math_type(arg_ty, member, arg_name.span())?;

                // Size validation
                const_asserts.push(quote::quote! {
                    assert!(
                        ::core::mem::size_of::<#arg_ty>() == #slang_size,
                        concat!(
                            "Primitive type '", stringify!(#arg_ty),
                            "' size mismatch with Slang shader (expected ", #slang_size, "B)."
                        )
                    );
                });

                // bytemuck Zero-Copy
                param_copies.push(quote::quote! {
                    let scalar_bytes = gpgpru::__private::bytemuck::bytes_of(&#arg_name);
                    let end = #offset + scalar_bytes.len();
                    ___push_constant_bytes[#offset..end].copy_from_slice(scalar_bytes);
                });
            }
        }
    }

    // ----------------------------------------------------------------------------------
    // 3. Final AST Code Generation
    // ----------------------------------------------------------------------------------
    let root_include_path = absolute_slang_path.as_str();

    let dep_includes = reflection
        .dependencies
        .iter()
        .filter(|dep_path| dep_path.as_str() != root_include_path)
        .map(|dep_path| {
            quote::quote! {
                const _: &str = include_str!(#dep_path);
            }
        });

    Ok(quote::quote! {
        // For telling cargo to recompile
        const _: &str = include_str!(#root_include_path);
        #(#dep_includes)*

        // Compile time validation
        const _: () = {
            #(#const_asserts)*

            #[allow(dead_code, unused_variables)]
            fn _use_types(#(_: #raw_arg_tys),*) {}
        };

        #[allow(non_snake_case)]
        #fn_vis fn #fn_name<'a>(
            #(#fn_args),*
        ) -> gpgpru::__private::CommandBuilder<'a, impl gpgpru::__private::HazardTrace + Copy + 'static> {
            static SPIRV_CODE: &[u32] = &[#(#spirv_words),*];

            let mut ___push_constant_bytes = [0u8; #total_size];
            #(#param_copies)*

            #[repr(C)]
            #[derive(Copy, Clone)]
            struct Params {
                raw: [u8; #total_size],
                #(#params_fields),*
            }

            impl gpgpru::__private::HazardTrace for Params {
                fn trace_hazards(&self, visitor: &mut dyn FnMut(u64, u64, gpgpru::__private::AccessType)) {
                    #(#hazard_traces)*
                }
            }

            gpgpru::__private::CommandBuilder::new(
                SPIRV_CODE,
                Params {
                    raw: ___push_constant_bytes,
                    #(#params_inits),*
                }
            )
        }
    })
}

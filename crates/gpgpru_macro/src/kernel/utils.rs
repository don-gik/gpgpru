use super::*;

struct LifetimeInjector {
    lifetime: syn::Lifetime,
}

impl syn::visit_mut::VisitMut for LifetimeInjector {
    fn visit_type_reference_mut(&mut self, i: &mut syn::TypeReference) {
        // recursively visit the inner type
        syn::visit_mut::visit_type_reference_mut(self, i);

        // Only apply the lifetime if it is a mutable reference without an existing lifetime
        if i.mutability.is_some() && i.lifetime.is_none() {
            i.lifetime = Some(self.lifetime.clone());
        }
    }
}

pub fn apply_lifetime_to_mut_refs(ty: syn::Type, lifetime_str: &str) -> syn::Type {
    use syn::visit_mut::VisitMut;

    let lifetime: syn::Lifetime = syn::parse_str(lifetime_str).expect("Invalid lifetime string");
    let mut ty = ty;
    let mut injector = LifetimeInjector { lifetime };
    injector.visit_type_mut(&mut ty);
    ty
}

// ------------------------------------------------------------------------------------------

pub(crate) fn parse_rust_math_type(ty: &syn::Type) -> Option<(raw::ScalarKind, u8, u8)> {
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

pub(crate) fn verify_primitive_or_math_type(
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

pub(crate) fn extract_bda_element_type(ty: &syn::Type) -> syn::Result<syn::Type> {
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

// ------------------------------------------------------------------------------------------

pub(crate) fn format_slang_warnings(raw_warning: &str) -> String {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let prefix_with_slash = if manifest_dir.is_empty() {
        String::new()
    } else {
        format!("{}/", manifest_dir.trim_end_matches('/'))
    };

    raw_warning
        .trim()
        .lines()
        .map(|line| {
            let cleaned = if !prefix_with_slash.is_empty() {
                line.replace(&prefix_with_slash, "")
            } else {
                line.to_string()
            };

            let trimmed = cleaned.trim_start();
            if trimmed.starts_with("warning[") || trimmed.starts_with("warning:") {
                let indent = &cleaned[..cleaned.len() - trimmed.len()];
                format!("{indent}\x1b[1;33m{trimmed}\x1b[0m")
            } else if trimmed.starts_with("note[") || trimmed.starts_with("note:") {
                let indent = &cleaned[..cleaned.len() - trimmed.len()];
                format!("{indent}\x1b[1;36m{trimmed}\x1b[0m")
            } else {
                cleaned
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

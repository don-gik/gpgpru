// src/types/lower.rs

use super::{GpuType, TypeArena, TypeIdx};
use crate::LiteralKind;
use syn::spanned::Spanned;

pub enum MethodAction {
    AsFunction(GpuType, &'static str),
    AsMethod(GpuType, &'static str),
}

impl TypeArena {
    pub fn lower_type(&mut self, syn_type: &syn::Type) -> syn::Result<TypeIdx> {
        let span = syn_type.span();

        match syn_type {
            syn::Type::Path(type_path) => {
                let segment = type_path
                    .path
                    .segments
                    .last()
                    .ok_or_else(|| syn::Error::new(span, "Invalid type path"))?;

                let ident_str = segment.ident.to_string();

                let ty = match ident_str.as_str() {
                    "f32" => return Ok(self.f32_ty),
                    "i32" => return Ok(self.i32_ty),
                    "u32" => return Ok(self.u32_ty),
                    "bool" => return Ok(self.bool_ty),

                    "Vec2" => GpuType::Vector {
                        element: LiteralKind::Float32,
                        len: 2,
                    },
                    "Vec3" => GpuType::Vector {
                        element: LiteralKind::Float32,
                        len: 3,
                    },
                    "Vec4" => GpuType::Vector {
                        element: LiteralKind::Float32,
                        len: 4,
                    },
                    "IVec2" => GpuType::Vector {
                        element: LiteralKind::Int32,
                        len: 2,
                    },
                    "UVec2" => GpuType::Vector {
                        element: LiteralKind::UInt32,
                        len: 2,
                    },

                    "Mat3" => GpuType::Matrix {
                        element: LiteralKind::Float32,
                        rows: 3,
                        cols: 3,
                    },
                    "Mat4" => GpuType::Matrix {
                        element: LiteralKind::Float32,
                        rows: 4,
                        cols: 4,
                    },

                    _ => {
                        return Err(syn::Error::new(
                            span,
                            format!("Unsupported type: {ident_str}"),
                        ));
                    }
                };

                Ok(self.alloc(ty))
            }

            syn::Type::Reference(type_ref) => {
                if let syn::Type::Slice(type_slice) = &*type_ref.elem {
                    let elem_idx = self.lower_type(&type_slice.elem)?;
                    let is_mut = type_ref.mutability.is_some();

                    Ok(self.alloc(GpuType::Slice {
                        element: elem_idx,
                        is_mut,
                    }))
                } else {
                    let target_idx = self.lower_type(&type_ref.elem)?;
                    let is_mut = type_ref.mutability.is_some();

                    Ok(self.alloc(GpuType::Ptr {
                        target: target_idx,
                        is_mut,
                    }))
                }
            }

            _ => Err(syn::Error::new(span, "Unsupported type syntax")),
        }
    }
}

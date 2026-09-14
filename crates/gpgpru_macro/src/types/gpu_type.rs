// src/types/gpu_type.rs

use super::{LiteralKind, ToSlangString};

use syn::spanned::Spanned;

#[derive(Debug, Clone, PartialEq)]
pub enum GpuType {
    /// f32, u32, ...
    Scalar(LiteralKind),

    /// Vec2, Vec3, Vec4, IVec2
    Vector { element: LiteralKind, len: u8 },

    /// Mat3, Mat4
    Matrix {
        element: LiteralKind,
        rows: u8,
        cols: u8,
    },

    /// &[T]
    Slice { element: Box<GpuType>, is_mut: bool },
}

impl TryFrom<&syn::Type> for GpuType {
    type Error = syn::Error;

    fn try_from(value: &syn::Type) -> Result<Self, Self::Error> {
        let span = value.span();

        match value {
            syn::Type::Path(type_path) => {
                let segment = type_path
                    .path
                    .segments
                    .last()
                    .ok_or_else(|| syn::Error::new(span, "Invalid type path"))?;

                let ident_str = segment.ident.to_string();

                match ident_str.as_str() {
                    "f32" => Ok(GpuType::Scalar(LiteralKind::Float32)),
                    "f64" => Ok(GpuType::Scalar(LiteralKind::Float64)),
                    "i32" => Ok(GpuType::Scalar(LiteralKind::Int32)),
                    "i64" => Ok(GpuType::Scalar(LiteralKind::Int64)),
                    "u32" => Ok(GpuType::Scalar(LiteralKind::UInt32)),
                    "u64" => Ok(GpuType::Scalar(LiteralKind::UInt64)),
                    "bool" => Ok(GpuType::Scalar(LiteralKind::Bool)),

                    "Vec2" => Ok(GpuType::Vector {
                        element: LiteralKind::Float32,
                        len: 2,
                    }),
                    "Vec3" => Ok(GpuType::Vector {
                        element: LiteralKind::Float32,
                        len: 3,
                    }),
                    "Vec4" => Ok(GpuType::Vector {
                        element: LiteralKind::Float32,
                        len: 4,
                    }),
                    "IVec2" => Ok(GpuType::Vector {
                        element: LiteralKind::Int32,
                        len: 2,
                    }),
                    "UVec2" => Ok(GpuType::Vector {
                        element: LiteralKind::UInt32,
                        len: 2,
                    }),

                    "Mat3" => Ok(GpuType::Matrix {
                        element: LiteralKind::Float32,
                        rows: 3,
                        cols: 3,
                    }),
                    "Mat4" => Ok(GpuType::Matrix {
                        element: LiteralKind::Float32,
                        rows: 4,
                        cols: 4,
                    }),

                    _ => Err(syn::Error::new(
                        span,
                        format!("Unsupported GPU type `{}`.", ident_str),
                    )),
                }
            }

            syn::Type::Reference(type_ref) => {
                if let syn::Type::Slice(type_slice) = &*type_ref.elem {
                    let elem_gpu_type = GpuType::try_from(&*type_slice.elem)?;

                    Ok(GpuType::Slice {
                        element: Box::new(elem_gpu_type),
                        is_mut: type_ref.mutability.is_some(),
                    })
                } else {
                    Err(syn::Error::new(
                        span,
                        "Only slices (&[T], &mut [T]) are supported for reference types.",
                    ))
                }
            }

            _ => Err(syn::Error::new(span, "Unsupported type syntax.")),
        }
    }
}

impl std::fmt::Display for GpuType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            GpuType::Scalar(lit) => lit.to_string(),

            GpuType::Vector { element, len } => match (element, len) {
                (LiteralKind::Float32, 2) => "glam::Vec2".to_string(),
                (LiteralKind::Float32, 3) => "glam::Vec3".to_string(),
                (LiteralKind::Float32, 4) => "glam::Vec4".to_string(),
                (LiteralKind::Int32, 2) => "glam::IVec2".to_string(),
                (LiteralKind::UInt32, 2) => "glam::UVec2".to_string(),
                _ => {
                    let elem = element.to_string();
                    let l = *len as usize;
                    format!("[{}; {}]", elem, l).to_string()
                }
            },

            GpuType::Matrix {
                rows,
                cols,
                element,
            } => match (rows, cols) {
                (3, 3) => "glam::Mat3".to_string(),
                (4, 4) => "glam::Mat4".to_string(),
                _ => {
                    let elem = element.to_string();
                    let l = (rows * cols) as usize;
                    format!("[{}; {}]", elem, l).to_string()
                }
            },

            GpuType::Slice { element, is_mut } => {
                if *is_mut {
                    format!("&mut [{}]", element.to_string()).to_string()
                } else {
                    format!("&[{}]", element.to_string())
                }
            }
        };

        write!(f, "{}", s)
    }
}

impl ToSlangString for GpuType {
    fn to_slang_string(&self) -> String {
        match self {
            GpuType::Scalar(scalar) => scalar.to_slang_string(),

            GpuType::Vector { element, len } => {
                let prefix = match element {
                    LiteralKind::Float32 => "float",
                    LiteralKind::Int32 => "int",
                    LiteralKind::UInt32 => "uint",
                    _ => "float",
                };
                format!("{}{}", prefix, len)
            }

            GpuType::Matrix { rows, cols, .. } => {
                format!("float{}x{}", rows, cols)
            }

            GpuType::Slice { element, .. } => {
                format!("Slice<{}>", element.to_slang_string())
            }
        }
    }
}

impl GpuType {
    pub fn byte_size(&self) -> usize {
        match self {
            GpuType::Scalar(lit) => lit.byte_size(),
            GpuType::Vector { element, len } => element.byte_size() * (*len as usize),
            GpuType::Matrix {
                rows,
                cols,
                element,
            } => element.byte_size() * (*rows as usize) * (*cols as usize),
            GpuType::Slice { .. } => 12,
        }
    }
}

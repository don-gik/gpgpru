// src/types/literal.rs

use super::ToSlangString;

use std::fmt;
use syn::spanned::Spanned;

#[derive(Debug, Clone, PartialEq)]
pub enum LiteralKind {
    Float32,
    Float64,
    Int32,
    Int64,
    UInt32,
    UInt64,
    Bool,
}

impl TryFrom<&syn::Type> for LiteralKind {
    type Error = syn::Error;

    fn try_from(value: &syn::Type) -> Result<Self, Self::Error> {
        let span = value.span();

        if let syn::Type::Path(type_path) = value {
            if let Some(segment) = type_path.path.segments.last() {
                let ident_str = segment.ident.to_string();

                return match ident_str.as_str() {
                    "f32" => Ok(LiteralKind::Float32),
                    "f64" => Ok(LiteralKind::Float64),
                    "i32" => Ok(LiteralKind::Int32),
                    "i64" => Ok(LiteralKind::Int64),
                    "u32" => Ok(LiteralKind::UInt32),
                    "u64" => Ok(LiteralKind::UInt64),
                    "bool" => Ok(LiteralKind::Bool),
                    _ => Err(syn::Error::new(
                        span,
                        format!("Unsupported primitive type `{}`.", ident_str),
                    )),
                };
            }
        }

        Err(syn::Error::new(
            span,
            "Expected a primitive type (e.g., f32, u32).",
        ))
    }
}

impl fmt::Display for LiteralKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            LiteralKind::Float32 => "f32",
            LiteralKind::Float64 => "f64",
            LiteralKind::Int32 => "i32",
            LiteralKind::Int64 => "i64",
            LiteralKind::UInt32 => "u32",
            LiteralKind::UInt64 => "u64",
            LiteralKind::Bool => "bool",
        };
        write!(f, "{}", s)
    }
}

impl ToSlangString for LiteralKind {
    fn to_slang_string(&self) -> String {
        match self {
            Self::Float32 => "float".to_string(),
            Self::Float64 => "double".to_string(),
            Self::Int32 => "int".to_string(),
            Self::Int64 => "int64_t".to_string(),
            Self::UInt32 => "uint".to_string(),
            Self::UInt64 => "uint64_t".to_string(),
            Self::Bool => "bool".to_string(),
        }
    }
}

impl LiteralKind {
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float32 => 4usize,
            Self::Float64 => 8usize,
            Self::Int32 => 4usize,
            Self::Int64 => 8usize,
            Self::UInt32 => 4usize,
            Self::UInt64 => 8usize,
            Self::Bool => 1usize,
        }
    }
}

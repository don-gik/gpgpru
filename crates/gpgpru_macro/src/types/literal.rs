// src/types/literal.rs

use syn::spanned::Spanned;

use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl std::fmt::Display for LiteralKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_rust_string())
    }
}

impl LiteralKind {
    pub fn alignment(&self) -> usize {
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

    pub fn to_slang_string(&self) -> Cow<'static, str> {
        match self {
            Self::Float32 => Cow::Borrowed("float"),
            Self::Float64 => Cow::Borrowed("double"),
            Self::Int32 => Cow::Borrowed("int"),
            Self::Int64 => Cow::Borrowed("int64_t"),
            Self::UInt32 => Cow::Borrowed("uint"),
            Self::UInt64 => Cow::Borrowed("uint64_t"),
            Self::Bool => Cow::Borrowed("bool"),
        }
    }

    pub fn to_rust_string(&self) -> Cow<'static, str> {
        match self {
            LiteralKind::Float32 => Cow::Borrowed("f32"),
            LiteralKind::Float64 => Cow::Borrowed("f64"),
            LiteralKind::Int32 => Cow::Borrowed("i32"),
            LiteralKind::Int64 => Cow::Borrowed("i64"),
            LiteralKind::UInt32 => Cow::Borrowed("u32"),
            LiteralKind::UInt64 => Cow::Borrowed("u64"),
            LiteralKind::Bool => Cow::Borrowed("bool"),
        }
    }
}

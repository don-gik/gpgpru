// src/slang_fields/slang_args.rs

use quote::ToTokens;
use syn::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlangPrimitive {
    Int,
    UInt,
    Int64T,
    UInt64T,
    Short,
    UShort,
    Half,
    Float,
    Double,
    Bool,
    Float3,
    Float4,
    Float4x4,
}

impl SlangPrimitive {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::UInt => "uint",
            Self::Int64T => "int64_t",
            Self::UInt64T => "uint64_t",
            Self::Short => "short",
            Self::UShort => "ushort",
            Self::Half => "half",
            Self::Float => "float",
            Self::Double => "double",
            Self::Bool => "bool",
            Self::Float3 => "float3",
            Self::Float4 => "float4",
            Self::Float4x4 => "float4x4",
        }
    }

    pub fn bit_width(&self) -> usize {
        match self {
            Self::Bool => 8,
            Self::Short | Self::UShort | Self::Half => 16,
            Self::Int | Self::UInt | Self::Float => 32,
            Self::Int64T | Self::UInt64T | Self::Double => 64,
            Self::Float3 | Self::Float4 => 128,
            Self::Float4x4 => 128,
        }
    }
}

impl TryFrom<&str> for SlangPrimitive {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "i32" | "int" => Ok(Self::Int),
            "u32" | "uint" => Ok(Self::UInt),
            "i64" | "int64_t" => Ok(Self::Int64T),
            "u64" | "uint64_t" => Ok(Self::UInt64T),
            "short" | "i16" => Ok(Self::Short),
            "ushort" | "u16" => Ok(Self::UShort),
            "half" | "f16" | "f16::f16" | "half::f16::f16" => Ok(Self::Half),
            "float" | "f32" => Ok(Self::Float),
            "double" | "f64" => Ok(Self::Double),
            "bool" => Ok(Self::Bool),
            "float3" | "glam::Vec3" | "Vec3" => Ok(Self::Float3),
            "float4" | "glam::Vec4" | "Vec4" => Ok(Self::Float4),
            "float4x4" | "glam::Mat4" | "Mat4" => Ok(Self::Float4x4),
            _ => Err(format!(
                "Unsupported primitive type `{}` for GPU kernel",
                value
            )),
        }
    }
}

impl TryFrom<&Type> for SlangPrimitive {
    type Error = syn::Error;

    fn try_from(ty: &Type) -> Result<Self, Self::Error> {
        let ty_str = ty.to_token_stream().to_string().replace(' ', "");

        Self::try_from(ty_str.as_str()).map_err(|err_msg| syn::Error::new_spanned(ty, err_msg))
    }
}

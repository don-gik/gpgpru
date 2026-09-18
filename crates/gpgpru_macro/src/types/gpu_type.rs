// src/types/gpu_type.rs

use crate::LiteralKind;

use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeIdx(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GpuType {
    Scalar(LiteralKind),
    Vector {
        element: LiteralKind,
        len: u8,
    },
    Matrix {
        element: LiteralKind,
        rows: u8,
        cols: u8,
    },

    Array {
        element: TypeIdx,
        len: usize,
    },
    Slice {
        element: TypeIdx,
        is_mut: bool,
    },
    Ptr {
        target: TypeIdx,
        is_mut: bool,
    }, // BDA

    #[allow(dead_code)]
    Struct {
        name: Cow<'static, str>,
        fields: Vec<(Cow<'static, str>, TypeIdx)>,
    },
}

#[derive(Debug)]
pub struct TypeArena {
    types: Vec<GpuType>,
    map: HashMap<GpuType, TypeIdx>,

    pub f32_ty: TypeIdx,
    pub i32_ty: TypeIdx,
    pub u32_ty: TypeIdx,
    pub bool_ty: TypeIdx,
}

impl Default for TypeArena {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeArena {
    pub fn new() -> Self {
        let mut arena = Self {
            types: Vec::new(),
            map: HashMap::new(),
            f32_ty: TypeIdx(0),
            i32_ty: TypeIdx(0),
            u32_ty: TypeIdx(0),
            bool_ty: TypeIdx(0),
        };

        arena.f32_ty = arena.alloc(GpuType::Scalar(LiteralKind::Float32));
        arena.i32_ty = arena.alloc(GpuType::Scalar(LiteralKind::Int32));
        arena.u32_ty = arena.alloc(GpuType::Scalar(LiteralKind::UInt32));
        arena.bool_ty = arena.alloc(GpuType::Scalar(LiteralKind::Bool));

        arena
    }

    pub fn alloc(&mut self, ty: GpuType) -> TypeIdx {
        if let Some(&idx) = self.map.get(&ty) {
            return idx;
        }
        let idx = TypeIdx(self.types.len() as u32);
        self.types.push(ty.clone());
        self.map.insert(ty, idx);
        idx
    }

    #[inline]
    pub fn get(&self, idx: TypeIdx) -> &GpuType {
        &self.types[idx.0 as usize]
    }

    pub fn to_rust_string(&self, idx: TypeIdx) -> Cow<'static, str> {
        match self.get(idx) {
            GpuType::Scalar(lit) => lit.to_rust_string(),
            GpuType::Vector { element, len } => match (element, len) {
                (LiteralKind::Float32, 2) => Cow::Borrowed("glam::Vec2"),
                (LiteralKind::Float32, 3) => Cow::Borrowed("glam::Vec3"),
                (LiteralKind::Float32, 4) => Cow::Borrowed("glam::Vec4"),
                (LiteralKind::Int32, 2) => Cow::Borrowed("glam::IVec2"),
                (LiteralKind::Int32, 3) => Cow::Borrowed("glam::IVec3"),
                (LiteralKind::Int32, 4) => Cow::Borrowed("glam::IVec4"),
                (LiteralKind::UInt32, 2) => Cow::Borrowed("glam::UVec2"),
                (LiteralKind::UInt32, 3) => Cow::Borrowed("glam::UVec3"),
                (LiteralKind::UInt32, 4) => Cow::Borrowed("glam::UVec4"),
                (LiteralKind::Bool, 2) => Cow::Borrowed("glam::BVec2"),
                (LiteralKind::Bool, 3) => Cow::Borrowed("glam::BVec3"),
                (LiteralKind::Bool, 4) => Cow::Borrowed("glam::BVec4"),
                _ => Cow::Owned(format!("[{}; {}]", element.to_rust_string(), len)),
            },
            GpuType::Matrix { rows, cols, .. } => match (rows, cols) {
                (2, 2) => Cow::Borrowed("glam::Mat2"),
                (3, 3) => Cow::Borrowed("glam::Mat3"),
                (4, 4) => Cow::Borrowed("glam::Mat4"),
                _ => Cow::Owned(format!("[[f32; {}]; {}]", cols, rows)),
            },
            GpuType::Array { element, len } => {
                Cow::Owned(format!("[{}; {}]", self.to_rust_string(*element), len))
            }
            GpuType::Slice { .. } => {
                panic!("Slice types (`&[T]`) cannot be lowered directly to Rust type strings.");
            }
            GpuType::Ptr { target, is_mut } => {
                let mut_str = if *is_mut { "*mut " } else { "*const " };
                Cow::Owned(format!("{}{}", mut_str, self.to_rust_string(*target)))
            }
            GpuType::Struct { name, .. } => name.clone(),
        }
    }

    pub fn to_slang_string(&self, idx: TypeIdx) -> Cow<'static, str> {
        match self.get(idx) {
            GpuType::Scalar(lit) => lit.to_slang_string(),
            GpuType::Vector { element, len } => match (element, len) {
                (LiteralKind::Float32, 2) => Cow::Borrowed("float2"),
                (LiteralKind::Float32, 3) => Cow::Borrowed("float3"),
                (LiteralKind::Float32, 4) => Cow::Borrowed("float4"),
                (LiteralKind::Int32, 2) => Cow::Borrowed("int2"),
                (LiteralKind::Int32, 3) => Cow::Borrowed("int3"),
                (LiteralKind::Int32, 4) => Cow::Borrowed("int4"),
                (LiteralKind::UInt32, 2) => Cow::Borrowed("uint2"),
                (LiteralKind::UInt32, 3) => Cow::Borrowed("uint3"),
                (LiteralKind::UInt32, 4) => Cow::Borrowed("uint4"),
                (LiteralKind::Bool, 2) => Cow::Borrowed("bool2"),
                (LiteralKind::Bool, 3) => Cow::Borrowed("bool3"),
                (LiteralKind::Bool, 4) => Cow::Borrowed("bool4"),
                _ => Cow::Owned(format!("{}{}", element.to_slang_string(), len)),
            },
            GpuType::Matrix { rows, cols, .. } => match (rows, cols) {
                (3, 3) => Cow::Borrowed("float3x3"),
                (4, 4) => Cow::Borrowed("float4x4"),
                _ => Cow::Owned(format!("float{}x{}", rows, cols)),
            },
            GpuType::Array { element, len } => {
                Cow::Owned(format!("{}[{}]", self.to_slang_string(*element), len))
            }
            GpuType::Slice { .. } => {
                panic!(
                    "Slice types (`&[T]`) cannot be lowered directly to Slang type strings. Use BDA pointers or buffer bindings instead."
                );
            }
            GpuType::Ptr { target, .. } => {
                Cow::Owned(format!("{}*", self.to_slang_string(*target)))
            }
            GpuType::Struct { name, .. } => name.clone(),
        }
    }

    pub fn alignment(&self, idx: TypeIdx) -> usize {
        match self.get(idx) {
            GpuType::Scalar(lit) => lit.alignment(),
            GpuType::Vector { len, .. } => match len {
                2 => 8,
                3 | 4 => 16,
                _ => panic!("Invalid vector size"),
            },
            GpuType::Matrix { rows, cols, .. } => match (rows, cols) {
                (3, 3) | (4, 4) => 16,
                _ => panic!("Invalid matrix size"),
            },
            GpuType::Array { element, .. } => self.alignment(*element),
            GpuType::Slice { .. } => {
                panic!(
                    "Slice types (`&[T]`) cannot be lowered directly to Slang type strings. Use BDA pointers or buffer bindings instead."
                );
            }
            GpuType::Ptr { .. } => 8,
            GpuType::Struct { fields, .. } => fields
                .iter()
                .map(|(_, field_idx)| self.alignment(*field_idx))
                .max()
                .unwrap_or(1),
        }
    }
}

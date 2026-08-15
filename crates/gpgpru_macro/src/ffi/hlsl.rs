// src/ffi/hlsl.rs

use hassle_rs::compile_hlsl;
// use proc_macro::TokenStream;
use proc_macro2::Span;
// use quote::quote;
use syn::{Expr, ExprLit, ItemFn, Lit, Stmt};

pub fn extract_raw_string(func: &ItemFn) -> Result<String, syn::Error> {
    if let Some(Stmt::Expr(
        Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }),
        _,
    )) = func.block.stmts.first()
    {
        Ok(s.value())
    } else {
        Err(syn::Error::new_spanned(
            &func.block,
            "Expected a raw string literal returning Pure HLSL code: r#\"...\"#",
        ))
    }
}

pub fn compile_hlsl_to_spirv(
    source_code: &str,
    entry_point: &str,
    span: Span,
) -> Result<Vec<u32>, syn::Error> {
    // 1. DXC SPIR-V flag
    let args = vec!["-spirv"];
    let defines = vec![];

    // 2. Use hassle_rs
    // - "cs_6_0" or "cs_6_5"
    match compile_hlsl(
        "shader.hlsl",
        source_code,
        entry_point,
        "cs_6_0",
        &args,
        &defines,
    ) {
        Ok(spirv_bytes) => {
            // Vulkan SPIR-V -> 4 bytes align
            if spirv_bytes.len() % 4 != 0 {
                return Err(syn::Error::new(
                    span,
                    "Generated SPIR-V byte size is not a multiple of 4.",
                ));
            }

            // Vec<u8> -> Vec<u32> casting
            let spirv_u32 = spirv_bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<u32>>();

            Ok(spirv_u32)
        }
        Err(err) => {
            // DXC compile error message
            Err(syn::Error::new(
                span,
                format!("HLSL Compilation Error:\n{}", err),
            ))
        }
    }
}

// src/slang_fields/arguments.rs

use super::SlangPrimitive;

use syn::{Type, TypeReference, TypeSlice};

#[derive(Debug)]
pub struct SlangField {
    pub name: String,
    pub slang_type: String,
    pub rust_type: Type,
}

pub fn unpack_push_field(name: &str, ty: &Type) -> syn::Result<Vec<SlangField>> {
    let ty_str = quote::quote!(#ty).to_string().replace(' ', "");
    if ty_str.starts_with("Vec<") || ty_str.starts_with("Box<") {
        return Err(syn::Error::new_spanned(
            ty,
            format!(
                "Heap-allocated `{ty_str}` cannot be passed directly to GPU kernel. Use a slice reference (`&[{}]`) instead.",
                ty_str
                    .trim_start_matches("Vec<")
                    .trim_start_matches("Box<")
                    .trim_end_matches('>')
            ),
        ));
    }

    match ty {
        Type::Reference(TypeReference { elem, .. }) => {
            if let Type::Slice(TypeSlice {
                elem: slice_elem, ..
            }) = &**elem
            {
                let primitive = SlangPrimitive::try_from(&**slice_elem)?;

                let u64_type: syn::Type = syn::parse_quote! { u64 };
                let u32_type: syn::Type = syn::parse_quote! { u32 };

                Ok(vec![
                    SlangField {
                        name: name.to_string(),
                        slang_type: format!("{}*", primitive.as_str()),
                        rust_type: u64_type,
                    },
                    SlangField {
                        name: format!("{}_____len", name),
                        slang_type: SlangPrimitive::UInt.as_str().to_string(),
                        rust_type: u32_type,
                    },
                ])
            } else {
                Err(syn::Error::new_spanned(
                    ty,
                    "Invalid reference: Only slice references (&[T] or &mut [T]) are allowed for GPU buffers.",
                ))
            }
        }

        _ => {
            let primitive = SlangPrimitive::try_from(ty)?;
            Ok(vec![SlangField {
                name: name.to_string(),
                slang_type: primitive.as_str().to_string(),
                rust_type: ty.clone(),
            }])
        }
    }
}

pub(crate) fn get_field_bit_width(field: &crate::slang_fields::SlangField) -> usize {
    let t = field.slang_type.as_str();

    if t.ends_with('*') {
        return 64;
    }

    if let Ok(primitive) = SlangPrimitive::try_from(t) {
        primitive.bit_width()
    } else {
        32 // default 32 bit
    }
}

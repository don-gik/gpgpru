// src/macros/compute/argument.rs

use crate::types::{ParsedSignature, SystemBinding, TypeArena};
use std::borrow::Cow;
use std::collections::HashMap;

fn get_system_binding(ty: &syn::Type) -> Option<SystemBinding> {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return match segment.ident.to_string().as_str() {
                "GlobalIdX" => Some(SystemBinding::GlobalIdX),
                "GlobalIdY" => Some(SystemBinding::GlobalIdY),
                "GlobalIdZ" => Some(SystemBinding::GlobalIdZ),
                "GlobalId" => Some(SystemBinding::GlobalId),
                _ => None,
            };
        }
    }
    None
}

pub fn parse_kernel_signature(
    arena: &mut TypeArena,
    item_fn: &syn::ItemFn,
) -> syn::Result<ParsedSignature> {
    let mut system_inputs = HashMap::new();
    let mut pushed_fields = Vec::new();

    for arg in &item_fn.sig.inputs {
        let syn::FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "self is not allowed in compute kernels.",
            ));
        };

        let syn::Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "Unsupported pattern in argument.",
            ));
        };
        let arg_name = pat_ident.ident.to_string();

        if let Some(bind_type) = get_system_binding(&*pat_type.ty) {
            system_inputs.insert(Cow::Owned(arg_name), bind_type);
        } else {
            let type_idx = arena.lower_type(&*pat_type.ty)?;
            pushed_fields.push((Cow::Owned(arg_name), type_idx));
        }
    }

    Ok(ParsedSignature {
        system_inputs,
        pushed_fields,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_parse_kernel_signature_success() {
        let mut arena = TypeArena::new();

        let input_fn: syn::ItemFn = parse_quote! {
            fn scale(data: &mut [f32], factor: f32, global_id_x: GlobalIdX) {}
        };

        let sig = parse_kernel_signature(&mut arena, &input_fn).expect("Parsing Failed");

        dbg!(&sig.system_inputs);
        dbg!(&sig.pushed_fields);

        // Push fields
        assert_eq!(sig.pushed_fields.len(), 2);
        assert_eq!(sig.pushed_fields[0].0, "data");
        assert_eq!(sig.pushed_fields[1].0, "factor");

        assert_eq!(sig.pushed_fields[1].1, arena.f32_ty);
    }

    #[test]
    fn test_reject_self_argument() {
        let mut arena = TypeArena::new();

        let input_fn: syn::ItemFn = parse_quote! {
            fn invalid(&self, factor: f32) {}
        };

        let result = parse_kernel_signature(&mut arena, &input_fn);
        assert!(result.is_err());
    }
}

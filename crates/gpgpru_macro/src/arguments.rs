// src/arguments.rs

use std::collections::HashMap;

use syn::{FnArg, ItemFn, Pat, Type};

pub struct ParsedSignature {
    pub system_inputs: HashMap<String, String>,
    pub pushed_fields: Vec<(String, Type)>,
}

fn get_system_binding(ty: &syn::Type) -> Option<&'static str> {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return match segment.ident.to_string().as_str() {
                "GlobalIdX" => Some("thread_id.x"),
                "GlobalIdY" => Some("thread_id.y"),
                "GlobalIdZ" => Some("thread_id.z"),
                "GlobalId" => Some("thread_id"),
                _ => None,
            };
        }
    }
    None
}

pub fn parse_kernel_signature(item_fn: &ItemFn) -> syn::Result<ParsedSignature> {
    let mut system_inputs = HashMap::new();
    let mut pushed_fields = Vec::new();

    for arg in &item_fn.sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "self is not allowed in compute kernels.",
            ));
        };

        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "Unsupported pattern in argument.",
            ));
        };
        let arg_name = pat_ident.ident.to_string();

        let ty = (*pat_type.ty).clone();
        if let Some(bind_name) = get_system_binding(&ty) {
            system_inputs.insert(arg_name, bind_name.to_string());
        } else {
            pushed_fields.push((arg_name, ty));
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
        let input_fn: ItemFn = parse_quote! {
            fn scale(data: &mut [f32], factor: f32, global_id_x: GlobalIdX) {}
        };

        let sig = parse_kernel_signature(&input_fn).expect("Parsing Failed");

        dbg!(&sig.system_inputs);
        dbg!(&sig.pushed_fields);

        // System Input
        assert_eq!(
            sig.system_inputs.get("global_id_x").map(|s| s.as_str()),
            Some("thread_id.x")
        );
        // Push fields
        assert_eq!(sig.pushed_fields.len(), 2);
        assert_eq!(sig.pushed_fields[0].0, "data");
        assert_eq!(sig.pushed_fields[1].0, "factor");
    }

    #[test]
    fn test_reject_self_argument() {
        let input_fn: ItemFn = parse_quote! {
            fn invalid(&self, factor: f32) {}
        };

        let result = parse_kernel_signature(&input_fn);
        assert!(result.is_err());
    }
}

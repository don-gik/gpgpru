// src/entry/argument.rs

use crate::{GpuType, entry::argument::SystemBinding::GlobalIdX};

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum SystemBinding {
    GlobalIdX,
    GlobalIdY,
    GlobalIdZ,
    GlobalId,
}

pub struct ParsedSignature {
    pub system_inputs: HashMap<String, SystemBinding>,
    pub pushed_fields: Vec<(String, GpuType)>,
}

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

pub fn parse_kernel_signature(item_fn: &syn::ItemFn) -> syn::Result<ParsedSignature> {
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
            system_inputs.insert(arg_name, bind_type);
        } else {
            let gputype = GpuType::try_from(&*pat_type.ty)?;
            pushed_fields.push((arg_name, gputype));
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
        let input_fn: syn::ItemFn = parse_quote! {
            fn scale(data: &mut [f32], factor: f32, global_id_x: GlobalIdX) {}
        };

        let sig = parse_kernel_signature(&input_fn).expect("Parsing Failed");

        dbg!(&sig.system_inputs);
        dbg!(&sig.pushed_fields);

        // Push fields
        assert_eq!(sig.pushed_fields.len(), 2);
        assert_eq!(sig.pushed_fields[0].0, "data");
        assert_eq!(sig.pushed_fields[1].0, "factor");
    }

    #[test]
    fn test_reject_self_argument() {
        let input_fn: syn::ItemFn = parse_quote! {
            fn invalid(&self, factor: f32) {}
        };

        let result = parse_kernel_signature(&input_fn);
        assert!(result.is_err());
    }
}

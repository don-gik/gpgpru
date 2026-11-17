use super::*;

pub struct KernelArgs {
    pub slang_path: syn::LitStr,
    pub entry_point: Option<syn::LitStr>,
    pub profiles: Vec<SlangProfile>,
    // pub profile_paths: Vec<syn::Path>,
    pub high_precision: bool,
}

fn suggest_profile_variant(input: &str) -> Option<&'static str> {
    let normalized = input.to_ascii_lowercase().replace('_', "");
    SlangProfile::VARIANT_NAMES
        .iter()
        .copied()
        .find(|v| v.to_ascii_lowercase().replace('_', "") == normalized)
}

fn parse_profile_list(input: syn::parse::ParseStream) -> syn::Result<Vec<SlangProfile>> {
    let array: syn::ExprArray = input.parse()?;
    let mut profiles = Vec::with_capacity(array.elems.len());
    let mut combined_err: Option<syn::Error> = None;

    let mut push_err = |err: syn::Error| match &mut combined_err {
        Some(e) => e.combine(err),
        None => combined_err = Some(err),
    };

    let valid_list = SlangProfile::VARIANT_NAMES
        .iter()
        .map(|v| format!("  - {v}"))
        .collect::<Vec<_>>()
        .join("\n");

    for elem in &array.elems {
        let syn::Expr::Path(expr_path) = elem else {
            push_err(syn::Error::new_spanned(
                elem,
                format!(
                    "Expected a `SlangProfile` variant, not an expression.\nValid variants: \n{valid_list}"
                ),
            ));
            continue;
        };

        let Some(last_seg) = expr_path.path.segments.last() else {
            continue;
        };
        let variant_name = last_seg.ident.to_string();

        match SlangProfile::from_variant_name(&variant_name) {
            Some(p) => profiles.push(p),
            None => {
                let hint = suggest_profile_variant(&variant_name)
                    .map(|s| format!(" Did you mean `{s}`?"))
                    .unwrap_or_default();

                push_err(syn::Error::new_spanned(
                    last_seg,
                    format!(
                        "Unknown SlangProfile `{variant_name}`.{hint}\nValid variants: \n{valid_list}"
                    ),
                ));
            }
        }
    }

    if let Some(err) = combined_err {
        return Err(err);
    }

    Ok(profiles)
}

impl syn::parse::Parse for KernelArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let slang_path: syn::LitStr = input.parse()?;
        let mut entry_point = None;
        let mut profiles = Vec::new();
        // let mut profile_paths = Vec::new();
        let mut high_precision = false;

        while !input.is_empty() {
            input.parse::<syn::Token![,]>()?;
            if input.is_empty() {
                break;
            }

            let lookahead = input.lookahead1();
            if lookahead.peek(syn::LitStr) {
                let lit: syn::LitStr = input.parse()?;
                if entry_point.is_some() {
                    return Err(syn::Error::new(
                        lit.span(),
                        "Duplicate `entry_point`. Already specified earlier.",
                    ));
                }
                entry_point = Some(lit);
            } else if lookahead.peek(syn::Ident) {
                let key: syn::Ident = input.parse()?;
                match key.to_string().as_str() {
                    "entry_point" => {
                        input.parse::<syn::Token![=]>()?;
                        let lit: syn::LitStr = input.parse()?;
                        if entry_point.is_some() {
                            return Err(syn::Error::new(
                                lit.span(),
                                "Duplicate `entry_point` specified.",
                            ));
                        }
                        entry_point = Some(lit);
                    }
                    "profiles" => {
                        input.parse::<syn::Token![=]>()?;
                        let parsed_profiles = parse_profile_list(input)?;
                        profiles = parsed_profiles;
                        // profile_paths = parsed_paths;
                    }
                    "high_precision" => {
                        if input.peek(syn::Token![=]) {
                            input.parse::<syn::Token![=]>()?;
                            let val: syn::LitBool = input.parse()?;
                            high_precision = val.value;
                        } else {
                            high_precision = true;
                        }
                    }
                    other => {
                        return Err(syn::Error::new(
                            key.span(),
                            format!(
                                "Unknown kernel option '{}'. Expected `entry_point`, `profiles`, or `high_precision`.",
                                other
                            ),
                        ));
                    }
                }
            } else {
                return Err(lookahead.error());
            }
        }

        Ok(Self {
            slang_path,
            entry_point,
            profiles,
            // profile_paths,
            high_precision,
        })
    }
}

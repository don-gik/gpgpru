use super::*;

pub struct KernelArgs {
    pub slang_path: syn::LitStr,
    pub entry_point: Option<syn::LitStr>,
    pub profiles: Vec<SlangProfile>,
    // pub profile_paths: Vec<syn::Path>,
    pub high_precision: bool,
}

fn parse_profile_list(
    input: syn::parse::ParseStream,
) -> syn::Result<(Vec<SlangProfile>, Vec<syn::Path>)> {
    let array: syn::ExprArray = input.parse()?;
    let mut profiles = Vec::with_capacity(array.elems.len());
    let mut profile_paths = Vec::with_capacity(array.elems.len());
    let mut combined_err: Option<syn::Error> = None;

    for elem in array.elems {
        let syn::Expr::Path(expr_path) = elem else {
            let err = syn::Error::new_spanned(
                elem,
                "Expected a `SlangProfile` variant (e.g. `Spirv1_6` or `SlangProfile::SpvGroupNonUniform`).",
            );
            match &mut combined_err {
                Some(e) => e.combine(err),
                None => combined_err = Some(err),
            }
            continue;
        };

        if let Some(last_seg) = expr_path.path.segments.last() {
            let variant_name = last_seg.ident.to_string();
            match SlangProfile::from_variant_name(&variant_name) {
                Some(p) => profiles.push(p),
                None => {
                    let err = syn::Error::new_spanned(
                        last_seg,
                        format!(
                            "Unknown SlangProfile '{}'. Valid variants: Spirv1_5, Spirv1_6, SpvGroupNonUniform, SpvGroupNonUniformShuffle, SpvCooperativeMatrixKHR, ...",
                            variant_name
                        ),
                    );
                    match &mut combined_err {
                        Some(e) => e.combine(err),
                        None => combined_err = Some(err),
                    }
                }
            }
        }

        profile_paths.push(expr_path.path);
    }

    if let Some(err) = combined_err {
        return Err(err);
    }

    Ok((profiles, profile_paths))
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
                        let (parsed_profiles, _parsed_paths) = parse_profile_list(input)?;
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

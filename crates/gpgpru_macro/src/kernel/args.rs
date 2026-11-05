pub struct KernelArgs {
    pub slang_path: syn::LitStr,
    pub entry_point: Option<syn::LitStr>,
    pub high_precision: bool,
}

impl syn::parse::Parse for KernelArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let slang_path: syn::LitStr = input.parse()?;

        let mut entry_point = None;
        let mut high_precision = false;

        // Loop to easily handle variable arguments in any order after the first comma
        while input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;

            // Safe guard against trailing commas at the very end
            if input.is_empty() {
                break;
            }

            // Check if the token is a string literal (e.g., "main") -> Entry point
            if input.peek(syn::LitStr) {
                entry_point = Some(input.parse()?);
            }
            // Check if the token is the standalone identifier `high` -> Precision flag
            else if input.peek(syn::Ident) {
                let ident: syn::Ident = input.parse()?;
                if ident == "high" {
                    high_precision = true;
                } else {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "Unknown argument token: `{}`. Expected `high` or entrypoint string literal.",
                            ident
                        ),
                    ));
                }
            } else {
                return Err(syn::Error::new(
                    input.span(),
                    "Expected string literal for entrypoint or `high` flag.",
                ));
            }
        }

        Ok(KernelArgs {
            slang_path,
            entry_point,
            high_precision,
        })
    }
}

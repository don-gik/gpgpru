pub struct KernelArgs {
    pub slang_path: syn::LitStr,
    pub entry_point: Option<syn::LitStr>,
}

impl syn::parse::Parse for KernelArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let slang_path: syn::LitStr = input.parse()?;

        let entry_point = if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
            if input.is_empty() {
                None
            } else if input.peek(syn::Ident) && input.peek2(syn::Token![=]) {
                let _ident: syn::Ident = input.parse()?;
                let _eq: syn::Token![=] = input.parse()?;
                Some(input.parse()?)
            } else {
                Some(input.parse()?)
            }
        } else {
            None
        };

        Ok(KernelArgs {
            slang_path,
            entry_point,
        })
    }
}

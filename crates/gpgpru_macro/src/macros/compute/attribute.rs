// src/entry/attribute.rs

use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitInt, Token, parenthesized};

pub struct ComputeAttr {
    pub threads: (u32, u32, u32),
}

impl Parse for ComputeAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        if name != "threads" {
            return Err(syn::Error::new(name.span(), "Expected `threads` attribute"));
        }

        let content;
        parenthesized!(content in input);

        let x: LitInt = content.parse()?;
        content.parse::<Token![,]>()?;
        let y: LitInt = content.parse()?;
        content.parse::<Token![,]>()?;
        let z: LitInt = content.parse()?;

        Ok(ComputeAttr {
            threads: (x.base10_parse()?, y.base10_parse()?, z.base10_parse()?),
        })
    }
}

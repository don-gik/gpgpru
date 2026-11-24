use super::*;

// use std::collections::HashMap;
// use syn::spanned::Spanned;

struct KernelError {
    err: syn::Error,
    deps: Vec<String>,
}

impl KernelError {
    fn with_deps(err: syn::Error, deps: Vec<String>) -> Self {
        Self { err, deps }
    }
}

impl From<syn::Error> for KernelError {
    fn from(err: syn::Error) -> Self {
        Self {
            err,
            deps: Vec::new(),
        }
    }
}

// ─── Shared state ──────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ArgKind<'a> {
    PushConstant { member: &'a CachedMember },
    Descriptor { res: &'a CachedResource },
}

#[derive(Debug)]
struct ArgData {
    ident: syn::Ident,
    ty: syn::Type,
    is_raw: bool,
}

#[derive(Debug)]
struct MatchArg<'a> {
    ident: syn::Ident,
    ty: syn::Type,
    is_raw: bool,
    kind: ArgKind<'a>,
}

#[derive(Debug, Default)]
struct SharedState {
    arg_data: Vec<ArgData>,
    error: Option<syn::Error>,
}

impl SharedState {
    pub fn push_error(&mut self, err: syn::Error) {
        match &mut self.error {
            Some(existing) => existing.combine(err),
            None => self.error = Some(err),
        }
    }

    pub fn error_spanned<T: quote::ToTokens>(&mut self, tokens: T, msg: &str) {
        self.push_error(syn::Error::new_spanned(tokens, msg));
    }

    pub fn check_errors(&mut self) -> syn::Result<()> {
        match self.error.take() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

#[derive(Default)]
struct CodeGen {
    raw_arg_tys: Vec<syn::Type>,
    fn_args: Vec<TokenStream2>,
    pc_copies: Vec<TokenStream2>,
    desc_inits: Vec<TokenStream2>,
    vk_types: Vec<TokenStream2>,
    extra_fields: Vec<TokenStream2>,
    extra_inits: Vec<TokenStream2>,
    hazards: Vec<TokenStream2>,
    asserts: Vec<TokenStream2>,
    err: Option<syn::Error>,
}

impl CodeGen {
    pub fn push_error(&mut self, err: syn::Error) {
        match &mut self.err {
            Some(existing) => existing.combine(err),
            None => self.err = Some(err),
        }
    }

    pub fn finish(&mut self) -> syn::Result<()> {
        match self.err.take() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

// ─── Top-level ─────────────────────────────────────────────────────────

pub fn kernel_impl(attr: TokenStream2, body: TokenStream2) -> syn::Result<TokenStream2> {
    let kernel_args: KernelArgs = syn::parse2(attr)?;
    let input_fn: syn::ItemFn = syn::parse2(body)?;
    let mut state = SharedState::default();

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let root_path = Utf8PathBuf::from(manifest_dir)
        .join(&kernel_args.slang_path.value())
        .into_string();

    let emit_err = |err: syn::Error, deps: &[String]| -> TokenStream2 {
        let tracked_paths = std::iter::once(root_path.as_str())
            .chain(deps.iter().map(|s| s.as_str()))
            .filter(|p| Utf8Path::new(p).exists());

        let err_tokens = err.to_compile_error();
        quote::quote! {
            #( const _: &[u8] = include_bytes!(#tracked_paths); )*
            #err_tokens
        }
    };

    // 1. Compiling
    let reflection = match compile_shader(&kernel_args, &input_fn) {
        Ok(refl) => refl,
        Err(KernelError { err, deps }) => return Ok(emit_err(err, &deps)),
    };
    if let Some(w) = &reflection.warnings {
        if !w.trim().is_empty() {
            eprintln!("{}\n", format_slang_warnings(w));
        }
    }

    // 2. Parsing
    parse_args(&input_fn, &mut state);

    let (matched, errors) = match_args(&reflection, &state.arg_data);
    for e in errors {
        state.push_error(e);
    }
    state.check_errors()?;

    // 3. Code gen starts here
    let mut code = CodeGen::default();
    generate_code(&matched, &reflection, &mut code);
    code.finish()?;

    Ok(emit_final(&input_fn, &reflection, &code, &root_path))
}

// ─── Parsing Args ──────────────────────────────────────────────────────

fn parse_args(input_fn: &syn::ItemFn, state: &mut SharedState) {
    fn has_raw_attr(attrs: &[syn::Attribute]) -> bool {
        attrs.iter().any(|attr| attr.path().is_ident("raw"))
    }

    let fn_is_raw = has_raw_attr(&input_fn.attrs);
    let arg_count = input_fn.sig.inputs.len();

    let mut arg_data = Vec::with_capacity(arg_count);

    for arg in &input_fn.sig.inputs {
        match arg {
            syn::FnArg::Typed(pat_type) => {
                let ident = match &*pat_type.pat {
                    syn::Pat::Ident(ident) => ident.ident.clone(),
                    _ => {
                        state.error_spanned(
                            pat_type,
                            "kernel parameter is expected to be simple variable names.",
                        );
                        continue;
                    }
                };
                let ty = &*pat_type.ty;

                arg_data.push(ArgData {
                    ident,
                    ty: ty.clone(),
                    is_raw: fn_is_raw || has_raw_attr(&pat_type.attrs),
                });
            }
            syn::FnArg::Receiver(receiver) => {
                state.error_spanned(
                    receiver,
                    "`self` parameter is not allowed in kernel functions",
                );
            }
        }
    }

    state.arg_data = arg_data;
}

fn match_args<'a>(
    reflection: &'a ShaderReflection,
    arg_data: &[ArgData],
) -> (Vec<MatchArg<'a>>, Vec<syn::Error>) {
    let mut out = Vec::with_capacity(arg_data.len());
    let mut errors = Vec::new();

    for arg in arg_data {
        match reflection.find_parameter(&arg.ident.to_string(), arg.ident.span()) {
            Ok(Some(kind)) => out.push(MatchArg {
                ident: arg.ident.clone(),
                ty: arg.ty.clone(),
                is_raw: arg.is_raw,
                kind,
            }),
            Ok(None) => {
                errors.push(syn::Error::new_spanned(
                    &arg.ident,
                    &format!("'{}' not found in shader", arg.ident),
                ));
            }
            Err(e) => errors.push(e),
        }
    }

    (out, errors)
}

// ─── Compiling Shader ──────────────────────────────────────────────────────

fn compile_shader(
    kernel_args: &KernelArgs,
    input_fn: &syn::ItemFn,
) -> Result<ShaderReflection, KernelError> {
    fn get_absolute_path(s: &syn::LitStr) -> Utf8PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        Utf8PathBuf::from(manifest_dir).join(&s.value())
    }

    let kernel_name = &input_fn.sig.ident.to_string();
    let slang_path = &get_absolute_path(&kernel_args.slang_path);
    let entry_point = &kernel_args
        .entry_point
        .as_ref()
        .map_or_else(|| "main".to_string(), |s| s.value());
    let profiles = &kernel_args.profiles;
    let high_precision = kernel_args.high_precision;

    ShaderCache::get_or_compile(
        &kernel_name,
        slang_path,
        &entry_point,
        profiles,
        high_precision,
    )
    .map_err(|e| {
        KernelError::with_deps(
            syn::Error::new(input_fn.sig.ident.span(), e.error.to_string()),
            e.fallback_deps,
        )
    })
}

// ─── Generating TokenStream2 ──────────────────────────────────────────────────────

fn generate_code(matched: &[MatchArg<'_>], reflection: &ShaderReflection, code: &mut CodeGen) {
    let pc = reflection
        .resources
        .iter()
        .find(|r| r.kind == raw::ResourceKind::PushConstant)
        .map(|r| r.members.as_slice())
        .unwrap_or(&[]);

    for m in matched {
        code.raw_arg_tys.push(m.ty.clone());

        match m.kind {
            ArgKind::PushConstant { member } => match member.kind {
                raw::MemberKind::Pointer => emit_pc_bda(code, m, member, pc),
                raw::MemberKind::Primitive => emit_pc_primitive(code, m, member),
                raw::MemberKind::Struct => emit_pc_struct(code, m, member, pc),
            },
            ArgKind::Descriptor { res } => emit_descriptor(code, m, res),
        }
    }
}

fn emit_pc_bda(code: &mut CodeGen, m: &MatchArg<'_>, member: &CachedMember, pc: &[CachedMember]) {
    let ident = &m.ident;
    let ty = &m.ty;
    let offset = member.offset as usize;
    let end = offset + 8;
    // let member_size = member.size as usize;

    // #[raw] u64 → raw pointer, no tree traversal, no size
    if m.is_raw && matches!(&*ty, syn::Type::Path(p) if p.path.is_ident("u64")) {
        code.push_error(syn::Error::new_spanned(
            ty,
            format!(
                "'{}' is marked #[raw] as a bare `u64`, but #[raw] only relaxes the \
                     layout check on the element type of &GpuSlice<T>. Pass the buffer itself.",
                ident
            ),
        ));
        return;
    }

    // &/&mut impl AsGpuBuffer<T> → BDA + not is_raw - tree traversal
    let syn::Type::Reference(tref) = &*ty else {
        code.push_error(syn::Error::new_spanned(
            ty,
            format!("BDA argument '{}' must be a reference (& or &mut).", ident),
        ));
        return;
    };
    let is_mut = tref.mutability.is_some();

    let transformed = apply_lifetime_to_mut_refs((*ty).clone(), "'a");
    let bda_field = quote::format_ident!("bda_{}", ident);
    let sz_field = quote::format_ident!("sz_{}", ident);

    code.fn_args.push(quote::quote! { #ident: #transformed });
    code.extra_fields
        .push(quote::quote! { #bda_field: u64, #sz_field: u64 });
    code.extra_inits
        .push(quote::quote! { #bda_field: #ident.bda(), #sz_field: #ident.size() });

    if !m.is_raw {
        let inner = extract_bda_element_type(ty).map_err(|e| {
            code.push_error(syn::Error::new_spanned(
                ty,
                format!("BDA '{}': cannot infer element type.", ident),
            ));
            e
        });
        if let Ok(inner) = inner {
            if member.scalar_type != raw::ScalarKind::None {
                if let Err(e) = verify_primitive_or_math_type(&inner, member, ident.span()) {
                    code.push_error(e);
                }
            }
            let idx = root_idx_of(pc, &member.name);
            emit_struct_layout(pc, idx, &inner, member.size as usize, ident.span(), code);
        }
    }

    code.pc_copies.push(quote::quote! {
        let bda = #ident.bda();
        ___pc[#offset..#end].copy_from_slice(&bda.to_ne_bytes());
    });

    let access = if is_mut {
        quote::quote! { gpgpru::__private::AccessType::Write }
    } else {
        quote::quote! { gpgpru::__private::AccessType::Read }
    };
    code.hazards.push(quote::quote! {
        visitor(self.#bda_field, self.#sz_field, #access);
    });
}

fn emit_pc_primitive(code: &mut CodeGen, m: &MatchArg<'_>, member: &CachedMember) {
    let ident = &m.ident;
    let ty = &m.ty;
    let offset = member.offset as usize;
    let member_size = member.size as usize;

    code.fn_args.push(quote::quote! { #ident: #ty });

    if m.is_raw {
        code.pc_copies.push(quote::quote! {
            let raw = unsafe {
                ::core::slice::from_raw_parts(
                    (&#ident as *const #ty).cast::<u8>(),
                    ::core::mem::size_of_val(&#ident),
                )
            };
            let len = raw.len().min(#member_size as usize);
            ___pc[#offset..#offset + len].copy_from_slice(&raw[..len]);
        });
        return;
    }

    if let Err(e) = verify_primitive_or_math_type(ty, member, ident.span()) {
        code.push_error(e);
    }

    code.asserts.push(quote::quote_spanned! { ident.span() =>
        assert!(
            ::core::mem::size_of::<#ty>() == #member_size,
            concat!("'", stringify!(#ty), "' size mismatch (expected ", #member_size, "B).")
        );
    });

    code.pc_copies.push(quote::quote! {
        let bytes = gpgpru::__private::bytemuck::bytes_of(&#ident);
        ___pc[#offset..#offset + bytes.len()].copy_from_slice(bytes);
    });
}

fn emit_pc_struct(
    code: &mut CodeGen,
    m: &MatchArg<'_>,
    member: &CachedMember,
    pc: &[CachedMember],
) {
    let ident = &m.ident;
    let ty = &m.ty;
    let offset = member.offset as usize;
    let member_size = member.size as usize;

    code.fn_args.push(quote::quote! { #ident: #ty });

    if m.is_raw {
        code.pc_copies.push(quote::quote! {
            let raw = unsafe {
                ::core::slice::from_raw_parts(
                    (&#ident as *const _).cast::<u8>(),
                    ::core::mem::size_of_val(&#ident),
                )
            };
            let len = raw.len().min(#member_size as usize);
            ___pc[#offset..#offset + len].copy_from_slice(&raw[..len]);
        });
        return;
    }

    code.asserts.push(quote::quote_spanned! { ident.span() =>
        assert!(
            ::core::mem::size_of::<#ty>() == #member_size,
            concat!("'", stringify!(#ty), "' size mismatch (expected ", #member_size, "B).")
        );
    });

    let idx = root_idx_of(pc, &member.name);
    emit_struct_layout(pc, idx, ty, member.size as usize, ident.span(), code);

    code.pc_copies.push(quote::quote! {
        let bytes = gpgpru::__private::bytemuck::bytes_of(&#ident);
        ___pc[#offset..#offset + bytes.len()].copy_from_slice(bytes);
    });
}

fn emit_descriptor(code: &mut CodeGen, m: &MatchArg<'_>, res: &CachedResource) {
    let ident = &m.ident;
    let set = res.set;
    let binding = res.binding;

    if set != 0 {
        code.push_error(syn::Error::new_spanned(
                ident,
                format!(
                    "descriptor resource '{}' uses descriptor set {}, but gpgpru push descriptors only support set 0. \
                     Please change the shader binding to `[[vk::binding({}, 0)]]`.",
                    ident, set, binding
                ),
            ));
        return;
    }

    let vk_type = match res.kind {
        raw::ResourceKind::Uniform => {
            quote::quote! { UNIFORM_BUFFER }
        }
        raw::ResourceKind::StorageBuffer | raw::ResourceKind::StorageBufferReadWrite => {
            quote::quote! { STORAGE_BUFFER }
        }
        _ => {
            code.push_error(syn::Error::new_spanned(
                ident,
                format!(
                    "descriptor resource '{}' ({:?}) is not implemented yet. \
                         Only buffer descriptors (UBO / SSBO) are supported.",
                    ident, res.kind
                ),
            ));
            return;
        }
    };
    code.vk_types.push(quote::quote! {
        gpgpru::__private::vk::DescriptorType::#vk_type
    });

    let transformed = apply_lifetime_to_mut_refs(m.ty.clone(), "'a");
    code.fn_args.push(quote::quote! { #ident: #transformed });

    let bda_field = quote::format_ident!("bda_{}", ident);
    let sz_field = quote::format_ident!("sz_{}", ident);
    code.extra_fields
        .push(quote::quote! { #bda_field: u64, #sz_field: u64 });
    code.extra_inits
        .push(quote::quote! { #bda_field: #ident.bda(), #sz_field: #ident.size() });

    code.desc_inits.push(quote::quote! {
        gpgpru::__private::PushDescriptor {
            // set: #set,
            binding: #binding,
            ty: gpgpru::__private::vk::DescriptorType::#vk_type,
            payload: gpgpru::__private::DescriptorPayload::Buffer(
                gpgpru::__private::vk::DescriptorBufferInfo {
                    buffer: #ident.vk_buffer(),
                    offset: #ident.offset(),
                    range: #ident.size(),
                }
            ),
        }
    });

    // Record hazards
    let access = if res.kind == raw::ResourceKind::Uniform {
        quote::quote! { gpgpru::__private::AccessType::Read }
    } else {
        quote::quote! { gpgpru::__private::AccessType::Write }
    };
    code.hazards.push(quote::quote! {
        visitor(self.#bda_field, self.#sz_field, #access);
    });

    // Validate inner layout
    if !m.is_raw {
        if let Some(inner) = extract_bda_element_type(&m.ty).ok() {
            let root_idx = res
                .members
                .iter()
                .position(|x| x.parent_idx == -1)
                .unwrap_or(0);
            emit_struct_layout(
                &res.members,
                root_idx,
                &inner,
                res.total_size as usize,
                ident.span(),
                code,
            );
        }
    }
}

// Utils for generating code.

fn child_members<'a>(
    ms: &'a [CachedMember],
    parent: i32,
) -> impl Iterator<Item = (usize, &'a CachedMember)> {
    ms.iter()
        .enumerate()
        .filter(move |(_, m)| m.parent_idx == parent)
}

fn emit_struct_layout(
    ms: &[CachedMember],
    root_idx: usize,
    ty: &syn::Type,
    total: usize,
    outer_span: proc_macro2::Span,
    code: &mut CodeGen,
) {
    code.asserts.push(quote::quote_spanned! {
        outer_span =>
        assert!(
            ::core::mem::size_of::<#ty>() == #total,
            concat!("'", stringify!(#ty), "' total size mismatch (slang ", #total, "B).")
        );
    });
    fn walk(
        code: &mut CodeGen,
        ms: &[CachedMember],
        idx: usize,
        ty: &syn::Type,
        outer_span: proc_macro2::Span,
    ) {
        for (i, c) in child_members(ms, idx as i32) {
            let f = syn::Ident::new(&c.name, proc_macro2::Span::call_site());
            let off = c.offset as usize;
            code.asserts.push(quote::quote_spanned! {
                outer_span =>
                assert!(
                    ::core::mem::offset_of!(#ty, #f) == #off,
                    concat!("'", stringify!(#ty), ".", stringify!(#f),
                            "' offset mismatch (slang ", #off, "B).")
                );
            });
            if child_members(ms, i as i32).next().is_some() {
                walk(code, ms, i, ty, outer_span);
            }
        }
    }
    walk(code, ms, root_idx, ty, outer_span);
}

fn root_idx_of(ms: &[CachedMember], name: &str) -> usize {
    ms.iter()
        .position(|m| m.name == name && m.parent_idx == -1)
        .unwrap_or(0)
}

// ─── Final Emitting into AST ──────────────────────────────────────────────────────

fn emit_final(
    input_fn: &syn::ItemFn,
    reflection: &ShaderReflection,
    code: &CodeGen,
    root_path: &str,
) -> TokenStream2 {
    let fn_vis = &input_fn.vis;
    let fn_name = &input_fn.sig.ident;

    let total_pc = reflection
        .resources
        .iter()
        .find(|r| r.kind == raw::ResourceKind::PushConstant)
        .map(|r| r.total_size as usize)
        .unwrap_or(0);

    let n_desc = code.desc_inits.len();
    let spirv = &reflection.spirv_bytes;

    let dep_includes: Vec<TokenStream2> = reflection
        .dependencies
        .iter()
        .filter(|p| p.as_str() != root_path)
        .map(|p| quote::quote! { const _: &str = include_str!(#p); })
        .collect();

    let asserts = &code.asserts;
    let raw_arg_tys = &code.raw_arg_tys;
    let fn_args = &code.fn_args;
    let pc_copies = &code.pc_copies;
    let extra_fields = &code.extra_fields;
    let hazards = &code.hazards;
    let vk_types = &code.vk_types;
    let extra_inits = &code.extra_inits;
    let desc_inits = &code.desc_inits;

    quote::quote! {
        // cargo recompile trigger
        const _: &[u8] = include_bytes!(#root_path);
        #(#dep_includes)*

        // compile-time validation
        const _: () = {
            #(#asserts)*
            #[allow(dead_code, unused_variables)]
            fn _use_types(#(_: #raw_arg_tys),*) {}
        };

        #[allow(non_snake_case)]
        #fn_vis fn #fn_name<'a>(
            #(#fn_args),*
        ) -> gpgpru::__private::CommandBuilder<
            'a,
            impl gpgpru::__private::ComputeArgs
                + 'static,
        > {
            static SPIRV_CODE: &[u32] = &[#(#spirv),*];

            // push constant bytes
            let mut ___pc = [0u8; #total_pc];
            #(#pc_copies)*

            #[repr(C)]
            #[derive(Copy, Clone)]
            struct Params {
                raw: [u8; #total_pc],
                #(#extra_fields,)*
                descs: [gpgpru::__private::PushDescriptor; #n_desc],
            }

            unsafe impl gpgpru::__private::ComputeArgs for Params {
                type PushConstants = [u8; #total_pc];
                type Descriptors = [gpgpru::__private::PushDescriptor; #n_desc];

                const DESCRIPTOR_TYPES: &'static [gpgpru::__private::vk::DescriptorType] =
                    &[#(#vk_types),*];

                fn push_constants(&self) -> Self::PushConstants { self.raw }
                fn descriptors(&self) -> Self::Descriptors { self.descs }
                fn trace_hazards(
                    &self,
                    visitor: &mut dyn FnMut(
                        u64,
                        u64,
                        gpgpru::__private::AccessType,
                    ),
                ) {
                    #(#hazards)*
                }
            }

            gpgpru::__private::CommandBuilder::new(
                SPIRV_CODE,
                Params {
                    raw: ___pc,
                    #(#extra_inits,)*
                    descs: [#(#desc_inits),*],
                },
            )
        }
    }
}

// ─── Testing ──────────────────────────────────────────────────────

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn test_parsing_raw() {
        let attr = quote::quote! {"examples/clean.slang", "scale", high};
        let body = quote::quote! {
            #[raw]
            fn x(i: i32) -> i32 {
                i
            }
        };

        let mut state = SharedState::default();
        let input_fn: syn::ItemFn = syn::parse2(body.clone()).unwrap();

        let arg_data = parse_args(&input_fn, &mut state);

        dbg!(arg_data);
        dbg!(syn::parse2::<KernelArgs>(attr.clone()).unwrap());
    }
}

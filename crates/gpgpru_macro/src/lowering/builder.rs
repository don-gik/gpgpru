// src/lowering/builder.rs

use crate::{
    ExprId, GpuType, IrExpr, IrStmt, LiteralKind, ParsedSignature, StmtId, SystemBinding,
    TypeArena, TypeIdx, VarId, VarInfo, VarStorage,
};

use std::borrow::Cow;
use std::collections::HashMap;

use proc_macro2::Span;

#[derive(Debug)]
pub struct LoweringBuilder {
    pub exprs: Vec<IrExpr>,
    pub stmts: Vec<IrStmt>,
    pub vars: Vec<VarInfo>,

    // Temporary Stack
    scopes: Vec<HashMap<Cow<'static, str>, VarId>>,
}

impl LoweringBuilder {
    pub fn new(arena: &mut TypeArena, sig: &ParsedSignature) -> Self {
        let mut builder = Self {
            exprs: Vec::new(),
            stmts: Vec::new(),
            vars: Vec::new(),
            scopes: vec![HashMap::new()],
        };

        for (arg_name, bind_target) in &sig.system_inputs {
            let var_id = builder.alloc_var(
                arg_name.clone(),
                VarStorage::Builtin {
                    ty: bind_target.clone(),
                },
                match bind_target {
                    SystemBinding::GlobalId => arena.alloc(GpuType::Vector {
                        element: LiteralKind::UInt32,
                        len: 3,
                    }),
                    SystemBinding::GlobalIdX
                    | SystemBinding::GlobalIdY
                    | SystemBinding::GlobalIdZ => arena.alloc(GpuType::Scalar(LiteralKind::UInt32)),
                },
            );
            builder.scopes[0].insert(arg_name.clone(), var_id);
        }

        for (arg_name, type_idx) in &sig.pushed_fields {
            let arg_name_str = arg_name.to_string();
            let arg_name_cow: Cow<'static, str> = Cow::Owned(arg_name_str);

            match arena.get(*type_idx) {
                GpuType::Slice { element, is_mut } => {
                    let bda_id = builder.alloc_var(
                        arg_name_cow.clone(),
                        VarStorage::Bda { elem_ty: *element },
                        *type_idx,
                    );
                    let ptr_name: Cow<'static, str> =
                        Cow::Owned(format!("_gpu_{}_ptr", arg_name_cow.clone()));
                    let ptr_id = builder.alloc_var(
                        ptr_name.clone(),
                        VarStorage::PushConstant,
                        arena.alloc(GpuType::Ptr {
                            target: *element,
                            is_mut: *is_mut,
                        }),
                    );
                    let len_name: Cow<'static, str> =
                        Cow::Owned(format!("_gpu_{}_len", arg_name_cow.clone()));
                    let len_id = builder.alloc_var(
                        len_name.clone(),
                        VarStorage::PushConstant,
                        arena.alloc(GpuType::Scalar(LiteralKind::UInt32)),
                    );

                    builder.scopes[0].insert(arg_name_cow, bda_id);
                    builder.scopes[0].insert(ptr_name, ptr_id);
                    builder.scopes[0].insert(len_name, len_id);
                }
                _ => {
                    let var_id = builder.alloc_var(
                        arg_name_cow.clone(),
                        VarStorage::PushConstant,
                        *type_idx,
                    );

                    builder.scopes[0].insert(arg_name_cow, var_id);
                }
            };
        }

        builder
    }

    pub fn with_scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.scopes.push(HashMap::new());
        let result = f(self);
        self.scopes.pop();
        result
    }

    pub fn insert_scope(
        &mut self,
        name: Cow<'static, str>,
        var_id: VarId,
        span: Span,
    ) -> syn::Result<()> {
        if let Some(current_scope) = self.scopes.last_mut() {
            current_scope.insert(name, var_id);
            return Ok(());
        }
        Err(syn::Error::new(
            span,
            "No active scope found to insert variable",
        ))
    }

    pub fn push_expr(&mut self, expr: IrExpr) -> ExprId {
        let id = ExprId(self.exprs.len());
        self.exprs.push(expr);
        id
    }

    pub fn push_stmt(&mut self, stmt: IrStmt) -> StmtId {
        let id = StmtId(self.stmts.len());
        self.stmts.push(stmt);
        id
    }

    pub fn alloc_var(&mut self, name: Cow<'static, str>, kind: VarStorage, ty: TypeIdx) -> VarId {
        let id = VarId(self.vars.len());
        self.vars.push(VarInfo { name, kind, ty });
        id
    }

    pub fn lookup_var(&self, name: &str, span: Span) -> syn::Result<VarId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&id) = scope.get(name) {
                return Ok(id);
            }
        }

        Err(syn::Error::new(
            span,
            format!("Cannot find value `{}` in this scope", name),
        ))
    }
}

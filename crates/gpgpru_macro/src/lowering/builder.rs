// src/lowering/builder.rs

use crate::{
    ExprId, GpuType, IrExpr, IrStmt, ParsedSignature, StmtId, VarId, VarInfo, VarStorage,
    types::ToSlangString,
};

use std::collections::HashMap;

use proc_macro2::Span;

#[derive(Debug)]
pub struct LoweringBuilder {
    pub exprs: Vec<IrExpr>,
    pub stmts: Vec<IrStmt>,
    pub vars: Vec<VarInfo>,

    // Temporary Stack
    scopes: Vec<HashMap<String, VarId>>,
}

impl LoweringBuilder {
    pub fn new(sig: &ParsedSignature) -> Self {
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
            );
            builder.scopes[0].insert(arg_name.clone(), var_id);
        }

        for (arg_name, gpu_type) in &sig.pushed_fields {
            let storage = match gpu_type {
                GpuType::Slice { element, .. } => VarStorage::Bda {
                    elem_ty: element.to_slang_string(),
                },
                _ => VarStorage::PushConstant,
            };

            let var_id = builder.alloc_var(arg_name.clone(), storage);
            builder.scopes[0].insert(arg_name.clone(), var_id);
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

    pub fn insert_scope(&mut self, name: String, var_id: VarId, span: Span) -> syn::Result<()> {
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

    pub fn alloc_var(&mut self, name: String, kind: VarStorage) -> VarId {
        let id = VarId(self.vars.len());
        self.vars.push(VarInfo { name, kind });
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

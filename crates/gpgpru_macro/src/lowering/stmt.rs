// src/lowering/stmt.rs

use super::LoweringBuilder;
use crate::{GpuType, IrBlock, IrExpr, IrStmt, LiteralKind, StmtId, TypeArena, VarStorage};

use std::borrow::Cow;
use syn::spanned::Spanned;

fn is_assign_op(op: &syn::BinOp) -> bool {
    use syn::BinOp::*;
    matches!(
        op,
        AddAssign(_)
            | SubAssign(_)
            | MulAssign(_)
            | DivAssign(_)
            | RemAssign(_)
            | ShlAssign(_)
            | ShrAssign(_)
    )
}

fn to_pure_bin_op(op: &syn::BinOp) -> Option<syn::BinOp> {
    use syn::BinOp::*;

    let pure_op = match op {
        AddAssign(_) => Add(Default::default()),
        SubAssign(_) => Sub(Default::default()),
        MulAssign(_) => Mul(Default::default()),
        DivAssign(_) => Div(Default::default()),
        RemAssign(_) => Rem(Default::default()),
        ShlAssign(_) => Shl(Default::default()),
        ShrAssign(_) => Shr(Default::default()),
        _ => return None,
    };

    Some(pure_op)
}

impl LoweringBuilder {
    pub fn lower_const(
        &mut self,
        arena: &mut TypeArena,
        local_const: &syn::ItemConst,
    ) -> syn::Result<StmtId> {
        let name = local_const.ident.to_string();
        let ty = arena.lower_type(&*local_const.ty)?;
        let init = self.lower_expr(arena, &local_const.expr)?;

        let name_cow: Cow<'static, str> = Cow::Owned(name);

        let var_id = self.alloc_var(name_cow.clone(), VarStorage::Const, ty);
        self.insert_scope(name_cow, var_id, local_const.span())?;

        Ok(self.push_stmt(IrStmt::Const {
            var: var_id,
            ty,
            init,
        }))
    }

    pub fn lower_block(
        &mut self,
        arena: &mut TypeArena,
        syn_block: &syn::Block,
    ) -> syn::Result<StmtId> {
        let stmts = self.with_scope(|builder| {
            syn_block
                .stmts
                .iter()
                .map(|s| builder.lower_stmt(arena, s))
                .collect::<syn::Result<Vec<StmtId>>>()
        })?;
        Ok(self.push_stmt(IrStmt::Block(IrBlock { stmts })))
    }

    pub fn lower_stmt(
        &mut self,
        arena: &mut TypeArena,
        syn_stmt: &syn::Stmt,
    ) -> syn::Result<StmtId> {
        match syn_stmt {
            syn::Stmt::Local(local) => {
                let pat_ident = match &local.pat {
                    syn::Pat::Ident(pat) => pat,
                    syn::Pat::Type(pat_type) => match &*pat_type.pat {
                        syn::Pat::Ident(pat) => pat,
                        _ => {
                            return Err(syn::Error::new(
                                local.pat.span(),
                                "Unsupported pattern inside type annotation",
                            ));
                        }
                    },
                    _ => {
                        return Err(syn::Error::new(
                            local.pat.span(),
                            "Complex pattern not supported",
                        ));
                    }
                };

                let name = pat_ident.ident.to_string();
                let is_mut = pat_ident.mutability.is_some();

                let init = if let Some(init) = &local.init {
                    self.lower_expr(arena, &init.expr)?
                } else {
                    return Err(syn::Error::new(
                        syn_stmt.span(),
                        "Using 'let' statement without init is not allowed.",
                    ));
                };

                let name_cow: Cow<'static, str> = Cow::Owned(name);
                let init_ty = self.exprs[init.0].ty();

                let var_id = self.alloc_var(name_cow.clone(), VarStorage::Local, init_ty);
                self.insert_scope(name_cow, var_id, syn_stmt.span())?;

                Ok(self.push_stmt(IrStmt::Let {
                    var: var_id,
                    init,
                    is_mut,
                }))
            }

            syn::Stmt::Item(item) => match item {
                syn::Item::Const(local_const) => self.lower_const(arena, local_const),

                _ => Err(syn::Error::new_spanned(
                    item,
                    "Any items other than 'const' is not supported in kernel.",
                )),
            },

            syn::Stmt::Expr(expr, semi) => {
                let is_control_flow = matches!(
                    expr,
                    syn::Expr::If(_) | syn::Expr::ForLoop(_) | syn::Expr::While(_)
                );

                if semi.is_none() && !is_control_flow {
                    return Err(syn::Error::new_spanned(
                        expr,
                        "Semicolon `;` is required at the end of the expression.",
                    ));
                }

                match expr {
                    syn::Expr::Assign(assign) => {
                        let target = self.lower_expr(arena, &assign.left)?;
                        let value = self.lower_expr(arena, &assign.right)?;
                        Ok(self.push_stmt(IrStmt::Assign {
                            target,
                            value,
                            op: None,
                        }))
                    }

                    syn::Expr::Binary(binary) if is_assign_op(&binary.op) => {
                        let target = self.lower_expr(arena, &binary.left)?;
                        let value = self.lower_expr(arena, &binary.right)?;

                        let pure_op = to_pure_bin_op(&binary.op);

                        Ok(self.push_stmt(IrStmt::Assign {
                            target,
                            value,
                            op: pure_op,
                        }))
                    }

                    syn::Expr::Return(ret) => {
                        let value = ret
                            .expr
                            .as_ref()
                            .map(|e| self.lower_expr(arena, e))
                            .transpose()?;
                        Ok(self.push_stmt(IrStmt::Return(value)))
                    }

                    syn::Expr::If(syn_if) => {
                        let cond = self.lower_expr(arena, &syn_if.cond)?;
                        let then_block = self.lower_block(arena, &syn_if.then_branch)?;

                        let else_branch = if let Some((_, else_expr)) = &syn_if.else_branch {
                            let else_stmt = syn::Stmt::Expr(*else_expr.clone(), None);
                            let stmt_id = self.lower_stmt(arena, &else_stmt)?;
                            Some(stmt_id)
                        } else {
                            None
                        };

                        Ok(self.push_stmt(IrStmt::If {
                            cond,
                            then_block,
                            else_branch,
                        }))
                    }

                    syn::Expr::ForLoop(for_loop) => {
                        let loop_var_ident = match &*for_loop.pat {
                            syn::Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    &for_loop.pat,
                                    "Only simple identifiers are supported in for loops (e.g., `for i in ...`)",
                                ));
                            }
                        };

                        let loop_ty = arena.alloc(GpuType::Scalar(LiteralKind::Int32));
                        let loop_var_id =
                            self.alloc_var(Cow::Owned(loop_var_ident), VarStorage::Local, loop_ty);
                        let loop_var_expr = self.push_expr(IrExpr::Var {
                            id: loop_var_id,
                            ty: loop_ty,
                        });

                        let mut is_reverse = false;
                        let mut custom_step: Option<&syn::Expr> = None;
                        let mut current_expr = &*for_loop.expr;

                        // Range parsing
                        let range = loop {
                            match current_expr {
                                syn::Expr::Range(r) => break r,

                                syn::Expr::MethodCall(method) => {
                                    let method_name = method.method.to_string();

                                    if method_name == "step_by" {
                                        custom_step = Some(&method.args[0]);
                                    } else if method_name == "rev" {
                                        is_reverse = !is_reverse;
                                    } else {
                                        return Err(syn::Error::new_spanned(
                                            &method.method,
                                            "Only .step_by() and .rev() are supported in range method chains",
                                        ));
                                    }

                                    current_expr = &*method.receiver;
                                }

                                _ => {
                                    return Err(syn::Error::new_spanned(
                                        current_expr,
                                        "Expected a range or range method chain (e.g., (0..10).rev().step_by(2))",
                                    ));
                                }
                            }
                        };

                        // Start parse
                        let start = match &range.start {
                            Some(s) => self.lower_expr(arena, s)?,
                            None => self.push_expr(IrExpr::Literal {
                                value: "0".to_string(),
                                ty: loop_ty,
                            }),
                        };

                        // End parse
                        let end = match &range.end {
                            Some(e) => self.lower_expr(arena, e)?,
                            None => {
                                return Err(syn::Error::new_spanned(
                                    &for_loop.expr,
                                    "Unbounded ranges (e.g., 0..) are not supported",
                                ));
                            }
                        };

                        // Step parse
                        let step_val = match custom_step {
                            Some(expr) => self.lower_expr(arena, expr)?,
                            None => self.push_expr(IrExpr::Literal {
                                value: "1".to_string(),
                                ty: loop_ty,
                            }),
                        };

                        let (init_stmt, cond_expr, step_stmt);

                        if is_reverse {
                            let init_val = match range.limits {
                                syn::RangeLimits::HalfOpen(_) => {
                                    let one = self.push_expr(IrExpr::Literal {
                                        value: "1".to_string(),
                                        ty: loop_ty,
                                    });
                                    self.push_expr(IrExpr::Binary {
                                        op: syn::BinOp::Sub(Default::default()),
                                        left: end,
                                        right: one,
                                        ty: loop_ty,
                                    })
                                }
                                syn::RangeLimits::Closed(_) => end,
                            };
                            init_stmt = self.push_stmt(IrStmt::Let {
                                var: loop_var_id,
                                init: init_val,
                                is_mut: false,
                            });

                            cond_expr = self.push_expr(IrExpr::Binary {
                                op: syn::BinOp::Ge(Default::default()),
                                left: loop_var_expr,
                                right: start,
                                ty: arena.bool_ty,
                            });

                            step_stmt = self.push_stmt(IrStmt::Assign {
                                target: loop_var_expr,
                                value: step_val,
                                op: Some(syn::BinOp::Sub(Default::default())),
                            });
                        } else {
                            init_stmt = self.push_stmt(IrStmt::Let {
                                var: loop_var_id,
                                init: start,
                                is_mut: true,
                            });

                            let cond_op = match range.limits {
                                syn::RangeLimits::HalfOpen(_) => syn::BinOp::Lt(Default::default()),
                                syn::RangeLimits::Closed(_) => syn::BinOp::Le(Default::default()),
                            };
                            cond_expr = self.push_expr(IrExpr::Binary {
                                op: cond_op,
                                left: loop_var_expr,
                                right: end,
                                ty: arena.bool_ty,
                            });

                            step_stmt = self.push_stmt(IrStmt::Assign {
                                target: loop_var_expr,
                                value: step_val,
                                op: Some(syn::BinOp::Add(Default::default())),
                            });
                        }

                        let body = self.lower_block(arena, &for_loop.body)?;

                        Ok(self.push_stmt(IrStmt::For {
                            init: Some(init_stmt),
                            cond: Some(cond_expr),
                            step: Some(step_stmt),
                            body,
                        }))
                    }

                    syn::Expr::While(while_expr) => {
                        let cond_id = self.lower_expr(arena, &while_expr.cond)?;
                        let body = self.lower_block(arena, &while_expr.body)?;

                        Ok(self.push_stmt(IrStmt::While {
                            cond: cond_id,
                            body,
                        }))
                    }

                    syn::Expr::Break(_) => Ok(self.push_stmt(IrStmt::Break)),

                    syn::Expr::Continue(_) => Ok(self.push_stmt(IrStmt::Continue)),

                    _ => {
                        let exprid = self.lower_expr(arena, expr)?;
                        Ok(self.push_stmt(IrStmt::Expr(exprid)))
                    }
                }
            }

            _ => Err(syn::Error::new(
                syn_stmt.span(),
                "Unsupported statement type",
            )),
        }
    }
}

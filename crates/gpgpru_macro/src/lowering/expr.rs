// src/lowering/expr.rs

use super::{LoweringBuilder, resolve_call, resolve_field, resolve_method};
use crate::{ExprId, GpuType, IrExpr, LiteralKind, MethodAction, TypeArena, VarStorage};

use syn::spanned::Spanned;

impl LoweringBuilder {
    pub fn lower_expr(
        &mut self,
        arena: &mut TypeArena,
        syn_expr: &syn::Expr,
    ) -> syn::Result<ExprId> {
        match syn_expr {
            syn::Expr::Lit(syn::ExprLit { lit, .. }) => {
                let (value, ty) = match lit {
                    syn::Lit::Float(f) => {
                        let value = f.base10_digits().to_string();
                        let kind = match f.suffix() {
                            "f32" | "" => LiteralKind::Float32,
                            "f64" => LiteralKind::Float64,
                            s => {
                                return Err(syn::Error::new(
                                    lit.span(),
                                    format!("Unsupported float suffix: {}", s),
                                ));
                            }
                        };
                        (value, arena.alloc(GpuType::Scalar(kind)))
                    }

                    syn::Lit::Int(i) => {
                        let value = i.base10_digits().to_string();
                        let kind = match i.suffix() {
                            "u32" => LiteralKind::UInt32,
                            "u64" => LiteralKind::UInt64,
                            "i32" | "" => LiteralKind::Int32,
                            "i64" => LiteralKind::Int64,
                            s => {
                                return Err(syn::Error::new(
                                    lit.span(),
                                    format!("Unsupported int suffix: {}", s),
                                ));
                            }
                        };
                        (value, arena.alloc(GpuType::Scalar(kind)))
                    }

                    syn::Lit::Bool(b) => (
                        b.value.to_string(),
                        arena.alloc(GpuType::Scalar(LiteralKind::Bool)),
                    ),

                    _ => return Err(syn::Error::new(lit.span(), "Unsupported Literal Type.")),
                };

                Ok(self.push_expr(IrExpr::Literal { value, ty }))
            }

            syn::Expr::Path(path) => {
                let ident = path
                    .path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new(path.span(), "Complex path not supported"))?;
                let name = ident.to_string();

                let var_id = self.lookup_var(&name, ident.span())?;
                let var_ty = self.vars[var_id.0].ty;

                Ok(self.push_expr(IrExpr::Var {
                    id: var_id,
                    ty: var_ty,
                }))
            }

            syn::Expr::Binary(bin) => {
                let lhs = self.lower_expr(arena, &bin.left)?;
                let rhs = self.lower_expr(arena, &bin.right)?;

                let ty = self.exprs[lhs.0].ty();

                Ok(self.push_expr(IrExpr::Binary {
                    left: lhs,
                    right: rhs,
                    op: bin.op,
                    ty,
                }))
            }

            syn::Expr::Unary(un) => {
                let expr_id = self.lower_expr(arena, &un.expr)?;
                let inner_ty = self.exprs[expr_id.0].ty();

                let ty = match un.op {
                    syn::UnOp::Neg(_) | syn::UnOp::Not(_) => inner_ty,

                    syn::UnOp::Deref(token) => match arena.get(inner_ty) {
                        GpuType::Slice { element, .. } => *element,
                        _ => {
                            return Err(syn::Error::new(
                                token.spans[0],
                                "Cannot dereference a non-buffer/slice type",
                            ));
                        }
                    },

                    _ => return Err(syn::Error::new(un.span(), "Unsupported unary operator")),
                };

                Ok(self.push_expr(IrExpr::Unary {
                    op: un.op,
                    value: expr_id,
                    ty,
                }))
            }

            syn::Expr::Index(idx) => {
                let mut base = self.lower_expr(arena, &idx.expr)?;
                let index = self.lower_expr(arena, &idx.index)?;

                if let IrExpr::Var { id: var_id, .. } = self.exprs[base.0] {
                    let var_info = &self.vars[var_id.0];
                    if matches!(var_info.kind, VarStorage::Bda { .. }) {
                        let ptr_name = format!("_gpu_{}_ptr", var_info.name);
                        let ptr_var_id = self.lookup_var(&ptr_name, idx.span())?;
                        let ptr_ty = self.vars[ptr_var_id.0].ty;

                        base = self.push_expr(IrExpr::Var {
                            id: ptr_var_id,
                            ty: ptr_ty,
                        });
                    }
                }

                let base_ty = self.exprs[base.0].ty();

                let ty = match arena.get(base_ty) {
                    GpuType::Ptr { target, .. } => *target,

                    GpuType::Slice { element, .. } => *element,

                    GpuType::Array { element, .. } => *element,

                    GpuType::Vector { element, .. } => arena.alloc(GpuType::Scalar(*element)),

                    GpuType::Matrix {
                        rows,
                        cols: _,
                        element,
                    } => arena.alloc(GpuType::Vector {
                        element: *element,
                        len: *rows,
                    }),

                    _ => {
                        return Err(syn::Error::new(idx.span(), "Cannot index into this type"));
                    }
                };

                Ok(self.push_expr(IrExpr::Index { base, index, ty }))
            }

            syn::Expr::Field(field) => {
                let base = self.lower_expr(arena, &field.base)?;

                let field_name = match &field.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(index) => index.index.to_string(),
                };

                let base_ty_id = self.exprs[base.0].ty();
                let base_ty = arena.get(base_ty_id);

                match resolve_field(base_ty, field_name.as_str()) {
                    Some(ty) => Ok(self.push_expr(IrExpr::Field {
                        base: base,
                        name: field_name,
                        ty: arena.alloc(ty),
                    })),
                    None => Err(syn::Error::new(field.span(), "Field definition not found.")),
                }
            }

            syn::Expr::Call(call) => {
                let segments: Vec<String> = match &*call.func {
                    syn::Expr::Path(p) => p
                        .path
                        .segments
                        .iter()
                        .map(|seg| seg.ident.to_string())
                        .collect(),
                    _ => {
                        return Err(syn::Error::new(
                            call.func.span(),
                            "Unsupported function call target",
                        ));
                    }
                };

                let mut args = Vec::new();
                let mut arg_types = Vec::new();

                for arg in &call.args {
                    let lowered_arg = self.lower_expr(arena, arg)?;
                    args.push(lowered_arg);
                    arg_types.push(arena.get(self.exprs[lowered_arg.0].ty()).clone());
                }

                match resolve_call(&segments, &arg_types) {
                    Some((return_ty, emit_name)) => Ok(self.push_expr(IrExpr::Call {
                        name: emit_name.to_string(),
                        args,
                        ty: arena.alloc(return_ty),
                    })),
                    None => Err(syn::Error::new(
                        call.func.span(),
                        format!("Function '{:?}' not found or invalid arguments.", segments),
                    )),
                }
            }

            syn::Expr::MethodCall(call) => {
                let receiver = self.lower_expr(arena, &call.receiver)?;
                let receiver_ty = self.exprs[receiver.0].ty();

                let method_name = call.method.to_string();

                let mut args = Vec::new();
                let mut arg_types = vec![arena.get(receiver_ty).clone()];
                for arg in &call.args {
                    let lowered_arg = self.lower_expr(arena, arg)?;
                    args.push(lowered_arg);
                    arg_types.push(arena.get(self.exprs[lowered_arg.0].ty()).clone());
                }

                match resolve_method(&method_name, &arg_types) {
                    Some(MethodAction::AsFunction(return_ty, emit_name)) => {
                        let mut func_args = vec![receiver];
                        func_args.extend(args); // receiver

                        Ok(self.push_expr(IrExpr::Call {
                            name: emit_name.to_string(), // "dot", "normalize"
                            args: func_args,
                            ty: arena.alloc(return_ty),
                        }))
                    }

                    Some(MethodAction::AsMethod(return_ty, emit_name)) => {
                        Ok(self.push_expr(IrExpr::MethodCall {
                            base: receiver,
                            name: emit_name.to_string(), // "Sample", "Load"
                            args,
                            ty: arena.alloc(return_ty),
                        }))
                    }

                    None => {
                        if method_name == "len" && args.is_empty() {
                            if let IrExpr::Var { id: var_id, .. } = self.exprs[receiver.0] {
                                let var_info = &self.vars[var_id.0];

                                if matches!(var_info.kind, VarStorage::Bda { .. }) {
                                    let len_var_name = format!("_gpu_{}_len", var_info.name);
                                    let len_var_id = self.lookup_var(&len_var_name, call.span())?;

                                    return Ok(self.push_expr(IrExpr::Var {
                                        id: len_var_id,
                                        ty: arena.u32_ty,
                                    }));
                                }
                            }
                        }

                        Err(syn::Error::new(
                            call.span(),
                            format!("Method '{:?}' not found or invalid arguments.", method_name),
                        ))
                    }
                }
            }

            syn::Expr::Cast(cast) => {
                let target_type = LiteralKind::try_from(&*cast.ty)?;
                let expr_id = self.lower_expr(arena, &cast.expr)?;

                let ty = arena.alloc(GpuType::Scalar(target_type));

                Ok(self.push_expr(IrExpr::Cast { expr: expr_id, ty }))
            }

            // [0.0; 3] -> "{ 0.0, 0.0, 0.0 }"
            syn::Expr::Repeat(e) => {
                let count = match &*e.len {
                    syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(len_lit),
                        ..
                    }) => len_lit.base10_parse::<usize>().map_err(|_| {
                        syn::Error::new_spanned(len_lit, "Array length must be a valid integer")
                    })?,
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &e.len,
                            "Array length must be a literal integer",
                        ));
                    }
                };

                let value_id = self.lower_expr(arena, &e.expr)?;
                let element_ty = self.exprs[value_id.0].ty();

                let array_ty = arena.alloc(GpuType::Array {
                    element: element_ty,
                    len: count,
                });

                Ok(self.push_expr(IrExpr::Repeat {
                    value: value_id,
                    count,
                    ty: array_ty,
                }))
            }

            syn::Expr::Paren(paren) => self.lower_expr(arena, &paren.expr),

            _ => Err(syn::Error::new(
                syn_expr.span(),
                "Unsupported expression format",
            )),
        }
    }
}

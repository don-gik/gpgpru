// src/lowering/expr.rs

use super::LoweringBuilder;
use crate::{ExprId, IrExpr, LiteralKind};

use syn::spanned::Spanned;

fn parse_lit_for_slang(lit: &syn::Lit) -> syn::Result<(String, &'static str)> {
    match lit {
        syn::Lit::Float(f) => {
            let digits = f.base10_digits();
            let slang_type = match f.suffix() {
                "f64" => "double",
                "f32" | "" => "float",
                s => {
                    return Err(syn::Error::new_spanned(
                        f,
                        format!("Unsupported float suffix: {}", s),
                    ));
                }
            };
            Ok((digits.to_string(), slang_type))
        }
        syn::Lit::Int(i) => {
            let digits = i.base10_digits();
            let slang_type = match i.suffix() {
                "u32" | "usize" => "uint",
                "u64" => "uint64_t",
                "i64" => "int64_t",
                "i32" | "" => "int",
                s => {
                    return Err(syn::Error::new_spanned(
                        i,
                        format!("Unsupported int suffix: {}", s),
                    ));
                }
            };
            Ok((digits.to_string(), slang_type))
        }
        syn::Lit::Bool(b) => Ok((b.value.to_string(), "bool")),
        _ => Err(syn::Error::new_spanned(
            lit,
            "Unsupported literal for array initialization",
        )),
    }
}

impl LoweringBuilder {
    pub fn lower_expr(&mut self, syn_expr: &syn::Expr) -> syn::Result<ExprId> {
        match syn_expr {
            syn::Expr::Lit(syn::ExprLit { lit, .. }) => {
                let value = match lit {
                    syn::Lit::Str(s) => format!("\"{}\"", s.value()),
                    syn::Lit::Int(i) => i.base10_digits().to_string(),
                    syn::Lit::Float(f) => f.base10_digits().to_string(),
                    syn::Lit::Bool(b) => b.value.to_string(),
                    _ => return Err(syn::Error::new(lit.span(), "Unsupported Literal Type.")),
                };
                Ok(self.push_expr(IrExpr::Literal { value }))
            }

            syn::Expr::Path(path) => {
                let ident = path
                    .path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new(path.span(), "Complex path not supported"))?;
                let name = ident.to_string();
                let var_id = self.lookup_var(&name, ident.span())?;
                Ok(self.push_expr(IrExpr::Var(var_id)))
            }

            syn::Expr::Binary(bin) => {
                let lhs = self.lower_expr(&bin.left)?;
                let rhs = self.lower_expr(&bin.right)?;
                Ok(self.push_expr(IrExpr::Binary {
                    op: bin.op,
                    left: lhs,
                    right: rhs,
                }))
            }

            syn::Expr::Unary(un) => {
                let expr = self.lower_expr(&un.expr)?;
                Ok(self.push_expr(IrExpr::Unary {
                    op: un.op,
                    value: expr,
                }))
            }

            syn::Expr::Index(idx) => {
                let base = self.lower_expr(&idx.expr)?;
                let index = self.lower_expr(&idx.index)?;
                Ok(self.push_expr(IrExpr::Index { base, index }))
            }

            syn::Expr::Field(field) => {
                let base = self.lower_expr(&field.base)?;
                let field_name = match &field.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(index) => index.index.to_string(),
                };
                Ok(self.push_expr(IrExpr::Field {
                    base: base,
                    name: field_name,
                }))
            }

            syn::Expr::Call(call) => {
                let name = match &*call.func {
                    syn::Expr::Path(p) => p
                        .path
                        .get_ident()
                        .ok_or_else(|| {
                            syn::Error::new(call.func.span(), "Complex function path not supported")
                        })?
                        .to_string(),
                    _ => {
                        return Err(syn::Error::new(
                            call.func.span(),
                            "Unsupported function call target",
                        ));
                    }
                };
                let args = call
                    .args
                    .iter()
                    .map(|arg| self.lower_expr(arg))
                    .collect::<syn::Result<Vec<_>>>()?;
                Ok(self.push_expr(IrExpr::Call { name, args }))
            }

            syn::Expr::MethodCall(call) => {
                let base = self.lower_expr(&call.receiver)?;
                let name = call.method.to_string();

                let args = call
                    .args
                    .iter()
                    .map(|arg| self.lower_expr(arg))
                    .collect::<syn::Result<Vec<_>>>()?;

                Ok(self.push_expr(IrExpr::MethodCall { base, name, args }))
            }

            syn::Expr::Cast(cast) => {
                let target_type = LiteralKind::try_from(&*cast.ty)?;
                let expr_id = self.lower_expr(&cast.expr)?;

                Ok(self.push_expr(IrExpr::Cast {
                    expr: expr_id,
                    target_type,
                }))
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

                let (lit_val, slang_type) = match &*e.expr {
                    syn::Expr::Lit(syn::ExprLit { lit, .. }) => parse_lit_for_slang(lit)?,
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &e.expr,
                            "Array repeat value must be a literal",
                        ));
                    }
                };

                let elements = vec![lit_val.as_str(); count].join(", ");

                Ok(self.push_expr(IrExpr::Literal {
                    value: format!("{}[{}]({})", slang_type, count, elements),
                }))
            }

            syn::Expr::Paren(paren) => self.lower_expr(&paren.expr),

            _ => Err(syn::Error::new(
                syn_expr.span(),
                "Unsupported expression format",
            )),
        }
    }
}

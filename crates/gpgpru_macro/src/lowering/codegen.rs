// src/lowering/codegen.rs

use super::LoweringBuilder;

use crate::*;

fn unop_to_string(op: &syn::UnOp) -> &'static str {
    use syn::UnOp::*;
    match op {
        Deref(_) => "*",
        Not(_) => "!",
        Neg(_) => "-",
        _ => unreachable!(),
    }
}

fn binop_to_string(op: &syn::BinOp) -> &'static str {
    use syn::BinOp::*;
    match op {
        Add(_) => "+",
        Sub(_) => "-",
        Mul(_) => "*",
        Div(_) => "/",
        Rem(_) => "%",
        And(_) => "&&",
        Or(_) => "||",
        BitXor(_) => "^",
        BitAnd(_) => "&",
        BitOr(_) => "|",
        Shl(_) => "<<",
        Shr(_) => ">>",
        Eq(_) => "==",
        Lt(_) => "<",
        Le(_) => "<=",
        Ne(_) => "!=",
        Ge(_) => ">=",
        Gt(_) => ">",
        _ => unreachable!(),
    }
}

impl LoweringBuilder {
    pub fn emit_stmt(&self, stmt_id: &StmtId, arena: &TypeArena, w: &mut CodeWriter) {
        match &self.stmts[stmt_id.0] {
            IrStmt::Block(block) => {
                w.block(|w| {
                    for child_id in &block.stmts {
                        self.emit_stmt(child_id, arena, w);
                    }
                });
            }

            IrStmt::If {
                cond,
                then_block,
                else_branch,
            } => {
                w.write("if (");
                self.emit_expr(cond, arena, w);
                w.write(") ");

                w.block(|w| {
                    self.emit_stmt_body(then_block, arena, w);
                });

                if let Some(else_stmt) = else_branch {
                    w.write(" else ");
                    w.block(|w| {
                        self.emit_stmt_body(else_stmt, arena, w);
                    });
                }

                w.writeln("");
            }

            IrStmt::For {
                init,
                cond,
                step,
                body,
            } => {
                w.write("for (");
                if let Some(some_init) = init {
                    self.emit_stmt_inline(some_init, arena, w);
                }
                w.write("; ");
                if let Some(some_cond) = cond {
                    self.emit_expr(some_cond, arena, w);
                }
                w.write("; ");
                if let Some(some_step) = step {
                    self.emit_stmt_inline(some_step, arena, w);
                }
                w.write(") ");

                w.block(|w| {
                    self.emit_stmt_body(body, arena, w);
                });

                w.writeln("");
            }

            IrStmt::While { cond, body } => {
                w.write("while (");
                self.emit_expr(cond, arena, w);
                w.write(")");

                w.block(|w| {
                    self.emit_stmt_body(body, arena, w);
                });

                w.writeln("");
            }

            IrStmt::Let { var, init, is_mut } => {
                let kw = if *is_mut { "var" } else { "let" };

                w.write(&format!("{} ", kw));
                self.emit_var(var, arena, w);
                w.write(" = ");
                self.emit_expr(init, arena, w);
                w.writeln(";");
            }

            IrStmt::Assign { target, value, op } => {
                self.emit_expr(target, arena, w);

                if let Some(bin_op) = op {
                    let bin_op_str = quote::quote!(#bin_op).to_string();
                    w.write(&format!(" {}= ", bin_op_str));
                } else {
                    w.write(" = ");
                }

                self.emit_expr(value, arena, w);

                w.writeln(";");
            }

            IrStmt::Expr(expr_id) => {
                self.emit_expr(expr_id, arena, w);
            }

            IrStmt::Const { var, ty, init } => {
                w.write(&format!(
                    "constexpr {} {} = ",
                    arena.to_slang_string(*ty),
                    &self.vars[var.0].name
                ));
                self.emit_expr(init, arena, w);
                w.writeln(";");
            }

            IrStmt::Return(option_expr_id) => {
                w.write("return ");
                if let Some(expr_id) = option_expr_id {
                    self.emit_expr(expr_id, arena, w);
                }
                w.writeln(";");
            }

            IrStmt::Break => w.write("break;"),

            IrStmt::Continue => w.write("continue;"),
        }
    }

    pub fn emit_stmt_body(&self, stmt_id: &StmtId, arena: &TypeArena, w: &mut CodeWriter) {
        if let IrStmt::Block(block) = &self.stmts[stmt_id.0] {
            for child_id in &block.stmts {
                self.emit_stmt(child_id, arena, w);
            }
        } else {
            self.emit_stmt(stmt_id, arena, w);
        }
    }

    pub fn emit_stmt_inline(&self, stmt_id: &StmtId, arena: &TypeArena, w: &mut CodeWriter) {
        match &self.stmts[stmt_id.0] {
            IrStmt::Let { var, init, is_mut } => {
                let kw = if *is_mut { "var" } else { "let" };
                w.write(&format!("{} ", kw));
                self.emit_var(var, arena, w);
                w.write(" = ");
                self.emit_expr(init, arena, w);
            }
            IrStmt::Expr(expr_id) => {
                self.emit_expr(expr_id, arena, w);
            }
            IrStmt::Assign { target, value, op } => {
                self.emit_expr(target, arena, w);

                if let Some(bin_op) = op {
                    let bin_op_str = quote::quote!(#bin_op).to_string();
                    w.write(&format!("{}=", bin_op_str));
                }

                self.emit_expr(value, arena, w);
            }
            IrStmt::Const { var, ty, init } => {
                w.write(&format!(
                    "constexpr {} {} = ",
                    arena.to_slang_string(*ty),
                    &self.vars[var.0].name
                ));
                self.emit_expr(init, arena, w);
            }
            IrStmt::Return(option_expr_id) => {
                w.write("return ");
                if let Some(expr_id) = option_expr_id {
                    self.emit_expr(expr_id, arena, w);
                }
            }
            _ => {}
        }
    }

    pub fn emit_expr(&self, expr_id: &ExprId, arena: &TypeArena, w: &mut CodeWriter) {
        match &self.exprs[expr_id.0] {
            IrExpr::Literal { value, .. } => {
                w.write(value.as_str());
            }

            IrExpr::Var { id, .. } => {
                self.emit_var(id, arena, w);
            }

            IrExpr::Unary { op, value, .. } => {
                let op_str = unop_to_string(op);
                w.write(op_str);
                w.paren(|w| {
                    self.emit_expr(value, arena, w);
                });
            }

            IrExpr::Binary {
                left, right, op, ..
            } => w.paren(|w| {
                self.emit_expr(left, arena, w);
                w.write(&format!(" {} ", binop_to_string(op)));
                self.emit_expr(right, arena, w);
            }),

            IrExpr::Index { base, index, .. } => {
                self.emit_expr(base, arena, w);
                w.write("[");
                self.emit_expr(index, arena, w);
                w.write("]");
            }

            IrExpr::Field { base, name, .. } => {
                self.emit_expr(base, arena, w);
                w.write(".");
                w.write(name.as_str());
            }

            IrExpr::Call { name, args, .. } => {
                w.write(name.as_str());
                w.paren(|w| {
                    for arg in args {
                        self.emit_expr(arg, arena, w);
                    }
                });
            }

            IrExpr::MethodCall {
                base, name, args, ..
            } => {
                if name == "len" {
                    match &self.exprs[base.0] {
                        IrExpr::Var { id, .. } => match &self.vars[id.0].kind {
                            VarStorage::Bda { .. } => {
                                w.write(&format!("g_args._gpu_{}_len", &self.vars[id.0].name));
                                return;
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }

                self.emit_expr(base, arena, w);
                w.write(".");
                w.write(name.as_str());
                w.paren(|w| {
                    for arg in args {
                        self.emit_expr(arg, arena, w);
                    }
                });
            }

            IrExpr::Cast { expr, ty } => {
                w.write(&format!("{}(", arena.to_slang_string(*ty)));
                self.emit_expr(expr, arena, w);
                w.write(")");
            }

            IrExpr::Repeat { value, count, ty } => {
                w.write(&arena.to_slang_string(*ty));

                w.paren(|w| {
                    for i in 0..*count {
                        if i > 0 {
                            w.write(", ");
                        }
                        self.emit_expr(value, arena, w);
                    }
                });
            }
        }
    }

    pub fn emit_var(&self, var_id: &VarId, arena: &TypeArena, w: &mut CodeWriter) {
        let kind = &self.vars[var_id.0].kind;
        let name = &self.vars[var_id.0].name;

        match kind {
            VarStorage::Bda { elem_ty } => {
                w.write(&format!(
                    "(({}*)g_args._gpu_{}_ptr)",
                    arena.to_slang_string(*elem_ty),
                    name
                ));
            }

            VarStorage::PushConstant => {
                w.write(&format!("g_args.{}", name));
            }

            VarStorage::Builtin { ty } => match *ty {
                SystemBinding::GlobalIdX => w.write("_sv_thread_id.x"),
                SystemBinding::GlobalIdY => w.write("_sv_thread_id.y"),
                SystemBinding::GlobalIdZ => w.write("_sv_thread_id.z"),
                SystemBinding::GlobalId => w.write("_sv_thread_id"),
                // _ => w.write(name),
            },

            _ => {
                w.write(name);
            }
        }
    }
}

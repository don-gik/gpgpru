// src/lowering/codegen.rs

use super::LoweringBuilder;
use super::writer::CodeWriter;

use crate::IrStmt;
use crate::IrStmt::*;

use crate::IrExpr::*;

use crate::{VarId, VarStorage};

use crate::rust_emit::FieldInfo;
use crate::types::ToSlangString;
use crate::{ComputeAttr, ExprId, StmtId, SystemBinding};

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
    pub fn emit_slang_fn(
        &self,
        fn_name: &str,
        attr: &ComputeAttr,
        pushed_fields: &Vec<FieldInfo>,
        block_id: &StmtId,
    ) -> String {
        let mut code = String::new();

        code.push_str("struct PushConstants {\n");

        for info in pushed_fields {
            code.push_str(&format!(
                "    {} {};\n",
                &info.ty.to_slang_string(),
                &info.name
            ));
        }

        code.push_str("};\n\n");

        code.push_str("[vk::push_constant]\n");
        code.push_str("PushConstants g_args;\n\n");

        let (x, y, z) = attr.threads;
        code.push_str("[shader(\"compute\")]\n");
        code.push_str(&format!("[numthreads({}, {}, {})]\n", x, y, z));

        let mut writer = CodeWriter::new();
        self.emit_stmt(block_id, &mut writer);

        code.push_str(&format!(
            "void {}(uint3 _sv_thread_id : SV_DispatchThreadID) {}",
            fn_name, writer.buf,
        ));

        code
    }

    fn emit_stmt(&self, stmt_id: &StmtId, w: &mut CodeWriter) {
        match &self.stmts[stmt_id.0] {
            Block(block) => {
                w.block(|w| {
                    for child_id in &block.stmts {
                        self.emit_stmt(child_id, w);
                    }
                });
            }

            If {
                cond,
                then_block,
                else_branch,
            } => {
                w.write("if (");
                self.emit_expr(cond, w);
                w.write(") ");

                w.block(|w| {
                    self.emit_stmt_body(then_block, w);
                });

                if let Some(else_stmt) = else_branch {
                    w.write(" else ");
                    w.block(|w| {
                        self.emit_stmt_body(else_stmt, w);
                    });
                }

                w.writeln("");
            }

            For {
                init,
                cond,
                step,
                body,
            } => {
                w.write("for (");
                if let Some(some_init) = init {
                    self.emit_stmt_inline(some_init, w);
                }
                w.write("; ");
                if let Some(some_cond) = cond {
                    self.emit_expr(some_cond, w);
                }
                w.write("; ");
                if let Some(some_step) = step {
                    self.emit_stmt_inline(some_step, w);
                }
                w.write(") ");

                w.block(|w| {
                    self.emit_stmt_body(body, w);
                });

                w.writeln("");
            }

            While { cond, body } => {
                w.write("while (");
                self.emit_expr(cond, w);
                w.write(")");

                w.block(|w| {
                    self.emit_stmt_body(body, w);
                });

                w.writeln("");
            }

            Let { var, init, is_mut } => {
                let kw = if *is_mut { "var" } else { "let" };

                w.write(&format!("{} ", kw));
                self.emit_var(var, w);
                w.write(" = ");
                self.emit_expr(init, w);
                w.writeln(";");
            }

            Assign { target, value, op } => {
                self.emit_expr(target, w);

                if let Some(bin_op) = op {
                    let bin_op_str = quote::quote!(#bin_op).to_string();
                    w.write(&format!(" {}= ", bin_op_str));
                } else {
                    w.write(" = ");
                }

                self.emit_expr(value, w);

                w.writeln(";");
            }

            Expr(expr_id) => {
                self.emit_expr(expr_id, w);
            }

            Const { var, ty, init } => {
                w.write(&format!(
                    "constexpr {} {} = ",
                    ty.to_slang_string(),
                    &self.vars[var.0].name
                ));
                self.emit_expr(init, w);
                w.writeln(";");
            }

            Return(option_expr_id) => {
                w.write("return ");
                if let Some(expr_id) = option_expr_id {
                    self.emit_expr(expr_id, w);
                }
                w.writeln(";");
            }

            Break => w.write("break;"),

            Continue => w.write("continue;"),
        }
    }

    fn emit_stmt_body(&self, stmt_id: &StmtId, w: &mut CodeWriter) {
        if let IrStmt::Block(block) = &self.stmts[stmt_id.0] {
            for child_id in &block.stmts {
                self.emit_stmt(child_id, w);
            }
        } else {
            self.emit_stmt(stmt_id, w);
        }
    }

    fn emit_stmt_inline(&self, stmt_id: &StmtId, w: &mut CodeWriter) {
        match &self.stmts[stmt_id.0] {
            IrStmt::Let { var, init, is_mut } => {
                let kw = if *is_mut { "var" } else { "let" };
                w.write(&format!("{} ", kw));
                self.emit_var(var, w);
                w.write(" = ");
                self.emit_expr(init, w);
            }
            IrStmt::Expr(expr_id) => {
                self.emit_expr(expr_id, w);
            }
            IrStmt::Assign { target, value, op } => {
                self.emit_expr(target, w);

                if let Some(bin_op) = op {
                    let bin_op_str = quote::quote!(#bin_op).to_string();
                    w.write(&format!("{}=", bin_op_str));
                }

                self.emit_expr(value, w);
            }
            Const { var, ty, init } => {
                w.write(&format!(
                    "constexpr {} {} = ",
                    ty.to_slang_string(),
                    &self.vars[var.0].name
                ));
                self.emit_expr(init, w);
            }
            Return(option_expr_id) => {
                w.write("return ");
                if let Some(expr_id) = option_expr_id {
                    self.emit_expr(expr_id, w);
                }
            }
            _ => {}
        }
    }

    fn emit_expr(&self, expr_id: &ExprId, w: &mut CodeWriter) {
        match &self.exprs[expr_id.0] {
            Literal { value } => {
                w.write(value.as_str());
            }

            Var(var_id) => {
                self.emit_var(var_id, w);
            }

            Unary { op, value } => {
                let op_str = unop_to_string(op);
                w.write(op_str);
                w.paren(|w| {
                    self.emit_expr(value, w);
                });
            }

            Binary { left, right, op } => w.paren(|w| {
                self.emit_expr(left, w);
                w.write(&format!(" {} ", binop_to_string(op)));
                self.emit_expr(right, w);
            }),

            Index { base, index } => {
                self.emit_expr(base, w);
                w.write("[");
                self.emit_expr(index, w);
                w.write("]");
            }

            Field { base, name } => {
                self.emit_expr(base, w);
                w.write(".");
                w.write(name.as_str());
            }

            Call { name, args } => {
                w.write(name.as_str());
                w.paren(|w| {
                    for arg in args {
                        self.emit_expr(arg, w);
                    }
                });
            }

            MethodCall { base, name, args } => {
                if name == "len" {
                    match &self.exprs[base.0] {
                        Var(var_id) => match &self.vars[var_id.0].kind {
                            VarStorage::Bda { .. } => {
                                w.write(&format!("g_args._gpu_{}_len", &self.vars[var_id.0].name));
                                return;
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }

                self.emit_expr(base, w);
                w.write(".");
                w.write(name.as_str());
                w.paren(|w| {
                    for arg in args {
                        self.emit_expr(arg, w);
                    }
                });
            }

            Cast { expr, target_type } => {
                w.write(&format!("{}(", target_type.to_slang_string()));
                self.emit_expr(expr, w);
                w.write(")");
            }
        }
    }

    fn emit_var(&self, var_id: &VarId, w: &mut CodeWriter) {
        let kind = &self.vars[var_id.0].kind;
        let name = &self.vars[var_id.0].name;

        match kind {
            VarStorage::Bda { elem_ty } => {
                w.write(&format!("(({}*)g_args._gpu_{}_ptr)", elem_ty, name));
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
                w.write(name.as_str());
            }
        }
    }
}

// src/slang_transpile/emitter.rs

use quote::ToTokens;
use std::collections::HashMap;
use std::fmt::Write;

use syn::spanned::Spanned;
use syn::visit::Visit;

pub struct SlangEmitter<'a> {
    pub system_inputs: &'a HashMap<String, String>,
    pub push_fields: Vec<String>,
    pub output: String,
    indent: usize,
    pub err: Vec<syn::Error>,
}

impl<'a> SlangEmitter<'a> {
    pub fn new(system_inputs: &'a HashMap<String, String>, push_fields: Vec<String>) -> Self {
        Self {
            system_inputs,
            push_fields,
            output: String::new(),
            indent: 1usize,
            err: Vec::new(),
        }
    }

    pub fn emit_block(&mut self, block: &syn::Block) {
        self.output.push_str("\n");
        for stmt in &block.stmts {
            self.visit_stmt(stmt);
        }
        self.output.push_str("}\n");
    }

    fn indent_block(&mut self) {
        self.indent += 1;
    }

    fn close_indent(&mut self) {
        self.indent -= 1;
    }

    fn push_str(&mut self, str: &str) {
        self.output.push_str(str);
    }

    fn push_indent(&mut self) {
        let width = self.indent * 4;
        let _ = write!(self.output, "{:width$}", "", width = width);
    }

    fn with_indent<F>(&mut self, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.indent_block();
        f(self);
        self.close_indent();
    }

    pub fn check_errors(&self) -> Option<syn::Error> {
        let mut iter = self.err.iter();
        let mut combined = iter.next()?.clone();
        for err in iter {
            combined.combine(err.clone());
        }
        Some(combined)
    }
}

impl<'a, 'ast> Visit<'ast> for SlangEmitter<'a> {
    fn visit_expr(&mut self, i: &'ast syn::Expr) {
        match i {
            syn::Expr::Path(expr_path) => self.visit_expr_path(expr_path),
            syn::Expr::Binary(expr_binary) => self.visit_expr_binary(expr_binary),
            syn::Expr::Lit(expr_lit) => self.visit_expr_lit(expr_lit),
            syn::Expr::Paren(expr_paren) => self.visit_expr_paren(expr_paren),
            syn::Expr::Unary(expr_unary) => self.visit_expr_unary(expr_unary),
            syn::Expr::Index(expr_index) => self.visit_expr_index(expr_index),
            syn::Expr::Assign(expr_assign) => self.visit_expr_assign(expr_assign),
            syn::Expr::If(expr_if) => self.visit_expr_if(expr_if),
            syn::Expr::ForLoop(expr_for) => self.visit_expr_for_loop(expr_for),
            syn::Expr::MethodCall(expr_method) => self.visit_expr_method_call(expr_method),
            syn::Expr::Field(expr_field) => self.visit_expr_field(expr_field),

            _ => {
                let err = syn::Error::new(
                    i.span(),
                    "Gpu Kernel Compiler does not support this type of expression yet.",
                );
                self.err.push(err);
            }
        }
    }

    fn visit_stmt(&mut self, i: &'ast syn::Stmt) {
        match i {
            syn::Stmt::Local(local) => {
                self.push_indent();
                self.visit_local(local);
            }
            syn::Stmt::Expr(expr, _semi) => {
                self.push_indent();
                self.visit_expr(expr);

                if matches!(expr, syn::Expr::If(_) | syn::Expr::ForLoop(_)) {
                    self.push_str("\n");
                } else {
                    self.push_str(";\n");
                }
            }
            _ => {
                let err = syn::Error::new(
                    syn::spanned::Spanned::span(i),
                    "Gpu Kernel Compiler does not support this type of statement yet.",
                );
                self.err.push(err);

                syn::visit::visit_stmt(self, i);
            }
        }
    }

    fn visit_expr_path(&mut self, i: &'ast syn::ExprPath) {
        let path_str = i.to_token_stream().to_string().replace(' ', "");

        if let Some(mapped) = self.system_inputs.get(&path_str) {
            self.push_str(mapped);
        } else if self.push_fields.contains(&path_str) {
            self.push_str(&format!("g_args.{}", path_str));
        } else {
            self.push_str(&path_str);
        }
    }

    fn visit_local(&mut self, i: &'ast syn::Local) {
        self.push_str("let ");

        if let syn::Pat::Ident(pat_ident) = &i.pat {
            self.push_str(&pat_ident.ident.to_string());
        }

        if let Some(init) = &i.init {
            self.push_str(" = ");
            self.visit_expr(&init.expr);
        }

        self.push_str(";\n");
    }

    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        self.visit_expr(&i.left);
        let op_str = i.op.to_token_stream().to_string();
        self.push_str(&format!(" {} ", op_str));
        self.visit_expr(&i.right);
    }

    fn visit_expr_lit(&mut self, i: &'ast syn::ExprLit) {
        let lit_str = i.lit.to_token_stream().to_string();
        self.push_str(&lit_str);
    }

    fn visit_expr_paren(&mut self, i: &'ast syn::ExprParen) {
        self.push_str("(");
        self.visit_expr(&i.expr);
        self.push_str(")");
    }

    fn visit_expr_unary(&mut self, i: &'ast syn::ExprUnary) {
        let op_str = i.op.to_token_stream().to_string();
        self.push_str(&op_str);
        self.visit_expr(&i.expr);
    }

    fn visit_expr_index(&mut self, i: &'ast syn::ExprIndex) {
        self.visit_expr(&i.expr);
        self.push_str("[");
        self.visit_expr(&i.index);
        self.push_str("]");
    }

    fn visit_expr_assign(&mut self, i: &'ast syn::ExprAssign) {
        self.visit_expr(&i.left);
        self.push_str(" = ");
        self.visit_expr(&i.right);
    }

    fn visit_expr_if(&mut self, i: &'ast syn::ExprIf) {
        self.push_str("if (");
        self.visit_expr(&i.cond);
        self.push_str(") ");
        self.visit_block(&i.then_branch);

        if let Some((_, else_expr)) = &i.else_branch {
            self.push_indent();
            self.push_str(" else ");
            self.visit_expr(else_expr);
        }
    }

    fn visit_block(&mut self, i: &'ast syn::Block) {
        self.push_str("{\n");
        self.with_indent(|visitor| {
            // visitor.push_indent();

            for stmt in &i.stmts {
                visitor.visit_stmt(stmt);
            }
        });
        self.push_indent();
        self.push_str("}");
    }

    fn visit_expr_for_loop(&mut self, i: &'ast syn::ExprForLoop) {
        let var_name = match &*i.pat {
            syn::Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
            _ => "i".to_string(),
        };

        // for i in 0..N
        if let syn::Expr::Range(range) = &*i.expr {
            self.push_str(&format!("for (int {} = ", var_name));

            // Range start value (default 0)
            if let Some(start) = &range.start {
                self.visit_expr(start);
            } else {
                self.push_str("0");
            }

            // Comparison
            let op = match range.limits {
                syn::RangeLimits::HalfOpen(_) => "<",
                syn::RangeLimits::Closed(_) => "<=",
            };
            self.push_str(&format!("; {} {} ", var_name, op));

            // Range End
            if let Some(end) = &range.end {
                self.visit_expr(end);
            }

            // Add stuff
            self.push_str(&format!("; {}++) ", var_name));

            // Loop
            self.visit_block(&i.body);
        } else {
            // iterator is not standard, throws error
            panic!(
                "Only standard range-based for loops (e.g., 0..N) are supported in Slang transpiler."
            );
        }
    }

    fn visit_expr_method_call(&mut self, i: &'ast syn::ExprMethodCall) {
        let method_name = i.method.to_string();

        if method_name == "len" && i.args.is_empty() {
            if let syn::Expr::Path(expr_path) = &*i.receiver {
                let path_str = expr_path.to_token_stream().to_string().replace(' ', "");

                if let Some(mapped) = self.system_inputs.get(&path_str) {
                    self.push_str(&format!("{}_____len", mapped));
                } else if self.push_fields.contains(&path_str) {
                    self.push_str(&format!("g_args.{}_____len", path_str));
                } else {
                    self.push_str(&format!("{}_____len", path_str));
                }

                return;
            }
        }

        self.visit_expr(&i.receiver);
        self.push_str(&format!(".{}", method_name));

        self.push_str("(");
        for (idx, arg) in i.args.iter().enumerate() {
            if idx > 0 {
                self.push_str(", ");
            }
            self.visit_expr(arg);
        }
        self.push_str(")");
    }

    fn visit_expr_field(&mut self, i: &'ast syn::ExprField) {
        if let syn::Member::Named(ident) = &i.member {
            if ident == "len" {
                if let Some(path_str) = extract_pure_path_string(&i.base) {
                    if let Some(mapped) = self.system_inputs.get(&path_str) {
                        self.push_str(&format!("{}_____len", mapped));
                    } else if self.push_fields.contains(&path_str) {
                        self.push_str(&format!("g_args.{}_____len", path_str));
                    } else {
                        self.push_str(&format!("{}_____len", path_str));
                    }
                    return;
                }
            }
        }

        self.visit_expr(&i.base);
        self.push_str(".");
        match &i.member {
            syn::Member::Named(ident) => self.push_str(&ident.to_string()),
            syn::Member::Unnamed(index) => self.push_str(&index.index.to_string()),
        }
    }
}

fn extract_pure_path_string(expr: &syn::Expr) -> Option<String> {
    use quote::ToTokens;
    match expr {
        syn::Expr::Path(expr_path) => {
            Some(expr_path.to_token_stream().to_string().replace(' ', ""))
        }
        syn::Expr::Reference(expr_ref) => extract_pure_path_string(&expr_ref.expr),
        syn::Expr::Unary(expr_unary) => {
            if matches!(expr_unary.op, syn::UnOp::Deref(_)) {
                extract_pure_path_string(&expr_unary.expr)
            } else {
                None
            }
        }
        syn::Expr::Paren(expr_paren) => extract_pure_path_string(&expr_paren.expr),
        _ => None,
    }
}

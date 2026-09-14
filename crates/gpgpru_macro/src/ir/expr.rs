// src/ir/expr.rs

use super::{ExprId, VarId};

use crate::LiteralKind;

#[derive(Debug)]
pub enum IrExpr {
    Literal {
        value: String,
    },
    Var(VarId),
    Unary {
        op: syn::UnOp,
        value: ExprId,
    },
    Binary {
        left: ExprId,
        right: ExprId,
        op: syn::BinOp,
    },
    Index {
        base: ExprId,
        index: ExprId,
    },
    Field {
        base: ExprId,
        name: String,
    },
    Call {
        name: String,
        args: Vec<ExprId>,
    },
    MethodCall {
        base: ExprId,
        name: String,
        args: Vec<ExprId>,
    },
    Cast {
        expr: ExprId,
        target_type: LiteralKind,
    },
}

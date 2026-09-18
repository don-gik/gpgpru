// src/ir/expr.rs

use super::{ExprId, VarId};

use crate::TypeIdx;

#[derive(Debug)]
pub enum IrExpr {
    Literal {
        value: String,
        ty: TypeIdx,
    },
    Var {
        id: VarId,
        ty: TypeIdx,
    },
    Unary {
        op: syn::UnOp,
        value: ExprId,
        ty: TypeIdx,
    },
    Binary {
        left: ExprId,
        right: ExprId,
        op: syn::BinOp,
        ty: TypeIdx,
    },
    Index {
        base: ExprId,
        index: ExprId,
        ty: TypeIdx,
    },
    Field {
        base: ExprId,
        name: String,
        ty: TypeIdx,
    },
    Call {
        name: String,
        args: Vec<ExprId>,
        ty: TypeIdx,
    },
    MethodCall {
        base: ExprId,
        name: String,
        args: Vec<ExprId>,
        ty: TypeIdx,
    },
    Cast {
        expr: ExprId,
        ty: TypeIdx,
    },
    Repeat {
        value: ExprId,
        count: usize,
        ty: TypeIdx,
    },
}

impl IrExpr {
    pub fn ty(&self) -> TypeIdx {
        match self {
            IrExpr::Literal { ty, .. }
            | IrExpr::Var { ty, .. }
            | IrExpr::Unary { ty, .. }
            | IrExpr::Binary { ty, .. }
            | IrExpr::Index { ty, .. }
            | IrExpr::Field { ty, .. }
            | IrExpr::Call { ty, .. }
            | IrExpr::MethodCall { ty, .. }
            | IrExpr::Cast { ty, .. }
            | IrExpr::Repeat { ty, .. } => *ty,
        }
    }
}

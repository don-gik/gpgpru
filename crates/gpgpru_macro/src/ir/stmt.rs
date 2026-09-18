// src/ir/stmt.rs

use super::{ExprId, StmtId, VarId};
use crate::TypeIdx;

#[derive(Debug, Clone)]
pub struct IrBlock {
    pub stmts: Vec<StmtId>,
}

#[derive(Debug)]
pub enum IrStmt {
    Block(IrBlock),

    If {
        cond: ExprId,
        then_block: StmtId,
        else_branch: Option<StmtId>,
    },

    For {
        init: Option<StmtId>,
        cond: Option<ExprId>,
        step: Option<StmtId>,
        body: StmtId,
    },

    While {
        cond: ExprId,
        body: StmtId,
    },

    Let {
        var: VarId,
        init: ExprId,
        is_mut: bool,
    },

    Assign {
        target: ExprId,
        value: ExprId,
        op: Option<syn::BinOp>,
    },

    Expr(ExprId),

    Const {
        var: VarId,
        ty: TypeIdx,
        init: ExprId,
    },

    Return(Option<ExprId>),

    Break,
    Continue,
}

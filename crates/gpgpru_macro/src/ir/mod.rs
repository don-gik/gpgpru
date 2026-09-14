// src/ir/mod.rs

mod expr;
mod id;
mod stmt;
mod var;

pub use expr::IrExpr;
pub use id::{ExprId, StmtId, VarId};
pub use stmt::{IrBlock, IrStmt};
pub use var::{VarInfo, VarStorage};

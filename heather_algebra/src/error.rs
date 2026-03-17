use thiserror::Error;

#[derive(Debug, Error)]
pub enum AlgebraError {
    #[error("dimension mismatch: left has {left}, right has {right}")]
    DimensionMismatch { left: usize, right: usize },

    #[error("empty snapshot: no locations to operate on")]
    EmptySnapshot,

    #[error("invalid scalar: {0}")]
    InvalidScalar(String),

    #[error("composition failed: {0}")]
    ComposeFailed(String),

    #[error("heather_db error: {0}")]
    Db(#[from] heather_db::HeatherError),
}

pub type Result<T> = std::result::Result<T, AlgebraError>;

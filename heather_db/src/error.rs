use thiserror::Error;

#[derive(Debug, Error)]
pub enum HeatherError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("memory is empty: no locations stored")]
    EmptyMemory,

    #[error("location not found: {0}")]
    LocationNotFound(u64),

    #[error("maximum locations reached: {0}")]
    MaxLocationsReached(usize),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("lock poisoned: a thread panicked while holding a lock")]
    LockPoisoned,
}

impl From<heed::Error> for HeatherError {
    fn from(e: heed::Error) -> Self {
        HeatherError::Storage(e.to_string())
    }
}

impl From<bincode::Error> for HeatherError {
    fn from(e: bincode::Error) -> Self {
        HeatherError::Serialization(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, HeatherError>;

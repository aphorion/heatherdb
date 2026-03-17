pub mod compose;
pub mod consolidate;
pub mod error;
pub mod ops;
pub mod routing;
pub mod snapshot;
mod traits;

pub use compose::{ComposeParams, ComposeResult, ComposeDiagnostics, compose};
pub use error::AlgebraError;
pub use routing::{ComposedEAM, ComposeReadResult, compose_read};
pub use snapshot::EAMSnapshot;

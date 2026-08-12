pub mod bind;
pub mod compose;
pub mod consolidate;
pub mod error;
pub mod ops;
pub mod routing;
pub mod snapshot;
mod traits;

pub use bind::{bind, bind_vec, bind_with_limit, circular_convolve, involve, unbind, unbind_vec};
pub use compose::{ComposeDiagnostics, ComposeParams, ComposeResult, compose};
pub use error::AlgebraError;
pub use routing::{ComposeReadResult, ComposedEAM, compose_read};
pub use snapshot::EAMSnapshot;

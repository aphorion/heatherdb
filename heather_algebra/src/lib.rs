pub mod bind;
pub mod compose;
pub mod consolidate;
pub mod error;
pub mod ops;
pub mod permute;
pub mod routing;
pub mod snapshot;
mod traits;

pub use bind::{
    bind, bind_vec, bind_with_limit, circular_convolve, involve, pow_vec, unbind, unbind_exact_vec,
    unbind_vec,
};
pub use compose::{ComposeDiagnostics, ComposeParams, ComposeResult, compose};
pub use error::AlgebraError;
pub use permute::{Permutation, permute, permute_inv, permute_pow, permute_snapshot};
pub use routing::{ComposeReadResult, ComposedEAM, compose_read};
pub use snapshot::EAMSnapshot;

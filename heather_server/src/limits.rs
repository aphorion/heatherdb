//! Server-side request-size limits.
//!
//! Body-size limiting alone doesn't bound the engine's work: a 2 MB JSON
//! body can describe an algebra op whose n×m cross product allocates
//! gigabytes. These caps bound the *result* of a request, not its wire
//! size. All are CLI/env tunable; `0` disables a cap.
//!
//! Stored in a process-wide `OnceLock` (set once at boot in `main.rs`) so
//! the dual-router handlers don't need their state shapes changed.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Max locations an algebra add/sub/bind result may contain
    /// (estimated as n_a × n_b, or n_a × max_cross_k when limited).
    pub max_algebra_locations: usize,
    /// Max address/counter pairs per bulk_load request.
    pub max_bulk_items: usize,
    /// Max queries per batch_analyze request.
    pub max_batch_queries: usize,
}

pub const DEFAULT_MAX_ALGEBRA_LOCATIONS: usize = 250_000;
pub const DEFAULT_MAX_BULK_ITEMS: usize = 100_000;
pub const DEFAULT_MAX_BATCH_QUERIES: usize = 1_000;

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_algebra_locations: DEFAULT_MAX_ALGEBRA_LOCATIONS,
            max_bulk_items: DEFAULT_MAX_BULK_ITEMS,
            max_batch_queries: DEFAULT_MAX_BATCH_QUERIES,
        }
    }
}

static LIMITS: OnceLock<Limits> = OnceLock::new();

/// Set the process-wide limits. Call once at boot; later calls are ignored.
pub fn init(limits: Limits) {
    let _ = LIMITS.set(limits);
}

/// Current limits (defaults if `init` was never called, e.g. in tests).
pub fn get() -> Limits {
    LIMITS.get().copied().unwrap_or_default()
}

/// Estimated result size of a pairwise algebra op (add/sub/bind):
/// `max_cross_k == 0` means full cartesian product.
pub fn estimated_cross_locations(n_a: usize, n_b: usize, max_cross_k: usize) -> usize {
    if max_cross_k == 0 || max_cross_k >= n_b {
        n_a.saturating_mul(n_b)
    } else {
        n_a.saturating_mul(max_cross_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_estimate() {
        assert_eq!(estimated_cross_locations(10, 20, 0), 200);
        assert_eq!(estimated_cross_locations(10, 20, 5), 50);
        assert_eq!(estimated_cross_locations(10, 20, 50), 200);
    }

    #[test]
    fn defaults_without_init() {
        let l = get();
        assert!(l.max_algebra_locations > 0);
        assert!(l.max_bulk_items > 0);
        assert!(l.max_batch_queries > 0);
    }
}

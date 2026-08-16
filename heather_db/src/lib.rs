//! # HeatherDB
//!
//! Adaptive Elastic Associative Memory — a persistent associative memory library
//! with multi-collection support.
//!
//! HeatherDB stores high-dimensional vectors and retrieves them associatively.
//! Unlike a key-value store, you don't need an exact key to retrieve data.
//! Noisy, partial, or approximate queries reconstruct the closest stored pattern.
//!
//! ## Usage
//!
//! ```no_run
//! use std::path::Path;
//! use heather_db::{Hive, EAMConfig, ReadStrategy};
//!
//! // Open a hive (manages multiple collections)
//! let config = EAMConfig::new(64).unwrap();
//! let hive = Hive::open(Path::new("/tmp/mydb"), config, 256).unwrap();
//!
//! // Get or create a collection (auto-creates on first access)
//! let col = hive.get_or_create_collection("vectors").unwrap();
//!
//! // Write a pattern
//! let pattern = vec![0.1; 64];
//! col.write(&pattern).unwrap();
//!
//! // Read it back (works with noisy queries too)
//! let result = col.read(&pattern, ReadStrategy::HopfieldIter).unwrap();
//! ```
//!
//! ## Thread Safety
//!
//! [`Hive`] and [`Collection`] are `Send + Sync`. All methods take `&self` —
//! reads acquire a shared lock, writes acquire an exclusive lock.

pub mod collection;
pub mod config;
pub mod db_config;
pub mod error;
pub mod hive;
pub mod location;
pub mod merge;
pub mod read;
pub mod server;
pub mod store;
pub mod vec_ops;
pub mod write;

pub use collection::{
    Collection, CompressResult, EAMStats, LocationSummary, MIN_CLEANUP_BETA, MultiRoleHit,
    MultiRoleResults, ReadStrategy, RoleCleanup,
};
pub use config::EAMConfig;
pub use db_config::{DEFAULT_MAP_SIZE_MB, DbConfig, DreamConfig, DreamMode, validate_db_name};
pub use error::HeatherError;
pub use hive::Hive;
pub use location::{HardLocation, LocationId};
pub use read::{ActivatedLocation, AttentionContributor, AttentionTrace, ReadTrace};
pub use server::{DEFAULT_DB, Server, default_db_config, try_acquire_engine_lock};
pub use write::WriteOpts;

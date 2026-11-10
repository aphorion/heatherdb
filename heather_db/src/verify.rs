//! Offline integrity checks: decode every stored record of a database and
//! report what is in the current format, what predates it, and what is
//! damaged (see [`crate::codec`] and `docs/reference/storage-format.md`).

use serde::Serialize;

/// How many records of one kind decoded, and how.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RecordCounts {
    /// In the current, checksummed envelope.
    pub current: u64,
    /// From before the envelope, still readable.
    pub legacy: u64,
    /// Fail their checksum, are truncated, or do not decode.
    pub corrupt: u64,
    /// In a record version newer than this HeatherDB reads.
    pub unsupported: u64,
}

impl RecordCounts {
    pub fn bad(&self) -> u64 {
        self.corrupt + self.unsupported
    }
}

/// One collection's records.
#[derive(Debug, Clone, Serialize)]
pub struct CollectionReport {
    pub collection: String,
    pub locations: RecordCounts,
    pub documents: RecordCounts,
    /// The first [`MAX_PROBLEMS`] damaged records, by key, with why.
    pub problems: Vec<String>,
}

impl CollectionReport {
    pub fn is_clean(&self) -> bool {
        self.locations.bad() + self.documents.bad() == 0
    }
}

/// Damaged records listed per collection (all are counted).
pub const MAX_PROBLEMS: usize = 20;

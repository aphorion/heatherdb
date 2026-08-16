//! Persisted operation log — who did what, when.
//!
//! This is a generic access-log primitive, comparable in scope to Postgres's
//! `pgaudit` extension or MongoDB's audit log: it records raw events (who,
//! what route, against which object, when, with what outcome) and stays
//! agnostic about what those events *mean*. It does not decide what counts
//! as "trending" or "relevant" or build any rollup aimed at a particular UI
//! panel — a consumer that wants that queries the raw log (`GET
//! /db/{db}/audit`, with its user/collection/route/time filters) and
//! aggregates it for its own purposes. Keeping that logic out of the engine
//! is what keeps this primitive usable by any application built on top of
//! HeatherDB, not just the one that motivated it.
//!
//! One `_audit` sub-DB per database env, alongside `_registry`, `_locations`,
//! `_metadata`, `_documents` and `_doc_index`. Records are keyed
//! `[timestamp_ms: 8B BE | seq: 8B BE]`, so LMDB's own key order **is** time
//! order: newest-first paging is a reverse cursor walk from the end of the
//! tree, and a time window is a prefix-bounded scan rather than a full table
//! read. `seq` breaks ties between events that land in the same millisecond
//! and keeps keys unique.
//!
//! # What is stored for the query vector: a hash, by default
//!
//! By default an audit record carries only a 64-bit hash of the query
//! vector, never the vector itself. Two reasons, both load-bearing:
//!
//!   1. **Size.** At D=4096 a query is ~32 KB of `f64`. A repository doing a
//!      million reads would spend 32 GB auditing reads against data that is
//!      itself smaller than the log. The log would dwarf what it audits.
//!   2. **Sensitivity.** A query vector is a *reconstructable statement of
//!      what someone was looking for* — run it back through `attention` and
//!      it names the documents. Storing it turns the audit log into a second,
//!      unguarded copy of every search anyone has ever made over confidential
//!      personnel, legal and financial material. The hash keeps the property
//!      the log actually needs — repeated identical queries correlate — while
//!      being one-way.
//!
//! That is a default, not a mandate: `AuditConfig::store_raw_query` lets an
//! operator who wants the raw vectors for their own analysis opt in
//! per-database. The engine picks the safe default and gets out of the way.
//!
//! # Write-path design: buffer in memory, flush in batches
//!
//! Recording a read means writing on the read path, and LMDB has exactly one
//! writer. Opening a write transaction per read would funnel every concurrent
//! reader through that single lock and convert a lock-free MVCC read path into
//! a serialised one.
//!
//! So [`AuditLog::record`] only takes a `Mutex<Vec<_>>`, pushes, and returns —
//! no transaction, no I/O, no `fsync`. [`AuditLog::flush`] drains the buffer
//! *out* of the lock and writes the whole batch in one transaction, driven by
//! two triggers: a size threshold (`flush_threshold`, reported back to the
//! caller by `record`'s return value) and a periodic tick the server owns.
//! Because LMDB readers never block on a writer, that batch write is invisible
//! to concurrent reads; it contends only with other *writes*, and one
//! transaction per few hundred events instead of one per event makes that
//! contention negligible.
//!
//! **The trade-off, stated plainly:** an unclean shutdown (SIGKILL, power
//! loss) loses whatever is still buffered — at most `flush_threshold` records
//! or `flush_interval_secs` of activity. That is accepted deliberately.
//! Losing the last second of the access log degrades an audit trail slightly;
//! serialising every read behind the writer lock degrades the product. A
//! graceful shutdown flushes, so the loss window only opens on a hard kill.
//!
//! A hard `max_buffer` cap bounds memory if the flusher ever stalls: records
//! past the cap are dropped and counted in [`AuditLog::dropped`] rather than
//! growing the heap without limit.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{HeatherError, Result};
use crate::store::Store;

/// Default cap on rows a single query is allowed to examine. Bounds the cost
/// of a filtered scan that matches nothing.
pub const DEFAULT_MAX_SCAN: usize = 200_000;

/// One recorded access. `location_ids` / `document_ids` are the identifiers
/// the request actually returned — that is what makes the log answer "who
/// has seen this document", not merely "who ran a search".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditRecord {
    /// Monotonic per-database sequence number. Also the key tiebreaker.
    pub seq: u64,
    /// Unix milliseconds at the moment the request completed.
    pub timestamp_ms: u64,
    /// Authenticated username, or `-` when auth is disabled.
    pub user: String,
    /// The user's scope label at request time (`root` / `db:<name>`).
    pub scope: String,
    pub database: String,
    pub collection: String,
    /// Logical route name, e.g. `read`, `attention`, `documents/query`.
    pub route: String,
    /// HTTP status the request returned.
    pub status: u16,
    /// How many results the caller got back.
    pub result_count: usize,
    /// Hard-location ids returned (attention contributors, activated locations).
    pub location_ids: Vec<u64>,
    /// Document ids returned.
    pub document_ids: Vec<u64>,
    /// Stable 64-bit hash of the query vector, hex. See the module docs —
    /// this is populated whether or not the raw vector is also stored.
    pub query_hash: Option<String>,
    /// The raw query vector, only when `AuditConfig::store_raw_query` is set
    /// for this database. `None` in the safe default configuration.
    pub query_raw: Option<Vec<f64>>,
}

/// Per-database audit settings, persisted in `db.toml` under `[audit]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditConfig {
    /// On by default: the access log is a security requirement, not a
    /// feature flag an operator has to discover.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Hard ceiling on stored records. The oldest are pruned past this.
    #[serde(default = "default_max_entries")]
    pub max_entries: u64,
    /// Age bound in days. `0` disables the age bound (`max_entries` still applies).
    #[serde(default = "default_retention_days")]
    pub retention_days: u64,
    /// Buffered records that trigger an out-of-band flush.
    #[serde(default = "default_flush_threshold")]
    pub flush_threshold: usize,
    /// Periodic flush cadence, seconds.
    #[serde(default = "default_flush_interval_secs")]
    pub flush_interval_secs: u64,
    /// Hard memory cap on the buffer. Records past it are dropped and counted.
    #[serde(default = "default_max_buffer")]
    pub max_buffer: usize,
    /// When `true`, store the raw query vector on every record in addition
    /// to its hash. Off by default — hashing is the safe default described
    /// in the module docs; this lets an operator who wants raw vectors for
    /// their own analysis opt in explicitly, rather than the engine deciding
    /// unilaterally for every deployment.
    #[serde(default)]
    pub store_raw_query: bool,
    /// Who may read this database's audit log via `GET /db/{db}/audit`.
    /// Defaults to [`AuditVisibility::Root`] — the safest option, and the
    /// only behavior this log had before the policy became configurable.
    #[serde(default)]
    pub visibility: AuditVisibility,
}

/// Read-access policy for a database's audit log.
///
/// The log is a record of what *every* user of a database searched for, so
/// the default is deliberately conservative. This is one binary policy
/// decision, not a role system: there is no per-user or per-route dimension,
/// just "root only" vs. "a database's own users may read their database's
/// log".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AuditVisibility {
    /// Only a `Root`-scoped user may read this database's audit log. Matches
    /// the original, non-configurable behavior.
    #[default]
    Root,
    /// A `Database(name)`-scoped user may read their own database's audit
    /// log (still never another database's). `Root` can always read it.
    DbUsers,
}

fn default_true() -> bool {
    true
}
fn default_max_entries() -> u64 {
    1_000_000
}
fn default_retention_days() -> u64 {
    90
}
fn default_flush_threshold() -> usize {
    256
}
fn default_flush_interval_secs() -> u64 {
    2
}
fn default_max_buffer() -> usize {
    16_384
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            max_entries: default_max_entries(),
            retention_days: default_retention_days(),
            flush_threshold: default_flush_threshold(),
            flush_interval_secs: default_flush_interval_secs(),
            max_buffer: default_max_buffer(),
            store_raw_query: false,
            visibility: AuditVisibility::Root,
        }
    }
}

impl AuditConfig {
    /// The age bound as milliseconds, or `None` when `retention_days == 0`.
    pub fn retention_ms(&self) -> Option<u64> {
        if self.retention_days == 0 {
            None
        } else {
            Some(self.retention_days * 24 * 60 * 60 * 1000)
        }
    }
}

/// Filters for [`crate::Hive::query_audit`]. All fields are exact matches
/// except the time bounds, which are inclusive.
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    pub user: Option<String>,
    pub collection: Option<String>,
    pub route: Option<String>,
    pub since_ms: Option<u64>,
    pub until_ms: Option<u64>,
    /// Rows returned. Callers are expected to clamp this before calling.
    pub limit: usize,
    /// Rows examined before the scan gives up.
    pub max_scan: usize,
}

impl AuditQuery {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            max_scan: DEFAULT_MAX_SCAN,
            ..Default::default()
        }
    }

    fn matches(&self, rec: &AuditRecord) -> bool {
        // `is_none_or` rather than nested `if let` — the collapsed `if let`
        // form is a let-chain, which the 1.84 MSRV job doesn't accept.
        self.user.as_ref().is_none_or(|u| &rec.user == u)
            && self
                .collection
                .as_ref()
                .is_none_or(|c| &rec.collection == c)
            && self.route.as_ref().is_none_or(|r| &rec.route == r)
    }
}

/// In-memory staging buffer in front of the `_audit` sub-DB.
///
/// See the module docs for why this exists and what it costs.
pub struct AuditLog {
    config: AuditConfig,
    buffer: Mutex<Vec<AuditRecord>>,
    seq: AtomicU64,
    dropped: AtomicU64,
}

impl AuditLog {
    /// Build a log over an already-open store, resuming the sequence counter
    /// from the highest key on disk so restarts don't reuse keys.
    pub fn open(store: &Store, config: AuditConfig) -> Result<Self> {
        let next_seq = store.last_audit_seq()?.map(|s| s + 1).unwrap_or(0);
        Ok(Self {
            config,
            buffer: Mutex::new(Vec::new()),
            seq: AtomicU64::new(next_seq),
            dropped: AtomicU64::new(0),
        })
    }

    pub fn config(&self) -> &AuditConfig {
        &self.config
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// Records dropped because the buffer hit `max_buffer`.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Number of records currently staged in memory.
    pub fn pending(&self) -> usize {
        self.buffer.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// Stage a record. Never touches LMDB. Returns `true` when the buffer has
    /// reached `flush_threshold` and the caller should schedule a flush.
    ///
    /// `seq` and `timestamp_ms` on the passed record are overwritten — the log
    /// owns both, so callers can't forge ordering.
    pub fn record(&self, mut rec: AuditRecord) -> bool {
        if !self.config.enabled {
            return false;
        }
        rec.seq = self.seq.fetch_add(1, Ordering::Relaxed);
        rec.timestamp_ms = now_ms();

        let mut buf = match self.buffer.lock() {
            Ok(b) => b,
            Err(_) => return false, // poisoned: drop rather than panic a read
        };
        if buf.len() >= self.config.max_buffer {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        buf.push(rec);
        buf.len() >= self.config.flush_threshold
    }

    /// Drain the buffer into one write transaction and enforce the retention
    /// bound. Returns how many records were persisted.
    ///
    /// The buffer is taken *out* of the mutex before the transaction opens, so
    /// concurrent `record` calls never wait on LMDB.
    pub fn flush(&self, store: &Store) -> Result<usize> {
        let batch: Vec<AuditRecord> = {
            let mut buf = self.buffer.lock().map_err(|_| HeatherError::LockPoisoned)?;
            if buf.is_empty() {
                // Still prune on an empty tick so an idle database eventually
                // ages out records past their retention window.
                drop(buf);
                self.prune(store)?;
                return Ok(0);
            }
            std::mem::take(&mut *buf)
        };

        let n = batch.len();
        let mut txn = store.write_txn()?;
        for rec in &batch {
            let key = audit_key(rec.timestamp_ms, rec.seq);
            let value = bincode::serialize(rec)?;
            store.append_audit(&mut txn, &key, &value)?;
        }
        let min_ts = self
            .config
            .retention_ms()
            .map(|ms| now_ms().saturating_sub(ms))
            .unwrap_or(0);
        store.prune_audit(&mut txn, self.config.max_entries, min_ts)?;
        txn.commit()?;
        Ok(n)
    }

    fn prune(&self, store: &Store) -> Result<()> {
        let min_ts = self
            .config
            .retention_ms()
            .map(|ms| now_ms().saturating_sub(ms))
            .unwrap_or(0);
        // Nothing to do when neither bound can bite.
        let mut txn = store.write_txn()?;
        let pruned = store.prune_audit(&mut txn, self.config.max_entries, min_ts)?;
        if pruned == 0 {
            txn.abort();
            return Ok(());
        }
        txn.commit()?;
        Ok(())
    }

    /// Newest-first scan with filters applied. Flush before calling if you
    /// want buffered records included — [`crate::Hive::query_audit`] does.
    pub fn query(&self, store: &Store, q: &AuditQuery) -> Result<Vec<AuditRecord>> {
        let mut out = Vec::new();
        store.scan_audit_desc(q.since_ms, q.until_ms, q.max_scan, |bytes| {
            match bincode::deserialize::<AuditRecord>(bytes) {
                Ok(rec) if q.matches(&rec) => {
                    out.push(rec);
                    out.len() < q.limit
                }
                // A record that doesn't decode is skipped, not fatal — one bad
                // row must not blind an operator to the rest of the log.
                _ => true,
            }
        })?;
        Ok(out)
    }
}

/// `[timestamp_ms: 8B BE | seq: 8B BE]` — time-ordered by construction.
pub fn audit_key(timestamp_ms: u64, seq: u64) -> [u8; 16] {
    let mut key = [0u8; 16];
    key[..8].copy_from_slice(&timestamp_ms.to_be_bytes());
    key[8..].copy_from_slice(&seq.to_be_bytes());
    key
}

/// Timestamp half of an audit key. `0` for a malformed key, which sorts it
/// out of every time window rather than crashing a scan.
pub fn key_timestamp(key: &[u8]) -> u64 {
    key.get(..8)
        .and_then(|b| <[u8; 8]>::try_from(b).ok())
        .map(u64::from_be_bytes)
        .unwrap_or(0)
}

/// Sequence half of an audit key.
pub fn key_seq(key: &[u8]) -> u64 {
    key.get(8..16)
        .and_then(|b| <[u8; 8]>::try_from(b).ok())
        .map(u64::from_be_bytes)
        .unwrap_or(0)
}

/// FNV-1a over the raw IEEE-754 bits of a query vector, hex-encoded.
///
/// Deliberately *not* `DefaultHasher`: that is explicitly unstable across
/// releases, and an audit log whose hashes stop matching after an upgrade
/// can't correlate repeated queries — the one property the hash exists for.
/// FNV-1a is fixed, cheap, and one-way enough for this purpose: it is a
/// correlation key, not a security primitive, and the vector is never
/// recoverable from it.
pub fn query_hash(query: &[f64]) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for v in query {
        for b in v.to_bits().to_be_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
    }
    format!("{h:016x}")
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_sort_by_time_then_seq() {
        let mut keys = vec![
            audit_key(200, 0),
            audit_key(100, 5),
            audit_key(100, 1),
            audit_key(300, 0),
        ];
        keys.sort();
        assert_eq!(
            keys,
            vec![
                audit_key(100, 1),
                audit_key(100, 5),
                audit_key(200, 0),
                audit_key(300, 0),
            ]
        );
        assert_eq!(key_timestamp(&audit_key(4242, 7)), 4242);
        assert_eq!(key_seq(&audit_key(4242, 7)), 7);
    }

    #[test]
    fn query_hash_is_stable_and_discriminating() {
        let a = query_hash(&[0.5, -1.25, 3.0]);
        assert_eq!(a, query_hash(&[0.5, -1.25, 3.0]));
        assert_ne!(a, query_hash(&[0.5, -1.25, 3.000_001]));
        assert_ne!(a, query_hash(&[]));
        // 16 hex chars, always.
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn retention_ms_zero_means_unbounded() {
        let mut cfg = AuditConfig::default();
        assert_eq!(cfg.retention_ms(), Some(90 * 86_400_000));
        cfg.retention_days = 0;
        assert_eq!(cfg.retention_ms(), None);
    }

    /* ─── end-to-end over a real store ────────────────────────────────── */

    use crate::config::EAMConfig;
    use crate::hive::Hive;
    use tempfile::TempDir;

    fn hive_with(cfg: AuditConfig) -> (TempDir, Hive) {
        let dir = TempDir::new().unwrap();
        let hive = Hive::open_with_audit(dir.path(), EAMConfig::new(8).unwrap(), 64, cfg).unwrap();
        (dir, hive)
    }

    fn rec(user: &str, collection: &str, route: &str, docs: Vec<u64>) -> AuditRecord {
        AuditRecord {
            seq: 0,
            timestamp_ms: 0,
            user: user.into(),
            scope: "root".into(),
            database: "default".into(),
            collection: collection.into(),
            route: route.into(),
            status: 200,
            result_count: docs.len(),
            location_ids: vec![],
            document_ids: docs,
            query_hash: Some(query_hash(&[1.0, 2.0])),
            query_raw: None,
        }
    }

    #[test]
    fn records_flush_and_read_back_newest_first() {
        let (_d, hive) = hive_with(AuditConfig::default());
        hive.record_audit(rec("alice", "bids", "attention", vec![1]));
        hive.record_audit(rec("bob", "bids", "read", vec![]));
        // Nothing is on disk until a flush — that IS the design.
        assert_eq!(hive.audit().pending(), 2);

        // query_audit flushes first, so both are visible.
        let got = hive.query_audit(&AuditQuery::new(10)).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].user, "bob", "newest first");
        assert_eq!(got[1].user, "alice");
        assert_eq!(got[1].document_ids, vec![1]);
        assert_eq!(got[1].query_hash, Some(query_hash(&[1.0, 2.0])));
        assert_eq!(hive.audit().pending(), 0);
    }

    #[test]
    fn threshold_reports_when_a_flush_is_due() {
        let cfg = AuditConfig {
            flush_threshold: 3,
            ..Default::default()
        };
        let (_d, hive) = hive_with(cfg);
        assert!(!hive.record_audit(rec("a", "c", "read", vec![])));
        assert!(!hive.record_audit(rec("a", "c", "read", vec![])));
        assert!(hive.record_audit(rec("a", "c", "read", vec![])));
    }

    #[test]
    fn buffer_cap_drops_rather_than_growing_without_bound() {
        let cfg = AuditConfig {
            max_buffer: 2,
            flush_threshold: 1000,
            ..Default::default()
        };
        let (_d, hive) = hive_with(cfg);
        for _ in 0..5 {
            hive.record_audit(rec("a", "c", "read", vec![]));
        }
        assert_eq!(hive.audit().pending(), 2);
        assert_eq!(hive.audit().dropped(), 3);
    }

    #[test]
    fn filters_select_by_user_collection_and_route() {
        let (_d, hive) = hive_with(AuditConfig::default());
        hive.record_audit(rec("alice", "bids", "attention", vec![]));
        hive.record_audit(rec("alice", "hr", "read", vec![]));
        hive.record_audit(rec("bob", "bids", "read", vec![]));
        hive.flush_audit().unwrap();

        let by_user = AuditQuery {
            user: Some("alice".into()),
            ..AuditQuery::new(10)
        };
        assert_eq!(hive.query_audit(&by_user).unwrap().len(), 2);

        let by_col = AuditQuery {
            collection: Some("bids".into()),
            ..AuditQuery::new(10)
        };
        assert_eq!(hive.query_audit(&by_col).unwrap().len(), 2);

        let by_route = AuditQuery {
            route: Some("read".into()),
            ..AuditQuery::new(10)
        };
        assert_eq!(hive.query_audit(&by_route).unwrap().len(), 2);

        let combined = AuditQuery {
            user: Some("alice".into()),
            collection: Some("bids".into()),
            route: Some("attention".into()),
            ..AuditQuery::new(10)
        };
        assert_eq!(hive.query_audit(&combined).unwrap().len(), 1);

        let nothing = AuditQuery {
            user: Some("nobody".into()),
            ..AuditQuery::new(10)
        };
        assert!(hive.query_audit(&nothing).unwrap().is_empty());
    }

    #[test]
    fn limit_and_time_bounds_apply() {
        let (_d, hive) = hive_with(AuditConfig::default());
        for _ in 0..5 {
            hive.record_audit(rec("a", "c", "read", vec![]));
        }
        hive.flush_audit().unwrap();

        assert_eq!(hive.query_audit(&AuditQuery::new(2)).unwrap().len(), 2);

        let future = AuditQuery {
            since_ms: Some(now_ms() + 60_000),
            ..AuditQuery::new(10)
        };
        assert!(hive.query_audit(&future).unwrap().is_empty());

        let past = AuditQuery {
            until_ms: Some(now_ms().saturating_sub(60_000)),
            ..AuditQuery::new(10)
        };
        assert!(hive.query_audit(&past).unwrap().is_empty());

        let window = AuditQuery {
            since_ms: Some(now_ms().saturating_sub(60_000)),
            until_ms: Some(now_ms() + 60_000),
            ..AuditQuery::new(10)
        };
        assert_eq!(hive.query_audit(&window).unwrap().len(), 5);
    }

    #[test]
    fn retention_bound_caps_the_log() {
        let cfg = AuditConfig {
            max_entries: 3,
            flush_threshold: 1000, // flush only when we say so
            ..Default::default()
        };
        let (_d, hive) = hive_with(cfg);
        for i in 0..10u64 {
            hive.record_audit(rec("a", "c", "read", vec![i]));
            hive.flush_audit().unwrap();
        }
        let got = hive.query_audit(&AuditQuery::new(100)).unwrap();
        assert_eq!(got.len(), 3, "max_entries must bound the stored log");
        // The survivors are the newest three.
        assert_eq!(got[0].document_ids, vec![9]);
        assert_eq!(got[2].document_ids, vec![7]);
    }

    #[test]
    fn retention_age_bound_drops_stale_records() {
        // retention_days can't express sub-day ages, so drive prune directly
        // with an explicit cutoff: everything currently stored is "stale".
        let (_d, hive) = hive_with(AuditConfig::default());
        for _ in 0..4 {
            hive.record_audit(rec("a", "c", "read", vec![]));
        }
        hive.flush_audit().unwrap();
        assert_eq!(hive.query_audit(&AuditQuery::new(100)).unwrap().len(), 4);

        hive.prune_audit_before(now_ms() + 1).unwrap();
        assert!(hive.query_audit(&AuditQuery::new(100)).unwrap().is_empty());
    }

    #[test]
    fn disabled_audit_writes_nothing() {
        let cfg = AuditConfig {
            enabled: false,
            ..Default::default()
        };
        let (_d, hive) = hive_with(cfg);
        for _ in 0..10 {
            assert!(!hive.record_audit(rec("alice", "bids", "attention", vec![1])));
        }
        assert_eq!(hive.audit().pending(), 0);
        assert_eq!(hive.flush_audit().unwrap(), 0);
        assert!(hive.query_audit(&AuditQuery::new(100)).unwrap().is_empty());
    }

    #[test]
    fn sequence_resumes_across_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let hive = Hive::open(dir.path(), EAMConfig::new(8).unwrap(), 64).unwrap();
            hive.record_audit(rec("a", "c", "read", vec![]));
            hive.record_audit(rec("a", "c", "read", vec![]));
            hive.flush_audit().unwrap();
        }
        let hive = Hive::open(dir.path(), EAMConfig::new(8).unwrap(), 64).unwrap();
        hive.record_audit(rec("b", "c", "read", vec![]));
        let got = hive.query_audit(&AuditQuery::new(10)).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].seq, 2, "sequence must not restart at 0");
    }
}

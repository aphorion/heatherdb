use std::path::Path;

use heed::types::Bytes;
use heed::{Database, Env, EnvFlags, EnvOpenOptions, FlagSetMode, RoTxn, RwTxn};

use crate::config::Precision;
use crate::error::{HeatherError, Result};
use crate::location::{HardLocation, LocationId};

const NEXT_COLLECTION_ID_KEY: &str = "_next_collection_id";

/// Composite key: [collection_id: 4 bytes BE | location_id: 8 bytes BE]
fn location_key(collection_id: u32, location_id: u64) -> [u8; 12] {
    let mut key = [0u8; 12];
    key[..4].copy_from_slice(&collection_id.to_be_bytes());
    key[4..].copy_from_slice(&location_id.to_be_bytes());
    key
}

/// 4-byte prefix for scanning all locations in a collection.
fn collection_prefix(collection_id: u32) -> [u8; 4] {
    collection_id.to_be_bytes()
}

/// Composite key: [collection_id: 4B | write_index: 8B BE]
fn document_key(collection_id: u32, write_index: u64) -> [u8; 12] {
    let mut key = [0u8; 12];
    key[..4].copy_from_slice(&collection_id.to_be_bytes());
    key[4..].copy_from_slice(&write_index.to_be_bytes());
    key
}

/// Composite key: [collection_id: 4 bytes BE | metadata_key bytes]
fn metadata_key(collection_id: u32, key: &str) -> Vec<u8> {
    let prefix = collection_id.to_be_bytes();
    let mut result = Vec::with_capacity(4 + key.len());
    result.extend_from_slice(&prefix);
    result.extend_from_slice(key.as_bytes());
    result
}

pub struct Store {
    env: Env,
    /// How vectors are encoded on disk (see [`crate::config::Precision`]).
    precision: Precision,
    registry_db: Database<Bytes, Bytes>,
    locations_db: Database<Bytes, Bytes>,
    metadata_db: Database<Bytes, Bytes>,
    documents_db: Database<Bytes, Bytes>,
    doc_index_db: Database<Bytes, Bytes>,
    audit_db: Database<Bytes, Bytes>,
}

/// A [`HardLocation`] with f32 vectors, as an f32 store keeps it.
#[derive(serde::Serialize, serde::Deserialize)]
struct CompactLocation {
    id: LocationId,
    address: Vec<f32>,
    counter: Vec<f32>,
    write_count: f64,
    neighbors: Vec<u64>,
    surprise_mass: f64,
}

fn narrow(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// A stored document: its id and its serialised (vector, metadata).
pub type RawDocument = (u64, Vec<u8>);

impl Store {
    /// Whether every commit is synced to disk (the default).
    pub fn syncs(&self) -> Result<bool> {
        Ok(self.env.get_flags()? & EnvFlags::NO_SYNC.bits() == 0)
    }

    /// Turns per-commit syncing off (`false`) or back on (`true`).
    ///
    /// Without it (LMDB `NO_SYNC`) commits reach the OS but not the disk:
    /// much faster where a sync is slow (macOS `F_FULLFSYNC`), but a system
    /// crash (not a process crash) can lose recent commits or, on file
    /// systems that reorder writes, corrupt the database. For data that can
    /// be rebuilt, such as a memory replayed from a stream. Turning it back on
    /// syncs everything written in between before returning. LMDB keeps the
    /// flag on the open environment, not in the file, so a restarted server
    /// syncs again.
    pub fn set_sync(&self, sync: bool) -> Result<()> {
        let mode = if sync {
            FlagSetMode::Disable
        } else {
            FlagSetMode::Enable
        };
        // SAFETY: NO_SYNC only changes when commits are flushed, not their
        // contents; the risk is the documented crash window above.
        unsafe { self.env.set_flags(EnvFlags::NO_SYNC, mode)? };
        if sync {
            self.env.force_sync()?;
        }
        Ok(())
    }

    /// Opens a store that keeps vectors in f64.
    pub fn open(path: &Path, map_size_mb: usize) -> Result<Self> {
        Self::open_with_precision(path, map_size_mb, Precision::F64)
    }

    /// Opens a store that keeps vectors at `precision`. A store must always be
    /// opened with the precision it was created with (`db.toml` records it).
    pub fn open_with_precision(
        path: &Path,
        map_size_mb: usize,
        precision: Precision,
    ) -> Result<Self> {
        std::fs::create_dir_all(path)
            .map_err(|e| HeatherError::Storage(format!("failed to create dir: {e}")))?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(1024 * 1024 * map_size_mb)
                .max_dbs(6)
                .open(path)?
        };

        let mut wtxn = env.write_txn()?;
        let registry_db = env.create_database(&mut wtxn, Some("_registry"))?;
        let locations_db = env.create_database(&mut wtxn, Some("_locations"))?;
        let metadata_db = env.create_database(&mut wtxn, Some("_metadata"))?;
        let documents_db = env.create_database(&mut wtxn, Some("_documents"))?;
        let doc_index_db = env.create_database(&mut wtxn, Some("_doc_index"))?;
        let audit_db = env.create_database(&mut wtxn, Some("_audit"))?;
        wtxn.commit()?;

        Ok(Store {
            env,
            precision,
            registry_db,
            locations_db,
            metadata_db,
            documents_db,
            doc_index_db,
            audit_db,
        })
    }

    // --- Transaction primitives ---

    pub fn write_txn(&self) -> Result<RwTxn<'_>> {
        Ok(self.env.write_txn()?)
    }

    pub fn read_txn(&self) -> Result<RoTxn<'_>> {
        Ok(self.env.read_txn()?)
    }

    // --- Collection registry ---

    pub fn create_collection(&self, txn: &mut RwTxn, name: &str) -> Result<u32> {
        // Check if already exists
        if let Some(existing) = self.get_collection_id_txn(txn, name)? {
            return Ok(existing);
        }

        let next_id = self.load_next_collection_id_txn(txn)?;
        let id_bytes = bincode::serialize(&next_id)?;
        self.registry_db.put(txn, name.as_bytes(), &id_bytes)?;

        // Increment counter
        let new_next = next_id + 1;
        let next_bytes = bincode::serialize(&new_next)?;
        self.registry_db
            .put(txn, NEXT_COLLECTION_ID_KEY.as_bytes(), &next_bytes)?;

        Ok(next_id)
    }

    pub fn get_collection_id(&self, name: &str) -> Result<Option<u32>> {
        let rtxn = self.env.read_txn()?;
        self.get_collection_id_txn(&rtxn, name)
    }

    fn get_collection_id_txn(&self, txn: &RoTxn, name: &str) -> Result<Option<u32>> {
        match self.registry_db.get(txn, name.as_bytes())? {
            Some(data) => Ok(Some(bincode::deserialize(data)?)),
            None => Ok(None),
        }
    }

    pub fn list_collections(&self) -> Result<Vec<(String, u32)>> {
        let rtxn = self.env.read_txn()?;
        let mut collections = Vec::new();
        let iter = self.registry_db.iter(&rtxn)?;
        for result in iter {
            let (key, value) = result?;
            let name = std::str::from_utf8(key).map_err(|e| {
                HeatherError::Storage(format!("invalid UTF-8 in registry key: {e}"))
            })?;
            if name == NEXT_COLLECTION_ID_KEY {
                continue;
            }
            let id: u32 = bincode::deserialize(value)?;
            collections.push((name.to_string(), id));
        }
        Ok(collections)
    }

    pub fn delete_collection(&self, txn: &mut RwTxn, name: &str, collection_id: u32) -> Result<()> {
        // Remove from registry
        self.registry_db.delete(txn, name.as_bytes())?;

        // Remove all locations with this collection's prefix
        let prefix = collection_prefix(collection_id);
        let iter = self.locations_db.prefix_iter(txn, &prefix)?;
        let keys_to_delete: Vec<Vec<u8>> = iter
            .map(|r| r.map(|(k, _)| k.to_vec()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for key in &keys_to_delete {
            self.locations_db.delete(txn, key)?;
        }

        // Remove all doc index entries with this collection's prefix
        let iter = self.doc_index_db.prefix_iter(txn, &prefix)?;
        let keys_to_delete: Vec<Vec<u8>> = iter
            .map(|r| r.map(|(k, _)| k.to_vec()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for key in &keys_to_delete {
            self.doc_index_db.delete(txn, key)?;
        }

        // Remove all documents with this collection's prefix
        let iter = self.documents_db.prefix_iter(txn, &prefix)?;
        let keys_to_delete: Vec<Vec<u8>> = iter
            .map(|r| r.map(|(k, _)| k.to_vec()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for key in &keys_to_delete {
            self.documents_db.delete(txn, key)?;
        }

        // Remove all metadata with this collection's prefix
        let iter = self.metadata_db.prefix_iter(txn, &prefix)?;
        let keys_to_delete: Vec<Vec<u8>> = iter
            .map(|r| r.map(|(k, _)| k.to_vec()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for key in &keys_to_delete {
            self.metadata_db.delete(txn, key)?;
        }

        Ok(())
    }

    fn load_next_collection_id_txn(&self, txn: &RoTxn) -> Result<u32> {
        match self
            .registry_db
            .get(txn, NEXT_COLLECTION_ID_KEY.as_bytes())?
        {
            Some(data) => Ok(bincode::deserialize(data)?),
            None => Ok(0),
        }
    }

    // --- Location operations (collection-scoped) ---

    pub fn put_location(
        &self,
        txn: &mut RwTxn,
        collection_id: u32,
        loc: &HardLocation,
    ) -> Result<()> {
        let key = location_key(collection_id, loc.id.0);
        let data = self.encode_location(loc)?;
        self.locations_db.put(txn, &key, &data)?;
        Ok(())
    }

    pub fn delete_location(
        &self,
        txn: &mut RwTxn,
        collection_id: u32,
        id: LocationId,
    ) -> Result<()> {
        let key = location_key(collection_id, id.0);
        self.locations_db.delete(txn, key.as_slice())?;
        Ok(())
    }

    pub fn load_all_locations(&self, collection_id: u32) -> Result<Vec<HardLocation>> {
        let rtxn = self.env.read_txn()?;
        let prefix = collection_prefix(collection_id);
        let mut locations = Vec::new();
        let iter = self.locations_db.prefix_iter(&rtxn, &prefix)?;
        for result in iter {
            let (_key, value) = result?;
            locations.push(self.decode_location(value)?);
        }
        Ok(locations)
    }

    // --- Vector encoding (by precision) ---

    fn encode_location(&self, loc: &HardLocation) -> Result<Vec<u8>> {
        Ok(match self.precision {
            Precision::F64 => bincode::serialize(loc)?,
            Precision::F32 => bincode::serialize(&CompactLocation {
                id: loc.id,
                address: narrow(&loc.address),
                counter: narrow(&loc.counter),
                write_count: loc.write_count,
                neighbors: loc.neighbors.clone(),
                surprise_mass: loc.surprise_mass,
            })?,
        })
    }

    fn decode_location(&self, bytes: &[u8]) -> Result<HardLocation> {
        Ok(match self.precision {
            Precision::F64 => bincode::deserialize(bytes)?,
            Precision::F32 => {
                let c: CompactLocation = bincode::deserialize(bytes)?;
                HardLocation {
                    id: c.id,
                    address: widen(&c.address),
                    counter: widen(&c.counter),
                    write_count: c.write_count,
                    neighbors: c.neighbors,
                    surprise_mass: c.surprise_mass,
                }
            }
        })
    }

    /// A document's stored bytes: its vector (at the store's precision) and
    /// its metadata.
    pub fn encode_document(&self, vector: &[f64], metadata: &[u8]) -> Result<Vec<u8>> {
        Ok(match self.precision {
            Precision::F64 => bincode::serialize(&(vector, metadata))?,
            Precision::F32 => bincode::serialize(&(narrow(vector), metadata))?,
        })
    }

    /// The vector and metadata of a stored document.
    pub fn decode_document(&self, bytes: &[u8]) -> Result<(Vec<f64>, Vec<u8>)> {
        Ok(match self.precision {
            Precision::F64 => bincode::deserialize(bytes)?,
            Precision::F32 => {
                let (vector, metadata): (Vec<f32>, Vec<u8>) = bincode::deserialize(bytes)?;
                (widen(&vector), metadata)
            }
        })
    }

    // --- Metadata operations (collection-scoped) ---

    pub fn put_metadata(
        &self,
        txn: &mut RwTxn,
        collection_id: u32,
        key: &str,
        value: &[u8],
    ) -> Result<()> {
        let composite_key = metadata_key(collection_id, key);
        self.metadata_db.put(txn, &composite_key, value)?;
        Ok(())
    }

    pub fn load_metadata(&self, collection_id: u32, key: &str) -> Result<Option<Vec<u8>>> {
        let rtxn = self.env.read_txn()?;
        let composite_key = metadata_key(collection_id, key);
        let val = self.metadata_db.get(&rtxn, &composite_key)?;
        Ok(val.map(|v| v.to_vec()))
    }

    // --- Document operations (collection-scoped) ---

    pub fn put_document(
        &self,
        txn: &mut RwTxn,
        collection_id: u32,
        write_index: u64,
        data: &[u8],
    ) -> Result<()> {
        let key = document_key(collection_id, write_index);
        self.documents_db.put(txn, &key, data)?;
        Ok(())
    }

    pub fn get_document(&self, collection_id: u32, write_index: u64) -> Result<Option<Vec<u8>>> {
        let rtxn = self.env.read_txn()?;
        let key = document_key(collection_id, write_index);
        let val = self.documents_db.get(&rtxn, &key)?;
        Ok(val.map(|v| v.to_vec()))
    }

    pub fn get_document_txn(
        &self,
        txn: &RoTxn,
        collection_id: u32,
        write_index: u64,
    ) -> Result<Option<Vec<u8>>> {
        let key = document_key(collection_id, write_index);
        let val = self.documents_db.get(txn, &key)?;
        Ok(val.map(|v| v.to_vec()))
    }

    /// One page of a collection's documents in id order, and how many there
    /// are in all. Only the page's values are copied.
    pub fn list_documents_page(
        &self,
        collection_id: u32,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<RawDocument>, usize)> {
        let rtxn = self.env.read_txn()?;
        let prefix = collection_prefix(collection_id);
        let mut docs = Vec::with_capacity(limit.min(1024));
        let mut total = 0;
        for result in self.documents_db.prefix_iter(&rtxn, &prefix)? {
            let (key, value) = result?;
            if total >= offset && docs.len() < limit {
                let idx_bytes: [u8; 8] = key[4..12].try_into().map_err(|_| {
                    HeatherError::Storage("invalid document key length".to_string())
                })?;
                docs.push((u64::from_be_bytes(idx_bytes), value.to_vec()));
            }
            total += 1;
        }
        Ok((docs, total))
    }

    pub fn list_documents(&self, collection_id: u32) -> Result<Vec<(u64, Vec<u8>)>> {
        let rtxn = self.env.read_txn()?;
        let prefix = collection_prefix(collection_id);
        let mut docs = Vec::new();
        let iter = self.documents_db.prefix_iter(&rtxn, &prefix)?;
        for result in iter {
            let (key, value) = result?;
            // Extract write_index from key bytes [4..12]
            let idx_bytes: [u8; 8] = key[4..12]
                .try_into()
                .map_err(|_| HeatherError::Storage("invalid document key length".to_string()))?;
            let write_index = u64::from_be_bytes(idx_bytes);
            docs.push((write_index, value.to_vec()));
        }
        Ok(docs)
    }
    // --- Document index operations (posting lists per hard location) ---

    /// Append a document ID to a hard location's posting list.
    pub fn append_doc_to_location(
        &self,
        txn: &mut RwTxn,
        collection_id: u32,
        location_id: u64,
        doc_id: u64,
    ) -> Result<()> {
        let key = location_key(collection_id, location_id);
        let mut doc_ids: Vec<u64> = match self.doc_index_db.get(txn, &key)? {
            Some(data) => bincode::deserialize(data)?,
            None => Vec::new(),
        };
        doc_ids.push(doc_id);
        let data = bincode::serialize(&doc_ids)?;
        self.doc_index_db.put(txn, &key, &data)?;
        Ok(())
    }

    /// Append several document IDs to a location's posting list, reading and
    /// rewriting the list once.
    pub fn append_docs_to_location(
        &self,
        txn: &mut RwTxn,
        collection_id: u32,
        location_id: u64,
        new_doc_ids: &[u64],
    ) -> Result<()> {
        if new_doc_ids.is_empty() {
            return Ok(());
        }
        let key = location_key(collection_id, location_id);
        let mut doc_ids: Vec<u64> = match self.doc_index_db.get(txn, &key)? {
            Some(data) => bincode::deserialize(data)?,
            None => Vec::new(),
        };
        doc_ids.extend_from_slice(new_doc_ids);
        let data = bincode::serialize(&doc_ids)?;
        self.doc_index_db.put(txn, &key, &data)?;
        Ok(())
    }

    /// Get all document IDs indexed under a hard location.
    pub fn get_doc_ids_for_location(
        &self,
        collection_id: u32,
        location_id: u64,
    ) -> Result<Vec<u64>> {
        let rtxn = self.env.read_txn()?;
        self.get_doc_ids_for_location_txn(&rtxn, collection_id, location_id)
    }

    /// Get all document IDs indexed under a hard location (within an existing txn).
    pub fn get_doc_ids_for_location_txn(
        &self,
        txn: &RoTxn,
        collection_id: u32,
        location_id: u64,
    ) -> Result<Vec<u64>> {
        let key = location_key(collection_id, location_id);
        match self.doc_index_db.get(txn, &key)? {
            Some(data) => Ok(bincode::deserialize(data)?),
            None => Ok(Vec::new()),
        }
    }

    /// Delete a hard location's posting list entry.
    pub fn delete_doc_index_entry(
        &self,
        txn: &mut RwTxn,
        collection_id: u32,
        location_id: u64,
    ) -> Result<()> {
        let key = location_key(collection_id, location_id);
        self.doc_index_db.delete(txn, &key)?;
        Ok(())
    }

    /// Remove a document from the store and from every posting list that
    /// references it. Returns true when the document existed.
    ///
    /// There is no reverse (document -> locations) index, so this scans the
    /// collection's posting lists. That is linear in the number of locations,
    /// which is the right trade for the tombstone path: deletions are driven
    /// by source-repository changes, not by queries, and a reverse index
    /// would tax every write to speed up a rare operation.
    pub fn delete_document_everywhere(
        &self,
        txn: &mut RwTxn,
        collection_id: u32,
        write_index: u64,
    ) -> Result<bool> {
        let key = document_key(collection_id, write_index);
        let existed = self.documents_db.get(txn, &key)?.is_some();
        if existed {
            self.documents_db.delete(txn, &key)?;
        }

        // Collect the posting lists that mention this doc before mutating, so
        // the iteration does not overlap the writes.
        let prefix = collection_prefix(collection_id);
        let mut rewrites: Vec<(Vec<u8>, Vec<u64>)> = Vec::new();
        {
            let iter = self.doc_index_db.prefix_iter(txn, &prefix)?;
            for result in iter {
                let (k, v) = result?;
                let ids: Vec<u64> = bincode::deserialize(v)?;
                if ids.contains(&write_index) {
                    let kept: Vec<u64> = ids.into_iter().filter(|d| *d != write_index).collect();
                    rewrites.push((k.to_vec(), kept));
                }
            }
        }
        for (k, kept) in rewrites {
            if kept.is_empty() {
                self.doc_index_db.delete(txn, k.as_slice())?;
            } else {
                self.doc_index_db
                    .put(txn, k.as_slice(), &bincode::serialize(&kept)?)?;
            }
        }
        Ok(existed)
    }

    /// Set a hard location's full posting list (used during merge migration).
    pub fn put_doc_index_entry(
        &self,
        txn: &mut RwTxn,
        collection_id: u32,
        location_id: u64,
        doc_ids: &[u64],
    ) -> Result<()> {
        let key = location_key(collection_id, location_id);
        let data = bincode::serialize(&doc_ids.to_vec())?;
        self.doc_index_db.put(txn, &key, &data)?;
        Ok(())
    }

    // --- Audit log (database-scoped, not collection-scoped) ---
    //
    // Keys are `[timestamp_ms: 8B BE | seq: 8B BE]` (see `crate::audit`), so
    // the sub-DB's own key order is time order and paging newest-first is a
    // reverse cursor walk rather than a sort.

    /// Append one audit record. Callers batch these into a single txn —
    /// see `crate::audit::AuditLog::flush`.
    pub fn append_audit(&self, txn: &mut RwTxn, key: &[u8], value: &[u8]) -> Result<()> {
        self.audit_db.put(txn, key, value)?;
        Ok(())
    }

    /// Highest sequence number on disk, used to resume the counter at boot.
    pub fn last_audit_seq(&self) -> Result<Option<u64>> {
        let rtxn = self.env.read_txn()?;
        Ok(self
            .audit_db
            .last(&rtxn)?
            .map(|(key, _)| crate::audit::key_seq(key)))
    }

    /// Number of stored audit records.
    pub fn audit_len(&self) -> Result<u64> {
        let rtxn = self.env.read_txn()?;
        Ok(self.audit_db.len(&rtxn)?)
    }

    /// Walk audit records newest-first, calling `visit` with each raw value.
    /// `visit` returns `false` to stop the scan.
    ///
    /// `since_ms` terminates the walk (keys are time-ordered, so everything
    /// past it is older); `until_ms` skips newer rows. `max_scan` bounds the
    /// rows examined so a filter that matches nothing can't read the whole log.
    pub fn scan_audit_desc<F>(
        &self,
        since_ms: Option<u64>,
        until_ms: Option<u64>,
        max_scan: usize,
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(&[u8]) -> bool,
    {
        let rtxn = self.env.read_txn()?;
        let mut scanned = 0usize;
        for result in self.audit_db.rev_iter(&rtxn)? {
            let (key, value) = result?;
            scanned += 1;
            if scanned > max_scan {
                break;
            }
            // `is_some_and` rather than nested `if let` — the collapsed form
            // is a let-chain, which the 1.84 MSRV job doesn't accept.
            let ts = crate::audit::key_timestamp(key);
            if until_ms.is_some_and(|until| ts > until) {
                continue;
            }
            if since_ms.is_some_and(|since| ts < since) {
                break;
            }
            if !visit(value) {
                break;
            }
        }
        Ok(())
    }

    /// Enforce the retention bound: drop the oldest records until at most
    /// `max_entries` remain, then drop anything older than `min_ts_ms`
    /// (`0` disables the age bound). Returns how many were removed.
    pub fn prune_audit(&self, txn: &mut RwTxn, max_entries: u64, min_ts_ms: u64) -> Result<usize> {
        let total = self.audit_db.len(txn)?;
        let mut excess = total.saturating_sub(max_entries);
        if excess == 0 && min_ts_ms == 0 {
            return Ok(0);
        }

        let mut doomed: Vec<Vec<u8>> = Vec::new();
        {
            for result in self.audit_db.iter(txn)? {
                let (key, _) = result?;
                if excess > 0 {
                    doomed.push(key.to_vec());
                    excess -= 1;
                    continue;
                }
                if min_ts_ms > 0 && crate::audit::key_timestamp(key) < min_ts_ms {
                    doomed.push(key.to_vec());
                    continue;
                }
                // Keys ascend by timestamp — the first survivor ends the walk.
                break;
            }
        }
        for key in &doomed {
            self.audit_db.delete(txn, key)?;
        }
        Ok(doomed.len())
    }

    /// Close the LMDB environment and block until the OS has released it.
    ///
    /// `Drop` alone is not enough for a caller that is about to move or
    /// delete the environment's directory: Windows refuses to rename a
    /// directory while a mapping into it is still open, and the release is
    /// asynchronous. POSIX permits the rename regardless, so this is a
    /// no-op there in practice.
    pub fn close_and_wait(self) {
        let Store { env, .. } = self;
        env.prepare_for_closing().wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_store_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path(), 256).unwrap();

        let collection_id = {
            let mut txn = store.write_txn().unwrap();
            let id = store.create_collection(&mut txn, "test").unwrap();
            txn.commit().unwrap();
            id
        };

        let loc = HardLocation::new(LocationId(1), vec![1.0, 0.0, 0.0]);
        let mut txn = store.write_txn().unwrap();
        store.put_location(&mut txn, collection_id, &loc).unwrap();
        txn.commit().unwrap();

        let loaded = store.load_all_locations(collection_id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, LocationId(1));
        assert_eq!(loaded[0].address, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_store_delete() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path(), 256).unwrap();

        let collection_id = {
            let mut txn = store.write_txn().unwrap();
            let id = store.create_collection(&mut txn, "test").unwrap();
            txn.commit().unwrap();
            id
        };

        let loc = HardLocation::new(LocationId(1), vec![1.0, 0.0]);
        let mut txn = store.write_txn().unwrap();
        store.put_location(&mut txn, collection_id, &loc).unwrap();
        txn.commit().unwrap();

        let mut txn = store.write_txn().unwrap();
        store
            .delete_location(&mut txn, collection_id, LocationId(1))
            .unwrap();
        txn.commit().unwrap();

        let loaded = store.load_all_locations(collection_id).unwrap();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_metadata_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path(), 256).unwrap();

        let collection_id = {
            let mut txn = store.write_txn().unwrap();
            let id = store.create_collection(&mut txn, "test").unwrap();
            txn.commit().unwrap();
            id
        };

        let mut txn = store.write_txn().unwrap();
        store
            .put_metadata(&mut txn, collection_id, "next_id", &42u64.to_le_bytes())
            .unwrap();
        txn.commit().unwrap();

        let val = store
            .load_metadata(collection_id, "next_id")
            .unwrap()
            .unwrap();
        assert_eq!(u64::from_le_bytes(val.try_into().unwrap()), 42);
    }

    #[test]
    fn test_atomic_multi_write() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path(), 256).unwrap();

        let collection_id = {
            let mut txn = store.write_txn().unwrap();
            let id = store.create_collection(&mut txn, "test").unwrap();
            txn.commit().unwrap();
            id
        };

        let loc1 = HardLocation::new(LocationId(1), vec![1.0, 0.0]);
        let loc2 = HardLocation::new(LocationId(2), vec![0.0, 1.0]);

        let mut txn = store.write_txn().unwrap();
        store.put_location(&mut txn, collection_id, &loc1).unwrap();
        store.put_location(&mut txn, collection_id, &loc2).unwrap();
        store
            .put_metadata(&mut txn, collection_id, "count", &2u64.to_le_bytes())
            .unwrap();
        txn.commit().unwrap();

        let loaded = store.load_all_locations(collection_id).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_collection_registry() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path(), 256).unwrap();

        let mut txn = store.write_txn().unwrap();
        let id1 = store.create_collection(&mut txn, "users").unwrap();
        let id2 = store.create_collection(&mut txn, "products").unwrap();
        txn.commit().unwrap();

        assert_ne!(id1, id2);

        // Idempotent creation
        let mut txn = store.write_txn().unwrap();
        let id1_again = store.create_collection(&mut txn, "users").unwrap();
        txn.commit().unwrap();
        assert_eq!(id1, id1_again);

        // Lookup
        assert_eq!(store.get_collection_id("users").unwrap(), Some(id1));
        assert_eq!(store.get_collection_id("products").unwrap(), Some(id2));
        assert_eq!(store.get_collection_id("nonexistent").unwrap(), None);

        // List
        let collections = store.list_collections().unwrap();
        assert_eq!(collections.len(), 2);
    }

    #[test]
    fn test_collection_isolation() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path(), 256).unwrap();

        let mut txn = store.write_txn().unwrap();
        let col1 = store.create_collection(&mut txn, "a").unwrap();
        let col2 = store.create_collection(&mut txn, "b").unwrap();
        txn.commit().unwrap();

        // Write to col1
        let loc = HardLocation::new(LocationId(1), vec![1.0, 0.0]);
        let mut txn = store.write_txn().unwrap();
        store.put_location(&mut txn, col1, &loc).unwrap();
        txn.commit().unwrap();

        // col1 has 1 location, col2 has 0
        assert_eq!(store.load_all_locations(col1).unwrap().len(), 1);
        assert_eq!(store.load_all_locations(col2).unwrap().len(), 0);
    }

    #[test]
    fn test_delete_collection() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path(), 256).unwrap();

        let mut txn = store.write_txn().unwrap();
        let col_id = store.create_collection(&mut txn, "doomed").unwrap();
        txn.commit().unwrap();

        // Add data
        let loc = HardLocation::new(LocationId(1), vec![1.0, 0.0]);
        let mut txn = store.write_txn().unwrap();
        store.put_location(&mut txn, col_id, &loc).unwrap();
        store
            .put_metadata(&mut txn, col_id, "next_id", &1u64.to_le_bytes())
            .unwrap();
        txn.commit().unwrap();

        // Delete
        let mut txn = store.write_txn().unwrap();
        store.delete_collection(&mut txn, "doomed", col_id).unwrap();
        txn.commit().unwrap();

        assert_eq!(store.get_collection_id("doomed").unwrap(), None);
        assert_eq!(store.load_all_locations(col_id).unwrap().len(), 0);
        assert!(store.load_metadata(col_id, "next_id").unwrap().is_none());
    }
}

#[cfg(test)]
mod sync_tests {
    use super::*;

    /// Syncing is on by default and can be turned off and back on. (LMDB
    /// keeps the flag on the open environment, not in the file, so a new
    /// process syncs again; heed reuses an environment still open in this
    /// process, so that part cannot be shown here.)
    #[test]
    fn sync_can_be_turned_off_and_back_on() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = Store::open(dir.path(), 16).unwrap();
        assert!(store.syncs().unwrap());
        store.set_sync(false).unwrap();
        assert!(!store.syncs().unwrap());
        store.set_sync(true).unwrap();
        assert!(store.syncs().unwrap());
    }
}

#[cfg(test)]
mod precision_tests {
    use super::*;

    /// An f32 store keeps vectors at f32 (rounded once, widened on load) in
    /// about half the bytes; an f64 store keeps them exactly.
    #[test]
    fn f32_stores_round_trip_at_f32_in_half_the_space() {
        let vector: Vec<f64> = (0..256).map(|i| (i as f64).sin() / 3.0).collect();
        let rounded: Vec<f64> = vector.iter().map(|&x| f64::from(x as f32)).collect();

        let dir32 = tempfile::TempDir::new().unwrap();
        let s32 = Store::open_with_precision(dir32.path(), 16, Precision::F32).unwrap();
        let dir64 = tempfile::TempDir::new().unwrap();
        let s64 = Store::open(dir64.path(), 16).unwrap();

        let d32 = s32.encode_document(&vector, b"{}").unwrap();
        let d64 = s64.encode_document(&vector, b"{}").unwrap();
        assert!(
            d32.len() * 10 < d64.len() * 6,
            "{} vs {} bytes",
            d32.len(),
            d64.len()
        );
        assert_eq!(
            s32.decode_document(&d32).unwrap(),
            (rounded.clone(), b"{}".to_vec())
        );
        assert_eq!(
            s64.decode_document(&d64).unwrap(),
            (vector.clone(), b"{}".to_vec())
        );

        let mut loc = HardLocation::new(LocationId(7), vector.clone());
        loc.counter = vector.clone();
        loc.write_count = 2.5;
        loc.neighbors = vec![1, 2];
        let mut txn = s32.write_txn().unwrap();
        s32.put_location(&mut txn, 0, &loc).unwrap();
        txn.commit().unwrap();
        let back = s32.load_all_locations(0).unwrap().pop().unwrap();
        assert_eq!(
            (back.id, back.write_count, back.neighbors.clone()),
            (LocationId(7), 2.5, vec![1, 2])
        );
        assert_eq!(back.address, rounded);
        assert_eq!(back.counter, rounded);
    }
}

use std::path::Path;

use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions, RoTxn, RwTxn};

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
    registry_db: Database<Bytes, Bytes>,
    locations_db: Database<Bytes, Bytes>,
    metadata_db: Database<Bytes, Bytes>,
    documents_db: Database<Bytes, Bytes>,
    doc_index_db: Database<Bytes, Bytes>,
}

impl Store {
    pub fn open(path: &Path, map_size_mb: usize) -> Result<Self> {
        std::fs::create_dir_all(path)
            .map_err(|e| HeatherError::Storage(format!("failed to create dir: {e}")))?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(1024 * 1024 * map_size_mb)
                .max_dbs(5)
                .open(path)?
        };

        let mut wtxn = env.write_txn()?;
        let registry_db = env.create_database(&mut wtxn, Some("_registry"))?;
        let locations_db = env.create_database(&mut wtxn, Some("_locations"))?;
        let metadata_db = env.create_database(&mut wtxn, Some("_metadata"))?;
        let documents_db = env.create_database(&mut wtxn, Some("_documents"))?;
        let doc_index_db = env.create_database(&mut wtxn, Some("_doc_index"))?;
        wtxn.commit()?;

        Ok(Store {
            env,
            registry_db,
            locations_db,
            metadata_db,
            documents_db,
            doc_index_db,
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
        let data = bincode::serialize(loc)?;
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
            let loc: HardLocation = bincode::deserialize(value)?;
            locations.push(loc);
        }
        Ok(locations)
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

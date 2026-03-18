use heather_db::merge::knn_merge;

use crate::snapshot::EAMSnapshot;

/// Consolidate a snapshot after an algebraic operation:
/// 1. Run knn_merge to merge locations above tau_merge.
/// 2. Re-index IDs to be contiguous.
pub fn consolidate(snap: &mut EAMSnapshot) {
    // Step 1: Merge similar locations
    knn_merge(&mut snap.locations, &snap.config);

    // Step 2: Re-index
    snap.reindex();
}

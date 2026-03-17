use heather_db::merge::knn_merge;

use crate::snapshot::EAMSnapshot;

/// Consolidate a snapshot after an algebraic operation:
/// 1. Run knn_merge to merge locations above tau_merge.
/// 2. If still above l_max, prune lowest-write-count locations.
/// 3. Re-index IDs to be contiguous.
pub fn consolidate(snap: &mut EAMSnapshot) {
    // Step 1: Merge similar locations
    knn_merge(&mut snap.locations, &snap.config);

    // Step 2: Enforce l_max by pruning least-written locations
    if snap.locations.len() > snap.config.l_max {
        snap.locations.sort_by(|a, b| {
            b.write_count
                .partial_cmp(&a.write_count)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        snap.locations.truncate(snap.config.l_max);
    }

    // Step 3: Re-index
    snap.reindex();
}

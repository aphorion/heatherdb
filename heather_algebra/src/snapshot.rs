use heather_db::{Collection, EAMConfig, HardLocation, LocationId};

use crate::error::{AlgebraError, Result};

/// A persistence-decoupled snapshot of an EAM's state.
/// All algebraic operations consume and produce EAMSnapshots.
#[derive(Debug, Clone)]
pub struct EAMSnapshot {
    pub locations: Vec<HardLocation>,
    pub config: EAMConfig,
}

impl EAMSnapshot {
    /// Create a snapshot from raw parts, validating dimensions.
    pub fn new(locations: Vec<HardLocation>, config: EAMConfig) -> Result<Self> {
        for loc in &locations {
            if loc.address.len() != config.d || loc.counter.len() != config.d {
                return Err(AlgebraError::DimensionMismatch {
                    left: config.d,
                    right: loc.address.len(),
                });
            }
        }
        Ok(EAMSnapshot { locations, config })
    }

    /// Extract a snapshot from a live Collection.
    pub fn from_collection(col: &Collection) -> Result<Self> {
        let (locations, config) = col.snapshot().map_err(AlgebraError::Db)?;
        Ok(EAMSnapshot { locations, config })
    }

    /// Load this snapshot into a live Collection, replacing its state.
    pub fn into_collection(self, col: &Collection) -> Result<()> {
        col.load_snapshot(self.locations, self.config)
            .map_err(AlgebraError::Db)
    }

    /// Dimensionality of this snapshot.
    pub fn dim(&self) -> usize {
        self.config.d
    }

    /// Number of hard locations.
    pub fn num_locations(&self) -> usize {
        self.locations.len()
    }

    /// Re-assign all location IDs to be contiguous starting from 0.
    pub fn reindex(&mut self) {
        for (i, loc) in self.locations.iter_mut().enumerate() {
            loc.id = LocationId(i as u64);
        }
    }
}

//! heather-fornix: Import/export EAM snapshots between strata (Python) and HeatherDB.
//!
//! The fornix is the nerve fiber bundle that carries memory signals from the
//! hippocampus to long-term storage — just as this tool carries trained EAM
//! state from strata into HeatherDB's persistent LMDB store.
//!
//! # Import (strata JSON → HeatherDB LMDB)
//!
//! ```sh
//! heather-fornix import --file mnist.json --collection mnist --db ./data
//! ```
//!
//! # Export (HeatherDB LMDB → JSON)
//!
//! ```sh
//! heather-fornix export --collection mnist --db ./data --output snapshot.json
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

use heather_db::{EAMConfig, HardLocation, Hive, LocationId};

// ---------------------------------------------------------------------------
// JSON schema — matches strata's bridge.py export format
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct ExportedEAM {
    config: EAMConfig,
    locations: Vec<ExportedLocation>,
    #[serde(default)]
    prototypes: HashMap<String, Vec<f64>>,
    #[serde(default)]
    controller_state: Option<String>,
}

/// Flat location representation for JSON interop.
///
/// We don't reuse `HardLocation` directly because `LocationId` is a newtype
/// and we want the JSON to have plain integer `id` fields for Python compat.
#[derive(Serialize, Deserialize)]
struct ExportedLocation {
    id: u64,
    address: Vec<f64>,
    counter: Vec<f64>,
    write_count: f64,
}

impl From<ExportedLocation> for HardLocation {
    fn from(loc: ExportedLocation) -> Self {
        HardLocation {
            id: LocationId(loc.id),
            address: loc.address,
            counter: loc.counter,
            write_count: loc.write_count,
            neighbors: Vec::new(),
            surprise_mass: 0.0,
        }
    }
}

impl From<&HardLocation> for ExportedLocation {
    fn from(loc: &HardLocation) -> Self {
        ExportedLocation {
            id: loc.id.0,
            address: loc.address.clone(),
            counter: loc.counter.clone(),
            write_count: loc.write_count,
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "heather-fornix", about = "Import/export EAM snapshots")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Import a strata JSON export into a HeatherDB collection.
    Import {
        /// Path to the exported JSON file.
        #[arg(short, long)]
        file: PathBuf,

        /// Collection name to import into (created if absent).
        #[arg(short, long)]
        collection: String,

        /// Path to the HeatherDB data directory.
        #[arg(short, long, default_value = "./data")]
        db: PathBuf,

        /// LMDB map size in MB.
        #[arg(long, default_value_t = 256)]
        map_size: usize,
    },

    /// Export a HeatherDB collection to JSON.
    Export {
        /// Collection name to export.
        #[arg(short, long)]
        collection: String,

        /// Path to the HeatherDB data directory.
        #[arg(short, long, default_value = "./data")]
        db: PathBuf,

        /// Output JSON path.
        #[arg(short, long)]
        output: PathBuf,

        /// LMDB map size in MB.
        #[arg(long, default_value_t = 256)]
        map_size: usize,
    },
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

fn do_import(file: &Path, collection: &str, db: &Path, map_size: usize) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(file)?;
    let exported: ExportedEAM = serde_json::from_str(&raw)?;

    let num_locations = exported.locations.len();
    let dim = exported.config.d;

    // Convert locations
    let locations: Vec<HardLocation> = exported.locations.into_iter().map(Into::into).collect();

    // Open hive — we need a throwaway config for Hive::open, load_snapshot
    // will replace it on the target collection.
    let hive_config = EAMConfig::new(dim)?;
    let hive = Hive::open(db, hive_config, map_size)?;

    let col = hive.get_or_create_collection(collection)?;
    col.load_snapshot(locations, exported.config)?;

    eprintln!("Imported {num_locations} locations (d={dim}) into collection '{collection}'");

    if !exported.prototypes.is_empty() {
        eprintln!(
            "  prototypes: {} classes ({})",
            exported.prototypes.len(),
            exported
                .prototypes
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    if let Some(ctrl) = &exported.controller_state {
        eprintln!("  controller checkpoint hint: {ctrl}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

fn do_export(collection: &str, db: &Path, output: &Path, map_size: usize) -> anyhow::Result<()> {
    // We need a config to open Hive. Use a placeholder — the real config
    // comes from the collection's stored state via snapshot().
    let hive_config = EAMConfig::new(64)?;
    let hive = Hive::open(db, hive_config, map_size)?;

    let col = hive
        .get_collection(collection)?
        .ok_or_else(|| anyhow::anyhow!("collection '{}' not found", collection))?;

    let (locations, config) = col.snapshot()?;
    let num_locations = locations.len();

    let exported = ExportedEAM {
        config,
        locations: locations.iter().map(Into::into).collect(),
        prototypes: HashMap::new(),
        controller_state: None,
    };

    let json = serde_json::to_string_pretty(&exported)?;
    std::fs::write(output, json)?;

    eprintln!(
        "Exported {num_locations} locations from collection '{collection}' to {}",
        output.display()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Import {
            file,
            collection,
            db,
            map_size,
        } => do_import(&file, &collection, &db, map_size),
        Command::Export {
            collection,
            db,
            output,
            map_size,
        } => do_export(&collection, &db, &output, map_size),
    }
}

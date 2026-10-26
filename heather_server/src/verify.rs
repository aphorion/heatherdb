//! `heather verify [--db NAME] [--json]` — an offline integrity check.
//!
//! Decodes every stored location and document of every database (or one)
//! and reports, per collection, how many records are in the current
//! format, how many predate it, and how many are corrupt or in a newer
//! format than this HeatherDB reads, with the first damaged records by key.
//! Databases that are not mounted (a newer format) are reported too. Exits
//! non-zero when anything is damaged or unreadable. Like `backup`, it needs
//! the server stopped.

use std::path::Path;
use std::process::ExitCode;

use heather_db::Server;
use heather_db::verify::CollectionReport;
use serde::Serialize;

#[derive(Serialize)]
struct DatabaseReport {
    database: String,
    /// Why it could not be checked (not mounted), if so.
    #[serde(skip_serializing_if = "Option::is_none")]
    unreadable: Option<String>,
    collections: Vec<CollectionReport>,
}

pub fn run(data_dir: &Path, db: Option<&str>, json: bool) -> ExitCode {
    match verify(data_dir, db) {
        Ok(reports) => {
            let clean = reports
                .iter()
                .all(|r| r.unreadable.is_none() && r.collections.iter().all(|c| c.is_clean()));

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&reports).unwrap_or_default()
                );
            } else {
                print_reports(&reports);
                println!(
                    "{}",
                    if clean {
                        "✓ no damage found"
                    } else {
                        "✗ damaged or unreadable records found"
                    }
                );
            }

            if clean {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("error: verify: {e}");
            ExitCode::FAILURE
        }
    }
}

fn verify(data_dir: &Path, only: Option<&str>) -> Result<Vec<DatabaseReport>, String> {
    crate::backup::refuse_if_engine_running(data_dir)?;
    if !data_dir.join("db").is_dir() {
        return Err(format!("no databases under {}", data_dir.display()));
    }

    let server =
        Server::open(data_dir, 128).map_err(|e| format!("open {}: {e}", data_dir.display()))?;
    let mut names = server.databases().map_err(|e| e.to_string())?;
    names.extend(server.refused_databases().into_iter().map(|(name, _)| name));
    names.sort();
    names.dedup();

    if let Some(only) = only
        && !names.iter().any(|n| n == only)
    {
        return Err(format!("database not found: {only}"));
    }

    names
        .into_iter()
        .filter(|name| only.is_none_or(|o| o == name))
        .map(|name| match server.database(&name) {
            Some(hive) => Ok(DatabaseReport {
                collections: hive.verify().map_err(|e| format!("{name}: {e}"))?,
                unreadable: None,
                database: name,
            }),
            None => Ok(DatabaseReport {
                unreadable: Some(
                    server
                        .refused(&name)
                        .unwrap_or_else(|| "not mounted".to_string()),
                ),
                collections: Vec::new(),
                database: name,
            }),
        })
        .collect()
}

fn print_reports(reports: &[DatabaseReport]) {
    for db in reports {
        match &db.unreadable {
            Some(reason) => println!("{}: NOT CHECKED — {reason}", db.database),
            None => println!("{}:", db.database),
        }
        for c in &db.collections {
            let l = c.locations;
            let d = c.documents;
            println!(
                "  {:<32} locations {} ok ({} legacy), {} corrupt, {} unsupported · documents {} ok ({} legacy), {} corrupt, {} unsupported",
                c.collection,
                l.current + l.legacy,
                l.legacy,
                l.corrupt,
                l.unsupported,
                d.current + d.legacy,
                d.legacy,
                d.corrupt,
                d.unsupported
            );
            for p in &c.problems {
                println!("      {p}");
            }
        }
    }
}

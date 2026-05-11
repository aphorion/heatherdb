//! `heather_server user {create,list,delete,passwd}` — operator commands
//! for managing the user store.
//!
//! These run synchronously and exit, never starting the HTTP server. Useful
//! flow on a fresh VPS:
//!
//!   heather_server user create alice --password 'hunter22!' --scope root
//!   heather_server user passwd admin --password 'newpw!' --confirm 'newpw!'
//!   heather_server user list
//!   heather_server user delete bob
//!
//! All commands operate on `$HEATHER_DATA_DIR/users.json`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Subcommand;

use crate::users::{Scope, UserStore};

#[derive(Subcommand)]
pub enum UserCmd {
    /// Create a new user.
    Create {
        /// Username (`[a-zA-Z][a-zA-Z0-9_.\-]{0,63}`).
        name: String,
        /// Password (>=8 chars). Required; consider piping from a secret store.
        #[arg(long)]
        password: String,
        /// Scope: `root` (or `*`) for full access, otherwise a database name
        /// (e.g. `--scope memoria` for memoria-only access).
        #[arg(long, default_value = "root")]
        scope: String,
    },
    /// List users (name + scope; never prints hashes).
    List,
    /// Delete a user.
    Delete {
        name: String,
    },
    /// Rotate a user's password.
    Passwd {
        name: String,
        #[arg(long)]
        password: String,
    },
}

/// Execute a user-mgmt subcommand. Returns the process exit code.
pub fn run(data_dir: &PathBuf, cmd: &UserCmd) -> ExitCode {
    // Make sure the data dir exists — the user is plausibly running this
    // before the server has ever started.
    if let Err(e) = std::fs::create_dir_all(data_dir) {
        eprintln!("error: create {}: {e}", data_dir.display());
        return ExitCode::FAILURE;
    }

    let store = match UserStore::load(data_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: load user store: {e}");
            return ExitCode::FAILURE;
        }
    };

    match cmd {
        UserCmd::Create { name, password, scope } => {
            let scope = match Scope::parse(scope) {
                Ok(s) => s,
                Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; }
            };
            match store.create(name, password, scope.clone()) {
                Ok(u) => {
                    println!("✓ created user '{}' (scope {})", u.name, scope.label());
                    println!("  → file: {}", store.path().display());
                    ExitCode::SUCCESS
                }
                Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
            }
        }
        UserCmd::List => {
            let users = store.list();
            if users.is_empty() {
                println!("(no users)");
                return ExitCode::SUCCESS;
            }
            println!("{:<24} {:<14} {}", "NAME", "SCOPE", "CREATED");
            for u in users {
                let ts = chrono_format(u.created_at);
                println!("{:<24} {:<14} {}", u.name, u.scope.label(), ts);
            }
            ExitCode::SUCCESS
        }
        UserCmd::Delete { name } => match store.delete(name) {
            Ok(true) => { println!("✓ deleted user '{name}'"); ExitCode::SUCCESS }
            Ok(false) => { eprintln!("error: user not found: {name}"); ExitCode::FAILURE }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
        UserCmd::Passwd { name, password } => match store.set_password(name, password) {
            Ok(()) => { println!("✓ password rotated for '{name}'"); ExitCode::SUCCESS }
            Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
        },
    }
}

/// Tiny YYYY-MM-DD HH:MM:SSZ formatter — avoids pulling in `chrono`. Good
/// enough for `user list` output. Treats input as Unix epoch seconds.
pub(crate) fn chrono_format(epoch_secs: u64) -> String {
    // Days from 1970-01-01 to compute the calendar date (proleptic
    // Gregorian). Ports a small chunk of Howard Hinnant's date algorithms.
    let days = (epoch_secs / 86_400) as i64;
    let (y, m, d) = civil_from_days(days);
    let secs_of_day = epoch_secs % 86_400;
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day / 60) % 60;
    let ss = secs_of_day % 60;
    format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}:{ss:02}Z")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

# Sieve — Near-Duplicate Detection

A practical CLI tool that uses HeatherDB's SDM for near-duplicate detection of text records. Catches paraphrased duplicates that exact matching misses, using EAM's reconstruction fidelity as a natural similarity measure.

## How it works

When you query SDM with a vector, it reconstructs the nearest attractor — the stored pattern it's most "reminded of." The **reconstruction fidelity** (cosine similarity between your input and the reconstruction) reveals whether something similar exists:

| Fidelity | Verdict | Meaning |
|----------|---------|---------|
| > 0.75 | **DUPLICATE** | SDM has a strong attractor nearby — a near-duplicate exists |
| 0.50 - 0.75 | **SIMILAR** | Related content exists but not a clear duplicate |
| < 0.50 | **UNIQUE** | SDM has no strong attractor — this is new |

Unlike exact dedup, Sieve catches:
- "I can't log in" vs "Unable to sign into my account" (paraphrased)
- "Password reset not working" vs "The password reset link is broken" (same intent, different words)

Unlike embeddings + cosine threshold, EAM's basins of attraction **naturally define** what "similar enough" means — patterns that have been written multiple times create stronger attractors.

## Quick start

```bash
# Start HeatherDB
cargo run --release -p heather_server -- \
  --data-dir /tmp/sieve_db \
  --dimension 384 \
  --port 6380

# Setup
cd sample_projects/sieve
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Scan a file for duplicates

```bash
python run.py scan sample_data/support_tickets.jsonl
```

Output:
```
    1.    UNIQUE (first) I can't log into my account, it says invalid password
    2. DUPLICATE (0.82) Unable to sign in to my account, getting an invalid passw
       ^ matches: I can't log into my account, it says invalid... [0.91]
    3. DUPLICATE (0.79) My login isn't working, keeps saying wrong password
       ^ matches: I can't log into my account, it says invalid... [0.87]
    ...

============================================================
  Scan summary:
    Total records:  31
    Duplicates:     14
    Similar:        8
    Unique:         9
============================================================
```

### Ingest + check

```bash
# Load records into EAM
python run.py ingest sample_data/support_tickets.jsonl

# Check new records
python run.py check "I'm unable to log in, my password doesn't work"
python run.py check "How do I integrate with the Stripe API?"
```

### Interactive mode

```bash
python run.py interactive
```

```
check> I can't sign in to my account
  DUPLICATE (fidelity: 0.81)
  I can't sign in to my account
  matches:
    [0.93] I can't log into my account, it says invalid password
    [0.88] Unable to sign in to my account, getting an invalid password error
    [0.82] My login isn't working, keeps saying wrong password

check> How do I set up two-factor authentication?
  UNIQUE (fidelity: 0.34)
  How do I set up two-factor authentication?
```

## Commands

| Command | Description |
|---------|-------------|
| `python run.py ingest <file>` | Load records into EAM |
| `python run.py check "text"` | Check one record for duplicates |
| `python run.py scan <file>` | Find duplicates within a file |
| `python run.py stats` | Show SDM stats and record count |
| `python run.py interactive` | Interactive mode |

### Options

- `--field <name>` — for JSONL/CSV files, specify which field to use (default: "text")

### File formats

- `.jsonl` — one JSON object per line, reads the "text" field
- `.csv` — reads the "text" column (or first column)
- `.txt` — one record per line

## Use cases

- **Support tickets** — detect duplicate customer issues before routing
- **Bug reports** — find existing reports before creating new ones
- **Product listings** — catch duplicate or near-duplicate listings
- **Content moderation** — identify reposted or slightly modified content
- **Resume screening** — detect duplicate applications

## Project structure

```
sieve/
  .env                        # HEATHER_URL
  requirements.txt            # Dependencies
  sieve/
    __init__.py
    client.py                 # HeatherDB HTTP client
    embeddings.py             # sentence-transformers + SQLite cache
    detector.py               # Core: Sieve class with ingest/check/scan
    cli.py                    # CLI subcommands
  run.py                      # Entry point
  sample_data/
    support_tickets.jsonl     # 31 sample tickets with intentional duplicates
```

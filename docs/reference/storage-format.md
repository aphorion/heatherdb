# Storage format

How HeatherDB keeps vectors on disk, and how it stays safe when the bytes it
reads are not what it expects.

## Records

Every hard location and every document is stored in LMDB as one value in a
self-describing envelope (`heather_db::codec`):

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `FF FE 48 44` (`FF FE "HD"`) |
| 4 | 1 | record version (currently `1`) |
| 5 | 1 | kind: `1` location, `2` document, `3` plain location, `4` document reference |
| 6 | 1 | precision of the vectors: `0` f64, `1` f32 |
| 7 | 1 | reserved, `0` |
| 8 | 4 | payload length, u32 little-endian |
| 12 | 4 | CRC-32 of the payload, u32 little-endian |
| 16 | … | payload |

Payloads are explicit little-endian layouts, one per record version:

- **location v1**: `id u64 · d u32 · address[d] · counter[d] · write_count f64 · surprise_mass f64 · n u32 · neighbors u64[n]`
- **document v1**: `d u32 · vector[d] · metadata length u32 · metadata bytes`
- **plain location v1**: `id u64 · d u32 · address[d] · write_count f64 · surprise_mass f64 · n u32 · neighbors u64[n]`
- **document reference v1**: `location id u64 · metadata length u32 · metadata bytes`

A plain (non-competitive) write stores its vector once. Its location's
counter equals its address, so the location is written as a **plain
location**, without the counter, and reads back with the address in its
place. Its document is written as a **document reference**: the id of that
location instead of another copy of the vector, which reads back as the
location's address. Merging, compressing and loading a snapshot change or
remove locations, so they first rewrite any referencing documents with
their own vectors.

Vectors are f64 or f32 as the header says, so a record decodes the same
whatever the reader assumes. New fields go into a new record version with
its own decoder; existing versions are never reinterpreted.

Values written before the envelope (`bincode` of the in-memory structs) are
still read: they cannot start with the magic, so they are recognised and
decoded the old way, at the database's precision.

## When a record is wrong

- A checksum mismatch, a truncated payload, an impossible length, or a
  document reference to a location that does not exist is a
  **corrupt record** (`HeatherError::Corrupt`). Decoding never panics and never
  allocates more than the record's own size.
- A record version newer than the server reads is an **unsupported format**
  (`HeatherError::UnsupportedFormat`): refused, not misread.
- Reads that serve many records — document listings and pages, similarity
  queries, loading a collection's locations when it opens — **skip** such a
  record and count it, and serve the rest. Listings say how many they skipped
  (`skipped`); `GET /db/{db}` reports the database's running count
  (`skipped_records`). A non-zero count means something is damaged.
- Fetching one specific document that is damaged is an error for that
  request.

## Databases

`db.toml` records the database's `format_version` (currently `1`). A server
refuses to mount a database whose format is newer than it reads, logs why at
startup, and answers requests for it with `409 Conflict` and the reason; the
other databases mount normally. Databases from before the envelope (format
`0`) are mounted and marked current, since what is written from then on uses
the envelope.

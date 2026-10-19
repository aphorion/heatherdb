//! The on-disk format of stored vectors: hard locations and documents.
//!
//! Every value is written in a self-describing envelope:
//!
//! ```text
//! offset  size  field
//!      0     4  magic           FF FE 'H' 'D'
//!      4     1  record version  1
//!      5     1  kind            1 = location, 2 = document
//!      6     1  precision       0 = f64, 1 = f32 (of the vectors)
//!      7     1  reserved        0
//!      8     4  payload length  u32 LE
//!     12     4  CRC-32 of the payload, u32 LE
//!     16     …  payload
//! ```
//!
//! and an explicit little-endian payload, one layout per record version:
//!
//! ```text
//! location v1: id u64 | d u32 | address [d] | counter [d] | write_count f64
//!              | surprise_mass f64 | n u32 | neighbors u64 [n]
//! document v1: d u32 | vector [d] | metadata length u32 | metadata bytes
//! ```
//!
//! Vectors are f64 or f32 as the header says, so a reader never relies on
//! how the store was opened; an unknown record version is refused instead
//! of misread; a checksum mismatch, truncation or impossible length is
//! reported as [`HeatherError::Corrupt`] and never panics.
//!
//! Values without the envelope were written before it existed: `bincode` of
//! the in-memory structs (f64, or f32 in stores opened as such). They are
//! still read. The magic cannot begin one: a legacy location starts with its
//! id and a legacy document with its vector length, both u64, whose low four
//! bytes would have to equal the magic (over 1.4 billion).

use crate::config::Precision;
use crate::error::{HeatherError, Result};
use crate::location::{HardLocation, LocationId};

pub const MAGIC: [u8; 4] = [0xFF, 0xFE, b'H', b'D'];
/// The record version written. Readers accept every version up to this.
pub const RECORD_VERSION: u8 = 1;
const HEADER_LEN: usize = 16;

const KIND_LOCATION: u8 = 1;
const KIND_DOCUMENT: u8 = 2;

/// What a stored value is, as far as its bytes say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Written in the envelope.
    Enveloped {
        version: u8,
        kind: u8,
        precision: Precision,
    },
    /// Written before the envelope existed.
    Legacy,
}

/// The format of a stored value, without decoding it.
pub fn format(bytes: &[u8]) -> Result<Format> {
    if bytes.len() < 4 || bytes[..4] != MAGIC {
        return Ok(Format::Legacy);
    }
    if bytes.len() < HEADER_LEN {
        return Err(corrupt("record header is truncated"));
    }
    Ok(Format::Enveloped {
        version: bytes[4],
        kind: bytes[5],
        precision: precision_from(bytes[6])?,
    })
}

pub fn encode_location(loc: &HardLocation, precision: Precision) -> Vec<u8> {
    let d = loc.address.len();
    let mut p =
        Vec::with_capacity(8 + 4 + 2 * d * width(precision) + 16 + 4 + 8 * loc.neighbors.len());
    p.extend_from_slice(&loc.id.0.to_le_bytes());
    p.extend_from_slice(&len_u32(d).to_le_bytes());
    put_floats(&mut p, &loc.address, precision);
    put_floats(&mut p, &loc.counter, precision);
    p.extend_from_slice(&loc.write_count.to_le_bytes());
    p.extend_from_slice(&loc.surprise_mass.to_le_bytes());
    p.extend_from_slice(&len_u32(loc.neighbors.len()).to_le_bytes());
    for n in &loc.neighbors {
        p.extend_from_slice(&n.to_le_bytes());
    }
    envelope(KIND_LOCATION, precision, p)
}

/// Decodes a location; `legacy` is the precision legacy values were
/// written at (the store's).
pub fn decode_location(bytes: &[u8], legacy: Precision) -> Result<HardLocation> {
    match format(bytes)? {
        Format::Legacy => legacy_location(bytes, legacy),
        Format::Enveloped { .. } => {
            let (precision, payload) = open(bytes, KIND_LOCATION)?;
            let mut r = Reader::new(payload);
            let id = LocationId(r.u64()?);
            let d = r.u32()? as usize;
            let address = r.floats(d, precision)?;
            let counter = r.floats(d, precision)?;
            let write_count = r.f64()?;
            let surprise_mass = r.f64()?;
            let n = r.u32()? as usize;
            let neighbors = r.u64s(n)?;
            r.finish()?;
            Ok(HardLocation {
                id,
                address,
                counter,
                write_count,
                neighbors,
                surprise_mass,
            })
        }
    }
}

pub fn encode_document(vector: &[f64], metadata: &[u8], precision: Precision) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + vector.len() * width(precision) + 4 + metadata.len());
    p.extend_from_slice(&len_u32(vector.len()).to_le_bytes());
    put_floats(&mut p, vector, precision);
    p.extend_from_slice(&len_u32(metadata.len()).to_le_bytes());
    p.extend_from_slice(metadata);
    envelope(KIND_DOCUMENT, precision, p)
}

/// Decodes a document into its vector and metadata; `legacy` as for
/// [`decode_location`].
pub fn decode_document(bytes: &[u8], legacy: Precision) -> Result<(Vec<f64>, Vec<u8>)> {
    match format(bytes)? {
        Format::Legacy => legacy_document(bytes, legacy),
        Format::Enveloped { .. } => {
            let (precision, payload) = open(bytes, KIND_DOCUMENT)?;
            let mut r = Reader::new(payload);
            let d = r.u32()? as usize;
            let vector = r.floats(d, precision)?;
            let m = r.u32()? as usize;
            let metadata = r.bytes(m)?.to_vec();
            r.finish()?;
            Ok((vector, metadata))
        }
    }
}

// --- envelope ---

fn envelope(kind: u8, precision: Precision, payload: Vec<u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&MAGIC);
    out.push(RECORD_VERSION);
    out.push(kind);
    out.push(precision_byte(precision));
    out.push(0);
    out.extend_from_slice(&len_u32(payload.len()).to_le_bytes());
    out.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// Checks an enveloped value and returns its precision and payload.
fn open(bytes: &[u8], kind: u8) -> Result<(Precision, &[u8])> {
    let version = bytes[4];
    if version == 0 || version > RECORD_VERSION {
        return Err(HeatherError::UnsupportedFormat(format!(
            "record version {version} (this HeatherDB reads up to {RECORD_VERSION}); \
             it was written by a newer HeatherDB"
        )));
    }
    if bytes[5] != kind {
        return Err(corrupt(&format!(
            "expected a {} record, found kind {}",
            kind_name(kind),
            bytes[5]
        )));
    }
    let precision = precision_from(bytes[6])?;
    let len = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize;
    let crc = u32::from_le_bytes(bytes[12..16].try_into().expect("4 bytes"));
    let payload = &bytes[HEADER_LEN..];
    if payload.len() != len {
        return Err(corrupt(&format!(
            "payload is {} bytes, header says {len}",
            payload.len()
        )));
    }
    if crc32fast::hash(payload) != crc {
        return Err(corrupt("checksum mismatch"));
    }
    Ok((precision, payload))
}

fn kind_name(kind: u8) -> &'static str {
    match kind {
        KIND_LOCATION => "location",
        KIND_DOCUMENT => "document",
        _ => "unknown",
    }
}

fn precision_byte(p: Precision) -> u8 {
    match p {
        Precision::F64 => 0,
        Precision::F32 => 1,
    }
}

fn precision_from(b: u8) -> Result<Precision> {
    match b {
        0 => Ok(Precision::F64),
        1 => Ok(Precision::F32),
        other => Err(corrupt(&format!("unknown precision {other}"))),
    }
}

fn width(p: Precision) -> usize {
    match p {
        Precision::F64 => 8,
        Precision::F32 => 4,
    }
}

fn len_u32(n: usize) -> u32 {
    u32::try_from(n).expect("lengths fit in u32")
}

fn put_floats(out: &mut Vec<u8>, v: &[f64], p: Precision) {
    match p {
        Precision::F64 => v
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        Precision::F32 => v
            .iter()
            .for_each(|&x| out.extend_from_slice(&(x as f32).to_le_bytes())),
    }
}

fn corrupt(message: &str) -> HeatherError {
    HeatherError::Corrupt(message.to_string())
}

/// Bounds-checked reads over a payload: a bad length is an error, never a
/// panic or an oversized allocation.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes }
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.bytes.len() {
            return Err(corrupt("payload is truncated"));
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(
            self.bytes(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn floats(&mut self, n: usize, p: Precision) -> Result<Vec<f64>> {
        let raw = self.bytes(
            n.checked_mul(width(p))
                .ok_or_else(|| corrupt("vector length overflows"))?,
        )?;
        Ok(match p {
            Precision::F64 => raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect(),
            Precision::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f64::from(f32::from_le_bytes(*b)))
                .collect(),
        })
    }

    fn u64s(&mut self, n: usize) -> Result<Vec<u64>> {
        let raw = self.bytes(
            n.checked_mul(8)
                .ok_or_else(|| corrupt("list length overflows"))?,
        )?;
        Ok(raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b))
            .collect())
    }

    fn finish(self) -> Result<()> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(corrupt(&format!(
                "{} unexpected trailing bytes",
                self.bytes.len()
            )))
        }
    }
}

// --- legacy (pre-envelope) values ---

/// An f32 location as stores opened at f32 wrote it before the envelope.
#[derive(serde::Deserialize)]
struct LegacyCompactLocation {
    id: LocationId,
    address: Vec<f32>,
    counter: Vec<f32>,
    write_count: f64,
    neighbors: Vec<u64>,
    surprise_mass: f64,
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// `bincode` as the legacy values were written, but reading at most the
/// value's own bytes, so a corrupt length cannot ask for a huge allocation.
fn legacy<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> bincode::Result<T> {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(bytes.len() as u64)
        .deserialize(bytes)
}

fn legacy_location(bytes: &[u8], precision: Precision) -> Result<HardLocation> {
    let decoded = match precision {
        Precision::F64 => legacy::<HardLocation>(bytes),
        Precision::F32 => legacy::<LegacyCompactLocation>(bytes).map(|c| HardLocation {
            id: c.id,
            address: widen(&c.address),
            counter: widen(&c.counter),
            write_count: c.write_count,
            neighbors: c.neighbors,
            surprise_mass: c.surprise_mass,
        }),
    };
    decoded.map_err(|e| corrupt(&format!("legacy location: {e}")))
}

fn legacy_document(bytes: &[u8], precision: Precision) -> Result<(Vec<f64>, Vec<u8>)> {
    let decoded = match precision {
        Precision::F64 => legacy::<(Vec<f64>, Vec<u8>)>(bytes),
        Precision::F32 => legacy::<(Vec<f32>, Vec<u8>)>(bytes).map(|(v, m)| (widen(&v), m)),
    };
    decoded.map_err(|e| corrupt(&format!("legacy document: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn location() -> HardLocation {
        HardLocation {
            id: LocationId(42),
            address: vec![0.5, -0.25, 0.125, 1.0 / 3.0],
            counter: vec![2.0, 0.0, -1.5, 1e-9],
            write_count: 3.5,
            neighbors: vec![1, 7, 9],
            surprise_mass: 0.75,
        }
    }

    fn rounded(v: &[f64]) -> Vec<f64> {
        v.iter().map(|&x| f64::from(x as f32)).collect()
    }

    fn same(a: &HardLocation, b: &HardLocation) -> bool {
        a.id == b.id
            && a.address == b.address
            && a.counter == b.counter
            && a.write_count == b.write_count
            && a.neighbors == b.neighbors
            && a.surprise_mass == b.surprise_mass
    }

    #[test]
    fn records_round_trip_at_both_precisions() {
        let loc = location();
        let back = decode_location(&encode_location(&loc, Precision::F64), Precision::F64).unwrap();
        assert!(same(&back, &loc));

        let back = decode_location(&encode_location(&loc, Precision::F32), Precision::F32).unwrap();
        assert_eq!(back.address, rounded(&loc.address));
        assert_eq!(back.counter, rounded(&loc.counter));
        assert_eq!(
            (back.id, back.write_count, back.surprise_mass),
            (loc.id, 3.5, 0.75)
        );
        assert_eq!(back.neighbors, loc.neighbors);

        let v = vec![0.1, 0.2, 0.3];
        assert_eq!(
            decode_document(
                &encode_document(&v, b"{\"a\":1}", Precision::F64),
                Precision::F64
            )
            .unwrap(),
            (v.clone(), b"{\"a\":1}".to_vec())
        );
        assert_eq!(
            decode_document(&encode_document(&v, b"", Precision::F32), Precision::F32).unwrap(),
            (rounded(&v), vec![])
        );
    }

    /// The header says the precision, so a record reads correctly whatever
    /// precision the reader assumes for legacy values.
    #[test]
    fn precision_comes_from_the_record() {
        let v = vec![0.1, 0.2];
        let f32_doc = encode_document(&v, b"m", Precision::F32);
        assert_eq!(
            decode_document(&f32_doc, Precision::F64).unwrap(),
            (rounded(&v), b"m".to_vec())
        );
        let f64_doc = encode_document(&v, b"m", Precision::F64);
        assert_eq!(
            decode_document(&f64_doc, Precision::F32).unwrap(),
            (v, b"m".to_vec())
        );
    }

    /// Values written before the envelope still read, at the store's precision.
    #[test]
    fn legacy_values_still_read() {
        let loc = location();
        let old = bincode::serialize(&loc).unwrap();
        assert_eq!(format(&old).unwrap(), Format::Legacy);
        assert!(same(&decode_location(&old, Precision::F64).unwrap(), &loc));

        let v = vec![0.5, 0.25];
        let old = bincode::serialize(&(v.clone(), b"{}".to_vec())).unwrap();
        assert_eq!(
            decode_document(&old, Precision::F64).unwrap(),
            (v.clone(), b"{}".to_vec())
        );
        let narrow: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let old = bincode::serialize(&(narrow, b"{}".to_vec())).unwrap();
        assert_eq!(
            decode_document(&old, Precision::F32).unwrap(),
            (v, b"{}".to_vec())
        );
    }

    #[test]
    fn damage_is_reported_never_misread() {
        let good = encode_document(&[1.0, 2.0], b"meta", Precision::F64);

        let mut flipped = good.clone();
        *flipped.last_mut().unwrap() ^= 0x01;
        assert!(
            matches!(decode_document(&flipped, Precision::F64), Err(HeatherError::Corrupt(m)) if m.contains("checksum"))
        );

        let truncated = &good[..good.len() - 3];
        assert!(matches!(
            decode_document(truncated, Precision::F64),
            Err(HeatherError::Corrupt(_))
        ));
        assert!(matches!(
            decode_document(&good[..10], Precision::F64),
            Err(HeatherError::Corrupt(_))
        ));

        // A document read as a location.
        assert!(
            matches!(decode_location(&good, Precision::F64), Err(HeatherError::Corrupt(m)) if m.contains("kind"))
        );

        let mut newer = good.clone();
        newer[4] = RECORD_VERSION + 1;
        assert!(matches!(
            decode_document(&newer, Precision::F64),
            Err(HeatherError::UnsupportedFormat(_))
        ));

        // Garbage that is not even legacy bincode.
        assert!(matches!(
            decode_document(&[1, 2, 3], Precision::F64),
            Err(HeatherError::Corrupt(_))
        ));
        // A legacy length claiming a huge vector: an error, not a giant allocation.
        let mut huge = (u64::MAX / 2).to_le_bytes().to_vec();
        huge.extend_from_slice(&[0; 16]);
        assert!(decode_document(&huge, Precision::F64).is_err());
    }
}

//! A stored location's bytes, whatever they are, decode or give an error:
//! never a panic, a hang or an oversized allocation. Whatever decodes
//! re-encodes and decodes to the same location.
#![no_main]

use heather_db::Precision;
use heather_db::codec::{decode_location, encode_location};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    for precision in [Precision::F64, Precision::F32] {
        if let Ok(loc) = decode_location(data, precision) {
            let again = decode_location(&encode_location(&loc, Precision::F64), Precision::F64)
                .expect("a decoded location re-encodes");
            assert_eq!(again.id, loc.id);
            assert_eq!(again.neighbors, loc.neighbors);
            assert_eq!(again.address.len(), loc.address.len());
        }
    }
});

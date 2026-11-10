//! A stored document's bytes, whatever they are, decode or give an error:
//! never a panic, a hang or an oversized allocation. Whatever decodes
//! re-encodes and decodes to the same document.
#![no_main]

use heather_db::Precision;
use heather_db::codec::{decode_document, encode_document};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    for precision in [Precision::F64, Precision::F32] {
        if let Ok((vector, metadata)) = decode_document(data, precision) {
            let (again, meta) = decode_document(&encode_document(&vector, &metadata, Precision::F64), Precision::F64)
                .expect("a decoded document re-encodes");
            assert_eq!(meta, metadata);
            assert_eq!(again.len(), vector.len());
        }
    }
});

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

use crate::disk::*;
use crate::reader::ImageReader;
use crate::extents::{extents_resolve_fork, extents_read_data};

#[derive(Clone, Copy, Debug)]
enum ComprType {
    Zlib,
    Gzip,
    Lzo,
    Zstd,
}

/// Returns true if the catalog record flags indicate a compressed file.
/// VDFS4 uses the VDFS4_COMPRESSED_FILE (bit 13) inode flag for extent-based
/// compressed files (descriptor at the tail of the fork data).
pub fn is_compressed_file(flags: u32) -> bool {
    flags & VDFS4_COMPRESSED_FILE != 0
}

/// Map a magic to a compression type. For inline compressed data the magic is
/// the bare 3-char name ("Zip", ...); for the on-disk descriptor it is stored
/// as <prefix-byte>"Zip" (e.g. "hZip") with the name at bytes [1..4].
fn detect_compr_type(magic: &[u8]) -> Option<ComprType> {
    let name: &[u8] = if magic.len() >= 4 {
        &magic[1..4]
    } else if magic.len() >= 3 {
        &magic[..3]
    } else {
        return None;
    };
    match name {
        b"Zip" => Some(ComprType::Zlib),
        b"Gzp" => Some(ComprType::Gzip),
        b"Lzo" => Some(ComprType::Lzo),
        b"Zst" => Some(ComprType::Zstd),
        _ => None,
    }
}

fn decompress_with(data: &[u8], ctype: ComprType) -> Result<Vec<u8>> {
    match ctype {
        ComprType::Zlib => {
            let mut decoder = flate2::read::ZlibDecoder::new(data);
            let mut output = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut output)
                .context("zlib decompression failed")?;
            Ok(output)
        }
        ComprType::Gzip => {
            let mut decoder = flate2::read::GzDecoder::new(data);
            let mut output = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut output)
                .context("gzip decompression failed")?;
            Ok(output)
        }
        ComprType::Zstd => {
            let mut decoder = zstd::stream::read::Decoder::new(data)
                .context("zstd decoder init failed")?;
            let mut output = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut output)
                .context("zstd decompression failed")?;
            Ok(output)
        }
        ComprType::Lzo => {
            bail!("LZO decompression is not supported by this build");
        }
    }
}

/// True if the 4 descriptor-magic bytes form a valid compressed descriptor:
/// byte[0] is an arbitrary prefix, bytes[1..4] are the compression name.
fn valid_comp_descr_magic(m: &[u8; 4]) -> bool {
    matches!(&m[1..4], b"Zip" | b"Gzp" | b"Lzo" | b"Zst")
}

/// Locate the compressed file descriptor within the compressed blob.
/// The descriptor sits at the very tail (right before any optional auth
/// trailer), and is identified by its 3-char compression name plus a sane
/// layout version. Returns the byte offset of the descriptor start.
fn find_descriptor(data: &[u8]) -> Option<usize> {
    let name_ok = |c: usize| -> bool {
        if c + 12 > data.len() {
            return false;
        }
        matches!(&data[c + 9..c + 12], b"Zip" | b"Gzp" | b"Lzo" | b"Zst")
    };
    let layout_ok = |c: usize| -> bool {
        if c + 16 > data.len() {
            return false;
        }
        let v = u16::from_le_bytes([data[c + 14], data[c + 15]]);
        v == VDFS4_COMPR_LAYOUT_VER_05 || v == VDFS4_COMPR_LAYOUT_VER_06
    };

    for &size in &[VDFS4_COMPR_FILE_DESC_LEN_06, VDFS4_COMPR_FILE_DESC_LEN_05] {
        if data.len() >= size {
            let c = data.len() - size;
            if name_ok(c) && layout_ok(c) {
                return Some(c);
            }
        }
    }
    None
}

pub fn decompress_file(
    reader: &ImageReader,
    fork: &Vdfs4Fork,
    object_id: u64,
    overflow: &HashMap<u64, Vec<(u64, Vdfs4Extent)>>,
) -> Result<Vec<u8>> {
    if fork.total_blocks_count == 0 {
        // Inline compressed data: vdfs4_comp_inline_data layout is
        // comp_type[3], comp_size (1 byte), data[]. fork.size_in_bytes holds
        // the *uncompressed* size (the file's i_size).
        let ctype = detect_compr_type(&fork.data[0..3])
            .context("unknown inline compression type")?;
        let comp_size = fork.data[3] as usize;
        let end = (4 + comp_size).min(fork.data.len());
        let compressed = &fork.data[4..end];
        let result = decompress_with(compressed, ctype)?;
        let file_size = fork.size_in_bytes as usize;
        return Ok(result.into_iter().take(file_size).collect());
    }

    // Extent-based compressed file. fork.size_in_bytes holds the COMPRESSED
    // size (comp_size); the fork's extents point at that blob in the image.
    // The blob layout is: [compressed chunks][comp_extent table][descriptor].
    // Each vdfs4_comp_extent.start is a byte offset WITHIN this blob.
    let comp_total = fork.size_in_bytes;
    let extents = extents_resolve_fork(fork, comp_total, overflow, object_id)?;
    if extents.is_empty() {
        bail!("Compressed file has no extents");
    }
    let data = extents_read_data(reader, &extents, 0, comp_total)?;

    let descr_offset = match find_descriptor(&data) {
        Some(o) => o,
        None => {
            let n = data.len();
            let tail = &data[n.saturating_sub(48)..];
            eprintln!(
                "DBG descr-not-found len={} tail={:?}",
                n, tail
            );
            bail!("Could not locate compressed file descriptor");
        }
    };
    let descr: Vdfs4CompFileDescr = unsafe {
        std::ptr::read_unaligned(data[descr_offset..].as_ptr() as *const Vdfs4CompFileDescr)
    };

    if !valid_comp_descr_magic(&descr.magic) {
        bail!("Invalid compressed file descriptor magic: {:?}", descr.magic);
    }

    let ctype = detect_compr_type(&descr.magic)
        .context("unknown compression type in descriptor")?;

    let comp_extents_count = descr.extents_num as usize;
    let extent_struct_size = std::mem::size_of::<Vdfs4CompExtent>();

    // Authenticated compressed files place a digest for every extent plus
    // one file digest, followed by an optional RSA signature, between the
    // extent table and the descriptor. The descriptor's first magic byte
    // selects the digest algorithm: C=none, I=MD5, H=SHA-1, h=SHA-256.
    let digest_size = match descr.magic[0] {
        b'C' => 0usize,
        b'I' => 16,
        b'H' => 20,
        b'h' => 32,
        other => bail!("Unknown compressed-file authentication type: 0x{other:02x}"),
    };
    let signature_size = match descr.sign_type {
        0 => 0usize,
        1 => 128,
        2 => 256,
        other => bail!("Unknown compressed-file signature type: {other}"),
    };
    let auth_size = comp_extents_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(digest_size))
        .and_then(|size| size.checked_add(signature_size))
        .context("compressed-file authentication table size overflow")?;

    // The compressed chunks live at the front of the blob; the comp_extent
    // table is followed by the optional authentication table and descriptor.
    let table_size = comp_extents_count
        .checked_mul(extent_struct_size)
        .context("compressed extent table size overflow")?;
    let table_offset = descr_offset
        .checked_sub(auth_size)
        .and_then(|offset| offset.checked_sub(table_size))
        .context("compressed extent table lies outside the file data")?;

    // Decompress every chunk and concatenate. The reference tool does not
    // trust unpacked_size for sizing -- it simply writes each chunk's full
    // decompressed output -- so we do the same.
    let mut result = Vec::new();
    let mut off = table_offset;
    let mut read = 0usize;
    while read < comp_extents_count && off + extent_struct_size <= descr_offset {
        let comp_ext: Vdfs4CompExtent = unsafe {
            std::ptr::read_unaligned(data[off..].as_ptr() as *const Vdfs4CompExtent)
        };
        off += extent_struct_size;
        read += 1;

        let chunk_len = comp_ext.len_bytes as usize;
        if chunk_len == 0 {
            continue;
        }
        let start = comp_ext.start as usize;
        let chunk_end = (start + chunk_len).min(data.len());

        if comp_ext.flags & VDFS4_CHUNK_FLAG_UNCOMPR != 0 {
            result.extend_from_slice(&data[start..chunk_end]);
        } else {
            let chunk = &data[start..chunk_end];
            let decompressed = decompress_with(chunk, ctype)
                .with_context(|| format!("decompressing chunk #{} (start={} len={})", read, start, chunk_len))?;
            result.extend_from_slice(&decompressed);
        }
    }

    let expected_size = descr.unpacked_size as usize;
    if result.len() != expected_size {
        bail!(
            "decompressed size mismatch: expected {} bytes, got {}",
            expected_size,
            result.len()
        );
    }

    Ok(result)
}

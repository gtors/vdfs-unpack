use anyhow::{Context, Result, anyhow, bail};
use std::cmp::Ordering;

use crate::disk::*;
use crate::reader::ImageReader;

pub struct Btree<'a> {
    pub reader: &'a ImageReader<'a>,
    pub tree_ino: u64,
    pub root_node_id: u32,
    pub height: u16,
    pub node_size: usize,
}

pub struct Bnode {
    pub node_id: u32,
    pub prev_node_id: u32,
    pub next_node_id: u32,
    pub node_type: u32,
    pub recs_count: u16,
    pub offsets: Vec<u32>,
    pub data: Vec<u8>,
}

impl Bnode {
    pub fn get_record(&self, index: usize) -> Option<&[u8]> {
        let &off = self.offsets.get(index)?;
        let off = off as usize;

        if off >= self.data.len() {
            return None;
        }

        let available = self.data.len() - off;
        let rec = &self.data[off..];

        if available < 8 {
            return Some(&rec[..available]);
        }

        let rec_len = cattree_key_record_len(rec).min(available);
        Some(&rec[..rec_len])
    }

    pub fn get_child_node_id(&self, index: usize) -> Option<u32> {
        let rec = self.get_record(index)?;
        let key_len = cattree_key_key_len(rec);

        if key_len > 0 && key_len.checked_add(4)? <= rec.len() {
            Some(read_u32_le(rec, key_len))
        } else {
            None
        }
    }
}

impl<'a> Btree<'a> {
    pub fn open(reader: &'a ImageReader<'a>, tree_ino: u64) -> Result<Self> {
        let node_size = reader.sbi.super_page_size();
        let meta_page = reader.read_meta_page(tree_ino, 0)?;

        let head: Vdfs4RawBtreeHead =
            unsafe { read_packed(&meta_page, 0).context("Failed to read btree head")? };

        if head.magic != *VDFS4_BTREE_HEAD_NODE_MAGIC {
            bail!(
                "Invalid btree head magic for ino {}: {:?}",
                tree_ino,
                head.magic
            );
        }

        Ok(Self {
            reader,
            tree_ino,
            root_node_id: head.root_bnode_id,
            height: head.btree_height,
            node_size,
        })
    }

    pub fn read_node(&self, node_id: u32) -> Result<Bnode> {
        let raw = if node_id == 0 {
            self.reader.read_meta_page(self.tree_ino, 0)?
        } else {
            self.reader.read_bnode_data(self.tree_ino, node_id)?
        };

        if raw.len() < 32 || raw.len() < self.node_size {
            bail!(
                "Node {} payload size mismatch (expected at least {}, got {})",
                node_id,
                self.node_size,
                raw.len()
            );
        }

        let node_type = read_u32_le(&raw, 28);
        let recs_count = read_u16_le(&raw, 14) as usize;
        let prev_node_id = read_u32_le(&raw, 20);
        let next_node_id = read_u32_le(&raw, 24);

        let mut offsets = Vec::with_capacity(recs_count + 1);

        for i in 0..=recs_count {
            let offset_from_end = (i + 1)
                .checked_mul(4)
                .and_then(|val| val.checked_add(VDFS4_BNODE_FIRST_OFFSET))
                .ok_or_else(|| anyhow!("Offset arithmetic overflow in node {}", node_id))?;

            let addr = self.node_size.checked_sub(offset_from_end).ok_or_else(|| {
                anyhow!(
                    "Calculated offset address out of node bounds in node {}",
                    node_id
                )
            })?;

            if addr + 4 > raw.len() {
                bail!("Offset position out of data bounds in node {}", node_id);
            }

            let off = read_u32_le(&raw, addr);
            offsets.push(off);
        }

        Ok(Bnode {
            node_id,
            prev_node_id,
            next_node_id,
            node_type,
            recs_count: recs_count as u16,
            offsets,
            data: raw,
        })
    }

    pub fn find_leaf(
        &self,
        cmp_fn: &dyn Fn(&[u8], &[u8]) -> Ordering,
        search_key: &[u8],
    ) -> Result<(Bnode, usize)> {
        let mut node = self.read_node(self.root_node_id)?;

        for _level in 0..self.height {
            if node.recs_count == 0 {
                bail!("Empty node {} during lookup", node.node_id);
            }

            let search_res = binary_search_node(&node, cmp_fn, search_key)?;

            if node.node_type == VDFS4_NODE_LEAF {
                let index = match search_res {
                    Ok(idx) => idx,
                    Err(idx) => idx,
                };
                return Ok((node, index));
            }

            let child_index = match search_res {
                Ok(idx) => idx,
                Err(idx) => idx.saturating_sub(1),
            };

            let child_id = node.get_child_node_id(child_index).ok_or_else(|| {
                anyhow!(
                    "Invalid or null child node pointer at index {} in node {}",
                    child_index,
                    node.node_id
                )
            })?;

            if child_id == 0 {
                bail!(
                    "Null child node pointer at index {} in node {}",
                    child_index,
                    node.node_id
                );
            }

            node = self.read_node(child_id)?;
        }

        bail!("Reached maximum tree height limit without landing on a leaf node")
    }

    pub fn find_record(
        &self,
        cmp_fn: &dyn Fn(&[u8], &[u8]) -> Ordering,
        search_key: &[u8],
    ) -> Result<Option<(u32, usize)>> {
        let (leaf_node, index) = self.find_leaf(cmp_fn, search_key)?;

        if index < leaf_node.recs_count as usize {
            if let Some(rec) = leaf_node.get_record(index) {
                if cmp_fn(search_key, rec) == Ordering::Equal {
                    return Ok(Some((leaf_node.node_id, index)));
                }
            }
        }

        Ok(None)
    }

    pub fn get_first_leaf_node(&self) -> Result<Bnode> {
        let mut node = self.read_node(self.root_node_id)?;

        for _level in 0..self.height {
            if node.node_type == VDFS4_NODE_LEAF {
                return Ok(node);
            }

            if node.recs_count == 0 {
                bail!("Empty internal node {}", node.node_id);
            }

            let child_id = node
                .get_child_node_id(0)
                .ok_or_else(|| anyhow!("Failed to read child node 0 in node {}", node.node_id))?;

            if child_id == 0 {
                bail!("Null child pointer at index 0 in node {}", node.node_id);
            }

            node = self.read_node(child_id)?;
        }

        bail!("Failed to reach leaf node following left-most paths")
    }

    pub fn collect_all_records(&self) -> Result<Vec<Vec<u8>>> {
        let mut node = self.get_first_leaf_node()?;
        let mut records = Vec::new();
        let mut visited_count = 0usize;
        const MAX_LEAF_TRAVERSAL_LIMIT: usize = 1_000_000;

        loop {
            visited_count += 1;
            if visited_count > MAX_LEAF_TRAVERSAL_LIMIT {
                bail!("Exceeded maximum leaf node iteration limit (potential cycle detected)");
            }

            for i in 0..node.recs_count as usize {
                let rec = node.get_record(i).ok_or_else(|| {
                    anyhow!(
                        "Corrupted record offset at index {} in node {}",
                        i,
                        node.node_id
                    )
                })?;
                records.push(rec.to_vec());
            }

            if node.next_node_id == 0 {
                break;
            }

            node = self.read_node(node.next_node_id)?;
        }

        Ok(records)
    }
}

pub fn binary_search_node(
    node: &Bnode,
    cmp_fn: &dyn Fn(&[u8], &[u8]) -> Ordering,
    search_key: &[u8],
) -> Result<Result<usize, usize>> {
    let count = node.recs_count as usize;
    let mut err = None;

    let res = (0..count)
        .collect::<Vec<_>>()
        .binary_search_by(|&i| match node.get_record(i) {
            Some(rec) => cmp_fn(search_key, rec),
            None => {
                err = Some(anyhow!(
                    "Failed to read record at index {} in node {}",
                    i,
                    node.node_id
                ));
                Ordering::Less
            }
        });

    if let Some(e) = err {
        return Err(e);
    }

    Ok(res)
}

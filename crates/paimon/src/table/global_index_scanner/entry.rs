// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Manifest entry parsing and file-level pruning metadata.

use crate::btree::BTreeIndexMeta;
use crate::spec::{DataType, PredicateOperator};
use crate::table::bitmap_global_index_format::{
    is_bitmap_floating_residual_sensitive_op, opposite_zero_key,
};
use crate::table::index_file_path::IndexFileLocation;
use crate::{Error, Result};
use std::cmp::Ordering;
use std::collections::HashMap;

/// A resolved global index entry with parsed metadata.
pub(super) struct GlobalIndexEntry {
    pub(super) file_name: String,
    pub(super) index_type: GlobalIndexFileKind,
    pub(super) file_size: i64,
    pub(super) row_range_start: i64,
    pub(super) row_range_end: i64,
    pub(super) external_path: Option<String>,
    pub(super) meta: GlobalIndexEntryMeta,
}

impl GlobalIndexEntry {
    /// The entry's on-disk path: its external path if set, else the table's
    /// global index directory. Also the BTree reader-cache key.
    pub(super) fn resolved_path(&self, table_path: &str) -> String {
        IndexFileLocation::Global { table_path }
            .resolve(&self.file_name, self.external_path.as_deref())
    }
}

pub(super) fn sorted_entry_meta(entry: &GlobalIndexEntry) -> &BTreeIndexMeta {
    match &entry.meta {
        GlobalIndexEntryMeta::Sorted(meta) => meta,
        GlobalIndexEntryMeta::FM { .. } => unreachable!("FM entries do not have sorted metadata"),
    }
}

pub(super) enum GlobalIndexEntryMeta {
    Sorted(BTreeIndexMeta),
    FM {
        bytes: Vec<u8>,
        first_row_id: u64,
        row_count: u64,
    },
}

struct FMFileRowRange<'a> {
    file_name: &'a str,
    first_row_id: u64,
    row_count: u64,
}

pub(super) fn validate_fm_file_sets(
    entries_by_field: &HashMap<i32, Vec<GlobalIndexEntry>>,
) -> Result<()> {
    for (field_id, entries) in entries_by_field {
        let mut groups: HashMap<(i64, i64), Vec<FMFileRowRange<'_>>> = HashMap::new();
        for entry in entries {
            let GlobalIndexEntryMeta::FM {
                first_row_id,
                row_count,
                ..
            } = &entry.meta
            else {
                continue;
            };
            groups
                .entry((entry.row_range_start, entry.row_range_end))
                .or_default()
                .push(FMFileRowRange {
                    file_name: &entry.file_name,
                    first_row_id: *first_row_id,
                    row_count: *row_count,
                });
        }

        for ((range_start, range_end), mut files) in groups {
            let expected_row_count = range_end
                .checked_sub(range_start)
                .and_then(|count| count.checked_add(1))
                .and_then(|count| u64::try_from(count).ok())
                .ok_or_else(|| Error::DataInvalid {
                    message: format!(
                        "Invalid FM global index source row range [{range_start}, {range_end}] for field {field_id}"
                    ),
                    source: None,
                })?;
            files.sort_unstable_by_key(|file| file.first_row_id);
            let mut expected_first_row_id = 0u64;
            for file in files {
                let FMFileRowRange {
                    file_name,
                    first_row_id,
                    row_count,
                } = file;
                if row_count == 0 || first_row_id != expected_first_row_id {
                    return Err(Error::DataInvalid {
                        message: format!(
                            "FM global index files do not exactly cover source row range [{range_start}, {range_end}] for field {field_id}: expected relative row {expected_first_row_id}, file '{file_name}' starts at {first_row_id}"
                        ),
                        source: None,
                    });
                }
                expected_first_row_id =
                    first_row_id
                        .checked_add(row_count)
                        .ok_or_else(|| Error::DataInvalid {
                            message: format!(
                                "FM global index row range overflows for file '{file_name}'"
                            ),
                            source: None,
                        })?;
                if expected_first_row_id > expected_row_count {
                    return Err(Error::DataInvalid {
                        message: format!(
                            "FM global index file '{file_name}' extends beyond source row range [{range_start}, {range_end}]"
                        ),
                        source: None,
                    });
                }
            }
            if expected_first_row_id != expected_row_count {
                return Err(Error::DataInvalid {
                    message: format!(
                        "FM global index files cover {expected_first_row_id} rows, expected {expected_row_count} for source row range [{range_start}, {range_end}] and field {field_id}"
                    ),
                    source: None,
                });
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GlobalIndexFileKind {
    BTree,
    Bitmap,
    Multivalue,
    FM,
}

pub(super) fn is_floating_point(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Float(_) | DataType::Double(_))
}

pub(super) fn bitmap_meta_may_match(
    meta: &BTreeIndexMeta,
    op: PredicateOperator,
    data_type: &DataType,
    serialized_literals: &[Vec<u8>],
    cmp: &dyn Fn(&[u8], &[u8]) -> Ordering,
) -> bool {
    if is_floating_point(data_type) && is_bitmap_floating_residual_sensitive_op(op) {
        return !meta.only_nulls();
    }
    if meta.may_match(op, serialized_literals, cmp) {
        return true;
    }
    // A `±0.0` literal also matches rows holding the other signed zero, but the
    // comparator orders `-0.0` below `+0.0`, so a key range that stops at one of
    // them would prune a file the reader would have found. `Eq` looks at the first
    // literal only, so the alternates are probed as a call of their own rather
    // than appended.
    let alternate_zeros: Vec<Vec<u8>> = serialized_literals
        .iter()
        .filter_map(|literal| opposite_zero_key(literal, data_type))
        .collect();
    !alternate_zeros.is_empty() && meta.may_match(op, &alternate_zeros, cmp)
}

pub(super) fn bitmap_meta_may_match_between(
    meta: &BTreeIndexMeta,
    data_type: &DataType,
    from_key: &[u8],
    to_key: &[u8],
    cmp: &dyn Fn(&[u8], &[u8]) -> Ordering,
) -> bool {
    if is_floating_point(data_type)
        && is_bitmap_floating_residual_sensitive_op(PredicateOperator::Between)
    {
        !meta.only_nulls()
    } else {
        meta.may_match_between(from_key, to_key, cmp)
    }
}

pub(super) fn multivalue_meta_may_match(
    meta: &BTreeIndexMeta,
    op: PredicateOperator,
    serialized_literals: &[Vec<u8>],
    cmp: &dyn Fn(&[u8], &[u8]) -> Ordering,
) -> bool {
    match op {
        PredicateOperator::ArrayContains => {
            meta.may_match(PredicateOperator::Eq, serialized_literals, cmp)
        }
        PredicateOperator::ArraysOverlap => {
            meta.may_match(PredicateOperator::In, serialized_literals, cmp)
        }
        PredicateOperator::ArrayContainsAll => serialized_literals.iter().all(|literal| {
            meta.may_match(PredicateOperator::Eq, std::slice::from_ref(literal), cmp)
        }),
        _ => false,
    }
}

impl GlobalIndexFileKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::BTree => "BTree",
            Self::Bitmap => "bitmap",
            Self::Multivalue => "multivalue",
            Self::FM => "FM",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{Datum, DoubleType, FloatType};
    use crate::table::bitmap_global_index_format::{
        make_bitmap_key_comparator, serialize_bitmap_datum,
    };

    /// A file whose key range stops at one signed zero must still be visited when
    /// the predicate names the other: the comparator orders `-0.0` below `+0.0`, so
    /// the plain min/max check prunes a file the reader would have matched.
    fn assert_signed_zero_survives_meta_prune(
        data_type: DataType,
        negative_zero: Datum,
        positive_zero: Datum,
        one: Datum,
    ) {
        let cmp = make_bitmap_key_comparator(&data_type);
        let negative_zero_key = serialize_bitmap_datum(&negative_zero, &data_type);
        let positive_zero_key = serialize_bitmap_datum(&positive_zero, &data_type);
        let one_key = serialize_bitmap_datum(&one, &data_type);

        // Key range is exactly [-0.0, -0.0]; the literal is +0.0.
        let meta = BTreeIndexMeta::new(
            Some(negative_zero_key.clone()),
            Some(negative_zero_key.clone()),
            false,
        );
        for op in [PredicateOperator::Eq, PredicateOperator::In] {
            assert!(
                bitmap_meta_may_match(
                    &meta,
                    op,
                    &data_type,
                    std::slice::from_ref(&positive_zero_key),
                    &cmp
                ),
                "{op} on +0.0 must not prune a -0.0-only file ({data_type:?})"
            );
        }
        // In with a mix of literals.
        assert!(bitmap_meta_may_match(
            &meta,
            PredicateOperator::In,
            &data_type,
            &[positive_zero_key.clone(), one_key.clone()],
            &cmp
        ));

        // The other direction: range is [+0.0, +0.0], literal is -0.0.
        let meta = BTreeIndexMeta::new(
            Some(positive_zero_key.clone()),
            Some(positive_zero_key.clone()),
            false,
        );
        assert!(bitmap_meta_may_match(
            &meta,
            PredicateOperator::Eq,
            &data_type,
            std::slice::from_ref(&negative_zero_key),
            &cmp
        ));

        // A file that holds neither zero is still pruned for a zero literal.
        let meta = BTreeIndexMeta::new(Some(one_key.clone()), Some(one_key), false);
        assert!(
            !bitmap_meta_may_match(
                &meta,
                PredicateOperator::Eq,
                &data_type,
                &[positive_zero_key],
                &cmp
            ),
            "pruning must still happen when no zero is in range ({data_type:?})"
        );
    }

    #[test]
    fn test_signed_zero_survives_meta_prune() {
        assert_signed_zero_survives_meta_prune(
            DataType::Float(FloatType::new()),
            Datum::Float(-0.0),
            Datum::Float(0.0),
            Datum::Float(1.0),
        );
        assert_signed_zero_survives_meta_prune(
            DataType::Double(DoubleType::new()),
            Datum::Double(-0.0),
            Datum::Double(0.0),
            Datum::Double(1.0),
        );
    }
}

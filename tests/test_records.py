"""JSONL boundary recovery, exact-once ownership, and structural validation."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from climbmix_mixture import config
from climbmix_mixture.bytesource import LocalRangeReader, SourceFile
from climbmix_mixture.records import (
    ParsedRecord,
    iter_owned_records,
    record_identity_str,
    validate_record,
)
from climbmix_mixture.workplan import WorkItem

from tests.synthetic import build_default_synthetic_source, doc_line


def _regions(size: int, region_bytes: int) -> list[WorkItem]:
    items: list[WorkItem] = []
    start, index = 0, 0
    while start < size:
        items.append(WorkItem(index=index, filename="f", range_start=start,
                              range_end=min(start + region_bytes, size)))
        index += 1
        start += region_bytes
    return items


class BoundaryOwnershipTest(unittest.TestCase):
    """Adjacent ranges must neither lose nor duplicate any record."""

    def _owned_starts(self, body: bytes, region_bytes: int) -> list[int]:
        reader = LocalRangeReader(body)
        starts: list[int] = []
        for item in _regions(len(body), region_bytes):
            for rec in iter_owned_records(item, reader):
                starts.append(rec.record_start)
        return starts

    def test_exact_once_ownership_across_adjacent_ranges(self) -> None:
        # A sequence of varied documents, one much longer than the region.
        lines = [
            doc_line(1, [i] * (i % 7 + 1)) for i in range(1, 40)
        ]
        lines.insert(5, doc_line(3, [7] * 800))  # long record spanning many regions
        body = b"".join(l + b"\n" for l in lines)
        expected_starts: list[int] = []
        offset = 0
        for l in lines:
            expected_starts.append(offset)
            offset += len(l) + 1

        for region_bytes in (16, 37, 64, 128, 1024, len(body)):
            owned = self._owned_starts(body, region_bytes)
            self.assertEqual(sorted(owned), sorted(expected_starts),
                             f"records lost/duplicated at region_bytes={region_bytes}")
            self.assertEqual(len(owned), len(set(owned)), "duplicate ownership")

    def test_long_record_owned_once_and_interior_regions_own_nothing(self) -> None:
        # One giant record followed by a small one. Region is small so the giant
        # spans many regions; only the region containing its first byte owns it.
        long_tokens = [i % 100 for i in range(3000)]
        body = doc_line(1, long_tokens) + b"\n" + doc_line(2, [1, 2]) + b"\n"
        region_bytes = 64
        reader = LocalRangeReader(body)
        owners: list[tuple[int, int]] = []  # (record_start, owner_index)
        for item in _regions(len(body), region_bytes):
            for rec in iter_owned_records(item, reader):
                owners.append((rec.record_start, item.index))
        self.assertEqual([s for s, _ in owners], [0, len(body) - len(doc_line(2, [1, 2])) - 1])
        # The long record's start is 0 -> owned by region 0 only.
        self.assertEqual(owners[0][1], 0)

    def test_final_record_without_trailing_newline(self) -> None:
        body = doc_line(1, [1, 2]) + b"\n" + doc_line(2, [3])  # no trailing newline
        reader = LocalRangeReader(body)
        starts = [rec.record_start for item in _regions(len(body), 32)
                  for rec in iter_owned_records(item, reader)]
        self.assertEqual(starts, [0, len(doc_line(1, [1, 2])) + 1])

    def test_records_split_across_range_edges_reconstruct(self) -> None:
        # Single long record; every region except region 0 sees only interior
        # bytes; ownership must stay with region 0.
        body = doc_line(1, [1] * 5000) + b"\n"
        reader = LocalRangeReader(body)
        owned = [rec.record_start for item in _regions(len(body), 128)
                 for rec in iter_owned_records(item, reader)]
        self.assertEqual(owned, [0])

    def test_on_disk_reader_matches_in_memory(self) -> None:
        src = build_default_synthetic_source()
        for name, body in src.files.items():
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / name
                path.write_bytes(body)
                disk = LocalRangeReader(path)
                mem = LocalRangeReader(body)
                for item in _regions(len(body), 128):
                    self.assertEqual(
                        [r.record_start for r in iter_owned_records(item, disk)],
                        [r.record_start for r in iter_owned_records(item, mem)],
                    )


class StructuralValidationTest(unittest.TestCase):
    def _validate(self, raw: bytes) -> tuple[bool, str | None]:
        result = validate_record(ParsedRecord(record_start=0, raw=raw))
        return result.valid, result.rejection_reason

    def test_valid_record(self) -> None:
        valid, reason = self._validate(doc_line(5, [1, 2, 3]))
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_bad_json(self) -> None:
        _, reason = self._validate(b"not-json{")
        self.assertEqual(reason, "json_parse_error")

    def test_missing_cluster_id(self) -> None:
        _, reason = self._validate(b'{"tokens":[1,2]}')
        self.assertEqual(reason, "missing_cluster_id")

    def test_cluster_id_not_integer(self) -> None:
        _, reason = self._validate(b'{"cluster_id":1.5,"tokens":[1]}')
        self.assertEqual(reason, "cluster_id_not_integer")
        _, reason = self._validate(b'{"cluster_id":true,"tokens":[1]}')
        self.assertEqual(reason, "cluster_id_not_integer")
        _, reason = self._validate(b'{"cluster_id":"1","tokens":[1]}')
        self.assertEqual(reason, "cluster_id_not_integer")

    def test_cluster_id_out_of_range(self) -> None:
        _, reason = self._validate(b'{"cluster_id":0,"tokens":[1]}')
        self.assertEqual(reason, "cluster_id_out_of_range")
        _, reason = self._validate(b'{"cluster_id":21,"tokens":[1]}')
        self.assertEqual(reason, "cluster_id_out_of_range")

    def test_missing_or_malformed_tokens(self) -> None:
        _, reason = self._validate(b'{"cluster_id":1}')
        self.assertEqual(reason, "missing_tokens")
        _, reason = self._validate(b'{"cluster_id":1,"tokens":"x"}')
        self.assertEqual(reason, "tokens_not_list")
        _, reason = self._validate(b'{"cluster_id":1,"tokens":[]}')
        self.assertEqual(reason, "tokens_empty")

    def test_token_not_integer(self) -> None:
        _, reason = self._validate(b'{"cluster_id":1,"tokens":[1,true]}')
        self.assertEqual(reason, "token_not_integer")
        _, reason = self._validate(b'{"cluster_id":1,"tokens":[1,2.5]}')
        self.assertEqual(reason, "token_not_integer")

    def test_token_out_of_range(self) -> None:
        _, reason = self._validate(b'{"cluster_id":1,"tokens":[0,50257]}')
        self.assertEqual(reason, "token_out_of_range")
        _, reason = self._validate(b'{"cluster_id":1,"tokens":[-1]}')
        self.assertEqual(reason, "token_out_of_range")

    def test_token_count_mismatch(self) -> None:
        _, reason = self._validate(b'{"cluster_id":1,"tokens":[1,2],"token_count":3}')
        self.assertEqual(reason, "token_count_mismatch")

    def test_cluster_11_is_structurally_valid(self) -> None:
        # Cluster 11 is NOT a structural reject; it is a policy decision handled
        # by the build loop using only the numeric cluster_id.
        result = validate_record(ParsedRecord(record_start=0, raw=doc_line(11, [1, 2])))
        self.assertTrue(result.valid)
        self.assertEqual(result.cluster_id, 11)
        self.assertIn(11, config.EXCLUDED_CLUSTER_IDS)
        self.assertNotIn(11, config.ACCEPTED_CLUSTER_IDS)

    def test_identity_is_stable_and_independent_of_order(self) -> None:
        self.assertEqual(record_identity_str("rev", "part_0", 42),
                         record_identity_str("rev", "part_0", 42))
        self.assertNotEqual(record_identity_str("rev", "part_0", 42),
                            record_identity_str("rev", "part_0", 43))


if __name__ == "__main__":
    unittest.main()
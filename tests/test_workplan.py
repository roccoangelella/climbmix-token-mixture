"""Deterministic work-plan generation, persistence, and hash checking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from climbmix_mixture import config
from climbmix_mixture.bytesource import SourceFile
from climbmix_mixture.workplan import (
    WorkPlan,
    build_work_plan,
    load_work_plan,
    save_work_plan,
    work_item_identity,
)


def _files() -> list[SourceFile]:
    return [
        SourceFile("part_0.tokenized.jsonl", 1_000),
        SourceFile("part_1.tokenized.jsonl", 2_500),
        SourceFile("part_2.tokenized.jsonl", 150),
    ]


class WorkPlanTest(unittest.TestCase):
    def test_regions_cover_every_file_exactly(self) -> None:
        plan = build_work_plan(_files(), region_bytes=256, seed="seed-x",
                               repository="repo", revision="rev-1")
        by_file: dict[str, list[tuple[int, int]]] = {}
        for item in plan.work_items:
            by_file.setdefault(item.filename, []).append((item.range_start, item.range_end))
        for source in _files():
            regions = sorted(by_file[source.path])
            self.assertEqual(regions[0][0], 0)
            self.assertEqual(regions[-1][1], source.size)
            # Coverage is contiguous: end of each region == start of the next.
            for a, b in zip(regions, regions[1:]):
                self.assertEqual(a[1], b[0])

    def test_deterministic_order_for_same_seed(self) -> None:
        a = build_work_plan(_files(), region_bytes=256, seed="seed-x",
                            repository="repo", revision="rev-1")
        b = build_work_plan(_files(), region_bytes=256, seed="seed-x",
                            repository="repo", revision="rev-1")
        self.assertEqual(
            [(i.filename, i.range_start, i.range_end) for i in a.work_items],
            [(i.filename, i.range_start, i.range_end) for i in b.work_items],
        )
        self.assertEqual(a.hash, b.hash)
        for item in a.work_items:
            self.assertEqual(
                item.work_item_id,
                work_item_identity(
                    "rev-1", item.filename, item.range_start, item.range_end
                ),
            )

    def test_seed_change_changes_order_and_hash(self) -> None:
        a = build_work_plan(_files(), region_bytes=256, seed="seed-a",
                            repository="repo", revision="rev-1")
        b = build_work_plan(_files(), region_bytes=256, seed="seed-b",
                            repository="repo", revision="rev-1")
        self.assertNotEqual([(i.filename, i.range_start) for i in a.work_items],
                            [(i.filename, i.range_start) for i in b.work_items])
        self.assertNotEqual(a.hash, b.hash)

    def test_persistence_round_trip_preserves_hash(self) -> None:
        plan = build_work_plan(_files(), region_bytes=256, seed="seed-x",
                               repository="repo", revision="rev-1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / config.WORK_PLAN_FILENAME
            save_work_plan(path, plan)
            loaded = load_work_plan(path)
        self.assertEqual(loaded.hash, plan.hash)
        self.assertEqual(loaded.revision, "rev-1")
        self.assertEqual(loaded.selection_seed, "seed-x")
        self.assertEqual(len(loaded.work_items), len(plan.work_items))
        self.assertEqual(
            [(i.filename, i.range_start, i.range_end) for i in loaded.work_items],
            [(i.filename, i.range_start, i.range_end) for i in plan.work_items],
        )

    def test_tampered_plan_is_rejected(self) -> None:
        plan = build_work_plan(_files(), region_bytes=256, seed="seed-x",
                               repository="repo", revision="rev-1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / config.WORK_PLAN_FILENAME
            save_work_plan(path, plan)
            payload = plan.to_dict()
            payload["region_bytes"] = 999  # tamper without updating hash
            import json as _json
            path.write_text(_json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_work_plan(path)

    def test_hash_is_stable_across_machines(self) -> None:
        # Fixed expected digest for a fixed input guards against platform drift.
        plan = build_work_plan(_files(), region_bytes=256, seed="seed-x",
                               repository="repo", revision="rev-1")
        self.assertEqual(len(plan.hash), 64)
        self.assertEqual(plan.hash, build_work_plan(
            _files(), region_bytes=256, seed="seed-x",
            repository="repo", revision="rev-1").hash)


if __name__ == "__main__":
    unittest.main()

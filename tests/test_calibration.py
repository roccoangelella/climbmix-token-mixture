from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from climbmix_mixture import config
from climbmix_mixture.calibration import (
    MIXTURE_PROGRESS_FILENAME,
    MIXTURE_WEIGHTS_FILENAME,
    extract_record_metadata,
    scan_mixture,
)
from climbmix_mixture.workplan import build_work_plan
from tests.synthetic import SyntheticSource, doc_line


class MixtureMetadataTest(unittest.TestCase):
    def test_extracts_top_level_metadata_without_parsing_tokens(self) -> None:
        raw = (
            b' { "cluster_id" : 12, "tokens" : [1,2,3], '
            b'"token_count" : 3 } '
        )
        metadata = extract_record_metadata(raw)
        self.assertEqual(metadata.cluster_id, 12)
        self.assertEqual(metadata.token_count, 3)

    def test_rejects_ambiguous_or_invalid_metadata(self) -> None:
        cases = (
            b'{"cluster_id":1,"cluster_id":2,"tokens":[1,2,3],"token_count":3}',
            b'{"cluster_id":1,"tokens":[1,2,3]}',
            b'{"cluster_id":21,"tokens":[1,2,3],"token_count":3}',
            b'{"cluster_id":1,"token_count":0,"tokens":[]}',
        )
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    extract_record_metadata(raw)


class MixtureCalibrationTest(unittest.TestCase):
    def _source_and_plan(self):
        source = SyntheticSource()
        expected: dict[int, int] = {}
        lines: list[bytes] = []
        for cluster in sorted(config.ALL_CLUSTER_IDS):
            count = cluster + 1
            expected[cluster] = count
            lines.append(doc_line(cluster, list(range(count))))
        source.add_file("part_all.tokenized.jsonl", lines)
        plan = build_work_plan(
            source.source_files,
            region_bytes=97,
            seed=config.SELECTION_SEED,
            repository=config.DATASET_REPOSITORY,
            revision=config.DATASET_REVISION,
        )
        return source, plan, expected

    def test_complete_scan_emits_exact_conditional_weights(self) -> None:
        source, plan, expected = self._source_and_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = scan_mixture(
                root,
                plan,
                source.reader_factory(),
                workers=3,
                max_in_flight=4,
                checkpoint_every_work_items=2,
            )
            weights = json.loads((root / MIXTURE_WEIGHTS_FILENAME).read_text())
            self.assertEqual(report["record_count"], 20)
            self.assertEqual(report["all_source_tokens"], sum(expected.values()))
            self.assertNotIn("11", weights)
            self.assertEqual(
                weights,
                {
                    str(cluster): expected[cluster]
                    for cluster in sorted(config.ACCEPTED_CLUSTER_IDS)
                },
            )
            self.assertTrue(report["complete"])

    def test_resume_is_byte_equivalent_to_uninterrupted_scan(self) -> None:
        source, plan, _ = self._source_and_plan()
        with tempfile.TemporaryDirectory() as interrupted_tmp, tempfile.TemporaryDirectory() as clean_tmp:
            interrupted = Path(interrupted_tmp)
            with self.assertRaisesRegex(RuntimeError, "simulated mixture"):
                scan_mixture(
                    interrupted,
                    plan,
                    source.reader_factory(),
                    workers=2,
                    max_in_flight=3,
                    checkpoint_every_work_items=1,
                    simulate_crash_after_work_items=2,
                )
            resumed = scan_mixture(
                interrupted,
                plan,
                source.reader_factory(),
                resume=True,
                workers=4,
                max_in_flight=5,
                checkpoint_every_work_items=2,
            )
            clean = scan_mixture(
                Path(clean_tmp),
                plan,
                source.reader_factory(),
                workers=1,
                max_in_flight=1,
                checkpoint_every_work_items=1,
            )
            self.assertEqual(resumed["report_sha256"], clean["report_sha256"])
            self.assertEqual(
                (interrupted / MIXTURE_WEIGHTS_FILENAME).read_bytes(),
                (Path(clean_tmp) / MIXTURE_WEIGHTS_FILENAME).read_bytes(),
            )

    def test_resume_rejects_work_plan_drift(self) -> None:
        source, plan, _ = self._source_and_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "simulated mixture"):
                scan_mixture(
                    root,
                    plan,
                    source.reader_factory(),
                    checkpoint_every_work_items=1,
                    simulate_crash_after_work_items=1,
                )
            progress_path = root / MIXTURE_PROGRESS_FILENAME
            progress = json.loads(progress_path.read_text())
            progress["work_plan_hash"] = "changed"
            progress_path.write_text(json.dumps(progress))
            with self.assertRaisesRegex(ValueError, "work_plan_hash"):
                scan_mixture(
                    root,
                    plan,
                    source.reader_factory(),
                    resume=True,
                )

    def test_bad_record_fails_closed_with_source_identity(self) -> None:
        source = SyntheticSource()
        lines = [
            doc_line(cluster, [cluster])
            for cluster in sorted(config.ALL_CLUSTER_IDS)
        ]
        lines[0] = b'{"cluster_id":1,"tokens":[1]}'
        source.add_file("part_bad.tokenized.jsonl", lines)
        plan = build_work_plan(
            source.source_files,
            region_bytes=100,
            seed=config.SELECTION_SEED,
            repository=config.DATASET_REPOSITORY,
            revision=config.DATASET_REVISION,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "part_bad.tokenized.jsonl"):
                scan_mixture(
                    Path(tmp),
                    plan,
                    source.reader_factory(),
                    workers=2,
                    max_in_flight=2,
                )


if __name__ == "__main__":
    unittest.main()

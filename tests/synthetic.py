"""Shared synthetic-source helpers for the token-only pipeline tests.

Tests never contact Hugging Face.  All source files are JSONL byte strings built
in memory (or in temp directories) and consumed by :class:`LocalRangeReader`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from climbmix_mixture import config
from climbmix_mixture.bytesource import LocalRangeReader, RangeReader, SourceFile


FULL_ACCEPTED_SOURCE_TOKENS = 5_042


def doc_line(cluster_id: int, tokens: list[int], *, token_count: int | None = None) -> bytes:
    """Serialize one ClimbMix-shaped JSONL record (no trailing newline)."""

    payload = {"cluster_id": cluster_id, "tokens": tokens}
    if token_count is not None:
        payload["token_count"] = token_count
    else:
        payload["token_count"] = len(tokens)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def line_with_newline(raw: bytes) -> bytes:
    return raw + b"\n"


class SyntheticSource:
    """An in-memory multi-file synthetic Nemotron-ClimbMix source."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.source_files: list[SourceFile] = []
        # Absolute record starts for every record, for ownership assertions.
        self.record_starts: dict[str, list[int]] = {}
        # The tokens actually accepted per record, for end-to-end assertions.
        self.accepted_identity_tokens: dict[str, list[int]] = {}

    def add_file(self, name: str, lines: list[bytes]) -> None:
        body = b"".join(line_with_newline(ln) for ln in lines)
        self.files[name] = body
        starts: list[int] = []
        offset = 0
        for ln in lines:
            starts.append(offset)
            offset += len(ln) + 1  # +1 for the newline
        self.record_starts[name] = starts
        self.source_files.append(SourceFile(path=name, size=len(body)))

    def reader_factory(self) -> Callable[[SourceFile], RangeReader]:
        files = self.files

        def factory(source_file: SourceFile) -> RangeReader:
            return LocalRangeReader(files[source_file.path])

        return factory

    def reader_factory_on_disk(self, root: Path) -> Callable[[SourceFile], RangeReader]:
        files: dict[str, Path] = {}
        for name, body in self.files.items():
            path = root / name
            path.write_bytes(body)
            files[name] = path

        def factory(source_file: SourceFile) -> RangeReader:
            return LocalRangeReader(files[source_file.path])

        return factory


def build_default_synthetic_source() -> SyntheticSource:
    """A small varied source exercising clusters, EOD edges, and one bad record.

    Includes:
    - accepted clusters 1..10, 12..20 (sampled, not all 19).
    - excluded cluster 11 documents.
    - one document that already ends in EOD (50256) to prove no double EOD.
    - one structurally invalid record (token out of range).
    - one document much longer than a region to exercise boundary ownership.
    """

    src = SyntheticSource()

    # part_a: short, varied documents.
    lines_a: list[bytes] = []
    line_specs_a = [
        (1, [10, 20, 30]),
        (2, [100, 200, 300, 400]),
        (11, [1, 2, 3]),                 # excluded cluster 11
        (3, [5, 6, 7, 8, 9]),
        (20, [11, 12, 13]),
        (11, [50, 60]),                  # excluded cluster 11 again
        (4, [201, 202, 203, 204]),
        (5, [301, 302]),
        (6, [401, 402, 403]),
        (12, [501, 502, 503, 504]),
    ]
    for cid, toks in line_specs_a:
        lines_a.append(doc_line(cid, toks))
    src.add_file("part_a.tokenized.jsonl", lines_a)

    # part_b: a long document spanning several regions + a doc ending in EOD + a
    # structurally invalid record.
    lines_b: list[bytes] = []
    long_tokens = [i % 1000 for i in range(5000)]
    lines_b.append(doc_line(7, long_tokens))            # very long foreign-spanning record
    lines_b.append(doc_line(8, [9, 9, 9]))
    lines_b.append(doc_line(9, [12, 13], token_count=2))
    lines_b.append(doc_line(10, [14, 15, 16, 50256]))   # already ends in EOD
    lines_b.append(b'{"cluster_id": 1, "tokens": [50257, 1], "token_count": 2}')  # out of range
    lines_b.append(doc_line(13, [17, 18, 19]))
    lines_b.append(doc_line(14, [21, 22]))
    src.add_file("part_b.tokenized.jsonl", lines_b)

    return src


def make_effective(output_dir: Path, **overrides) -> config.EffectiveConfig:
    """Build a small-but-valid EffectiveConfig for tests."""

    defaults = {
        "output_dir": output_dir,
        "target_accepted_source_tokens": 1_000_000,
        "minimum_accepted_source_tokens": 1,
        "maximum_accepted_source_tokens": 1_000_000_000,
        # End-to-end tests use one logical region per synthetic file so they
        # inspect the complete fixture. Boundary behavior has dedicated tests.
        "region_bytes": 1_000_000,
        "writer_buffer_bytes": 48,
        "checkpoint_bytes_threshold": 200,
        "max_work_items": None,
        "resume": False,
        "strict": False,
        "allow_unsafe_low_disk": True,
        "reset": False,
        "full_scan": False,
        "crash_after_written_bytes": None,
    }
    defaults.update(overrides)
    return config.EffectiveConfig(**defaults)

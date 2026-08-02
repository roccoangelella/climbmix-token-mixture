"""Deterministic work-plan generation over all official root source files.

The plan divides every source file into fixed-size logical byte regions,
deterministically shuffles all regions with the frozen seed, and persists the
complete ordered plan to ``work_plan.json``.  The plan is reusable during resume
and must not silently change if the remote repository's ``main`` branch moves
(it is pinned to an immutable revision).

Ordering is deliberately machine-independent: each stable work-item identity is
SHA-256 ranked with the versioned seed. This avoids depending on a particular
Python PRNG or shuffle implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from . import config

from .bytesource import SourceFile
from .storage import canonical_json_bytes, read_json, write_json_atomic


@dataclass(frozen=True)
class WorkItem:
    """A logical byte region of one source file, in saved processing order.

    ``index`` is the position in the saved (shuffled) plan; the stable source
    identity is ``(revision, filename, range_start, range_end)``.
    """

    index: int
    filename: str
    range_start: int
    range_end: int
    work_item_id: str = ""


@dataclass(frozen=True)
class WorkPlan:
    schema_version: int
    dataset: str
    revision: str
    source_glob: str
    selection_seed: str
    region_bytes: int
    source_files: tuple[SourceFile, ...]
    work_items: tuple[WorkItem, ...]
    hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "revision": self.revision,
            "source_glob": self.source_glob,
            "selection_seed": self.selection_seed,
            "region_bytes": self.region_bytes,
            "source_files": [{"path": f.path, "size": f.size} for f in self.source_files],
            "work_items": [
                {
                    "index": item.index,
                    "filename": item.filename,
                    "range_start": item.range_start,
                    "range_end": item.range_end,
                    "work_item_id": item.work_item_id,
                }
                for item in self.work_items
            ],
            "hash": self.hash,
        }


def work_item_identity(
    revision: str, filename: str, range_start: int, range_end: int
) -> str:
    """Return the stable, human-readable identity of one logical region."""

    return f"{revision}:{filename}:{range_start}:{range_end}"


def build_work_plan(
    source_files: list[SourceFile],
    *,
    region_bytes: int,
    seed: str,
    repository: str,
    revision: str,
) -> WorkPlan:
    """Divide files into regions, hash-shuffle them, and compute the self-hash."""

    if region_bytes <= 0:
        raise ValueError("region_bytes must be positive")
    if not source_files:
        raise ValueError("Cannot build a work plan from zero source files")

    normalized_files = sorted(source_files, key=lambda source: source.path)
    if len({source.path for source in normalized_files}) != len(normalized_files):
        raise ValueError("Source file paths must be unique")
    raw_items: list[WorkItem] = []
    for source in normalized_files:
        if source.size <= 0:
            raise ValueError(f"Source file {source.path!r} has invalid size {source.size}")
        start = 0
        while start < source.size:
            end = min(start + region_bytes, source.size)
            identity = work_item_identity(revision, source.path, start, end)
            raw_items.append(
                WorkItem(
                    index=0,
                    filename=source.path,
                    range_start=start,
                    range_end=end,
                    work_item_id=identity,
                )
            )
            start = end

    # Hash-sort instead of relying on ``random.Random.shuffle`` implementation
    # details.  This is a deterministic shuffle across Python versions and
    # machines, with the source identity as a collision tie-breaker.
    shuffled = sorted(
        raw_items,
        key=lambda item: (
            hashlib.sha256(
                seed.encode("utf-8")
                + b"\0"
                + item.work_item_id.encode("utf-8")
            ).digest(),
            item.work_item_id,
        ),
    )
    work_items = tuple(
        WorkItem(
            index=i,
            filename=item.filename,
            range_start=item.range_start,
            range_end=item.range_end,
            work_item_id=item.work_item_id,
        )
        for i, item in enumerate(shuffled)
    )

    plan = WorkPlan(
        schema_version=config.WORK_PLAN_SCHEMA_VERSION,
        dataset=repository,
        revision=revision,
        source_glob=config.SOURCE_DATA_GLOB,
        selection_seed=seed,
        region_bytes=region_bytes,
        source_files=tuple(normalized_files),
        work_items=work_items,
        hash="",  # filled after hashing
    )
    return replace(plan, hash=_plan_hash(plan))


def _plan_hash(plan: WorkPlan) -> str:
    """SHA-256 over the canonical payload excluding the ``hash`` field."""

    payload = plan.to_dict()
    return hashlib.sha256(canonical_json_bytes(payload, exclude_keys=("hash",))).hexdigest()


def save_work_plan(path: Path, plan: WorkPlan) -> None:
    """Persist the plan atomically."""

    write_json_atomic(path, plan.to_dict())


def load_work_plan(path: Path) -> WorkPlan:
    """Load and fully validate a persisted plan, including its self-hash."""

    payload = read_json(path)
    if payload.get("schema_version") != config.WORK_PLAN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported work-plan schema in {path}")
    stored_hash = payload.get("hash")
    if not isinstance(stored_hash, str):
        raise ValueError(f"Work plan in {path} is missing its hash")
    recomputed = hashlib.sha256(
        canonical_json_bytes(payload, exclude_keys=("hash",))
    ).hexdigest()
    if recomputed != stored_hash:
        raise ValueError(
            f"Work plan in {path} is tampered or corrupt: hash mismatch "
            f"(stored {stored_hash[:12]}, recomputed {recomputed[:12]})"
        )
    source_files = tuple(
        SourceFile(path=str(f["path"]), size=int(f["size"])) for f in payload["source_files"]
    )
    work_items = tuple(
        WorkItem(
            index=int(i["index"]),
            filename=str(i["filename"]),
            range_start=int(i["range_start"]),
            range_end=int(i["range_end"]),
            work_item_id=str(i["work_item_id"]),
        )
        for i in payload["work_items"]
    )
    plan = WorkPlan(
        schema_version=int(payload["schema_version"]),
        dataset=str(payload["dataset"]),
        revision=str(payload["revision"]),
        source_glob=str(payload["source_glob"]),
        selection_seed=str(payload["selection_seed"]),
        region_bytes=int(payload["region_bytes"]),
        source_files=source_files,
        work_items=work_items,
        hash=stored_hash,
    )
    _validate_plan(plan, path)
    return plan


def _validate_plan(plan: WorkPlan, path: Path) -> None:
    """Validate identities, indexes, and exact source-file coverage."""

    if not plan.source_files or not plan.work_items:
        raise ValueError(f"Work plan in {path} is empty")
    source_sizes: dict[str, int] = {}
    for source in plan.source_files:
        if source.path in source_sizes:
            raise ValueError(f"Work plan in {path} repeats source file {source.path!r}")
        if source.size <= 0:
            raise ValueError(f"Work plan in {path} has invalid size for {source.path!r}")
        source_sizes[source.path] = source.size

    by_file: dict[str, list[tuple[int, int]]] = {name: [] for name in source_sizes}
    identities: set[str] = set()
    for expected_index, item in enumerate(plan.work_items):
        if item.index != expected_index:
            raise ValueError(
                f"Work plan in {path} has index {item.index} at position {expected_index}"
            )
        if item.filename not in source_sizes:
            raise ValueError(
                f"Work plan in {path} references unknown file {item.filename!r}"
            )
        expected_identity = work_item_identity(
            plan.revision, item.filename, item.range_start, item.range_end
        )
        if item.work_item_id != expected_identity:
            raise ValueError(
                f"Work plan in {path} has an invalid identity for item {item.index}"
            )
        if item.work_item_id in identities:
            raise ValueError(f"Work plan in {path} repeats {item.work_item_id!r}")
        identities.add(item.work_item_id)
        if not (0 <= item.range_start < item.range_end <= source_sizes[item.filename]):
            raise ValueError(
                f"Work plan in {path} has invalid range "
                f"{item.range_start}:{item.range_end} for {item.filename}"
            )
        by_file[item.filename].append((item.range_start, item.range_end))

    for filename, size in source_sizes.items():
        regions = sorted(by_file[filename])
        cursor = 0
        for start, end in regions:
            if start != cursor:
                raise ValueError(
                    f"Work plan in {path} does not cover {filename} exactly at byte {cursor}"
                )
            cursor = end
        if cursor != size:
            raise ValueError(
                f"Work plan in {path} ends {filename} at {cursor}, expected {size}"
            )

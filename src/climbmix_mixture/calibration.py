"""Exact, resumable cluster-token calibration for Nemotron-ClimbMix."""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping

from . import config

from .bytesource import RangeReader, SourceFile
from .storage import sha256_file
from .records import iter_owned_records
from .storage import canonical_json_bytes, read_json, write_json_atomic
from .workplan import WorkItem, WorkPlan

LOGGER = logging.getLogger(__name__)

MIXTURE_SCAN_SCHEMA_VERSION = 1
MIXTURE_PROGRESS_FILENAME = "mixture_progress.json"
MIXTURE_REPORT_FILENAME = "mixture_report.json"
MIXTURE_WEIGHTS_FILENAME = "climbmix_code_free_weights.json"

_CLUSTER_KEY = b'"cluster_id"'
_TOKENS_KEY = b'"tokens"'
_TOKEN_COUNT_KEY = b'"token_count"'


@dataclass(frozen=True)
class RecordMetadata:
    cluster_id: int
    token_count: int


@dataclass(frozen=True)
class WorkItemMixture:
    index: int
    source_bytes: int
    record_count: int
    cluster_source_tokens: dict[int, int]
    cluster_document_counts: dict[int, int]


def _skip_ws(raw: bytes, index: int) -> int:
    while index < len(raw) and raw[index] in b" \t\r\n":
        index += 1
    return index


def _parse_integer(raw: bytes, index: int) -> tuple[int, int]:
    index = _skip_ws(raw, index)
    start = index
    if index < len(raw) and raw[index] == ord("-"):
        index += 1
    digit_start = index
    while index < len(raw) and ord("0") <= raw[index] <= ord("9"):
        index += 1
    if index == digit_start:
        raise ValueError("metadata value is not an integer")
    return int(raw[start:index]), index


def _colon_after(raw: bytes, key_at: int, key: bytes) -> int:
    index = _skip_ws(raw, key_at + len(key))
    if index >= len(raw) or raw[index] != ord(":"):
        raise ValueError(f"missing colon after {key.decode('ascii')}")
    return index + 1


def extract_record_metadata(raw: bytes) -> RecordMetadata:
    """Read the official top-level metadata without constructing ``tokens``."""

    data = raw.strip()
    if not data.startswith(b"{") or not data.endswith(b"}"):
        raise ValueError("record is not a JSON object")

    cluster_at = data.find(_CLUSTER_KEY)
    tokens_at = data.find(_TOKENS_KEY)
    count_at = data.find(_TOKEN_COUNT_KEY)
    if not (0 < cluster_at < tokens_at < count_at):
        raise ValueError("record does not use the pinned cluster/tokens/token_count layout")
    for key, at in (
        (_CLUSTER_KEY, cluster_at),
        (_TOKENS_KEY, tokens_at),
        (_TOKEN_COUNT_KEY, count_at),
    ):
        if data.find(key, at + len(key)) >= 0:
            raise ValueError(f"duplicate {key.decode('ascii')}")
    if data[1:cluster_at].strip():
        raise ValueError("unexpected field before cluster_id")

    cluster_id, cluster_end = _parse_integer(
        data, _colon_after(data, cluster_at, _CLUSTER_KEY)
    )
    if data[cluster_end:tokens_at].strip() != b",":
        raise ValueError("unexpected fields between cluster_id and tokens")

    token_value = _skip_ws(data, _colon_after(data, tokens_at, _TOKENS_KEY))
    if token_value >= len(data) or data[token_value] != ord("["):
        raise ValueError("tokens is not an array")
    close = data.rfind(b"]", token_value + 1, count_at)
    if close < 0 or data[close + 1:count_at].strip() != b",":
        raise ValueError("tokens array is malformed or followed by unexpected fields")

    token_count, count_end = _parse_integer(
        data, _colon_after(data, count_at, _TOKEN_COUNT_KEY)
    )
    if data[count_end:-1].strip():
        raise ValueError("unexpected field after token_count")
    if cluster_id not in config.ALL_CLUSTER_IDS:
        raise ValueError(f"cluster_id {cluster_id} is outside 1..20")
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return RecordMetadata(cluster_id, token_count)


def scan_work_item(
    item: WorkItem,
    source_file: SourceFile,
    reader_factory: Callable[[SourceFile], RangeReader],
) -> WorkItemMixture:
    reader = reader_factory(source_file)
    if reader.file_size() != source_file.size:
        raise RuntimeError(
            f"source size changed for {source_file.path}: "
            f"expected {source_file.size}, got {reader.file_size()}"
        )
    token_counts = {cluster: 0 for cluster in config.ALL_CLUSTER_IDS}
    document_counts = {cluster: 0 for cluster in config.ALL_CLUSTER_IDS}
    records = 0
    for record in iter_owned_records(item, reader):
        try:
            metadata = extract_record_metadata(record.raw)
        except ValueError as error:
            raise ValueError(
                f"invalid mixture metadata at {item.filename}:{record.record_start}: {error}"
            ) from error
        token_counts[metadata.cluster_id] += metadata.token_count
        document_counts[metadata.cluster_id] += 1
        records += 1
    return WorkItemMixture(
        item.index,
        item.range_end - item.range_start,
        records,
        token_counts,
        document_counts,
    )


def _ordered_parallel_results(
    plan: WorkPlan,
    *,
    start_index: int,
    reader_factory: Callable[[SourceFile], RangeReader],
    workers: int,
    max_in_flight: int,
) -> Iterator[WorkItemMixture]:
    if workers <= 0 or max_in_flight <= 0:
        raise ValueError("workers and max_in_flight must be positive")
    source_by_name = {source.path: source for source in plan.source_files}
    items = plan.work_items[start_index:]
    pending: dict[int, Future[WorkItemMixture]] = {}
    submitted = 0
    expected = start_index

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mixture-scan") as pool:
        while submitted < len(items) or pending:
            while submitted < len(items) and len(pending) < max_in_flight:
                item = items[submitted]
                source = source_by_name.get(item.filename)
                if source is None:
                    raise RuntimeError(f"work item references unknown source {item.filename}")
                pending[item.index] = pool.submit(scan_work_item, item, source, reader_factory)
                submitted += 1
            result = pending.pop(expected).result()
            if result.index != expected:
                raise RuntimeError("mixture scan result order changed")
            yield result
            expected += 1


def _empty_cluster_map() -> dict[str, int]:
    return {str(cluster): 0 for cluster in sorted(config.ALL_CLUSTER_IDS)}


def _initial_state(plan: WorkPlan) -> dict[str, object]:
    return {
        "schema_version": MIXTURE_SCAN_SCHEMA_VERSION,
        "dataset": plan.dataset,
        "revision": plan.revision,
        "source_glob": plan.source_glob,
        "work_plan_hash": plan.hash,
        "next_work_item_index": 0,
        "completed_work_items": 0,
        "source_bytes_covered": 0,
        "record_count": 0,
        "cluster_source_tokens": _empty_cluster_map(),
        "cluster_document_counts": _empty_cluster_map(),
        "complete": False,
    }


def _validate_state(raw: object, plan: WorkPlan) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("mixture progress must be a JSON object")
    state = dict(raw)
    expected = {
        "schema_version": MIXTURE_SCAN_SCHEMA_VERSION,
        "dataset": plan.dataset,
        "revision": plan.revision,
        "source_glob": plan.source_glob,
        "work_plan_hash": plan.hash,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"mixture progress {key} does not match the pinned work plan")
    next_index = state.get("next_work_item_index")
    if isinstance(next_index, bool) or not isinstance(next_index, int):
        raise ValueError("mixture progress has an invalid next_work_item_index")
    if not 0 <= next_index <= len(plan.work_items):
        raise ValueError("mixture progress next_work_item_index is outside the work plan")
    if state.get("completed_work_items") != next_index:
        raise ValueError("mixture progress completed_work_items is inconsistent")
    expected_bytes = sum(
        item.range_end - item.range_start for item in plan.work_items[:next_index]
    )
    if state.get("source_bytes_covered") != expected_bytes:
        raise ValueError("mixture progress source byte coverage is inconsistent")
    expected_keys = set(_empty_cluster_map())
    for field in ("cluster_source_tokens", "cluster_document_counts"):
        value = state.get(field)
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ValueError(f"mixture progress has an invalid {field}")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in value.values()
        ):
            raise ValueError(f"mixture progress has negative or non-integer {field}")
    return state


def _report_hash(report: Mapping[str, object]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(dict(report), exclude_keys=("report_sha256",))
    ).hexdigest()


def _load_completed_report(output_dir: Path, state: Mapping[str, object]) -> dict[str, object]:
    report = read_json(output_dir / MIXTURE_REPORT_FILENAME)
    if not isinstance(report, dict):
        raise ValueError("completed mixture report must be a JSON object")
    if report.get("report_sha256") != _report_hash(report):
        raise ValueError("completed mixture report hash mismatch")
    if sha256_file(output_dir / MIXTURE_WEIGHTS_FILENAME) != report.get("weights_sha256"):
        raise ValueError("completed mixture weights hash mismatch")
    if report.get("work_plan_hash") != state.get("work_plan_hash"):
        raise ValueError("completed mixture report belongs to a different work plan")
    return report


def _finish(output_dir: Path, plan: WorkPlan, state: dict[str, object]) -> dict[str, object]:
    if int(state["next_work_item_index"]) != len(plan.work_items):
        raise RuntimeError("cannot finish an incomplete mixture scan")
    total_source_bytes = sum(source.size for source in plan.source_files)
    if int(state["source_bytes_covered"]) != total_source_bytes:
        raise RuntimeError("mixture scan did not cover the exact pinned source byte size")

    token_counts = {
        int(cluster): int(count)
        for cluster, count in dict(state["cluster_source_tokens"]).items()
    }
    missing = [
        cluster for cluster in sorted(config.ALL_CLUSTER_IDS) if token_counts[cluster] <= 0
    ]
    if missing:
        raise RuntimeError(f"complete source scan found no tokens for clusters {missing}")

    weights = {
        str(cluster): token_counts[cluster]
        for cluster in sorted(config.ACCEPTED_CLUSTER_IDS)
    }
    weights_path = output_dir / MIXTURE_WEIGHTS_FILENAME
    write_json_atomic(weights_path, weights)
    weights_sha256 = sha256_file(weights_path)
    report: dict[str, object] = {
        "schema_version": MIXTURE_SCAN_SCHEMA_VERSION,
        "complete": True,
        "dataset": plan.dataset,
        "revision": plan.revision,
        "source_glob": plan.source_glob,
        "work_plan_hash": plan.hash,
        "source_files": [
            {"path": source.path, "size": source.size} for source in plan.source_files
        ],
        "source_bytes_scanned": total_source_bytes,
        "record_count": int(state["record_count"]),
        "all_cluster_source_tokens": {
            str(cluster): token_counts[cluster] for cluster in sorted(token_counts)
        },
        "all_cluster_document_counts": dict(state["cluster_document_counts"]),
        "all_source_tokens": sum(token_counts.values()),
        "accepted_cluster_ids": sorted(config.ACCEPTED_CLUSTER_IDS),
        "excluded_cluster_ids": sorted(config.EXCLUDED_CLUSTER_IDS),
        "accepted_source_tokens": sum(
            token_counts[cluster] for cluster in config.ACCEPTED_CLUSTER_IDS
        ),
        "conditioning_rule": (
            "Production weights are the exact released-corpus token totals for retained "
            "clusters, conditioned on excluding cluster 11."
        ),
        "weights_file": MIXTURE_WEIGHTS_FILENAME,
        "weights_sha256": weights_sha256,
    }
    report["report_sha256"] = _report_hash(report)
    write_json_atomic(output_dir / MIXTURE_REPORT_FILENAME, report)
    state.update(
        complete=True,
        weights_sha256=weights_sha256,
        report_sha256=report["report_sha256"],
    )
    write_json_atomic(output_dir / MIXTURE_PROGRESS_FILENAME, state)
    return report


def scan_mixture(
    output_dir: Path | str,
    plan: WorkPlan,
    reader_factory: Callable[[SourceFile], RangeReader],
    *,
    resume: bool = False,
    workers: int = 8,
    max_in_flight: int = 16,
    checkpoint_every_work_items: int = 4,
    simulate_crash_after_work_items: int | None = None,
) -> dict[str, object]:
    """Scan every pinned source record and emit exact code-free scheduler weights."""

    if checkpoint_every_work_items <= 0:
        raise ValueError("checkpoint_every_work_items must be positive")
    if simulate_crash_after_work_items is not None and simulate_crash_after_work_items <= 0:
        raise ValueError("simulate_crash_after_work_items must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / MIXTURE_PROGRESS_FILENAME

    if resume:
        state = _validate_state(read_json(progress_path), plan)
        if state.get("complete") is True:
            return _load_completed_report(output_dir, state)
    else:
        for path in (
            progress_path,
            output_dir / MIXTURE_REPORT_FILENAME,
            output_dir / MIXTURE_WEIGHTS_FILENAME,
        ):
            if path.exists():
                raise FileExistsError(
                    f"mixture calibration state already exists at {path}; "
                    "use --resume or a new directory"
                )
        state = _initial_state(plan)
        write_json_atomic(progress_path, state)

    token_counts = dict(state["cluster_source_tokens"])
    document_counts = dict(state["cluster_document_counts"])
    for result in _ordered_parallel_results(
        plan,
        start_index=int(state["next_work_item_index"]),
        reader_factory=reader_factory,
        workers=workers,
        max_in_flight=max_in_flight,
    ):
        for cluster in config.ALL_CLUSTER_IDS:
            key = str(cluster)
            token_counts[key] = int(token_counts[key]) + result.cluster_source_tokens[cluster]
            document_counts[key] = (
                int(document_counts[key]) + result.cluster_document_counts[cluster]
            )
        state.update(
            cluster_source_tokens=token_counts,
            cluster_document_counts=document_counts,
            record_count=int(state["record_count"]) + result.record_count,
            source_bytes_covered=int(state["source_bytes_covered"]) + result.source_bytes,
            next_work_item_index=result.index + 1,
            completed_work_items=result.index + 1,
        )
        completed = result.index + 1
        if completed % checkpoint_every_work_items == 0:
            write_json_atomic(progress_path, state)
            LOGGER.info(
                "mixture calibration: %d/%d work items, %d records, %.1f GiB covered",
                completed,
                len(plan.work_items),
                int(state["record_count"]),
                int(state["source_bytes_covered"]) / (1024**3),
            )
        if simulate_crash_after_work_items == completed:
            write_json_atomic(progress_path, state)
            raise RuntimeError("simulated mixture calibration interruption")

    write_json_atomic(progress_path, state)
    return _finish(output_dir, plan, state)

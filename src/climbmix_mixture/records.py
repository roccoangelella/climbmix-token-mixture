"""JSONL record ownership and structural validation over byte ranges.

Adjacent byte ranges must neither lose nor duplicate records.  For each work
item we recover complete JSONL lines around its boundaries, determine the
absolute starting byte offset of every record, and assign a record to this work
item only when its first byte lies in the half-open interval
``[range_start, range_end)``.  Fetching extra bytes on either side is allowed for
boundary reconstruction; ownership uses the record's absolute starting position.

Production validation is *structural only*.  No accepted document is decoded,
re-encoded, classified by an LLM, or passed through a code/quality filter.  The
only semantic signal is the numeric ``cluster_id``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

from . import config

from .bytesource import RangeReader
from .workplan import WorkItem


@dataclass(frozen=True)
class ParsedRecord:
    """A complete JSONL line plus its absolute starting byte offset."""

    record_start: int
    raw: bytes


@dataclass(frozen=True)
class ValidationResult:
    """Structural validation outcome.  Accepted iff ``valid`` is True."""

    valid: bool
    cluster_id: int | None
    tokens: tuple[int, ...] | None
    rejection_reason: str | None


def record_identity_str(revision: str, filename: str, record_start: int) -> str:
    """Stable permanent source identity for one record.

    The processing order of shuffled regions is intentionally NOT part of the
    identity, so train/validation assignment and deduplication are independent
    of the shuffle.
    """

    return f"{revision}:{filename}:{record_start}"


# ---------------------------------------------------------------------------
# Record stream within a work item
# ---------------------------------------------------------------------------


def iter_owned_records(
    item: WorkItem,
    reader: RangeReader,
    *,
    scan_chunk: int = config.BOUNDARY_SCAN_CHUNK_BYTES,
    fetch_chunk: int = config.FORWARD_FETCH_CHUNK_BYTES,
) -> Iterator[ParsedRecord]:
    """Yield every record whose first byte lies in ``[range_start, range_end)``.

    Boundary records split across range edges are reconstructed; only the
    absolute record-start byte determines ownership.  At most one
    currently-accumulating line is held in memory, so a single multi-gigabyte
    record does not require holding the whole region.
    """

    start, end = item.range_start, item.range_end
    file_size = reader.file_size()
    if scan_chunk <= 0 or fetch_chunk <= 0:
        raise ValueError("scan_chunk and fetch_chunk must be positive")
    if not (0 <= start < end <= file_size):
        raise ValueError(
            f"Work item {item.index} has range {start}:{end}, "
            f"outside source size {file_size}"
        )
    if end <= start:
        return
    floor = 0 if start == 0 else _find_record_floor(reader, start, scan_chunk)

    cursor = floor            # absolute offset of the next byte to fetch
    buf = b""
    line_start = floor        # absolute start of the line currently accumulated

    while True:
        newline = buf.find(b"\n")
        if newline != -1:
            line_bytes = _strip_cr(buf[:newline])
            if start <= line_start < end:
                yield ParsedRecord(record_start=line_start, raw=line_bytes)
            consumed = newline + 1
            buf = buf[consumed:]
            line_start += consumed
            if line_start >= end:
                return
            continue

        # No complete newline in the buffer yet.
        if line_start >= end:
            # Previous owned record (if any) was already yielded complete.
            return
        if line_start < start and cursor >= end:
            # The whole range lies interior to one foreign record.
            return

        if cursor >= file_size:
            # EOF: final record has no trailing newline.
            if buf:
                line_bytes = _strip_cr(buf)
                if start <= line_start < end:
                    yield ParsedRecord(record_start=line_start, raw=line_bytes)
            return
        chunk = reader.read_range(cursor, fetch_chunk)
        if not chunk:
            if buf:
                line_bytes = _strip_cr(buf)
                if start <= line_start < end:
                    yield ParsedRecord(record_start=line_start, raw=line_bytes)
            return
        buf += chunk
        cursor += len(chunk)


def _find_record_floor(reader: RangeReader, start: int, scan_chunk: int) -> int:
    """Return the absolute start of the record that contains ``start``.

    Scans backward in chunks for the last newline strictly before ``start``; the
    record containing ``start`` begins right after it.  Returns 0 if ``start``
    lies in the file's first record.
    """

    pos = start
    while pos > 0:
        lo = max(0, pos - scan_chunk)
        data = reader.read_range(lo, pos - lo)
        newline = data.rfind(b"\n")
        if newline != -1:
            return lo + newline + 1
        if lo == 0:
            return 0
        pos = lo
    return 0


def _strip_cr(line: bytes) -> bytes:
    """Drop a single trailing carriage return if CRLF separators were used."""

    if line.endswith(b"\r"):
        return line[:-1]
    return line


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def validate_record(record: ParsedRecord) -> ValidationResult:
    """Structural-only validation; never decodes the document to text."""

    try:
        parsed = json.loads(record.raw)
    except (ValueError, UnicodeDecodeError):
        return ValidationResult(False, None, None, "json_parse_error")
    if not isinstance(parsed, dict):
        return ValidationResult(False, None, None, "json_parse_error")

    raw_cluster = parsed.get("cluster_id")
    if raw_cluster is None:
        return ValidationResult(False, None, None, "missing_cluster_id")
    if isinstance(raw_cluster, bool) or not isinstance(raw_cluster, int):
        return ValidationResult(False, None, None, "cluster_id_not_integer")
    if not (1 <= raw_cluster <= 20):
        return ValidationResult(False, raw_cluster, None, "cluster_id_out_of_range")

    raw_tokens = parsed.get("tokens")
    if raw_tokens is None:
        return ValidationResult(False, raw_cluster, None, "missing_tokens")
    if not isinstance(raw_tokens, list):
        return ValidationResult(False, raw_cluster, None, "tokens_not_list")
    if not raw_tokens:
        return ValidationResult(False, raw_cluster, None, "tokens_empty")

    for value in raw_tokens:
        if isinstance(value, bool) or not isinstance(value, int):
            return ValidationResult(False, raw_cluster, None, "token_not_integer")
        if not (config.TOKEN_MIN <= value <= config.TOKEN_MAX):
            return ValidationResult(False, raw_cluster, None, "token_out_of_range")

    if "token_count" in parsed:
        token_count = parsed["token_count"]
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            return ValidationResult(False, raw_cluster, None, "token_count_not_integer")
        if token_count != len(raw_tokens):
            return ValidationResult(False, raw_cluster, None, "token_count_mismatch")

    return ValidationResult(True, raw_cluster, tuple(raw_tokens), None)

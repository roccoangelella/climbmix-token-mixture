"""Deterministic byte-range source access for the token-only corpus.

The production path never uses a sequential Hugging Face streaming iterator from
source row zero.  Instead it resolves the immutable source file list and exact
byte sizes once, then reads deterministic byte ranges with HTTP ``Range``
requests.  Tests inject the local :class:`LocalRangeReader` against in-memory
bytes or temporary files and never contact Hugging Face.

Record-boundary recovery (which crosses range edges to reconstruct complete JSONL
lines) lives in :mod:`climbmix_mixture.records`; this module only supplies raw bytes.
"""

from __future__ import annotations

import fnmatch
import http.client
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import config


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceFile:
    """One immutable source file: its path within the repo and its byte size."""

    path: str
    size: int


class RangeReader(Protocol):
    """Abstract byte-range access over one source file.

    Implementations stream bytes by absolute offset without holding the whole
    file in memory.  The HTTP implementation contacts Hugging Face; the local
    implementation backs tests with in-memory bytes or temporary files.
    """

    def file_size(self) -> int: ...

    def read_range(self, offset: int, length: int) -> bytes: ...


# ---------------------------------------------------------------------------
# HTTP range reader
# ---------------------------------------------------------------------------


class HttpRangeReader:
    """HTTP ``Range`` reader for one Hugging Face source file.

    Retries with exponential backoff on transient network errors.  Uses only the
    Python standard library so the production dependency footprint stays empty.
    """

    def __init__(self, source_file: SourceFile, repository: str, revision: str) -> None:
        self._source_file = source_file
        self._url = config.RESOLVE_URL_TEMPLATE.format(
            repository=repository, revision=revision, path=source_file.path
        )

    def file_size(self) -> int:
        return self._source_file.size

    def read_range(self, offset: int, length: int) -> bytes:
        if offset < 0 or length <= 0:
            raise ValueError(f"Invalid range request offset={offset} length={length}")
        if offset >= self._source_file.size:
            return b""
        # Clamp at EOF.  Callers intentionally use fixed fetch chunks, so the
        # final request for a file normally extends beyond its last byte.
        actual_length = min(length, self._source_file.size - offset)
        end = offset + actual_length - 1
        request = urllib.request.Request(
            self._url,
            headers={
                "Range": f"bytes={offset}-{end}",
                "Accept-Encoding": "identity",
                "User-Agent": config.HTTP_USER_AGENT,
            },
            method="GET",
        )
        last_error: Exception | None = None
        for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=config.HTTP_TIMEOUT_SECONDS
                ) as response:
                    data = response.read()
                if response.status != 206:
                    raise RuntimeError(
                        f"Expected 206 Partial Content for {self._source_file.path} "
                        f"bytes={offset}-{end}, got {response.status}"
                    )
                content_range = response.headers.get("Content-Range", "")
                expected = f"bytes {offset}-{end}/{self._source_file.size}"
                if content_range != expected:
                    raise RuntimeError(
                        f"Content-Range mismatch for {self._source_file.path}: "
                        f"got {content_range!r}, expected {expected!r}"
                    )
                if len(data) != actual_length:
                    raise RuntimeError(
                        f"Short read for {self._source_file.path}: wanted "
                        f"{actual_length}, got {len(data)}"
                    )
                return data
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.IncompleteRead,
                OSError,
                RuntimeError,
            ) as error:
                last_error = error
                if attempt >= config.HTTP_MAX_RETRIES:
                    break
                delay = min(
                    config.HTTP_BACKOFF_MAX_SECONDS,
                    config.HTTP_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                )
                LOGGER.warning(
                    "Range read attempt %d/%d failed for %s bytes=%d-%d: %s: %s; retrying in %.1fs",
                    attempt,
                    config.HTTP_MAX_RETRIES,
                    self._source_file.path,
                    offset,
                    end,
                    type(error).__name__,
                    error,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Range read failed for {self._source_file.path} bytes={offset}-{end} after "
            f"{config.HTTP_MAX_RETRIES} attempts: {type(last_error).__name__}: {last_error}"
        )


# ---------------------------------------------------------------------------
# Local range reader (tests / synthetic sources)
# ---------------------------------------------------------------------------


class LocalRangeReader:
    """Range reader backed by an in-memory byte string or a file on disk.

    ``data`` may be ``bytes`` (kept in memory, fine for small synthetic files)
    or a :class:`pathlib.Path` to a file (read with ``pread``-style seeking so
    large synthetic files never need to be loaded whole).
    """

    def __init__(self, data: bytes | Path) -> None:
        self._data = data
        if isinstance(data, Path):
            self._size = data.stat().st_size
        else:
            self._size = len(data)

    def file_size(self) -> int:
        return self._size

    def read_range(self, offset: int, length: int) -> bytes:
        if offset < 0 or length <= 0:
            raise ValueError(f"Invalid range request offset={offset} length={length}")
        end = min(offset + length, self._size)
        if offset >= self._size:
            return b""
        if isinstance(self._data, Path):
            with self._data.open("rb") as handle:
                handle.seek(offset)
                return handle.read(end - offset)
        return self._data[offset:end]


# ---------------------------------------------------------------------------
# Source file-list resolution
# ---------------------------------------------------------------------------


def list_source_files(repository: str, revision: str) -> list[SourceFile]:
    """Resolve the immutable root source file list and exact byte sizes.

    Uses the Hugging Face tree API (metadata only; never downloads content).
    Only root files matching :data:`config.SOURCE_DATA_GLOB` are returned, which
    excludes ``climbmix_small`` and every other subdirectory.
    """

    url = (
        config.TREE_URL_TEMPLATE.format(repository=repository, revision=revision)
        + "?recursive=false&expand=false"
    )
    entries: list[dict[str, object]] = []
    cursor: str | None = None
    while True:
        separator = "&" if "?" in url else "?"
        page_url = (
            url
            if cursor is None
            else f"{url}{separator}cursor={urllib.parse.quote(cursor)}"
        )
        payload = _read_json_with_retries(page_url, description="source tree")
        if isinstance(payload, list):
            page = payload
        elif isinstance(payload, dict) and isinstance(payload.get("tree"), list):
            page = payload["tree"]
        else:
            raise RuntimeError(
                f"Unexpected source tree response type for {repository}@{revision}"
            )
        for entry in page:
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Unexpected source tree entry for {repository}@{revision}"
                )
            if entry.get("type") != "file":
                continue
            path = entry.get("path")
            if not isinstance(path, str):
                continue
            if not _matches_glob(path):
                continue
            size = entry.get("size")
            if not isinstance(size, int) or size <= 0:
                raise RuntimeError(f"Source file {path!r} has no valid size in tree API")
            entries.append({"path": path, "size": size})
        cursor = payload.get("cursor") if isinstance(payload, dict) else None
        if not cursor:
            break

    files = sorted((SourceFile(path=str(e["path"]), size=int(e["size"])) for e in entries),
                   key=lambda f: f.path)
    if not files:
        raise RuntimeError(
            f"No root files matching {config.SOURCE_DATA_GLOB!r} found at "
            f"{repository}@{revision}"
        )
    LOGGER.info("Resolved %d root source files (%.1f GiB total) at %s@%s",
                len(files), sum(f.size for f in files) / (1024 ** 3), repository, revision)
    return files


def _read_json_with_retries(url: str, *, description: str) -> object:
    """Read small HTTP JSON metadata with the production retry policy."""

    last_error: Exception | None = None
    for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
        request = urllib.request.Request(
            url, headers={"User-Agent": config.HTTP_USER_AGENT}
        )
        try:
            with urllib.request.urlopen(
                request, timeout=config.HTTP_TIMEOUT_SECONDS
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            OSError,
            json.JSONDecodeError,
        ) as error:
            last_error = error
            if attempt >= config.HTTP_MAX_RETRIES:
                break
            delay = min(
                config.HTTP_BACKOFF_MAX_SECONDS,
                config.HTTP_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
            )
            LOGGER.warning(
                "%s request attempt %d/%d failed: %s: %s; retrying in %.1fs",
                description,
                attempt,
                config.HTTP_MAX_RETRIES,
                type(error).__name__,
                error,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Could not read {description} after {config.HTTP_MAX_RETRIES} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def _matches_glob(path: str) -> bool:
    """Match only root files; reject anything inside a subdirectory."""

    if "/" in path:
        return False
    return _glob_match(path, config.SOURCE_DATA_GLOB)


def _glob_match(name: str, pattern: str) -> bool:
    """Minimal single-segment glob supporting ``*`` and ``?`` only."""

    return fnmatch.fnmatchcase(name, pattern)


def make_http_reader(source_file: SourceFile, repository: str, revision: str) -> HttpRangeReader:
    """Factory used by production; kept explicit so tests can swap it out."""

    return HttpRangeReader(source_file, repository, revision)

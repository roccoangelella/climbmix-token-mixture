"""Atomic JSON I/O and stable hashing helpers."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json_bytes(payload: Any, *, exclude_keys: tuple[str, ...] = ()) -> bytes:
    filtered = {k: v for k, v in payload.items() if k not in exclude_keys} if exclude_keys else payload
    return json.dumps(filtered, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            hasher.update(block)
    return hasher.hexdigest()

"""Offline verification of the published calibration artifacts."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from . import config
from .storage import canonical_json_bytes, read_json, sha256_file
from .workplan import load_work_plan


def verify(results_dir: Path) -> dict[str, object]:
    verification = read_json(results_dir / "verification.json")
    report = read_json(results_dir / "mixture_report.json")
    progress = read_json(results_dir / "mixture_progress.json")
    weights_path = results_dir / "climbmix_code_free_weights.json"
    weights = read_json(weights_path)
    plan = load_work_plan(results_dir / "work_plan.json")

    artifact_hashes = verification["artifact_sha256"]
    for name, expected in artifact_hashes.items():
        actual = sha256_file(results_dir / name)
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch: {actual} != {expected}")

    canonical_report = hashlib.sha256(
        canonical_json_bytes(report, exclude_keys=("report_sha256",))
    ).hexdigest()
    if canonical_report != report["report_sha256"]:
        raise ValueError("report canonical self-hash mismatch")
    if plan.hash != report["work_plan_hash"]:
        raise ValueError("work-plan/report hash mismatch")
    if progress["report_sha256"] != report["report_sha256"]:
        raise ValueError("progress/report self-hash mismatch")
    if progress["weights_sha256"] != report["weights_sha256"]:
        raise ValueError("progress/report weights hash mismatch")
    if sha256_file(weights_path) != report["weights_sha256"]:
        raise ValueError("weights file hash mismatch")

    source_bytes = sum(source.size for source in plan.source_files)
    if source_bytes != report["source_bytes_scanned"]:
        raise ValueError("source byte total mismatch")
    if report["source_files"] != [
        {"path": source.path, "size": source.size} for source in plan.source_files
    ]:
        raise ValueError("source file list mismatch")
    if progress["completed_work_items"] != len(plan.work_items):
        raise ValueError("work-item completion mismatch")
    if progress["source_bytes_covered"] != source_bytes:
        raise ValueError("progress source-byte coverage mismatch")

    all_tokens = {int(k): int(v) for k, v in report["all_cluster_source_tokens"].items()}
    all_docs = {int(k): int(v) for k, v in report["all_cluster_document_counts"].items()}
    if set(all_tokens) != set(config.ALL_CLUSTER_IDS) or min(all_tokens.values()) <= 0:
        raise ValueError("invalid all-cluster token totals")
    if set(all_docs) != set(config.ALL_CLUSTER_IDS) or min(all_docs.values()) <= 0:
        raise ValueError("invalid all-cluster document totals")
    if sum(all_tokens.values()) != report["all_source_tokens"]:
        raise ValueError("all-source token sum mismatch")
    expected_weights = {str(c): all_tokens[c] for c in sorted(config.ACCEPTED_CLUSTER_IDS)}
    if weights != expected_weights:
        raise ValueError("accepted weights do not match report totals")
    if sum(expected_weights.values()) != report["accepted_source_tokens"]:
        raise ValueError("accepted-source token sum mismatch")
    if "11" in weights or all_tokens[11] <= 0:
        raise ValueError("cluster-11 inclusion/exclusion contract mismatch")

    return {
        "status": "PASS",
        "source_files": len(plan.source_files),
        "work_items": len(plan.work_items),
        "records": report["record_count"],
        "all_source_tokens": report["all_source_tokens"],
        "accepted_source_tokens": report["accepted_source_tokens"],
        "weights_sha256": report["weights_sha256"],
        "report_self_hash": report["report_sha256"],
        "work_plan_self_hash": plan.hash,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    result = verify(args.results_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

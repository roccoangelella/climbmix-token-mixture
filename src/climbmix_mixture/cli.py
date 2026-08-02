"""CLI for recovering exact Nemotron-ClimbMix cluster token weights."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import config
from .bytesource import list_source_files, make_http_reader
from .calibration import scan_mixture
from .workplan import build_work_plan, load_work_plan, save_work_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan the complete pinned Nemotron-ClimbMix release and derive exact "
            "cluster-token weights conditioned on excluding cluster 11."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-in-flight-work-items", type=int, default=16)
    parser.add_argument("--checkpoint-every-work-items", type=int, default=4)
    parser.add_argument("--region-bytes", type=int, default=config.REGION_BYTES)
    parser.add_argument(
        "--simulate-crash-after-work-items",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        if args.workers <= 0:
            raise ValueError("--workers must be positive")
        if args.max_in_flight_work_items <= 0:
            raise ValueError("--max-in-flight-work-items must be positive")
        if args.checkpoint_every_work_items <= 0:
            raise ValueError("--checkpoint-every-work-items must be positive")
        if args.region_bytes <= 0:
            raise ValueError("--region-bytes must be positive")

        output_dir = args.output_dir.resolve()
        plan_path = output_dir / config.WORK_PLAN_FILENAME
        if args.resume:
            plan = load_work_plan(plan_path)
            if plan.dataset != config.DATASET_REPOSITORY:
                raise ValueError("saved work plan belongs to a different dataset")
            if plan.revision != config.DATASET_REVISION:
                raise ValueError("saved work plan belongs to a different source revision")
            if plan.source_glob != config.SOURCE_DATA_GLOB:
                raise ValueError("saved work plan uses a different source glob")
        else:
            if plan_path.exists():
                raise FileExistsError(
                    f"work plan already exists at {plan_path}; use --resume or a new output directory"
                )
            source_files = list_source_files(config.DATASET_REPOSITORY, config.DATASET_REVISION)
            plan = build_work_plan(
                source_files,
                region_bytes=args.region_bytes,
                seed=config.SELECTION_SEED,
                repository=config.DATASET_REPOSITORY,
                revision=config.DATASET_REVISION,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            save_work_plan(plan_path, plan)

        report = scan_mixture(
            output_dir,
            plan,
            lambda source: make_http_reader(
                source, config.DATASET_REPOSITORY, config.DATASET_REVISION
            ),
            resume=args.resume,
            workers=args.workers,
            max_in_flight=args.max_in_flight_work_items,
            checkpoint_every_work_items=args.checkpoint_every_work_items,
            simulate_crash_after_work_items=args.simulate_crash_after_work_items,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - concise CLI boundary
        sys.stderr.write(f"mixture calibration error: {type(error).__name__}: {error}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Plan a per-tensor precision map from a calibration report, under a budget.

The plan keeps every tensor packed and spends the remaining budget promoting
the tensors whose quantization costs the most logit error per extra byte. It
is a ranking heuristic over measured numbers, not a quality guarantee.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.planner.memory import parse_memory_size
from compiler.precision_map import PrecisionMap, select_precision

MAX_REPORT_BYTES = 64 * 1024 * 1024


def read_report(path):
    path = Path(path)
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("Calibration report exceeds the supported size")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Calibration report must be a JSON object")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    planner = commands.add_parser("plan", help="Select codecs under a byte budget")
    planner.add_argument("--calibration", required=True, type=Path)
    planner.add_argument("--budget", required=True, help="Weight byte budget, e.g. 64MiB")
    planner.add_argument("--out", type=Path, help="Write the precision map here")

    shower = commands.add_parser("show", help="Validate a precision map and summarize it")
    shower.add_argument("map", type=Path)

    args = parser.parse_args(argv)
    if args.command == "plan":
        report = read_report(args.calibration)
        precision = select_precision(report, parse_memory_size(args.budget))
        encoded = precision.to_json()
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return 0
    precision = PrecisionMap.from_json(Path(args.map).read_text(encoding="utf-8"))
    summary = {"tensors": len(precision.codecs), "dense_tensors": list(precision.dense_tensors),
               "policy_id": precision.to_dict()["policy_id"], "provenance": precision.provenance}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Precision planning failed: {error}", file=sys.stderr)
        raise SystemExit(1)

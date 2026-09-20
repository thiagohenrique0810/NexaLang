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

from compiler.planner.compression import CompressionPlanner
from compiler.planner.memory import parse_memory_size
from compiler.precision_map import PrecisionMap

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
    planner.add_argument("--budget", help="Weight byte budget, e.g. 64MiB")
    planner.add_argument("--max-rmse", type=float,
                         help="Estimated logit RMSE ceiling; plans the cheapest map that reaches it")
    planner.add_argument("--out", type=Path, help="Write the precision map here")
    planner.add_argument("--cost", choices=("payload", "physical"), default="payload",
                         help="Byte count to optimize: the codec payload, or what the "
                              "tensor files actually hold (default: payload)")
    planner.add_argument("--max-decode-ns", type=int,
                         help="Ceiling on the measured nanoseconds to decode every weight once; "
                              "needs a calibration report with a 'decode_time' block")
    planner.add_argument("--axes", type=Path,
                         help="Write the axis provenance here: what the planner considered and "
                              "what it refused to consider")

    shower = commands.add_parser("show", help="Validate a precision map and summarize it")
    shower.add_argument("map", type=Path)

    args = parser.parse_args(argv)
    if args.command == "plan":
        if (args.budget is None) == (args.max_rmse is None):
            parser.error("pass exactly one of --budget or --max-rmse")
        report = read_report(args.calibration)
        plan = CompressionPlanner(report, cost=args.cost, max_decode_ns=args.max_decode_ns,
                                  max_bytes=parse_memory_size(args.budget) if args.budget else None,
                                  max_rmse=args.max_rmse).plan()
        encoded = plan.to_json()
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(encoded + "\n", encoding="utf-8")
        if args.axes is not None:
            args.axes.parent.mkdir(parents=True, exist_ok=True)
            args.axes.write_text(json.dumps(plan.axes, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
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

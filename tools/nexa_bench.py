#!/usr/bin/env python3
"""Benchmark native packed Q4 tiles with an explicit CPU buffer budget.

No model download, PyTorch, or GPU inference is performed by this first milestone.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.nexapack.executor import run_packed_matmul
from runtime.nexapack.format import write_q4_matrix


def write_report(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--pack", type=Path, help="A standalone NexaPack V1 matrix")
    sources.add_argument("--bundle", type=Path, help="A model bundle directory")
    parser.add_argument("--tensor", help="Matrix name or alias inside --bundle")
    parser.add_argument("--generate-demo", action="store_true", help="Create deterministic Q4 weights; requires a new output path")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tile-rows", type=int, default=64)
    parser.add_argument("--memory-budget", default="512MiB")
    parser.add_argument("--reserve", default="0B")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    if args.bundle is not None and not args.tensor:
        parser.error("--bundle requires --tensor")
    if args.pack is not None and args.tensor is not None:
        parser.error("--tensor requires --bundle")
    if args.bundle is not None and args.generate_demo:
        parser.error("--generate-demo only creates standalone --pack matrices")
    try:
        source = args.bundle or args.pack
        # Reports must never replace the weights they describe (including aliases).
        paths = [p for p in (source, args.report, args.csv) if p is not None]
        for i, path in enumerate(paths):
            for other in paths[:i]:
                if path.resolve() == other.resolve() or (path.exists() and other.exists() and path.samefile(other)):
                    raise ValueError("Pack, JSON report, and CSV report must use distinct paths")
        if args.bundle is not None:
            for report_path in (args.report, args.csv):
                if report_path is not None and report_path.resolve().is_relative_to(args.bundle.resolve()):
                    raise ValueError("Reports must be written outside the immutable model bundle")
        if args.generate_demo:
            if args.pack.exists():
                raise ValueError("Demo output already exists; choose a new --pack path")
            if args.rows <= 0 or args.cols <= 0:
                raise ValueError("rows and cols must be positive")
            # Each generator binds its row while consumed by the streaming writer.
            def rows():
                for r in range(args.rows):
                    yield (((r * 13 + c * 7) % 29 - 14) / 16.0 for c in range(args.cols))
            write_q4_matrix(args.pack, args.rows, args.cols, args.group_size, rows())
        report = run_packed_matmul(source, batch=args.batch, tile_rows=args.tile_rows,
                                  memory_budget=args.memory_budget, reserve_bytes=args.reserve,
                                  verify=args.verify, tensor_name=args.tensor)
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.report:
            write_report(args.report, rendered)
        if args.csv:
            record = {"backend": report["backend"], **report["shape"],
                      "tile_rows": report["tile_rows"],
                      "budget_bytes": report["memory"]["budget_bytes"],
                      "managed_buffers_peak_bound_bytes": report["memory"]["managed_buffers_peak_bound_bytes"],
                      "file_payload_bytes_read": report["io"]["file_payload_bytes_read"],
                      **report["timing"], "verified": report["validation"]["verified"]}
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
            write_report(args.csv, stream.getvalue())
        print(rendered, end="")
        return 0
    except (OSError, ValueError, ArithmeticError, RuntimeError, MemoryError, subprocess.CalledProcessError) as exc:
        print(f"Nexa benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

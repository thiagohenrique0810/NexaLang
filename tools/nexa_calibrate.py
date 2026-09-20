#!/usr/bin/env python3
"""Measure what quantizing each tensor costs: bytes, codec error and logits.

Builds a dense reference bundle and one bundle per packed codec from the same
checkpoint, then runs the model once per (tensor, codec) with only that tensor
packed. Sensitivity therefore costs O(tensors x codecs) executions; --tensor
and --codec restrict it to the ones in question.

The logit delta is measured on the weights given. On an untrained checkpoint it
shows how error propagates through the graph, not answer quality: perplexity
needs a trained model (LLM.04b).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.calibration import (
    MAX_CALIBRATION_TENSORS, build_variant, logit_delta, quantization_error, row_statistics,
)
from compiler.importers.llama import import_llama_checkpoint
from compiler.importers.safetensors import SafeTensorCheckpoint
from runtime.nexapack.bundle import ModelBundleReader
from runtime.nexapack.transformer import TransformerSession


def token_list(value):
    tokens = [int(item) for item in value.split(",") if item != ""]
    if not tokens:
        raise argparse.ArgumentTypeError("Provide at least one token id")
    return tokens


def run_model(path, tokens, *, memory_budget, tile_rows):
    with TransformerSession(path, memory_budget=memory_budget, max_sequence_length=len(tokens),
                            tile_rows=tile_rows) as session:
        logits = session.prefill(list(tokens))
        report = session.report()
    return logits, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Local Llama Safetensors checkpoint to calibrate")
    parser.add_argument("--tokens", required=True, type=token_list,
                        help="Prompt ids used to measure the logit delta")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--block-rows", type=int, default=64)
    parser.add_argument("--tile-rows", type=int, default=32)
    parser.add_argument("--memory-budget", default="512MiB")
    parser.add_argument("--tensor", action="append", dest="tensors",
                        help="Measure only these tensors; repeat per tensor")
    parser.add_argument("--codec", action="append", dest="codecs", choices=("q3", "q4", "q8"),
                        help="Packed codecs to measure (default: q3, q4 and q8)")
    parser.add_argument("--static-only", action="store_true",
                        help="Report distribution and codec error without executing the model")
    parser.add_argument("--work-dir", type=Path, help="Keep intermediate bundles here instead of a temp dir")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.resolve(strict=True)
    holder = None if args.work_dir else tempfile.TemporaryDirectory(prefix="nexa-calibration-")
    work = Path(args.work_dir) if args.work_dir else Path(holder.name)
    work.mkdir(parents=True, exist_ok=True)
    try:
        source = SafeTensorCheckpoint(checkpoint)
        shapes = {name: tuple(shape) for name, shape in source.tensor_shapes.items()}
        matrices = sorted(name for name, shape in shapes.items() if len(shape) == 2)
        selected = args.tensors or matrices
        unknown = sorted(set(selected) - set(matrices))
        if unknown:
            raise ValueError(f"Not a matrix of this checkpoint: {', '.join(unknown)}")
        if len(selected) > MAX_CALIBRATION_TENSORS:
            raise ValueError("Too many tensors requested for one calibration run")

        codecs = tuple(dict.fromkeys(args.codecs or ("q3", "q4", "q8")))
        tensors = []
        for name in selected:
            statistics = row_statistics(source.iter_rows(name))
            dense_bytes = statistics["values"] * 4
            measured = {codec: {"quantization": quantization_error(source.iter_rows(name),
                                                                   args.group_size, codec)}
                        for codec in codecs}
            tensors.append({"name": name, "shape": list(shapes[name]),
                            "dense_bytes": dense_bytes, "statistics": statistics,
                            "codecs": measured})

        report = {"checkpoint": str(checkpoint), "tokens": args.tokens,
                  "group_size": args.group_size, "tensors": tensors,
                  "measured_codecs": list(codecs), "sensitivity_measured": not args.static_only,
                  "scope": ("static distribution and per-codec round-trip error"
                            if args.static_only else
                            "per-codec error plus the logit delta of packing one tensor at a time"),
                  "quality_measured": False,
                  "quality_note": "logit deltas on these weights; perplexity needs a trained checkpoint"}

        def rank(entry):
            """Worst case across measured codecs; the cheapest one bounds it."""
            return max(item.get("sensitivity", {}).get("rmse", 0.0) for item in entry["codecs"].values())

        if not args.static_only:
            dense = work / "reference-dense"
            if not dense.exists():
                import_llama_checkpoint(checkpoint, dense, group_size=args.group_size,
                                        block_rows=args.block_rows,
                                        tensor_codecs={name: "f32" for name in matrices})
            reference, dense_report = run_model(dense, args.tokens, memory_budget=args.memory_budget,
                                                tile_rows=args.tile_rows)
            report["reference"] = {"codec": "RAW_F32_MATRIX",
                                   "logits_sha256": dense_report["logits_sha256"]}
            report["all_packed"] = {}
            for codec in codecs:
                packed = work / f"reference-{codec}"
                if not packed.exists():
                    import_llama_checkpoint(checkpoint, packed, group_size=args.group_size,
                                            block_rows=args.block_rows,
                                            tensor_codecs={name: codec for name in matrices})
                packed_logits, packed_report = run_model(packed, args.tokens,
                                                         memory_budget=args.memory_budget,
                                                         tile_rows=args.tile_rows)
                report["all_packed"][codec] = {
                    "logits_sha256": packed_report["logits_sha256"],
                    "sensitivity": logit_delta(reference, packed_logits)}
                with ModelBundleReader(packed) as bundle:
                    sizes = {item["name"]: item["packed_payload_bytes"]
                             for item in bundle.inspect()["tensors"]}
                for entry in tensors:
                    name = entry["name"]
                    variant = work / f"variant-{codec}-{name.replace('.', '_')}"
                    if variant.exists():
                        shutil.rmtree(variant)
                    build_variant(dense, packed, name, variant)
                    try:
                        logits, _ = run_model(variant, args.tokens, memory_budget=args.memory_budget,
                                              tile_rows=args.tile_rows)
                    finally:
                        if not args.work_dir:
                            shutil.rmtree(variant, ignore_errors=True)
                    measurement = entry["codecs"][codec]
                    measurement["packed_bytes"] = sizes[name]
                    measurement["saved_bytes"] = entry["dense_bytes"] - sizes[name]
                    measurement["sensitivity"] = logit_delta(reference, logits)
                    saved = max(measurement["saved_bytes"], 1)
                    # Logit error per byte this codec saves on this tensor: the
                    # ranking a precision map needs, not a quality claim.
                    measurement["cost_per_saved_kib"] = (measurement["sensitivity"]["rmse"] /
                                                         (saved / 1024))
            tensors.sort(key=rank, reverse=True)
            report["most_sensitive"] = [entry["name"] for entry in tensors[:5]]
            report["ranking"] = ("tensors sorted by the worst logit RMSE any measured codec "
                                 "causes on that tensor alone")
        else:
            tensors.sort(key=lambda item: max(codec["quantization"]["relative_rmse"]
                                              for codec in item["codecs"].values()), reverse=True)

        encoded = json.dumps(report, indent=2, sort_keys=True)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return 0
    finally:
        if holder is not None:
            holder.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, ArithmeticError) as error:
        print(f"Calibration failed: {error}", file=sys.stderr)
        raise SystemExit(1)

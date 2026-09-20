#!/usr/bin/env python3
"""Rewrite a ModelGraph under verification, or report where it could stream.

Both subcommands read the same graph: a serialized ModelGraph, a ModelConfig
JSON, or a named model from a .nxl definition lowered at a sequence length.
Neither executes the model. `rewrite` measures every rewrite it accepts with an
independent evaluator; `regions` only reports analysis and the bytes it implies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.graph_algebra import GraphEquivalenceVerifier, default_passes, run_passes
from compiler.model_config import ModelConfig
from compiler.model_definition import compile_model_definition
from compiler.model_ir import ModelGraph
from compiler.model_lowering import lower_model
from compiler.planner.streaming import detect_streaming_regions, streaming_summary

MAX_SOURCE_BYTES = 64 * 1024 * 1024


def read_text(path):
    path = Path(path)
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError(f"{path} exceeds the supported source size")
    return path.read_text(encoding="utf-8")


def add_source_arguments(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph", type=Path, help="A serialized ModelGraph")
    source.add_argument("--config", type=Path, help="A ModelConfig JSON to lower")
    source.add_argument("--definition", type=Path, help="A .nxl model definition to lower")
    parser.add_argument("--model", help="Model name inside a --definition file")
    parser.add_argument("--sequence-length", type=int, default=4,
                        help="Tokens to lower for when the source is a config (default: 4)")


def load_graph(args, parser):
    if args.graph is not None:
        if args.model is not None:
            parser.error("--model applies to --definition, not to a serialized graph")
        return ModelGraph.from_json(read_text(args.graph))
    if args.config is not None:
        config = ModelConfig.from_json(read_text(args.config))
    else:
        models = compile_model_definition(read_text(args.definition))
        if args.model is None:
            if len(models) != 1:
                parser.error(f"--model is required; the file declares {', '.join(sorted(models))}")
            args.model = next(iter(models))
        if args.model not in models:
            parser.error(f"unknown model {args.model}; the file declares {', '.join(sorted(models))}")
        config = models[args.model]
    return lower_model(config, args.sequence_length)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    rewriter = commands.add_parser("rewrite", help="Run the algebraic passes and report every site")
    add_source_arguments(rewriter)
    rewriter.add_argument("--out", type=Path, help="Write the rewritten graph here")
    rewriter.add_argument("--no-verify", action="store_true",
                          help="Skip equivalence measurement; the report then proves nothing")

    regions = commands.add_parser("regions", help="Detect streaming regions and size them")
    add_source_arguments(regions)
    regions.add_argument("--tile-rows", type=int, default=1,
                         help="Rows per tile used to size interior buffers (default: 1)")

    args = parser.parse_args(argv)
    graph = load_graph(args, parser)
    if args.command == "rewrite":
        verifier = None if args.no_verify else GraphEquivalenceVerifier()
        rewritten, report = run_passes(graph, default_passes(), verifier=verifier)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rewritten.to_json() + "\n", encoding="utf-8")
        print(report.to_json())
        return 0
    detected = detect_streaming_regions(graph, tile_rows=args.tile_rows)
    summary = dict(streaming_summary(graph, detected, args.tile_rows))
    print(json.dumps({"graph": graph.name, "summary": summary,
                      "regions": [region.to_dict() for region in detected]},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Graph tooling failed: {error}", file=sys.stderr)
        raise SystemExit(1)

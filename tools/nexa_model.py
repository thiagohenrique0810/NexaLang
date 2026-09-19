#!/usr/bin/env python3
"""Lower NexaLM declarations to metadata and optional stateless prefill graphs.

This model-definition frontend is separate from nxc. It does not train models
or run the emitted graph. --sequence-length includes operators and lifetimes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.model_definition import compile_model_definition
from compiler.model_lowering import derive_activation_requests, lower_model
from tools.nexa_bench import write_report


def compile_definitions(path, sequence_length=None):
    path = Path(path)
    with path.open("rb") as source:
        raw = source.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Model definition exceeds 1 MiB")
    definitions = compile_model_definition(raw.decode("utf-8"))
    result = {
        "schema_version": 1, "format": "NexaModelDefinitions",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "structural architecture and tensor schema; no executable Transformer graph",
        "models": {name: {
            "config": config.to_dict(),
            "parameter_count": config.parameter_count(),
            "tensor_shapes": {tensor: list(shape) for tensor, shape in config.required_tensor_shapes().items()},
            "aliases": config.tensor_aliases(),
        } for name, config in definitions.items()},
    }
    if sequence_length is not None:
        result["scope"] = "stateless single-sequence prefill ModelIR and activation lifetimes; no KV cache"
        result["sequence_length"] = sequence_length
        for name, config in definitions.items():
            graph = lower_model(config, sequence_length)
            result["models"][name]["model_ir"] = graph.to_dict()
            result["models"][name]["activation_requests"] = [
                request.to_dict() for request in derive_activation_requests(graph)]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sequence-length", type=int, help="Include a prefill graph and derived activation lifetimes")
    args = parser.parse_args(argv)
    try:
        if args.out is not None and (args.input.resolve() == args.out.resolve() or
                (args.out.exists() and args.input.samefile(args.out))):
            raise ValueError("Output must not replace the model definition")
        result = compile_definitions(args.input, sequence_length=args.sequence_length)
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.out:
            write_report(args.out, rendered)
        print(rendered, end="")
        return 0
    except (OSError, ValueError, ArithmeticError) as exc:
        print(f"Model definition failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate a deterministic SYNTHETIC Llama-shaped checkpoint at real scale.

The output is exactly what `compiler/importers/llama.py` already imports:
Safetensors shards plus `config.json`, with an `origin.json` beside them that
declares the checkpoint synthetic. Weights are pseudorandom, drawn from the
initialization a real model starts training from -- never from training.

It exists so the pipeline can be measured on 100M+ parameters without
downloading anyone's weights: bytes per codec, arena peak, wall time, and
whether the whole thing fits 512 MB. It measures physics, never quality.
No tensor is materialized whole: rows are generated, packed and written in
bounded chunks, so a 480 MiB checkpoint costs kilobytes of process memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.importers.safetensors import MAX_HEADER_BYTES, MAX_SHARDS
from compiler.model_config import ModelConfig
from compiler.model_definition import compile_model_definition
from compiler.planner.memory import parse_memory_size

TOOL = "tools/nexa_synth.py"
TOOL_VERSION = 1
FORMAT = "NexaSyntheticCheckpoint"
SCHEMA_VERSION = 1
MAX_DEFINITION_BYTES = 1024 * 1024
# Values are emitted in bounded runs so no row list and no packed buffer grows
# with the tensor. 16384 float32 values are 64 KiB of scratch.
CHUNK_VALUES = 16384
DEFAULT_MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024
MIN_MAX_SHARD_BYTES = 4096
# Irwin-Hall order. Twelve uniforms sum to variance exactly one, so the sample
# needs no scaling correction, and the support is exactly [-6, +6].
_UNIFORMS = 12
_DTYPES = {"f32": ("F32", 4, "f"), "f16": ("F16", 2, "e")}
# Llama's `initializer_range`. Embeddings are looked up, not multiplied into an
# activation, so they do not get the fan-in scaling the projections get.
EMBEDDING_STD = 0.02


class SyntheticCheckpointError(ValueError):
    """Invalid synthetic-checkpoint request or destination."""


def _tensor_seed(seed, name):
    """Derive one independent stream per tensor, independent of write order.

    Seeding from the tensor name rather than from a running counter is what
    makes a shard's bytes depend only on the tensor it holds: reordering
    shards, or generating one tensor alone, reproduces the same payload.
    """
    digest = hashlib.sha256(f"{FORMAT}\x00{SCHEMA_VERSION}\x00{seed}\x00{name}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def _normals(count, seed, name, std):
    """Yield `count` N(0, std^2) samples, truncated at six standard deviations.

    The sampler is the sum of twelve uniforms minus six: only IEEE-754 addition
    and multiplication, so the same seed yields the same doubles on any
    conforming platform. Box-Muller would route the value through libm's log
    and cos, whose last bit is not guaranteed portable, and the point of this
    tool is that a measurement can be reproduced somewhere else.
    """
    uniform = random.Random(_tensor_seed(seed, name)).random
    for _ in range(count):
        total = 0.0
        for _ in range(_UNIFORMS):
            total += uniform()
        yield (total - 6.0) * std


def _constants(count, value):
    for _ in range(count):
        yield value


def tensor_std(config: ModelConfig, name):
    """Standard deviation a real model of this shape would be initialized with.

    Projections scale by fan-in, the two residual-output projections take the
    additional 1/sqrt(2L) that keeps the residual stream's variance from
    growing with depth, and RMSNorm gains start at exactly one. This is not
    decoration: every packed codec quantizes by max|v| inside a group, so a
    uniform fill would report a quantization error no trained model produces.
    """
    if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
        return None
    if name == "model.embed_tokens.weight":
        return EMBEDDING_STD
    if name == "lm_head.weight":
        return 1.0 / math.sqrt(config.hidden_size)
    fan_in = config.intermediate_size if name.endswith("mlp.down_proj.weight") else config.hidden_size
    std = 1.0 / math.sqrt(fan_in)
    if name.endswith("self_attn.o_proj.weight") or name.endswith("mlp.down_proj.weight"):
        std /= math.sqrt(2.0 * config.num_hidden_layers)
    return std


def _source_name(config: ModelConfig, name):
    """Map an alias to the tensor whose values it must repeat, bit for bit.

    A stored tied head is not a second tensor with the same shape: the importer
    decodes both and rejects the checkpoint if one float differs, so the alias
    has to draw from the embedding's stream, at the embedding's scale.
    """
    aliases = config.tensor_aliases()
    return aliases.get(name, name)


def _values(config, name, count, seed):
    name = _source_name(config, name)
    std = tensor_std(config, name)
    return _constants(count, 1.0) if std is None else _normals(count, seed, name, std)


def plan_shards(config: ModelConfig, *, dtype="f32", max_shard_bytes=DEFAULT_MAX_SHARD_BYTES,
                include_tied_head=False):
    """Assign physical tensors to shards in sorted order; measure every size.

    A tensor never straddles two shards, so one larger than the limit gets a
    shard to itself instead of a silent truncation.
    """
    if dtype not in _DTYPES:
        raise SyntheticCheckpointError(f"dtype must be one of {', '.join(sorted(_DTYPES))}")
    if type(max_shard_bytes) is not int or max_shard_bytes < MIN_MAX_SHARD_BYTES:
        raise SyntheticCheckpointError(f"max_shard_bytes must be at least {MIN_MAX_SHARD_BYTES}")
    width = _DTYPES[dtype][1]
    shapes = dict(config.required_tensor_shapes())
    if include_tied_head:
        if not config.tie_word_embeddings:
            raise SyntheticCheckpointError("an untied model already stores lm_head.weight")
        shapes["lm_head.weight"] = shapes["model.embed_tokens.weight"]
    shards, current, used = [], [], 0
    for name in sorted(shapes):
        size = math.prod(shapes[name]) * width
        if current and used + size > max_shard_bytes:
            shards.append(current)
            current, used = [], 0
        current.append(name)
        used += size
    shards.append(current)
    if len(shards) > MAX_SHARDS:
        raise SyntheticCheckpointError(f"checkpoint needs {len(shards)} shards; the importer allows {MAX_SHARDS}")
    return shapes, shards


def _shard_header(shapes, names, dtype):
    """Build the exact Safetensors header, then the payload offsets it fixes."""
    code, width, _ = _DTYPES[dtype]
    header, offset = {}, 0
    for name in sorted(names):
        size = math.prod(shapes[name]) * width
        header[name] = {"dtype": code, "shape": list(shapes[name]), "data_offsets": [offset, offset + size]}
        offset += size
    header["__metadata__"] = {"format": "pt", "origin": FORMAT, "trained": "false"}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    if len(encoded) > MAX_HEADER_BYTES:
        raise SyntheticCheckpointError("Safetensors header exceeds the supported limit; use more shards")
    return encoded, offset


def _write_shard(path, config, shapes, names, dtype, seed):
    """Stream one shard: header first, then each tensor in bounded chunks."""
    encoded, payload_bytes = _shard_header(shapes, names, dtype)
    _, _, code = _DTYPES[dtype]
    digest = hashlib.sha256()
    written = 0
    with open(path, "wb") as stream:
        prefix = struct.pack("<Q", len(encoded)) + encoded
        stream.write(prefix)
        digest.update(prefix)
        for name in sorted(names):
            remaining = math.prod(shapes[name])
            source = _values(config, name, remaining, seed)
            while remaining:
                count = min(remaining, CHUNK_VALUES)
                chunk = struct.pack(f"<{count}{code}", *(next(source) for _ in range(count)))
                stream.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                remaining -= count
        stream.flush()
        os.fsync(stream.fileno())
    if written != payload_bytes:
        raise SyntheticCheckpointError(f"shard payload is {written} bytes; the header declares {payload_bytes}")
    return {"path": Path(path).name, "size_bytes": len(prefix) + written,
            "sha256": digest.hexdigest(), "payload_bytes": payload_bytes,
            "tensors": len(names)}


def _hf_config(config: ModelConfig):
    return {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "_name_or_path": config.name,
            "vocab_size": config.vocab_size, "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "tie_word_embeddings": config.tie_word_embeddings,
            "hidden_act": "silu", "rope_theta": config.rope_theta,
            "rms_norm_eps": config.rms_norm_eps,
            "attention_bias": False, "mlp_bias": False, "torch_dtype": "float32"}


def synthesize_checkpoint(destination, config: ModelConfig, *, seed, dtype="f32",
                          max_shard_bytes=DEFAULT_MAX_SHARD_BYTES, include_tied_head=False,
                          definition=None):
    """Write a complete synthetic checkpoint directory, publishing it atomically."""
    if not isinstance(config, ModelConfig):
        raise SyntheticCheckpointError("config must be a validated ModelConfig")
    if type(seed) is not int or isinstance(seed, bool) or not 0 <= seed < (1 << 63):
        raise SyntheticCheckpointError("seed must be an integer in [0, 2**63)")
    destination = Path(destination)
    if os.path.lexists(destination):
        raise SyntheticCheckpointError(f"Checkpoint destination already exists: {destination}")
    shapes, shards = plan_shards(config, dtype=dtype, max_shard_bytes=max_shard_bytes,
                                 include_tied_head=include_tied_head)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".nexa-synth-", dir=destination.parent))
    try:
        files = []
        if len(shards) == 1:
            files.append(_write_shard(staging / "model.safetensors", config, shapes,
                                      shards[0], dtype, seed))
            mapping = None
        else:
            mapping, total = {}, 0
            for index, names in enumerate(shards, 1):
                filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
                entry = _write_shard(staging / filename, config, shapes, names, dtype, seed)
                files.append(entry)
                total += entry["payload_bytes"]
                mapping.update({name: filename for name in names})
            index_text = json.dumps({"metadata": {"total_size": total}, "weight_map": mapping},
                                    sort_keys=True, indent=2) + "\n"
            (staging / "model.safetensors.index.json").write_text(index_text, encoding="utf-8")
        hf = _hf_config(config)
        (staging / "config.json").write_text(json.dumps(hf, indent=2, sort_keys=True,
                                                        allow_nan=False) + "\n", encoding="utf-8")
        for name in ("model.safetensors.index.json", "config.json"):
            path = staging / name
            if path.exists():
                data = path.read_bytes()
                files.append({"path": name, "size_bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()})
        stored = sum(math.prod(shape) for shape in shapes.values())
        origin = {
            "format": FORMAT, "schema_version": SCHEMA_VERSION,
            "synthetic": True, "trained": False, "quality_measured": False,
            "tool": TOOL, "tool_version": TOOL_VERSION, "seed": seed,
            "model_name": config.name, "definition": definition,
            "config": config.to_dict(), "hf_config": hf,
            "parameter_count": config.parameter_count(),
            "stored_scalar_count": stored, "stored_dtype": _DTYPES[dtype][0],
            "tied_head_stored": include_tied_head, "shard_count": len(shards),
            "initialization": {
                "sampler": "irwin_hall_12_minus_6",
                "scope": "sum of twelve Mersenne-Twister uniforms; N(0,1) truncated at +/-6 sigma",
                "embeddings_std": EMBEDDING_STD,
                "projection_std": "1/sqrt(fan_in)",
                "residual_projection_extra_scale": "1/sqrt(2 * num_hidden_layers) on o_proj and down_proj",
                "norm_weights": 1.0,
            },
            "files": sorted(files, key=lambda entry: entry["path"]),
            "warning": ("Pseudorandom weights at the scale of a real model. Valid for measuring "
                        "physics -- bytes, resident memory, arena peak, wall time, whether it fits "
                        "512 MB -- and for nothing about quality: perplexity, coherence and "
                        "per-tensor codec sensitivity measured here do not transfer to a trained "
                        "model. See weights/README.md and docs/NEXALM_MODELOS_SINTETICOS.md."),
        }
        (staging / "origin.json").write_text(json.dumps(origin, indent=2, sort_keys=True,
                                                        allow_nan=False) + "\n", encoding="utf-8")
        os.rename(staging, destination)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return {**origin, "path": str(destination.resolve()),
            "checkpoint_bytes": sum(entry["size_bytes"] for entry in origin["files"])}


def load_definition(path, name):
    source = Path(path)
    with source.open("rb") as stream:
        raw = stream.read(MAX_DEFINITION_BYTES + 1)
    if len(raw) > MAX_DEFINITION_BYTES:
        raise SyntheticCheckpointError("Model definition exceeds 1 MiB")
    definitions = compile_model_definition(raw.decode("utf-8"))
    if name is None:
        if len(definitions) != 1:
            raise SyntheticCheckpointError(
                "--model is required; the definition declares " + ", ".join(sorted(definitions)))
        name = next(iter(definitions))
    if name not in definitions:
        raise SyntheticCheckpointError(f"{name} is not declared; found " + ", ".join(sorted(definitions)))
    return definitions[name], {"path": str(source), "model": name,
                               "sha256": hashlib.sha256(raw).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, help="A .nxl file declaring the architecture")
    parser.add_argument("--model", help="Which model of --definition to synthesize")
    parser.add_argument("--name", help="Model name when the shape is given on the command line")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--query-heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--ffn-hidden", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--rope-theta", type=float, help="Default: 10000.0")
    parser.add_argument("--rms-norm-eps", type=float, help="Default: 1e-5")
    parser.add_argument("--untied", action="store_true", help="Store a separate lm_head.weight")
    parser.add_argument("--store-tied-head", action="store_true",
                        help="Also store the redundant tied lm_head.weight, to exercise the importer's check")
    parser.add_argument("--seed", type=int, required=True, help="Bytes are a function of this seed alone")
    parser.add_argument("--dtype", choices=tuple(sorted(_DTYPES)), default="f32")
    parser.add_argument("--max-shard-bytes", default=None,
                        help="Split the payload across shards at this size (e.g. 256MiB)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Report the plan without writing weights")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    shape_flags = (args.vocab_size, args.hidden_size, args.layers, args.query_heads,
                   args.kv_heads, args.ffn_hidden, args.context)
    if (args.definition is None) == all(value is None for value in shape_flags):
        parser.error("pass either --definition or the full set of shape options")
    architecture_flags = (*shape_flags, args.name, args.rope_theta, args.rms_norm_eps)
    if args.definition is not None and (args.untied
                                        or any(value is not None for value in architecture_flags)):
        parser.error("--definition takes the whole architecture from the file; do not also pass "
                     "--name/--untied/--rope-theta/--rms-norm-eps or shape options")
    if args.definition is None:
        if any(value is None for value in shape_flags):
            parser.error("the command-line shape needs --vocab-size, --hidden-size, --layers, "
                         "--query-heads, --kv-heads, --ffn-hidden and --context")
        if args.model is not None:
            parser.error("--model selects a model inside --definition")
    try:
        definition = None
        if args.definition is not None:
            config, definition = load_definition(args.definition, args.model)
        else:
            config = ModelConfig(
                name=args.name or "NexaLM_Synthetic", vocab_size=args.vocab_size,
                hidden_size=args.hidden_size, intermediate_size=args.ffn_hidden,
                num_hidden_layers=args.layers, num_attention_heads=args.query_heads,
                num_key_value_heads=args.kv_heads, max_position_embeddings=args.context,
                rope_theta=10000.0 if args.rope_theta is None else args.rope_theta,
                rms_norm_eps=1e-5 if args.rms_norm_eps is None else args.rms_norm_eps,
                tie_word_embeddings=not args.untied)
        max_shard_bytes = (DEFAULT_MAX_SHARD_BYTES if args.max_shard_bytes is None
                           else parse_memory_size(args.max_shard_bytes))
        if args.dry_run:
            shapes, shards = plan_shards(config, dtype=args.dtype, max_shard_bytes=max_shard_bytes,
                                         include_tied_head=args.store_tied_head)
            width = _DTYPES[args.dtype][1]
            result = {"format": FORMAT, "dry_run": True, "synthetic": True, "trained": False,
                      "model_name": config.name, "config": config.to_dict(),
                      "parameter_count": config.parameter_count(),
                      "stored_scalar_count": sum(math.prod(shape) for shape in shapes.values()),
                      "stored_dtype": _DTYPES[args.dtype][0], "shard_count": len(shards),
                      "payload_bytes": sum(math.prod(shape) for shape in shapes.values()) * width,
                      "shards": [{"tensors": len(names),
                                  "payload_bytes": sum(math.prod(shapes[name]) for name in names) * width}
                                 for names in shards]}
        else:
            result = synthesize_checkpoint(args.out, config, seed=args.seed, dtype=args.dtype,
                                           max_shard_bytes=max_shard_bytes,
                                           include_tied_head=args.store_tied_head,
                                           definition=definition)
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.report:
            from tools.nexa_bench import write_report
            write_report(args.report, rendered)
        print(rendered, end="")
        return 0
    except (OSError, ValueError, ArithmeticError, MemoryError) as exc:
        print(f"Synthetic checkpoint generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

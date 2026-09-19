"""Deterministic tiny Llama-shaped checkpoint for offline tests (untrained)."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.model_config import ModelConfig


TINY_CONFIG = {
    "model_type": "llama", "architectures": ["LlamaForCausalLM"],
    "_name_or_path": "NexaLM_R0_TinyFixture",
    "vocab_size": 16, "hidden_size": 8, "intermediate_size": 16,
    "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
    "max_position_embeddings": 8, "tie_word_embeddings": True,
    "hidden_act": "silu", "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
    "attention_bias": False, "mlp_bias": False, "bos_token_id": 1, "eos_token_id": 2,
}


def write_safetensors(path, tensors):
    """Small independent format fixture, not the production exporter."""
    header = {"__metadata__": {"format": "pt", "fixture": "untrained deterministic weights"}}
    payload = bytearray()
    for name, (shape, dtype, values) in sorted(tensors.items()):
        values = list(values)
        if len(values) != math.prod(shape):
            raise ValueError("Fixture value count does not match shape")
        start = len(payload)
        for value in values:
            if dtype == "F32":
                payload.extend(struct.pack("<f", value))
            elif dtype == "F16":
                payload.extend(struct.pack("<e", value))
            elif dtype == "BF16":
                # Fixture values are dyadic numbers represented exactly in BF16.
                bits = struct.unpack("<I", struct.pack("<f", value))[0]
                if bits & 65535:
                    raise ValueError("Fixture BF16 value would need rounding")
                payload.extend(struct.pack("<H", bits >> 16))
            else:
                raise ValueError("Unsupported fixture dtype")
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return len(payload)


def create_checkpoint(destination, *, sharded=True, include_tied_head=False):
    destination = Path(destination)
    if os.path.lexists(destination):
        raise ValueError("Fixture output must not already exist")
    config = ModelConfig.from_hf_config(dict(TINY_CONFIG))
    tensors = {}
    for i, (name, shape) in enumerate(sorted(config.required_tensor_shapes().items())):
        if len(shape) == 1:
            values = [1.0 + ((j % 3) - 1) / 32.0 for j in range(math.prod(shape))]
            dtype = "F32"
        else:
            values = [((i * 7 + j * 3) % 23 - 11) / 16.0 for j in range(math.prod(shape))]
            dtype = "BF16" if name == "model.embed_tokens.weight" else "F16"
        tensors[name] = (shape, dtype, values)
    if include_tied_head:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".nexa-fixture-", dir=destination.parent))
    try:
        (temporary / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2) + "\n")
        tokens = ["<unk>", "<s>", "</s>"] + [f"token{i}" for i in range(3, 16)]
        tokenizer = {"version": "1.0", "truncation": None, "padding": None,
                     "added_tokens": [], "normalizer": None,
                     "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
                     "decoder": None, "model": {"type": "WordLevel",
                     "vocab": {token: i for i, token in enumerate(tokens)}, "unk_token": "<unk>"}}
        (temporary / "tokenizer.json").write_text(json.dumps(tokenizer, sort_keys=True))
        (temporary / "tokenizer_config.json").write_text(json.dumps({
            "model_max_length": 8, "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}))
        if sharded:
            names = sorted(tensors)
            shards = [names[::2], names[1::2]]
            mapping, total_size = {}, 0
            for i, names in enumerate(shards, 1):
                filename = f"model-{i:05d}-of-00002.safetensors"
                total_size += write_safetensors(temporary / filename, {name: tensors[name] for name in names})
                mapping.update({name: filename for name in names})
            (temporary / "model.safetensors.index.json").write_text(json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": mapping}, sort_keys=True))
        else:
            write_safetensors(temporary / "model.safetensors", tensors)
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"path": str(destination.resolve()), "untrained_fixture": True,
            "physical_parameter_count": config.parameter_count(), "stored_tensors": len(tensors),
            "sharded": sharded}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--single-file", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(create_checkpoint(args.out, sharded=not args.single_file), indent=2))
    except (OSError, ValueError) as exc:
        print(f"Fixture creation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

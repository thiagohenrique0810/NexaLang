"""Deterministic untrained D64 model for CPU KV backing-store demonstrations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.model_config import ModelConfig
from runtime.nexapack.bundle import write_model_bundle


def create_bundle(destination, *, capacity=512):
    config = ModelConfig(name="kv_offload_512_fixture", vocab_size=13, hidden_size=64,
                         intermediate_size=96, num_hidden_layers=1, num_attention_heads=1,
                         num_key_value_heads=1, max_position_embeddings=capacity,
                         tie_word_embeddings=True)
    rng = random.Random(411)

    def f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    def source(shape):
        if len(shape) == 1:
            return iter([[f32(rng.uniform(0.85, 1.15)) for _ in range(shape[0])]])
        return ([f32(rng.uniform(-0.15, 0.15)) for _ in range(shape[1])] for _ in range(shape[0]))

    sources = {name: lambda shape=shape: source(shape)
               for name, shape in config.required_tensor_shapes().items()}
    write_model_bundle(destination, config, sources, group_size=32, block_rows=16)
    return {"path": str(Path(destination).resolve()), "untrained_fixture": True,
            "capacity": capacity, "seed": 411, "physical_parameter_count": config.parameter_count()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--capacity", type=int, default=512)
    args = parser.parse_args()
    try:
        print(json.dumps(create_bundle(args.out, capacity=args.capacity), indent=2))
    except (OSError, ValueError) as error:
        print(f"Fixture creation failed: {error}", file=sys.stderr)
        raise SystemExit(1)

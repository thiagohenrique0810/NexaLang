#!/usr/bin/env python3
"""TinyLlama chat runner with TurboQuant-compressed prompt context.

This script uses the HF TinyLlama checkpoint (local cache) for correct tokenization
and generation, and compresses/decompresses the prompt embedding context with
TurboQuant before inference.
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TurboQuant:
    def __init__(self, lib_path: Path, dim: int, bits: int = 3, seed: int = 42):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.tq_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.tq_create.restype = ctypes.c_void_p
        self.lib.tq_destroy.argtypes = [ctypes.c_void_p]
        self.lib.tq_destroy.restype = None
        self.lib.tq_quantize_packed.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
        ]
        self.lib.tq_quantize_packed.restype = None
        self.lib.tq_dequantize_packed.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.tq_dequantize_packed.restype = None

        self.dim = dim
        self.bits = bits
        self.ctx = self.lib.tq_create(dim, bits, seed)
        if not self.ctx:
            raise RuntimeError("tq_create failed (dim must be power-of-two, bits in [1..8])")

    def close(self) -> None:
        if self.ctx:
            self.lib.tq_destroy(self.ctx)
            self.ctx = None

    def roundtrip(self, x: np.ndarray) -> tuple[np.ndarray, float, int]:
        # x shape: [n_vec, dim], dtype float32
        n_vec, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"dim mismatch: got {dim}, expected {self.dim}")

        total_bits = n_vec * dim * self.bits
        packed_size = (total_bits + 7) // 8

        packed = np.zeros((packed_size,), dtype=np.uint8)
        out = np.zeros_like(x, dtype=np.float32)

        self.lib.tq_quantize_packed(
            self.ctx,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n_vec,
        )
        self.lib.tq_dequantize_packed(
            self.ctx,
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_vec,
        )

        mse = float(np.mean((x - out) ** 2))
        return out, mse, packed_size


def resolve_runtime_lib(repo_root: Path) -> Path:
    lib = repo_root / "runtime" / "libturboquant.dylib"
    if not lib.exists():
        raise FileNotFoundError(f"TurboQuant dylib not found: {lib}")
    return lib


def generate_with_tq_context(
    model,
    tokenizer,
    tq_lib: Path,
    text: str,
    bits: int,
    max_new: int,
) -> tuple[str, int, int, int, float]:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    with torch.no_grad():
        emb_layer = model.get_input_embeddings()
        emb = emb_layer(input_ids).to(torch.float32).detach().cpu().numpy().astype(np.float32)

    seq_len = emb.shape[1]
    dim = emb.shape[2]
    flat = emb.reshape(seq_len, dim)

    tq = TurboQuant(tq_lib, dim=dim, bits=bits)
    try:
        rec, mse, packed_size = tq.roundtrip(flat)
    finally:
        tq.close()

    rec_t = torch.from_numpy(rec.reshape(1, seq_len, dim)).to(dtype=model.dtype, device=model.device)
    attn_mask = attn_mask.to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            inputs_embeds=rec_t,
            attention_mask=attn_mask,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    original_bytes = flat.size * 4
    return generated, seq_len, dim, packed_size, mse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--prompt", default="Explique em uma frase o que e NexaLang.")
    ap.add_argument("--bits", type=int, default=3, choices=[1, 2, 3, 4])
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--interactive", action="store_true", help="Run an interactive chat loop")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tq_lib = resolve_runtime_lib(repo_root)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)
    model.eval()

    if args.interactive:
        print("=== TinyLlama Interactive + TurboQuant Context ===")
        print("Digite sua mensagem. Use q/quit/sair para encerrar.")
        history = ""
        while True:
            try:
                user = input("\n> ").strip()
            except EOFError:
                break
            if user.lower() in {"q", "quit", "sair"}:
                break
            if not user:
                continue

            text = history + f"<|user|>\n{user}\n<|assistant|>\n"
            generated, seq_len, dim, packed_size, mse = generate_with_tq_context(
                model, tokenizer, tq_lib, text, args.bits, args.max_new
            )

            marker = "<|assistant|>"
            if marker in generated:
                assistant = generated.split(marker)[-1].strip()
            else:
                assistant = generated.strip()

            print("\n--- Assistant ---")
            print(assistant)

            original_bytes = seq_len * dim * 4
            print(f"\n[ctx tokens={seq_len} bits={args.bits} ratio={original_bytes / max(1, packed_size):.2f}x mse={mse:.6f}]")
            history = text + assistant + "\n"
    else:
        text = f"<|user|>\n{args.prompt}\n<|assistant|>\n"
        generated, seq_len, dim, packed_size, mse = generate_with_tq_context(
            model, tokenizer, tq_lib, text, args.bits, args.max_new
        )

        original_bytes = seq_len * dim * 4
        print("=== TinyLlama + TurboQuant Context ===")
        print(f"Model: {args.model}")
        print(f"Prompt tokens: {seq_len}")
        print(f"Embedding dim: {dim}")
        print(f"TurboQuant bits: {args.bits}")
        print(f"Context bytes fp32: {original_bytes}")
        print(f"Context bytes packed: {packed_size}")
        print(f"Compression ratio: {original_bytes / max(1, packed_size):.2f}x")
        print(f"Roundtrip MSE: {mse:.6f}")
        print("\n--- Response ---")
        print(generated)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

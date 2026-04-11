#!/usr/bin/env python3
"""
BPE Tokenizer for NexaLang Transformer v6.

Trains Byte-Pair Encoding on Portuguese corpus,
encodes pretrain + instruction data to i32 binary format,
and saves merge table for NexaLang runtime decode/encode.

Output files:
  data/bpe_merges.bin          — merge table (for NexaLang programs)
  data/text_pt_v2_bpe/train.bin, val.bin   — BPE pretrain data
  data/instruct_pt_v2_bpe/train.bin, val.bin — BPE instruction data
"""
import struct
import os
import sys
import time

# ─── Configuration ─────────────────────────────────────────────────────
PRETRAIN_CORPUS = "data/text_pt_v2/corpus_pt_v2.txt"
INSTRUCT_TEXT   = "data/instruct_pt_v2/instruct_data.txt"
ARTIFACTS_MODELS_DIR = os.environ.get("NEXA_ARTIFACTS_MODELS", "artifacts/models")
BPE_MERGES_PATH = os.path.join(ARTIFACTS_MODELS_DIR, "bpe_merges.bin")
LEGACY_BPE_MERGES_PATH = "data/bpe_merges.bin"
PRETRAIN_OUT    = "data/text_pt_v2_bpe"
INSTRUCT_OUT    = "data/instruct_pt_v2_bpe"

NUM_MERGES = 768       # V = 256 + 768 = 1024
TRAIN_RATIO = 0.9

# ─── BPE Training ─────────────────────────────────────────────────────

def train_bpe(text_bytes, num_merges):
    """Train BPE merges on corpus bytes. Returns list of (a, b) merge pairs."""
    # Start with byte tokens
    tokens = list(text_bytes)
    merges = []

    print(f"  Training BPE: {len(tokens):,} initial tokens, {num_merges} merges")

    for m in range(num_merges):
        # Count pairs
        pair_counts = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            print(f"  No more pairs at merge {m}")
            break

        # Find most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]
        new_token = 256 + m

        merges.append(best_pair)

        # Merge all occurrences
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

        if m % 100 == 0 or m == num_merges - 1:
            ratio = len(text_bytes) / len(tokens)
            print(f"    Merge {m}: ({best_pair[0]}, {best_pair[1]}) -> {new_token}  "
                  f"count={best_count:,}  tokens={len(tokens):,}  ratio={ratio:.2f}x")

    final_ratio = len(text_bytes) / len(tokens)
    print(f"  BPE training done: {len(text_bytes):,} -> {len(tokens):,} tokens ({final_ratio:.2f}x compression)")
    return merges


def bpe_encode(text_bytes, merges):
    """Encode bytes using trained BPE merges."""
    tokens = list(text_bytes)
    for m_idx, (a, b) in enumerate(merges):
        new_token = 256 + m_idx
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


def build_decode_table(merges):
    """Build decode table: token_id -> list of bytes."""
    decode = {}
    # Base: tokens 0-255 map to their byte value
    for i in range(256):
        decode[i] = [i]
    # Merge tokens: recursively expand
    for m_idx, (a, b) in enumerate(merges):
        tok = 256 + m_idx
        decode[tok] = decode[a] + decode[b]
    return decode


def save_bpe_merges(path, merges, decode_table, vocab_size):
    """Save BPE merge table in binary format for NexaLang."""
    num_merges = len(merges)
    # Find max token byte length
    max_len = max(len(v) for v in decode_table.values())
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        # Header: num_merges, vocab_size, max_token_len
        f.write(struct.pack('<iii', num_merges, vocab_size, max_len))
        # Merge pairs (a, b) as i32
        for a, b in merges:
            f.write(struct.pack('<i', a))
        for a, b in merges:
            f.write(struct.pack('<i', b))
        # Decode lengths
        for tok in range(vocab_size):
            f.write(struct.pack('<i', len(decode_table[tok])))
        # Decode bytes (fixed-width, padded with 0)
        for tok in range(vocab_size):
            bts = decode_table[tok]
            padded = bts + [0] * (max_len - len(bts))
            f.write(bytes(padded))

    file_size = os.path.getsize(path)
    print(f"  Saved {path}: {file_size:,} bytes")
    print(f"    num_merges={num_merges}, vocab_size={vocab_size}, max_token_len={max_len}")


def write_bpe_binary(tokens, path, vocab_size):
    """Write BPE tokens as i32 binary for NexaLang."""
    n = len(tokens)
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', n))
        f.write(struct.pack('<i', vocab_size))
        for t in tokens:
            f.write(struct.pack('<i', t))
    file_size = os.path.getsize(path)
    print(f"  Wrote {path}: {n:,} tokens ({file_size / 1024 / 1024:.1f} MB)")


def main():
    t0 = time.time()
    print("=" * 60)
    print("  NexaLang BPE Tokenizer Training")
    print(f"  Target vocab: V={256 + NUM_MERGES} ({NUM_MERGES} merges)")
    print("=" * 60)

    # 1. Load pretrain corpus
    print(f"\n[1] Loading pretrain corpus: {PRETRAIN_CORPUS}")
    with open(PRETRAIN_CORPUS, 'r', encoding='utf-8') as f:
        pretrain_text = f.read()
    pretrain_bytes = pretrain_text.encode('utf-8')
    print(f"  {len(pretrain_text):,} chars = {len(pretrain_bytes):,} bytes")

    # 2. Load instruction text
    print(f"\n[2] Loading instruction text: {INSTRUCT_TEXT}")
    with open(INSTRUCT_TEXT, 'r', encoding='utf-8') as f:
        instruct_text = f.read()
    instruct_bytes = instruct_text.encode('utf-8')
    print(f"  {len(instruct_text):,} chars = {len(instruct_bytes):,} bytes")

    # 3. Train BPE on combined corpus (pretrain + instruction)
    print(f"\n[3] Training BPE on combined corpus...")
    combined_bytes = pretrain_bytes + instruct_bytes
    print(f"  Combined: {len(combined_bytes):,} bytes")
    merges = train_bpe(combined_bytes, NUM_MERGES)

    # 4. Build decode table
    vocab_size = 256 + len(merges)
    decode_table = build_decode_table(merges)
    print(f"\n  Vocab size: {vocab_size}")

    # Show some example tokens
    print("\n  Sample BPE tokens:")
    for tok in [256, 257, 258, 259, 260, 300, 400, 500, 700, vocab_size-1]:
        if tok < vocab_size:
            decoded = bytes(decode_table[tok])
            try:
                text_repr = decoded.decode('utf-8')
            except:
                text_repr = repr(decoded)
            print(f"    Token {tok}: {text_repr!r} ({len(decode_table[tok])} bytes)")

    # 5. Save BPE merges file
    print(f"\n[4] Saving BPE merges...")
    save_bpe_merges(BPE_MERGES_PATH, merges, decode_table, vocab_size)

    # Keep a legacy copy for older scripts that still read data/bpe_merges.bin
    os.makedirs(os.path.dirname(LEGACY_BPE_MERGES_PATH), exist_ok=True)
    if BPE_MERGES_PATH != LEGACY_BPE_MERGES_PATH:
        with open(BPE_MERGES_PATH, "rb") as src, open(LEGACY_BPE_MERGES_PATH, "wb") as dst:
            dst.write(src.read())
        print(f"  Legacy copy: {LEGACY_BPE_MERGES_PATH}")

    # 6. Encode pretrain data
    print(f"\n[5] Encoding pretrain data...")
    t_enc = time.time()
    pretrain_tokens = bpe_encode(pretrain_bytes, merges)
    ratio_pt = len(pretrain_bytes) / len(pretrain_tokens)
    print(f"  Pretrain: {len(pretrain_bytes):,} bytes -> {len(pretrain_tokens):,} tokens ({ratio_pt:.2f}x) [{time.time()-t_enc:.1f}s]")

    # Train/val split
    split_idx = int(len(pretrain_tokens) * TRAIN_RATIO)
    pt_train = pretrain_tokens[:split_idx]
    pt_val = pretrain_tokens[split_idx:]

    os.makedirs(PRETRAIN_OUT, exist_ok=True)
    write_bpe_binary(pt_train, os.path.join(PRETRAIN_OUT, "train.bin"), vocab_size)
    write_bpe_binary(pt_val, os.path.join(PRETRAIN_OUT, "val.bin"), vocab_size)

    # 7. Encode instruction data
    print(f"\n[6] Encoding instruction data...")
    t_enc = time.time()
    instruct_tokens = bpe_encode(instruct_bytes, merges)
    ratio_ins = len(instruct_bytes) / len(instruct_tokens)
    print(f"  Instruct: {len(instruct_bytes):,} bytes -> {len(instruct_tokens):,} tokens ({ratio_ins:.2f}x) [{time.time()-t_enc:.1f}s]")

    split_idx = int(len(instruct_tokens) * TRAIN_RATIO)
    ins_train = instruct_tokens[:split_idx]
    ins_val = instruct_tokens[split_idx:]

    os.makedirs(INSTRUCT_OUT, exist_ok=True)
    write_bpe_binary(ins_train, os.path.join(INSTRUCT_OUT, "train.bin"), vocab_size)
    write_bpe_binary(ins_val, os.path.join(INSTRUCT_OUT, "val.bin"), vocab_size)

    # 8. Verification
    print(f"\n[7] Verification...")
    # Decode first 200 pretrain tokens and compare
    reconstructed = []
    for t in pretrain_tokens[:200]:
        reconstructed.extend(decode_table[t])
    original = list(pretrain_bytes[:len(reconstructed)])
    match = reconstructed[:len(original)] == original
    print(f"  Decode verification: {'PASS' if match else 'FAIL'}")
    if match:
        decoded_text = bytes(reconstructed).decode('utf-8', errors='replace')[:100]
        print(f"  Preview: {decoded_text!r}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Done! Total time: {elapsed:.1f}s")
    print(f"  Pretrain: {len(pretrain_tokens):,} BPE tokens ({ratio_pt:.2f}x)")
    print(f"  Instruct: {len(instruct_tokens):,} BPE tokens ({ratio_ins:.2f}x)")
    print(f"  Vocab: V={vocab_size}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

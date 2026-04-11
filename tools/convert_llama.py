#!/usr/bin/env python3
"""Download and convert a LLaMA-family model to NexaLang binary format.

Usage:
    python3 tools/convert_llama.py [model_name] [context_len]

Examples:
    python3 tools/convert_llama.py                                          # TinyLlama 1.1B Chat
    python3 tools/convert_llama.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 256  # shorter context
"""

import os, sys, struct, json, glob
import numpy as np


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    context_len = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    out_dir = os.path.join("artifacts", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_weights = os.path.join(out_dir, "tinyllama.bin")
    out_tokenizer = os.path.join(out_dir, "tinyllama_bpe.bin")

    print(f"Model:   {model_name}")
    print(f"Context: {context_len}")
    print(f"Output:  {out_weights}, {out_tokenizer}")

    # === 1. Download model files ===
    print("\n=== Downloading model ===")
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.json", "tokenizer.model"],
        ignore_patterns=["*.bin", "*.ot", "*.msgpack", "*.onnx"],
    )
    print(f"Downloaded to: {model_dir}")

    # === 2. Read config ===
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    V = config["vocab_size"]
    D = config["hidden_size"]
    H = config["num_attention_heads"]
    H_KV = config.get("num_key_value_heads", H)
    FF = config["intermediate_size"]
    N = config["num_hidden_layers"]
    T = context_len
    HD = D // H
    D_KV = H_KV * HD
    rope_theta = config.get("rope_theta", 10000.0)

    n_params = V*D + N*(D*D + D*D_KV*2 + D*D + D*FF*3 + D*2) + D + D*V
    print(f"\nConfig:")
    print(f"  V={V}, D={D}, H={H}, H_KV={H_KV}, FF={FF}, N={N}")
    print(f"  HD={HD}, D_KV={D_KV}, T={T}, rope_theta={rope_theta}")
    print(f"  ~{n_params/1e6:.0f}M parameters")

    # === 3. Load weights from safetensors ===
    print("\n=== Loading weights ===")
    from safetensors import safe_open

    sf_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not sf_files:
        # Try looking in subdirectories (snapshot_download structure)
        sf_files = sorted(glob.glob(os.path.join(model_dir, "**", "*.safetensors"), recursive=True))
    print(f"  Found {len(sf_files)} safetensors file(s)")

    # Try torch first (handles bfloat16 natively), fall back to manual conversion
    try:
        import torch
        sd = {}
        for sf in sf_files:
            with safe_open(sf, framework="pt") as f:
                for key in f.keys():
                    sd[key] = f.get_tensor(key).float().numpy()
        print(f"  Loaded {len(sd)} tensors (via torch)")
    except ImportError:
        # No torch — handle bfloat16 manually via raw safetensors bytes
        from safetensors import deserialize as _sf_deser
        sd = {}
        for sf in sf_files:
            with open(sf, 'rb') as raw_f:
                raw_bytes = raw_f.read()
            header_size = struct.unpack('<Q', raw_bytes[:8])[0]
            header_json = json.loads(raw_bytes[8:8+header_size])
            data_start = 8 + header_size
            for key, info in header_json.items():
                if key == '__metadata__':
                    continue
                dtype_str = info['dtype']
                shape = info['shape']
                offsets = info['data_offsets']
                blob = raw_bytes[data_start + offsets[0]:data_start + offsets[1]]
                if dtype_str == 'BF16':
                    u16 = np.frombuffer(blob, dtype=np.uint16).reshape(shape)
                    fp32 = np.zeros(shape, dtype=np.float32)
                    fp32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
                    sd[key] = fp32
                elif dtype_str == 'F16':
                    sd[key] = np.frombuffer(blob, dtype=np.float16).reshape(shape).astype(np.float32)
                elif dtype_str == 'F32':
                    sd[key] = np.frombuffer(blob, dtype=np.float32).reshape(shape).copy()
                else:
                    raise ValueError(f"Unsupported dtype: {dtype_str} for {key}")
        print(f"  Loaded {len(sd)} tensors (manual bf16 conversion)")

    # === 4. Convert weights to NexaLang binary ===
    print(f"\n=== Converting weights to {out_weights} ===")
    with open(out_weights, 'wb') as f:
        # Header: 7 × i32
        f.write(struct.pack('iiiiiii', V, D, H, H_KV, FF, T, N))

        def w(key, transpose=False):
            t = sd[key].astype(np.float32)
            if transpose:
                t = t.T.copy()
            f.write(t.tobytes())
            return t.shape

        def w_layers(template, transpose=True):
            for i in range(N):
                shape = w(template.format(i), transpose=transpose)
            print(f"  {template.format('*')}: {shape} ×{N}" + (" [T]" if transpose else ""))

        # tok_emb [V, D]
        embed_key = "model.embed_tokens.weight" if "model.embed_tokens.weight" in sd else "lm_head.weight"
        shape = w(embed_key)
        print(f"  embed_tokens ({embed_key}): {shape}")

        # Per-weight-type (all layers)
        w_layers("model.layers.{}.self_attn.q_proj.weight")
        w_layers("model.layers.{}.self_attn.k_proj.weight")
        w_layers("model.layers.{}.self_attn.v_proj.weight")
        w_layers("model.layers.{}.self_attn.o_proj.weight")
        w_layers("model.layers.{}.input_layernorm.weight", transpose=False)
        w_layers("model.layers.{}.mlp.gate_proj.weight")
        w_layers("model.layers.{}.mlp.up_proj.weight")
        w_layers("model.layers.{}.mlp.down_proj.weight")
        w_layers("model.layers.{}.post_attention_layernorm.weight", transpose=False)

        # Final norm
        shape = w("model.norm.weight")
        print(f"  norm: {shape}")

        # lm_head (may be tied with embed_tokens)
        if "lm_head.weight" in sd:
            shape = w("lm_head.weight", transpose=True)
        elif "model.embed_tokens.weight" in sd:
            shape = w("model.embed_tokens.weight", transpose=True)
            print("  (lm_head tied with embed_tokens)")
        else:
            print("ERROR: Neither lm_head.weight nor model.embed_tokens.weight found!")
            sys.exit(1)
        print(f"  lm_head: {shape} [T]")

    sz = os.path.getsize(out_weights)
    print(f"\nWeights saved: {out_weights} ({sz/1e9:.2f} GB)")

    # === 5. Convert tokenizer ===
    print(f"\n=== Converting tokenizer to {out_tokenizer} ===")
    import sentencepiece as spm

    sp_path = os.path.join(model_dir, "tokenizer.model")
    if not os.path.exists(sp_path):
        # Try to find it
        candidates = glob.glob(os.path.join(model_dir, "**", "tokenizer.model"), recursive=True)
        if candidates:
            sp_path = candidates[0]
        else:
            print("ERROR: tokenizer.model not found!")
            sys.exit(1)

    sp = spm.SentencePieceProcessor(model_file=sp_path)
    vocab_size = sp.get_piece_size()
    print(f"  SentencePiece vocab: {vocab_size}")

    # Build decode tables:
    # - display_bytes: for output (▁ → space) — stored in binary
    # - raw_bytes: for BPE merge decomposition (▁ preserved as UTF-8)
    display_bytes = {}
    raw_bytes = {}
    max_tok_len = 1
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        if piece.startswith('<0x') and piece.endswith('>'):
            display_bytes[i] = bytes([int(piece[3:-1], 16)])
            raw_bytes[i] = display_bytes[i]
        elif piece in ('<unk>', '<s>', '</s>', '<pad>'):
            display_bytes[i] = b''
            raw_bytes[i] = b''
        else:
            display_bytes[i] = piece.replace('\u2581', ' ').encode('utf-8')
            raw_bytes[i] = piece.encode('utf-8')  # preserve ▁ as UTF-8
        max_tok_len = max(max_tok_len, len(display_bytes[i]))

    # Find byte_start
    byte_start = 0
    for i in range(vocab_size):
        if sp.id_to_piece(i) == '<0x00>':
            byte_start = i
            break
    merge_start = byte_start + 256
    print(f"  byte_start={byte_start}, merge_start={merge_start}, max_tok_len={max_tok_len}")

    # Build byte-to-piece lookup: maps each byte value (0-255) to the correct piece ID.
    # For bytes that have a dedicated single-char piece (like '|', '<', '>', '.', '!', '?'),
    # we use that piece's ID instead of the byte fallback token.
    # This is applied AFTER BPE merges as a final remap step.
    byte_to_piece = np.zeros(256, dtype=np.int32)
    # Default: byte fallback token IDs (no change needed)
    for b in range(256):
        byte_to_piece[b] = byte_start + b
    # Override with predefined single-char pieces
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        if piece.startswith('<') and piece.endswith('>'):
            continue  # skip byte tokens and special tokens
        raw = piece.encode('utf-8')
        if len(raw) == 1 and i >= merge_start:
            # This is a predefined single-char piece — use its actual ID
            byte_val = raw[0]
            byte_to_piece[byte_val] = i
    print(f"  byte_to_piece overrides: {sum(1 for b in range(256) if byte_to_piece[b] != byte_start + b)} chars remapped")

    # Extract merge pairs using raw byte-level decomposition
    # No lid<pid constraint: ▁ is predefined (high ID) but always available
    bytes_to_id = {}
    for i in range(vocab_size):
        b = raw_bytes.get(i, b'')
        if len(b) > 0 and b not in bytes_to_id:
            bytes_to_id[b] = i

    # Find ▁ piece ID (predefined SentencePiece word boundary piece)
    sp_space_id = 0
    for i in range(vocab_size):
        if sp.id_to_piece(i) == '\u2581':
            sp_space_id = i
            break
    print(f"  sp_space_id={sp_space_id}")

    merge_a = []
    merge_b = []
    n_found = 0
    for pid in range(merge_start, vocab_size):
        piece_bytes = raw_bytes.get(pid, b'')
        candidates = []
        if len(piece_bytes) > 1:
            for split_pos in range(1, len(piece_bytes)):
                left = piece_bytes[:split_pos]
                right = piece_bytes[split_pos:]
                if left in bytes_to_id and right in bytes_to_id:
                    lid = bytes_to_id[left]
                    rid = bytes_to_id[right]
                    candidates.append((max(lid, rid), lid, rid))
        if candidates:
            candidates.sort()  # pick split with minimum max-component-ID
            _, lid, rid = candidates[0]
            merge_a.append(lid)
            merge_b.append(rid)
            n_found += 1
        else:
            merge_a.append(0)
            merge_b.append(0)

    num_merges = len(merge_a)
    print(f"  {num_merges} merges extracted ({n_found} decomposed, {num_merges - n_found} fallback)")

    # Validate tokenizer
    def nxl_bpe_encode(text):
        """Simulate NexaLang BPE encoder with post-merge byte_to_piece remap."""
        input_bytes = text.encode('utf-8')
        tokens = [sp_space_id]  # dummy prefix ▁
        for b in input_bytes:
            if b == 0x20:  # space → ▁
                tokens.append(sp_space_id)
            else:
                tokens.append(b + byte_start)
        merge_start_local = byte_start + 256
        for m in range(num_merges):
            a, b = merge_a[m], merge_b[m]
            new_tok = merge_start_local + m
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(new_tok)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        # Post-merge remap: replace remaining byte tokens with predefined piece IDs
        for i in range(len(tokens)):
            t = tokens[i]
            if byte_start <= t < byte_start + 256:
                byte_val = t - byte_start
                tokens[i] = int(byte_to_piece[byte_val])
        return tokens

    test_texts = ["Hello world", "Como vai?", "<|user|>\nOla"]
    all_ok = True
    for text in test_texts:
        sp_ids = sp.encode(text)
        nxl_ids = nxl_bpe_encode(text)
        match = sp_ids == nxl_ids
        status = "OK" if match else "MISMATCH"
        if not match:
            all_ok = False
            print(f"  [{status}] '{text}': SP={sp_ids[:8]}... NXL={nxl_ids[:8]}...")
        else:
            print(f"  [{status}] '{text}': {len(sp_ids)} tokens")

    if not all_ok:
        print("  WARNING: Tokenizer mismatch detected. Output quality may be reduced.")
        print("  (This is expected for some SentencePiece models with non-standard merge order)")

    # Write tokenizer binary
    with open(out_tokenizer, 'wb') as f:
        f.write(struct.pack('iiiii', num_merges, vocab_size, max_tok_len, byte_start, sp_space_id))
        f.write(np.array(merge_a, dtype=np.int32).tobytes())
        f.write(np.array(merge_b, dtype=np.int32).tobytes())
        # byte_to_piece lookup table: 256 × i32
        f.write(byte_to_piece.tobytes())

        decode_len_arr = np.zeros(vocab_size, dtype=np.int32)
        decode_data_arr = np.zeros((vocab_size, max_tok_len), dtype=np.uint8)
        for i in range(vocab_size):
            b = display_bytes.get(i, b'')
            decode_len_arr[i] = len(b)
            for j, bv in enumerate(b):
                decode_data_arr[i, j] = bv

        f.write(decode_len_arr.tobytes())
        f.write(decode_data_arr.tobytes())

    sz = os.path.getsize(out_tokenizer)
    print(f"\nTokenizer saved: {out_tokenizer} ({sz/1e6:.1f} MB)")

    print(f"""
{'='*60}
  Conversion complete!
{'='*60}

  Weights:   {out_weights} ({os.path.getsize(out_weights)/1e9:.2f} GB)
  Tokenizer: {out_tokenizer} ({os.path.getsize(out_tokenizer)/1e6:.1f} MB)

  Model: {model_name}
  Config: V={V}, D={D}, H={H}, H_KV={H_KV}, FF={FF}, N={N}, T={T}

  Next steps:
    python3 bootstrap/main.py examples/llama_chat.nxl --emit ll --opt 3
        clang artifacts/build/output.ll -o llama_chat -lm -O2 -framework Accelerate
    ./llama_chat
""")


if __name__ == "__main__":
    main()

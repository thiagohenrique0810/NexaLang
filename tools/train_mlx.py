#!/usr/bin/env python3
"""
train_mlx.py — Transformer v7 treinamento com MLX (Apple Silicon GPU)

Mesma arquitetura do v6 (V=1024, D=512, H=8, FF=2048, T=512, N=10, ~33M params)
mas rodando na GPU via MLX ao invés de CPU+BLAS.

Salva pesos no mesmo formato binário que o NexaLang chat espera.

Uso:
    python3 tools/train_mlx.py pretrain   # pré-treina → artifacts/models/model_v7.bin
    python3 tools/train_mlx.py finetune   # fine-tune  → artifacts/models/model_v7_ft.bin
    python3 tools/train_mlx.py both       # faz os dois sequencialmente
"""

import sys, os, time, struct, math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

ARTIFACTS_MODELS_DIR = os.environ.get("NEXA_ARTIFACTS_MODELS", "artifacts/models")


def model_path(filename):
    return os.path.join(ARTIFACTS_MODELS_DIR, filename)


def resolve_existing_path(preferred, legacy=None):
    if os.path.exists(preferred):
        return preferred
    if legacy and os.path.exists(legacy):
        return legacy
    return preferred

# ═══════════════════════════════════════════════════════════════════════
# Hiperparâmetros
# ═══════════════════════════════════════════════════════════════════════
V = 1024       # vocab (BPE)
D = 512        # embedding dim
H = 8          # heads
HD = 64        # head dim
FF = 2048      # feedforward
T = 512        # context length
N = 10         # layers
B = 8          # batch size (GPU pode mais que CPU)

# Pretrain
PT_STEPS    = 10000
PT_MAX_LR   = 3e-4
PT_MIN_LR   = 3e-5
PT_WARMUP   = 500
PT_EVAL_INT = 200
PT_EVAL_SAMPLES = 5

# Finetune
FT_STEPS    = 4000
FT_MAX_LR   = 1e-4
FT_MIN_LR   = 1e-5
FT_WARMUP   = 200
FT_EVAL_INT = 100
FT_EVAL_SAMPLES = 5


# ═══════════════════════════════════════════════════════════════════════
# Modelo
# ═══════════════════════════════════════════════════════════════════════
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)
        self.Wo = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D)

    def __call__(self, x, mask):
        Bsz, Tsz, _ = x.shape
        # Pre-norm attention
        h = self.ln1(x)
        q = self.Wq(h).reshape(Bsz, Tsz, H, HD).transpose(0, 2, 1, 3)
        k = self.Wk(h).reshape(Bsz, Tsz, H, HD).transpose(0, 2, 1, 3)
        v = self.Wv(h).reshape(Bsz, Tsz, H, HD).transpose(0, 2, 1, 3)

        scale = math.sqrt(HD)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        scores = scores + mask
        probs = mx.softmax(scores, axis=-1)
        attn_out = (probs @ v).transpose(0, 2, 1, 3).reshape(Bsz, Tsz, D)
        x = x + self.Wo(attn_out)

        # Pre-norm FFN with GELU
        h = self.ln2(x)
        x = x + self.ff2(nn.gelu(self.ff1(h)))
        return x


class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(T, D)
        self.layers = [TransformerBlock() for _ in range(N)]
        self.ln_f = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, V)

    def __call__(self, tokens):
        Bsz, Tsz = tokens.shape
        positions = mx.arange(Tsz)
        x = self.tok_emb(tokens) + self.pos_emb(positions)

        # Causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(Tsz)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        logits = self.out_proj(x)
        return logits


def count_params(model):
    return sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))


# ═══════════════════════════════════════════════════════════════════════
# Data loading (BPE binary format)
# ═══════════════════════════════════════════════════════════════════════
def load_tokens(path):
    """Load BPE tokens from binary: [n_tokens: i32] [vocab_size: i32] [data: n*i32]"""
    with open(path, "rb") as f:
        n_tokens = struct.unpack("i", f.read(4))[0]
        vocab_size = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(n_tokens * 4), dtype=np.int32)
    return data


def get_batch(tokens, batch_size, seq_len, rng_key):
    """Sample a random batch of (input, target) pairs."""
    n = len(tokens) - seq_len - 1
    starts = np.random.randint(0, n, size=batch_size)
    inputs = np.stack([tokens[s:s+seq_len] for s in starts])
    targets = np.stack([tokens[s+1:s+seq_len+1] for s in starts])
    return mx.array(inputs), mx.array(targets)


# ═══════════════════════════════════════════════════════════════════════
# Weight I/O (compatible with NexaLang binary format)
# ═══════════════════════════════════════════════════════════════════════
def save_weights_nxl(model, path):
    """Save weights in NexaLang-compatible binary format."""
    params = dict(nn.utils.tree_flatten(model.parameters()))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        # Header: V, D, H, FF, T, N
        f.write(struct.pack("iiiiii", V, D, H, FF, T, N))

        def w(key):
            arr = np.array(params[key].astype(mx.float32))
            f.write(arr.tobytes())

        def wt(key):
            """Write transposed (MLX stores Linear as [out, in], NexaLang as [in, out])"""
            arr = np.array(params[key].astype(mx.float32)).T.copy()
            f.write(arr.tobytes())

        # tok_emb: [V, D]
        w("tok_emb.weight")
        # pos_emb: [T, D]
        w("pos_emb.weight")

        # Per-weight-type: all layers' Wq, then all bq, etc. (NexaLang load order)
        weight_specs = [
            ("Wq.weight", True), ("Wq.bias", False),
            ("Wk.weight", True), ("Wk.bias", False),
            ("Wv.weight", True), ("Wv.bias", False),
            ("Wo.weight", True), ("Wo.bias", False),
            ("ln1.weight", False), ("ln1.bias", False),
            ("ff1.weight", True), ("ff1.bias", False),
            ("ff2.weight", True), ("ff2.bias", False),
            ("ln2.weight", False), ("ln2.bias", False),
        ]
        for suffix, transpose in weight_specs:
            for i in range(N):
                key = f"layers.{i}.{suffix}"
                if transpose:
                    wt(key)
                else:
                    w(key)

        # Final layer norm
        w("ln_f.weight")
        w("ln_f.bias")

        # Output projection: [D, V]
        wt("out_proj.weight")
        w("out_proj.bias")

    sz = os.path.getsize(path) / 1024 / 1024
    print(f"  Weights saved to {path} ({sz:.1f} MB)")


def load_weights_nxl(model, path):
    """Load weights from NexaLang-compatible binary format."""
    with open(path, "rb") as f:
        hdr = struct.unpack("iiiiii", f.read(24))
        assert hdr == (V, D, H, FF, T, N), f"Header mismatch: {hdr}"

        def r(shape):
            n = 1
            for s in shape: n *= s
            arr = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(shape)
            return mx.array(arr)

        def rt(shape):
            """Read transposed (NexaLang [in,out] → MLX [out,in])"""
            n = 1
            for s in shape: n *= s
            arr = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(shape).T.copy()
            return mx.array(arr)

        updates = {}
        updates["tok_emb.weight"] = r((V, D))
        updates["pos_emb.weight"] = r((T, D))

        # Per-weight-type: all layers' Wq, then all bq, etc. (NexaLang format)
        weight_specs = [
            ("Wq.weight", (D, D), True), ("Wq.bias", (D,), False),
            ("Wk.weight", (D, D), True), ("Wk.bias", (D,), False),
            ("Wv.weight", (D, D), True), ("Wv.bias", (D,), False),
            ("Wo.weight", (D, D), True), ("Wo.bias", (D,), False),
            ("ln1.weight", (D,), False), ("ln1.bias", (D,), False),
            ("ff1.weight", (D, FF), True), ("ff1.bias", (FF,), False),
            ("ff2.weight", (FF, D), True), ("ff2.bias", (D,), False),
            ("ln2.weight", (D,), False), ("ln2.bias", (D,), False),
        ]
        for suffix, shape, transpose in weight_specs:
            for i in range(N):
                key = f"layers.{i}.{suffix}"
                if transpose:
                    updates[key] = rt(shape)
                else:
                    updates[key] = r(shape)

        updates["ln_f.weight"] = r((D,))
        updates["ln_f.bias"]   = r((D,))
        updates["out_proj.weight"] = rt((D, V))
        updates["out_proj.bias"]   = r((V,))

    model.load_weights(list(updates.items()))
    print(f"  Weights loaded from {path}")


# ═══════════════════════════════════════════════════════════════════════
# BPE decode (for text generation display)
# ═══════════════════════════════════════════════════════════════════════
def load_bpe_decode():
    bpe_path = resolve_existing_path(
        model_path("bpe_merges.bin"),
        "data/bpe_merges.bin",
    )
    with open(bpe_path, "rb") as f:
        num_merges, vocab_size, max_tok_len = struct.unpack("iii", f.read(12))
        merge_a = np.frombuffer(f.read(num_merges*4), dtype=np.int32)
        merge_b = np.frombuffer(f.read(num_merges*4), dtype=np.int32)
        decode_len_arr = np.frombuffer(f.read(vocab_size*4), dtype=np.int32)
        decode_data_arr = np.frombuffer(f.read(vocab_size*max_tok_len), dtype=np.uint8).reshape(vocab_size, max_tok_len)
    decode_table = {}
    for t in range(vocab_size):
        dlen = decode_len_arr[t]
        decode_table[t] = bytes(decode_data_arr[t, :dlen])
    return decode_table


def decode_tokens(tokens, decode_table):
    out = b""
    for t in tokens:
        if t in decode_table:
            out += decode_table[t]
    return out.decode("utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════
# LR Schedule
# ═══════════════════════════════════════════════════════════════════════
def get_lr(step, max_lr, min_lr, warmup, total_steps):
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════
def loss_fn(model, inputs, targets):
    logits = model(inputs)
    return mx.mean(nn.losses.cross_entropy(logits, targets))


def train(mode="pretrain"):
    assert mode in ("pretrain", "finetune")
    is_pt = mode == "pretrain"

    print("=" * 64)
    print(f"  NexaLang Transformer v7 — {'Pretraining' if is_pt else 'Fine-tuning'} (MLX GPU)")
    print(f"  {N}-Layer, {H}-Head, D={D}, FF={FF}, V={V} (~33M params)")
    print("=" * 64)

    # Load data
    if is_pt:
        train_data = load_tokens("data/text_pt_v2_bpe/train.bin")
        val_data = load_tokens("data/text_pt_v2_bpe/val.bin")
        steps, max_lr, min_lr, warmup, eval_int, eval_samp = PT_STEPS, PT_MAX_LR, PT_MIN_LR, PT_WARMUP, PT_EVAL_INT, PT_EVAL_SAMPLES
        out_path = model_path("model_v7.bin")
    else:
        train_data = load_tokens("data/instruct_pt_v2_bpe/train.bin")
        val_data = load_tokens("data/instruct_pt_v2_bpe/val.bin")
        steps, max_lr, min_lr, warmup, eval_int, eval_samp = FT_STEPS, FT_MAX_LR, FT_MIN_LR, FT_WARMUP, FT_EVAL_INT, FT_EVAL_SAMPLES
        out_path = model_path("model_v7_ft.bin")

    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens:   {len(val_data):,}")

    # Model
    model = TransformerLM()
    mx.eval(model.parameters())
    n_params = count_params(model)
    print(f"  Parameters:   {n_params:,}")

    if not is_pt:
        pt_path = resolve_existing_path(model_path("model_v7.bin"), "model_v7.bin")
        print(f"\n  Loading pretrained weights from {pt_path}...")
        load_weights_nxl(model, pt_path)

    # Optimizer
    optimizer = optim.AdamW(learning_rate=max_lr, weight_decay=0.01)

    # Training
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"\n  Training for {steps} steps, B={B}, T={T}...")
    t0 = time.time()

    for step in range(steps):
        lr = get_lr(step, max_lr, min_lr, warmup, steps)
        optimizer.learning_rate = lr

        inputs, targets = get_batch(train_data, B, T, step)
        loss, grads = loss_and_grad(model, inputs, targets)
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % eval_int == 0:
            # Validation
            val_losses = []
            for vi in range(eval_samp):
                v_in, v_tgt = get_batch(val_data, B, T, step + vi + 1)
                vl = loss_fn(model, v_in, v_tgt)
                val_losses.append(vl.item())
            val_loss = sum(val_losses) / len(val_losses)

            elapsed = time.time() - t0
            print(f"  Step {step:5d}/{steps}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  lr={lr:.6f}  time={elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Training complete! Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    print(f"\n  Saving weights...")
    save_weights_nxl(model, out_path)

    # Generate sample
    print(f"\n  Sample generation:")
    print("  " + "─" * 58)
    decode_table = load_bpe_decode()
    tokens = [0] * T
    # Seed with [P] Ola\n[R]
    seed_bytes = b"[P] Ola!\n[R] "
    for i, b in enumerate(seed_bytes):
        tokens[T - len(seed_bytes) + i] = b
    ctx = mx.array([tokens])

    gen_tokens = []
    for _ in range(200):
        logits = model(ctx)
        last_logits = logits[0, -1, :] / 0.7  # temperature
        probs = mx.softmax(last_logits, axis=-1)
        next_tok = mx.random.categorical(probs).item()
        gen_tokens.append(next_tok)
        tokens = tokens[1:] + [next_tok]
        ctx = mx.array([tokens])
        # Check double newline stop
        decoded = decode_table.get(next_tok, b"")
        if len(gen_tokens) > 2:
            last_bytes = b""
            for t in gen_tokens[-3:]:
                last_bytes += decode_table.get(t, b"")
            if b"\n\n" in last_bytes:
                break

    text = decode_tokens(gen_tokens, decode_table)
    print(f"  {text}")
    print("  " + "─" * 58)

    return model


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 tools/train_mlx.py [pretrain|finetune|both]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "pretrain":
        train("pretrain")
    elif mode == "finetune":
        train("finetune")
    elif mode == "both":
        train("pretrain")
        print("\n" + "=" * 64)
        print("  Iniciando fine-tuning...")
        print("=" * 64 + "\n")
        train("finetune")
    else:
        print(f"Modo desconhecido: {mode}")
        print("Uso: python3 tools/train_mlx.py [pretrain|finetune|both]")
        sys.exit(1)

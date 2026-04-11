#!/usr/bin/env python3
"""Verify forward pass against Python reference."""
import os
import struct, numpy as np

weights_path = 'artifacts/models/tinyllama.bin'
if not os.path.exists(weights_path):
    weights_path = 'tinyllama.bin'

with open(weights_path, 'rb') as f:
    V, D, H, H_KV, FF, T, N = struct.unpack('iiiiiii', f.read(28))
    HD = D // H
    D_KV = H_KV * HD
    
    tok_emb = np.frombuffer(f.read(V*D*4), dtype=np.float32).reshape(V, D).copy()
    all_Wq = np.frombuffer(f.read(N*D*D*4), dtype=np.float32).reshape(N, D, D).copy()
    all_Wk = np.frombuffer(f.read(N*D*D_KV*4), dtype=np.float32).reshape(N, D, D_KV).copy()
    all_Wv = np.frombuffer(f.read(N*D*D_KV*4), dtype=np.float32).reshape(N, D, D_KV).copy()
    all_Wo = np.frombuffer(f.read(N*D*D*4), dtype=np.float32).reshape(N, D, D).copy()
    all_ln1 = np.frombuffer(f.read(N*D*4), dtype=np.float32).reshape(N, D).copy()
    all_gate = np.frombuffer(f.read(N*D*FF*4), dtype=np.float32).reshape(N, D, FF).copy()
    all_up = np.frombuffer(f.read(N*D*FF*4), dtype=np.float32).reshape(N, D, FF).copy()
    all_down = np.frombuffer(f.read(N*FF*D*4), dtype=np.float32).reshape(N, FF, D).copy()
    all_ln2 = np.frombuffer(f.read(N*D*4), dtype=np.float32).reshape(N, D).copy()
    ln_f = np.frombuffer(f.read(D*4), dtype=np.float32).copy()
    W_out = np.frombuffer(f.read(D*V*4), dtype=np.float32).reshape(D, V).copy()

print(f'Loaded: V={V}, D={D}, H={H}, H_KV={H_KV}, FF={FF}, T={T}, N={N}, HD={HD}, D_KV={D_KV}')

# Token: BOS=1, "Hello"=15043
tokens = [1, 15043]
seq_len = len(tokens)

def rms_norm(x, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return gamma * x / rms

def rope(buf, n_heads, hd):
    T_len = buf.shape[0]
    half = hd // 2
    result = buf.copy()
    for t in range(T_len):
        for h_idx in range(n_heads):
            base = h_idx * hd
            for i in range(half):
                freq = np.exp(-2.0 * i * np.log(10000.0) / hd)
                angle = t * freq
                c, s = np.cos(angle), np.sin(angle)
                x0 = result[t, base + 2*i]
                x1 = result[t, base + 2*i + 1]
                result[t, base + 2*i] = x0 * c - x1 * s
                result[t, base + 2*i + 1] = x0 * s + x1 * c
    return result

scale = 1.0 / np.sqrt(HD)
heads_per_kv = H // H_KV
mask = np.triu(np.ones((seq_len, seq_len)) * -1e5, k=1)

residual = tok_emb[tokens].copy()  # [seq_len, D]
print(f'Embedding [0,:5]: {residual[0,:5]}')

for layer in range(N):
    normed = rms_norm(residual, all_ln1[layer])
    q = normed @ all_Wq[layer]
    k = normed @ all_Wk[layer]
    v = normed @ all_Wv[layer]
    q = rope(q, H, HD)
    k = rope(k, H_KV, HD)
    
    attn_out = np.zeros_like(residual)
    for h_idx in range(H):
        kv_h = h_idx // heads_per_kv
        hq = q[:, h_idx*HD:(h_idx+1)*HD]
        hk = k[:, kv_h*HD:(kv_h+1)*HD]
        hv_local = v[:, kv_h*HD:(kv_h+1)*HD]
        scores = (hq @ hk.T) * scale
        scores += mask
        scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
        attn_out[:, h_idx*HD:(h_idx+1)*HD] = probs @ hv_local
    
    o_out = attn_out @ all_Wo[layer]
    residual = residual + o_out
    normed = rms_norm(residual, all_ln2[layer])
    gate_out = normed @ all_gate[layer]
    up_out = normed @ all_up[layer]
    silu = gate_out / (1.0 + np.exp(-gate_out))
    ffn = silu * up_out
    down_out = ffn @ all_down[layer]
    residual = residual + down_out
    print(f'Layer {layer:2d}: residual[1,:3] = {residual[1,:3]}')

# Final
normed = rms_norm(residual, ln_f)
logits = normed @ W_out
last_logits = logits[-1]
top10 = np.argsort(last_logits)[-10:][::-1]
print(f'\nTop 10 token IDs: {top10}')
print(f'Top 10 logits: {last_logits[top10]}')

# Decode top tokens
tokenizer_path = 'artifacts/models/tinyllama_bpe.bin'
if not os.path.exists(tokenizer_path):
    tokenizer_path = 'tinyllama_bpe.bin'

with open(tokenizer_path, 'rb') as f:
    hdr = struct.unpack('iiiii', f.read(20))
    num_merges, bpe_vocab, max_tok_len, byte_start, sp_space_id = hdr
    merge_a = np.frombuffer(f.read(num_merges*4), dtype=np.int32)
    merge_b = np.frombuffer(f.read(num_merges*4), dtype=np.int32)
    decode_len = np.frombuffer(f.read(bpe_vocab*4), dtype=np.int32)
    decode_data = f.read(bpe_vocab * max_tok_len)

def decode_token(tid):
    dl = decode_len[tid]
    offset = tid * max_tok_len
    raw = decode_data[offset:offset+dl]
    try:
        return raw.decode('utf-8', errors='replace')
    except:
        return f'<{tid}>'

print('\nTop 10 tokens decoded:')
for tid in top10:
    print(f'  {tid:6d}: {last_logits[tid]:8.3f}  "{decode_token(tid)}"')

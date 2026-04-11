#!/usr/bin/env python3
"""Test tokenizer encoding."""
import os
import struct, numpy as np

tokenizer_path = 'artifacts/models/tinyllama_bpe.bin'
if not os.path.exists(tokenizer_path):
    tokenizer_path = 'tinyllama_bpe.bin'

with open(tokenizer_path, 'rb') as f:
    hdr = struct.unpack('iiiii', f.read(20))
    num_merges, bpe_vocab, max_tok_len, byte_start, sp_space_id = hdr
    merge_a = np.frombuffer(f.read(num_merges*4), dtype=np.int32).copy()
    merge_b = np.frombuffer(f.read(num_merges*4), dtype=np.int32).copy()
    decode_len_arr = np.frombuffer(f.read(bpe_vocab*4), dtype=np.int32).copy()
    decode_data = f.read(bpe_vocab * max_tok_len)

print(f'num_merges={num_merges}, vocab={bpe_vocab}, max_tok_len={max_tok_len}, byte_start={byte_start}, sp_space_id={sp_space_id}')

def nxl_bpe_encode(text_bytes):
    output = [sp_space_id]
    for b in text_bytes:
        if b == 32:
            output.append(sp_space_id)
        else:
            output.append(b + byte_start)
    merge_start = byte_start + 256
    for m in range(num_merges):
        a, b = int(merge_a[m]), int(merge_b[m])
        new_tok = merge_start + m
        i = 0
        new_out = []
        while i < len(output):
            if i < len(output) - 1 and output[i] == a and output[i+1] == b:
                new_out.append(new_tok)
                i += 2
            else:
                new_out.append(output[i])
                i += 1
        output = new_out
    return output

def decode_token(tid):
    dl = decode_len_arr[tid]
    offset = tid * max_tok_len
    raw = decode_data[offset:offset+dl]
    return raw.decode('utf-8', errors='replace')

# Test prime context
test = '<|system|>\nYou are a helpful assistant.\n<|user|>\nHello!\n<|assistant|>\nHello! How can I help you?\n'
tokens = nxl_bpe_encode(test.encode('utf-8'))
print(f'Encoded {len(test)} bytes -> {len(tokens)} tokens')
print(f'Token IDs: {tokens[:30]}')
decoded = ''.join(decode_token(t) for t in tokens)
print(f'Decoded: {repr(decoded[:100])}')

# Test user prompt
test2 = '<|user|>\nTell me a short story about a cat\n<|assistant|>\n'
tokens2 = nxl_bpe_encode(test2.encode('utf-8'))
print(f'\nUser prompt: {len(tokens2)} tokens')
print(f'Token IDs: {tokens2}')
decoded2 = ''.join(decode_token(t) for t in tokens2)
print(f'Decoded: {repr(decoded2)}')

# Special tokens
print(f'\nToken 0: {repr(decode_token(0))}, len={decode_len_arr[0]}')
print(f'Token 1 (BOS): {repr(decode_token(1))}, len={decode_len_arr[1]}')
print(f'Token 2 (EOS): {repr(decode_token(2))}, len={decode_len_arr[2]}')

# Run forward pass with context = 512 zeros + prime + prompt
# Show what the actual context looks like
full_ctx = [0] * 512
for t in tokens:
    full_ctx.pop(0)
    full_ctx.append(t)
for t in tokens2:
    full_ctx.pop(0)
    full_ctx.append(t)

# Count leading zeros
n_zeros = sum(1 for t in full_ctx if t == 0)
n_nonzero = 512 - n_zeros
print(f'\nContext: {n_zeros} zeros + {n_nonzero} tokens')
print(f'First non-zero at index: {next(i for i, t in enumerate(full_ctx) if t != 0)}')
print(f'Last 20 context tokens: {full_ctx[-20:]}')
decoded_last = ''.join(decode_token(t) for t in full_ctx[-20:])
print(f'Last 20 decoded: {repr(decoded_last)}')

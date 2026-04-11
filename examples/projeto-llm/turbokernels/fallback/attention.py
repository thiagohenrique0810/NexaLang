"""CPU fallback attention — plain PyTorch implementation."""

import math
import torch


def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      causal: bool = True) -> torch.Tensor:
    """Standard scaled dot-product attention (CPU/MPS fallback).

    Args:
        q, k, v: [batch, heads, seq_len, head_dim]
        causal: apply causal mask

    Returns:
        [batch, heads, seq_len, head_dim]
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        seq_len = q.shape[-2]
        kv_len = k.shape[-2]
        mask = torch.triu(
            torch.ones(seq_len, kv_len, device=q.device, dtype=torch.bool),
            diagonal=kv_len - seq_len + 1
        )
        attn.masked_fill_(mask, float('-inf'))

    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)

"""Fused operations — combined kernels for better performance."""

import torch
import torch.nn.functional as F


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMS normalization."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight).to(x.dtype)


def fused_silu_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Fused SiLU activation with gate multiplication (used in LLaMA MLP)."""
    return F.silu(x) * gate


def fused_rotary_embedding(q: torch.Tensor, k: torch.Tensor,
                           cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q and k."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                        ignore_index: int = -100) -> torch.Tensor:
    """Fused cross-entropy loss (memory efficient — no logits materialization)."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)),
                          targets.view(-1), ignore_index=ignore_index)

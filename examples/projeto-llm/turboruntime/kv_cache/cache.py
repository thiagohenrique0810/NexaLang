"""KV Cache Manager with TurboQuant compression."""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("turboruntime.kv_cache")


@dataclass
class CacheStats:
    total_entries: int = 0
    memory_used_mb: float = 0.0
    memory_saved_mb: float = 0.0
    compression_ratio: float = 1.0
    evictions: int = 0


class KVCache:
    """Standard KV cache with optional TurboQuant compression."""

    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 max_seq_len: int = 2048, device: str = "cpu",
                 dtype: Any = None, compression: str = "none", bits: int = 16):
        import torch

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype or torch.float16
        self.compression = compression
        self.bits = bits
        self.current_len = 0
        self.stats = CacheStats()

        # Pre-allocate cache tensors: [num_layers, 2 (k/v), num_heads, max_seq_len, head_dim]
        if compression == "turboquant" and bits < 16:
            # Compressed cache — store scales + quantized values
            self.keys = [torch.zeros(num_heads, max_seq_len, head_dim, dtype=torch.int8, device=device)
                         for _ in range(num_layers)]
            self.values = [torch.zeros(num_heads, max_seq_len, head_dim, dtype=torch.int8, device=device)
                           for _ in range(num_layers)]
            self.key_scales = [torch.zeros(num_heads, max_seq_len, 1, dtype=torch.float16, device=device)
                               for _ in range(num_layers)]
            self.value_scales = [torch.zeros(num_heads, max_seq_len, 1, dtype=torch.float16, device=device)
                                 for _ in range(num_layers)]
            self._compressed = True
        else:
            self.keys = [torch.zeros(num_heads, max_seq_len, head_dim, dtype=self.dtype, device=device)
                         for _ in range(num_layers)]
            self.values = [torch.zeros(num_heads, max_seq_len, head_dim, dtype=self.dtype, device=device)
                           for _ in range(num_layers)]
            self.key_scales = None
            self.value_scales = None
            self._compressed = False

        uncompressed_bytes = num_layers * 2 * num_heads * max_seq_len * head_dim * 2  # fp16
        actual_bytes = self._calc_memory()
        self.stats.memory_used_mb = actual_bytes / (1024 * 1024)
        self.stats.memory_saved_mb = (uncompressed_bytes - actual_bytes) / (1024 * 1024)
        self.stats.compression_ratio = uncompressed_bytes / max(actual_bytes, 1)

        logger.info(f"KV Cache initialized: {num_layers} layers, {num_heads} heads, "
                     f"head_dim={head_dim}, max_seq={max_seq_len}, "
                     f"compression={compression}({bits}bit), "
                     f"mem={self.stats.memory_used_mb:.1f}MB "
                     f"(saved {self.stats.memory_saved_mb:.1f}MB, {self.stats.compression_ratio:.1f}x)")

    def _calc_memory(self) -> int:
        total = 0
        for k in self.keys:
            total += k.nelement() * k.element_size()
        for v in self.values:
            total += v.nelement() * v.element_size()
        if self.key_scales:
            for s in self.key_scales:
                total += s.nelement() * s.element_size()
        if self.value_scales:
            for s in self.value_scales:
                total += s.nelement() * s.element_size()
        return total

    def update(self, layer_idx: int, key: Any, value: Any, pos: int):
        """Update cache at position for a given layer.
        key/value shape: [num_heads, seq_len, head_dim]
        """
        import torch

        seq_len = key.shape[1]
        end_pos = pos + seq_len

        if end_pos > self.max_seq_len:
            # Evict oldest entries (shift left)
            shift = end_pos - self.max_seq_len
            self._evict(layer_idx, shift)
            pos = self.max_seq_len - seq_len
            end_pos = self.max_seq_len
            self.stats.evictions += 1

        if self._compressed:
            k_scale = key.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            v_scale = value.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            max_val = (1 << (self.bits - 1)) - 1
            self.keys[layer_idx][:, pos:end_pos, :] = torch.clamp(
                torch.round(key / k_scale * max_val), -max_val - 1, max_val
            ).to(torch.int8)
            self.values[layer_idx][:, pos:end_pos, :] = torch.clamp(
                torch.round(value / v_scale * max_val), -max_val - 1, max_val
            ).to(torch.int8)
            self.key_scales[layer_idx][:, pos:end_pos, :] = k_scale.to(torch.float16)
            self.value_scales[layer_idx][:, pos:end_pos, :] = v_scale.to(torch.float16)
        else:
            self.keys[layer_idx][:, pos:end_pos, :] = key
            self.values[layer_idx][:, pos:end_pos, :] = value

        self.current_len = max(self.current_len, end_pos)
        self.stats.total_entries = self.current_len

    def get(self, layer_idx: int, end_pos: int = None):
        """Retrieve cached K,V for a layer up to end_pos."""
        import torch

        if end_pos is None:
            end_pos = self.current_len

        if self._compressed:
            max_val = (1 << (self.bits - 1)) - 1
            k = self.keys[layer_idx][:, :end_pos, :].float()
            v = self.values[layer_idx][:, :end_pos, :].float()
            ks = self.key_scales[layer_idx][:, :end_pos, :].float()
            vs = self.value_scales[layer_idx][:, :end_pos, :].float()
            return (k / max_val * ks).to(self.dtype), (v / max_val * vs).to(self.dtype)
        else:
            return self.keys[layer_idx][:, :end_pos, :], self.values[layer_idx][:, :end_pos, :]

    def _evict(self, layer_idx: int, shift: int):
        """Shift cache left by `shift` positions."""
        import torch
        self.keys[layer_idx] = torch.roll(self.keys[layer_idx], -shift, dims=1)
        self.values[layer_idx] = torch.roll(self.values[layer_idx], -shift, dims=1)
        if self._compressed:
            self.key_scales[layer_idx] = torch.roll(self.key_scales[layer_idx], -shift, dims=1)
            self.value_scales[layer_idx] = torch.roll(self.value_scales[layer_idx], -shift, dims=1)

    def clear(self):
        import torch
        for i in range(self.num_layers):
            self.keys[i].zero_()
            self.values[i].zero_()
            if self._compressed:
                self.key_scales[i].zero_()
                self.value_scales[i].zero_()
        self.current_len = 0
        self.stats = CacheStats()

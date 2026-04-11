"""Tests for TurboRuntime — KV cache and quantizer (no model download needed)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")


class TestKVCache:
    def test_standard_cache(self):
        from turboruntime.kv_cache.cache import KVCache

        cache = KVCache(
            num_layers=2, num_heads=4, head_dim=16,
            max_seq_len=64, device="cpu",
            dtype=torch.float32, compression="none"
        )
        assert cache.current_len == 0

        # Insert
        k = torch.randn(4, 3, 16)  # [heads, seq, dim]
        v = torch.randn(4, 3, 16)
        cache.update(0, k, v, pos=0)
        assert cache.current_len == 3

        # Retrieve
        k_out, v_out = cache.get(0, end_pos=3)
        assert k_out.shape == (4, 3, 16)
        assert torch.allclose(k_out, k, atol=1e-5)

    def test_turboquant_compression(self):
        from turboruntime.kv_cache.cache import KVCache

        cache = KVCache(
            num_layers=2, num_heads=4, head_dim=16,
            max_seq_len=64, device="cpu",
            dtype=torch.float32, compression="turboquant", bits=4
        )
        assert cache._compressed is True

        k = torch.randn(4, 5, 16)
        v = torch.randn(4, 5, 16)
        cache.update(0, k, v, pos=0)

        k_out, v_out = cache.get(0, end_pos=5)
        # Lossy compression — check shape and reasonable values
        assert k_out.shape == (4, 5, 16)
        assert not torch.allclose(k_out, k, atol=1e-3)  # Not exact (compressed)
        # But should be close
        assert torch.allclose(k_out, k, atol=0.5)

    def test_cache_eviction(self):
        from turboruntime.kv_cache.cache import KVCache

        cache = KVCache(
            num_layers=1, num_heads=2, head_dim=8,
            max_seq_len=10, device="cpu", dtype=torch.float32
        )

        # Fill beyond capacity
        for i in range(12):
            k = torch.randn(2, 1, 8)
            v = torch.randn(2, 1, 8)
            cache.update(0, k, v, pos=min(i, 9))

    def test_cache_clear(self):
        from turboruntime.kv_cache.cache import KVCache

        cache = KVCache(
            num_layers=2, num_heads=4, head_dim=16,
            max_seq_len=32, device="cpu", dtype=torch.float32
        )
        k = torch.randn(4, 5, 16)
        v = torch.randn(4, 5, 16)
        cache.update(0, k, v, pos=0)
        cache.clear()
        assert cache.current_len == 0

    def test_compression_memory_savings(self):
        from turboruntime.kv_cache.cache import KVCache

        standard = KVCache(
            num_layers=4, num_heads=8, head_dim=32,
            max_seq_len=256, device="cpu", dtype=torch.float16,
            compression="none"
        )
        compressed = KVCache(
            num_layers=4, num_heads=8, head_dim=32,
            max_seq_len=256, device="cpu", dtype=torch.float16,
            compression="turboquant", bits=4
        )
        # Compressed should use less memory
        assert compressed.stats.memory_used_mb < standard.stats.memory_used_mb


class TestQuantizer:
    def test_int8_quantize(self):
        from turboruntime.quant.quantizer import Quantizer
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        q = Quantizer()
        model_q, stats = q.quantize(model, quant_type="int8")
        assert stats.num_quantized_layers == 2

    def test_int4_quantize(self):
        from turboruntime.quant.quantizer import Quantizer
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(16, 16))
        q = Quantizer()
        model_q, stats = q.quantize(model, quant_type="int4")
        assert stats.num_quantized_layers == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

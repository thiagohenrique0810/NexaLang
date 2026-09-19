"""Native TQ state for paged KV, admitted before loading a library or allocating.

Only the linear immutable codec state persists. Work buffers belong to the
session's memory-planned arena. The struct reserve matches the public C bound;
the runtime checks the exact native size before constructing any context.
"""
import ctypes
from functools import lru_cache
import struct

from .tq import validate_tq_parameters, validate_tq_codebook

TQ_STRUCT_RESERVE_BYTES = 128


def tq_kv_memory(dim, bits):
    validate_tq_parameters(dim, bits, 0)
    levels = 1 << bits
    return {
        "context_reserved_bytes": TQ_STRUCT_RESERVE_BYTES + 4 * dim + 4 * (2 * levels - 1),
        "constructor_staging_bytes": max(8 * levels, 4 * (levels + 1)),
        "vector_scratch_bytes": 4 * dim,
        "accumulator_bytes": 8 * dim,
    }


@lru_cache(maxsize=1)
def _load_tq_kernels():
    from runtime.build_runtime import build_runtime
    lib = ctypes.CDLL(str(build_runtime("nexa_tq_attention")))
    fp, bp, dp = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
                  ctypes.POINTER(ctypes.c_double))
    bpp, sz, ctx, integer = ctypes.POINTER(bp), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int
    signatures = {
        "tq_mse_context_memory_size": ([integer, integer], sz),
        "tq_create_mse": ([integer, integer, integer], ctx),
        "tq_create_mse_from_codebook": ([integer, integer, integer, fp, sz], ctx),
        "tq_export_mse_codebook": ([ctx, fp, sz], integer),
        "tq_context_memory_bytes": ([ctx], sz),
        "tq_destroy": ([ctx], None),
        "tq_quantize_tq02": ([ctx, fp, sz, bp, sz, integer, fp, sz], integer),
        "nexa_causal_gqa_attention_paged_tq": (
            [ctx, fp, sz, bpp, sz, bpp, sz, sz, sz, sz, sz, sz, sz, sz,
             fp, sz, fp, sz, dp, sz, fp, sz], integer),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    return lib


class TQKVContext:
    def __init__(self, dim, bits, seed, codebook_f32le=None):
        validate_tq_parameters(dim, bits, seed)
        values = None if codebook_f32le is None else validate_tq_codebook(codebook_f32le, bits)
        self._ctx = None
        self.library = _load_tq_kernels()
        expected = int(self.library.tq_mse_context_memory_size(dim, bits))
        if not expected or expected > tq_kv_memory(dim, bits)["context_reserved_bytes"]:
            raise ValueError("Native TQ context exceeds the admitted struct/state reserve")
        staging = None
        try:
            if values is None:
                context = self.library.tq_create_mse(dim, bits, seed)
            else:
                staging = (ctypes.c_float * len(values))(*values)
                context = self.library.tq_create_mse_from_codebook(dim, bits, seed, staging, len(values))
            if not context:
                raise MemoryError("Unable to allocate the admitted TQ KV context")
            self._ctx = context
            staging = None
            self.state_bytes = int(self.library.tq_context_memory_bytes(context))
            if self.state_bytes != expected:
                raise ValueError("Native TQ context accounting differs from its preflight")
            staging = (ctypes.c_float * (1 << bits))()
            if self.library.tq_export_mse_codebook(context, staging, len(staging)):
                raise ArithmeticError("Unable to export the TQ KV codebook")
            self.codebook_f32le = struct.pack("<" + "f" * len(staging), *staging).hex()
            validate_tq_codebook(self.codebook_f32le, bits)
        except BaseException:
            self.close()
            raise
        finally:
            staging = values = None

    def close(self):
        if self._ctx is not None:
            self.library.tq_destroy(self._ctx)
            self._ctx = None


class TQKernelDispatch:
    """Keep the existing timed call path and legacy library ABI unchanged."""
    def __init__(self, transformer, tq):
        self.transformer, self.tq = transformer, tq

    def __getattr__(self, name):
        library = self.tq if name in ("tq_quantize_tq02", "nexa_causal_gqa_attention_paged_tq") else self.transformer
        return getattr(library, name)

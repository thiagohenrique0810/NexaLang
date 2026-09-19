"""Portable TQ MSE rows and bounded, one-vector native CPU execution.

TQ02 stores an explicit little-endian F32 norm and LSB-first indices. Decoding
also needs the dimension, bit width, seed and exact persisted F32 codebook.
Importing this module or using validation helpers never loads native code.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import math
import struct
import threading

MAX_TQ_DIM = 1 << 20
TQ_CODEC_ID = "TQ_MSE_SRHT"
TQ_CODEC_VERSION = 1
TQ_TRANSFORM_ID = "SRHT_XOSHIRO256SS_V1"


def validate_tq_parameters(dim, bits, seed):
    """Validate portable limits and return (dim, bits, seed). Booleans fail."""
    for value, name in ((dim, "dim"), (bits, "bits"), (seed, "seed")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"TQ {name} must be an integer")
    if not 1 <= dim <= MAX_TQ_DIM or dim & (dim - 1):
        raise ValueError(f"TQ dim must be a power of two in [1, {MAX_TQ_DIM}]")
    if not 1 <= bits <= 8:
        raise ValueError("TQ bits must be in [1, 8]")
    if not -(1 << 31) <= seed < (1 << 31):
        raise ValueError("TQ seed must be a signed int32")
    return dim, bits, seed


def tq_row_bytes(dim, bits):
    """Return the complete independently addressable TQ02 row length."""
    validate_tq_parameters(dim, bits, 0)
    return 8 + (dim * bits + 7) // 8


def validate_tq_codebook(codebook_f32le, bits):
    """Return centroids from exact F32LE hex; reject unsafe legacy midpoints."""
    validate_tq_parameters(1, bits, 0)
    if not isinstance(codebook_f32le, str) or len(codebook_f32le) != (1 << bits) * 8:
        raise ValueError("TQ codebook_f32le must contain exactly 2^bits F32 hex values")
    if any(c not in "0123456789abcdef" for c in codebook_f32le):
        raise ValueError("TQ codebook_f32le must be lowercase hexadecimal without whitespace")
    values = struct.unpack("<" + "f" * (1 << bits), bytes.fromhex(codebook_f32le))
    for i, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError("TQ codebook centroids must be finite")
        if i:
            if not values[i - 1] < value:
                raise ValueError("TQ codebook centroids must be strictly increasing")
            try:
                midpoint_sum = struct.unpack("<f", struct.pack("<f", values[i - 1] + value))[0]
            except OverflowError as exc:
                raise ValueError("TQ codebook midpoint sum overflows F32") from exc
            if not math.isfinite(midpoint_sum):
                raise ValueError("TQ codebook midpoint sum overflows F32")
    return values


def _byte_view(data):
    try:
        return memoryview(data).cast("B")
    except (TypeError, ValueError) as exc:
        raise ValueError("TQ row must be a contiguous bytes-like object") from exc


def validate_tq_row(data, dim, bits):
    """Validate canonical TQ02 bytes without decoding; return None."""
    required = tq_row_bytes(dim, bits)
    row = _byte_view(data)
    try:
        if len(row) != required:
            raise ValueError(f"TQ row must contain exactly {required} bytes")
        if row[:4] != b"TQ02":
            raise ValueError("Invalid TQ02 magic")
        norm_bits = struct.unpack_from("<I", row, 4)[0]
        if norm_bits & 0x80000000 or norm_bits & 0x7F800000 == 0x7F800000:
            raise ValueError("TQ norm must be finite, nonnegative and not negative zero")
        tail = (dim * bits) % 8
        if tail and row[-1] >> tail:
            raise ValueError("TQ index padding bits must be zero")
        if norm_bits == 0 and any(row[8:]):
            raise ValueError("TQ zero norm requires all-zero indices")
    finally:
        row.release()
        row = data = None


def migrate_tq01_row(data, dim, bits, *, source_endianness):
    """Change canonical legacy magic/norm byte order, without requantization.

    The caller must persist the original codebook/seed separately. Endianness
    cannot be inferred. Noncanonical legacy negative zero is rejected.
    """
    if source_endianness not in ("little", "big"):
        raise ValueError("source_endianness must be 'little' or 'big'")
    row = _byte_view(data)
    norm = migrated = None
    try:
        if len(row) != tq_row_bytes(dim, bits) or row[:4] != b"TQ01":
            raise ValueError("Invalid legacy TQ01 row length or magic")
        norm = bytes(row[4:8])
        if source_endianness == "big":
            norm = norm[::-1]
        migrated = b"TQ02" + norm + bytes(row[8:])
        validate_tq_row(migrated, dim, bits)
        return migrated
    finally:
        row.release()
        row = migrated = norm = data = None


@lru_cache(maxsize=1)
def _load_library():
    from runtime.build_runtime import build_runtime

    library = ctypes.CDLL(str(build_runtime("turboquant")))
    fp, bp = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)
    sz, ctx, integer = ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int
    signatures = {
        "tq_mse_context_memory_size": ([integer, integer], sz),
        "tq_create_mse": ([integer, integer, integer], ctx),
        "tq_create_mse_from_codebook": ([integer, integer, integer, fp, sz], ctx),
        "tq_export_mse_codebook": ([ctx, fp, sz], integer),
        "tq_context_memory_bytes": ([ctx], sz),
        "tq_destroy": ([ctx], None),
        "tq_quantize_tq02": ([ctx, fp, sz, bp, sz, integer, fp, sz], integer),
        "tq_dequantize_tq02": ([ctx, bp, sz, fp, sz, integer], integer),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(library, name)
        fn.argtypes, fn.restype = args, result
    return library


class TQCodec:
    """One-vector codec with a preflighted managed native-buffer budget.

    The report excludes Python objects/metadata, allocator overhead, loaded
    libraries, stack, caller input and the decoded consumer list; it is not RSS.
    This wrapper serializes operations and close, so concurrent callers cannot
    multiply its one-vector memory bound. Native contexts themselves are immutable.
    """

    def __init__(self, dim, bits=3, seed=42, *, codebook_f32le=None, memory_budget=None):
        self._lock = threading.RLock()
        self._active = False
        self._dim, self._bits, self._seed = validate_tq_parameters(dim, bits, seed)
        self._ctx = None
        levels = 1 << bits
        imported = None if codebook_f32le is None else validate_tq_codebook(codebook_f32le, bits)
        budget = None
        if memory_budget is not None:
            from compiler.planner.memory import parse_memory_size
            if isinstance(memory_budget, bool):
                raise ValueError("memory_budget must be a byte count or memory size")
            budget = parse_memory_size(memory_budget)
        self._library = _load_library()
        context_bytes = int(self._library.tq_mse_context_memory_size(dim, bits))
        if not context_bytes:
            raise ValueError("Native TQ context size overflow")
        row_bytes = tq_row_bytes(dim, bits)
        # Export/import staging has a ctypes F32 codebook and F32LE bytes.
        constructor_temporary = max(8 * levels, 4 * (levels + 1) if imported is None else 0)
        encode_peak = context_bytes + 8 * dim + 2 * row_bytes
        decode_peak = context_bytes + 4 * dim + row_bytes
        constructor_peak = context_bytes + constructor_temporary
        peak = max(encode_peak, decode_peak, constructor_peak)
        self._memory = {
            "context_bytes": context_bytes,
            "state_bytes": context_bytes,
            "scratch_bytes": 4 * dim,
            "input_bytes": 4 * dim,
            "output_bytes": 4 * dim,
            "packed_bytes": row_bytes,
            "packed_return_copy_bytes": row_bytes,
            "codebook_staging_bytes": 8 * levels,
            "constructor_temporary_bytes": constructor_temporary,
            "constructor_peak_bytes": constructor_peak,
            "encode_peak_bytes": encode_peak,
            "decode_peak_bytes": decode_peak,
            "managed_buffers_peak_bound_bytes": peak,
            "budget_bytes": budget,
            "scope": "managed native buffers; excludes Python metadata, caller inputs, consumer lists, allocator overhead and libraries",
        }
        if budget is not None and peak > budget:
            raise MemoryError(f"TQ codec requires {peak} managed bytes; budget is {budget}")
        codebook = None
        try:
            if imported is None:
                context = self._library.tq_create_mse(dim, bits, seed)
            else:
                codebook = (ctypes.c_float * levels)(*imported)
                context = self._library.tq_create_mse_from_codebook(dim, bits, seed, codebook, levels)
        finally:
            codebook = None
        if not context:
            raise MemoryError("Unable to allocate TQ MSE context")
        self._ctx = context
        try:
            codebook = (ctypes.c_float * levels)()
            status = self._library.tq_export_mse_codebook(context, codebook, levels)
            if status:
                raise ArithmeticError(f"TQ codebook export failed (status {status})")
            self._codebook_f32le = struct.pack("<" + "f" * levels, *codebook).hex()
            validate_tq_codebook(self._codebook_f32le, bits)
        except BaseException:
            self.close()
            raise
        finally:
            codebook = None

    @property
    def dim(self):
        return self._dim

    @property
    def bits(self):
        return self._bits

    @property
    def seed(self):
        return self._seed

    @property
    def codebook_f32le(self):
        return self._codebook_f32le

    def memory_report(self):
        return dict(self._memory)

    def _require_open(self):
        if self._ctx is None:
            raise ValueError("TQ codec is closed")

    def encode_row(self, values):
        with self._lock:
            if self._active:
                raise RuntimeError("TQ codec operations cannot reenter the same codec")
            self._active = True
            try:
                return self._encode_row(values)
            finally:
                self._active = False

    def _encode_row(self, values):
        self._require_open()
        inputs = scratch = packed = None
        try:
            # Iteration is bounded to dim+1 even for infinite generators.
            inputs = (ctypes.c_float * self.dim)()
            try:
                iterator = iter(values)
            except TypeError as exc:
                raise ValueError("TQ row must be iterable") from exc
            for i in range(self.dim):
                try:
                    value = next(iterator)
                except StopIteration as exc:
                    raise ValueError(f"TQ row requires exactly {self.dim} values") from exc
                if isinstance(value, bool):
                    raise ValueError("TQ input values must be finite F32 numbers")
                try:
                    inputs[i] = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("TQ input values must be finite F32 numbers") from exc
                if not math.isfinite(inputs[i]):
                    raise ValueError("TQ input values must be finite F32 numbers")
            marker = object()
            if next(iterator, marker) is not marker:
                raise ValueError(f"TQ row requires exactly {self.dim} values")
            scratch = (ctypes.c_float * self.dim)()
            packed = (ctypes.c_uint8 * self._memory["packed_bytes"])()
            status = self._library.tq_quantize_tq02(
                self._ctx, inputs, self.dim, packed, len(packed), 1, scratch, self.dim)
            if status:
                raise ArithmeticError(f"TQ02 encode failed (status {status})")
            return bytes(packed)
        finally:
            inputs = scratch = packed = None

    def decode_row(self, data):
        with self._lock:
            if self._active:
                raise RuntimeError("TQ codec operations cannot reenter the same codec")
            self._active = True
            try:
                return self._decode_row(data)
            finally:
                self._active = False

    def _decode_row(self, data):
        self._require_open()
        validate_tq_row(data, self.dim, self.bits)
        packed = output = None
        try:
            packed = (ctypes.c_uint8 * self._memory["packed_bytes"]).from_buffer_copy(data)
            output = (ctypes.c_float * self.dim)()
            status = self._library.tq_dequantize_tq02(
                self._ctx, packed, len(packed), output, self.dim, 1)
            if status:
                raise ArithmeticError(f"TQ02 decode failed (status {status})")
            return list(output)
        finally:
            packed = output = None

    def close(self):
        with self._lock:
            if self._active:
                raise RuntimeError("Cannot close a TQ codec from its active input iterator")
            if getattr(self, "_ctx", None) is not None:
                self._library.tq_destroy(self._ctx)
                self._ctx = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

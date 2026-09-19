"""CPU Q4 tile execution under an explicit managed-buffer budget.

This is a matrix executor, not full-model inference or a process RSS limiter.
The C kernel consumes packed groups directly. Python orchestrates bounded I/O.
"""
from __future__ import annotations

import ctypes
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from compiler.hardware_profile import HardwareProfile
from compiler.model_ir import ModelGraph, ModelOp, TensorDesc
from compiler.planner.memory import MemoryPlanner, MemoryRequest, parse_memory_size
from runtime.build_runtime import build_runtime
from runtime.nexapack.format import (
    CODEC_ID, CODEC_VERSION, NexaPackError, NexaPackReader, READ_CHUNK_BYTES, MAX_READ_BYTES,
)

ALIGNMENT = 64


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@lru_cache(maxsize=8)
def _load_kernel(library_path=None):
    library = ctypes.CDLL(str(library_path or build_runtime("nexa_q4")))
    fn = library.nexa_q4_matmul
    fp = ctypes.POINTER(ctypes.c_float)
    bp = ctypes.POINTER(ctypes.c_uint8)
    sz = ctypes.c_size_t
    fn.argtypes = [fp, sz, sz, bp, sz, sz, sz, sz, fp, sz]
    fn.restype = ctypes.c_int
    # Keep the library alive while calls through its exported symbol are possible.
    return library, fn


def make_plan(reader, batch, tile_rows, memory_budget, reserve_bytes=0):
    """Preflight without reading any weight payload or loading native code."""
    if (getattr(reader, 'codec_id', None) != CODEC_ID
            or type(getattr(reader, 'codec_version', None)) is not int
            or reader.codec_version != CODEC_VERSION):
        raise NexaPackError('CPU matrix execution supports only Q4_GROUPED version 1')
    _positive_int(batch, "batch")
    _positive_int(tile_rows, "tile_rows")
    budget = parse_memory_size(memory_budget)
    reserve = parse_memory_size(reserve_bytes)
    rows = min(tile_rows, reader.rows, MAX_READ_BYTES // reader.row_bytes)
    requests = [
        MemoryRequest("input", batch * reader.cols * 4, 0, 3, ALIGNMENT),
        MemoryRequest("packed_tile", rows * reader.row_bytes, 1, 2, ALIGNMENT),
        MemoryRequest("output_tile", batch * rows * 4, 1, 3, ALIGNMENT),
    ]
    # The reader's fixed checksum scratch lives outside the arena. Base alignment
    # requires at most 63 extra bytes; both are included in the same budget.
    overhead = READ_CHUNK_BYTES + ALIGNMENT - 1
    plan = MemoryPlanner().plan(
        requests, budgets={"host": budget}, reserves={"host": reserve + overhead}
    )
    return plan, rows, budget, reserve


def _graph(rows, cols, batch, packed_bytes):
    graph = ModelGraph(
        name="q4_streamed_matmul",
        tensors=[
            TensorDesc("input", (batch, cols)),
            TensorDesc("weights", (rows, cols), storage_dtype="q4", storage_nbytes=packed_bytes),
            TensorDesc("output", (batch, rows)),
        ],
        ops=[ModelOp("matmul", "MatMul", ["input", "weights"], ["output"],
                     {"transpose_b": True})],
        inputs=["input"], outputs=["output"], constants=["weights"],
    )
    graph.validate()
    return graph


def _check_tile(packed, inputs, outputs, batch, rows, cols, group_size, row_bytes):
    """Independent scalar reference, unpacking one value at a time (no matrix)."""
    group_bytes = 4 + (group_size + 1) // 2
    largest = 0.0
    for b in range(batch):
        for r in range(rows):
            total = 0.0
            for c in range(cols):
                group, lane = divmod(c, group_size)
                offset = r * row_bytes + group * group_bytes
                scale = struct.unpack_from("<f", packed, offset)[0]
                q = (packed[offset + 4 + lane // 2] >> (4 * (lane % 2))) & 15
                if q & 8:
                    q -= 16
                total += float(inputs[b * cols + c]) * (float(scale) * q)
            reference = struct.unpack("<f", struct.pack("<f", total))[0]
            actual = float(outputs[b * rows + r])
            error = abs(actual - reference)
            largest = max(largest, error)
            if not math.isfinite(actual) or error > 1e-5 * max(1.0, abs(reference)):
                raise ArithmeticError(f"Q4 reference mismatch at batch={b}, row={r}: {actual} != {reference}")
    return largest


@contextmanager
def _open_matrix(path, tensor_name):
    if tensor_name is None:
        with NexaPackReader(path) as reader:
            yield reader
    else:
        from runtime.nexapack.bundle import ModelBundleReader
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("tensor_name must be a nonempty string")
        with ModelBundleReader(path) as bundle:
            with bundle.open_q4(tensor_name) as reader:
                yield reader


def run_packed_matmul(path, *, batch=1, tile_rows=64, memory_budget="512MiB",
                      reserve_bytes=0, verify=False, library_path=None, on_tile=None,
                      tensor_name=None):
    """Run deterministic inputs against a packed matrix, retaining only one tile.

    on_tile is an optional synchronous output consumer. It receives a ctypes view
    which is reused on the next iteration. Memory retained by a caller is outside
    this executor's budget. Verification checks arithmetic against decoded Q4;
    this is not a measurement of model quality or error versus original weights.
    """
    if not isinstance(verify, bool):
        raise ValueError("verify must be a bool")
    path = Path(path)
    with _open_matrix(path, tensor_name) as reader:
        plan, tile_rows, budget, reserve = make_plan(
            reader, batch, tile_rows, memory_budget, reserve_bytes
        )
        metadata_digest = hashlib.sha256(json.dumps(
            reader.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        graph = _graph(reader.rows, reader.cols, batch, reader.rows * reader.row_bytes)
        _, kernel = _load_kernel(str(library_path) if library_path else None)
        arena_extent = plan.peak_bytes["host"]
        arena_size = arena_extent + ALIGNMENT - 1
        arena = (ctypes.c_uint8 * arena_size)()
        raw_address = ctypes.addressof(arena)
        address = (raw_address + ALIGNMENT - 1) & -ALIGNMENT
        allocations = plan.allocations
        inputs = (ctypes.c_float * (batch * reader.cols)).from_address(address + allocations["input"].offset)
        packed = (ctypes.c_uint8 * (tile_rows * reader.row_bytes)).from_address(address + allocations["packed_tile"].offset)
        outputs = (ctypes.c_float * (batch * tile_rows)).from_address(address + allocations["output_tile"].offset)
        packed_view = memoryview(packed).cast("B")
        for i in range(len(inputs)):
            inputs[i] = ((i % 17) - 8) / 8.0

        read_seconds = compute_seconds = verification_seconds = consumer_seconds = 0.0
        max_error = 0.0
        # Per-batch digests make the checksum independent of the chosen tile size.
        digests = [hashlib.sha256() for _ in range(batch)]
        tiles = 0
        started = time.perf_counter()
        for start in range(0, reader.rows, tile_rows):
            count = min(tile_rows, reader.rows - start)
            size = count * reader.row_bytes
            tick = time.perf_counter()
            reader.read_rows_into(start, count, packed_view[:size])
            read_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            status = kernel(inputs, len(inputs), batch, packed, size, count,
                            reader.cols, reader.group_size, outputs, batch * count)
            compute_seconds += time.perf_counter() - tick
            if status:
                raise ArithmeticError(f"Native Q4 matmul failed (status {status}) at row {start}")
            tick = time.perf_counter()
            if verify:
                max_error = max(max_error, _check_tile(
                    packed_view, inputs, outputs, batch, count, reader.cols,
                    reader.group_size, reader.row_bytes
                ))
            verification_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            for b, digest in enumerate(digests):
                for r in range(count):
                    digest.update(struct.pack("<f", outputs[b * count + r]))
            if on_tile is not None:
                on_tile(start, count, batch, outputs)
            consumer_seconds += time.perf_counter() - tick
            tiles += 1
        elapsed = time.perf_counter() - started
        managed_peak = arena_size + READ_CHUNK_BYTES
        assert managed_peak + reserve <= budget
        return {
            "schema_version": 1,
            "backend": "cpu",
            "workload": "q4_streamed_matmul",
            "pack": str(path.resolve()),
            "source": {"kind": "model_bundle" if tensor_name is not None else "q4_matrix",
                       "path": str(path.resolve()), "tensor": tensor_name},
            "pack_metadata_sha256": metadata_digest,
            "input_pattern": "float32((flat_index % 17 - 8) / 8)",
            "shape": {"batch": batch, "rows": reader.rows, "cols": reader.cols},
            "group_size": reader.group_size,
            "tile_rows": tile_rows,
            "tiles_executed": tiles,
            "model_ir": graph.to_dict(),
            "memory_plan": plan.to_dict(),
            "hardware": HardwareProfile.detect_cpu().to_dict(),
            "memory": {
                "scope": "managed CPU arena plus bounded reader scratch; not process RSS or GPU VRAM",
                "budget_bytes": budget,
                "user_reserve_bytes": reserve,
                "arena_extent_bytes": arena_extent,
                "arena_allocation_bytes": arena_size,
                "reader_scratch_capacity_bytes": READ_CHUNK_BYTES,
                "managed_buffers_peak_bound_bytes": managed_peak,
                "weights_packed_bytes": reader.rows * reader.row_bytes,
                "weights_f32_equivalent_bytes": reader.rows * reader.cols * 4,
                "peak_vram_bytes": None,
                "peak_rss_bytes": None,
                "full_dequantized_weight_buffer_bytes": 0,
                "excluded": ["Python objects and metadata", "interpreter and native libraries", "OS file cache", "caller-retained outputs"],
            },
            "io": {"file_payload_bytes_read": reader.payload_bytes_read,
                   "packed_bytes_consumed_by_kernel": reader.rows * reader.row_bytes,
                   "host_to_gpu_bytes": 0},
            "timing": {"read_seconds": read_seconds, "compute_seconds": compute_seconds,
                       "verification_seconds": verification_seconds,
                       "output_consumer_seconds": consumer_seconds,
                       "execution_wall_seconds": elapsed},
            "validation": {"reference": "scalar decoded Q4" if verify else None,
                           "verified": verify, "max_abs_error": max_error if verify else None,
                           "model_quality_measured": False},
            "output_sha256_by_batch": [digest.hexdigest() for digest in digests],
        }

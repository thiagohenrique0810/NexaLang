"""Native CPU Llama graph execution over streamed Q4 model bundles.

This baseline recomputes the prefix; incremental.py adds persistent KV execution.
The memory budget covers the planned arena and reader scratch, not Python, RSS,
tokenizer assets, or lists of logits retained by the caller. PyTorch is not used.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

from compiler.hardware_profile import HardwareProfile
from compiler.model_ir import TensorDesc
from compiler.model_lowering import lower_model, derive_activation_requests
from compiler.planner.memory import (MemoryAllocation, MemoryPlan, MemoryPlanner,
                                     MemoryRequest, parse_memory_size)
from runtime.build_runtime import build_runtime
from .bundle import ModelBundleReader
from .format import MAX_READ_BYTES, READ_CHUNK_BYTES

ALIGNMENT = 64
# Matmul and single-row decode kernels for each packed weight codec.
_PACKED_KERNELS = {"Q4_GROUPED": ("nexa_q4_matmul", "nexa_q4_decode_row"),
                   "Q8_GROUPED": ("nexa_q8_matmul", "nexa_q8_decode_row"),
                   "Q3_GROUPED": ("nexa_q3_matmul", "nexa_q3_decode_row")}


@lru_cache(maxsize=1)
def _load_kernels():
    lib = ctypes.CDLL(str(build_runtime("nexa_transformer")))
    fp, bp, sz, dbl = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
                       ctypes.c_size_t, ctypes.c_double)
    fpp = ctypes.POINTER(fp)
    bpp = ctypes.POINTER(bp)
    signatures = {
        "nexa_q4_matmul": [fp, sz, sz, bp, sz, sz, sz, sz, fp, sz],
        "nexa_q4_decode_row": [bp, sz, sz, sz, fp, sz],
        "nexa_f32_matmul": [fp, sz, sz, fp, sz, sz, sz, fp, sz],
        "nexa_q8_matmul": [fp, sz, sz, bp, sz, sz, sz, sz, fp, sz],
        "nexa_q8_decode_row": [bp, sz, sz, sz, fp, sz],
        "nexa_q3_matmul": [fp, sz, sz, bp, sz, sz, sz, sz, fp, sz],
        "nexa_q3_decode_row": [bp, sz, sz, sz, fp, sz],
        "nexa_q4_quantize": [fp, sz, sz, sz, sz, bp, sz],
        "nexa_q3_quantize": [fp, sz, sz, sz, sz, bp, sz],
        "nexa_rmsnorm": [fp, sz, fp, sz, sz, sz, dbl, fp, sz],
        "nexa_rope": [fp, sz, sz, sz, sz, dbl, fp, sz],
        "nexa_rope_offset": [fp, sz, sz, sz, sz, dbl, sz, fp, sz],
        "nexa_swiglu": [fp, sz, fp, sz, sz, fp, sz],
        "nexa_add": [fp, sz, fp, sz, sz, fp, sz],
        "nexa_causal_gqa_attention": [fp, sz, fp, sz, fp, sz, sz, sz, sz, sz, fp, sz, fp, sz],
        "nexa_causal_gqa_attention_cached": [fp, sz, fp, sz, fp, sz, sz, sz, sz, sz, sz, fp, sz, fp, sz],
        "nexa_causal_gqa_attention_paged": [fp, sz, fpp, sz, fpp, sz, sz, sz, sz, sz, sz, sz, sz, fp, sz, fp, sz],
        "nexa_causal_gqa_attention_paged_q4": [fp, sz, bpp, sz, bpp, sz, sz, sz, sz, sz, sz, sz, sz, sz, fp, sz, fp, sz],
        "nexa_causal_gqa_attention_paged_q3": [fp, sz, bpp, sz, bpp, sz, sz, sz, sz, sz, sz, sz, sz, sz, fp, sz, fp, sz],
        "nexa_causal_gqa_attention_paged_mixed": [fp, sz, bpp, sz, bpp, sz, bp, sz,
                                                 ctypes.POINTER(sz), sz, sz, sz, sz, sz,
                                                 sz, sz, sz, fp, sz, fp, sz],
        "nexa_kv_reencode_rows": [bp, sz, ctypes.c_int, ctypes.c_int, sz, sz, sz,
                                  bp, sz, fp, sz, ctypes.POINTER(dbl), sz],
        "nexa_causal_gqa_attention_page": [fp, sz, bp, sz, bp, sz, ctypes.c_int,
                                           sz, sz, sz, sz, sz, sz, sz, sz, ctypes.c_int,
                                           ctypes.POINTER(dbl), sz, ctypes.POINTER(dbl), sz],
        "nexa_causal_gqa_attention_finish": [ctypes.POINTER(dbl), sz, ctypes.POINTER(dbl), sz,
                                             sz, sz, sz, fp, sz],
    }
    for name, signature in signatures.items():
        function = getattr(lib, name)
        function.argtypes = signature
        function.restype = ctypes.c_int
    return lib


def _positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


class TransformerSession:
    """Single-sequence, causal Llama evaluation with transactional token history.

    prefill replaces history and returns every position's logits; decode appends
    one ID and returns its logits. Both execute a full-prefix graph. Returned
    Python lists are caller-owned and excluded from the managed-buffer budget.
    Instances are synchronous and must not be used concurrently.
    """
    def __init__(self, bundle_path, *, memory_budget="512MiB", max_sequence_length=None,
                 tile_rows=32, reserve_bytes=0):
        self._closed = False
        self._tokens = ()
        self._last_report = None
        self._workspace_template = None
        self.path = Path(bundle_path)
        self.budget = parse_memory_size(memory_budget)
        self.reserve = parse_memory_size(reserve_bytes)
        self.tile_rows = _positive(tile_rows, "tile_rows")
        if sys.byteorder != "little":
            raise ValueError("The native Transformer arena currently requires a little-endian host")
        self._bundle = ModelBundleReader(self.path)
        try:
            self.config = self._bundle.config
            if self.config.vocab_size > (1 << 32):
                raise ValueError("Token IDs must fit the U32 graph input")
            self.max_sequence_length = _positive(
                self.config.max_position_embeddings if max_sequence_length is None else max_sequence_length,
                "max_sequence_length")
            if self.max_sequence_length > self.config.max_position_embeddings:
                raise ValueError("Requested sequence capacity exceeds the model context")
            summary = self._bundle.inspect()
            self._storage = {}
            self._matrix_layouts = {}
            self._dense_matrices = set()
            self._packed_kernels = {}
            for item in summary["tensors"]:
                name, shape = item["name"], tuple(item["shape"])
                q4 = item["codec"] in _PACKED_KERNELS
                dense = item["codec"] == "RAW_F32_MATRIX"
                if q4:
                    self._packed_kernels[name] = _PACKED_KERNELS[item["codec"]]
                self._storage[name] = TensorDesc(name, shape, storage_dtype="q4" if q4 else "f32",
                                                 storage_nbytes=item["packed_payload_bytes"])
                if q4:
                    row_bytes = item["packed_payload_bytes"] // shape[0]
                    rows = min(self.tile_rows, shape[0], MAX_READ_BYTES // row_bytes)
                    if not rows:
                        raise ValueError("Packed weight row exceeds the reader limit")
                    self._matrix_layouts[name] = (rows, row_bytes)
                elif dense:
                    # A dense matrix is verified per stored block, so the block
                    # the writer chose is also the tile this reader consumes.
                    blocks = self._bundle.matrix_blocks(name)
                    row_bytes = shape[1] * 4
                    rows = max(block["row_count"] for block in blocks)
                    if rows * row_bytes > MAX_READ_BYTES:
                        raise ValueError("Dense weight block exceeds the reader limit; convert with fewer block_rows")
                    self._matrix_layouts[name] = (rows, row_bytes)
                    self._dense_matrices.add(name)
            self._weights_bytes = summary["packed_payload_bytes"]
            manifest = json.dumps(self._bundle.manifest, sort_keys=True, separators=(",", ":"))
            self._manifest_sha256 = hashlib.sha256(manifest.encode()).hexdigest()
            # Validate the declared session capacity before any weights or native
            # library are loaded, and before allocating an activation arena.
            graph, plan = self._make_plan(self.max_sequence_length)
            self._capacity_plan = plan.to_dict()
            self._last_report = self._report(graph, plan, executed=False)
        except BaseException:
            self._bundle.close()
            self._closed = True
            raise

    @property
    def token_ids(self):
        return self._tokens

    def _check_open(self):
        if self._closed:
            raise ValueError("Transformer session is closed")

    def _validate_tokens(self, token_ids):
        if not isinstance(token_ids, (list, tuple)) or not token_ids:
            raise ValueError("token_ids must be a nonempty list or tuple")
        if len(token_ids) > self.max_sequence_length:
            raise ValueError("Token sequence exceeds the session context capacity")
        for token in token_ids:
            if type(token) is not int or not 0 <= token < self.config.vocab_size:
                raise ValueError("Every token ID must be an integer in the model vocabulary")
        return tuple(token_ids)

    def _make_plan(self, length, *, execution_context=None):
        if execution_context is not None:
            raise ValueError("The baseline executor does not accept a stateful execution plan")
        graph = lower_model(self.config, length, weight_storage=self._storage)
        requests = list(derive_activation_requests(graph))
        end = len(graph.ops) + 2
        return graph, self._plan_workspace(length, requests, end, attention_length=length)

    def _plan_workspace(self, length, requests, end, *, attention_length, extra_reserve=0):
        requests = list(requests)
        max_packed = max(rows * row_bytes for rows, row_bytes in self._matrix_layouts.values())
        max_rows = max(rows for rows, _ in self._matrix_layouts.values())
        for name, size in (("__packed_tile", max_packed), ("__output_tile", length * max_rows * 4),
                           ("__norm", self.config.hidden_size * 4), ("__attention", attention_length * 4)):
            requests.append(MemoryRequest(name, size, 0, end))
        reserves = {"host": self.reserve + READ_CHUNK_BYTES + ALIGNMENT - 1 + extra_reserve}
        if self._workspace_template is None:
            # Preflight uses the largest declared chunk and attention prefix.
            # Keep these offsets: first-fit fragmentation is not monotonic as
            # buffer sizes shrink, so replanning could exceed this accepted cap.
            plan = MemoryPlanner().plan(requests, {"host": self.budget}, reserves)
            self._workspace_template = plan
            return plan
        template = self._workspace_template
        if ({request.name for request in requests} != set(template.allocations)
                or reserves != dict(template.reserves)):
            raise ValueError("Workspace requests differ from the session capacity plan")
        allocations = {}
        for request in requests:
            capacity = template.allocations[request.name]
            if (request.size_bytes > capacity.size_bytes
                    or (request.start, request.end, request.alignment, request.tier) !=
                       (capacity.start, capacity.end, capacity.alignment, capacity.tier)):
                raise ValueError("Workspace request exceeds its session capacity contract")
            allocations[request.name] = MemoryAllocation(**request.to_dict(), offset=capacity.offset)
        extent = max(allocation.offset + allocation.size_bytes for allocation in allocations.values())
        return MemoryPlan(allocations, {"host": extent}, template.budgets, reserves).validate(requests)

    def _report(self, graph, plan, *, executed, execution_context=None):
        extent = plan.peak_bytes["host"]
        return {
            "schema_version": 1, "workload": "llama_causal_forward", "backend": "cpu_native",
            "bundle": str(self.path.resolve()), "manifest_sha256": self._manifest_sha256,
            "config": self.config.to_dict(), "executed": executed,
            "decode_strategy": "full_prefix_recomputation", "persistent_kv_cache": False,
            "max_sequence_length": self.max_sequence_length,
            "model_ir": graph.to_dict(), "memory_plan": plan.to_dict(),
            "hardware": HardwareProfile.detect_cpu().to_dict(),
            "memory": {
                "scope": "managed CPU activation/staging arena plus reader scratch; excludes RSS/VRAM",
                "budget_bytes": self.budget, "user_reserve_bytes": self.reserve,
                "arena_extent_bytes": extent, "arena_allocation_bytes": extent + ALIGNMENT - 1,
                "reader_scratch_capacity_bytes": READ_CHUNK_BYTES,
                "managed_buffers_peak_bound_bytes": extent + ALIGNMENT - 1 + READ_CHUNK_BYTES,
                "weights_packed_bytes": self._weights_bytes,
                "full_dequantized_weight_buffer_bytes": 0,
                "persistent_kv_bytes": 0, "attention_score_scratch_bytes":
                    plan.allocations["__attention"].size_bytes,
                "peak_vram_bytes": None, "peak_rss_bytes": None,
                "excluded": ["Python objects, graph/manifest metadata and token history",
                             "tokenizer assets and metadata validation", "interpreter/native libraries",
                             "OS file cache", "caller-owned Python logits and reference verification"],
            },
            "validation": {"model_quality_measured": False, "verified": False},
        }

    def _execution_steps(self, graph, execution_context):
        for op in graph.ops:
            yield op, None

    def _state_action(self, action, buffer, execution_context, call):
        raise ValueError("The baseline executor has no state actions")

    def _rope_offset(self, action, execution_context):
        return 0

    def _run_attention(self, op, action, buffer, attention, out, call, length, execution_context):
        q, k, v = (buffer(name) for name in op.inputs)
        attributes = op.attributes
        call("nexa_causal_gqa_attention", q, len(q), k, len(k), v, len(v), length,
             attributes["num_heads"], attributes["num_key_value_heads"],
             attributes["head_dim"], attention, len(attention), out, len(out))

    def _execute(self, tokens, *, execution_context=None):
        length = len(tokens)
        graph, plan = self._make_plan(length, execution_context=execution_context)
        kernels = _load_kernels()
        extent = plan.peak_bytes["host"]
        arena = (ctypes.c_uint8 * (extent + ALIGNMENT - 1))()
        base = (ctypes.addressof(arena) + ALIGNMENT - 1) & -ALIGNMENT
        try:
            return self._execute_arena(tokens, graph, plan, kernels, base, execution_context)
        finally:
            # A saved exception keeps frame locals alive. Own the arena only
            # here and drop it even when a caller retries inside except; views
            # in the numerical frame borrow an address, never the allocation.
            arena = None

    def _execute_arena(self, tokens, graph, plan, kernels, base, execution_context):
        length = len(tokens)
        tensors = {t.name: t for t in graph.tensors}

        def buffer(name, ctype=ctypes.c_float):
            allocation = plan.allocations[name]
            return (ctype * (allocation.size_bytes // ctypes.sizeof(ctype))).from_address(base + allocation.offset)

        inputs = buffer(graph.inputs[0], ctypes.c_uint32)
        for index, token in enumerate(tokens):
            inputs[index] = token
        packed = buffer("__packed_tile", ctypes.c_uint8)
        packed_view = memoryview(packed).cast("B")
        tile_output = buffer("__output_tile")
        norm = buffer("__norm")
        attention = buffer("__attention")
        io = {"q4_payload_bytes_read": 0, "raw_payload_bytes_read": 0,
              "packed_bytes_consumed": 0, "matmul_tiles": 0, "host_to_gpu_bytes": 0,
              "embedding_rows_read": 0}
        read_seconds = compute_seconds = 0.0
        started = time.perf_counter()

        def call(name, *args):
            nonlocal compute_seconds
            tick = time.perf_counter()
            status = getattr(kernels, name)(*args)
            compute_seconds += time.perf_counter() - tick
            if status:
                raise ArithmeticError(f"Native {name} failed with status {status}")

        for op, action in self._execution_steps(graph, execution_context):
            if op is None:
                self._state_action(action, buffer, execution_context, call)
                continue
            kind, attributes = op.kind.value, op.attributes
            out = buffer(op.outputs[0])
            if kind in ("Embedding", "MatMul") and op.inputs[1] in self._dense_matrices:
                weight_name = op.inputs[1]
                rows, cols = self._storage[weight_name].shape
                blocks = self._bundle.matrix_blocks(weight_name)
                left = buffer(op.inputs[0]) if kind == "MatMul" else None
                for index, block in enumerate(blocks):
                    count, size = block["row_count"], block["bytes"]
                    if kind == "Embedding" and not any(
                            block["start_row"] <= token < block["start_row"] + count for token in tokens):
                        continue
                    tick = time.perf_counter()
                    io["raw_payload_bytes_read"] += self._bundle.read_matrix_block_into(
                        weight_name, index, packed_view[:size])
                    read_seconds += time.perf_counter() - tick
                    io["packed_bytes_consumed"] += size
                    weights = (ctypes.c_float * (count * cols)).from_address(ctypes.addressof(packed))
                    if kind == "Embedding":
                        for position, token in enumerate(tokens):
                            if not block["start_row"] <= token < block["start_row"] + count:
                                continue
                            ctypes.memmove(ctypes.addressof(out) + position * cols * 4,
                                           ctypes.addressof(packed) + (token - block["start_row"]) * cols * 4,
                                           cols * 4)
                            io["embedding_rows_read"] += 1
                        continue
                    call("nexa_f32_matmul", left, len(left), length, weights, len(weights),
                         count, cols, tile_output, length * count)
                    for row in range(length):
                        ctypes.memmove(ctypes.addressof(out) + (row * rows + block["start_row"]) * 4,
                                       ctypes.addressof(tile_output) + row * count * 4, count * 4)
                    io["matmul_tiles"] += 1
            elif kind in ("Embedding", "MatMul"):
                weight_name = op.inputs[1]
                with self._bundle.open_packed(weight_name) as reader:
                    if kind == "Embedding":
                        for index, token in enumerate(tokens):
                            tick = time.perf_counter()
                            reader.read_rows_into(token, 1, packed_view[:reader.row_bytes])
                            read_seconds += time.perf_counter() - tick
                            target = (ctypes.c_float * reader.cols).from_address(
                                ctypes.addressof(out) + index * reader.cols * 4)
                            # Embedding rows decode through the codec's own helper.
                            call(self._packed_kernels[weight_name][1], packed, reader.row_bytes,
                                 reader.cols, reader.group_size, target, len(target))
                            io["packed_bytes_consumed"] += reader.row_bytes
                            io["embedding_rows_read"] += 1
                    else:
                        left = buffer(op.inputs[0])
                        rows, _ = self._matrix_layouts[weight_name]
                        for start in range(0, reader.rows, rows):
                            count = min(rows, reader.rows - start)
                            size = count * reader.row_bytes
                            tick = time.perf_counter()
                            reader.read_rows_into(start, count, packed_view[:size])
                            read_seconds += time.perf_counter() - tick
                            call(self._packed_kernels[weight_name][0], left, len(left), length, packed, size,
                                 count, reader.cols, reader.group_size, tile_output, length * count)
                            for row in range(length):
                                ctypes.memmove(ctypes.addressof(out) + (row * reader.rows + start) * 4,
                                               ctypes.addressof(tile_output) + row * count * 4, count * 4)
                            io["packed_bytes_consumed"] += size
                            io["matmul_tiles"] += 1
                    io["q4_payload_bytes_read"] += reader.payload_bytes_read
            elif kind == "RMSNorm":
                x = buffer(op.inputs[0])
                tick = time.perf_counter()
                io["raw_payload_bytes_read"] += self._bundle.read_f32_into(
                    op.inputs[1], memoryview(norm).cast("B"))
                read_seconds += time.perf_counter() - tick
                call("nexa_rmsnorm", x, len(x), norm, len(norm), length,
                     self.config.hidden_size, attributes["epsilon"], out, len(out))
            elif kind == "RoPE":
                x = buffer(op.inputs[0])
                call("nexa_rope_offset", x, len(x), length, attributes["num_heads"],
                     attributes["head_dim"], attributes["theta"],
                     self._rope_offset(action, execution_context), out, len(out))
            elif kind == "CausalAttention":
                self._run_attention(op, action, buffer, attention, out, call, length, execution_context)
            elif kind in ("Add", "SwiGLU"):
                left, right = (buffer(name) for name in op.inputs)
                call("nexa_add" if kind == "Add" else "nexa_swiglu", left, len(left),
                     right, len(right), tensors[op.outputs[0]].numel, out, len(out))
            else:
                raise ValueError(f"Unsupported executable model operator: {kind}")

        logits = buffer(graph.outputs[0])
        digest = hashlib.sha256(memoryview(logits).cast("B")).hexdigest()
        output = [logits[row * self.config.vocab_size:(row + 1) * self.config.vocab_size]
                  for row in range(length)]
        report = self._report(graph, plan, executed=True, execution_context=execution_context)
        report.update({"token_ids": list(tokens), "sequence_length": length,
                       "processed_tokens": length,
                       "logits_shape": [length, self.config.vocab_size],
                       "logits_scope": "processed_chunk" if execution_context is not None else "full_prefix",
                       "operators_executed": len(graph.ops), "logits_sha256": digest,
                       "io": io, "timing": {"read_seconds": read_seconds,
                       "compute_seconds": compute_seconds,
                       "execution_wall_seconds": time.perf_counter() - started}})
        return output, report

    def prefill(self, token_ids):
        self._check_open()
        tokens = self._validate_tokens(token_ids)
        output, report = self._execute(tokens)
        self._tokens, self._last_report = tokens, report
        return output

    def decode(self, token_id):
        self._check_open()
        if not self._tokens:
            raise ValueError("decode requires a successful prefill first")
        tokens = self._validate_tokens([*self._tokens, token_id])
        output, report = self._execute(tokens)
        self._tokens, self._last_report = tokens, report
        return output[-1]

    def reset(self):
        self._check_open()
        graph, plan = self._make_plan(self.max_sequence_length)
        report = self._report(graph, plan, executed=False)
        self._tokens, self._last_report = (), report

    def report(self):
        self._check_open()
        return json.loads(json.dumps(self._last_report, allow_nan=False))

    def close(self):
        if not self._closed:
            self._bundle.close()
            self._tokens = ()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

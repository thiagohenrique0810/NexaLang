"""Transactional incremental F32 KV execution over streamed Q4 weights.

Two planned cache banks permit prompt replacement without copying the old cache.
Decode only writes an uncommitted suffix of the active bank; its prefix remains
usable if any later layer or output projection fails. The global length changes
only after logits and the complete report have been produced.
"""
from __future__ import annotations

import ctypes

from compiler.kv_plan import make_kv_cache_plan, plan_step
from compiler.model_lowering import lower_model
from .transformer import ALIGNMENT, TransformerSession, _positive


class IncrementalTransformerSession(TransformerSession):
    """One synchronous sequence with bounded, persistent, double-bank F32 KV.

    prefill replaces the prompt atomically. append accepts a nonempty chunk;
    decode is append of one ID. Output lists remain caller-owned, outside the
    managed-buffer budget. reset invalidates cached positions without copying
    or clearing the full allocation. Sessions are not concurrently callable.
    """
    def __init__(self, bundle_path, *, max_chunk_length=None, **kwargs):
        self._cache_plan = None
        self._cache_arena = None
        self._cache_address = 0
        self._active_bank = 0
        self._requested_chunk_length = max_chunk_length
        self.max_chunk_length = None
        super().__init__(bundle_path, **kwargs)
        try:
            # super().__init__ already checked the combined cache/workspace
            # budget. No payload or native library has been loaded yet.
            self._allocate_cache()
            graph, workspace = self._make_plan(self.max_sequence_length)
            self._last_report = self._report(graph, workspace, executed=False)
        except BaseException:
            self.close()
            raise

    @property
    def cache_length(self):
        return len(self._tokens)

    @property
    def active_bank(self):
        return self._active_bank

    def _allocate_cache(self):
        self._cache_arena = (ctypes.c_uint8 * self._cache_plan.allocation_bytes)()
        raw_address = ctypes.addressof(self._cache_arena)
        self._cache_address = (raw_address + ALIGNMENT - 1) & -ALIGNMENT

    def _configure_cache(self):
        if self._cache_plan is None:
            self._cache_plan = make_kv_cache_plan(self.config, self.max_sequence_length)
            self.max_chunk_length = _positive(
                self.max_sequence_length if self._requested_chunk_length is None else self._requested_chunk_length,
                "max_chunk_length")
            if self.max_chunk_length > self.max_sequence_length:
                raise ValueError("Chunk capacity exceeds the session context capacity")

    def _context(self, length, *, mode="prefill"):
        self._configure_cache()
        graph = lower_model(self.config, length, weight_storage=self._storage)
        return plan_step(graph, self._cache_plan, self.cache_length,
                         mode=mode, active_bank=self._active_bank)

    def _make_plan(self, length, *, execution_context=None):
        self._configure_cache()
        if execution_context is None:
            # Initial capacity planning combines the largest permitted chunk
            # with the full context's attention scratch and persistent cache.
            length = min(length, self.max_chunk_length)
        context = execution_context or self._context(length)
        graph = context.graph
        if graph.tensors[[t.name for t in graph.tensors].index("tokens")].shape != (length,):
            raise ValueError("Execution chunk differs from its KV step plan")
        workspace = self._plan_workspace(
            length, context.activation_requests, len(context.steps) + 2,
            attention_length=self.max_sequence_length if execution_context is None else context.new_length,
            extra_reserve=self._cache_plan.allocation_bytes)
        return graph, workspace

    def _execution_steps(self, graph, execution_context):
        if execution_context is None:
            raise ValueError("Incremental execution requires an explicit KV step plan")
        operations = {op.name: op for op in graph.ops}
        for action in execution_context.steps:
            if action.kind in ("CacheWrite", "Commit"):
                yield None, action
            elif action.kind in ("Compute", "RoPE", "CachedAttention"):
                yield operations[action.op_name], action
            else:
                raise ValueError(f"Unknown KV execution action: {action.kind}")

    def _state_action(self, action, buffer, execution_context, call):
        if action.kind == "Commit":
            # The numerical schedule is complete, but Python output/report
            # creation can still fail. The public entrypoint commits afterwards.
            return
        if action.kind != "CacheWrite":
            raise ValueError(f"Unsupported state action: {action.kind}")
        bindings = action.bindings
        size = bindings["write_bytes"]
        for kind in ("key", "value"):
            source = buffer(bindings[f"{kind}_input"])
            offset = bindings[f"{kind}_write_offset"]
            if size != ctypes.sizeof(source) or offset < 0 or offset + size > self._cache_plan.arena_bytes:
                raise ValueError("KV write does not fit its planned source and cache range")
            ctypes.memmove(self._cache_address + offset, ctypes.addressof(source), size)

    def _rope_offset(self, action, execution_context):
        if action.kind != "RoPE":
            raise ValueError("RoPE operator lacks an explicit position offset")
        return action.bindings["position_offset"]

    def _run_attention(self, op, action, buffer, attention, out, call, length, execution_context):
        if action.kind != "CachedAttention":
            raise ValueError("Attention operator lacks a KV state binding")
        bindings, attributes = action.bindings, op.attributes
        q = buffer(op.inputs[0])
        views = []
        for kind in ("key", "value"):
            allocation = self._cache_plan.buffer(bindings["bank"], bindings["layer"], kind)
            if allocation.offset != bindings[f"{kind}_offset"]:
                raise ValueError("Attention cache offset differs from its allocation")
            views.append((ctypes.c_float * (allocation.size_bytes // 4)).from_address(
                self._cache_address + allocation.offset))
        key, value = views
        call("nexa_causal_gqa_attention_cached", q, len(q), key, len(key), value, len(value),
             bindings["past_length"], length, attributes["num_heads"],
             attributes["num_key_value_heads"], attributes["head_dim"],
             attention, len(attention), out, len(out))

    def _report(self, graph, plan, *, executed, execution_context=None):
        report = super()._report(graph, plan, executed=executed)
        context = execution_context
        total = context.new_length if executed and context is not None else self.cache_length
        report.update({"decode_strategy": "incremental_kv", "persistent_kv_cache": True,
                       "max_chunk_length": self.max_chunk_length,
                       "kv_cache_plan": self._cache_plan.to_dict(),
                       "model_ir_scope": "chunk computation template; state dependencies and positions in execution_plan",
                       "context_length": total, "cache_length": total,
                       "cache_position_offset": context.position_offset if context is not None else 0,
                       "active_bank": context.target_bank if executed and context is not None else self._active_bank})
        if context is not None:
            report["execution_plan"] = context.to_dict()
        memory = report["memory"]
        memory["scope"] = "managed CPU workspace, both persistent KV banks and reader scratch; excludes RSS/VRAM"
        memory.update({"persistent_kv_bytes": self._cache_plan.cache_bytes,
                       "kv_bank_count": 2, "kv_bytes_per_bank": self._cache_plan.bytes_per_bank,
                       "kv_arena_extent_bytes": self._cache_plan.arena_bytes,
                       "kv_arena_allocation_bytes": self._cache_plan.allocation_bytes,
                       "kv_valid_prefix_bytes": total * self.config.num_hidden_layers * 2 *
                           self.config.num_key_value_heads * self.config.head_dim * 4,
                       "kv_prefix_copy_buffer_bytes": 0})
        memory["managed_buffers_peak_bound_bytes"] += self._cache_plan.allocation_bytes
        return report

    def _perform(self, token_ids, *, mode):
        self._check_open()
        chunk = self._validate_tokens(token_ids)
        if len(chunk) > self.max_chunk_length:
            raise ValueError("Input chunk exceeds max_chunk_length; split the prompt into smaller chunks")
        if mode == "decode" and not self._tokens:
            raise ValueError("append/decode requires a successful prefill first")
        candidate = chunk if mode == "prefill" else self._validate_tokens(self._tokens + chunk)
        context = self._context(len(chunk), mode=mode)
        output, report = self._execute(chunk, execution_context=context)
        report.update({"token_ids": list(candidate), "sequence_length": len(candidate),
                       "processed_tokens": len(chunk), "input_chunk_token_ids": list(chunk)})
        report["io"].update({"kv_bytes_written": len(chunk) * self.config.num_hidden_layers * 2 *
                            self.config.num_key_value_heads * self.config.head_dim * 4,
                            "kv_prefix_bytes_copied": 0})
        # Nothing above this point changes the visible prefix or its active bank.
        self._tokens, self._last_report, self._active_bank = candidate, report, context.target_bank
        return output

    def prefill(self, token_ids):
        # The guard wraps the whole transaction, not just the numerical phase:
        # migration and page publication are cancellable for the same reason.
        with self._call_guard():
            return self._perform(token_ids, mode="prefill")

    def append(self, token_ids):
        with self._call_guard():
            return self._perform(token_ids, mode="decode")

    def decode(self, token_id):
        return self.append([token_id])[-1]

    def reset(self):
        self._check_open()
        graph, plan = self._make_plan(self.max_sequence_length)
        report = self._report(graph, plan, executed=False)
        report.update({"context_length": 0, "cache_length": 0,
                       "cache_position_offset": 0, "active_bank": 0})
        report["memory"]["kv_valid_prefix_bytes"] = 0
        self._tokens, self._active_bank, self._last_report = (), 0, report

    def close(self):
        super().close()
        self._cache_arena = None
        self._cache_address = 0

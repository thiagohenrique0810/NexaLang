"""On-demand F32/Q4/Q3/TQ KV pages with native attention and atomic commits.

A page owns K/V for every layer, with aligned token-major buffers. Append keeps
all committed page addresses stable; replacement stages fresh pages until commit.
Admission reserves the maximum context plus a complete replacement chunk, while
reports distinguish that bound from the bytes actually resident during a call.
"""
from __future__ import annotations

import ctypes

from compiler.model_lowering import lower_model
from compiler.paged_kv_plan import make_paged_kv_cache_plan, plan_paged_step
from compiler.planner.memory import MemoryRequest
from .format import READ_CHUNK_BYTES
from .incremental import IncrementalTransformerSession
from .transformer import ALIGNMENT, TransformerSession, _positive
from .tq_kv import TQKVContext, TQKernelDispatch, tq_kv_memory

# Codec ids the native KV accessors dispatch on; f32 is zero.
_KV_CODEC_IDS = {"f32": 0, "q3": 3, "q4": 4, "q8": 8}


class _KVPage:
    """One page allocation, owned by one sequence or shared by several.

    Only complete pages are ever shared: append writes into the newest partial
    page or into fresh ones, so a shared page is immutable for every owner.
    """
    def __init__(self, allocation_bytes):
        self.arena = (ctypes.c_uint8 * allocation_bytes)()
        self._references = 1
        try:
            self.address = (ctypes.addressof(self.arena) + ALIGNMENT - 1) & -ALIGNMENT
        except BaseException:
            self.release()
            raise

    @property
    def shared(self):
        return self._references > 1

    def retain(self):
        if not self.address:
            raise ValueError("A released KV page cannot be shared")
        self._references += 1
        return self

    def release(self):
        # Saved tracebacks can retain the page object. Explicitly detach its
        # owner so a failed transaction never keeps physical pages resident.
        # A shared page only loses its allocation with its last owner.
        if self._references > 1:
            self._references -= 1
            return
        self._references = 0
        self.address = 0
        self.arena = None


class PagedTransformerSession(IncrementalTransformerSession):
    """Single-sequence paged KV; inherits the incremental operator schedule.

    page_tokens controls physical page size. max_chunk_length bounds activations
    and replacement staging. Q4/Q3/TQ quantize each new token/head independently;
    partial pages never require requantizing a committed prefix. reset/close free
    pages. TQ retains one immutable MSE context until close, with planned head
    reconstruction scratch. Eviction, shared prefixes, and concurrent calls
    are not supported.
    """
    def __init__(self, bundle_path, *, page_tokens=16, max_chunk_length=None,
                 kv_codec="f32", kv_group_size=None, kv_bits=None, kv_seed=None,
                 kv_codebook_f32le=None, **kwargs):
        self.page_tokens = _positive(page_tokens, "page_tokens")
        if kv_codec not in ("f32", "q4", "q3", "q8", "tq"):
            raise ValueError("kv_codec must be f32, q4, q3, q8 or tq")
        if kv_codec not in ("q4", "q3", "q8") and kv_group_size is not None:
            raise ValueError("kv_group_size requires the q4, q3 or q8 KV codec")
        if kv_codec != "tq" and any(value is not None for value in (kv_bits, kv_seed, kv_codebook_f32le)):
            raise ValueError("kv_bits/kv_seed/kv_codebook_f32le require the tq KV codec")
        self.kv_codec = kv_codec
        self.kv_group_size = kv_group_size
        self._kv_bits = kv_bits
        self._kv_seed = kv_seed
        self._kv_codebook_f32le = kv_codebook_f32le
        self._tq_context = None
        self._requested_chunk_length = max_chunk_length
        self.max_chunk_length = None
        self._cache_plan = None
        self._pages = []
        self._pending_pages = None
        # Deliberately use the shared Transformer initializer, without the
        # contiguous incremental session's two-bank allocation.
        TransformerSession.__init__(self, bundle_path, **kwargs)

    @property
    def kv_bits(self):
        return self._cache_plan.bits

    @property
    def kv_seed(self):
        return self._cache_plan.seed

    @property
    def kv_codebook_f32le(self):
        return self._cache_plan.codebook_f32le

    @property
    def active_bank(self):
        raise AttributeError("Paged KV uses logical page tables, not banks")

    @property
    def resident_page_count(self):
        return len(self._pages)

    @property
    def resident_kv_bytes(self):
        return self.resident_page_count * self._cache_plan.page_allocation_bytes

    def _allocate_page(self):
        return _KVPage(self._cache_plan.page_allocation_bytes)

    def _configure_cache(self):
        if self._cache_plan is None:
            self._cache_plan = make_paged_kv_cache_plan(
                self.config, self.max_sequence_length, self.page_tokens,
                codec=self.kv_codec, group_size=self.kv_group_size, bits=self._kv_bits,
                seed=self._kv_seed, codebook_f32le=self._kv_codebook_f32le)
            self.kv_group_size = self._cache_plan.group_size
            self.max_chunk_length = _positive(
                self.max_sequence_length if self._requested_chunk_length is None else self._requested_chunk_length,
                "max_chunk_length")
            if self.max_chunk_length > self.max_sequence_length:
                raise ValueError("Chunk capacity exceeds the session context capacity")
            # Validate byte/metadata limits for the worst append alignment as
            # well as a position-zero prefill before expanding any step plan.
            self._cache_plan.allocation_limit_bytes(self.max_chunk_length)

    def _fork_options(self):
        """Everything a derived sequence needs to reproduce this exact layout."""
        packed = self.kv_codec in ("q4", "q3")
        return {"memory_budget": self.budget, "reserve_bytes": self.reserve,
                "max_sequence_length": self.max_sequence_length, "tile_rows": self.tile_rows,
                "page_tokens": self.page_tokens, "max_chunk_length": self._requested_chunk_length,
                "kv_codec": self.kv_codec, "kv_group_size": self.kv_group_size if packed else None,
                "kv_bits": self._kv_bits if self.kv_codec == "tq" else None,
                "kv_seed": self._kv_seed if self.kv_codec == "tq" else None,
                "kv_codebook_f32le": self.kv_codebook_f32le if self.kv_codec == "tq" else None}

    def fork(self, **overrides):
        """Derive a sequence continuing this committed prefix.

        Complete pages are shared, not copied; the partial page is copied so
        both sequences keep exclusive write access to their own tail. The new
        session admits its own budget before any page is retained or copied,
        and the two sequences are independent afterwards: either can append,
        reset or close in any order.
        """
        self._check_open()
        if not self._tokens:
            raise ValueError("fork requires a committed prefix")
        child = type(self)(self.path, **{**self._fork_options(), **overrides})
        try:
            child._adopt_prefix(self)
        except BaseException:
            child.close()
            raise
        return child

    def _adopt_prefix(self, parent):
        if not isinstance(parent, PagedTransformerSession) or type(self) is not type(parent):
            raise ValueError("A prefix can only be adopted from the same paged executor")
        if self._pages or self._tokens:
            raise ValueError("Only a session without its own prefix can adopt one")
        parent._check_open()
        self._check_open()
        self._configure_cache()
        if self.kv_codec == "tq":
            self._ensure_tq_context()
        if (self._manifest_sha256 != parent._manifest_sha256
                or self._layout_identity() != parent._layout_identity()):
            raise ValueError("A derived sequence requires the same bundle and KV page layout")
        if len(parent._tokens) > self.max_sequence_length:
            raise ValueError("The inherited prefix exceeds this sequence's context capacity")
        # Pages before the newest partial one hold only committed tokens.
        complete = parent.cache_length // self.page_tokens
        pages, copied = [], 0
        try:
            for index, page in enumerate(parent._pages):
                if index < complete:
                    pages.append(page.retain())
                    continue
                copy = self._allocate_page()
                try:
                    pages.append(copy)
                except BaseException:
                    copy.release()
                    raise
                ctypes.memmove(copy.address, page.address, self._cache_plan.page_extent_bytes)
                copied += 1
            graph, plan = self._make_plan(self.max_sequence_length)
            self._tokens, self._pages = parent._tokens, pages
            self._adopt_state(parent)
            report = self._report(graph, plan, executed=False)
        except BaseException:
            self._tokens, self._pages = (), []
            self._adopt_state(None)
            for page in pages:
                page.release()
            raise
        # A report for a sequence that has not executed yet has no io block;
        # adoption costs are stated separately from executed KV traffic.
        report["kv_prefix_adoption"] = {"inherited_tokens": len(self._tokens),
                                        "shared_pages": len(pages) - copied, "copied_pages": copied,
                                        "copied_bytes": copied * self._cache_plan.page_extent_bytes}
        self._last_report = report

    def _layout_identity(self):
        """Canonical description of every layout an inherited page may use."""
        return self._cache_plan.to_json(indent=None)

    def _adopt_state(self, parent):
        """Executor state an inherited prefix implies, before its first report."""

    def _tq_memory(self):
        return tq_kv_memory(self.config.head_dim, self.kv_bits)

    def _ensure_tq_context(self):
        if self.kv_codec != "tq" or self._tq_context is not None:
            return
        context = TQKVContext(self.config.head_dim, self.kv_bits, self.kv_seed, self.kv_codebook_f32le)
        try:
            concrete = make_paged_kv_cache_plan(
                self.config, self.max_sequence_length, self.page_tokens, codec="tq",
                bits=self.kv_bits, seed=self.kv_seed, codebook_f32le=context.codebook_f32le)
        except BaseException:
            context.close()
            raise
        self._cache_plan, self._tq_context = concrete, context

    def _context(self, length, *, mode="prefill"):
        self._configure_cache()
        graph = lower_model(self.config, length, weight_storage=self._storage)
        return plan_paged_step(graph, self._cache_plan, self.cache_length, mode=mode)

    def _make_plan(self, length, *, execution_context=None):
        self._configure_cache()
        if execution_context is None:
            length = min(length, self.max_chunk_length)
        context = execution_context or self._context(length)
        graph = context.graph
        if next(t.shape for t in graph.tensors if t.name == "tokens") != (length,):
            raise ValueError("Execution chunk differs from its paged KV step plan")
        end = len(context.steps) + 2
        prefix = self.max_sequence_length if execution_context is None else context.new_length
        table_bytes = self._cache_plan.page_count(prefix) * ctypes.sizeof(ctypes.c_void_p)
        requests = [*context.activation_requests,
                    MemoryRequest("__key_pages", table_bytes, 0, end),
                    MemoryRequest("__value_pages", table_bytes, 0, end)]
        extra_reserve = self._cache_plan.allocation_limit_bytes(self.max_chunk_length)
        if self.kv_codec == "tq":
            memory = self._tq_memory()
            requests.extend((MemoryRequest("__tq_vector", memory["vector_scratch_bytes"], 0, end),
                             MemoryRequest("__tq_accumulator", memory["accumulator_bytes"], 0, end)))
            extra_reserve += memory["context_reserved_bytes"] + memory["constructor_staging_bytes"]
        workspace = self._plan_workspace(
            length, requests, end, attention_length=prefix,
            extra_reserve=extra_reserve)
        return graph, workspace

    def _execute_arena(self, tokens, graph, plan, kernels, base, execution_context):
        if self.kv_codec == "tq":
            if self._tq_context is None or self.kv_codebook_f32le is None:
                raise ValueError("TQ execution requires a concrete admitted codebook/context")
            kernels = TQKernelDispatch(kernels, self._tq_context.library)
        return super()._execute_arena(tokens, graph, plan, kernels, base, execution_context)

    def _state_action(self, action, buffer, execution_context, call):
        if action.kind == "Commit":
            return  # Commit follows Python result/report creation in _perform.
        if action.kind != "CacheWrite" or self._pending_pages is None:
            raise ValueError("Paged KV write requires a prepared transaction")
        bindings = action.bindings
        source_row_bytes = bindings["kv_width"] * 4
        target_row_bytes = self._cache_plan.token_bytes
        for kind in ("key", "value"):
            source = buffer(bindings[f"{kind}_input"])
            if ctypes.sizeof(source) != bindings["query_length"] * source_row_bytes:
                raise ValueError("KV source differs from the planned chunk")
            for segment in bindings["segments"]:
                page = self._pending_pages[segment["page_index"]]
                offset = bindings[f"{kind}_offset"] + segment["page_token_offset"] * target_row_bytes
                source_offset = segment["source_token_offset"] * source_row_bytes
                size = segment["token_count"] * target_row_bytes
                source_size = segment["token_count"] * source_row_bytes
                if (not page.address or offset + size > self._cache_plan.page_extent_bytes
                        or source_offset + source_size > ctypes.sizeof(source)):
                    raise ValueError("Paged KV write exceeds the planned source or page")
                if page.shared:
                    # Defense in depth: adoption copies the partial page, so a
                    # write must never reach a page another sequence reads.
                    raise ValueError("A shared prefix page is immutable; new tokens need an owned page")
                if self.kv_codec != "f32":
                    inputs = (ctypes.c_float * (source_size // 4)).from_address(ctypes.addressof(source) + source_offset)
                    packed = (ctypes.c_uint8 * size).from_address(page.address + offset)
                    rows = segment["token_count"] * self.config.num_key_value_heads
                    if self.kv_codec == "tq":
                        scratch = buffer("__tq_vector")
                        # The public TQ ABI counts vectors with signed int.
                        # Chunking here bounds that count without new staging.
                        max_rows = (1 << 31) - 1
                        stride = self._cache_plan.head_row_bytes
                        for start in range(0, rows, max_rows):
                            count = min(rows - start, max_rows)
                            chunk_input = (ctypes.c_float * (count * self.config.head_dim)).from_address(
                                ctypes.addressof(inputs) + start * self.config.head_dim * 4)
                            chunk_packed = (ctypes.c_uint8 * (count * stride)).from_address(
                                ctypes.addressof(packed) + start * stride)
                            call("tq_quantize_tq02", self._tq_context._ctx, chunk_input, len(chunk_input),
                                 chunk_packed, len(chunk_packed), count, scratch, len(scratch))
                    else:
                        call(f"nexa_{self.kv_codec}_quantize", inputs, len(inputs), rows,
                             self.config.head_dim, self.kv_group_size, packed, size)
                else:
                    ctypes.memmove(page.address + offset, ctypes.addressof(source) + source_offset, size)

    def _run_attention(self, op, action, buffer, attention, out, call, length, execution_context):
        if action.kind != "CachedAttention" or self._pending_pages is None:
            raise ValueError("Paged attention requires a prepared transaction")
        bindings, attributes = action.bindings, op.attributes
        query = buffer(op.inputs[0])
        key_table = buffer("__key_pages", ctypes.c_void_p)
        value_table = buffer("__value_pages", ctypes.c_void_p)
        count = bindings["page_count"]
        if len(key_table) != count or len(value_table) != count or len(self._pending_pages) != count:
            raise ValueError("Paged attention tables differ from the planned visible prefix")
        for index, page in enumerate(self._pending_pages):
            if not page.address:
                raise ValueError("Paged attention cannot access a released page")
            key_table[index] = page.address + bindings["key_offset"]
            value_table[index] = page.address + bindings["value_offset"]
        pointer_table = ctypes.POINTER(ctypes.POINTER(ctypes.c_float if self.kv_codec == "f32" else ctypes.c_uint8))
        common = (query, len(query), ctypes.cast(key_table, pointer_table), count,
                  ctypes.cast(value_table, pointer_table), count, self.page_tokens)
        if self.kv_codec == "tq":
            vector, accumulator = buffer("__tq_vector"), buffer("__tq_accumulator", ctypes.c_double)
            call("nexa_causal_gqa_attention_paged_tq", self._tq_context._ctx, *common,
                 self.page_tokens * self._cache_plan.token_bytes, bindings["past_length"], length,
                 attributes["num_heads"], attributes["num_key_value_heads"], attributes["head_dim"],
                 attention, len(attention), vector, len(vector), accumulator, len(accumulator), out, len(out))
            return
        if self.kv_codec == "q8":
            # One homogeneous codec through the shared per-layout accessors,
            # instead of a kernel duplicated for every packed KV codec.
            call("nexa_causal_gqa_attention_paged_codec", query, len(query),
                 ctypes.cast(key_table, pointer_table), count,
                 ctypes.cast(value_table, pointer_table), count,
                 _KV_CODEC_IDS[self.kv_codec], self.page_tokens,
                 self.page_tokens * self._cache_plan.token_bytes, self.kv_group_size,
                 bindings["past_length"], length, attributes["num_heads"],
                 attributes["num_key_value_heads"], attributes["head_dim"],
                 attention, len(attention), out, len(out))
            return
        capacity = ((self.page_tokens * self._cache_plan.token_bytes, self.kv_group_size)
                    if self.kv_codec != "f32" else (self.page_tokens * self._cache_plan.kv_width,))
        call("nexa_causal_gqa_attention_paged" if self.kv_codec == "f32" else f"nexa_causal_gqa_attention_paged_{self.kv_codec}",
             *common, *capacity, bindings["past_length"], length, attributes["num_heads"],
             attributes["num_key_value_heads"], attributes["head_dim"], attention, len(attention), out, len(out))

    def _report(self, graph, plan, *, executed, execution_context=None):
        report = TransformerSession._report(self, graph, plan, executed=executed)
        context = execution_context
        total = context.new_length if executed and context is not None else self.cache_length
        pages = self._cache_plan.page_count(total)
        peak_pages = context.resident_pages_peak if executed and context is not None else len(self._pages)
        allocation = self._cache_plan.page_allocation_bytes
        reserved = self._cache_plan.allocation_limit_bytes(self.max_chunk_length)
        report.update({"decode_strategy": "paged_incremental_kv", "persistent_kv_cache": True,
                       "paged_kv_cache": True, "max_chunk_length": self.max_chunk_length,
                       "kv_codec": self.kv_codec, "kv_group_size": self.kv_group_size,
                       "kv_partial_page_policy": "f32_on_write" if self.kv_codec == "f32" else "quantize_each_token_on_write",
                       "kv_cache_plan": self._cache_plan.to_dict(),
                       "model_ir_scope": "chunk template; paged state dependencies and positions in execution_plan",
                       "context_length": total, "cache_length": total,
                       "cache_position_offset": context.position_offset if context is not None else 0})
        if context is not None:
            report["execution_plan"] = context.to_dict()
        memory = report["memory"]
        # Shared pages are resident once for the whole process; each sequence
        # still admits its own conservative reservation, which does not shrink.
        resident = self._pending_pages if self._pending_pages is not None else self._pages
        shared = sum(getattr(page, "shared", False) for page in resident if page is not None)
        memory.update({"scope": "managed CPU workspace, resident/staged KV pages, pointer tables and reader scratch; excludes RSS/VRAM",
                       "kv_page_tokens": self.page_tokens, "kv_page_payload_bytes": self._cache_plan.page_payload_bytes,
                       "kv_page_allocation_bytes": allocation, "kv_resident_page_count": pages,
                       "kv_resident_allocation_bytes": pages * allocation,
                       "kv_transaction_peak_allocation_bytes": peak_pages * allocation,
                       "kv_reserved_capacity_bytes": reserved,
                       "kv_reserved_page_count": self._cache_plan.reservation_pages(self.max_chunk_length),
                       "persistent_kv_bytes": pages * self._cache_plan.page_payload_bytes,
                       "kv_valid_prefix_bytes": total * self.config.num_hidden_layers * 2 * self._cache_plan.token_bytes,
                       "kv_valid_prefix_f32_bytes": total * self.config.num_hidden_layers * 2 * self._cache_plan.kv_width * 4,
                       "kv_encoded_bytes_per_token": self.config.num_hidden_layers * 2 * self._cache_plan.token_bytes,
                       "kv_f32_bytes_per_token": self.config.num_hidden_layers * 2 * self._cache_plan.kv_width * 4,
                       "kv_shared_page_count": shared,
                       "kv_shared_allocation_bytes": shared * allocation,
                       "kv_owned_allocation_bytes": (len(resident) - shared) * allocation,
                       "kv_full_dequantized_buffer_bytes": 0,
                       "kv_page_table_bytes": sum(plan.allocations[name].size_bytes
                                                  for name in ("__key_pages", "__value_pages")),
                       "kv_prefix_copy_buffer_bytes": 0,
                       "capacity_managed_buffers_bound_bytes": self._workspace_template.peak_bytes["host"] +
                           ALIGNMENT - 1 + READ_CHUNK_BYTES + reserved})
        memory["managed_buffers_peak_bound_bytes"] += peak_pages * allocation
        if self.kv_codec == "tq":
            tq = self._tq_memory()
            state = self._tq_context.state_bytes if self._tq_context is not None else 0
            constructor = tq["context_reserved_bytes"] + tq["constructor_staging_bytes"]
            report.update({"kv_bits": self.kv_bits, "kv_seed": self.kv_seed})
            memory.update({"scope": "managed CPU workspace, TQ context, KV pages/tables and reader scratch; excludes RSS/VRAM",
                           "kv_tq_context_bytes": state,
                           "kv_tq_context_reserved_bytes": tq["context_reserved_bytes"],
                           "kv_tq_constructor_staging_bytes": tq["constructor_staging_bytes"],
                           "kv_tq_constructor_peak_bound_bytes": constructor,
                           "kv_tq_vector_scratch_bytes": tq["vector_scratch_bytes"],
                           "kv_tq_accumulator_bytes": tq["accumulator_bytes"],
                           "kv_tq_scratch_policy": "one F32 head shared by quantize/decode; one F64 head accumulator"})
            memory["capacity_managed_buffers_bound_bytes"] += constructor
            memory["managed_buffers_peak_bound_bytes"] += state
            # Context construction precedes page/arena allocation. The capacity
            # reservation is deliberately conservative; it is not residence.
            if executed:
                memory["managed_buffers_peak_bound_bytes"] = max(
                    memory["managed_buffers_peak_bound_bytes"], constructor)
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
        self._make_plan(len(chunk), execution_context=context)  # Before allocating any new page.
        old_cache_plan, old_tq_context = self._cache_plan, self._tq_context
        old_pages = self._pages
        staged = [None] * context.new_pages
        pending = [None] * self._cache_plan.page_count(context.new_length)
        reused = len(old_pages) if mode == "decode" else 0
        for index in range(reused):
            pending[index] = old_pages[index]
        committed = False
        try:
            if self.kv_codec == "tq" and self._tq_context is None:
                self._ensure_tq_context()  # After admission; no page or payload exists yet.
                context = self._context(len(chunk), mode=mode)
            for index in range(len(staged)):
                staged[index] = self._allocate_page()
                pending[reused + index] = staged[index]
            self._pending_pages = pending
            output, report = self._execute(chunk, execution_context=context)
            report.update({"token_ids": list(candidate), "sequence_length": len(candidate),
                           "processed_tokens": len(chunk), "input_chunk_token_ids": list(chunk)})
            report["io"].update({"kv_bytes_written": len(chunk) * self.config.num_hidden_layers * 2 *
                                self._cache_plan.token_bytes, "kv_prefix_bytes_copied": 0,
                                "kv_source_f32_bytes_quantized": len(chunk) * self.config.num_hidden_layers * 2 *
                                    self._cache_plan.kv_width * 4 if self.kv_codec != "f32" else 0,
                                "kv_pages_allocated": len(staged),
                                "kv_pages_released": len(old_pages) if mode == "prefill" else 0})
            self._tokens, self._last_report, self._pages = candidate, report, pending
            committed = True
            return output
        finally:
            self._pending_pages = None
            if committed and mode == "prefill":
                for page in old_pages:
                    page.release()
            elif not committed:
                for page in staged:
                    if page is not None:
                        page.release()
                if old_tq_context is None and self._tq_context is not None:
                    self._tq_context.close()
                    self._tq_context = None
                    self._cache_plan = old_cache_plan

    def reset(self):
        self._check_open()
        graph, plan = self._make_plan(self.max_sequence_length)
        report = self._report(graph, plan, executed=False)
        report.update({"context_length": 0, "cache_length": 0, "cache_position_offset": 0})
        memory = report["memory"]
        for name in ("kv_resident_page_count", "kv_resident_allocation_bytes",
                     "kv_transaction_peak_allocation_bytes", "persistent_kv_bytes", "kv_valid_prefix_bytes",
                     "kv_valid_prefix_f32_bytes"):
            memory[name] = 0
        memory["managed_buffers_peak_bound_bytes"] = (plan.peak_bytes["host"] + ALIGNMENT - 1 + READ_CHUNK_BYTES)
        if self._tq_context is not None:
            memory["managed_buffers_peak_bound_bytes"] += self._tq_context.state_bytes
        empty_pages = []
        old_pages = self._pages
        self._tokens, self._last_report, self._pages = (), report, empty_pages
        for page in old_pages:
            page.release()

    def close(self):
        TransformerSession.close(self)
        for page in self._pages:
            page.release()
        self._pages.clear()
        if self._tq_context is not None:
            self._tq_context.close()
            self._tq_context = None

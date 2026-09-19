"""Transactional cold-page eviction with bounded two-pass causal CPU attention.

Cold Q3 bytes live in a private ephemeral store. A bounded reload cache serves
all layers/heads; the whole prefix is never resident or concatenated. Codec
aging and its numerical history are identical to the resident tiered executor.
Slots only change residency and I/O: the same published bytes reach the kernel,
whether a page was reloaded once or on every pass.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import time

from compiler.offloaded_kv_plan import OffloadedKVPlan, RELOAD_POLICY_ID
from compiler.planner.memory import MemoryRequest
from .format import READ_CHUNK_BYTES
from .kv_store import KVPageStore, KV_STORE_BUFFER_BYTES
from .reload_cache import ReloadPageCache
from .tiered import TieredTransformerSession, _CODEC_IDS
from .transformer import ALIGNMENT, TransformerSession


class _OffloadedPage:
    address = 0
    arena = None
    allocation_bytes = 0

    def __init__(self, ref, layout):
        self.ref, self.layout, self.codec = ref, layout, layout.codec

    def release(self):
        # File ownership belongs to the store, never to borrowed page handles.
        pass


class OffloadedTieredTransformerSession(TieredTransformerSession):
    """Age tiers with all cold pages evicted; synchronous, private single sequence.

    The directory is a disposable cache, not a resumable inference checkpoint.
    No file or reload slot is created before the global budget is admitted.
    kv_reload_slots admits how many cold pages may stay resident across the
    layers and calls of this session; slots are allocated only on a real miss.
    """
    def __init__(self, bundle_path, *, kv_backing_store, kv_reload_slots=1, **kwargs):
        self._backing_parent = Path(kv_backing_store)
        self._reload_slots = kv_reload_slots
        self._offload_plan = self._reload_cache = None
        self._store = None
        self._retired_refs = []
        self._backing_io = self._reload_baseline = None
        super().__init__(bundle_path, **kwargs)

    @property
    def resident_page_count(self):
        return sum(not isinstance(page, _OffloadedPage) for page in self._pages)

    def _configure_cache(self):
        super()._configure_cache()
        if self._offload_plan is None:
            self._offload_plan = OffloadedKVPlan(self._tier_plan, self.max_chunk_length, self._reload_slots)
            # Constructing the cache reserves no memory; slots come from the
            # admitted page allocator once attention actually misses.
            self._reload_cache = ReloadPageCache(self._offload_plan.reload_slots,
                                                 lambda: self._allocate_page("q3"))

    def _make_plan(self, length, *, execution_context=None):
        self._configure_cache()
        if execution_context is None:
            length = min(length, self.max_chunk_length)
        context = execution_context or self._context(length)
        graph, end = context.graph, len(context.steps) + 2
        if next(t.shape for t in graph.tensors if t.name == "tokens") != (length,):
            raise ValueError("Execution chunk differs from the offloaded KV schedule")
        heads = length * self.config.num_attention_heads
        requests = [*context.activation_requests,
                    MemoryRequest("__stream_sums", 8 * heads * (self.config.head_dim + 1), 0, end)]
        reserve = (self._offload_plan.allocation_limit_bytes + self._offload_plan.reload_cache_capacity_bytes +
                   self._tier_plan.migration_scratch_bytes + KV_STORE_BUFFER_BYTES)
        # __attention contains double maxima per query/head, not prefix scores.
        return graph, self._plan_workspace(length, requests, end, attention_length=2 * heads,
                                           extra_reserve=reserve)

    def _ensure_store(self):
        if self._store is None:
            self._store = KVPageStore(self._backing_parent, identity=self._manifest_sha256)

    def _loader(self, page):
        """Verified read of this page into a slot the cache owns and selects."""
        def load(address):
            tick = time.perf_counter()
            count = self._store.read_page(page.ref, address, page.layout.page_extent_bytes)
            self._backing_io["read_seconds"] += time.perf_counter() - tick
            self._backing_io["bytes_read"] += count
            self._backing_io["reloads"] += 1
        return load

    def _run_attention(self, op, action, buffer, attention, out, call, length, execution_context):
        if action.kind != "CachedAttention" or self._pending_pages is None or self._transaction is None:
            raise ValueError("Offloaded attention requires a prepared transaction")
        bindings, attributes = action.bindings, op.attributes
        query = buffer(op.inputs[0])
        maxima, sums = buffer("__attention", ctypes.c_double), buffer("__stream_sums", ctypes.c_double)
        for index in range(len(maxima)):
            maxima[index] = float("-inf")
        ctypes.memset(ctypes.addressof(sums), 0, ctypes.sizeof(sums))
        if len(self._pending_pages) != len(self._transaction.attention_pages):
            raise ValueError("Logical pages differ from the admitted prefix")
        for phase in (0, 1):
            for descriptor, page in zip(self._transaction.attention_pages, self._pending_pages):
                if page.codec != descriptor.codec:
                    raise ValueError("Page codec differs from its logical descriptor")
                if isinstance(page, _OffloadedPage):
                    if self._reload_cache is None:
                        raise ValueError("Reload requires an admitted slot")
                    if (page.ref.descriptor.page_index != descriptor.page_index or
                            page.ref.descriptor.layout_identity != descriptor.layout_identity or
                            page.ref.descriptor.valid_tokens != descriptor.valid_tokens):
                        raise ValueError("Backing reference differs from the logical page")
                    address = self._reload_cache.acquire(page.ref, self._loader(page))
                else:
                    address = page.address
                if not address:
                    raise ValueError("Attention requires a live page or reload slot")
                size = descriptor.valid_tokens * page.layout.token_bytes
                key = (ctypes.c_uint8 * size).from_address(address + page.layout.buffer_offset(bindings["layer"], "key"))
                value = (ctypes.c_uint8 * size).from_address(address + page.layout.buffer_offset(bindings["layer"], "value"))
                call("nexa_causal_gqa_attention_page", query, len(query), key, size, value, size,
                     _CODEC_IDS[page.codec], self.policy.group_size, descriptor.logical_start,
                     descriptor.valid_tokens, bindings["past_length"], length, attributes["num_heads"],
                     attributes["num_key_value_heads"], attributes["head_dim"], phase,
                     maxima, len(maxima), sums, len(sums))
        call("nexa_causal_gqa_attention_finish", maxima, len(maxima), sums, len(sums),
             length, attributes["num_heads"], attributes["head_dim"], out, len(out))

    def _report(self, graph, plan, *, executed, execution_context=None):
        report = TransformerSession._report(self, graph, plan, executed=executed)
        transition = self._transaction if executed else None
        descriptors = transition.final_pages if transition else self._page_descriptors
        total = transition.new_length if transition else self.cache_length
        resident = sum(p.page_allocation_bytes for p in descriptors if p.tier != "cold")
        cold = [p for p in descriptors if p.tier == "cold"]
        cold_extent = sum(self._tier_plan.layout(p.codec).page_extent_bytes for p in cold)
        physical = (self._offload_plan.transition_memory(transition) if transition else
                    {"old_resident_bytes": resident, "attention_pages_bytes": resident,
                     "migration_pages_bytes": resident, "final_resident_bytes": resident,
                     "reload_slots_used": 0, "reload_slot_bytes": 0})
        report.update({"decode_strategy": "offloaded_tiered_paged_incremental_kv", "persistent_kv_cache": True,
                       "paged_kv_cache": True, "kv_codec": "mixed", "kv_group_size": self.policy.group_size,
                       "kv_policy": self.policy.to_dict(), "max_chunk_length": self.max_chunk_length,
                       "kv_partial_page_policy": "hot_f32_until_full", "kv_cache_plan": self._offload_plan.to_dict(),
                       "kv_backing_store": {"format": "NEXAKV_PAGE_V1", "lifetime": "private_ephemeral_session",
                                            "eviction": "all_cold_q3", "reload_slots": self._offload_plan.reload_slots,
                                            "reload_replacement_policy": RELOAD_POLICY_ID,
                                            "reload_cache_scope": "session_until_reference_retired",
                                            "attention_passes": 2, "requantize_on_reload": False},
                       "kv_pages": [dict(p.to_dict(), residency="backing_store" if p.tier == "cold" else "ram")
                                    for p in descriptors],
                       "context_length": total, "cache_length": total,
                       "cache_position_offset": execution_context.position_offset if execution_context else 0,
                       "model_ir_scope": "chunk graph; logical codec transition and physical residency are separate"})
        if execution_context is not None:
            schedule = execution_context.to_dict()
            for step in schedule["steps"]:
                if step["kind"] == "CachedAttention":
                    for name in ("key_offset", "value_offset"):
                        step["bindings"].pop(name)
                    step["bindings"]["page_layout_source"] = "logical_codec_transition.attention_pages"
                    step["bindings"]["attention_passes"] = 2
            # The codec transition's byte fields describe the resident baseline;
            # physical_memory is the authoritative offloaded allocation plan.
            report["execution_plan"] = {"schema_version": 1, "kind": "offloaded_tiered_kv_step",
                                        "operator_schedule": schedule,
                                        "logical_codec_transition": transition.to_dict(),
                                        "logical_transition_byte_scope": "hypothetical all-resident codec layouts",
                                        "physical_memory": physical}
        memory = report["memory"]
        workspace = memory["managed_buffers_peak_bound_bytes"]
        migration_scratch = self._tier_plan.migration_scratch_bytes if transition and transition.migrations else 0
        # Slots outlive the call, so residency is the larger of what this
        # transition may occupy and what earlier calls already allocated.
        cache = self._reload_cache
        reload_bytes = max(physical["reload_slot_bytes"], cache.allocated_bytes if cache else 0)
        attention_peak = workspace + physical["attention_pages_bytes"] + reload_bytes
        if physical["reload_slot_bytes"]:
            attention_peak += KV_STORE_BUFFER_BYTES
        migration_peak = physical["migration_pages_bytes"] + reload_bytes + max(migration_scratch,
                           KV_STORE_BUFFER_BYTES if transition and any(m.target.tier == "cold"
                                                                      for m in transition.migrations) else 0)
        memory.update({"scope": "managed CPU arena, resident KV, bounded reload cache and I/O; excludes RSS/VRAM",
                       "kv_page_tokens": self.page_tokens,
                       "kv_resident_page_count": len(descriptors) - len(cold),
                       "kv_offloaded_page_count": len(cold), "kv_offloaded_payload_bytes": cold_extent,
                       "kv_resident_allocation_bytes": resident,
                       "persistent_kv_bytes": sum(self._tier_plan.layout(p.codec).page_payload_bytes
                                                  for p in descriptors if p.tier != "cold"),
                       "kv_logical_payload_bytes": sum(self._tier_plan.layout(p.codec).page_payload_bytes for p in descriptors),
                       "kv_valid_prefix_bytes": sum(p.valid_tokens * self._tier_plan.layout(p.codec).token_bytes *
                                                    self.config.num_hidden_layers * 2 for p in descriptors),
                       "kv_valid_prefix_f32_bytes": total * self.config.num_hidden_layers * 2 * self._cache_plan.token_bytes,
                       "kv_f32_bytes_per_token": self.config.num_hidden_layers * 2 * self._cache_plan.token_bytes,
                       "kv_tier_counts": {tier: sum(p.tier == tier for p in descriptors) for tier in ("hot", "warm", "cold")},
                       "kv_full_dequantized_buffer_bytes": 0, "kv_prefix_copy_buffer_bytes": 0,
                       "kv_page_table_bytes": 0, "attention_score_scratch_bytes": 0,
                       "kv_streaming_scratch_bytes": plan.allocations["__attention"].size_bytes +
                                                     plan.allocations["__stream_sums"].size_bytes,
                       "kv_reload_slot_bytes": physical["reload_slot_bytes"],
                       "kv_reload_slot_capacity_bytes": self._offload_plan.reload_slot_bytes,
                       "kv_reload_slots": self._offload_plan.reload_slots,
                       "kv_reload_slots_used": physical["reload_slots_used"],
                       "kv_reload_cache_capacity_bytes": self._offload_plan.reload_cache_capacity_bytes,
                       "kv_reload_cache_allocated_bytes": cache.allocated_bytes if cache else 0,
                       "kv_reload_cache_entries": cache.entry_count if cache else 0,
                       "kv_backing_io_scratch_capacity_bytes": KV_STORE_BUFFER_BYTES,
                       "kv_reserved_capacity_bytes": self._offload_plan.allocation_limit_bytes,
                       "kv_migration_scratch_capacity_bytes": self._tier_plan.migration_scratch_bytes,
                       "kv_migration_scratch_bytes": migration_scratch,
                       "kv_transaction_peak_allocation_bytes": physical["migration_pages_bytes"],
                       "kv_attention_phase_pages_bytes": physical["attention_pages_bytes"],
                       "kv_attention_phase_bound_bytes": attention_peak,
                       "kv_migration_phase_bound_bytes": migration_peak,
                       "capacity_managed_buffers_bound_bytes": self._workspace_template.peak_bytes["host"] +
                           self._workspace_template.reserves["host"] - self.reserve,
                       "managed_buffers_peak_bound_bytes": max(attention_peak, migration_peak)})
        return report

    def _persist_cold(self, transition, final_pages, staged_refs):
        for index, descriptor in enumerate(transition.final_pages):
            page = final_pages[index]
            if descriptor.tier != "cold" or isinstance(page, _OffloadedPage):
                continue
            tick = time.perf_counter()
            ref = self._store.write_page(descriptor, page.address, page.layout.page_extent_bytes)
            try:
                staged_refs.append(ref)
            except BaseException:
                self._store.remove(ref)
                raise
            self._backing_io["write_seconds"] += time.perf_counter() - tick
            self._backing_io["bytes_written"] += ref.file_bytes
            self._backing_io["evictions"] += 1
            final_pages[index] = _OffloadedPage(ref, page.layout)

    def _finalize_report(self, report, candidate, chunk, transition, migration, migration_seconds):
        report = super()._finalize_report(report, candidate, chunk, transition, migration, migration_seconds)
        io = self._backing_io
        reuse = self._reload_cache.delta(self._reload_baseline)
        report["io"].update({"kv_backing_bytes_read": io["bytes_read"], "kv_backing_bytes_written": io["bytes_written"],
                             "kv_page_reloads": io["reloads"], "kv_pages_evicted": io["evictions"],
                             # Requests served by a slot, and the bytes those
                             # requests would have read from the backing store.
                             "kv_reload_cache_hits": reuse["hits"], "kv_reload_cache_misses": reuse["misses"],
                             "kv_reload_bytes_avoided": reuse["bytes_avoided"],
                             "kv_reload_slot_admissions": reuse["admissions"],
                             "kv_reload_slot_evictions": reuse["evictions"]})
        report["timing"].update({"kv_backing_read_seconds": io["read_seconds"],
                                 "kv_backing_write_seconds": io["write_seconds"]})
        report["timing"]["read_seconds"] += io["read_seconds"]
        return report

    def _collect_retired(self, *, best_effort=False):
        # Cleanup failure must not turn a published token into a failed decode.
        # Work in place: even an allocation failure/cancellation during cleanup
        # preserves ownership for retry. Before a new transaction cancellation
        # propagates; after publication it only postpones garbage collection.
        for index in range(len(self._retired_refs) - 1, -1, -1):
            try:
                ref = self._retired_refs[index]
                # Removal can have completed immediately before an interrupt.
                # Reconcile the queue with the store's ownership before retry.
                if self._store.contains(ref):
                    self._store.remove(ref)
            except OSError:
                continue
            except BaseException:
                if best_effort:
                    return
                raise
            del self._retired_refs[index]

    def _perform(self, token_ids, *, mode):
        self._check_open()
        chunk = self._validate_tokens(token_ids)
        if len(chunk) > self.max_chunk_length:
            raise ValueError("Input chunk exceeds max_chunk_length; split the prompt")
        if mode == "decode" and not self._tokens:
            raise ValueError("append/decode requires a successful prefill first")
        candidate = chunk if mode == "prefill" else self._validate_tokens(self._tokens + chunk)
        transition = self._tier_plan.plan_transition(self.cache_length, len(chunk), mode, self._page_descriptors)
        physical = self._offload_plan.transition_memory(transition)
        context = self._context(len(chunk), mode=mode)
        self._make_plan(len(chunk), execution_context=context)
        self._ensure_store()
        self._collect_retired()
        old_pages = self._pages
        pending = list(old_pages) if mode == "decode" else []
        staged, staged_refs = [], []
        committed = False
        self._backing_io = {"bytes_read": 0, "bytes_written": 0, "reloads": 0, "evictions": 0,
                            "read_seconds": 0.0, "write_seconds": 0.0}
        self._reload_baseline = self._reload_cache.counters
        if not physical["reload_slot_bytes"]:
            # No cold page can be reachable from this prefix; nothing may stay.
            self._reload_cache.clear()
        try:
            for _ in transition.fresh_pages:
                page = self._allocate_page()
                try:
                    staged.append(page)
                except BaseException:
                    page.release()
                    raise
                pending.append(page)
            self._pending_pages, self._transaction = pending, transition
            output, report = self._execute(chunk, execution_context=context)
            # Slots survive migration: their pages are immutable copies whose
            # capacity the same admission already reserved.
            final_pages = list(pending)
            started = time.perf_counter()
            migration = self._migrate_pages(transition, pending, final_pages, staged)
            elapsed = time.perf_counter() - started
            started = time.perf_counter()
            self._persist_cold(transition, final_pages, staged_refs)
            persistence_elapsed = time.perf_counter() - started
            report = self._finalize_report(report, candidate, chunk, transition, migration, elapsed)
            report["timing"]["kv_eviction_wall_seconds"] = persistence_elapsed
            report["timing"]["execution_wall_seconds"] += persistence_elapsed
            retained = {id(page) for page in final_pages}
            retired = self._retired_refs + [p.ref for p in old_pages if isinstance(p, _OffloadedPage) and id(p) not in retained]
            self._tokens, self._last_report, self._pages, self._page_descriptors, self._retired_refs = (
                candidate, report, final_pages, transition.final_pages, retired)
            committed = True
        finally:
            self._pending_pages = self._transaction = self._backing_io = None
            self._reload_baseline = None
            if committed:
                # Only pages still committed may be served from a slot; a
                # retired reference is never requested again.
                self._reload_cache.retain([page.ref for page in final_pages if isinstance(page, _OffloadedPage)])
                for page in old_pages:
                    if id(page) not in retained:
                        page.release()
                for page in staged:
                    if id(page) not in retained:
                        page.release()
            else:
                # A failed load can leave a slot holding a partial page, and a
                # discarded destination invalidates whatever it was staging.
                self._reload_cache.clear()
                for page in staged:
                    page.release()
                self._retired_refs.extend(staged_refs)
            self._collect_retired(best_effort=True)
        return output

    def reset(self):
        self._check_open()
        # Prepare the empty report without changing committed state on failure.
        graph, plan = self._make_plan(self.max_sequence_length)
        # Every slot caches a page this reset retires, so free them first and
        # let the empty report describe the residency the caller will observe.
        self._reload_cache.clear()
        descriptors = self._page_descriptors
        try:
            self._page_descriptors = ()
            report = self._report(graph, plan, executed=False)
        finally:
            self._page_descriptors = descriptors
        report.update({"context_length": 0, "cache_length": 0, "cache_position_offset": 0})
        report["memory"]["kv_valid_prefix_f32_bytes"] = 0
        retired = self._retired_refs + [p.ref for p in self._pages if isinstance(p, _OffloadedPage)]
        old = self._pages
        self._tokens, self._last_report, self._pages, self._page_descriptors, self._retired_refs = (), report, [], (), retired
        for page in old:
            page.release()
        self._collect_retired(best_effort=True)

    def close(self):
        try:
            super().close()
        finally:
            if self._reload_cache is not None:
                self._reload_cache.clear()
            if self._store is not None:
                self._store.close()
                self._store = None
                self._retired_refs.clear()

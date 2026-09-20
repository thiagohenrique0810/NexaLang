"""Transactional CPU page aging with hot F32, warm Q4 and cold Q3 KV.

Attention consumes heterogeneous pages directly. Re-encoding runs after each
complete chunk and before commit; its quantization history therefore depends on
chunk boundaries. All source pages survive until publication succeeds.
"""
from __future__ import annotations

import ctypes
import math
import time

from compiler.tiered_kv_plan import TieredKVPolicy, TieredKVCachePlan
from compiler.planner.memory import MemoryRequest
from .format import READ_CHUNK_BYTES
from .paged import PagedTransformerSession, _KVPage
from .transformer import ALIGNMENT, TransformerSession, _load_kernels, _positive

_CODEC_IDS = {"f32": 0, "q4": 4, "q3": 3}


class _TieredPage(_KVPage):
    def __init__(self, layout):
        self.layout = layout
        self.codec = layout.codec
        self.allocation_bytes = layout.page_allocation_bytes
        super().__init__(self.allocation_bytes)


class TieredTransformerSession(PagedTransformerSession):
    """One sequence with a fixed page-age policy, no eviction or concurrent calls."""
    def __init__(self, bundle_path, *, page_tokens=16, max_chunk_length=None,
                 hot_pages=1, warm_pages=1, kv_group_size=32,
                 kv_quality_max_rmse=None, kv_retain_pages=0, **kwargs):
        self._policy = TieredKVPolicy(hot_pages, warm_pages, kv_group_size,
                                      kv_quality_max_rmse, kv_retain_pages)
        self._tier_plan = None
        self._page_descriptors = ()
        self._retained = {}
        self._transaction = None
        if any(name in kwargs for name in ("kv_codec", "kv_bits", "kv_seed", "kv_codebook_f32le")):
            raise ValueError("Age tiers use fixed F32/Q4/Q3 codecs and do not accept homogeneous codec options")
        super().__init__(bundle_path, page_tokens=page_tokens, max_chunk_length=max_chunk_length,
                         kv_codec="f32", **kwargs)

    @property
    def policy(self):
        return self._policy

    @property
    def resident_kv_bytes(self):
        return sum(page.allocation_bytes for page in self._pages)

    def _configure_cache(self):
        if self._cache_plan is None:
            self._tier_plan = TieredKVCachePlan(self.config, self.max_sequence_length,
                                                self.page_tokens, self.policy)
            self._cache_plan = self._tier_plan.layout("f32")  # New tokens always enter hot pages.
            self.kv_group_size = self.policy.group_size
            self.max_chunk_length = _positive(
                self.max_sequence_length if self._requested_chunk_length is None else self._requested_chunk_length,
                "max_chunk_length")
            if self.max_chunk_length > self.max_sequence_length:
                raise ValueError("Chunk capacity exceeds the session context capacity")
            self._tier_plan.allocation_limit_bytes(self.max_chunk_length)

    def _allocate_page(self, codec="f32"):
        return _TieredPage(self._tier_plan.layout(codec))

    def _fork_options(self):
        options = super()._fork_options()
        # Tiers fix the codecs; the policy itself defines the page layouts.
        for name in ("kv_codec", "kv_bits", "kv_seed", "kv_codebook_f32le"):
            options.pop(name)
        options.update({"hot_pages": self.policy.hot_pages, "warm_pages": self.policy.warm_pages,
                        "kv_group_size": self.policy.group_size,
                        "kv_quality_max_rmse": self.policy.quality_max_rmse,
                        "kv_retain_pages": self.policy.retain_pages})
        return options

    def _layout_identity(self):
        # The F32 layout alone ignores the policy and the packed group size,
        # which decide how an inherited Q4/Q3 page is read back.
        return self._tier_plan.to_json(indent=None)

    def _adopt_state(self, parent):
        # Aging re-encodes a page into a *new* one and releases the source, so
        # a migration is private to the sequence performing it: the shared page
        # stays valid, and each sequence may pay that re-encode separately.
        self._page_descriptors = parent._page_descriptors if parent is not None else ()
        # The retained set describes the inherited pages, not a private decision.
        self._retained = dict(parent._retained) if parent is not None else {}

    def _make_plan(self, length, *, execution_context=None):
        self._configure_cache()
        if execution_context is None:
            length = min(length, self.max_chunk_length)
        context = execution_context or self._context(length)
        graph = context.graph
        if next(t.shape for t in graph.tensors if t.name == "tokens") != (length,):
            raise ValueError("Execution chunk differs from its tiered KV schedule")
        end = len(context.steps) + 2
        prefix = self.max_sequence_length if execution_context is None else context.new_length
        pages = self._cache_plan.page_count(prefix)
        requests = [*context.activation_requests,
                    MemoryRequest("__key_pages", pages * ctypes.sizeof(ctypes.c_void_p), 0, end),
                    MemoryRequest("__value_pages", pages * ctypes.sizeof(ctypes.c_void_p), 0, end),
                    MemoryRequest("__page_codecs", pages, 0, end),
                    MemoryRequest("__page_bytes", pages * ctypes.sizeof(ctypes.c_size_t), 0, end)]
        reserve = (self._tier_plan.allocation_limit_bytes(self.max_chunk_length) +
                   self._tier_plan.migration_scratch_bytes)
        return graph, self._plan_workspace(length, requests, end, attention_length=prefix,
                                           extra_reserve=reserve)

    def _state_action(self, action, buffer, execution_context, call):
        if action.kind == "CacheWrite":
            for segment in action.bindings["segments"]:
                if self._pending_pages[segment["page_index"]].codec != "f32":
                    raise ValueError("New KV tokens require an uncompressed hot page")
        return super()._state_action(action, buffer, execution_context, call)

    def _run_attention(self, op, action, buffer, attention, out, call, length, execution_context):
        if action.kind != "CachedAttention" or self._pending_pages is None or self._transaction is None:
            raise ValueError("Mixed attention requires a prepared tiered transaction")
        bindings, attributes = action.bindings, op.attributes
        query = buffer(op.inputs[0])
        keys, values = buffer("__key_pages", ctypes.c_void_p), buffer("__value_pages", ctypes.c_void_p)
        codecs, sizes = buffer("__page_codecs", ctypes.c_uint8), buffer("__page_bytes", ctypes.c_size_t)
        count = bindings["page_count"]
        if any(len(table) != count for table in (keys, values, codecs, sizes)) or len(self._pending_pages) != count:
            raise ValueError("Mixed attention tables differ from the planned prefix")
        for index, page in enumerate(self._pending_pages):
            descriptor = self._transaction.attention_pages[index]
            if not page.address or descriptor.codec != page.codec:
                raise ValueError("Mixed attention page differs from its admitted descriptor")
            keys[index] = page.address + page.layout.buffer_offset(bindings["layer"], "key")
            values[index] = page.address + page.layout.buffer_offset(bindings["layer"], "value")
            codecs[index] = _CODEC_IDS[page.codec]
            sizes[index] = self.page_tokens * page.layout.token_bytes
        pointer_table = ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8))
        call("nexa_causal_gqa_attention_paged_mixed", query, len(query),
             ctypes.cast(keys, pointer_table), count, ctypes.cast(values, pointer_table), count,
             codecs, count, sizes, count, self.page_tokens, self.policy.group_size,
             bindings["past_length"], length, attributes["num_heads"],
             attributes["num_key_value_heads"], attributes["head_dim"], attention, len(attention), out, len(out))

    def _report(self, graph, plan, *, executed, execution_context=None):
        report = TransformerSession._report(self, graph, plan, executed=executed)
        transition = self._transaction if executed else None
        descriptors = transition.final_pages if transition is not None else self._page_descriptors
        total = transition.new_length if transition is not None else self.cache_length
        resident = sum(page.page_allocation_bytes for page in descriptors)
        live = self._pending_pages if self._pending_pages is not None else self._pages
        residency = self._residency_fields(descriptors, live)
        reserved = self._tier_plan.allocation_limit_bytes(self.max_chunk_length)
        scratch = self._tier_plan.migration_scratch_bytes
        report.update({"decode_strategy": "tiered_paged_incremental_kv", "persistent_kv_cache": True,
                       "paged_kv_cache": True, "kv_codec": "mixed", "kv_group_size": self.policy.group_size,
                       "kv_policy": self.policy.to_dict(), "max_chunk_length": self.max_chunk_length,
                       "kv_partial_page_policy": "hot_f32_until_full",
                       "kv_cache_plan": self._tier_plan.to_dict(),
                       "kv_pages": residency["kv_pages"],
                       "kv_retained_pages": [[index, self._retained[index]]
                                             for index in sorted(self._retained)],
                       "context_length": total, "cache_length": total,
                       "cache_position_offset": execution_context.position_offset if execution_context else 0,
                       "model_ir_scope": "chunk graph; per-page attention layouts and post-compute migrations in execution_plan"})
        if execution_context is not None:
            # The inherited graph schedule describes F32 writes. Its attention
            # offsets are replaced by per-page descriptors for mixed reads.
            schedule = execution_context.to_dict()
            for step in schedule["steps"]:
                if step["kind"] == "CachedAttention":
                    for name in ("key_offset", "value_offset"):
                        step["bindings"].pop(name)
                    step["bindings"]["page_layout_source"] = "page_transition.attention_pages"
            report["execution_plan"] = {"schema_version": 1, "kind": "tiered_kv_step",
                                        "operator_schedule": schedule,
                                        "page_transition": transition.to_dict() if transition else None}
        memory = report["memory"]
        workspace = memory["managed_buffers_peak_bound_bytes"]
        attention_pages = transition.attention_resident_bytes if transition else resident
        migration_pages = transition.transaction_peak_bytes if transition else resident
        migration_scratch = scratch if transition and transition.migrations else 0
        memory.update({"scope": "managed CPU workspace, mixed KV pages/tables, migration buffers and reader scratch; excludes RSS/VRAM",
                       "kv_page_tokens": self.page_tokens,
                       **residency["memory"],
                       "kv_valid_prefix_f32_bytes": total * self.config.num_hidden_layers * 2 * self._cache_plan.token_bytes,
                       "kv_f32_bytes_per_token": self.config.num_hidden_layers * 2 * self._cache_plan.token_bytes,
                       "kv_full_dequantized_buffer_bytes": 0, "kv_prefix_copy_buffer_bytes": 0,
                       "kv_page_table_bytes": sum(plan.allocations[name].size_bytes for name in
                                                  ("__key_pages", "__value_pages", "__page_codecs", "__page_bytes")),
                       "kv_reserved_capacity_bytes": reserved,
                       "kv_migration_scratch_capacity_bytes": scratch,
                       "kv_migration_scratch_bytes": migration_scratch,
                       "kv_transaction_peak_allocation_bytes": migration_pages,
                       "kv_attention_phase_pages_bytes": attention_pages,
                       "kv_attention_phase_bound_bytes": workspace + attention_pages,
                       "kv_migration_phase_bound_bytes": migration_pages + migration_scratch,
                       "capacity_managed_buffers_bound_bytes": self._workspace_template.peak_bytes["host"] +
                           ALIGNMENT - 1 + READ_CHUNK_BYTES + reserved + scratch,
                       "managed_buffers_peak_bound_bytes": max(workspace + attention_pages,
                                                               migration_pages + migration_scratch)})
        return report

    def _residency_fields(self, descriptors, live):
        """Report fields that depend on which page is resident at which codec.

        The quality gate decides that only after the report is built, so the
        computation lives here and is applied a second time on the way out.
        """
        resident = sum(page.page_allocation_bytes for page in descriptors)
        shared = [page for page in live if getattr(page, "shared", False)]
        shared_bytes = sum(page.allocation_bytes for page in shared)
        return {"kv_pages": [page.to_dict() for page in descriptors],
                "memory": {
                    "kv_resident_page_count": len(descriptors),
                    "kv_resident_allocation_bytes": resident,
                    "persistent_kv_bytes": sum(self._tier_plan.layout(page.codec).page_payload_bytes
                                               for page in descriptors),
                    # Shared pages are resident once for the whole process;
                    # summing sequences would count the same page twice.
                    "kv_shared_page_count": len(shared),
                    "kv_shared_allocation_bytes": shared_bytes,
                    "kv_owned_allocation_bytes": resident - shared_bytes,
                    "kv_valid_prefix_bytes": sum(page.valid_tokens *
                                                 self._tier_plan.layout(page.codec).token_bytes *
                                                 self.config.num_hidden_layers * 2 for page in descriptors),
                    "kv_tier_counts": {tier: sum(page.tier == tier for page in descriptors)
                                       for tier in ("hot", "warm", "cold")}}}

    def _publish_retention(self, report, transition, migration, final_pages):
        """Final descriptors and retention set once the gate has spoken."""
        kept = dict(transition.final_retained_pages)
        kept.update(migration["retained"])
        descriptors = transition.final_pages
        if migration["retained"]:
            descriptors = self._tier_plan.desired_pages(transition.new_length, kept)
            residency = self._residency_fields(descriptors, final_pages)
            report["kv_pages"] = residency["kv_pages"]
            report["memory"].update(residency["memory"])
        report["kv_retained_pages"] = [[index, kept[index]] for index in sorted(kept)]
        return descriptors, kept

    def _migrate_pages(self, transition, pending, final_pages, staged):
        ceiling = self.policy.quality_max_rmse
        budget = self.policy.retain_pages - len(transition.final_retained_pages)
        result = {"pages_reencoded": 0, "source_bytes": 0, "target_bytes": 0,
                  "max_abs_error": 0.0, "sum_squared_error": 0.0, "value_count": 0, "rmse": 0.0,
                  "quality_max_rmse": ceiling, "retain_pages_available": max(budget, 0),
                  "pages_retained": 0, "retentions_declined": 0, "discarded_target_bytes": 0,
                  "worst_page_rmse": 0.0, "retained": [], "page_rmse": [],
                  "scope": "error between source and destination reconstructed KV, including the F32 bridge"}
        if not transition.migrations:
            return result
        kernels = _load_kernels()
        scratch_owner = stats_owner = scratch = stats = None
        try:
            scratch_owner = (ctypes.c_float * self.config.head_dim)()
            stats_owner = (ctypes.c_double * 3)()
            # Borrow addresses so a retained native-call exception cannot own
            # these allocations after the finally block releases their owners.
            scratch = (ctypes.c_float * self.config.head_dim).from_address(ctypes.addressof(scratch_owner))
            stats = (ctypes.c_double * 3).from_address(ctypes.addressof(stats_owner))
            for migration in transition.migrations:
                self._check_cancelled()
                index = migration.source.page_index
                source = pending[index]
                target = self._allocate_page(migration.target.codec)
                try:
                    staged.append(target)
                except BaseException:
                    target.release()
                    raise
                rows = migration.source.valid_tokens * self.config.num_key_value_heads
                input_bytes = rows * source.layout.head_row_bytes
                output_bytes = rows * target.layout.head_row_bytes
                page_max = page_squared = 0.0
                page_values = 0
                for layer in range(self.config.num_hidden_layers):
                    for kind in ("key", "value"):
                        inputs = (ctypes.c_uint8 * input_bytes).from_address(
                            source.address + source.layout.buffer_offset(layer, kind))
                        output = (ctypes.c_uint8 * output_bytes).from_address(
                            target.address + target.layout.buffer_offset(layer, kind))
                        status = kernels.nexa_kv_reencode_rows(
                            inputs, input_bytes, _CODEC_IDS[source.codec], _CODEC_IDS[target.codec],
                            rows, self.config.head_dim, self.policy.group_size, output, output_bytes,
                            scratch, len(scratch), stats, len(stats))
                        if status:
                            raise ArithmeticError(f"Native nexa_kv_reencode_rows failed with status {status}")
                        page_max = max(page_max, stats[0])
                        page_squared += stats[1]
                        page_values += int(stats[2])
                page_rmse = math.sqrt(page_squared / page_values) if page_values else 0.0
                result["worst_page_rmse"] = max(result["worst_page_rmse"], page_rmse)
                result["page_rmse"].append([index, page_rmse])
                # The destination exists and was measured before anything was
                # published: a page too damaged to age simply is not adopted.
                if ceiling is not None and page_rmse > ceiling:
                    if budget > 0:
                        budget -= 1
                        result["pages_retained"] += 1
                        result["discarded_target_bytes"] += output_bytes * 2 * self.config.num_hidden_layers
                        result["retained"].append([index, source.codec])
                        continue  # target stays in staged and is released on commit
                    result["retentions_declined"] += 1
                final_pages[index] = target
                result["max_abs_error"] = max(result["max_abs_error"], page_max)
                result["sum_squared_error"] += page_squared
                result["value_count"] += page_values
                result["source_bytes"] += input_bytes * 2 * self.config.num_hidden_layers
                result["target_bytes"] += output_bytes * 2 * self.config.num_hidden_layers
                result["pages_reencoded"] += 1
            result["rmse"] = math.sqrt(result["sum_squared_error"] / result["value_count"]) if result["value_count"] else 0.0
            return result
        finally:
            scratch_owner = stats_owner = scratch = stats = None

    def _finalize_report(self, report, candidate, chunk, transition, migration, migration_seconds):
        report.update({"token_ids": list(candidate), "sequence_length": len(candidate),
                       "processed_tokens": len(chunk), "input_chunk_token_ids": list(chunk),
                       "kv_migration": migration})
        report["io"].update({"kv_bytes_written": len(chunk) * self.config.num_hidden_layers * 2 * self._cache_plan.token_bytes,
                             "kv_prefix_bytes_copied": 0,
                             "kv_migration_source_bytes": migration["source_bytes"],
                             "kv_migration_target_bytes": migration["target_bytes"],
                             "kv_pages_allocated": len(transition.fresh_pages) + len(transition.migrations),
                             "kv_pages_reencoded": migration["pages_reencoded"],
                             "kv_pages_quality_retained": migration["pages_retained"],
                             "kv_migration_discarded_bytes": migration["discarded_target_bytes"]})
        report["timing"]["migration_wall_seconds"] = migration_seconds
        report["timing"]["execution_wall_seconds"] += migration_seconds
        return report

    def _perform(self, token_ids, *, mode):
        self._check_open()
        chunk = self._validate_tokens(token_ids)
        if len(chunk) > self.max_chunk_length:
            raise ValueError("Input chunk exceeds max_chunk_length; split the prompt")
        if mode == "decode" and not self._tokens:
            raise ValueError("append/decode requires a successful prefill first")
        candidate = chunk if mode == "prefill" else self._validate_tokens(self._tokens + chunk)
        transition = self._tier_plan.plan_transition(self.cache_length, len(chunk), mode,
                                                     self._page_descriptors, self._retained)
        context = self._context(len(chunk), mode=mode)
        self._make_plan(len(chunk), execution_context=context)
        old_pages = self._pages
        pending = list(old_pages) if mode == "decode" else []
        staged = []
        committed = False
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
            final_pages = list(pending)
            started = time.perf_counter()
            migration = self._migrate_pages(transition, pending, final_pages, staged)
            elapsed = time.perf_counter() - started
            report = self._finalize_report(report, candidate, chunk, transition, migration, elapsed)
            descriptors, kept = self._publish_retention(report, transition, migration, final_pages)
            # All fallible result preparation precedes the single state publication.
            retained = {id(page) for page in final_pages}
            self._tokens, self._last_report, self._pages, self._page_descriptors, self._retained = (
                candidate, report, final_pages, descriptors, kept)
            committed = True
        finally:
            self._pending_pages = self._transaction = None
            if committed:
                for page in old_pages:
                    if id(page) not in retained:
                        page.release()
                for page in staged:
                    if id(page) not in retained:
                        page.release()
            else:
                for page in staged:
                    page.release()
        return output

    def reset(self):
        self._check_open()
        old_pages = self._pages
        # Compute the empty report before publishing the reset.
        graph, plan = self._make_plan(self.max_sequence_length)
        report = self._report(graph, plan, executed=False)
        report.update({"context_length": 0, "cache_length": 0, "cache_position_offset": 0,
                       "kv_pages": [], "kv_retained_pages": []})
        memory = report["memory"]
        for name in ("kv_resident_page_count", "kv_resident_allocation_bytes", "persistent_kv_bytes",
                     "kv_shared_page_count", "kv_shared_allocation_bytes", "kv_owned_allocation_bytes",
                     "kv_valid_prefix_bytes", "kv_valid_prefix_f32_bytes", "kv_transaction_peak_allocation_bytes",
                     "kv_attention_phase_pages_bytes", "kv_migration_phase_bound_bytes"):
            memory[name] = 0
        memory["kv_tier_counts"] = {tier: 0 for tier in ("hot", "warm", "cold")}
        memory["managed_buffers_peak_bound_bytes"] = plan.peak_bytes["host"] + ALIGNMENT - 1 + READ_CHUNK_BYTES
        memory["kv_attention_phase_bound_bytes"] = memory["managed_buffers_peak_bound_bytes"]
        self._tokens, self._last_report, self._pages, self._page_descriptors = (), report, [], ()
        self._retained = {}
        for page in old_pages:
            page.release()

    def close(self):
        super().close()
        self._page_descriptors = ()
        self._retained = {}

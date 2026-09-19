"""Page geometry, transaction admission and copy-aware paged KV schedules."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.model_config import ModelConfig
from compiler.model_lowering import lower_model
from compiler.paged_kv_plan import (
    MAX_PLAN_SEGMENTS, PagedKVCachePlan, PagedKVStepPlan,
    make_paged_kv_cache_plan, plan_paged_step,
)
from compiler.planner.memory import MemoryPlanner


def tiny(**changes):
    return ModelConfig(**{"name": "TinyPagedKV", "vocab_size": 16, "hidden_size": 16,
                          "intermediate_size": 20, "num_hidden_layers": 2,
                          "num_attention_heads": 4, "num_key_value_heads": 2,
                          "max_position_embeddings": 32, **changes})


class PagedKVLayoutRegressions(unittest.TestCase):
    def test_mha_mqa_and_gqa_heads_fit_distinct_token_major_buffers(self):
        for heads in (1, 2, 4):
            with self.subTest(kv_heads=heads):
                config = tiny(num_key_value_heads=heads)
                plan = make_paged_kv_cache_plan(config, 11, 3)
                width = heads * config.head_dim
                payload = 3 * width * 4
                stride = (payload + 63) // 64 * 64
                self.assertEqual(plan.kv_width, width)
                self.assertEqual(plan.buffer_stride, stride)
                self.assertEqual(plan.page_payload_bytes, 4 * payload)
                self.assertEqual(plan.page_extent_bytes, 4 * stride)
                self.assertEqual(plan.page_allocation_bytes, 4 * stride + 63)
                arena = bytearray(plan.page_extent_bytes)
                visited = set()
                for layer in range(2):
                    for kind_index, kind in enumerate(("key", "value")):
                        offset = plan.buffer_offset(layer, kind)
                        self.assertEqual(offset, (2 * layer + kind_index) * stride)
                        self.assertEqual(offset % 64, 0)
                        for token in range(3):
                            for head in range(heads):
                                begin = plan.head_offset(layer, kind, head) + token * width * 4
                                indices = set(range(begin, begin + config.head_dim * 4))
                                self.assertFalse(indices & visited)
                                self.assertLessEqual(max(indices) + 1, plan.page_extent_bytes)
                                visited.update(indices)
                                arena[begin:begin + config.head_dim * 4] = bytes([1 + head]) * (config.head_dim * 4)
                self.assertEqual(len(visited), plan.page_payload_bytes)

    def test_page_rounding_padding_and_page_larger_than_context(self):
        plan = make_paged_kv_cache_plan(tiny(num_key_value_heads=1), 11, 3)
        self.assertEqual(plan.max_pages, 4)
        self.assertEqual([plan.page_count(length) for length in (0, 1, 3, 4, 9, 10, 11)],
                         [0, 1, 1, 2, 3, 4, 4])
        self.assertEqual((plan.page_payload_bytes, plan.page_extent_bytes), (192, 256))
        larger_page = make_paged_kv_cache_plan(tiny(), 3, 16)
        self.assertEqual(larger_page.max_pages, 1)
        self.assertEqual(larger_page.page_count(3), 1)
        self.assertEqual(larger_page.reservation_pages(2), 2)

    def test_reservation_covers_old_full_cache_and_fresh_prefill_chunk(self):
        plan = make_paged_kv_cache_plan(tiny(), 31, 4)
        self.assertEqual(plan.max_pages, 8)
        for chunk, pages in ((1, 9), (4, 9), (5, 10), (31, 16)):
            self.assertEqual(plan.reservation_pages(chunk), pages)
            self.assertEqual(plan.allocation_limit_bytes(chunk), pages * plan.page_allocation_bytes)
        step = plan_paged_step(lower_model(plan.config, 5), plan, 31, mode="prefill")
        self.assertEqual((step.new_pages, step.resident_pages_peak), (2, 10))
        self.assertEqual(step.resident_pages_peak * plan.page_allocation_bytes, plan.allocation_limit_bytes(5))

    def test_admission_accounts_for_extra_segment_when_decode_starts_inside_page(self):
        plan = make_paged_kv_cache_plan(tiny(), 9, 2)
        with patch("compiler.paged_kv_plan.MAX_PLAN_SEGMENTS", 4):
            # Fresh prefill8 needs four segments, but append8 after token1
            # needs five. The session must reject that declared chunk cap.
            plan_paged_step(lower_model(plan.config, 8), plan, 0, mode="prefill")
            with self.assertRaisesRegex(ValueError, "segment metadata limit"):
                plan.allocation_limit_bytes(8)
            self.assertGreater(plan.allocation_limit_bytes(7), 0)
            for past in (1, 2):
                step = plan_paged_step(lower_model(plan.config, 7), plan, past)
                self.assertEqual(len(step.bindings["layers.0.attention"]["segments"]), 4)
            exact_context = make_paged_kv_cache_plan(plan.config, 8, 2)
            self.assertGreater(exact_context.allocation_limit_bytes(8), 0)

    def test_invalid_dimensions_lengths_heads_and_byte_ranges_are_rejected(self):
        for capacity, page_tokens in ((0, 4), (-1, 4), (True, 4), (33, 4), (12, 0),
                                      (12, -1), (12, True), (12, 1.5)):
            with self.subTest(capacity=capacity, page_tokens=page_tokens), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(tiny(), capacity, page_tokens)
        plan = make_paged_kv_cache_plan(tiny(), 11, 4)
        for length in (-1, True, 1.5, 12):
            with self.subTest(length=length), self.assertRaises(ValueError):
                plan.page_count(length)
        for length in (0, -1, True, 1.5, 12):
            with self.subTest(chunk=length), self.assertRaises(ValueError):
                plan.reservation_pages(length)
        for layer, kind in ((True, "key"), (-1, "key"), (2, "key"), (0, "keys")):
            with self.assertRaises(ValueError):
                plan.buffer_offset(layer, kind)
        for head in (-1, True, 1.5, 2):
            with self.assertRaises(ValueError):
                plan.head_offset(0, "key", head)
        with self.assertRaisesRegex(ValueError, "physical tensors"):
            make_paged_kv_cache_plan(tiny(num_hidden_layers=10 ** 15), 1, 1)
        with self.assertRaisesRegex(ValueError, "byte range"):
            make_paged_kv_cache_plan(tiny(max_position_embeddings=10 ** 30), 10 ** 30, 1)

    def test_layout_json_rejects_tampered_geometry_and_boolean_metadata(self):
        plan = make_paged_kv_cache_plan(tiny(), 11, 3)
        self.assertEqual(PagedKVCachePlan.from_json(plan.to_json()).to_dict(), plan.to_dict())
        changes = [lambda d: d.update(schema_version=True), lambda d: d.update(max_pages=3),
                   lambda d: d.update(page_tokens=True), lambda d: d.update(alignment=32),
                   lambda d: d.update(page_allocation_bytes=d["page_extent_bytes"]),
                   lambda d: d["buffers"]["layers.0.key"].update(offset=64),
                   lambda d: d["buffers"]["layers.0.key"].update(layer=False),
                   lambda d: d["buffers"].pop("layers.1.value"), lambda d: d.update(banks=2)]
        for change in changes:
            serialized = deepcopy(plan.to_dict())
            change(serialized)
            with self.subTest(serialized=serialized), self.assertRaises(ValueError):
                PagedKVCachePlan.from_dict(serialized)
        with self.assertRaises(ValueError):
            PagedKVCachePlan.from_json('{"capacity": 3, "capacity": 4}')


class PagedKVScheduleRegressions(unittest.TestCase):
    def test_segments_cover_chunk_without_crossing_pages_or_copying_prefix(self):
        config = tiny()
        cache = make_paged_kv_cache_plan(config, 19, 4)
        cases = [(3, 6, [(0, 3, 0, 1), (1, 0, 1, 4), (2, 0, 5, 1)]),
                 (4, 4, [(1, 0, 0, 4)]), (7, 1, [(1, 3, 0, 1)]),
                 (7, 2, [(1, 3, 0, 1), (2, 0, 1, 1)])]
        for past, count, expected in cases:
            with self.subTest(past=past, count=count):
                step = plan_paged_step(lower_model(config, count), cache, past)
                segments = step.bindings["layers.0.attention"]["segments"]
                self.assertEqual([tuple(segment.values()) for segment in segments], expected)
                self.assertEqual(step.position_offset, past)
                self.assertEqual(step.new_length, past + count)
                self.assertEqual(step.page_count, cache.page_count(past + count))
                self.assertEqual(step.new_pages, cache.page_count(past + count) - cache.page_count(past))
                self.assertEqual(step.resident_pages_peak, step.page_count)

    def test_prefill_replacement_requires_fresh_pages_even_when_old_pages_are_sufficient(self):
        config = tiny()
        cache = make_paged_kv_cache_plan(config, 16, 4)
        step = plan_paged_step(lower_model(config, 3), cache, 16, mode="prefill")
        self.assertEqual((step.position_offset, step.new_length, step.new_pages, step.resident_pages_peak),
                         (0, 3, 1, 5))
        self.assertEqual(dict(step.bindings["layers.0.attention"]["segments"][0]),
                         {"page_index": 0, "page_token_offset": 0, "source_token_offset": 0, "token_count": 3})
        first = plan_paged_step(lower_model(config, 3), cache, 0, mode="prefill")
        self.assertEqual(first.resident_pages_peak, 1)
        partial = plan_paged_step(lower_model(config, 1), cache, 2)
        self.assertEqual((partial.new_pages, partial.resident_pages_peak), (0, 1))

    def test_layer_bindings_and_action_order_use_rotated_keys_and_final_commit(self):
        config = tiny(num_hidden_layers=3)
        cache = make_paged_kv_cache_plan(config, 19, 4)
        graph = lower_model(config, 3)
        step = plan_paged_step(graph, cache, 5)
        self.assertEqual([action.op_name for action in step.steps if action.kind not in ("CacheWrite", "Commit")],
                         [op.name for op in graph.ops])
        for index, action in enumerate(step.steps):
            if action.kind == "RoPE":
                self.assertEqual(dict(action.bindings), {"position_offset": 5})
            if action.kind == "CachedAttention":
                layer = action.bindings["layer"]
                self.assertEqual(action.bindings["key_input"], f"layers.{layer}.k_rope")
                self.assertEqual(action.bindings["value_input"], f"layers.{layer}.v")
                self.assertEqual(action.bindings["key_offset"], cache.buffer_offset(layer, "key"))
                self.assertEqual(action.bindings["value_offset"], cache.buffer_offset(layer, "value"))
                self.assertEqual(step.steps[index - 1].kind, "CacheWrite")
                self.assertEqual(step.steps[index - 1].bindings, action.bindings)
        self.assertEqual(step.steps[-2].op_name, "logits")
        self.assertEqual(step.steps[-1].to_dict(), {"kind": "Commit", "op_name": None,
                                                 "bindings": {"new_length": 8, "page_count": 2}})

    def test_invalid_state_noncanonical_graph_and_segment_expansion_limit(self):
        config = tiny()
        cache = make_paged_kv_cache_plan(config, 8, 3)
        graph = lower_model(config, 2)
        for past, mode in ((0, "decode"), (-1, "prefill"), (True, "prefill"), (9, "prefill"),
                           (7, "decode"), (1, "append")):
            with self.subTest(past=past, mode=mode), self.assertRaises(ValueError):
                plan_paged_step(graph, cache, past, mode=mode)
        changed_ops = tuple(replace(op, attributes={**op.attributes, "theta": 5000})
                            if op.kind.value == "RoPE" else op for op in graph.ops)
        for malformed in (replace(graph, ops=changed_ops), lower_model(tiny(num_hidden_layers=1), 2)):
            with self.assertRaises(ValueError):
                plan_paged_step(malformed, cache, 1)
        huge = tiny(max_position_embeddings=MAX_PLAN_SEGMENTS + 1)
        huge_graph = lower_model(huge, MAX_PLAN_SEGMENTS + 1)
        huge_cache = make_paged_kv_cache_plan(huge, MAX_PLAN_SEGMENTS + 1, 1)
        with patch("compiler.paged_kv_plan.PagedKVSegment", side_effect=AssertionError("expanded segments")):
            with self.assertRaisesRegex(ValueError, "segment metadata limit"):
                plan_paged_step(huge_graph, huge_cache, 0, mode="prefill")

    def test_step_json_rejects_gaps_overlaps_wrong_pages_and_early_commit(self):
        config = tiny()
        step = plan_paged_step(lower_model(config, 6), make_paged_kv_cache_plan(config, 19, 4), 3)
        self.assertEqual(PagedKVStepPlan.from_json(step.to_json()).to_dict(), step.to_dict())
        copy = next(index for index, action in enumerate(step.steps) if action.kind == "CacheWrite")
        changes = [lambda d: d.update(schema_version=True), lambda d: d.update(resident_pages_peak=1),
                   lambda d: d.update(new_pages=0), lambda d: d["steps"].reverse(), lambda d: d["steps"].pop(),
                   lambda d: d["steps"][copy]["bindings"].update(key_offset=64),
                   lambda d: d["steps"][copy]["bindings"]["segments"][0].update(page_index=True),
                   lambda d: d["steps"][copy]["bindings"]["segments"][0].update(token_count=2),
                   lambda d: d["steps"][copy]["bindings"]["segments"][1].update(source_token_offset=2),
                   lambda d: d["steps"][copy]["bindings"]["segments"].pop(),
                   lambda d: d["activation_requests"][0].update(end=1)]
        for change in changes:
            serialized = deepcopy(step.to_dict())
            change(serialized)
            with self.subTest(serialized=serialized), self.assertRaises(ValueError):
                PagedKVStepPlan.from_dict(serialized)

    def test_nested_segments_are_immutable_shared_records_and_json_is_detached(self):
        config = tiny()
        step = plan_paged_step(lower_model(config, 6), make_paged_kv_cache_plan(config, 19, 4), 3)
        first = step.bindings["layers.0.attention"]["segments"][0]
        second = step.bindings["layers.1.attention"]["segments"][0]
        self.assertIs(first, second)
        with self.assertRaises(FrozenInstanceError):
            first.token_count = 2
        with self.assertRaises(TypeError):
            first["token_count"] = 2
        with self.assertRaises(TypeError):
            step.bindings["layers.0.attention"]["new_length"] = 2
        data = step.to_dict()
        copy = next(action for action in data["steps"] if action["kind"] == "CacheWrite")
        copy["bindings"]["segments"][0]["token_count"] = 2
        self.assertEqual(first.token_count, 1)


class PagedKVLifetimeRegressions(unittest.TestCase):
    def test_copies_release_chunk_kv_only_after_page_writes_finish(self):
        config = tiny()
        step = plan_paged_step(lower_model(config, 6), make_paged_kv_cache_plan(config, 19, 4), 3)
        requests = {request.name: request for request in step.activation_requests}
        self.assertEqual(set(requests), {tensor.name for tensor in step.graph.tensors} - set(step.graph.constants))
        for event, action in enumerate(step.steps, 1):
            if action.kind == "CachedAttention":
                layer = action.bindings["layer"]
                self.assertEqual(requests[f"layers.{layer}.k_rope"].end, event)
                self.assertEqual(requests[f"layers.{layer}.v"].end, event)
                self.assertEqual(requests[f"layers.{layer}.q_rope"].end, event + 1)
                self.assertEqual(requests[f"layers.{layer}.attention"].start, event)
        self.assertEqual(requests["logits"].end, len(step.steps) + 2)

    def test_arena_reuse_and_cross_page_copies_preserve_values_and_committed_prefix(self):
        config = tiny()
        cache = make_paged_kv_cache_plan(config, 19, 4)
        step = plan_paged_step(lower_model(config, 6), cache, 3)
        workspace = MemoryPlanner().plan(step.activation_requests, {"host": 1 << 20})
        arena = bytearray(workspace.peak_bytes["host"])
        pages = [bytearray([99]) * cache.page_extent_bytes for _ in range(step.page_count)]
        expected, ops = {}, {op.name: op for op in step.graph.ops}
        width_bytes = cache.kv_width * 4

        def span(name):
            allocation = workspace.allocations[name]
            return slice(allocation.offset, allocation.offset + allocation.size_bytes)

        def write(name, tag):
            expected[name] = bytes([tag]) * workspace.allocations[name].size_bytes
            arena[span(name)] = expected[name]

        write("tokens", 1)
        for event, action in enumerate(step.steps, 2):
            if action.kind == "Commit":
                continue
            op = ops[action.op_name]
            if action.kind == "CacheWrite":
                for kind in ("key", "value"):
                    name = action.bindings[kind + "_input"]
                    self.assertEqual(arena[span(name)], expected[name])
                    for segment in action.bindings["segments"]:
                        source = segment["source_token_offset"] * width_bytes
                        offset = action.bindings[kind + "_offset"] + segment["page_token_offset"] * width_bytes
                        size = segment["token_count"] * width_bytes
                        pages[segment["page_index"]][offset:offset + size] = arena[span(name)][source:source + size]
                continue
            inputs = op.inputs[:1] if action.kind == "CachedAttention" else op.inputs
            for name in inputs:
                if name not in step.graph.constants:
                    self.assertEqual(arena[span(name)], expected[name], f"{action.op_name} lost {name}")
            for name in op.outputs:
                write(name, event)
            if action.kind == "CachedAttention":
                for kind in ("key", "value"):
                    reconstructed = bytearray()
                    for segment in action.bindings["segments"]:
                        offset = action.bindings[kind + "_offset"] + segment["page_token_offset"] * width_bytes
                        size = segment["token_count"] * width_bytes
                        reconstructed.extend(pages[segment["page_index"]][offset:offset + size])
                    self.assertEqual(reconstructed, expected[action.bindings[kind + "_input"]])
        for layer in range(config.num_hidden_layers):
            for kind in ("key", "value"):
                offset = cache.buffer_offset(layer, kind)
                self.assertEqual(pages[0][offset:offset + 3 * width_bytes], bytes([99]) * (3 * width_bytes))
        self.assertEqual(arena[span("logits")], expected["logits"])
        self.assertLess(workspace.peak_bytes["host"], sum(request.size_bytes for request in step.activation_requests))


if __name__ == "__main__":
    unittest.main()

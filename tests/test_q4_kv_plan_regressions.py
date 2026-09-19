"""Head-wise Q4 KV geometry, exact F32 compatibility, and packed write spans."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.model_config import ModelConfig
from compiler.model_lowering import lower_model
from compiler.paged_kv_plan import (
    MAX_Q4_GROUP_SIZE, PagedKVAction, PagedKVCachePlan, PagedKVStepPlan,
    make_paged_kv_cache_plan, plan_paged_step,
)
from runtime.nexapack.format import decode_q4_row, quantize_q4_row


def config(head_dim=6, kv_heads=2, **changes):
    return ModelConfig(**{"name": "Q4KV", "vocab_size": 16, "hidden_size": 4 * head_dim,
                          "intermediate_size": 28, "num_hidden_layers": 2,
                          "num_attention_heads": 4, "num_key_value_heads": kv_heads,
                          "max_position_embeddings": 128, **changes})


class Q4KVLayoutRegressions(unittest.TestCase):
    def test_head_rows_match_existing_codec_for_odd_groups_tails_and_large_groups(self):
        for dimension, group, expected in ((2, 1, 10), (4, 3, 12), (6, 5, 14),
                                            (64, 32, 40), (66, 32, 60), (6, 16, 12)):
            with self.subTest(dimension=dimension, group=group):
                plan = make_paged_kv_cache_plan(config(dimension), 19, 3, codec="q4", group_size=group)
                encoded = quantize_q4_row([(index - 2) / 10 for index in range(dimension)], group)
                self.assertEqual(plan.head_row_bytes, expected)
                self.assertEqual(len(encoded), expected)
                self.assertEqual(plan.token_bytes, 2 * expected)
                self.assertEqual(plan.page_payload_bytes, 4 * 3 * 2 * expected)
                self.assertEqual(len(decode_q4_row(encoded, dimension, group)), dimension)

    def test_head_and_token_offsets_preserve_mha_mqa_gqa_without_hidden_padding(self):
        for heads in (1, 2, 4):
            with self.subTest(kv_heads=heads):
                plan = make_paged_kv_cache_plan(config(6, heads), 11, 3, codec="q4", group_size=5)
                self.assertEqual(plan.head_row_bytes, 14)
                self.assertEqual(plan.token_bytes, heads * 14)
                occupied = set()
                for layer in range(2):
                    for kind in ("key", "value"):
                        base = plan.buffer_offset(layer, kind)
                        self.assertEqual(base % 64, 0)
                        for token in range(3):
                            for head in range(heads):
                                begin = plan.head_offset(layer, kind, head) + token * plan.token_bytes
                                self.assertEqual(begin, base + (token * heads + head) * 14)
                                span = set(range(begin, begin + 14))
                                self.assertFalse(span & occupied)
                                self.assertLessEqual(max(span) + 1, plan.page_extent_bytes)
                                occupied.update(span)
                self.assertEqual(len(occupied), plan.page_payload_bytes)
                if heads > 1:
                    # Scales at the second head are deliberately not float-aligned.
                    self.assertEqual(plan.head_offset(0, "key", 1) % 4, 2)

    def test_physical_gain_includes_page_padding_scales_and_replacement_reserve(self):
        model = config(64)
        dense = make_paged_kv_cache_plan(model, 128, 16)
        packed = make_paged_kv_cache_plan(model, 128, 16, codec="q4", group_size=32)
        self.assertEqual((dense.token_bytes, packed.token_bytes), (512, 80))
        self.assertEqual((dense.page_extent_bytes, packed.page_extent_bytes), (32768, 5120))
        self.assertEqual((dense.page_allocation_bytes, packed.page_allocation_bytes), (32831, 5183))
        self.assertEqual(packed.reservation_pages(8), 9)
        self.assertEqual(packed.allocation_limit_bytes(8), 46647)
        self.assertLess(packed.allocation_limit_bytes(8), dense.allocation_limit_bytes(8) / 6)
        # Small pages can eliminate a payload reduction through alignment.
        small = config(4, 1)
        dense_small = make_paged_kv_cache_plan(small, 4, 1)
        packed_small = make_paged_kv_cache_plan(small, 4, 1, codec="q4", group_size=3)
        self.assertLess(packed_small.page_payload_bytes, dense_small.page_payload_bytes)
        self.assertEqual(packed_small.page_allocation_bytes, dense_small.page_allocation_bytes)

    def test_defaults_invalid_groups_codecs_and_overflow(self):
        model = config()
        default = make_paged_kv_cache_plan(model, 8, 4, codec="q4")
        explicit = make_paged_kv_cache_plan(model, 8, 4, codec="q4", group_size=32)
        self.assertEqual(default.group_size, 32)
        self.assertEqual(default.to_dict(), explicit.to_dict())
        self.assertEqual(PagedKVCachePlan(model, 8, 4, "q4", 32).to_dict(), default.to_dict())
        for group in (True, False, 0, -1, 1.5, "32", MAX_Q4_GROUP_SIZE + 1):
            with self.subTest(group=group), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(model, 8, 4, codec="q4", group_size=group)
        maximum = make_paged_kv_cache_plan(model, 1, 1, codec="q4", group_size=MAX_Q4_GROUP_SIZE)
        self.assertEqual(maximum.head_row_bytes, 4 + MAX_Q4_GROUP_SIZE // 2)
        for codec in ("q2", "Q4", None, True):
            with self.subTest(codec=codec), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(model, 8, 4, codec=codec)
        for group in (0, 32, True):
            with self.assertRaises(ValueError):
                make_paged_kv_cache_plan(model, 8, 4, group_size=group)
        with self.assertRaisesRegex(ValueError, "byte range"):
            make_paged_kv_cache_plan(config(max_position_embeddings=1 << 63), 1 << 63, 1,
                                    codec="q4", group_size=5)
        with self.assertRaisesRegex(ValueError, "byte range"):
            make_paged_kv_cache_plan(model, 1, 1 << 63, codec="q4", group_size=5)

    def test_q4_metadata_identifies_logical_shape_and_physical_grouped_storage(self):
        plan = make_paged_kv_cache_plan(config(), 11, 3, codec="q4", group_size=5)
        data = plan.to_dict()
        self.assertEqual((data["dtype"], data["logical_dtype"], data["codec"], data["codec_id"],
                          data["codec_version"], data["layout"]),
                         ("q4", "f32", "q4", "Q4_GROUPED", 1, "token_head_grouped"))
        self.assertEqual(data["buffers"]["layers.0.key"]["shape"], [3, 12])
        self.assertEqual(data["buffers"]["layers.0.key"]["size_bytes"], 84)
        self.assertEqual(PagedKVCachePlan.from_json(plan.to_json()).to_dict(), data)

    def test_existing_f32_json_is_byte_identical_and_accepts_no_packed_fields(self):
        model = ModelConfig("Compat", 16, 16, 20, 2, 4, 2, 32)
        plan = make_paged_kv_cache_plan(model, 11, 3)
        step = plan_paged_step(lower_model(model, 6), plan, 3)
        # Captured before the Q4 extension; these pin both canonical F32 schemas.
        self.assertEqual(hashlib.sha256(plan.to_json().encode()).hexdigest(),
                         "24253b1a1eaa0d5eb4e8019b0184757bd57e52939637d7af8efd1b2e0e03734c")
        self.assertEqual(hashlib.sha256(step.to_json().encode()).hexdigest(),
                         "123413cef13de8dd71f12f0770d70e2db0ed2d5c5685ea8755bad376f5be419a")
        self.assertEqual((plan.codec, plan.group_size, plan.head_row_bytes, plan.token_bytes), ("f32", None, 16, 32))
        self.assertEqual(PagedKVCachePlan.from_dict(plan.to_dict()).to_dict(), plan.to_dict())
        for field, value in (("codec", "f32"), ("group_size", None), ("head_row_bytes", 16)):
            data = {**plan.to_dict(), field: value}
            with self.assertRaises(ValueError):
                PagedKVCachePlan.from_dict(data)


class Q4KVScheduleRegressions(unittest.TestCase):
    def test_q4_bindings_preserve_f32_compute_schedule_segments_and_lifetimes(self):
        model = config()
        graph = lower_model(model, 5)
        packed = plan_paged_step(graph, make_paged_kv_cache_plan(model, 19, 3, codec="q4", group_size=5), 2)
        dense = plan_paged_step(graph, make_paged_kv_cache_plan(model, 19, 3), 2)
        self.assertEqual([(s.kind, s.op_name) for s in packed.steps], [(s.kind, s.op_name) for s in dense.steps])
        self.assertEqual(packed.activation_requests, dense.activation_requests)
        self.assertEqual((packed.position_offset, packed.new_length, packed.new_pages, packed.resident_pages_peak),
                         (2, 7, 2, 3))
        for name, binding in packed.bindings.items():
            self.assertEqual(binding["segments"], dense.bindings[name]["segments"])
            self.assertEqual({key: binding[key] for key in ("codec", "group_size", "head_dim",
                                                          "head_row_bytes", "token_bytes")},
                             {"codec": "q4", "group_size": 5, "head_dim": 6, "head_row_bytes": 14, "token_bytes": 28})
        self.assertEqual(PagedKVStepPlan.from_json(packed.to_json()).to_dict(), packed.to_dict())

    def test_closed_action_contract_rejects_partial_mixed_and_inconsistent_q4_metadata(self):
        model = config()
        step = plan_paged_step(lower_model(model, 2), make_paged_kv_cache_plan(model, 11, 3, codec="q4", group_size=5), 2)
        binding = dict(step.bindings["layers.0.attention"])
        mutations = [lambda d: d.pop("codec"), lambda d: d.pop("group_size"),
                     lambda d: d.update(codec="f32"), lambda d: d.update(head_dim=3),
                     lambda d: d.update(head_row_bytes=15), lambda d: d.update(token_bytes=48),
                     lambda d: d.update(group_size=MAX_Q4_GROUP_SIZE + 1),
                     lambda d: d.update(group_size=True), lambda d: d.update(codec_version=1)]
        for mutate in mutations:
            malformed = dict(binding)
            mutate(malformed)
            with self.subTest(binding=malformed), self.assertRaises(ValueError):
                PagedKVAction("CacheWrite", "layers.0.attention", malformed)

    def test_json_rejects_codec_layout_byte_and_mixed_schedule_tampering(self):
        model = config()
        cache = make_paged_kv_cache_plan(model, 11, 3, codec="q4", group_size=5)
        for field, value in (("codec", "f32"), ("codec_id", "TQ01"), ("codec_version", True),
                             ("layout", "page_head_grouped"), ("dtype", "f32"), ("logical_dtype", "q4"),
                             ("head_row_bytes", 15), ("token_bytes", 48), ("group_size", None)):
            data = {**cache.to_dict(), field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                PagedKVCachePlan.from_dict(data)
        step = plan_paged_step(lower_model(model, 5), cache, 2)
        for kind in ("CacheWrite", "CachedAttention"):
            data = deepcopy(step.to_dict())
            action = next(item for item in data["steps"] if item["kind"] == kind)
            for field in ("codec", "group_size", "head_dim", "head_row_bytes", "token_bytes"):
                action["bindings"].pop(field)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                PagedKVStepPlan.from_dict(data)
        data = step.to_dict()
        data["cache_plan"] = make_paged_kv_cache_plan(model, 11, 3).to_dict()
        with self.assertRaises(ValueError):
            PagedKVStepPlan.from_dict(data)

    def test_replacement_peak_and_unaligned_segment_admission_remain_bounded(self):
        cache = make_paged_kv_cache_plan(config(), 9, 2, codec="q4", group_size=5)
        replacement = plan_paged_step(lower_model(cache.config, 3), cache, 9, mode="prefill")
        self.assertEqual((replacement.position_offset, replacement.new_pages, replacement.resident_pages_peak), (0, 2, 7))
        self.assertEqual(replacement.resident_pages_peak * cache.page_allocation_bytes, cache.allocation_limit_bytes(3))
        with patch("compiler.paged_kv_plan.MAX_PLAN_SEGMENTS", 4):
            with self.assertRaisesRegex(ValueError, "segment metadata limit"):
                cache.allocation_limit_bytes(8)
            self.assertGreater(cache.allocation_limit_bytes(7), 0)

    def test_encoded_destination_spans_cover_heads_across_pages_without_touching_prefix(self):
        model = config()
        cache = make_paged_kv_cache_plan(model, 19, 3, codec="q4", group_size=5)
        step = plan_paged_step(lower_model(model, 5), cache, 2)
        pages = [bytearray([0xA5]) * cache.page_extent_bytes for _ in range(step.page_count)]
        expected = {}
        written = [set() for _ in pages]
        for action in step.steps:
            if action.kind != "CacheWrite":
                continue
            binding = action.bindings
            for kind in ("key", "value"):
                rows = []
                for token in range(binding["query_length"]):
                    rows.append(b"".join(quantize_q4_row(
                        [(token + head + dim - 3) / 7 for dim in range(model.head_dim)], cache.group_size)
                        for head in range(model.num_key_value_heads)))
                encoded = b"".join(rows)
                expected[action.op_name, kind] = encoded
                for segment in binding["segments"]:
                    page_index = segment["page_index"]
                    offset = binding[kind + "_offset"] + segment["page_token_offset"] * binding["token_bytes"]
                    source = segment["source_token_offset"] * binding["token_bytes"]
                    size = segment["token_count"] * binding["token_bytes"]
                    self.assertEqual(size, len(b"".join(rows[segment["source_token_offset"]:
                                                           segment["source_token_offset"] + segment["token_count"]])))
                    span = set(range(offset, offset + size))
                    self.assertFalse(written[page_index] & span)
                    self.assertLessEqual(offset + size, cache.page_extent_bytes)
                    written[page_index].update(span)
                    pages[page_index][offset:offset + size] = encoded[source:source + size]
        for binding in step.bindings.values():
            for kind in ("key", "value"):
                prefix = binding[kind + "_offset"]
                self.assertEqual(pages[0][prefix:prefix + 2 * cache.token_bytes], bytes([0xA5]) * (2 * cache.token_bytes))
                reconstructed = bytearray()
                for segment in binding["segments"]:
                    offset = prefix + segment["page_token_offset"] * cache.token_bytes
                    size = segment["token_count"] * cache.token_bytes
                    reconstructed.extend(pages[segment["page_index"]][offset:offset + size])
                self.assertEqual(reconstructed, expected[f'layers.{binding["layer"]}.attention', kind])
        self.assertEqual(sum(map(len, written)), 5 * cache.token_bytes * model.num_hidden_layers * 2)


if __name__ == "__main__":
    unittest.main()

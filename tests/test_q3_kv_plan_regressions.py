"""Q3 page bytes, codec identity, immutable prefixes and prior schema stability."""
from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.model_config import ModelConfig
from compiler.model_lowering import lower_model
from compiler.paged_kv_plan import (
    MAX_Q3_GROUP_SIZE, PagedKVAction, PagedKVCachePlan, PagedKVStepPlan,
    make_paged_kv_cache_plan, plan_paged_step,
)
from q3_reference import decode_q3_row, quantize_q3_row


def config(head_dim=6, kv_heads=2, **changes):
    return ModelConfig(**{"name": "Q3KV", "vocab_size": 16, "hidden_size": 4 * head_dim,
                          "intermediate_size": 28, "num_hidden_layers": 2,
                          "num_attention_heads": 4, "num_key_value_heads": kv_heads,
                          "max_position_embeddings": 128, **changes})


class Q3KVLayoutRegressions(unittest.TestCase):
    def test_head_rows_match_independent_three_bit_codec_for_tails_and_odd_groups(self):
        cases = ((2, 1, 10), (4, 3, 12), (6, 5, 12), (6, 7, 7),
                 (8, 7, 14), (64, 32, 32), (66, 32, 48), (6, 16, 10))
        for dimension, group, expected_bytes in cases:
            with self.subTest(dimension=dimension, group=group):
                plan = make_paged_kv_cache_plan(config(dimension), 19, 3, codec="q3", group_size=group)
                values = [(index - 3) / 10 for index in range(dimension)]
                encoded = quantize_q3_row(values, group)
                self.assertEqual(len(encoded), expected_bytes)
                self.assertEqual(plan.head_row_bytes, expected_bytes)
                self.assertEqual(plan.token_bytes, 2 * expected_bytes)
                self.assertEqual(plan.page_payload_bytes, 4 * 3 * 2 * expected_bytes)
                self.assertEqual(len(decode_q3_row(encoded, dimension, group)), dimension)

    def test_lsb_first_codes_cross_bytes_and_group_tail_padding_is_not_head_padding(self):
        values = [-3, -2, -1, 0, 1, 2, 3, 0]
        encoded = quantize_q3_row(values, 7)
        # Codes 5,6,7,0,1,2,3 cross bytes at positions 2 and 5.
        # The second group stores only a zero tail value and six padded codes.
        self.assertEqual(encoded, struct.pack("<f", 1.0) + bytes.fromhex("f5 11 0d") + bytes(7))
        self.assertEqual(decode_q3_row(encoded, 8, 7), values)
        plan = make_paged_kv_cache_plan(config(8), 11, 3, codec="q3", group_size=7)
        self.assertEqual(plan.head_row_bytes, len(encoded))
        self.assertEqual(plan.head_offset(0, "key", 1), 14)
        self.assertEqual(plan.token_bytes, 28)
        self.assertEqual(plan.buffer_stride, 128)
        bad_code = bytearray(encoded)
        bad_code[4] = (bad_code[4] & ~7) | 4
        with self.assertRaises(ValueError):
            decode_q3_row(bad_code, 8, 7)
        bad_spare_bits = bytearray(encoded)
        bad_spare_bits[6] |= 0x80
        with self.assertRaises(ValueError):
            decode_q3_row(bad_spare_bits, 8, 7)

    def test_all_heads_and_tokens_have_distinct_bytes_for_mha_mqa_and_gqa(self):
        for heads in (1, 2, 4):
            with self.subTest(kv_heads=heads):
                plan = make_paged_kv_cache_plan(config(6, heads), 11, 3, codec="q3", group_size=7)
                occupied = set()
                for layer in range(2):
                    for kind in ("key", "value"):
                        self.assertEqual(plan.buffer_offset(layer, kind) % 64, 0)
                        for token in range(3):
                            for head in range(heads):
                                begin = plan.head_offset(layer, kind, head) + token * plan.token_bytes
                                self.assertEqual(begin, plan.buffer_offset(layer, kind) + (token * heads + head) * 7)
                                span = set(range(begin, begin + 7))
                                self.assertFalse(span & occupied)
                                self.assertLessEqual(max(span) + 1, plan.page_extent_bytes)
                                occupied.update(span)
                self.assertEqual(len(occupied), plan.page_payload_bytes)

    def test_q3_reduces_physical_allocation_vs_q4_after_padding_for_d64(self):
        model = config(64)
        q3 = make_paged_kv_cache_plan(model, 128, 16, codec="q3", group_size=32)
        q4 = make_paged_kv_cache_plan(model, 128, 16, codec="q4", group_size=32)
        self.assertEqual((q3.head_row_bytes, q4.head_row_bytes), (32, 40))
        self.assertEqual((q3.page_extent_bytes, q4.page_extent_bytes), (4096, 5120))
        self.assertEqual((q3.page_allocation_bytes, q4.page_allocation_bytes), (4159, 5183))
        self.assertEqual((q3.allocation_limit_bytes(8), q4.allocation_limit_bytes(8)), (37431, 46647))
        self.assertEqual(q3.reservation_pages(8), q4.reservation_pages(8))
        small3 = make_paged_kv_cache_plan(config(), 8, 1, codec="q3", group_size=5)
        small4 = make_paged_kv_cache_plan(config(), 8, 1, codec="q4", group_size=5)
        self.assertLess(small3.page_payload_bytes, small4.page_payload_bytes)
        self.assertEqual(small3.page_allocation_bytes, small4.page_allocation_bytes)

    def test_defaults_group_limits_codecs_and_overflow_are_strict(self):
        model = config()
        implicit = make_paged_kv_cache_plan(model, 8, 4, codec="q3")
        self.assertEqual(implicit.group_size, 32)
        self.assertEqual(implicit.to_dict(), PagedKVCachePlan(model, 8, 4, "q3", 32).to_dict())
        for group in (True, False, 0, -1, 1.5, "32", MAX_Q3_GROUP_SIZE + 1):
            with self.subTest(group=group), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(model, 8, 4, codec="q3", group_size=group)
        maximum = make_paged_kv_cache_plan(model, 1, 1, codec="q3", group_size=MAX_Q3_GROUP_SIZE)
        self.assertEqual(maximum.head_row_bytes, 4 + 3 * MAX_Q3_GROUP_SIZE // 8)
        for codec in ("q2", "Q3", "Q3_GROUPED", None, True):
            with self.subTest(codec=codec), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(model, 8, 4, codec=codec)
        with self.assertRaises(ValueError):
            make_paged_kv_cache_plan(model, 8, 4, codec="f32", group_size=32)
        with self.assertRaisesRegex(ValueError, "byte range"):
            make_paged_kv_cache_plan(config(max_position_embeddings=1 << 63), 1 << 63, 1, codec="q3")
        with self.assertRaisesRegex(ValueError, "byte range"):
            make_paged_kv_cache_plan(model, 1, 1 << 63, codec="q3")

    def test_f32_and_q4_cache_and_execution_json_remain_byte_identical(self):
        model = ModelConfig("Compat", 16, 16, 20, 2, 4, 2, 32)
        snapshots = {
            "f32": ("24253b1a1eaa0d5eb4e8019b0184757bd57e52939637d7af8efd1b2e0e03734c",
                    "123413cef13de8dd71f12f0770d70e2db0ed2d5c5685ea8755bad376f5be419a"),
            "q4": ("ea10d22d23df7553dbfc22ced3649da8ab0e20cdb494f23c9b10c6fa33477ce1",
                   "9fc4257d413c006275b853608f8f47d06f2f889a9545083179b50d94ec63335b"),
        }
        for codec, hashes in snapshots.items():
            cache = make_paged_kv_cache_plan(model, 11, 3, codec=codec)
            step = plan_paged_step(lower_model(model, 6), cache, 3)
            for plan, expected_hash in zip((cache, step), hashes):
                self.assertEqual(hashlib.sha256(plan.to_json().encode()).hexdigest(), expected_hash)


class Q3KVScheduleRegressions(unittest.TestCase):
    def test_q3_roundtrip_identifies_codec_and_rejects_cross_codec_metadata(self):
        model = config(64)
        cache = make_paged_kv_cache_plan(model, 19, 3, codec="q3", group_size=32)
        data = cache.to_dict()
        self.assertEqual((data["dtype"], data["logical_dtype"], data["codec"], data["codec_id"],
                          data["codec_version"], data["layout"]),
                         ("q3", "f32", "q3", "Q3_GROUPED", 1, "token_head_grouped"))
        self.assertEqual(data["buffers"]["layers.0.key"]["shape"], [3, 128])
        self.assertEqual(data["buffers"]["layers.0.key"]["size_bytes"], 192)
        self.assertEqual(PagedKVCachePlan.from_json(cache.to_json()).to_dict(), data)
        for field, value in (("codec", "q4"), ("codec_id", "Q4_GROUPED"), ("codec_version", True),
                             ("dtype", "q4"), ("head_row_bytes", 40), ("token_bytes", 80),
                             ("group_size", None), ("bits", 3)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                PagedKVCachePlan.from_dict({**data, field: value})
        step = plan_paged_step(lower_model(model, 5), cache, 2)
        self.assertEqual(PagedKVStepPlan.from_json(step.to_json()).to_dict(), step.to_dict())
        mixed = step.to_dict()
        mixed["cache_plan"] = make_paged_kv_cache_plan(model, 19, 3, codec="q4", group_size=32).to_dict()
        with self.assertRaises(ValueError):
            PagedKVStepPlan.from_dict(mixed)

    def test_q3_bindings_are_closed_and_preserve_schedule_lifetimes_and_reservation(self):
        model = config(64)
        graph = lower_model(model, 5)
        cache = make_paged_kv_cache_plan(model, 19, 3, codec="q3")
        step = plan_paged_step(graph, cache, 2)
        dense = plan_paged_step(graph, make_paged_kv_cache_plan(model, 19, 3), 2)
        self.assertEqual(step.activation_requests, dense.activation_requests)
        self.assertEqual([(s.kind, s.op_name) for s in step.steps], [(s.kind, s.op_name) for s in dense.steps])
        binding = dict(step.bindings["layers.0.attention"])
        self.assertEqual((binding["codec"], binding["head_dim"], binding["head_row_bytes"], binding["token_bytes"]),
                         ("q3", 64, 32, 64))
        for field, value in (("codec", "q4"), ("head_row_bytes", 40), ("token_bytes", 80),
                             ("codec", "f32"), ("group_size", True), ("bits", 3)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                PagedKVAction("CacheWrite", "layers.0.attention", {**binding, field: value})
        malformed = dict(binding)
        malformed.pop("head_row_bytes")
        with self.assertRaises(ValueError):
            PagedKVAction("CacheWrite", "layers.0.attention", malformed)
        replacement = plan_paged_step(graph, cache, 19, mode="prefill")
        self.assertEqual((replacement.position_offset, replacement.new_pages, replacement.resident_pages_peak), (0, 2, 9))
        self.assertEqual(replacement.resident_pages_peak * cache.page_allocation_bytes, cache.allocation_limit_bytes(5))
        limited = make_paged_kv_cache_plan(model, 9, 2, codec="q3")
        with patch("compiler.paged_kv_plan.MAX_PLAN_SEGMENTS", 4):
            with self.assertRaisesRegex(ValueError, "segment metadata limit"):
                limited.allocation_limit_bytes(8)
            self.assertGreater(limited.allocation_limit_bytes(7), 0)

    def test_q3_segment_writes_preserve_prefix_and_leave_page_padding_untouched(self):
        model = config(8)
        cache = make_paged_kv_cache_plan(model, 19, 3, codec="q3", group_size=7)
        step = plan_paged_step(lower_model(model, 5), cache, 2)
        pages = [bytearray([0xA5]) * cache.page_extent_bytes for _ in range(step.page_count)]
        written = [set() for _ in pages]
        expected = {}
        for action in step.steps:
            if action.kind != "CacheWrite":
                continue
            binding = action.bindings
            for kind in ("key", "value"):
                rows = [b"".join(quantize_q3_row(
                    [(token + head + dim - 3) / 7 for dim in range(model.head_dim)], cache.group_size)
                    for head in range(model.num_key_value_heads)) for token in range(5)]
                encoded = b"".join(rows)
                expected[action.op_name, kind] = encoded
                for segment in binding["segments"]:
                    page = segment["page_index"]
                    offset = binding[kind + "_offset"] + segment["page_token_offset"] * cache.token_bytes
                    source = segment["source_token_offset"] * cache.token_bytes
                    size = segment["token_count"] * cache.token_bytes
                    span = set(range(offset, offset + size))
                    self.assertFalse(written[page] & span)
                    self.assertLessEqual(offset + size, cache.page_extent_bytes)
                    written[page].update(span)
                    pages[page][offset:offset + size] = encoded[source:source + size]
        for name, binding in step.bindings.items():
            for kind in ("key", "value"):
                base = binding[kind + "_offset"]
                self.assertEqual(pages[0][base:base + 2 * cache.token_bytes], bytes([0xA5]) * (2 * cache.token_bytes))
                reconstructed = bytearray()
                for segment in binding["segments"]:
                    offset = base + segment["page_token_offset"] * cache.token_bytes
                    size = segment["token_count"] * cache.token_bytes
                    reconstructed.extend(pages[segment["page_index"]][offset:offset + size])
                self.assertEqual(reconstructed, expected[name, kind])
        for page, changed in zip(pages, written):
            self.assertTrue(all(value == 0xA5 for index, value in enumerate(page) if index not in changed))
        self.assertEqual(sum(map(len, written)), 5 * cache.token_bytes * model.num_hidden_layers * 2)
        # Corrupt only a write binding; a validated graph cannot bless it.
        malformed = deepcopy(step.to_dict())
        copy = next(action for action in malformed["steps"] if action["kind"] == "CacheWrite")
        copy["bindings"]["segments"][0]["page_token_offset"] = 0
        with self.assertRaises(ValueError):
            PagedKVStepPlan.from_dict(malformed)


if __name__ == "__main__":
    unittest.main()

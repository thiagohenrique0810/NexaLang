"""Portable TQ02 paged KV geometry, codebook identity and schedule admission."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.model_config import ModelConfig
from compiler.model_lowering import lower_model
from compiler.paged_kv_plan import (
    PagedKVAction, PagedKVCachePlan, PagedKVStepPlan,
    make_paged_kv_cache_plan, plan_paged_step,
)
from runtime.nexapack.tq import MAX_TQ_DIM, validate_tq_row


def config(head_dim=8, kv_heads=2, **changes):
    return ModelConfig(**{"name": "TQKV", "vocab_size": 16, "hidden_size": 4 * head_dim,
                          "intermediate_size": 28, "num_hidden_layers": 2,
                          "num_attention_heads": 4, "num_key_value_heads": kv_heads,
                          "max_position_embeddings": 128, **changes})


def codebook(bits=3):
    levels = 1 << bits
    return struct.pack("<" + "f" * levels,
                       *[(index - (levels - 1) / 2) / levels for index in range(levels)]).hex()


def portable_row(dimension, bits, row_index):
    # Independent integer packing, without a native quantizer: geometry and
    # canonical padding are properties of the portable record, not its values.
    codes = sum(((row_index + index) % (1 << bits)) << (index * bits)
                for index in range(dimension))
    return b"TQ02" + struct.pack("<f", row_index + 1) + codes.to_bytes((dimension * bits + 7) // 8, "little")


class TQKVLayoutRegressions(unittest.TestCase):
    def test_rows_match_portable_bitstream_for_all_bits_and_partial_bytes(self):
        for dimension in (2, 4, 8, 64):
            for bits in range(1, 9):
                with self.subTest(dimension=dimension, bits=bits):
                    plan = make_paged_kv_cache_plan(config(dimension), 19, 3, codec="tq", bits=bits)
                    row = portable_row(dimension, bits, 3)
                    validate_tq_row(row, dimension, bits)
                    self.assertEqual(plan.head_row_bytes, len(row))
                    self.assertEqual(plan.token_bytes, 2 * len(row))
                    self.assertEqual(plan.page_payload_bytes, 4 * 3 * 2 * len(row))
                    tail = dimension * bits % 8
                    if tail:
                        malformed = bytearray(row)
                        malformed[-1] |= 1 << tail
                        with self.assertRaises(ValueError):
                            validate_tq_row(malformed, dimension, bits)

    def test_head_and_token_offsets_preserve_mha_mqa_gqa_without_extra_padding(self):
        for heads in (1, 2, 4):
            with self.subTest(kv_heads=heads):
                plan = make_paged_kv_cache_plan(config(4, heads), 11, 3, codec="tq", bits=3)
                self.assertEqual(plan.head_row_bytes, 10)
                occupied = set()
                for layer in range(2):
                    for kind in ("key", "value"):
                        self.assertEqual(plan.buffer_offset(layer, kind) % 64, 0)
                        for token in range(3):
                            for head in range(heads):
                                begin = plan.head_offset(layer, kind, head) + token * plan.token_bytes
                                self.assertEqual(begin, plan.buffer_offset(layer, kind) + (token * heads + head) * 10)
                                span = set(range(begin, begin + 10))
                                self.assertFalse(span & occupied)
                                self.assertLessEqual(max(span) + 1, plan.page_extent_bytes)
                                occupied.update(span)
                self.assertEqual(len(occupied), plan.page_payload_bytes)

    def test_page_reservation_accounts_for_headers_alignment_and_replacement(self):
        model = config(64)
        plans = {codec: make_paged_kv_cache_plan(model, 128, 16, codec=codec)
                 for codec in ("f32", "q4", "q3", "tq")}
        tq = plans["tq"]
        self.assertEqual((tq.head_row_bytes, tq.token_bytes), (32, 64))
        self.assertEqual((tq.page_extent_bytes, tq.page_allocation_bytes), (4096, 4159))
        self.assertEqual((tq.reservation_pages(8), tq.allocation_limit_bytes(8)), (9, 37431))
        self.assertEqual(tq.allocation_limit_bytes(8), plans["q3"].allocation_limit_bytes(8))
        self.assertLess(tq.allocation_limit_bytes(8), plans["q4"].allocation_limit_bytes(8))
        self.assertLess(plans["q4"].allocation_limit_bytes(8), plans["f32"].allocation_limit_bytes(8))
        # TQ's header can make tiny heads larger than their F32 payload. The
        # plan reports physical bytes instead of assuming every codec shrinks.
        tiny = config(2, 1)
        dense = make_paged_kv_cache_plan(tiny, 4, 1)
        packed = make_paged_kv_cache_plan(tiny, 4, 1, codec="tq")
        self.assertGreater(packed.page_payload_bytes, dense.page_payload_bytes)
        self.assertEqual(packed.page_allocation_bytes, dense.page_allocation_bytes)

    def test_defaults_explicit_codebook_and_planning_never_load_native_code(self):
        with patch("runtime.nexapack.tq._load_library", side_effect=AssertionError("native load")):
            provisional = make_paged_kv_cache_plan(config(), 11, 3, codec="tq")
            self.assertEqual((provisional.bits, provisional.seed, provisional.codebook_f32le), (3, 42, None))
            self.assertEqual(PagedKVCachePlan.from_json(provisional.to_json()), provisional)
            concrete = replace(provisional, codebook_f32le=codebook(), seed=-(1 << 31))
            step = plan_paged_step(lower_model(config(), 5), concrete, 2)
            self.assertEqual(PagedKVStepPlan.from_json(step.to_json()).to_dict(), step.to_dict())
            self.assertEqual(concrete.page_allocation_bytes, provisional.page_allocation_bytes)
            self.assertEqual(step.bindings["layers.0.attention"]["seed"], -(1 << 31))
        with self.assertRaises(FrozenInstanceError):
            concrete.bits = 8
        # Verify fresh imports as well as calls: no cached CDLL may hide a load.
        result = subprocess.run([sys.executable, "-S", "-c",
                                 "import ctypes; ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(AssertionError('native load')); "
                                 "import compiler.paged_kv_plan"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_parameters_options_and_allocation_ranges_are_strict(self):
        for bits in (True, False, 0, -1, 9, 3.0, "3"):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(config(), 8, 4, codec="tq", bits=bits)
        for seed in (True, False, -(1 << 31) - 1, 1 << 31, 42.0, "42"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(config(), 8, 4, codec="tq", seed=seed)
        for seed in (-(1 << 31), -1, 0, (1 << 31) - 1):
            self.assertEqual(make_paged_kv_cache_plan(config(), 8, 4, codec="tq", seed=seed).seed, seed)
        for dimension in (6, 66, 2 * MAX_TQ_DIM):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(config(dimension), 8, 4, codec="tq")
        self.assertGreater(make_paged_kv_cache_plan(config(MAX_TQ_DIM), 1, 1, codec="tq").head_row_bytes, 0)
        for group_size in (False, 0, 32):
            with self.assertRaises(ValueError):
                make_paged_kv_cache_plan(config(), 8, 4, codec="tq", group_size=group_size)
        for codec in ("f32", "q4", "q3"):
            for option in ({"bits": 3}, {"seed": 0}, {"codebook_f32le": codebook()}):
                with self.subTest(codec=codec, option=option), self.assertRaises(ValueError):
                    make_paged_kv_cache_plan(config(), 8, 4, codec=codec, **option)
        for codec in ("TQ", "tq02", "TQ_MSE_SRHT", None, True):
            with self.assertRaises(ValueError):
                make_paged_kv_cache_plan(config(), 8, 4, codec=codec)
        with self.assertRaisesRegex(ValueError, "byte range"):
            make_paged_kv_cache_plan(config(max_position_embeddings=1 << 63), 1 << 63, 1, codec="tq")
        with self.assertRaisesRegex(ValueError, "byte range"):
            make_paged_kv_cache_plan(config(), 1, 1 << 63, codec="tq")

    def test_explicit_codebook_is_exact_f32le_sorted_finite_and_safe(self):
        invalid = [False, [], "", codebook()[:-2], codebook().upper(), " " + codebook()[1:],
                   "g" + codebook()[1:], struct.pack("<8f", *([0] * 8)).hex(),
                   struct.pack("<8f", *reversed(range(8))).hex(),
                   struct.pack("<8f", *range(7), float("nan")).hex(),
                   struct.pack("<8f", *range(7), float("inf")).hex(),
                   struct.pack("<8f", *range(6), 2e38, 3e38).hex()]
        for book in invalid:
            with self.subTest(book=book), self.assertRaises(ValueError):
                make_paged_kv_cache_plan(config(), 8, 4, codec="tq", codebook_f32le=book)

    def test_prior_f32_q4_q3_cache_and_execution_json_remain_byte_identical(self):
        model = ModelConfig("Compat", 16, 16, 20, 2, 4, 2, 32)
        snapshots = {
            "f32": ("24253b1a1eaa0d5eb4e8019b0184757bd57e52939637d7af8efd1b2e0e03734c",
                    "123413cef13de8dd71f12f0770d70e2db0ed2d5c5685ea8755bad376f5be419a"),
            "q4": ("ea10d22d23df7553dbfc22ced3649da8ab0e20cdb494f23c9b10c6fa33477ce1",
                   "9fc4257d413c006275b853608f8f47d06f2f889a9545083179b50d94ec63335b"),
            "q3": ("466b23181dff60294d19d248e411b0c243637564b288ed54ed4df0322999e074",
                   "f1bbc22570f295beec1d847250c68cd47c975b00ecca9cbb91da0050e8d5bd2f"),
        }
        for codec, hashes in snapshots.items():
            cache = make_paged_kv_cache_plan(model, 11, 3, codec=codec)
            step = plan_paged_step(lower_model(model, 6), cache, 3)
            for plan, expected_hash in zip((cache, step), hashes):
                self.assertEqual(hashlib.sha256(plan.to_json().encode()).hexdigest(), expected_hash)


class TQKVScheduleRegressions(unittest.TestCase):
    def test_roundtrip_codec_identity_and_closed_metadata_reject_corruption(self):
        cache = make_paged_kv_cache_plan(config(64), 19, 3, codec="tq", seed=-42, codebook_f32le=codebook())
        data = cache.to_dict()
        self.assertEqual((data["dtype"], data["logical_dtype"], data["codec_id"], data["codec_version"],
                          data["transform_id"], data["layout"]),
                         ("tq", "f32", "TQ_MSE_SRHT", 1, "SRHT_XOSHIRO256SS_V1", "token_head_tq02"))
        self.assertNotIn("group_size", data)
        self.assertEqual(data["buffers"]["layers.0.key"]["shape"], [3, 128])
        self.assertEqual(data["buffers"]["layers.0.key"]["size_bytes"], 192)
        self.assertEqual(PagedKVCachePlan.from_json(cache.to_json()).to_dict(), data)
        for name, value in (("codec_id", "TQ01"), ("codec_version", True), ("dtype", "q3"),
                            ("codec", "q3"), ("head_dim", 32), ("head_row_bytes", 40),
                            ("token_bytes", 128), ("bits", True), ("seed", True),
                            ("transform_id", "SRHT"), ("group_size", None), ("codebook_f32le", ""),
                            ("layout", "token_head_grouped"), ("page_allocation_bytes", 1)):
            with self.subTest(field=name), self.assertRaises(ValueError):
                PagedKVCachePlan.from_dict({**data, name: value})
        for name in ("bits", "seed", "codebook_f32le", "transform_id"):
            missing = dict(data)
            missing.pop(name)
            with self.assertRaises(ValueError):
                PagedKVCachePlan.from_dict(missing)
        with self.assertRaises(ValueError):
            PagedKVCachePlan.from_json(cache.to_json().replace('"bits": 3', '"bits": 3, "bits": 3'))

    def test_bindings_share_codebook_and_preserve_lifetimes_segments_and_reserve(self):
        model = config(64)
        graph = lower_model(model, 5)
        cache = make_paged_kv_cache_plan(model, 19, 3, codec="tq", seed=-7, codebook_f32le=codebook())
        step = plan_paged_step(graph, cache, 2)
        dense = plan_paged_step(graph, make_paged_kv_cache_plan(model, 19, 3), 2)
        self.assertEqual(step.activation_requests, dense.activation_requests)
        self.assertEqual([(s.kind, s.op_name) for s in step.steps], [(s.kind, s.op_name) for s in dense.steps])
        self.assertEqual((step.position_offset, step.new_length, step.new_pages, step.resident_pages_peak), (2, 7, 2, 3))
        for name, binding in step.bindings.items():
            self.assertEqual(binding["segments"], dense.bindings[name]["segments"])
            self.assertEqual({key: binding[key] for key in ("codec", "bits", "seed", "head_dim", "head_row_bytes", "token_bytes")},
                             {"codec": "tq", "bits": 3, "seed": -7, "head_dim": 64, "head_row_bytes": 32, "token_bytes": 64})
            self.assertNotIn("codebook_f32le", binding)
            self.assertNotIn("group_size", binding)
        self.assertEqual(PagedKVStepPlan.from_json(step.to_json()).to_dict(), step.to_dict())
        replacement = plan_paged_step(graph, cache, 19, mode="prefill")
        self.assertEqual((replacement.position_offset, replacement.new_pages, replacement.resident_pages_peak), (0, 2, 9))
        self.assertEqual(replacement.resident_pages_peak * cache.page_allocation_bytes, cache.allocation_limit_bytes(5))
        limited = make_paged_kv_cache_plan(model, 9, 2, codec="tq")
        with patch("compiler.paged_kv_plan.MAX_PLAN_SEGMENTS", 4):
            with self.assertRaisesRegex(ValueError, "segment metadata limit"):
                limited.allocation_limit_bytes(8)
            self.assertGreater(limited.allocation_limit_bytes(7), 0)

    def test_action_geometry_and_cross_plan_codec_parameters_are_validated(self):
        model = config(64)
        cache = make_paged_kv_cache_plan(model, 19, 3, codec="tq", codebook_f32le=codebook())
        step = plan_paged_step(lower_model(model, 5), cache, 2)
        binding = dict(step.bindings["layers.0.attention"])
        for name, value in (("codec", "q3"), ("codec", "f32"), ("head_dim", 48),
                            ("head_dim", 256), ("head_row_bytes", 40), ("token_bytes", 80),
                            ("bits", 0), ("bits", True), ("seed", True), ("seed", 1 << 31),
                            ("group_size", 32), ("codebook_f32le", codebook())):
            with self.subTest(field=name), self.assertRaises(ValueError):
                PagedKVAction("CacheWrite", "layers.0.attention", {**binding, name: value})
        for name in ("bits", "seed", "head_dim", "head_row_bytes", "token_bytes"):
            missing = dict(binding)
            missing.pop(name)
            with self.assertRaises(ValueError):
                PagedKVAction("CachedAttention", "layers.0.attention", missing)
        for kind in ("CacheWrite", "CachedAttention"):
            malformed = deepcopy(step.to_dict())
            action = next(action for action in malformed["steps"] if action["kind"] == kind)
            action["bindings"]["seed"] = -1
            with self.assertRaises(ValueError):
                PagedKVStepPlan.from_dict(malformed)
        malformed = deepcopy(step.to_dict())
        malformed["cache_plan"] = replace(cache, seed=-1).to_dict()
        with self.assertRaises(ValueError):
            PagedKVStepPlan.from_dict(malformed)

    def test_segment_writes_preserve_prefix_other_heads_and_alignment_padding(self):
        model = config(4)
        cache = make_paged_kv_cache_plan(model, 19, 3, codec="tq", codebook_f32le=codebook())
        step = plan_paged_step(lower_model(model, 5), cache, 2)
        pages = [bytearray([0xA5]) * cache.page_extent_bytes for _ in range(step.page_count)]
        written = [set() for _ in pages]
        expected = {}
        for action in step.steps:
            if action.kind != "CacheWrite":
                continue
            binding = action.bindings
            for kind in ("key", "value"):
                rows = [b"".join(portable_row(model.head_dim, cache.bits, token + head)
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
        malformed = deepcopy(step.to_dict())
        action = next(action for action in malformed["steps"] if action["kind"] == "CacheWrite")
        action["bindings"]["segments"][0]["page_token_offset"] = 0
        with self.assertRaises(ValueError):
            PagedKVStepPlan.from_dict(malformed)


if __name__ == "__main__":
    unittest.main()

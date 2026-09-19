"""Pure CPU page-aging plans, strict serialization and transactional admission."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.model_config import ModelConfig
from compiler.paged_kv_plan import make_paged_kv_cache_plan
from compiler.tiered_kv_plan import (
    TieredKVPolicy, TieredKVPage, TieredKVMigration,
    TieredKVCachePlan, TieredKVTransition,
)


def tiny(**changes):
    return ModelConfig(**{"name": "TinyTieredKV", "vocab_size": 16, "hidden_size": 128,
                          "intermediate_size": 160, "num_hidden_layers": 2,
                          "num_attention_heads": 2, "num_key_value_heads": 1,
                          "max_position_embeddings": 32, **changes})


class TieredKVPolicyRegressions(unittest.TestCase):
    def test_defaults_and_policy_are_frozen_and_strict(self):
        policy = TieredKVPolicy()
        self.assertEqual((policy.hot_pages, policy.warm_pages, policy.group_size), (1, 1, 32))
        self.assertEqual(TieredKVPolicy.from_json(policy.to_json()), policy)
        with self.assertRaises(FrozenInstanceError):
            policy.hot_pages = 2
        for values in ({"hot_pages": 0}, {"hot_pages": True}, {"warm_pages": -1},
                       {"warm_pages": False}, {"group_size": 0}, {"group_size": True},
                       {"group_size": (1 << 20) + 1}, {"warm_pages": 1.5},
                       {"hot_pages": 65537}, {"warm_pages": 65537}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TieredKVPolicy(**values)
        self.assertEqual(TieredKVPolicy(1, 0, 1 << 20).warm_pages, 0)

    def test_age_includes_newest_partial_page_and_partial_remains_f32(self):
        cache = TieredKVCachePlan(tiny(), 32, 3)
        cases = {0: [], 1: ["f32"], 3: ["f32"], 4: ["q4", "f32"],
                 6: ["q4", "f32"], 7: ["q3", "q4", "f32"],
                 10: ["q3", "q3", "q4", "f32"]}
        for length, codecs in cases.items():
            with self.subTest(length=length):
                pages = cache.desired_pages(length)
                self.assertEqual([p.codec for p in pages], codecs)
                self.assertEqual([p.age_rank for p in pages], list(reversed(range(len(pages)))))
                self.assertEqual([p.logical_start for p in pages], list(range(0, length, 3)))
                self.assertEqual(sum(p.valid_tokens for p in pages), length)
                if pages:
                    self.assertEqual(pages[-1].codec, "f32")
                    self.assertTrue(all(p.valid_tokens == 3 for p in pages[:-1]))

    def test_zero_warm_and_multiple_hot_pages_preserve_the_declared_window(self):
        cache = TieredKVCachePlan(tiny(), 20, 2, TieredKVPolicy(2, 0, 7))
        self.assertEqual([p.codec for p in cache.desired_pages(9)], ["q3", "q3", "q3", "f32", "f32"])
        self.assertEqual([p.codec for p in cache.desired_pages(4)], ["f32", "f32"])
        cache = TieredKVCachePlan(tiny(), 20, 2, TieredKVPolicy(2, 3, 7))
        self.assertEqual([p.tier for p in cache.desired_pages(13)],
                         ["cold", "cold", "warm", "warm", "warm", "hot", "hot"])

    def test_layouts_reuse_grouped_contract_and_identity_covers_parameters(self):
        cache = TieredKVCachePlan(tiny(), 20, 3, TieredKVPolicy(1, 1, 7))
        for codec in ("f32", "q4", "q3"):
            expected = make_paged_kv_cache_plan(cache.config, 20, 3, codec=codec,
                                               group_size=None if codec == "f32" else 7)
            self.assertEqual(cache.layout(codec).to_dict(), expected.to_dict())
        pages = cache.desired_pages(9)
        self.assertEqual([p.codec_id for p in pages], ["Q3_GROUPED", "Q4_GROUPED", "F32_NATIVE"])
        self.assertEqual(len({p.layout_identity for p in pages}), 3)
        other = TieredKVCachePlan(cache.config, 20, 3, TieredKVPolicy(1, 1, 8))
        self.assertNotEqual(pages[0].layout_identity, other.desired_pages(9)[0].layout_identity)
        for invalid in ("tq", "q8", [], None, True):
            with self.assertRaises(ValueError):
                cache.layout(invalid)

    def test_capacity_page_metadata_and_allocation_limits_precede_expansion(self):
        for capacity, page_tokens in ((True, 2), (0, 2), (33, 2), (4, True), (4, 0), (4, 1.5)):
            with self.subTest(capacity=capacity, page_tokens=page_tokens), self.assertRaises(ValueError):
                TieredKVCachePlan(tiny(), capacity, page_tokens)
        with self.assertRaisesRegex(ValueError, "page metadata limit"):
            TieredKVCachePlan(tiny(max_position_embeddings=10**20), 10**20, 1)
        with self.assertRaisesRegex(ValueError, "byte range"):
            TieredKVCachePlan(tiny(), 1, 10**20)
        with self.assertRaisesRegex(ValueError, "physical tensors"):
            TieredKVCachePlan(tiny(num_hidden_layers=10**20), 1, 1)
        with patch("compiler.tiered_kv_plan.MAX_PLAN_SEGMENTS", 4):
            self.assertEqual(TieredKVCachePlan(tiny(), 8, 2).max_pages, 4)
            with self.assertRaises(ValueError):
                TieredKVCachePlan(tiny(), 9, 2)
        cache = TieredKVCachePlan(tiny(), 5, 10)
        self.assertEqual(cache.desired_pages(5)[0].valid_tokens, 5)


class TieredKVTransitionRegressions(unittest.TestCase):
    def test_append_attention_uses_old_codecs_and_migration_follows_attention(self):
        cache = TieredKVCachePlan(tiny(), 20, 2)
        old = cache.desired_pages(6)
        transition = cache.plan_transition(6, 3, "decode", old)
        self.assertEqual([p.codec for p in old], ["q3", "q4", "f32"])
        self.assertEqual([p.codec for p in transition.attention_pages], ["q3", "q4", "f32", "f32", "f32"])
        self.assertEqual([p.codec for p in transition.final_pages], ["q3", "q3", "q3", "q4", "f32"])
        self.assertEqual([(m.source.page_index, m.source.codec, m.target.codec) for m in transition.migrations],
                         [(1, "q4", "q3"), (2, "f32", "q3"), (3, "f32", "q4")])
        self.assertEqual([p.page_index for p in transition.fresh_pages], [3, 4])
        self.assertEqual((transition.position_offset, transition.new_length), (6, 9))
        self.assertEqual(old, cache.desired_pages(6))

    def test_partial_append_reuses_page_and_does_not_migrate_partial(self):
        cache = TieredKVCachePlan(tiny(), 20, 4)
        old = cache.desired_pages(2)
        transition = cache.plan_transition(2, 1, "decode", old)
        self.assertEqual(transition.fresh_pages, ())
        self.assertEqual(transition.migrations, ())
        self.assertEqual(transition.attention_pages[0].valid_tokens, 3)
        self.assertEqual(old[0].valid_tokens, 2)
        self.assertEqual(transition.transaction_peak_bytes, cache.layout("f32").page_allocation_bytes)
        crossed = cache.plan_transition(2, 3, "decode", old)
        self.assertEqual(len(crossed.fresh_pages), 1)
        self.assertEqual(crossed.migrations[0].source.valid_tokens, 4)
        self.assertEqual(crossed.final_pages[-1].codec, "f32")

    def test_prefill_replacement_keeps_old_full_cache_in_peak(self):
        cache = TieredKVCachePlan(tiny(), 20, 2)
        old = cache.desired_pages(20)
        transition = cache.plan_transition(20, 7, "prefill", old)
        self.assertEqual(transition.position_offset, 0)
        self.assertEqual(len(transition.fresh_pages), 4)
        self.assertEqual([p.codec for p in transition.attention_pages], ["f32"] * 4)
        self.assertEqual(transition.old_resident_bytes, sum(p.page_allocation_bytes for p in old))
        fresh_bytes = 4 * cache.layout("f32").page_allocation_bytes
        self.assertEqual(transition.attention_resident_bytes, transition.old_resident_bytes + fresh_bytes)
        replacement = 2 * cache.layout("q3").page_allocation_bytes + cache.layout("q4").page_allocation_bytes
        self.assertEqual(transition.migration_replacement_bytes, replacement)
        self.assertEqual(transition.transaction_peak_bytes, transition.old_resident_bytes + fresh_bytes + replacement)
        self.assertEqual(transition.final_resident_bytes,
                         sum(p.page_allocation_bytes for p in cache.desired_pages(7)))

    def test_initial_large_chunk_can_skip_warm_but_never_evicts(self):
        cache = TieredKVCachePlan(tiny(), 20, 2, TieredKVPolicy(1, 0))
        transition = cache.plan_transition(0, 11, "prefill", ())
        self.assertEqual(transition.old_resident_bytes, 0)
        self.assertEqual(len(transition.migrations), 5)
        self.assertTrue(all((m.source.codec, m.target.codec) == ("f32", "q3") for m in transition.migrations))
        self.assertEqual(sum(p.valid_tokens for p in transition.final_pages), 11)
        self.assertEqual([p.page_index for p in transition.final_pages], list(range(6)))

    def test_reservation_bounds_every_small_append_and_prefill_transaction(self):
        # Includes grouped rows larger than F32; do not assume compression.
        for dim, group_size in ((64, 32), (2, 128)):
            for page_tokens in (1, 2, 4, 16):
                for hot, warm in ((1, 0), (1, 1), (2, 3)):
                    config = tiny(hidden_size=dim * 2)
                    cache = TieredKVCachePlan(config, 9, page_tokens, TieredKVPolicy(hot, warm, group_size))
                    self.assertEqual(cache.migration_scratch_bytes, 4 * dim + 24)
                    actual_capacity = sum(p.page_allocation_bytes for p in cache.desired_pages(9))
                    self.assertEqual(cache.capacity_resident_bytes_bound, actual_capacity)
                    residences = [sum(p.page_allocation_bytes for p in cache.desired_pages(n)) for n in range(10)]
                    self.assertEqual(residences, sorted(residences))
                    for past in range(10):
                        old = cache.desired_pages(past)
                        for chunk in range(1, 10):
                            bound = cache.allocation_limit_bytes(chunk)
                            modes = ("prefill", "decode") if past and past + chunk <= 9 else ("prefill",)
                            for mode in modes:
                                transition = cache.plan_transition(past, chunk, mode, old)
                                self.assertLessEqual(transition.transaction_peak_bytes, bound)
                                self.assertLessEqual(transition.final_resident_bytes, actual_capacity)

    def test_reservation_invalid_chunks_and_overflow_are_rejected(self):
        cache = TieredKVCachePlan(tiny(), 9, 2)
        for chunk in (0, -1, True, 1.5, 10):
            with self.assertRaises(ValueError):
                cache.allocation_limit_bytes(chunk)
        with patch("compiler.tiered_kv_plan.MAX_ALLOCATION_BYTES", cache.capacity_resident_bytes_bound):
            with self.assertRaisesRegex(ValueError, "byte range"):
                cache.allocation_limit_bytes(1)
            with self.assertRaisesRegex(ValueError, "byte range"):
                cache.plan_transition(9, 9, "prefill", cache.desired_pages(9))

    def test_invalid_prefix_policy_identity_and_modes_are_rejected(self):
        cache = TieredKVCachePlan(tiny(), 9, 2)
        old = cache.desired_pages(4)
        cases = [(0, 1, "decode", ()), (4, 0, "decode", old), (4, True, "decode", old),
                 (4, 6, "decode", old), (4, 1, "reset", old), (4, 1, "decode", list(old)),
                 (4, 1, "decode", tuple(reversed(old))), (3, 1, "decode", old),
                 (True, 1, "prefill", ()), (4, 1, "decode", (old[0],)),
                 (4, 1, "decode", (replace(old[0], layout_identity="0" * 64), old[1]))]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                cache.plan_transition(*args)
        other = TieredKVCachePlan(cache.config, 9, 2, TieredKVPolicy(1, 1, 7))
        with self.assertRaises(ValueError):
            cache.plan_transition(4, 1, "decode", other.desired_pages(4))

    def test_plan_is_pure_and_all_descriptors_are_immutable(self):
        with patch("runtime.nexapack.tq.TQCodec.__init__", side_effect=AssertionError("native codec touched")):
            cache = TieredKVCachePlan(tiny(), 9, 2)
            transition = cache.plan_transition(4, 3, "decode", cache.desired_pages(4))
        for value, name, replacement in ((cache, "capacity", 3), (transition, "new_length", 2),
                                         (transition.final_pages[0], "codec", "f32"),
                                         (transition.migrations[0], "target", None)):
            with self.assertRaises(FrozenInstanceError):
                setattr(value, name, replacement)
        with self.assertRaises(TypeError):
            cache._layouts["f32"] = None


class TieredKVSerializationRegressions(unittest.TestCase):
    def test_policy_page_migration_cache_and_transition_json_roundtrip(self):
        cache = TieredKVCachePlan(tiny(), 19, 3, TieredKVPolicy(1, 2, 7))
        transition = cache.plan_transition(7, 5, "decode", cache.desired_pages(7))
        for value in (cache.policy, cache, transition, transition.final_pages[0], transition.migrations[0]):
            with self.subTest(type=type(value).__name__):
                restored = type(value).from_json(value.to_json())
                self.assertEqual(restored, value)
                self.assertEqual(restored.to_dict(), value.to_dict())

    def test_cache_json_rejects_booleans_versions_unknown_fields_and_geometry_drift(self):
        cache = TieredKVCachePlan(tiny(), 19, 3)
        mutations = [lambda d: d.update(schema_version=True), lambda d: d.update(capacity=True),
                     lambda d: d.update(max_pages=9), lambda d: d.update(extra=True),
                     lambda d: d["policy"].update(policy_id="future"),
                     lambda d: d["policy"].update(hot_pages=True),
                     lambda d: d["layouts"]["q3"].update(codec_version=True),
                     lambda d: d["layouts"]["q4"].update(token_bytes=4),
                     lambda d: d["layout_identities"].update(q3="0" * 64),
                     lambda d: d.update(capacity_resident_bytes_bound=1),
                     lambda d: d.update(migration_scratch_bytes=4 * cache.config.head_dim)]
        for mutate in mutations:
            data = deepcopy(cache.to_dict())
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                TieredKVCachePlan.from_dict(data)

    def test_transition_json_rejects_tampered_migrations_or_memory(self):
        cache = TieredKVCachePlan(tiny(), 19, 3)
        transition = cache.plan_transition(7, 5, "decode", cache.desired_pages(7))
        mutations = [lambda d: d.update(schema_version=True), lambda d: d.update(new_length=11),
                     lambda d: d.update(transaction_peak_bytes=d["final_resident_bytes"]),
                     lambda d: d["migrations"].clear(), lambda d: d["fresh_pages"].clear(),
                     lambda d: d["attention_pages"][0].update(logical_start=1),
                     lambda d: d["final_pages"][0].update(age_rank=True),
                     lambda d: d["committed_pages"][0].update(layout_identity="0" * 64),
                     lambda d: d["committed_pages"][0].update(codec_version=True),
                     lambda d: d["migrations"][0]["target"].update(valid_tokens=2)]
        for mutate in mutations:
            data = deepcopy(transition.to_dict())
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                TieredKVTransition.from_dict(data)

    def test_standalone_page_and_migration_reject_wrong_codec_and_promotion(self):
        cache = TieredKVCachePlan(tiny(), 9, 2)
        transition = cache.plan_transition(4, 1, "decode", cache.desired_pages(4))
        migration = transition.migrations[0]
        for changes in ({"tier": "hot"}, {"codec_version": True}, {"group_size": False},
                        {"valid_tokens": 0}, {"layout_identity": "A" * 64}, {"age_rank": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(migration.target, **changes)
        with self.assertRaises(ValueError):
            TieredKVMigration(migration.target, migration.source)
        with self.assertRaises(ValueError):
            TieredKVMigration(migration.source, replace(migration.target, logical_start=1))

    def test_json_rejects_duplicate_keys_nonfinite_and_deep_nesting(self):
        for cls in (TieredKVPolicy, TieredKVCachePlan, TieredKVTransition, TieredKVPage, TieredKVMigration):
            for source in ('{"hot_pages": 1, "hot_pages": 2}', '{"value": NaN}', '[' * 1100 + '0' + ']' * 1100):
                with self.subTest(cls=cls.__name__), self.assertRaises(ValueError):
                    cls.from_json(source)


if __name__ == "__main__":
    unittest.main()

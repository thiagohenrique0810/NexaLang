"""Physical residency admission independently of logical codec history."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.offloaded_kv_plan import OffloadedKVPlan, RELOAD_POLICY_ID
from compiler.tiered_kv_plan import TieredKVCachePlan, TieredKVPolicy
from test_tiered_kv_plan_regressions import tiny


class OffloadedKVPlanRegressions(unittest.TestCase):
    def test_bound_covers_every_partial_append_and_prompt_replacement(self):
        for page_tokens in (1, 2, 3, 9):
            for hot, warm, group in ((1, 0, 1), (1, 1, 32), (2, 3, 4096)):
                tiers = TieredKVCachePlan(tiny(), 8, page_tokens, TieredKVPolicy(hot, warm, group))
                for chunk_limit in (1, 3, 8):
                    plan = OffloadedKVPlan(tiers, chunk_limit)
                    for past in range(9):
                        for chunk in range(1, chunk_limit + 1):
                            for mode in ("prefill", "decode"):
                                if mode == "decode" and (not past or past + chunk > 8):
                                    continue
                                step = tiers.plan_transition(past, chunk, mode, tiers.desired_pages(past))
                                memory = plan.transition_memory(step)
                                old = [p for p in step.committed_pages if p.codec != "q3"]
                                allocated = sum(p.page_allocation_bytes for p in old + list(step.fresh_pages))
                                self.assertEqual(memory["attention_pages_bytes"], allocated)
                                allocated += sum(m.target.page_allocation_bytes for m in step.migrations)
                                self.assertEqual(memory["migration_pages_bytes"], allocated)
                                self.assertLessEqual(allocated, plan.allocation_limit_bytes)
                                self.assertLessEqual(memory["final_resident_bytes"], plan.resident_capacity_bytes)

    def test_reload_cache_capacity_scales_with_slots_and_bounds_touched_pages(self):
        tiers = TieredKVCachePlan(tiny(), 8, 1, TieredKVPolicy(1, 1, 4))
        single = OffloadedKVPlan(tiers, 2)
        self.assertEqual(single.reload_cache_capacity_bytes, single.reload_slot_bytes)
        for slots in (1, 2, 8):
            plan = OffloadedKVPlan(tiers, 2, slots)
            self.assertEqual(plan.reload_cache_capacity_bytes, slots * plan.reload_slot_bytes)
            # Page allocation is unchanged: slots are admitted separately.
            self.assertEqual(plan.allocation_limit_bytes, single.allocation_limit_bytes)
            data = plan.to_dict()
            self.assertEqual(data["reload_slots"], slots)
            self.assertEqual(data["reload_replacement_policy"], RELOAD_POLICY_ID)
            self.assertEqual(data["reload_cache_capacity_bytes"], plan.reload_cache_capacity_bytes)
            for past, expected_cold in ((0, 0), (2, 0), (3, 1), (6, 4)):
                step = tiers.plan_transition(past, 1, "prefill" if not past else "decode",
                                             tiers.desired_pages(past))
                memory = plan.transition_memory(step)
                # Only a cold page of this very prefix can occupy a slot.
                self.assertEqual(memory["reload_slots_used"], min(slots, expected_cold))
                self.assertEqual(memory["reload_slot_bytes"],
                                 memory["reload_slots_used"] * plan.reload_slot_bytes)

    def test_rejects_slot_counts_outside_the_context_and_integer_contract(self):
        tiers = TieredKVCachePlan(tiny(), 8, 2)  # Four pages of two tokens.
        for slots in (0, -1, True, 1.0, None, 5):
            with self.assertRaises(ValueError):
                OffloadedKVPlan(tiers, 2, slots)
        self.assertEqual(OffloadedKVPlan(tiers, 2, 4).reload_slots, 4)

    def test_resident_and_reload_bound_do_not_grow_with_cold_context(self):
        plans = [OffloadedKVPlan(TieredKVCachePlan(tiny(max_position_embeddings=4096), capacity, 2), 2)
                 for capacity in (16, 4096)]
        for field in ("resident_capacity_bytes", "reload_slot_bytes", "allocation_limit_bytes"):
            self.assertEqual(getattr(plans[0], field), getattr(plans[1], field))
        self.assertEqual(plans[1].to_dict()["reload_slots"], 1)

    def test_rejects_wrong_capacity_chunk_or_layout_and_is_immutable(self):
        tiers = TieredKVCachePlan(tiny(), 8, 2)
        for bad in (0, True, -1, 9, 1.5):
            with self.assertRaises(ValueError):
                OffloadedKVPlan(tiers, bad)
        with self.assertRaises(ValueError):
            OffloadedKVPlan(None, 1)
        plan = OffloadedKVPlan(tiers, 2)
        with self.assertRaises(FrozenInstanceError):
            plan.max_chunk_length = 3
        for other, chunk in ((tiers, 3), (TieredKVCachePlan(tiny(), 8, 1), 1)):
            with self.assertRaises(ValueError):
                plan.transition_memory(other.plan_transition(0, chunk, "prefill", ()))


if __name__ == "__main__":
    unittest.main()

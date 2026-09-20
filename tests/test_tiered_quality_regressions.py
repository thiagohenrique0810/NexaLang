"""Quality criteria for age transitions: a page too damaged to age keeps its codec."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.planner.memory import MemoryBudgetError  # noqa: F401  (imported fixture deps)
from compiler.tiered_kv_plan import (
    TieredKVCachePlan, TieredKVPolicy, TieredKVTransition, normalize_retained,
)
from test_tiered_transformer_regressions import _TierFixture


# Measured on this fixture before the gate existed: each decode ages two pages,
# the older one (f32 -> q4 -> q3 bridge) erring roughly twice as much.
LOOSE, TIGHT, MIDDLE = 1.0, 0.0, 0.05


class TieredQualityPolicyRegressions(unittest.TestCase):
    def plan(self, **options):
        from compiler.model_config import ModelConfig
        config = ModelConfig(name="TinyTieredKV", vocab_size=16, hidden_size=128,
                             intermediate_size=160, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1,
                             max_position_embeddings=32)
        policy = TieredKVPolicy(1, 1, 4, options.pop("quality_max_rmse", 0.5),
                                options.pop("retain_pages", 2))
        return TieredKVCachePlan(config, 8, 1, policy)

    def test_a_ceiling_without_a_budget_and_a_budget_without_a_ceiling_are_rejected(self):
        with self.assertRaises(ValueError):
            TieredKVPolicy(1, 1, 4, None, 2)
        for bad in (-1.0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                TieredKVPolicy(1, 1, 4, bad, 1)
        # A ceiling with no budget is legal and simply never retains.
        self.assertEqual(TieredKVPolicy(1, 1, 4, 0.5, 0).retain_pages, 0)

    def test_retention_only_keeps_more_precision_than_the_age_gives(self):
        plan = self.plan()
        # Page 0 of a four-page prefix is cold q3; keeping it f32 or q4 is legal.
        self.assertEqual(plan.desired_pages(4, {0: "f32"})[0].codec, "f32")
        self.assertEqual(plan.desired_pages(4, {0: "q4"})[0].codec, "q4")
        self.assertEqual(plan.desired_pages(4, {0: "q3"})[0].codec, "q3")
        # The newest page is hot: nothing is more precise than F32.
        with self.assertRaises(ValueError):
            plan.desired_pages(4, {3: "q4"})
        with self.assertRaises(ValueError):
            plan.desired_pages(4, {0: "q2"})
        with self.assertRaises(ValueError):
            plan.desired_pages(4, {9: "f32"})

    def test_the_admitted_budget_bounds_how_many_pages_may_be_kept(self):
        plan = self.plan(retain_pages=1)
        plan.desired_pages(4, {0: "f32"})
        with self.assertRaises(ValueError):
            plan.desired_pages(4, {0: "f32", 1: "f32"})

    def test_the_reservation_covers_what_the_retentions_may_cost(self):
        plain = self.plan(quality_max_rmse=None, retain_pages=0)
        keeping = self.plan(retain_pages=2)
        spread = (plain.layout("f32").page_allocation_bytes
                  - plain.layout("q3").page_allocation_bytes)
        self.assertEqual(keeping.quality_retention_bytes, 2 * max(spread, 0))
        self.assertEqual(keeping.allocation_limit_bytes(4) - plain.allocation_limit_bytes(4),
                         keeping.quality_retention_bytes)

    def test_a_transition_rejects_committed_pages_that_ignore_the_retention(self):
        plan = self.plan()
        committed = plan.desired_pages(4, {0: "f32"})
        plan.plan_transition(4, 1, "decode", committed, {0: "f32"})
        with self.assertRaises(ValueError):
            plan.plan_transition(4, 1, "decode", committed)
        with self.assertRaises(ValueError):
            plan.plan_transition(4, 1, "decode", plan.desired_pages(4), {0: "f32"})

    def test_a_retained_page_never_migrates_again(self):
        plan = self.plan()
        transition = plan.plan_transition(4, 1, "decode", plan.desired_pages(4, {0: "f32"}), {0: "f32"})
        self.assertEqual([page.codec for page in transition.final_pages],
                         ["f32", "q3", "q3", "q4", "f32"])
        self.assertNotIn(0, [m.source.page_index for m in transition.migrations])
        self.assertEqual(transition.final_retained_pages, ((0, "f32"),))

    def test_a_replaced_prefix_drops_every_retention(self):
        plan = self.plan()
        transition = plan.plan_transition(4, 4, "prefill", plan.desired_pages(4, {0: "f32"}), {0: "f32"})
        self.assertEqual(transition.final_retained_pages, ())
        self.assertEqual([page.codec for page in transition.final_pages], ["q3", "q3", "q4", "f32"])

    def test_the_transition_json_carries_the_retention(self):
        plan = self.plan()
        transition = plan.plan_transition(4, 1, "decode", plan.desired_pages(4, {0: "f32"}), {0: "f32"})
        restored = TieredKVTransition.from_json(transition.to_json())
        self.assertEqual(restored.retained_pages, ((0, "f32"),))
        self.assertEqual(restored.final_pages, transition.final_pages)
        data = json.loads(transition.to_json())
        data["retained_pages"] = []
        with self.assertRaises(ValueError):
            TieredKVTransition.from_json(json.dumps(data))

    def test_normalizing_retentions_is_canonical_and_rejects_nonsense(self):
        self.assertEqual(normalize_retained({2: "q4", 0: "f32"}), ((0, "f32"), (2, "q4")))
        self.assertEqual(normalize_retained([[1, "q3"]]), ((1, "q3"),))
        self.assertEqual(normalize_retained(None), ())
        for bad in ([[0, "f32"], [0, "q4"]], [[0]], [[-1, "f32"]], [[0, "tq"]], [[0, 4]]):
            with self.assertRaises(ValueError):
                normalize_retained(bad)


class TieredQualityRuntimeRegressions(_TierFixture):
    def quality_session(self, **options):
        session = self.session(page_tokens=options.pop("page_tokens", 1), **options)
        self.addCleanup(session.close)
        return session

    @staticmethod
    def migration(session):
        return session.report()["kv_migration"]

    def test_without_a_ceiling_every_page_ages_as_before(self):
        session = self.quality_session()
        session.prefill([1, 3])
        session.decode(5)
        report = self.migration(session)
        self.assertIsNone(report["quality_max_rmse"])
        self.assertEqual(report["pages_retained"], 0)
        self.assertEqual(self.codecs(session), ["q3", "q4", "f32"])

    def test_a_loose_ceiling_changes_nothing_measurable(self):
        plain = self.quality_session()
        plain.prefill([1, 3])
        plain.decode(5)
        gated = self.quality_session(kv_quality_max_rmse=LOOSE, kv_retain_pages=2)
        gated.prefill([1, 3])
        gated.decode(5)
        self.assertEqual(self.codecs(gated), self.codecs(plain))
        self.assertEqual(self.records(gated), self.records(plain))
        self.assertEqual(self.migration(gated)["pages_retained"], 0)
        self.assertEqual(self.migration(gated)["rmse"], self.migration(plain)["rmse"])

    def test_a_zero_ceiling_keeps_every_page_the_budget_allows(self):
        session = self.quality_session(kv_quality_max_rmse=TIGHT, kv_retain_pages=2)
        session.prefill([1, 3])
        # The prefill already ages one page, and a zero ceiling rejects it.
        self.assertEqual(self.migration(session)["pages_retained"], 1)
        self.assertEqual(session.report()["kv_retained_pages"], [[0, "f32"]])
        session.decode(5)
        report = self.migration(session)
        self.assertEqual(report["retain_pages_available"], 1)
        self.assertEqual(report["pages_retained"], 1)
        self.assertEqual(report["pages_reencoded"], 0)
        self.assertEqual(report["retentions_declined"], 0)
        self.assertEqual(self.codecs(session), ["f32", "f32", "f32"])
        self.assertEqual(session.report()["kv_retained_pages"], [[0, "f32"], [1, "f32"]])

    def test_a_middle_ceiling_stops_a_page_at_the_warm_tier(self):
        session = self.quality_session(kv_quality_max_rmse=MIDDLE, kv_retain_pages=2)
        session.prefill([1, 3])
        # F32 -> Q4 stays under the ceiling, so the page does age once.
        self.assertEqual(self.migration(session)["pages_retained"], 0)
        self.assertEqual(self.codecs(session), ["q4", "f32"])
        session.decode(5)
        report = self.migration(session)
        errors = dict(report["page_rmse"])
        self.assertGreater(errors[0], MIDDLE)  # Q4 -> Q3 does not.
        self.assertLess(errors[1], MIDDLE)
        self.assertEqual(report["pages_retained"], 1)
        self.assertEqual(report["pages_reencoded"], 1)
        self.assertEqual(self.codecs(session), ["q4", "q4", "f32"])
        self.assertEqual(session.report()["kv_retained_pages"], [[0, "q4"]])

    def test_an_exhausted_budget_ages_the_page_anyway_and_says_so(self):
        session = self.quality_session(kv_quality_max_rmse=TIGHT, kv_retain_pages=1)
        session.prefill([1, 3])
        self.assertEqual(self.migration(session)["pages_retained"], 1)
        session.decode(5)
        report = self.migration(session)
        # The slot is already spent: the budget is a hard bound on residence,
        # so the next page ages despite failing the ceiling, and the report says so.
        self.assertEqual(report["retain_pages_available"], 0)
        self.assertEqual(report["pages_retained"], 0)
        self.assertEqual(report["retentions_declined"], 1)
        self.assertEqual(self.codecs(session), ["f32", "q4", "f32"])

    def test_a_retained_page_stays_bit_identical_to_its_source(self):
        session = self.quality_session(kv_quality_max_rmse=TIGHT, kv_retain_pages=2)
        session.prefill([1, 3])
        before = self.records(session)
        session.decode(5)
        after = self.records(session, length=2)
        # Retention adopts the existing page: not a re-encode that happens to
        # round-trip, the very same bytes the prefill wrote.
        self.assertEqual(after, before)

    def test_a_retained_page_is_never_migrated_again_by_a_later_decode(self):
        session = self.quality_session(kv_quality_max_rmse=TIGHT, kv_retain_pages=1)
        session.prefill([1, 3])
        self.assertEqual(session.report()["kv_retained_pages"], [[0, "f32"]])
        kept = self.records(session, length=1)
        for token, codecs in ((5, ["f32", "q4", "f32"]), (7, ["f32", "q3", "q4", "f32"])):
            session.decode(token)
            # Retention is sticky: the bytes never change, so re-measuring the
            # same migration would only spend the same work for the same verdict.
            self.assertNotIn(0, [index for index, _ in self.migration(session)["page_rmse"]])
            self.assertEqual(self.records(session, length=1), kept)
            self.assertEqual(self.codecs(session), codecs)

    def test_the_budget_is_reserved_before_anything_is_allocated(self):
        plain = self.quality_session()
        gated = self.quality_session(kv_quality_max_rmse=TIGHT, kv_retain_pages=2)
        # Retention raises the admitted reservation by exactly the spread the
        # plan promises; on this fixture a Q3 page is not cheaper than F32, so
        # the spread is zero and admission is unchanged. Either way it is paid
        # up front, never discovered while a transaction is open.
        self.assertEqual(gated.report()["memory"]["kv_reserved_capacity_bytes"],
                         plain.report()["memory"]["kv_reserved_capacity_bytes"]
                         + gated._tier_plan.quality_retention_bytes)

    def test_resetting_and_replacing_a_prefix_returns_the_whole_budget(self):
        session = self.quality_session(kv_quality_max_rmse=TIGHT, kv_retain_pages=2)
        session.prefill([1, 3])
        session.decode(5)
        self.assertEqual(self.migration(session)["retain_pages_available"], 1)
        # A replaced prefix shares no page with the old one, so no retention
        # survives it and the budget is whole again for the new pages.
        session.prefill([2, 4, 6])
        self.assertEqual(self.migration(session)["retain_pages_available"], 2)
        self.assertEqual(session.report()["kv_retained_pages"], [[0, "f32"], [1, "f32"]])
        session.reset()
        self.assertEqual(session.report()["kv_retained_pages"], [])
        session.prefill([1, 3])
        self.assertEqual(self.migration(session)["retain_pages_available"], 2)

    def test_a_derived_sequence_inherits_the_retentions(self):
        parent = self.quality_session(kv_quality_max_rmse=TIGHT, kv_retain_pages=2)
        parent.prefill([1, 3])
        parent.decode(5)
        child = parent.fork()
        self.addCleanup(child.close)
        self.assertEqual(child.policy, parent.policy)
        self.assertEqual(child.report()["kv_pages"], parent.report()["kv_pages"])
        child.decode(7)
        # The parent spent the whole budget, and the child inherits the pages
        # *and* the bill: it may not keep a third page above its age.
        self.assertEqual(self.migration(child)["retain_pages_available"], 0)
        self.assertEqual(self.migration(child)["retentions_declined"], 1)
        self.assertEqual(self.codecs(child), ["f32", "f32", "q4", "f32"])
        self.assertEqual(self.codecs(parent), ["f32", "f32", "f32"])

    def test_a_failed_migration_leaves_no_retention_behind(self):
        from unittest.mock import patch
        session = self.quality_session(kv_quality_max_rmse=TIGHT, kv_retain_pages=2)
        session.prefill([1, 3])
        tokens, codecs = session.token_ids, self.codecs(session)
        with patch.object(type(session), "_allocate_page", side_effect=MemoryError("injected")):
            with self.assertRaises(MemoryError):
                session.decode(5)
        self.assertEqual(session.token_ids, tokens)
        self.assertEqual(self.codecs(session), codecs)
        self.assertEqual(session.report()["kv_retained_pages"], [[0, "f32"]])
        session.decode(5)
        self.assertEqual(session.report()["kv_retained_pages"], [[0, "f32"], [1, "f32"]])
        self.assertEqual(self.codecs(session), ["f32", "f32", "f32"])

    def test_cli_reports_the_retained_pages(self):
        command = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path),
                   "--tokens", "1,3", "--decode-tokens", "5", "--kv-policy", "age",
                   "--kv-page-tokens", "1", "--kv-group-size", "3",
                   "--kv-quality-max-rmse", str(MIDDLE), "--kv-retain-pages", "2",
                   "--max-sequence-length", "8", "--tile-rows", "3", "--memory-budget", "1MiB"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["kv_retained_pages"], [[0, "q4"]])
        self.assertEqual(report["kv_migration"]["quality_max_rmse"], MIDDLE)
        self.assertEqual([page["codec"] for page in report["kv_pages"]], ["q4", "q4", "f32"])

    def test_a_backing_store_rejects_the_ceiling_instead_of_ignoring_it(self):
        from runtime.nexapack.offloaded import OffloadedTieredTransformerSession
        with self.assertRaises(ValueError):
            OffloadedTieredTransformerSession(self.path, kv_backing_store=self.directory / "gated",
                                              kv_quality_max_rmse=TIGHT, kv_retain_pages=1)

    def test_cli_rejects_an_unbounded_or_misplaced_quality_ceiling(self):
        base = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path),
                "--tokens", "1,3", "--max-sequence-length", "8", "--tile-rows", "3",
                "--memory-budget", "1MiB"]
        for extra in (["--kv-policy", "age", "--kv-quality-max-rmse", "0.1"],
                      ["--kv-policy", "age", "--kv-retain-pages", "2"],
                      ["--kv-quality-max-rmse", "0.1", "--kv-retain-pages", "2"],
                      ["--kv-policy", "age", "--kv-quality-max-rmse", "0.1", "--kv-retain-pages", "2",
                       "--kv-backing-store", str(self.directory / "rejected")]):
            result = subprocess.run(base + extra, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 2, extra)


if __name__ == "__main__":
    unittest.main()

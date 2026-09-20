"""Prefix reuse between sessions with no kinship: pages, bytes and refusals."""
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.nexapack.prefix_reuse import common_page_prefix, common_prefix_length
from test_paged_transformer_regressions import _PagedFixture
from test_transformer_forward_regressions import random_bundle

CODECS = (("f32", {}), ("q4", {"kv_group_size": 4}), ("q3", {"kv_group_size": 4}),
          ("tq", {"kv_bits": 3, "kv_seed": 42}))
# PROMPT_B diverges *inside* the third page and PROMPT_C exactly on its
# boundary. Both share two complete pages at two tokens per page: the fifth
# token PROMPT_B also agrees on belongs to a page its owner is still writing,
# so no adoption may carry it. Keeping both cases is what makes the oracle
# able to fail when a partial page is adopted.
PROMPT_A = [1, 3, 5, 7, 2, 4]
PROMPT_B = [1, 3, 5, 7, 2, 6]
PROMPT_C = [1, 3, 5, 7, 8, 6]
SHARED_PAGES = 2


class PagePrefixArithmeticRegressions(unittest.TestCase):
    """The rule alone, with no session, allocation or file involved."""

    def test_the_shared_page_count_is_the_floor_of_the_common_token_prefix(self):
        cases = (([1, 2, 3, 4], [1, 2, 3, 4], 2, 2), ([1, 2, 3, 4], [1, 2, 9, 4], 2, 1),
                 ([1, 2, 3], [1, 2, 3, 4, 5], 2, 1), ([1, 2, 3], [1, 2, 3], 1, 3),
                 ([1, 2], [9, 2], 1, 0), ([], [1], 1, 0), ([1, 2, 3], [1, 2, 3], 4, 0),
                 ([1, 2, 3, 4, 5], [1, 2, 3, 4, 9], 2, 2))
        for left, right, page_tokens, expected in cases:
            with self.subTest(left=left, right=right, page_tokens=page_tokens):
                self.assertEqual(common_page_prefix(left, right, page_tokens), expected)
                # A partial page is never counted, in either direction.
                self.assertEqual(common_page_prefix(right, left, page_tokens), expected)
                self.assertGreaterEqual(common_prefix_length(left, right), expected * page_tokens)

    def test_the_rule_rejects_anything_that_is_not_a_page_of_token_ids(self):
        for left, right, page_tokens in (([1], [1], 0), ([1], [1], -2), ([1], [1], 1.0),
                                         ([1], [1], True), ([1], [1], "2"), ("12", [1], 1),
                                         ([1], {1: 2}, 1), ([1.0], [1.0], 1), ([True], [True], 1)):
            with self.subTest(left=left, right=right, page_tokens=page_tokens):
                with self.assertRaises(ValueError):
                    common_page_prefix(left, right, page_tokens)


class PrefixReuseRegressions(_PagedFixture):
    def sequence(self, codec="f32", options=None, **kwargs):
        return self.session(kv_codec=codec, **{**(options or {}), **kwargs})

    @staticmethod
    def page_bytes(session, page):
        return ctypes.string_at(page.address, session._cache_plan.page_extent_bytes)

    def test_adopted_pages_are_byte_identical_to_a_prefix_computed_alone(self):
        """The central oracle: a page's bytes depend only on absolute position.

        The control never sees either prompt's tail, and the adopter never
        derived from the source. If any page byte depended on the sequence's
        history, its chunking or its owner, the comparison would fail.
        """
        for codec, options in CODECS:
            for prompt in (PROMPT_B, PROMPT_C):
                with self.subTest(codec=codec, prompt=prompt):
                    source = self.sequence(codec, options)
                    self.addCleanup(source.close)
                    source.prefill(PROMPT_A)
                    reuse = self.sequence(codec, options)  # Built alone, never forked.
                    self.addCleanup(reuse.close)
                    reuse.adopt_prefix(source, prompt)
                    control = self.sequence(codec, options)
                    self.addCleanup(control.close)
                    control.prefill(PROMPT_A[:SHARED_PAGES * source.page_tokens])
                    self.assertEqual(len(reuse._pages), SHARED_PAGES)
                    self.assertEqual(len(control._pages), SHARED_PAGES)
                    for index, (adopted, computed) in enumerate(zip(reuse._pages, control._pages)):
                        with self.subTest(page=index):
                            self.assertEqual(self.page_bytes(reuse, adopted),
                                             self.page_bytes(control, computed))

    def test_reuse_continues_with_the_logits_of_an_independent_session(self):
        for codec, options in CODECS:
            with self.subTest(codec=codec):
                source = self.sequence(codec, options)
                self.addCleanup(source.close)
                source.prefill(PROMPT_A)
                reuse = self.sequence(codec, options)
                self.addCleanup(reuse.close)
                adoption = reuse.adopt_prefix(source, PROMPT_B)
                self.assertEqual(adoption["inherited_tokens"], SHARED_PAGES * source.page_tokens)
                self.assertEqual(reuse.token_ids, tuple(PROMPT_B[:adoption["inherited_tokens"]]))
                produced = reuse.append(PROMPT_B[adoption["inherited_tokens"]:])
                with self.sequence(codec, options) as control:
                    control.prefill(PROMPT_B[:adoption["inherited_tokens"]])
                    expected = control.append(PROMPT_B[adoption["inherited_tokens"]:])
                self.assertEqual(produced, expected)
                self.assertEqual(reuse.token_ids, tuple(PROMPT_B))
                # The source keeps its own prefix and can still continue it.
                self.assertEqual(source.token_ids, tuple(PROMPT_A))
                source.decode(2)
                self.assertEqual(source.token_ids, tuple(PROMPT_A) + (2,))

    def test_the_source_is_untouched_and_its_shared_pages_stay_immutable(self):
        source = self.sequence()
        self.addCleanup(source.close)
        source.prefill(PROMPT_A)
        before = [self.page_bytes(source, page) for page in source._pages[:SHARED_PAGES]]
        reuse = self.sequence()
        self.addCleanup(reuse.close)
        reuse.adopt_prefix(source, PROMPT_B)
        reuse.append(PROMPT_B[SHARED_PAGES * source.page_tokens:])
        source.append([6, 8])
        self.assertEqual([self.page_bytes(source, page) for page in source._pages[:SHARED_PAGES]], before)
        self.assertEqual(source.token_ids, tuple(PROMPT_A) + (6, 8))
        self.assertEqual(reuse.token_ids, tuple(PROMPT_B))
        self.assertTrue(all(page.shared for page in reuse._pages[:SHARED_PAGES]))

    def test_reports_publish_copied_bytes_and_the_shared_residency(self):
        source = self.sequence("q4", {"kv_group_size": 4})
        self.addCleanup(source.close)
        source.prefill(PROMPT_A)
        reuse = self.sequence("q4", {"kv_group_size": 4})
        self.addCleanup(reuse.close)
        adoption = reuse.adopt_prefix(source, PROMPT_B)
        # A whole-page boundary carries no partial page, so nothing is copied.
        # The fifth token both prompts agree on is dropped: it lives in a page
        # the source is still writing, and only complete pages are immutable.
        self.assertEqual(adoption, {"inherited_tokens": 4, "shared_pages": SHARED_PAGES,
                                    "copied_pages": 0, "copied_bytes": 0,
                                    "source": "shared_page_prefix", "common_tokens": 5,
                                    "requested_tokens": len(PROMPT_B), "source_tokens": len(PROMPT_A)})
        reuse.append(PROMPT_B[4:])
        memory = reuse.report()["memory"]
        allocation = memory["kv_page_allocation_bytes"]
        self.assertEqual(memory["kv_shared_page_count"], SHARED_PAGES)
        self.assertEqual(memory["kv_shared_allocation_bytes"], SHARED_PAGES * allocation)
        self.assertEqual(memory["kv_owned_allocation_bytes"], allocation)
        self.assertEqual(memory["kv_shared_allocation_bytes"] + memory["kv_owned_allocation_bytes"],
                         memory["kv_resident_allocation_bytes"])
        # Measured, not asserted in the abstract: the process holds the shared
        # pages once, so two unrelated sequences cost less than two copies.
        duplicated = 2 * source.report()["memory"]["kv_resident_allocation_bytes"]
        shared = (source.report()["memory"]["kv_resident_allocation_bytes"]
                  + memory["kv_owned_allocation_bytes"])
        self.assertEqual((duplicated, shared), (6 * allocation, 4 * allocation))
        # Admission stays conservative: sharing lowers residency, not the bound.
        self.assertEqual(memory["kv_reserved_capacity_bytes"],
                         source.report()["memory"]["kv_reserved_capacity_bytes"])

    def test_reuse_refuses_a_different_layout_bundle_or_prefix(self):
        other = self.directory / "other-bundle"
        random_bundle(other, layers=2, heads=4, kv_heads=2, tied=False, seed=777)
        source = self.sequence("q4", {"kv_group_size": 4})
        self.addCleanup(source.close)
        source.prefill(PROMPT_A)
        cases = {"page_tokens": {"page_tokens": 4}, "codec": {"kv_codec": "q3"},
                 "group_size": {"kv_group_size": 2}}
        for name, overrides in cases.items():
            with self.subTest(case=name):
                options = {"kv_codec": "q4", "kv_group_size": 4, **overrides}
                with self.session(**options) as stranger, self.assertRaises(ValueError) as failure:
                    stranger.adopt_prefix(source, PROMPT_B)
                self.assertIn("same bundle and KV page layout", str(failure.exception))
        with self.sequence("q4", {"kv_group_size": 4}, path=other) as unrelated:
            with self.assertRaises(ValueError) as failure:
                unrelated.adopt_prefix(source, PROMPT_B)
            self.assertIn("same bundle and KV page layout", str(failure.exception))
        with self.sequence("q4", {"kv_group_size": 4}) as short:
            # Three common tokens are one token short of a second page, and the
            # first page is not shared at all when the prompts differ at zero.
            with self.assertRaises(ValueError) as failure:
                short.adopt_prefix(source, [1, 9, 9, 9])
            self.assertIn("less than one complete KV page", str(failure.exception))
        with self.sequence("q4", {"kv_group_size": 4}) as used:
            used.prefill([2, 4])
            with self.assertRaises(ValueError) as failure:
                used.adopt_prefix(source, PROMPT_B)
            self.assertIn("without its own prefix", str(failure.exception))
        closed_source = self.sequence("q4", {"kv_group_size": 4})
        closed_source.prefill(PROMPT_A)
        closed_source.close()
        with self.sequence("q4", {"kv_group_size": 4}) as adopter:
            with self.assertRaises(ValueError) as failure:
                adopter.adopt_prefix(closed_source, PROMPT_B)
            self.assertIn("closed", str(failure.exception))
        # Every refusal leaves the source able to continue.
        self.assertEqual(source.token_ids, tuple(PROMPT_A))
        source.decode(2)

    def test_reuse_refuses_a_prefix_longer_than_the_adopter_capacity(self):
        source = self.session(page_tokens=2, max_sequence_length=8)
        self.addCleanup(source.close)
        source.prefill([1, 3, 5, 7, 2, 4])
        with self.session(page_tokens=2, max_sequence_length=4) as small:
            with self.assertRaises(ValueError) as failure:
                small.adopt_prefix(source, [1, 3, 5, 7, 2, 4])
            self.assertIn("capacity", str(failure.exception))
        # Honest limit: the adopted prefix is a prefix of the adopter's own
        # prompt, so the prompt hits the capacity first and the inherited-
        # prefix refusal below it is only reachable through fork.
        self.assertFalse(any(page.shared for page in source._pages))

    def test_age_tiers_refuse_a_truncated_prefix_and_say_why(self):
        from runtime.nexapack.tiered import TieredTransformerSession

        def tiers(**options):
            return TieredTransformerSession(self.path, memory_budget="1MiB", max_sequence_length=8,
                                            page_tokens=2, kv_group_size=4, **options)

        with tiers(hot_pages=1, warm_pages=1) as source:
            source.prefill(PROMPT_A)
            self.assertEqual([page.codec for page in source._page_descriptors], ["q3", "q4", "f32"])
            with tiers(hot_pages=1, warm_pages=1) as reuse:
                with self.assertRaises(ValueError) as failure:
                    reuse.adopt_prefix(source, PROMPT_B)
                message = str(failure.exception)
                # The correct reason: a shorter prefix is a *younger* one.
                self.assertIn("makes the inherited ones younger", message)
                self.assertIn("promoted back", message)
                self.assertIn("[0, 'q3', 'q4']", message)
            self.assertEqual(source.token_ids, tuple(PROMPT_A))
            self.assertFalse(any(page.shared for page in source._pages))
        # The second branch: pages kept *above* their new age. Only the quality
        # retention budget could express it, and adoption admits none.
        gate = {"hot_pages": 1, "warm_pages": 1, "kv_quality_max_rmse": 0.0, "kv_retain_pages": 2}
        with tiers(**gate) as source:
            source.prefill(PROMPT_A)
            self.assertEqual([page.codec for page in source._page_descriptors], ["f32", "f32", "f32"])
            with tiers(**gate) as reuse:
                with self.assertRaises(ValueError) as failure:
                    reuse.adopt_prefix(source, PROMPT_B)
                self.assertIn("kept above their new age", str(failure.exception))

    def test_age_tiers_admit_a_truncated_prefix_that_never_aged(self):
        """The refusal is exactly as wide as the promotion it cannot perform."""
        from runtime.nexapack.tiered import TieredTransformerSession

        def tiers():
            return TieredTransformerSession(self.path, memory_budget="1MiB", max_sequence_length=8,
                                            page_tokens=2, kv_group_size=4, hot_pages=4, warm_pages=0)

        with tiers() as source, tiers() as reuse, tiers() as control:
            source.prefill(PROMPT_A)
            self.assertEqual([page.codec for page in source._page_descriptors], ["f32"] * 3)
            reuse.adopt_prefix(source, PROMPT_B)
            # Age ranks are recomputed for the shorter prefix, not inherited.
            self.assertEqual([page.age_rank for page in reuse._page_descriptors], [1, 0])
            produced = reuse.append(PROMPT_B[4:])
            control.prefill(PROMPT_B[:4])
            self.assertEqual(produced, control.append(PROMPT_B[4:]))

    def test_a_backing_store_refuses_a_truncated_prefix(self):
        from runtime.nexapack.offloaded import OffloadedTieredTransformerSession

        def offloaded(name):
            return OffloadedTieredTransformerSession(
                self.path, kv_backing_store=self.directory / name, memory_budget="1MiB",
                max_sequence_length=8, page_tokens=2, kv_group_size=4, hot_pages=4, warm_pages=0)

        with offloaded("source-store") as source, offloaded("reuse-store") as reuse:
            source.prefill(PROMPT_A)
            with self.assertRaises(ValueError) as failure:
                reuse.adopt_prefix(source, PROMPT_B)
            self.assertIn("whole prefixes only", str(failure.exception))
            # fork, which takes the entire prefix, still works on this store.
            child = source.fork()
            self.addCleanup(child.close)
            self.assertEqual(child.token_ids, tuple(PROMPT_A))

    def test_cli_reuses_a_prefix_between_two_independent_prompts(self):
        command = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path),
                   "--tokens", "1,3,5,7", "--kv-cache", "--kv-page-tokens", "2", "--kv-codec", "q4",
                   "--kv-group-size", "4", "--max-sequence-length", "8", "--tile-rows", "3",
                   "--memory-budget", "1MiB", "--reuse-prefix-tokens", "1,3,5,2,4"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        reused = report["reused_prefix"]
        self.assertEqual(report["token_ids"], [1, 3, 5, 7])  # The first prompt is untouched.
        self.assertEqual(reused["token_ids"], [1, 3, 5, 2, 4])
        self.assertEqual(reused["prefix_adoption"],
                         {"inherited_tokens": 2, "shared_pages": 1, "copied_pages": 0,
                          "copied_bytes": 0, "source": "shared_page_prefix", "common_tokens": 3,
                          "requested_tokens": 5, "source_tokens": 4})
        self.assertEqual(reused["appended_token_ids"], [5, 2, 4])
        self.assertEqual(reused["kv_shared_allocation_bytes"] + reused["kv_owned_allocation_bytes"],
                         reused["kv_resident_allocation_bytes"])
        with self.sequence("q4", {"kv_group_size": 4}) as control:
            control.prefill([1, 3])
            appended = control.append([5, 2, 4])
        digest = hashlib.sha256()
        for row in appended:
            for value in row:
                digest.update(struct.pack("<f", value))
        self.assertEqual(reused["logits_sha256"], digest.hexdigest())
        # argparse prints the whole usage on any error, so assert the specific
        # message: a weaker check would pass for an unrelated rejection.
        rejected = subprocess.run(
            [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path), "--tokens", "1,3,5,7",
             "--max-sequence-length", "8", "--tile-rows", "3", "--memory-budget", "1MiB",
             "--recompute", "--reuse-prefix-tokens", "1,3,5,2,4"],
            capture_output=True, text=True, timeout=120)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("--recompute keeps no cache", rejected.stderr)
        self.assertNotIn("Traceback", rejected.stderr)


if __name__ == "__main__":
    unittest.main()

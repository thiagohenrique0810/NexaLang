"""Derived sequences sharing a paged prefix: equivalence, ownership and limits."""
import ctypes
import hashlib
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_paged_transformer_regressions import _PagedFixture
from test_transformer_forward_regressions import random_bundle

CODECS = (("f32", {}), ("q4", {"kv_group_size": 4}), ("q3", {"kv_group_size": 4}),
          ("tq", {"kv_bits": 3, "kv_seed": 42}))


class PagedSequenceRegressions(_PagedFixture):
    def sequence(self, codec="f32", options=None, **kwargs):
        return self.session(kv_codec=codec, **{**(options or {}), **kwargs})

    @staticmethod
    def page_bytes(session, page):
        return ctypes.string_at(page.address, session._cache_plan.page_extent_bytes)

    def test_fork_continues_the_prefix_with_identical_logits_for_every_codec(self):
        for codec, options in CODECS:
            with self.subTest(codec=codec), self.sequence(codec, options) as parent:
                parent.prefill([1, 3, 5])
                partial = self.page_bytes(parent, parent._pages[-1])
                child = parent.fork()
                self.addCleanup(child.close)
                self.assertEqual(child.token_ids, parent.token_ids)
                # Complete pages are the very same allocation; the partial page
                # is an exclusive copy with identical bytes.
                self.assertIs(child._pages[0], parent._pages[0])
                self.assertTrue(child._pages[0].shared)
                self.assertIsNot(child._pages[-1], parent._pages[-1])
                self.assertFalse(child._pages[-1].shared)
                self.assertEqual(self.page_bytes(child, child._pages[-1]), partial)
                adoption = child.report()["kv_prefix_adoption"]
                self.assertEqual(adoption["inherited_tokens"], 3)
                self.assertEqual((adoption["shared_pages"], adoption["copied_pages"]), (1, 1))
                self.assertGreater(adoption["copied_bytes"], 0)
                with self.sequence(codec, options) as control:
                    control.prefill([1, 3, 5])
                    expected = control.decode(7)
                self.assertEqual(child.decode(7), expected)
                self.assertEqual(parent.decode(7), expected)

    def test_sequences_diverge_without_touching_each_other_bytes(self):
        with self.sequence(page_tokens=2) as parent:
            parent.prefill([1, 3, 5, 7])
            child = parent.fork()
            self.addCleanup(child.close)
            shared = [page for page in child._pages if page.shared]
            self.assertEqual(len(shared), 2)  # Both pages are complete at four tokens.
            before = [self.page_bytes(parent, page) for page in shared]
            child_logits = child.append([2, 4])
            parent_logits = parent.append([6, 8])
            self.assertEqual([self.page_bytes(parent, page) for page in shared], before)
            self.assertEqual(parent.token_ids, (1, 3, 5, 7, 6, 8))
            self.assertEqual(child.token_ids, (1, 3, 5, 7, 2, 4))
            for session, tokens, produced in ((parent, [1, 3, 5, 7, 6, 8], parent_logits),
                                              (child, [1, 3, 5, 7, 2, 4], child_logits)):
                with self.sequence(page_tokens=2) as control:
                    control.prefill(tokens[:4])
                    self.assertEqual(control.append(tokens[4:]), produced)

    def test_shared_pages_survive_parent_reset_prefill_and_close_in_any_order(self):
        for action in ("reset", "prefill", "close"):
            with self.subTest(action=action):
                parent = self.sequence(page_tokens=2)
                parent.prefill([1, 3, 5, 7])
                child = parent.fork()
                try:
                    shared = child._pages[0]
                    expected = self.page_bytes(child, shared)
                    if action == "reset":
                        parent.reset()
                    elif action == "prefill":
                        parent.prefill([2, 6])
                    else:
                        parent.close()
                    # The parent dropped its reference; the page stays alive
                    # and unchanged for the sequence that still reads it.
                    self.assertTrue(shared.address)
                    self.assertFalse(shared.shared)
                    self.assertEqual(self.page_bytes(child, shared), expected)
                    child.decode(4)
                    self.assertEqual(child.token_ids, (1, 3, 5, 7, 4))
                finally:
                    child.close()
                    if action != "close":
                        parent.close()
                self.assertFalse(shared.address)

    def test_a_write_can_never_reach_a_shared_page(self):
        with self.sequence(page_tokens=2) as parent:
            parent.prefill([1, 3, 5])
            child = parent.fork()
            self.addCleanup(child.close)
            tokens, report = child.token_ids, child.report()
            # Force the rejected state adoption is designed to avoid.
            child._pages[-1].retain()
            with self.assertRaises(ValueError) as failure:
                child.decode(7)
            self.assertIn("shared prefix page", str(failure.exception))
            self.assertEqual((child.token_ids, child.report()), (tokens, report))
            child._pages[-1].release()
            self.assertEqual(child.decode(7), parent.decode(7))

    def test_fork_requires_a_committed_prefix_and_an_identical_layout(self):
        other = self.directory / "other-bundle"
        random_bundle(other, layers=2, heads=4, kv_heads=2, tied=False, seed=777)
        with self.sequence(page_tokens=2) as parent:
            with self.assertRaises(ValueError):
                parent.fork()
            parent.prefill([1, 3, 5])
            for overrides in ({"page_tokens": 4}, {"kv_codec": "q4", "kv_group_size": 4},
                              {"max_sequence_length": 2}):
                with self.subTest(overrides=sorted(overrides)), self.assertRaises(ValueError):
                    parent.fork(**overrides)
            with self.sequence(page_tokens=2) as stranger:
                stranger.prefill([2, 4])
                with self.assertRaises(ValueError):
                    stranger._adopt_prefix(parent)  # Already owns a prefix.
                with self.sequence(page_tokens=2, path=other) as unrelated:
                    with self.assertRaises(ValueError):
                        unrelated._adopt_prefix(parent)
            # Every rejection leaves the parent able to continue.
            self.assertEqual(parent.token_ids, (1, 3, 5))
            parent.decode(7)

    def test_capacity_limits_apply_to_each_derived_sequence(self):
        with self.sequence(page_tokens=2, max_sequence_length=6) as parent:
            parent.prefill([1, 3, 5, 7])
            child = parent.fork()
            self.addCleanup(child.close)
            child.append([2, 4])
            self.assertEqual(len(child.token_ids), 6)
            with self.assertRaises(ValueError):
                child.decode(6)
            self.assertEqual(len(child.token_ids), 6)
            self.assertEqual(len(parent.token_ids), 4)
            parent.append([8, 2])
            self.assertEqual(len(parent.token_ids), 6)

    def test_failed_adoption_releases_copies_and_preserves_the_parent(self):
        with self.sequence(page_tokens=2) as parent:
            parent.prefill([1, 3, 5])
            tokens, shared = parent.token_ids, parent._pages[0]
            with patch("ctypes.memmove", side_effect=OSError("injected copy failure")):
                with self.assertRaises(OSError):
                    parent.fork()
            self.assertFalse(shared.shared)  # The retained reference was returned.
            self.assertEqual(parent.token_ids, tokens)
            child = parent.fork()
            self.addCleanup(child.close)
            self.assertEqual(child.token_ids, tokens)

    def test_reports_separate_shared_from_owned_residency(self):
        with self.sequence(page_tokens=2) as parent:
            parent.prefill([1, 3, 5, 7])
            child = parent.fork()
            self.addCleanup(child.close)
            memory = child.report()["memory"]
            allocation = memory["kv_page_allocation_bytes"]
            self.assertEqual(memory["kv_shared_page_count"], 2)
            self.assertEqual(memory["kv_shared_allocation_bytes"], 2 * allocation)
            self.assertEqual(memory["kv_owned_allocation_bytes"], 0)
            self.assertEqual(memory["kv_shared_allocation_bytes"] + memory["kv_owned_allocation_bytes"],
                             memory["kv_resident_allocation_bytes"])
            child.append([2, 4])
            memory = child.report()["memory"]
            self.assertEqual(memory["kv_shared_page_count"], 2)
            self.assertEqual(memory["kv_owned_allocation_bytes"], allocation)
            # Admission stays conservative: sharing lowers residency, not the
            # reservation each sequence must still be granted.
            self.assertEqual(memory["kv_reserved_capacity_bytes"],
                             parent.report()["memory"]["kv_reserved_capacity_bytes"])

    def test_a_fork_of_a_fork_shares_the_same_pages(self):
        with self.sequence(page_tokens=2) as parent:
            parent.prefill([1, 3, 5, 7])
            child = parent.fork()
            self.addCleanup(child.close)
            grandchild = child.fork()
            self.addCleanup(grandchild.close)
            self.assertIs(grandchild._pages[0], parent._pages[0])
            self.assertEqual(grandchild.token_ids, parent.token_ids)
            self.assertEqual(grandchild.decode(2), child.decode(2))

    def test_cli_derives_a_sequence_and_reports_shared_residency(self):
        import json
        import subprocess
        command = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path), "--tokens", "1,3,5,7",
                   "--kv-cache", "--kv-page-tokens", "2", "--kv-codec", "q4", "--kv-group-size", "4",
                   "--max-sequence-length", "8", "--tile-rows", "3", "--memory-budget", "1MiB",
                   "--fork-tokens", "2,4"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        derived = report["derived_sequence"]
        self.assertEqual(report["token_ids"], [1, 3, 5, 7])  # The prompt sequence is untouched.
        self.assertEqual(derived["token_ids"], [1, 3, 5, 7, 2, 4])
        self.assertEqual(derived["prefix_adoption"]["inherited_tokens"], 4)
        self.assertEqual(derived["prefix_adoption"]["copied_pages"], 0)
        self.assertEqual(derived["kv_shared_page_count"], 2)
        self.assertGreater(derived["kv_shared_allocation_bytes"], derived["kv_owned_allocation_bytes"])
        with self.sequence("q4", {"kv_group_size": 4}, page_tokens=2) as control:
            control.prefill([1, 3, 5, 7])
            appended = control.append([2, 4])
        digest = hashlib.sha256()
        for row in appended:
            for value in row:
                digest.update(struct.pack("<f", value))
        self.assertEqual(derived["logits_sha256"], digest.hexdigest())
        # argparse prints the whole usage on any error, so assert the specific
        # message: a weaker check would pass for an unrelated rejection. Each
        # case is valid except for the executor a derived sequence needs.
        common = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path), "--tokens", "1,3,5,7",
                  "--max-sequence-length", "8", "--tile-rows", "3", "--memory-budget", "1MiB",
                  "--fork-tokens", "2,4"]
        unpaged = [*common, "--kv-cache"]
        tiers = [*common, "--kv-cache", "--kv-page-tokens", "2", "--kv-policy", "age", "--kv-group-size", "4"]
        for invalid in (unpaged, tiers):
            rejected = subprocess.run(invalid, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("--fork-tokens requires --kv-cache, --kv-page-tokens and the homogeneous KV policy",
                          rejected.stderr)
            self.assertNotIn("Traceback", rejected.stderr)

    def test_age_tiers_reject_derived_sequences(self):
        from runtime.nexapack.tiered import TieredTransformerSession
        with TieredTransformerSession(self.path, memory_budget="1MiB", max_sequence_length=8,
                                      page_tokens=2, kv_group_size=4) as session:
            session.prefill([1, 3, 5])
            with self.assertRaises(ValueError) as failure:
                session.fork()
            self.assertIn("derived sequences", str(failure.exception))
            self.assertEqual(session.token_ids, (1, 3, 5))


if __name__ == "__main__":
    unittest.main()

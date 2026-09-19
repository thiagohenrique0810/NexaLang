"""Derived sequences under age tiers: private migration over a shared prefix."""
import ctypes
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_tiered_transformer_regressions import _TierFixture


class TieredSequenceRegressions(_TierFixture):
    def tiers(self, **options):
        return self.session(None, page_tokens=options.pop("page_tokens", 1),
                            hot_pages=options.pop("hot_pages", 1),
                            warm_pages=options.pop("warm_pages", 1), **options)

    @staticmethod
    def snapshot(session):
        return [(page.codec, ctypes.string_at(page.address, page.layout.page_extent_bytes))
                for page in session._pages]

    def test_fork_inherits_codecs_and_continues_with_identical_logits(self):
        with self.tiers() as parent:
            parent.prefill([1, 3, 5])
            self.assertEqual(self.codecs(parent), ["q3", "q4", "f32"])
            child = parent.fork()
            self.addCleanup(child.close)
            # The whole prefix is complete at one token per page: nothing is copied.
            self.assertEqual(self.codecs(child), ["q3", "q4", "f32"])
            self.assertEqual(child._page_descriptors, parent._page_descriptors)
            self.assertEqual(child.report()["kv_prefix_adoption"],
                             {"inherited_tokens": 3, "shared_pages": 3, "copied_pages": 0, "copied_bytes": 0})
            self.assertTrue(all(page.shared for page in child._pages))
            with self.tiers() as control:
                control.prefill([1, 3, 5])
                expected = control.append([7, 2])
            self.assertEqual(child.append([7, 2]), expected)
            self.assertEqual(parent.append([7, 2]), expected)

    def test_migration_is_private_to_the_sequence_that_performs_it(self):
        with self.tiers() as parent:
            parent.prefill([1, 3, 5])
            before = self.snapshot(parent)
            child = parent.fork()
            self.addCleanup(child.close)
            child.append([7, 2])
            # Aging re-encodes into new pages and releases the sources, so the
            # parent keeps its own codecs, bytes and addresses untouched.
            self.assertEqual(self.snapshot(parent), before)
            self.assertEqual(self.codecs(parent), ["q3", "q4", "f32"])
            self.assertEqual(self.codecs(child), ["q3", "q3", "q3", "q4", "f32"])
            still_shared = [page for page in child._pages if page.shared]
            self.assertEqual(len(still_shared), 1)  # Only the page neither sequence migrated.
            self.assertIs(still_shared[0], parent._pages[0])
            parent.append([4, 6])
            self.assertEqual(self.codecs(parent), ["q3", "q3", "q3", "q4", "f32"])
            self.assertEqual(child.token_ids, (1, 3, 5, 7, 2))

    def test_partial_hot_page_is_copied_so_both_sequences_keep_writing(self):
        with self.tiers(page_tokens=2) as parent:
            parent.prefill([1, 3, 5])
            child = parent.fork()
            self.addCleanup(child.close)
            adoption = child.report()["kv_prefix_adoption"]
            self.assertEqual((adoption["shared_pages"], adoption["copied_pages"]), (1, 1))
            self.assertGreater(adoption["copied_bytes"], 0)
            self.assertIsNot(child._pages[-1], parent._pages[-1])
            self.assertEqual(self.snapshot(child)[-1], self.snapshot(parent)[-1])
            child.decode(7)
            parent.decode(2)
            self.assertNotEqual(self.snapshot(child)[-1], self.snapshot(parent)[-1])
            self.assertEqual(child.token_ids, (1, 3, 5, 7))
            self.assertEqual(parent.token_ids, (1, 3, 5, 2))

    def test_reports_separate_shared_pages_and_survive_parent_release(self):
        with self.tiers() as parent:
            parent.prefill([1, 3, 5])
            child = parent.fork()
            self.addCleanup(child.close)
            memory = child.report()["memory"]
            self.assertEqual(memory["kv_shared_page_count"], 3)
            self.assertEqual(memory["kv_shared_allocation_bytes"], memory["kv_resident_allocation_bytes"])
            shared, expected = child._pages[0], self.snapshot(child)[0]
            parent.reset()
            self.assertEqual(parent.report()["memory"]["kv_shared_page_count"], 0)
            self.assertTrue(shared.address)
            self.assertFalse(shared.shared)
            self.assertEqual(self.snapshot(child)[0], expected)
            child.append([7, 2])
            self.assertEqual(child.token_ids, (1, 3, 5, 7, 2))

    def test_fork_requires_the_same_policy_and_rejects_a_used_session(self):
        with self.tiers() as parent:
            parent.prefill([1, 3, 5])
            for overrides in ({"hot_pages": 2}, {"warm_pages": 2}, {"kv_group_size": 4}, {"page_tokens": 2}):
                with self.subTest(overrides=sorted(overrides)), self.assertRaises(ValueError):
                    parent.fork(**overrides)
            with self.tiers() as stranger:
                stranger.prefill([2, 4])
                with self.assertRaises(ValueError):
                    stranger._adopt_prefix(parent)
            self.assertEqual(parent.token_ids, (1, 3, 5))
            parent.append([7, 2])

    def test_failed_adoption_leaves_no_shared_reference_behind(self):
        with self.tiers(page_tokens=2) as parent:
            parent.prefill([1, 3, 5])
            with patch("ctypes.memmove", side_effect=OSError("injected copy failure")):
                with self.assertRaises(OSError):
                    parent.fork()
            self.assertFalse(any(page.shared for page in parent._pages))
            self.assertEqual(parent.token_ids, (1, 3, 5))
            child = parent.fork()
            self.addCleanup(child.close)
            self.assertEqual(child.token_ids, (1, 3, 5))


if __name__ == "__main__":
    unittest.main()

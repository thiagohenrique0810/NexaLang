"""Derived sequences over a backing store: shared files and their ownership."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_offloaded_transformer_regressions import _OffloadFixture


class OffloadedSequenceRegressions(_OffloadFixture):
    def parent(self, name="shared", **options):
        session = self.session(kv_backing_store=self.directory / name,
                               page_tokens=options.pop("page_tokens", 1), **options)
        self.addCleanup(session.close)
        return session

    @staticmethod
    def files(session):
        return set(session._store.directory.iterdir()) if session._store is not None else set()

    def test_a_derived_sequence_reads_the_parent_cold_files(self):
        parent = self.parent()
        parent.prefill([1, 3, 5])
        cold = self.refs(parent)
        self.assertTrue(cold)
        child = parent.fork()
        self.addCleanup(child.close)
        self.assertEqual(child.token_ids, parent.token_ids)
        # Nothing was copied: both sequences point at the same published files.
        self.assertEqual([page.ref for page in child._pages if hasattr(page, "ref")], cold)
        self.assertEqual(child.report()["kv_prefix_adoption"]["copied_bytes"], 0)
        for ref in cold:
            self.assertEqual(parent._store.references(ref), 2)
        self.assertEqual(child.decode(7), parent.decode(7))

    def test_the_parent_may_close_first_and_the_files_survive(self):
        parent = self.parent("outlive")
        parent.prefill([1, 3, 5])
        store, cold = parent._store, self.refs(parent)
        child = parent.fork()
        self.addCleanup(child.close)
        expected = [self.packed_page(child, page) for page in child._pages]
        parent.close()
        self.assertEqual(store.holders, 1)
        self.assertTrue(all(ref.path.exists() for ref in cold))
        self.assertEqual([self.packed_page(child, page) for page in child._pages], expected)
        child.append([7, 2])
        self.assertEqual(child.token_ids, (1, 3, 5, 7, 2))
        child.close()
        # The last holder removes what no sequence reads any more.
        self.assertFalse(any(ref.path.exists() for ref in cold))

    def test_the_child_may_close_first_and_the_parent_keeps_reading(self):
        parent = self.parent("child-first")
        parent.prefill([1, 3, 5])
        cold = self.refs(parent)
        child = parent.fork()
        child.append([7])
        child.close()
        self.assertTrue(all(ref.path.exists() for ref in cold))
        for ref in cold:
            self.assertEqual(parent._store.references(ref), 1)
        self.assertEqual(parent.token_ids, (1, 3, 5))
        parent.decode(7)

    def test_retiring_a_shared_page_in_one_sequence_keeps_the_other_intact(self):
        parent = self.parent("retire")
        parent.prefill([1, 3, 5])
        cold = self.refs(parent)
        child = parent.fork()
        self.addCleanup(child.close)
        expected = [self.packed_page(child, page) for page in child._pages]
        parent.prefill([2, 4, 6])  # Replacement retires every previous file.
        self.assertTrue(all(ref.path.exists() for ref in cold))
        self.assertEqual([self.packed_page(child, page) for page in child._pages], expected)
        self.assertEqual(child.token_ids, (1, 3, 5))
        child.decode(7)
        parent.decode(8)

    def test_a_reset_parent_leaves_the_derived_prefix_readable(self):
        parent = self.parent("reset")
        parent.prefill([1, 3, 5])
        child = parent.fork()
        self.addCleanup(child.close)
        expected = [self.packed_page(child, page) for page in child._pages]
        parent.reset()
        self.assertEqual(parent.token_ids, ())
        self.assertEqual([self.packed_page(child, page) for page in child._pages], expected)
        child.decode(7)
        self.assertEqual(child.token_ids, (1, 3, 5, 7))

    def test_each_sequence_admits_its_own_budget(self):
        parent = self.parent("budgets")
        parent.prefill([1, 3, 5])
        child = parent.fork()
        self.addCleanup(child.close)
        parent_memory = parent.report()["memory"]
        child_memory = child.report()["memory"]
        # Sharing lowers what exists on disk, never the admitted reservation.
        self.assertEqual(parent_memory["kv_reserved_capacity_bytes"],
                         child_memory["kv_reserved_capacity_bytes"])
        self.assertEqual(child_memory["kv_inherited_stores"], 1)
        self.assertEqual(parent_memory["kv_inherited_stores"], 0)
        self.assertGreaterEqual(child_memory["kv_shared_page_count"], 1)

    def test_a_failed_child_transaction_leaves_the_parent_untouched(self):
        from unittest.mock import patch
        parent = self.parent("failure")
        parent.prefill([1, 3, 5])
        cold = self.refs(parent)
        child = parent.fork()
        self.addCleanup(child.close)
        tokens = parent.token_ids
        # The child has no store of its own until it first evicts a page.
        from runtime.nexapack.kv_store import KVPageStore
        with patch.object(KVPageStore, "write_page", side_effect=OSError("injected write failure")):
            with self.assertRaises(OSError):
                child.append([7, 2])
        self.assertEqual(child.token_ids, (1, 3, 5))
        self.assertEqual(parent.token_ids, tokens)
        self.assertTrue(all(ref.path.exists() for ref in cold))
        parent.decode(7)
        child.decode(4)

    def test_cli_derives_a_sequence_over_a_backing_store(self):
        directory = self.directory / "cli-fork"
        command = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path),
                   "--tokens", "1,3,5,7", "--kv-page-tokens", "1", "--kv-policy", "age",
                   "--kv-hot-pages", "1", "--kv-warm-pages", "1", "--kv-group-size", "3",
                   "--kv-backing-store", str(directory), "--fork-tokens", "2,4",
                   "--max-sequence-length", "8", "--tile-rows", "3", "--memory-budget", "1MiB"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        derived = report["derived_sequence"]
        self.assertEqual(derived["token_ids"], [1, 3, 5, 7, 2, 4])
        self.assertEqual(derived["prefix_adoption"]["copied_bytes"], 0)
        # Both sequences closed: the store leaves nothing behind.
        self.assertFalse(any(path.is_file() for path in directory.rglob("*")))


if __name__ == "__main__":
    unittest.main()

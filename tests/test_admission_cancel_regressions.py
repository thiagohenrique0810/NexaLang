"""Joint admission under one process ceiling, and cancelling a call in flight."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.nexapack.admission import PoolAdmissionError, SessionMemoryPool
from runtime.nexapack.transformer import SessionCancelled
from test_tiered_transformer_regressions import _TierFixture


class SessionMemoryPoolRegressions(unittest.TestCase):
    def test_a_pool_admits_until_the_ceiling_and_then_refuses_with_the_numbers(self):
        pool = SessionMemoryPool(1000)
        first = pool.admit("a", 600)
        with self.assertRaises(PoolAdmissionError) as raised:
            pool.admit("b", 401)
        error = raised.exception
        self.assertEqual((error.label, error.required_bytes, error.reserved_bytes, error.limit_bytes),
                         ("b", 401, 600, 1000))
        # Exactly filling the ceiling is admitted; one byte more is not.
        second = pool.admit("b", 400)
        self.assertEqual(pool.available_bytes, 0)
        with self.assertRaises(PoolAdmissionError):
            pool.admit("c", 1)
        self.assertEqual({pool.release(first), pool.release(second)}, {600, 400})

    def test_releasing_twice_does_not_credit_twice(self):
        pool = SessionMemoryPool(1000)
        handle = pool.admit("a", 600)
        self.assertEqual(pool.release(handle), 600)
        self.assertEqual(pool.release(handle), 0)
        self.assertEqual(pool.release(None), 0)
        self.assertEqual(pool.reserved_bytes, 0)
        self.assertEqual(pool.members, 0)

    def test_a_refused_admission_reserves_nothing(self):
        pool = SessionMemoryPool(1000)
        pool.admit("a", 600)
        for _ in range(3):
            with self.assertRaises(PoolAdmissionError):
                pool.admit("b", 500)
        self.assertEqual(pool.reserved_bytes, 600)
        self.assertEqual(pool.counters["rejections"], 3)
        self.assertEqual(pool.counters["admissions"], 1)

    def test_the_counters_track_the_peak_and_survive_serialization(self):
        pool = SessionMemoryPool("1KiB")
        a, b = pool.admit("a", 600), pool.admit("b", 400)
        pool.release(a)
        pool.release(b)
        data = json.loads(json.dumps(pool.to_dict()))
        self.assertEqual(data["peak_reserved_bytes"], 1000)
        self.assertEqual(data["reserved_bytes"], 0)
        self.assertEqual(data["limit_bytes"], 1024)
        self.assertEqual(data["policy_id"], "PROCESS_JOINT_ADMISSION_UPPER_BOUND_V1")

    def test_concurrent_admissions_never_exceed_the_ceiling(self):
        pool = SessionMemoryPool(10 * 100)
        granted, barrier = [], threading.Barrier(16)
        lock = threading.Lock()

        def attempt():
            barrier.wait(10)
            try:
                handle = pool.admit("t", 100)
            except PoolAdmissionError:
                return
            with lock:
                granted.append(handle)

        threads = [threading.Thread(target=attempt) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        # Sixteen threads raced for ten slots; the ceiling held exactly.
        self.assertEqual(len(granted), 10)
        self.assertEqual(pool.reserved_bytes, 1000)

    def test_a_pool_rejects_a_limit_that_is_not_a_positive_size(self):
        for bad in (0, -1, "0B", True, 1.5):
            with self.assertRaises((ValueError, TypeError)):
                SessionMemoryPool(bad)


class SessionAdmissionRegressions(_TierFixture):
    def test_two_sessions_that_each_fit_alone_do_not_both_fit_together(self):
        solo = self.session(page_tokens=1)
        bound = solo.admission_bytes
        solo.close()
        # A ceiling with room for one session and a byte to spare.
        pool = SessionMemoryPool(bound + 1)
        first = self.session(page_tokens=1, memory_pool=pool)
        self.addCleanup(first.close)
        with self.assertRaises(PoolAdmissionError):
            self.session(page_tokens=1, memory_pool=pool)
        self.assertEqual(pool.members, 1)
        self.assertEqual(pool.reserved_bytes, bound)

    def test_closing_a_session_returns_its_reservation(self):
        solo = self.session(page_tokens=1)
        bound = solo.admission_bytes
        solo.close()
        pool = SessionMemoryPool(bound + 1)
        first = self.session(page_tokens=1, memory_pool=pool)
        first.prefill([1, 3])
        first.close()
        self.assertEqual(pool.reserved_bytes, 0)
        second = self.session(page_tokens=1, memory_pool=pool)
        self.addCleanup(second.close)
        second.prefill([1, 3])
        self.assertEqual(second.token_ids, (1, 3))
        self.assertEqual(pool.counters["admissions"], 2)

    def test_a_refused_session_leaves_no_file_handle_and_no_reservation(self):
        solo = self.session(page_tokens=1)
        bound = solo.admission_bytes
        solo.close()
        pool = SessionMemoryPool(bound)
        first = self.session(page_tokens=1, memory_pool=pool)
        self.addCleanup(first.close)
        with self.assertRaises(PoolAdmissionError):
            self.session(page_tokens=1, memory_pool=pool)
        # The refused session closed its bundle on the way out; the survivor
        # is untouched and still executes.
        self.assertEqual(pool.reserved_bytes, bound)
        first.prefill([1, 3])
        self.assertEqual(first.token_ids, (1, 3))

    def test_a_derived_sequence_admits_against_the_same_ceiling(self):
        solo = self.session(page_tokens=1)
        bound = solo.admission_bytes
        solo.close()
        pool = SessionMemoryPool(2 * bound)
        parent = self.session(page_tokens=1, memory_pool=pool)
        self.addCleanup(parent.close)
        parent.prefill([1, 3])
        child = parent.fork()
        self.addCleanup(child.close)
        # Sharing the prefix lowers residence, never the reservation: the child
        # may append, and then the pages stop being shared.
        self.assertEqual(pool.reserved_bytes, 2 * bound)
        self.assertEqual(pool.members, 2)
        with self.assertRaises(PoolAdmissionError):
            parent.fork()
        child.append([7])
        self.assertEqual(child.token_ids, (1, 3, 7))

    def test_without_a_pool_nothing_changes(self):
        session = self.session(page_tokens=1)
        self.addCleanup(session.close)
        session.prefill([1, 3])
        report = session.report()
        self.assertNotIn("memory_pool", report)
        self.assertNotIn("session_admission_bytes", report["memory"])

    def test_the_report_publishes_the_shared_ceiling(self):
        pool = SessionMemoryPool("1MiB")
        session = self.session(page_tokens=1, memory_pool=pool)
        self.addCleanup(session.close)
        session.prefill([1, 3])
        report = session.report()
        self.assertEqual(report["memory"]["session_admission_bytes"], session.admission_bytes)
        self.assertEqual(report["memory_pool"]["reserved_bytes"], session.admission_bytes)
        self.assertEqual(report["memory_pool"]["members"], 1)

    def test_cli_shares_one_ceiling_between_a_sequence_and_its_fork(self):
        command = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path),
                   "--tokens", "1,3,5", "--kv-policy", "age", "--kv-page-tokens", "1",
                   "--kv-group-size", "3", "--fork-tokens", "7", "--memory-pool", "1MiB",
                   "--max-sequence-length", "8", "--tile-rows", "3", "--memory-budget", "1MiB"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        pool = report["derived_sequence"]["memory_pool"]
        self.assertEqual(pool["members"], 2)
        self.assertEqual(pool["admissions"], 2)
        self.assertEqual(pool["reserved_bytes"], 2 * report["memory"]["session_admission_bytes"])

    def test_cli_refuses_a_ceiling_that_does_not_fit_the_session(self):
        command = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path),
                   "--tokens", "1,3", "--kv-policy", "age", "--kv-page-tokens", "1",
                   "--kv-group-size", "3", "--memory-pool", "1KiB",
                   "--max-sequence-length", "8", "--tile-rows", "3", "--memory-budget", "1MiB"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already reserved by open sessions", result.stderr + result.stdout)


class SessionCancellationRegressions(_TierFixture):
    def cancel_at(self, session, step_index):
        """Stop the call between operators, from a thread that is not running it."""
        reached, released = threading.Event(), threading.Event()
        original = type(session)._execution_steps

        def steps(self, graph, execution_context):
            for index, step in enumerate(original(self, graph, execution_context)):
                if index == step_index:
                    reached.set()
                    released.wait(10)
                yield step

        return reached, released, patch.object(type(session), "_execution_steps", steps)

    def run_cancelled(self, session, call, step_index=1):
        reached, released, patcher = self.cancel_at(session, step_index)
        outcome = {}

        def worker():
            try:
                call()
            except BaseException as error:  # noqa: BLE001 - recorded for the assertion
                outcome["error"] = error

        with patcher:
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(reached.wait(10))
            self.assertTrue(session.cancel())
            released.set()
            thread.join(20)
        self.assertFalse(thread.is_alive())
        return outcome.get("error")

    def test_cancelling_a_prefill_leaves_an_empty_session(self):
        session = self.session(page_tokens=1)
        self.addCleanup(session.close)
        error = self.run_cancelled(session, lambda: session.prefill([1, 3, 5]))
        self.assertIsInstance(error, SessionCancelled)
        self.assertEqual(session.token_ids, ())
        self.assertEqual(session._pages, [])
        # The session is usable afterwards: cancelling is a rollback, not a fault.
        session.prefill([1, 3, 5])
        self.assertEqual(session.token_ids, (1, 3, 5))

    def test_cancelling_a_decode_preserves_the_committed_prefix(self):
        session = self.session(page_tokens=1)
        self.addCleanup(session.close)
        session.prefill([1, 3])
        expected, codecs = self.records(session), self.codecs(session)
        error = self.run_cancelled(session, lambda: session.decode(5))
        self.assertIsInstance(error, SessionCancelled)
        self.assertEqual(session.token_ids, (1, 3))
        self.assertEqual(self.records(session), expected)
        self.assertEqual(self.codecs(session), codecs)
        self.assertEqual(session.report()["cache_length"], 2)
        self.assertEqual(session.decode(5), self.session_reference())

    def session_reference(self):
        other = self.session(page_tokens=1)
        self.addCleanup(other.close)
        other.prefill([1, 3])
        return other.decode(5)

    def test_cancelling_during_the_migration_rolls_the_whole_call_back(self):
        from runtime.nexapack.tiered import TieredTransformerSession
        session = self.session(page_tokens=1)
        self.addCleanup(session.close)
        session.prefill([1, 3])
        expected = self.records(session)
        original = TieredTransformerSession._migrate_pages

        def spy(inner, *args, **kwargs):
            # The numerical phase is done and the pages are written; cancelling
            # here must still discard them and keep the previous prefix.
            inner.cancel()
            return original(inner, *args, **kwargs)

        with patch.object(TieredTransformerSession, "_migrate_pages", spy):
            with self.assertRaises(SessionCancelled):
                session.decode(5)
        self.assertEqual(session.token_ids, (1, 3))
        self.assertEqual(self.records(session), expected)
        session.decode(5)
        self.assertEqual(session.token_ids, (1, 3, 5))

    def test_cancel_outside_a_call_does_nothing_and_arms_nothing(self):
        session = self.session(page_tokens=1)
        self.addCleanup(session.close)
        self.assertFalse(session.cancel())
        session.prefill([1, 3])
        self.assertFalse(session.cancel())
        # A stale cancel must not kill the next call.
        session.decode(5)
        self.assertEqual(session.token_ids, (1, 3, 5))

    def test_a_concurrent_call_is_refused_instead_of_corrupting_the_state(self):
        session = self.session(page_tokens=1)
        self.addCleanup(session.close)
        session.prefill([1, 3])
        reached, released, patcher = self.cancel_at(session, 1)
        errors = {}

        def worker():
            try:
                session.decode(5)
            except BaseException as error:  # noqa: BLE001
                errors["inner"] = error

        with patcher:
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(reached.wait(10))
            with self.assertRaises(ValueError):
                session.decode(7)
            released.set()
            thread.join(20)
        self.assertIsNone(errors.get("inner"))
        self.assertEqual(session.token_ids, (1, 3, 5))

    def test_the_baseline_executor_is_cancellable_too(self):
        from runtime.nexapack.transformer import TransformerSession
        session = TransformerSession(self.path, memory_budget="1MiB", max_sequence_length=8,
                                     tile_rows=3)
        self.addCleanup(session.close)
        error = self.run_cancelled(session, lambda: session.prefill([1, 3, 5]))
        self.assertIsInstance(error, SessionCancelled)
        self.assertEqual(session.token_ids, ())
        self.assertEqual(len(session.prefill([1, 3, 5])), 3)


if __name__ == "__main__":
    unittest.main()

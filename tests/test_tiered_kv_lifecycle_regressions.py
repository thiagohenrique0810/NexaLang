"""Tiered migration owner lifetimes and rollback with retained tracebacks."""
import ctypes
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.tiered_kv_plan import TieredKVPolicy
from runtime.nexapack.transformer import _load_kernels
from test_tiered_transformer_regressions import _TierFixture


class _TrackingScalar:
    """Track real owning arrays while leaving from_address views non-owning."""
    def __init__(self, scalar, label, references, fail):
        self.scalar, self.label, self.references, self.fail = scalar, label, references, fail

    def __mul__(self, count):
        real = self.scalar * count
        label, references, fail = self.label, self.references, self.fail

        class Array:
            @staticmethod
            def from_address(address):
                return real.from_address(address)

            def __new__(cls):
                if fail == label:
                    raise MemoryError(f"injected {label} allocation failure")
                owner = real()
                references.append((label, weakref.ref(owner)))
                return owner

        return Array


class _TrackingCTypes:
    def __init__(self, references, *, fail=None):
        self.c_float = _TrackingScalar(ctypes.c_float, "head scratch", references, fail)
        self.c_double = _TrackingScalar(ctypes.c_double, "migration statistics", references, fail)

    def __getattr__(self, name):
        return getattr(ctypes, name)


class TieredKVLifecycleRegressions(_TierFixture):
    def capture_error(self, action, error_type):
        try:
            action()
        except error_type as error:
            self.assertIsNotNone(error.__traceback__)
            return error
        self.fail(f"Expected injected {error_type.__name__}")

    def assert_migration_owners_released(self, references):
        gc.collect()
        for label, owner in references:
            self.assertIsNone(owner(), f"retained {label} owner after migration ended")

    def snapshot(self, session):
        return (session.token_ids, session.report(), self.records(session),
                tuple(page.address for page in session._pages), session._page_descriptors)

    def assert_snapshot(self, session, before):
        self.assertEqual(self.snapshot(session), before)
        self.assertIsNone(session._pending_pages)
        self.assertIsNone(session._transaction)

    def test_success_releases_head_and_statistics_owners_before_return(self):
        references = []
        with self.session(page_tokens=1) as session:
            with patch("runtime.nexapack.tiered.ctypes", _TrackingCTypes(references)):
                session.prefill([1, 3, 5, 7])
            self.assertEqual([label for label, _ in references], ["head scratch", "migration statistics"])
            self.assert_migration_owners_released(references)
            self.assertEqual(session.report()["kv_migration"]["pages_reencoded"], 3)
            self.assertTrue(all(page.address and page.arena is not None for page in session._pages))

    def test_scratch_allocation_failure_rolls_back_without_retained_pages(self):
        for failing_owner in ("head scratch", "migration statistics"):
            with self.subTest(owner=failing_owner), self.session(page_tokens=1) as session:
                session.prefill([1])
                before = self.snapshot(session)
                references, pages = [], []
                original = session._allocate_page

                def allocate(codec="f32"):
                    return self.track_page(original(codec), pages)

                with patch.object(session, "_allocate_page", side_effect=allocate):
                    with patch("runtime.nexapack.tiered.ctypes", _TrackingCTypes(references, fail=failing_owner)):
                        saved = self.capture_error(lambda: session.append([3, 5]), MemoryError)
                self.assert_snapshot(session, before)
                self.assert_migration_owners_released(references)
                self.assert_pages_released(pages)
                if failing_owner == "migration statistics":
                    self.assertEqual([label for label, _ in references], ["head scratch"])
                session.append([3, 5])
                self.assertEqual(session.token_ids, (1, 3, 5))
                self.assertIsNotNone(saved.__traceback__)

    def test_native_exception_keeps_borrowed_views_but_not_owners(self):
        real = _load_kernels()
        for mode in ("append", "prefill"):
            with self.subTest(mode=mode), self.session(page_tokens=1) as session:
                session.prefill([1, 3])
                before = self.snapshot(session)
                references, pages, native_calls = [], [], []
                original = session._allocate_page

                def allocate(codec="f32"):
                    return self.track_page(original(codec), pages)

                def reencode(*args):
                    # Preserve views from both a completed row batch and the
                    # failing one, exactly as a retained native-call traceback.
                    native_calls.append(args)
                    if len(native_calls) == 2:
                        raise ArithmeticError("injected migration call exception")
                    return real.nexa_kv_reencode_rows(*args)

                proxy = SimpleNamespace(nexa_kv_reencode_rows=reencode)
                with patch.object(session, "_allocate_page", side_effect=allocate):
                    with patch("runtime.nexapack.tiered.ctypes", _TrackingCTypes(references)):
                        with patch("runtime.nexapack.tiered._load_kernels", return_value=proxy):
                            saved = self.capture_error(lambda: getattr(session, mode)([5, 7]), ArithmeticError)
                self.assertEqual(len(native_calls), 2)
                self.assert_snapshot(session, before)
                self.assert_migration_owners_released(references)
                self.assert_pages_released(pages)
                # Never dereference stale views; their existence must not keep
                # any owned allocation alive while retrying the transaction.
                self.assertTrue(native_calls)
                getattr(session, mode)([5, 7])
                self.assertIsNotNone(saved.__traceback__)

    def test_native_error_status_discards_replacement_and_preserves_report(self):
        with self.session(page_tokens=1) as session:
            session.prefill([1, 3])
            before = self.snapshot(session)
            references, pages = [], []
            original = session._allocate_page

            def allocate(codec="f32"):
                return self.track_page(original(codec), pages)

            proxy = SimpleNamespace(nexa_kv_reencode_rows=lambda *args: -4)
            with patch.object(session, "_allocate_page", side_effect=allocate):
                with patch("runtime.nexapack.tiered.ctypes", _TrackingCTypes(references)):
                    with patch("runtime.nexapack.tiered._load_kernels", return_value=proxy):
                        saved = self.capture_error(lambda: session.decode(5), ArithmeticError)
            self.assertIn("status -4", str(saved))
            self.assert_snapshot(session, before)
            self.assert_migration_owners_released(references)
            self.assert_pages_released(pages)
            session.decode(5)
            self.assertIsNotNone(saved.__traceback__)

    def test_later_target_allocation_failure_releases_completed_replacements(self):
        with self.session(page_tokens=1) as session:
            session.prefill([1, 3])
            before = self.snapshot(session)
            references, pages, targets = [], [], []
            original = session._allocate_page

            def allocate(codec="f32"):
                if codec != "f32":
                    targets.append(codec)
                    if len(targets) == 2:
                        raise MemoryError("injected second migration target failure")
                return self.track_page(original(codec), pages)

            with patch.object(session, "_allocate_page", side_effect=allocate):
                with patch("runtime.nexapack.tiered.ctypes", _TrackingCTypes(references)):
                    saved = self.capture_error(lambda: session.append([5, 7]), MemoryError)
            self.assertEqual(targets, ["q3", "q3"])
            self.assert_snapshot(session, before)
            self.assert_migration_owners_released(references)
            self.assert_pages_released(pages)
            session.append([5, 7])
            self.assertIsNotNone(saved.__traceback__)

    def test_final_report_failure_releases_all_new_pages_after_migration(self):
        for mode in ("append", "prefill"):
            with self.subTest(mode=mode), self.session(page_tokens=1) as session:
                session.prefill([1, 3])
                before = self.snapshot(session)
                references, pages, completed = [], [], []
                original_allocate, original_finalize = session._allocate_page, session._finalize_report

                def allocate(codec="f32"):
                    return self.track_page(original_allocate(codec), pages)

                def finalize(*args):
                    report = original_finalize(*args)
                    completed.append(report["kv_migration"]["pages_reencoded"])
                    raise MemoryError("injected final result failure after migration")

                with patch.object(session, "_allocate_page", side_effect=allocate):
                    with patch("runtime.nexapack.tiered.ctypes", _TrackingCTypes(references)):
                        with patch.object(session, "_finalize_report", side_effect=finalize):
                            saved = self.capture_error(lambda: getattr(session, mode)([5, 7]), MemoryError)
                self.assertTrue(completed and completed[0] > 0)
                self.assert_snapshot(session, before)
                self.assert_migration_owners_released(references)
                self.assert_pages_released(pages)
                getattr(session, mode)([5, 7])
                self.assertIsNotNone(saved.__traceback__)

    def test_policy_cannot_be_replaced_after_memory_admission(self):
        with self.session() as session:
            old = session.policy
            with self.assertRaises(AttributeError):
                session.policy = TieredKVPolicy(2, 0, 128)
            self.assertIs(session.policy, old)
            session.prefill([1, 3, 5])
            self.assertEqual(session.report()["kv_policy"], old.to_dict())


if __name__ == "__main__":
    unittest.main()

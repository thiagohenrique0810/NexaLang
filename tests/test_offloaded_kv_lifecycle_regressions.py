"""Backing-store failures retain the committed prefix and release transient owners."""
import errno
import ctypes
import gc
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_offloaded_transformer_regressions import _OffloadFixture


class Cancelled(BaseException):
    pass


class OffloadedLifecycleRegressions(_OffloadFixture):
    def capture(self, action, error_type):
        try:
            action()
        except error_type as error:
            self.assertIsNotNone(error.__traceback__)
            return error
        self.fail(f"Expected injected {error_type.__name__}")

    def snapshot(self, session):
        rows = []
        for descriptor, page in zip(session._page_descriptors, session._pages):
            contents = self.packed_page(session, page)
            valid = descriptor.valid_tokens * page.layout.token_bytes
            rows.append(tuple(contents[page.layout.buffer_offset(layer, kind):
                                       page.layout.buffer_offset(layer, kind) + valid]
                              for layer in range(session.config.num_hidden_layers) for kind in ("key", "value")))
        return (session.token_ids, session.report(), tuple(session._pages),
                session._page_descriptors, rows,
                {ref.path: ref.path.read_bytes() for ref in self.refs(session)})

    def assert_snapshot(self, session, before):
        self.assertEqual(self.snapshot(session), before)
        self.assertIsNone(session._pending_pages)
        self.assertIsNone(session._transaction)
        cache = session._reload_cache
        # A discarded transaction cannot leave a partially read page behind.
        self.assertEqual((cache.entry_count, cache.slot_count, cache.allocated_bytes), (0, 0, 0))

    def track_allocations(self, session, references):
        allocate = session._allocate_page
        def tracked(codec="f32"):
            page = allocate(codec)
            references.append((weakref.ref(page), weakref.ref(page.arena)))
            return page
        return tracked

    @staticmethod
    def private_files(session):
        return set(session._store.directory.iterdir()) if session._store is not None else set()

    def test_second_write_enospc_or_cancellation_removes_new_files_and_allows_retry(self):
        for exception in (OSError(errno.ENOSPC, "injected full store"), Cancelled("injected cancellation")):
            with self.subTest(error=type(exception).__name__), self.session(page_tokens=1) as session:
                session.prefill([1, 3, 5])
                before, files = self.snapshot(session), self.private_files(session)
                references, writes = [], []
                write_page = session._store.write_page
                def failing_write(*args):
                    if writes:
                        raise exception
                    ref = write_page(*args)
                    writes.append(ref)
                    return ref
                with patch.object(session, "_allocate_page", self.track_allocations(session, references)), \
                     patch.object(session._store, "write_page", failing_write):
                    retained = self.capture(lambda: session.append([7, 2]), type(exception))
                self.assertEqual(len(writes), 1)
                self.assertFalse(writes[0].path.exists())
                self.assertEqual(self.private_files(session), files)
                self.assert_snapshot(session, before)
                self.assert_pages_released(references)
                session.append([7, 2])
                self.assertEqual(session.token_ids, (1, 3, 5, 7, 2))
                self.assertIsNotNone(retained.__traceback__)

    def test_read_interruption_and_baseexception_release_reload_slot_and_preserve_prefix(self):
        for exception in (InterruptedError("injected read interruption"), OSError(errno.EIO, "injected read error"),
                          Cancelled("injected reload cancellation")):
            # Two cold pages and one slot guarantee a second, failing read:
            # a single cold page would be served from the slot after one load.
            with self.subTest(error=type(exception).__name__), self.session(page_tokens=1) as session:
                session.prefill([1, 3, 5, 7])
                before, files = self.snapshot(session), self.private_files(session)
                references, reads = [], []
                read_page = session._store.read_page
                def failing_read(*args):
                    if reads:
                        raise exception
                    count = read_page(*args)
                    reads.append(count)
                    return count
                with patch.object(session, "_allocate_page", self.track_allocations(session, references)), \
                     patch.object(session._store, "read_page", failing_read):
                    retained = self.capture(lambda: session.append([2, 4]), type(exception))
                self.assertEqual(len(reads), 1)
                self.assertEqual(self.private_files(session), files)
                self.assert_snapshot(session, before)
                self.assert_pages_released(references)
                session.decode(2)
                self.assertEqual(session.token_ids, (1, 3, 5, 7, 2))
                self.assertIsNotNone(retained.__traceback__)

    def test_failed_reload_discards_every_slot_of_a_multi_slot_cache(self):
        with self.session(page_tokens=1, kv_reload_slots=4) as session:
            session.prefill([1, 3, 5, 7])
            before, files = self.snapshot(session), self.private_files(session)
            references, reads = [], []
            read_page = session._store.read_page
            def failing_read(*args):
                if reads:
                    raise OSError(errno.EIO, "injected reload error")
                reads.append(read_page(*args))
                return reads[0]
            with patch.object(session, "_allocate_page", self.track_allocations(session, references)), \
                 patch.object(session._store, "read_page", failing_read):
                self.capture(lambda: session.append([2, 4]), OSError)
            # The first page loaded successfully; a cancelled transaction still
            # drops it, because no slot may outlive the call that filled it.
            self.assertEqual(len(reads), 1)
            self.assertEqual(self.private_files(session), files)
            self.assert_snapshot(session, before)
            self.assert_pages_released(references)
            session.append([2, 4])
            self.assertEqual(session.token_ids, (1, 3, 5, 7, 2, 4))
            # Only the two cold pages this retry actually read occupy slots;
            # pages evicted after the attention are never loaded back.
            cache = session._reload_cache
            self.assertEqual(cache.entry_count, 2)
            self.assertTrue(all(ref is None or session._store.contains(ref) for ref in cache._refs))

    def test_corrupted_cold_page_fails_checksum_then_restored_bytes_allow_retry(self):
        with self.session(page_tokens=1) as session, self.resident(page_tokens=1) as resident:
            session.prefill([1, 3, 5])
            resident.prefill([1, 3, 5])
            before = self.snapshot(session)
            ref = self.refs(session)[0]
            original = ref.path.read_bytes()
            corrupted = bytearray(original)
            corrupted[ref.payload_offset] ^= 1
            ref.path.write_bytes(corrupted)
            references = []
            try:
                with patch.object(session, "_allocate_page", self.track_allocations(session, references)):
                    retained = self.capture(lambda: session.decode(7), ValueError)
            finally:
                ref.path.write_bytes(original)
            self.assert_snapshot(session, before)
            self.assert_pages_released(references)
            self.assertEqual(session.decode(7), resident.decode(7))
            self.assert_same_pages(session, resident)
            self.assertIsNotNone(retained.__traceback__)

    def test_final_report_failure_after_persistence_removes_only_staged_files(self):
        for mode in ("append", "prefill"):
            with self.subTest(mode=mode), self.session(page_tokens=1) as session:
                session.prefill([1, 3, 5])
                before, files = self.snapshot(session), self.private_files(session)
                references, staged_files = [], []
                finalize = session._finalize_report
                def failing_finalize(*args):
                    report = finalize(*args)
                    staged_files.extend(self.private_files(session) - files)
                    self.assertGreater(len(staged_files), 0)
                    raise MemoryError("injected final report allocation failure")
                with patch.object(session, "_allocate_page", self.track_allocations(session, references)), \
                     patch.object(session, "_finalize_report", failing_finalize):
                    retained = self.capture(lambda: getattr(session, mode)([7, 2, 4]), MemoryError)
                self.assertFalse(any(path.exists() for path in staged_files))
                self.assertEqual(self.private_files(session), files)
                self.assert_snapshot(session, before)
                self.assert_pages_released(references)
                getattr(session, mode)([7, 2, 4])
                self.assertIsNotNone(retained.__traceback__)

    def test_failed_replacement_store_write_keeps_previous_store_pages_readable(self):
        with self.session(page_tokens=1) as session, self.resident(page_tokens=1) as resident:
            session.prefill([1, 3, 5, 7])
            resident.prefill([1, 3, 5, 7])
            before, files = self.snapshot(session), self.private_files(session)
            with patch.object(session._store, "write_page", side_effect=OSError(errno.ENOSPC, "store full")):
                retained = self.capture(lambda: session.prefill([2, 4, 6]), OSError)
            self.assertEqual(self.private_files(session), files)
            self.assert_snapshot(session, before)
            self.assertEqual(session.decode(2), resident.decode(2))
            self.assertIsNotNone(retained.__traceback__)

    def test_retired_file_removal_failure_is_deferred_without_rejecting_a_committed_prompt(self):
        with self.session(page_tokens=1) as session:
            session.prefill([1, 3, 5, 7])
            retired = self.refs(session)
            with patch.object(session._store, "remove", side_effect=OSError(errno.EIO, "temporary unlink error")):
                result = session.prefill([2])
            self.assertEqual(len(result), 1)
            self.assertEqual(session.token_ids, (2,))
            self.assertTrue(all(ref.path.exists() for ref in retired))
            self.assertEqual(set(session._retired_refs), set(retired))
            session.decode(4)
            self.assertFalse(any(ref.path.exists() for ref in retired))
            self.assertEqual(session._retired_refs, [])

    def test_cleanup_memory_failure_or_cancellation_after_publication_is_deferred(self):
        for error_type in (MemoryError, Cancelled):
            for operation in ("prefill", "reset"):
                with self.subTest(error=error_type.__name__, operation=operation), self.session(page_tokens=1) as session:
                    session.prefill([1, 3, 5, 7])
                    retired = self.refs(session)
                    with patch.object(session._store, "remove", side_effect=error_type("cleanup interrupted")) as remove:
                        if operation == "prefill":
                            result = session.prefill([2])
                            self.assertEqual(len(result), 1)
                        else:
                            session.reset()
                    self.assertEqual(remove.call_count, 1)
                    self.assertEqual(session.token_ids, (2,) if operation == "prefill" else ())
                    self.assertTrue(all(ref.path.exists() for ref in retired))
                    self.assertEqual(set(session._retired_refs), set(retired))
                    if operation == "prefill":
                        session.decode(4)
                    else:
                        session.prefill([4])
                    self.assertFalse(any(ref.path.exists() for ref in retired))
                    self.assertEqual(session._retired_refs, [])

    def test_cleanup_cancellation_before_transaction_preserves_state_and_allocates_nothing(self):
        for error_type in (MemoryError, Cancelled):
            with self.subTest(error=error_type.__name__), self.session(page_tokens=1) as session:
                session.prefill([1, 3, 5, 7])
                retired = self.refs(session)
                with patch.object(session._store, "remove", side_effect=OSError(errno.EIO, "unlink deferred")):
                    session.prefill([2])
                before, files = self.snapshot(session), self.private_files(session)
                with patch.object(session._store, "remove", side_effect=error_type("cleanup interrupted")), \
                     patch.object(session, "_allocate_page", side_effect=AssertionError("allocation before cleanup")) as allocate:
                    retained = self.capture(lambda: session.decode(4), error_type)
                allocate.assert_not_called()
                self.assert_snapshot(session, before)
                self.assertEqual(self.private_files(session), files)
                self.assertEqual(set(session._retired_refs), set(retired))
                self.assertIsNone(session._backing_io)
                session.decode(4)
                self.assertEqual(session.token_ids, (2, 4))
                self.assertFalse(any(ref.path.exists() for ref in retired))
                self.assertEqual(session._retired_refs, [])
                self.assertIsNotNone(retained.__traceback__)

    def test_completed_removal_followed_by_cancellation_reconciles_queue_on_retry(self):
        for when in ("after_publication", "before_transaction"):
            with self.subTest(when=when), self.session(page_tokens=1) as session:
                session.prefill([1, 3, 5, 7])
                retired = self.refs(session)
                if when == "before_transaction":
                    with patch.object(session._store, "remove", side_effect=OSError(errno.EIO, "unlink deferred")):
                        session.prefill([2])
                    before = self.snapshot(session)
                removed = []
                original_remove = session._store.remove
                def completed_then_cancelled(ref):
                    original_remove(ref)
                    removed.append(ref)
                    self.assertFalse(session._store.contains(ref))
                    raise Cancelled("cancelled after removal completed")
                with patch.object(session._store, "remove", completed_then_cancelled):
                    if when == "after_publication":
                        self.assertEqual(len(session.prefill([2])), 1)
                    else:
                        retained = self.capture(lambda: session.decode(4), Cancelled)
                        self.assert_snapshot(session, before)
                        self.assertIsNotNone(retained.__traceback__)
                self.assertEqual(session.token_ids, (2,))
                self.assertEqual(len(removed), 1)
                self.assertFalse(removed[0].path.exists())
                self.assertFalse(session._store.contains(removed[0]))
                self.assertIn(removed[0], session._retired_refs)
                session.decode(4)
                self.assertEqual(session.token_ids, (2, 4))
                self.assertEqual(session._retired_refs, [])
                self.assertTrue(all(not session._store.contains(ref) for ref in retired))
                self.assertFalse(any(ref.path.exists() for ref in retired))

    def test_stream_kernel_exception_retains_borrowed_views_without_owning_arena_or_reload_page(self):
        from runtime.nexapack.transformer import _load_kernels
        real = _load_kernels()
        owners, pages, borrowed = [], [], []
        class ArenaCTypes:
            @staticmethod
            def addressof(value):
                if isinstance(value, ctypes.Array) and value._b_needsfree_:
                    owners.append(weakref.ref(value))
                return ctypes.addressof(value)
            def __getattr__(self, name):
                return getattr(ctypes, name)
        class Kernels:
            @staticmethod
            def nexa_causal_gqa_attention_page(*args):
                borrowed.append(args)
                if len(borrowed) == 2:
                    raise Cancelled("interrupted streaming kernel")
                return real.nexa_causal_gqa_attention_page(*args)
            def __getattr__(self, name):
                return getattr(real, name)
        with self.session(page_tokens=1) as session:
            session.prefill([1, 3, 5])
            before = self.snapshot(session)
            with patch.object(session, "_allocate_page", self.track_allocations(session, pages)), \
                 patch("runtime.nexapack.transformer.ctypes", ArenaCTypes()), \
                 patch("runtime.nexapack.transformer._load_kernels", return_value=Kernels()):
                retained = self.capture(lambda: session.decode(7), Cancelled)
            self.assertEqual(len(borrowed), 2)
            self.assert_snapshot(session, before)
            self.assert_pages_released(pages)
            gc.collect()
            self.assertTrue(owners)
            self.assertTrue(all(owner() is None for owner in owners))
            # The deliberately retained pointer views are stale, never read.
            session.decode(7)
            self.assertIsNotNone(retained.__traceback__)


if __name__ == "__main__":
    unittest.main()

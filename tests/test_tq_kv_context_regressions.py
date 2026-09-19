"""TQ session admission and native-state ownership under retained failures.

The fake native library models allocation ownership and return codes. Numerical
kernel behavior is covered by the independent TQ attention/oracle suites.
"""
import gc
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.planner.memory import MemoryBudgetError
from runtime.nexapack.bundle import ModelBundleReader
from runtime.nexapack.format import NexaPackReader
from runtime.nexapack.paged import PagedTransformerSession
from runtime.nexapack.tq_kv import TQKVContext, TQ_STRUCT_RESERVE_BYTES, tq_kv_memory
from test_transformer_forward_regressions import random_bundle


def codebook(bits=3):
    count = 1 << bits
    return struct.pack("<" + "f" * count,
                       *[(index - (count - 1) / 2) / count for index in range(count)]).hex()


class FakeNative:
    """No native allocations; only weak references to caller-owned staging."""
    def __init__(self):
        self.live = {}
        self.created = []
        self.destroyed = []
        self.staging = []
        self.fail_allocate = False
        self.fail_export = False
        self.invalid_export = False
        self.preflight_extra = 0
        self.actual_extra = 0

    def tq_mse_context_memory_size(self, dim, bits):
        return 88 + 4 * dim + 4 * (2 * (1 << bits) - 1) + self.preflight_extra

    def _create(self, dim, bits, seed, book):
        if self.fail_allocate:
            return None
        handle = len(self.created) + 1
        self.live[handle] = (dim, bits, seed, book)
        self.created.append(handle)
        return handle

    def tq_create_mse(self, dim, bits, seed):
        return self._create(dim, bits, seed, codebook(bits))

    def tq_create_mse_from_codebook(self, dim, bits, seed, staging, count):
        self.staging.append(weakref.ref(staging))
        book = struct.pack("<" + "f" * count, *staging).hex()
        return self._create(dim, bits, seed, book)

    def tq_context_memory_bytes(self, handle):
        dim, bits, _, _ = self.live[handle]
        return self.tq_mse_context_memory_size(dim, bits) + self.actual_extra

    def tq_export_mse_codebook(self, handle, staging, count):
        self.staging.append(weakref.ref(staging))
        if self.fail_export:
            return -1
        values = struct.unpack("<" + "f" * count, bytes.fromhex(self.live[handle][3]))
        for index, value in enumerate(values):
            staging[index] = 0 if self.invalid_export else value
        return 0

    def tq_destroy(self, handle):
        self.destroyed.append(handle)
        del self.live[handle]


class TQContextRegressions(unittest.TestCase):
    def assert_staging_released(self, library):
        gc.collect()
        self.assertTrue(all(reference() is None for reference in library.staging),
                        "retained exception kept codebook staging alive")

    def test_memory_bound_separates_state_constructor_vector_and_f64_accumulator(self):
        with patch("runtime.nexapack.tq_kv._load_tq_kernels", side_effect=AssertionError("native load")):
            for dim in (2, 64, 1024):
                for bits in range(1, 9):
                    memory = tq_kv_memory(dim, bits)
                    levels = 1 << bits
                    self.assertEqual(memory["context_reserved_bytes"], TQ_STRUCT_RESERVE_BYTES + 4 * dim + 4 * (2 * levels - 1))
                    self.assertEqual(memory["constructor_staging_bytes"], max(8 * levels, 4 * (levels + 1)))
                    self.assertEqual(memory["vector_scratch_bytes"], 4 * dim)
                    self.assertEqual(memory["accumulator_bytes"], 8 * dim)

    def test_imported_and_generated_contexts_export_exact_book_and_close_once(self):
        for supplied in (None, codebook()):
            with self.subTest(imported=supplied is not None):
                library = FakeNative()
                with patch("runtime.nexapack.tq_kv._load_tq_kernels", return_value=library):
                    context = TQKVContext(64, 3, -42, supplied)
                    self.assertEqual(context.codebook_f32le, codebook())
                    self.assertEqual(context.state_bytes, library.tq_mse_context_memory_size(64, 3))
                    self.assertEqual(library.live[context._ctx][2], -42)
                    self.assert_staging_released(library)
                    context.close()
                    context.close()
                    self.assertIsNone(context._ctx)
                    self.assertEqual(library.destroyed, [1])
                    self.assertFalse(library.live)

    def test_native_preflight_rejects_struct_growth_before_context_allocation(self):
        library = FakeNative()
        library.preflight_extra = TQ_STRUCT_RESERVE_BYTES
        with patch("runtime.nexapack.tq_kv._load_tq_kernels", return_value=library):
            with self.assertRaisesRegex(ValueError, "reserve"):
                TQKVContext(64, 3, 42)
        self.assertEqual(library.created, [])
        self.assertEqual(library.destroyed, [])

    def test_retained_allocation_export_and_accounting_failures_release_state_and_staging(self):
        errors = []
        for failure in ("fail_allocate", "fail_export", "invalid_export", "actual_extra"):
            for supplied in (None, codebook()):
                with self.subTest(failure=failure, imported=supplied is not None):
                    library = FakeNative()
                    setattr(library, failure, 1)
                    with patch("runtime.nexapack.tq_kv._load_tq_kernels", return_value=library):
                        try:
                            TQKVContext(64, 3, 42, supplied)
                        except (MemoryError, ArithmeticError, ValueError) as error:
                            errors.append(error)
                        else:
                            self.fail("injected context failure succeeded")
                    self.assertFalse(library.live)
                    self.assertEqual(library.destroyed, library.created)
                    self.assert_staging_released(library)
        self.assertEqual(len(errors), 8)

    def test_invalid_arguments_and_codebook_fail_before_native_load(self):
        with patch("runtime.nexapack.tq_kv._load_tq_kernels", side_effect=AssertionError("native load")):
            for arguments in ((6, 3, 42, None), (64, True, 42, None), (64, 3, True, None),
                              (64, 3, 42, ""), (64, 3, 42, codebook().upper())):
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    TQKVContext(*arguments)


class TQSessionContextRegressions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nexa-tq-context-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "bundle"
        self.config, _ = random_bundle(self.path, layers=2, heads=4, kv_heads=2, tied=False, seed=501)

    def session(self, **options):
        return PagedTransformerSession(self.path, kv_codec="tq", max_sequence_length=8,
                                       max_chunk_length=3, page_tokens=2,
                                       memory_budget=options.pop("memory_budget", "1MiB"), **options)

    @staticmethod
    def fake_execute(session, tokens, *, execution_context):
        graph, plan = session._make_plan(len(tokens), execution_context=execution_context)
        report = session._report(graph, plan, executed=True, execution_context=execution_context)
        report["io"] = {}
        return [[0.0] * session.config.vocab_size for _ in tokens], report

    def test_budget_and_uninitialized_session_do_not_load_native_or_read_payload(self):
        with patch("runtime.nexapack.tq_kv._load_tq_kernels", side_effect=AssertionError("TQ load")), \
             patch("runtime.nexapack.transformer._load_kernels", side_effect=AssertionError("Transformer load")), \
             patch.object(NexaPackReader, "read_rows_into", side_effect=AssertionError("payload read")), \
             patch.object(ModelBundleReader, "read_f32_into", side_effect=AssertionError("norm read")):
            with self.session() as session:
                report = session.report()
                bound = report["memory"]["capacity_managed_buffers_bound_bytes"]
                self.assertIsNone(session._tq_context)
                self.assertIsNone(session.kv_codebook_f32le)
                self.assertEqual(session.resident_page_count, 0)
                self.assertEqual(report["memory"]["kv_tq_context_bytes"], 0)
                session.reset()
                self.assertIsNone(session.kv_codebook_f32le)
            with self.session(memory_budget=bound) as exact:
                self.assertEqual(exact.report()["memory"]["capacity_managed_buffers_bound_bytes"], bound)
            with self.assertRaises(MemoryBudgetError):
                self.session(memory_budget=bound - 1)

    def test_first_execution_failure_releases_context_pages_and_restores_provisional_plan(self):
        library = FakeNative()
        errors, allocations = [], []
        def fail_execute(session, tokens, *, execution_context):
            self.assertIsNotNone(execution_context.cache_plan.codebook_f32le)
            allocations.extend(weakref.ref(page.arena) for page in session._pending_pages)
            raise ArithmeticError("late execution failure")
        with patch("runtime.nexapack.tq_kv._load_tq_kernels", return_value=library), self.session() as session:
            initial = session.report()
            with patch.object(PagedTransformerSession, "_execute", fail_execute):
                try:
                    session.prefill([1, 3, 5])
                except ArithmeticError as error:
                    errors.append(error)
            self.assertEqual(session.report(), initial)
            self.assertEqual(session.token_ids, ())
            self.assertEqual(session.resident_page_count, 0)
            self.assertIsNone(session._pending_pages)
            self.assertIsNone(session._tq_context)
            self.assertIsNone(session.kv_codebook_f32le)
            self.assertEqual(library.created, library.destroyed)
            gc.collect()
            self.assertTrue(all(reference() is None for reference in allocations))
            with patch.object(PagedTransformerSession, "_execute", self.fake_execute):
                session.prefill([2, 4])
            self.assertEqual(session.token_ids, (2, 4))
            self.assertIsNotNone(session.kv_codebook_f32le)
        self.assertEqual(len(errors), 1)
        self.assertEqual(library.created, library.destroyed)

    def test_context_is_shared_across_append_reset_and_replacement_until_close(self):
        library = FakeNative()
        with patch("runtime.nexapack.tq_kv._load_tq_kernels", return_value=library), \
             patch.object(PagedTransformerSession, "_execute", self.fake_execute):
            with self.session(kv_seed=-1, kv_codebook_f32le=codebook()) as session:
                self.assertIsNone(session._tq_context)
                session.prefill([1, 3])
                native_context = session._tq_context
                plan = session._cache_plan
                pages = [weakref.ref(page.arena) for page in session._pages]
                session.append([5])
                session.reset()
                self.assertEqual(session.resident_page_count, 0)
                self.assertIs(session._tq_context, native_context)
                self.assertIs(session._cache_plan, plan)
                self.assertEqual(session.kv_codebook_f32le, codebook())
                self.assertEqual(library.created, [1])
                self.assertEqual(library.destroyed, [])
                gc.collect()
                self.assertTrue(all(reference() is None for reference in pages))
                memory = session.report()["memory"]
                self.assertEqual(memory["kv_tq_context_bytes"], native_context.state_bytes)
                self.assertEqual(memory["managed_buffers_peak_bound_bytes"],
                                 memory["arena_allocation_bytes"] + memory["reader_scratch_capacity_bytes"] + native_context.state_bytes)
                session.prefill([7])
                self.assertEqual(library.created, [1])
                self.assertIs(session._tq_context, native_context)
            session.close()
            self.assertEqual(library.destroyed, [1])
            self.assertIsNone(native_context._ctx)
            self.assertIsNone(session._tq_context)

    def test_failed_concrete_plan_publication_closes_created_context_without_pages(self):
        library = FakeNative()
        with patch("runtime.nexapack.tq_kv._load_tq_kernels", return_value=library), self.session() as session:
            from runtime.nexapack.paged import make_paged_kv_cache_plan
            def fail_concrete_plan(*arguments, **keywords):
                if keywords.get("codebook_f32le") is not None:
                    raise ValueError("concrete plan rejected")
                return make_paged_kv_cache_plan(*arguments, **keywords)
            with patch("runtime.nexapack.paged.make_paged_kv_cache_plan", fail_concrete_plan), \
                 patch.object(session, "_allocate_page", side_effect=AssertionError("page allocation")):
                with self.assertRaisesRegex(ValueError, "concrete plan rejected"):
                    session.prefill([1])
            self.assertFalse(library.live)
            self.assertEqual(library.created, library.destroyed)
            self.assertIsNone(session._tq_context)
            self.assertIsNone(session.kv_codebook_f32le)
            self.assertEqual(session.resident_page_count, 0)


if __name__ == "__main__":
    unittest.main()

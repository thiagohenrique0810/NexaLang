"""Managed buffers must be released even while failed-step tracebacks survive."""
import ctypes
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.nexapack import format as fmt
from runtime.nexapack.format import NexaPackError, NexaPackReader, write_q4_matrix


class ReaderBufferLifetimeRegressions(unittest.TestCase):
    def test_checksum_exception_retained_during_retry_releases_reader_scratch(self):
        owners, failures = [], []

        class TrackedBuffer(bytearray):
            def __init__(buffer, *args, **kwargs):
                self.assertTrue(all(owner() is None for owner in owners), "prior scratch survived into retry")
                super().__init__(*args, **kwargs)
                owners.append(weakref.ref(buffer))

        with tempfile.TemporaryDirectory(prefix="nexa-scratch-lifetime-") as directory:
            path = Path(directory) / "weights.nxp"
            write_q4_matrix(path, 2, 5, 3, [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], block_rows=2)
            original = path.read_bytes()
            with NexaPackReader(path) as reader:
                expected = reader.read_rows(0, 2)
                damaged = bytearray(original)
                damaged[-1] ^= 1
                path.write_bytes(damaged)
                destination = bytearray(2 * reader.row_bytes)
                with patch.object(fmt, "bytearray", TrackedBuffer, create=True):
                    try:
                        reader.read_rows_into(0, 2, destination)
                    except NexaPackError as error:
                        failures.append(error)  # Keep the original traceback throughout the retry.
                        self.assertIn("checksum", str(error))
                        self.assertEqual(len(owners), 1)
                        self.assertIsNone(owners[0](), "checksum traceback retained the 64 KiB scratch")
                        path.write_bytes(original)
                        self.assertEqual(reader.read_rows_into(0, 2, destination), len(destination))
                        self.assertEqual(destination, expected)
                    else:
                        self.fail("damaged payload passed checksum validation")
                self.assertEqual(len(owners), 2)
                self.assertTrue(all(owner() is None for owner in owners))
                self.assertIsNotNone(failures[0].__traceback__)

    def test_read_error_retaining_chunk_view_does_not_keep_scratch_exported(self):
        owners, failures, borrowed_views = [], [], []

        class TrackedBuffer(bytearray):
            def __init__(buffer, *args, **kwargs):
                super().__init__(*args, **kwargs)
                owners.append(weakref.ref(buffer))

        with tempfile.TemporaryDirectory(prefix="nexa-scratch-read-error-") as directory:
            path = Path(directory) / "weights.nxp"
            write_q4_matrix(path, 1, 2, 2, [[1, 2]])
            with NexaPackReader(path) as reader:
                stream = reader._stream

                class FailedRead:
                    @property
                    def closed(self):
                        return stream.closed

                    def seek(self, offset):
                        return stream.seek(offset)

                    def readinto(self, view):
                        borrowed_views.append(view)
                        raise OSError("injected payload read error")

                destination = bytearray(reader.row_bytes)
                with patch.object(fmt, "bytearray", TrackedBuffer, create=True):
                    try:
                        with patch.object(reader, "_stream", FailedRead()):
                            reader.read_rows_into(0, 1, destination)
                    except OSError as error:
                        failures.append(error)
                        self.assertIn("injected payload read error", str(error))
                        self.assertIsNone(owners[0]())
                        with self.assertRaises(ValueError):
                            len(borrowed_views[0])  # The retained view itself was explicitly released.
                        self.assertEqual(reader.read_rows_into(0, 1, destination), len(destination))
                    else:
                        self.fail("injected read error did not propagate")
                self.assertTrue(all(owner() is None for owner in owners))
                self.assertIsNotNone(failures[0].__traceback__)


class _TrackedCTypes:
    """Track real uint8 arena owners; borrowed from_address views stay unchanged."""
    def __init__(self, testcase, owners):
        class ByteType:
            def __mul__(self, count):
                array_type = ctypes.c_uint8 * count

                class ArrayFactory:
                    def __call__(self):
                        testcase.assertTrue(all(owner() is None for owner in owners),
                                            "prior workspace survived into the next allocation")
                        owner = array_type()
                        owners.append(weakref.ref(owner))
                        return owner

                    def from_address(self, address):
                        return array_type.from_address(address)

                return ArrayFactory()

        self.c_uint8 = ByteType()

    def sizeof(self, value):
        return 1 if value is self.c_uint8 else ctypes.sizeof(value)

    def __getattr__(self, name):
        return getattr(ctypes, name)


class TransformerBufferLifetimeRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which("clang") or shutil.which("cc")):
            raise unittest.SkipTest("C compiler unavailable")

    def setUp(self):
        from test_transformer_forward_regressions import random_bundle
        temporary = tempfile.TemporaryDirectory(prefix="nexa-workspace-lifetime-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "bundle"
        random_bundle(self.path, layers=1, heads=2, kv_heads=1)

    def session_types(self):
        from runtime.nexapack.transformer import TransformerSession
        from runtime.nexapack.incremental import IncrementalTransformerSession
        return (TransformerSession, IncrementalTransformerSession)

    def test_retained_report_failure_releases_workspace_before_retry_at_exact_budget(self):
        from runtime.nexapack import transformer
        transformer._load_kernels()  # Bind ctypes before instrumenting only arena allocation.
        for session_type in self.session_types():
            with self.subTest(session=session_type.__name__):
                with session_type(self.path, memory_budget="1MiB", max_sequence_length=4, reserve_bytes=123) as probe:
                    budget = probe.report()["memory"]["managed_buffers_peak_bound_bytes"] + 123
                owners, failures = [], []
                with patch.object(transformer, "ctypes", _TrackedCTypes(self, owners)):
                    with session_type(self.path, memory_budget=budget, max_sequence_length=4, reserve_bytes=123) as model:
                        model.prefill([1, 2])
                        previous = model.report()
                        try:
                            with patch.object(model, "_report", side_effect=MemoryError("injected late report failure")):
                                model.prefill([1, 2, 3, 4])
                        except MemoryError as error:
                            failures.append(error)
                            self.assertEqual(len(owners), 2)
                            self.assertTrue(all(owner() is None for owner in owners), "traceback retained workspace")
                            self.assertEqual(model.token_ids, (1, 2))
                            self.assertEqual(model.report(), previous)
                            # Retry inside the except block, with the failed traceback still alive.
                            self.assertEqual(len(model.prefill([1, 2, 3, 4])), 4)
                            self.assertEqual(model.report()["memory"]["managed_buffers_peak_bound_bytes"] + 123, budget)
                        else:
                            self.fail("late report failure did not propagate")
                self.assertEqual(len(owners), 3)
                self.assertTrue(all(owner() is None for owner in owners))
                self.assertIsNotNone(failures[0].__traceback__)

    def test_failed_reset_preserves_history_bank_and_report(self):
        for session_type in self.session_types():
            with self.subTest(session=session_type.__name__), session_type(self.path, memory_budget="1MiB",
                                                                          max_sequence_length=4) as model:
                model.prefill([1, 2])
                previous, bank = model.report(), getattr(model, "active_bank", None)
                for failure_point in ("_make_plan", "_report"):
                    with patch.object(model, failure_point, side_effect=MemoryError("injected reset preparation failure")):
                        with self.assertRaisesRegex(MemoryError, "reset preparation"):
                            model.reset()
                    self.assertEqual(model.token_ids, (1, 2))
                    self.assertEqual(model.report(), previous)
                    self.assertEqual(getattr(model, "active_bank", None), bank)
                self.assertEqual(len(model.decode(3)), model.config.vocab_size)
                model.reset()
                self.assertEqual(model.token_ids, ())
                self.assertFalse(model.report()["executed"])
                if hasattr(model, "active_bank"):
                    self.assertEqual(model.active_bank, 0)
                    self.assertEqual(model.report()["cache_length"], 0)
                    self.assertEqual(model.report()["active_bank"], 0)


if __name__ == "__main__":
    unittest.main()

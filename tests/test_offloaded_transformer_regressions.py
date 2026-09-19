"""Cold KV backing storage preserves resident-tier bytes, logits and ownership."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.planner.memory import MemoryBudgetError
from runtime.nexapack.format import NexaPackReader
from test_tiered_transformer_regressions import _TierFixture, TIERED_GOLDEN
from test_transformer_forward_regressions import random_bundle
from transformer_reference import torch_available


class _OffloadFixture(_TierFixture):
    def session(self, path=None, **options):
        from runtime.nexapack.offloaded import OffloadedTieredTransformerSession
        return OffloadedTieredTransformerSession(
            path or self.path, kv_backing_store=options.pop("kv_backing_store", self.directory / "backing"),
            memory_budget=options.pop("memory_budget", "1MiB"),
            max_sequence_length=options.pop("max_sequence_length", 8),
            page_tokens=options.pop("page_tokens", 2),
            kv_group_size=options.pop("kv_group_size", 3), **options)

    def resident(self, path=None, **options):
        return _TierFixture.session(self, path, **options)

    @staticmethod
    def refs(session):
        return [page.ref for page in session._pages if hasattr(page, "ref")]

    @staticmethod
    def packed_page(session, page):
        extent = page.layout.page_extent_bytes
        if hasattr(page, "ref"):
            # Diagnostic storage belongs to the test, not the execution budget.
            destination = (ctypes.c_uint8 * extent)()
            count = session._store.read_page(page.ref, ctypes.addressof(destination), extent)
            if count != page.ref.file_bytes:
                raise AssertionError("store read byte count differs from its immutable reference")
            return bytes(destination)
        return ctypes.string_at(page.address, extent)

    def assert_same_pages(self, offloaded, resident):
        self.assertEqual(offloaded._page_descriptors, resident._page_descriptors)
        self.assertEqual(len(offloaded._pages), len(resident._pages))
        for left, right in zip(offloaded._pages, resident._pages):
            self.assertEqual(left.codec, right.codec)
            self.assertEqual(self.packed_page(offloaded, left), self.packed_page(resident, right))
            self.assertEqual(hasattr(left, "ref"), left.codec == "q3")
            if hasattr(left, "ref"):
                self.assertEqual(left.allocation_bytes, 0)


class OffloadedNativeRegressions(_OffloadFixture):
    def test_logits_and_packed_pages_equal_resident_tiers_for_chunk_and_head_geometries(self):
        cases = ((1, 2, 2, False, 1, 1), (2, 4, 1, True, 2, 0), (2, 4, 2, False, 3, 0))
        for index, (layers, heads, kv_heads, tied, page_tokens, warm) in enumerate(cases):
            path = self.directory / f"case-{index}"
            random_bundle(path, layers=layers, heads=heads, kv_heads=kv_heads, tied=tied, seed=1801 + index)
            options = {"page_tokens": page_tokens, "hot_pages": 1, "warm_pages": warm, "max_chunk_length": 3}
            with self.subTest(case=index), self.session(path, **options) as offloaded, self.resident(path, **options) as resident:
                for index, chunk in enumerate(([1, 4, 2], [6], [8, 3, 5])):
                    method = "prefill" if index == 0 else "append"
                    self.assertEqual(getattr(offloaded, method)(chunk), getattr(resident, method)(chunk))
                    self.assert_same_pages(offloaded, resident)
                self.assertTrue(self.refs(offloaded))
                self.assertEqual(offloaded.decode(7), resident.decode(7))
                self.assert_same_pages(offloaded, resident)
                self.assertGreater(offloaded.report()["io"]["kv_backing_bytes_read"], 0)
                self.assertGreater(offloaded.report()["io"]["kv_page_reloads"], 0)
                self.assertEqual(offloaded.prefill([2, 6]), resident.prefill([2, 6]))
                self.assert_same_pages(offloaded, resident)

    def test_existing_independent_tier_golden_runs_without_torch(self):
        _, path = self.tiny_bundle()
        with self.session(path, page_tokens=1, hot_pages=1, warm_pages=1, max_chunk_length=2) as session:
            output = []
            for index, chunk in enumerate(TIERED_GOLDEN["token_chunks"]):
                output += session.prefill(chunk) if index == 0 else session.append(chunk)
                self.assertEqual(self.codecs(session), TIERED_GOLDEN["page_codecs"][index])
            self.assert_rows_close(output, TIERED_GOLDEN["logits"])
            self.assertEqual(len(self.refs(session)), 2)

    def test_eviction_and_reload_preserve_file_bytes_and_never_requantize_cold_pages(self):
        with self.session(page_tokens=1) as session:
            session.prefill([1, 3, 5])
            first = self.refs(session)[0]
            original = first.path.read_bytes()
            packed = self.packed_page(session, session._pages[0])
            session.append([7, 2])
            self.assertIs(session._pages[0].ref, first)
            self.assertEqual(first.path.read_bytes(), original)
            self.assertEqual(self.packed_page(session, session._pages[0]), packed)
            session.decode(4)
            self.assertEqual(first.path.read_bytes(), original)
            self.assertEqual(session.report()["io"]["kv_prefix_bytes_copied"], 0)

    def test_reload_slots_reuse_cold_pages_without_changing_logits_or_bytes(self):
        chunks = ([1, 4, 2], [6], [8, 3], [5])
        options = {"page_tokens": 1, "hot_pages": 1, "warm_pages": 1, "max_chunk_length": 3}
        runs = {}
        for slots in (1, 2, 8):
            store = self.directory / f"slots-{slots}"
            with self.session(kv_backing_store=store, kv_reload_slots=slots, **options) as session:
                logits = []
                for index, chunk in enumerate(chunks):
                    logits += session.prefill(chunk) if index == 0 else session.append(chunk)
                report = session.report()
                runs[slots] = (logits, [self.packed_page(session, page) for page in session._pages],
                               report["io"], report["memory"])
        with self.resident(**options) as resident:
            expected = []
            for index, chunk in enumerate(chunks):
                expected += resident.prefill(chunk) if index == 0 else resident.append(chunk)
        for slots, (logits, pages, io, memory) in runs.items():
            with self.subTest(slots=slots):
                # Slots change residency and I/O only; every kernel still reads
                # the same published bytes, in the same order, for both passes.
                self.assertEqual(logits, expected)
                self.assertEqual(pages, runs[1][1])
                self.assertEqual(memory["kv_reload_slots"], slots)
                self.assertLessEqual(memory["kv_reload_cache_entries"], slots)
                self.assertLessEqual(memory["kv_reload_cache_allocated_bytes"],
                                     memory["kv_reload_cache_capacity_bytes"])
                self.assertLessEqual(memory["managed_buffers_peak_bound_bytes"],
                                     memory["capacity_managed_buffers_bound_bytes"])
                self.assertEqual(io["kv_reload_cache_hits"] + io["kv_reload_cache_misses"],
                                 runs[1][2]["kv_reload_cache_hits"] + runs[1][2]["kv_reload_cache_misses"])
        # One slot cannot retain anything across an ascending rescan of several
        # cold pages; capacity beyond the prefix turns those misses into hits.
        self.assertEqual(runs[1][2]["kv_reload_cache_hits"], 0)
        self.assertGreater(runs[8][2]["kv_reload_cache_hits"], runs[2][2]["kv_reload_cache_hits"])
        self.assertLess(runs[8][2]["kv_page_reloads"], runs[2][2]["kv_page_reloads"])
        self.assertLess(runs[2][2]["kv_page_reloads"], runs[1][2]["kv_page_reloads"])
        self.assertLess(runs[8][2]["kv_backing_bytes_read"], runs[1][2]["kv_backing_bytes_read"])
        self.assertGreater(runs[8][2]["kv_reload_bytes_avoided"], 0)
        self.assertGreater(runs[8][3]["capacity_managed_buffers_bound_bytes"],
                           runs[1][3]["capacity_managed_buffers_bound_bytes"])

    def test_slots_beyond_the_cold_prefix_only_cost_admitted_memory(self):
        options = {"page_tokens": 1, "hot_pages": 1, "warm_pages": 1}
        reports = {}
        for slots in (1, 4):
            with self.session(kv_backing_store=self.directory / f"single-{slots}",
                              kv_reload_slots=slots, **options) as session:
                session.prefill([1, 3, 5])  # Leaves exactly one cold page behind.
                session.decode(7)
                reports[slots] = session.report()
        first, second = reports[1], reports[4]
        # A lone cold page already fits in one slot: extra capacity avoids no
        # byte and is admitted anyway, which is the cost of a larger cache.
        for name in ("kv_page_reloads", "kv_backing_bytes_read", "kv_reload_cache_hits", "kv_reload_bytes_avoided"):
            self.assertEqual(first["io"][name], second["io"][name])
        self.assertEqual(first["io"]["kv_page_reloads"], 1)
        self.assertGreater(first["io"]["kv_reload_cache_hits"], 0)
        self.assertEqual(second["memory"]["kv_reload_cache_entries"], 1)
        self.assertEqual(second["memory"]["kv_reload_cache_capacity_bytes"],
                         4 * second["memory"]["kv_reload_slot_capacity_bytes"])
        self.assertGreater(second["memory"]["kv_reserved_capacity_bytes"] +
                           second["memory"]["kv_reload_cache_capacity_bytes"],
                           first["memory"]["kv_reserved_capacity_bytes"] +
                           first["memory"]["kv_reload_cache_capacity_bytes"])

    def test_slots_are_released_when_their_page_is_retired_or_the_session_resets(self):
        with self.session(page_tokens=1, kv_reload_slots=4) as session:
            session.prefill([1, 3, 5])
            session.decode(7)
            cache = session._reload_cache
            self.assertEqual(cache.entry_count, 1)
            self.assertTrue(all(ref is None or session._store.contains(ref) for ref in cache._refs))
            session.reset()
            self.assertEqual((cache.entry_count, cache.slot_count, cache.allocated_bytes), (0, 0, 0))
            self.assertEqual(session.report()["memory"]["kv_reload_cache_allocated_bytes"], 0)
            session.prefill([1, 3, 5])
            session.decode(7)
            self.assertEqual(cache.entry_count, 1)
            session.prefill([2, 4, 6])  # Replacement retires every previous file.
            session.decode(8)
            self.assertEqual(cache.entry_count, 1)
            for ref in cache._refs:
                self.assertTrue(ref is None or session._store.contains(ref))
        self.assertEqual((cache.entry_count, cache.slot_count), (0, 0))

    def test_memory_reports_distinguish_offloaded_payload_slots_and_managed_capacity(self):
        from test_q4_paged_transformer_regressions import wide_bundle
        path = self.directory / "wide"
        wide_bundle(path)
        options = {"page_tokens": 2, "kv_group_size": 32}
        with self.session(path, **options) as offloaded, self.resident(path, **options) as resident:
            self.assertEqual(offloaded.prefill([1, 3, 5, 7, 2, 4, 6]), resident.prefill([1, 3, 5, 7, 2, 4, 6]))
            memory = offloaded.report()["memory"]
            refs = self.refs(offloaded)
            self.assertEqual(memory["kv_offloaded_page_count"], len(refs))
            self.assertEqual(memory["kv_offloaded_payload_bytes"], sum(ref.extent_bytes for ref in refs))
            self.assertEqual(memory["kv_resident_allocation_bytes"], offloaded.resident_kv_bytes)
            self.assertLess(offloaded.resident_kv_bytes, resident.resident_kv_bytes)
            self.assertEqual(memory["kv_reload_slot_bytes"], 0)  # The initial chunk has no evicted prefix.
            self.assertGreater(memory["kv_reload_slot_capacity_bytes"], 0)
            self.assertGreater(memory["kv_streaming_scratch_bytes"], 0)
            self.assertLessEqual(memory["managed_buffers_peak_bound_bytes"], memory["capacity_managed_buffers_bound_bytes"])
            self.assertLessEqual(memory["capacity_managed_buffers_bound_bytes"], memory["budget_bytes"])
            self.assertEqual(memory["kv_full_dequantized_buffer_bytes"], 0)
            self.assertIsNone(memory["peak_rss_bytes"])
            self.assertIsNone(memory["peak_vram_bytes"])
            self.assertEqual(offloaded.decode(8), resident.decode(8))
            self.assertGreater(offloaded.report()["memory"]["kv_reload_slot_bytes"], 0)

    def test_budget_rejection_happens_before_store_directory_native_and_payload(self):
        with self.session() as accepted:
            bound = accepted.report()["memory"]["capacity_managed_buffers_bound_bytes"]
        destination = self.directory / "must-not-exist"
        with patch("runtime.nexapack.transformer._load_kernels", side_effect=AssertionError("native load")), \
             patch.object(NexaPackReader, "read_rows_into", side_effect=AssertionError("payload read")):
            with self.assertRaises(MemoryBudgetError):
                self.session(memory_budget=bound - 1, kv_backing_store=destination)
        self.assertFalse(destination.exists())
        with self.session(memory_budget=bound, kv_backing_store=destination) as exact:
            exact.prefill([1, 3, 5])
            exact.append([7, 2])
            self.assertLessEqual(exact.report()["memory"]["managed_buffers_peak_bound_bytes"], bound)

    def test_reset_replacement_close_and_parallel_sessions_preserve_file_ownership(self):
        parent = self.directory / "shared"
        parent.mkdir()
        sentinel = parent / "user-data.txt"
        sentinel.write_text("unrelated user file")
        with self.session(page_tokens=1, kv_backing_store=parent) as first, \
             self.session(page_tokens=1, kv_backing_store=parent) as second:
            first.prefill([1, 3, 5, 7])
            second.prefill([2, 4, 6])
            self.assertNotEqual(first._store.directory, second._store.directory)
            first_refs, second_refs = self.refs(first), self.refs(second)
            first.reset()
            self.assertFalse(any(ref.path.exists() for ref in first_refs))
            self.assertTrue(all(ref.path.exists() for ref in second_refs))
            self.assertEqual(first.token_ids, ())
            self.assertEqual(first.report()["memory"]["kv_offloaded_page_count"], 0)
            self.assertEqual(first.report()["memory"]["kv_offloaded_payload_bytes"], 0)
            first.prefill([1, 3, 5])
            replaced = self.refs(first)
            first.prefill([7])
            self.assertFalse(any(ref.path.exists() for ref in replaced))
            second.decode(8)
            final_refs = self.refs(second)
        first.close()
        second.close()
        self.assertFalse(any(ref.path.exists() for ref in final_refs))
        self.assertEqual(sentinel.read_text(), "unrelated user file")

    def test_cli_backing_store_requires_age_policy_and_reports_io(self):
        directory = self.directory / "cli-backing"
        command = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path), "--tokens", "1,3,5",
                   "--decode-tokens", "7", "--kv-cache", "--kv-page-tokens", "1", "--kv-policy", "age",
                   "--kv-hot-pages", "1", "--kv-warm-pages", "1", "--kv-group-size", "3",
                   "--kv-backing-store", str(directory), "--prefill-chunk-size", "2",
                   "--memory-budget", "1MiB", "--max-sequence-length", "8"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["token_ids"], [1, 3, 5, 7])
        self.assertEqual(report["memory"]["kv_offloaded_page_count"], 2)
        self.assertGreater(report["io"]["kv_backing_bytes_read"], 0)
        self.assertGreater(report["io"]["kv_backing_bytes_written"], 0)
        self.assertFalse(report["validation"]["verified"])
        self.assertFalse(any(path.is_file() for path in directory.rglob("*")))
        invalid = command.copy()
        index = invalid.index("--kv-policy")
        del invalid[index:index + 2]
        failed = subprocess.run(invalid, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("--kv-", failed.stderr)
        self.assertNotIn("Traceback", failed.stderr)

    def test_cli_reload_slots_report_reuse_totals_and_require_a_backing_store(self):
        base = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path), "--tokens", "1,3,5",
                "--decode-tokens", "7,2", "--kv-cache", "--kv-page-tokens", "1", "--kv-policy", "age",
                "--kv-hot-pages", "1", "--kv-warm-pages", "1", "--kv-group-size", "3",
                "--prefill-chunk-size", "2", "--memory-budget", "1MiB", "--max-sequence-length", "8"]
        totals = {}
        for slots in (1, 4):
            command = base + ["--kv-backing-store", str(self.directory / f"cli-slots-{slots}"),
                              "--kv-reload-slots", str(slots)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["token_ids"], [1, 3, 5, 7, 2])
            self.assertEqual(report["kv_backing_store"]["reload_slots"], slots)
            totals[slots] = report["run_totals"]
        self.assertGreater(totals[4]["kv_reload_cache_hits"], totals[1]["kv_reload_cache_hits"])
        self.assertLess(totals[4]["kv_page_reloads"], totals[1]["kv_page_reloads"])
        self.assertGreater(totals[4]["kv_reload_bytes_avoided"], totals[1]["kv_reload_bytes_avoided"])
        self.assertEqual(totals[4]["kv_reload_slot_admissions"] + totals[4]["kv_reload_slot_evictions"],
                         totals[4]["kv_reload_cache_misses"])
        store = ["--kv-backing-store", str(self.directory / "cli-rejected")]
        for arguments in (["--kv-reload-slots", "2"], store + ["--kv-reload-slots", "0"]):
            rejected = subprocess.run(base + arguments, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("--kv-reload-slots", rejected.stderr)
            self.assertNotIn("Traceback", rejected.stderr)


@unittest.skipUnless(torch_available(), "Optional PyTorch reference is unavailable")
class OffloadedOracleRegressions(_OffloadFixture):
    def test_cli_optional_oracle_keeps_exact_tier_chunk_history(self):
        source, path = self.tiny_bundle()
        command = [sys.executable, str(ROOT / "tools/nexa_run.py"), str(path), "--tokens", "1,3,5",
                   "--decode-tokens", "7", "--kv-cache", "--kv-page-tokens", "1", "--kv-policy", "age",
                   "--kv-group-size", "3", "--kv-backing-store", str(self.directory / "verified"),
                   "--prefill-chunk-size", "2", "--memory-budget", "1MiB", "--verify",
                   "--reference-checkpoint", str(source)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertTrue(report["validation"]["verified"])
        self.assertEqual(report["validation"]["reference"], "pytorch_decoded_q4_kv_age_tiers")
        self.assertEqual(report["validation"]["token_chunks"], [[1, 3], [5], [7]])
        self.assertEqual(report["validation"]["execution_error"]["max_abs_error"], 0.0)


if __name__ == "__main__":
    unittest.main()

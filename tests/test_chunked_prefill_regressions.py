"""Chunk-bounded activation storage with complete-context F32 KV attention."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.planner.memory import MemoryBudgetError
from compiler.model_config import ModelConfig
from runtime.nexapack.bundle import ModelBundleReader, write_model_bundle
from runtime.nexapack.format import NexaPackReader
from runtime.nexapack.incremental import IncrementalTransformerSession
from runtime.nexapack.transformer import TransformerSession
from test_transformer_forward_regressions import random_bundle


def digest(rows):
    result = hashlib.sha256()
    for row in rows:
        for value in row:
            result.update(struct.pack("<f", value))
    return result.hexdigest()


class ChunkedPrefillRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which("clang") or shutil.which("cc")):
            raise unittest.SkipTest("C compiler unavailable")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nexa-chunked-prefill-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "bundle"
        self.config, _ = random_bundle(self.path, layers=2, heads=4, kv_heads=2, tied=False, seed=927)

    def session(self, **kwargs):
        return IncrementalTransformerSession(self.path, memory_budget=kwargs.pop("memory_budget", "1MiB"),
                                             max_sequence_length=12, tile_rows=3, **kwargs)

    def baseline(self, tokens):
        with TransformerSession(self.path, memory_budget="1MiB", max_sequence_length=12, tile_rows=3) as session:
            return session.prefill(tokens)

    def assert_rows_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for row, reference in zip(actual, expected):
            self.assertEqual(len(row), len(reference))
            for value, target in zip(row, reference):
                self.assertTrue(math.isfinite(value) and math.isfinite(target))
                self.assertLessEqual(abs(value - target), 1e-5 + 1e-4 * abs(target))

    def cli(self, *arguments):
        return subprocess.run([sys.executable, str(ROOT / "tools/nexa_run.py"), str(self.path),
                               "--tile-rows", "3", *map(str, arguments)],
                              capture_output=True, text=True, timeout=30)

    def alignment_fixture(self):
        """A valid shape where replanning a smaller graph increases first-fit extent."""
        config = ModelConfig("AlignmentBound", 16, 4, 10, 2, 2, 2, 32)
        sources = {}
        for name, shape in config.required_tensor_shapes().items():
            if len(shape) == 1:
                sources[name] = lambda shape=shape: iter([[1.0] * shape[0]])
            else:
                sources[name] = lambda shape=shape: (
                    [((row * shape[1] + column) % 13 - 6) / 10 for column in range(shape[1])]
                    for row in range(shape[0]))
        path = self.directory / "alignment-bundle"
        write_model_bundle(path, config, sources, group_size=2, block_rows=2)
        return path

    def test_long_prompt_chunks_match_full_forward_and_actual_chunk_shapes(self):
        tokens = [1, 3, 5, 7, 2, 8, 4, 6, 10, 9, 11, 12]
        expected = self.baseline(tokens)
        for chunk_size in (1, 2, 5):
            with self.subTest(chunk_size=chunk_size), self.session(max_chunk_length=chunk_size) as session:
                initial = session.report()
                self.assertEqual(initial["memory"]["attention_score_scratch_bytes"], 12 * 4)
                self.assertEqual(initial["kv_cache_plan"]["capacity"], 12)
                self.assertEqual(initial["kv_cache_plan"]["banks"], 2)
                actual = []
                for start in range(0, len(tokens), chunk_size):
                    chunk = tokens[start:start + chunk_size]
                    actual.extend(session.prefill(chunk) if start == 0 else session.append(chunk))
                    report = session.report()
                    length = start + len(chunk)
                    tensors = {item["name"]: item for item in report["model_ir"]["tensors"]}
                    self.assertEqual(tensors["tokens"]["shape"], [len(chunk)])
                    self.assertEqual(tensors["logits"]["shape"], [len(chunk), self.config.vocab_size])
                    self.assertEqual(report["memory"]["attention_score_scratch_bytes"], length * 4)
                    self.assertEqual(report["processed_tokens"], len(chunk))
                    self.assertEqual(report["cache_position_offset"], start)
                    self.assertEqual(report["io"]["kv_prefix_bytes_copied"], 0)
                self.assertEqual(session.token_ids, tuple(tokens))
                self.assert_rows_close(actual, expected)

    def test_chunk_capacity_admits_a_budget_that_full_context_activations_reject(self):
        tokens = [1, 3, 5, 7, 2, 8, 4, 6, 10, 9, 11, 12]
        with self.session(max_chunk_length=2) as chunked, self.session() as full:
            small, large = chunked.report(), full.report()
        budget = small["memory"]["managed_buffers_peak_bound_bytes"]
        self.assertLess(budget, large["memory"]["managed_buffers_peak_bound_bytes"])
        self.assertEqual(small["memory"]["persistent_kv_bytes"], large["memory"]["persistent_kv_bytes"])
        self.assertEqual(small["memory"]["kv_arena_allocation_bytes"], large["memory"]["kv_arena_allocation_bytes"])
        self.assertLess(small["memory"]["arena_extent_bytes"], large["memory"]["arena_extent_bytes"])
        with patch.object(NexaPackReader, "read_rows_into", side_effect=AssertionError("unexpected payload read")):
            with self.assertRaises(MemoryBudgetError):
                self.session(memory_budget=budget)
            with self.assertRaises(MemoryBudgetError):
                self.session(memory_budget=budget - 1, max_chunk_length=2)
        with self.session(memory_budget=budget, max_chunk_length=2) as session:
            actual = []
            for start in range(0, len(tokens), 2):
                chunk = tokens[start:start + 2]
                actual.extend(session.prefill(chunk) if start == 0 else session.append(chunk))
                report = session.report()
                self.assertLessEqual(report["memory"]["managed_buffers_peak_bound_bytes"], budget)
            self.assertEqual(session.report()["memory"]["attention_score_scratch_bytes"], len(tokens) * 4)
        self.assert_rows_close(actual, self.baseline(tokens))

    def test_preflight_covers_full_attention_scratch_even_for_one_token_chunks(self):
        with self.session(max_chunk_length=1) as session:
            report = session.report()
        memory, allocations = report["memory"], report["memory_plan"]["allocations"]
        self.assertEqual(allocations["__attention"]["size_bytes"], 12 * 4)
        self.assertEqual(allocations["logits"]["size_bytes"], self.config.vocab_size * 4)
        self.assertEqual(memory["persistent_kv_bytes"],
                         2 * self.config.num_hidden_layers * 2 * 12 *
                         self.config.num_key_value_heads * self.config.head_dim * 4)
        accounted = (memory["arena_allocation_bytes"] + memory["reader_scratch_capacity_bytes"]
                     + memory["kv_arena_allocation_bytes"])
        self.assertEqual(accounted, memory["managed_buffers_peak_bound_bytes"])
        with self.session(memory_budget=accounted, max_chunk_length=1) as session:
            session.prefill([1])
            for token in range(1, 12):
                session.decode(token)
            final = session.report()
            self.assertEqual(final["memory"]["attention_score_scratch_bytes"], 12 * 4)
            self.assertLessEqual(final["memory"]["managed_buffers_peak_bound_bytes"], accounted)

    def test_smaller_chunks_cannot_exceed_the_accepted_capacity_budget_due_to_alignment(self):
        path = self.alignment_fixture()
        tokens = [index % 16 for index in range(32)]
        options = {"max_sequence_length": 32, "max_chunk_length": 9, "tile_rows": 7}
        with IncrementalTransformerSession(path, memory_budget="1MiB", **options) as session:
            capacity = session.report()["memory"]
        budget = capacity["managed_buffers_peak_bound_bytes"]
        with TransformerSession(path, memory_budget="1MiB", max_sequence_length=32, tile_rows=7) as reference:
            expected = reference.prefill(tokens)
        with IncrementalTransformerSession(path, memory_budget=budget, **options) as session:
            actual = []
            for start in range(0, 32, 8):
                chunk = tokens[start:start + 8]
                actual.extend(session.prefill(chunk) if start == 0 else session.append(chunk))
                memory = session.report()["memory"]
                self.assertLessEqual(memory["arena_extent_bytes"], capacity["arena_extent_bytes"])
                self.assertLessEqual(memory["managed_buffers_peak_bound_bytes"], budget)
            self.assertEqual(session.cache_length, 32)
            self.assert_rows_close(actual, expected)

    def test_baseline_smaller_prefill_and_decode_fit_the_exact_capacity_budget(self):
        path = self.alignment_fixture()
        tokens = list(range(9))
        options = {"max_sequence_length": 9, "tile_rows": 7}
        with TransformerSession(path, memory_budget="1MiB", **options) as reference:
            capacity = reference.report()["memory"]
            expected = reference.prefill(tokens)
        budget = capacity["managed_buffers_peak_bound_bytes"]
        with TransformerSession(path, memory_budget=budget, **options) as session:
            actual = session.prefill(tokens[:8])
            memory = session.report()["memory"]
            self.assertLessEqual(memory["arena_extent_bytes"], capacity["arena_extent_bytes"])
            self.assertLessEqual(memory["managed_buffers_peak_bound_bytes"], budget)
            actual.append(session.decode(tokens[-1]))
            self.assertEqual(session.token_ids, tuple(tokens))
            self.assertLessEqual(session.report()["memory"]["managed_buffers_peak_bound_bytes"], budget)
            self.assert_rows_close(actual, expected)

    def test_invalid_chunk_capacity_rejects_before_payload_or_cache_allocation(self):
        with patch.object(NexaPackReader, "read_rows_into") as packed:
            with patch.object(ModelBundleReader, "read_f32_into") as norm:
                with patch.object(IncrementalTransformerSession, "_allocate_cache") as allocate:
                    for size in (False, True, 0, -1, 1.5, "2", 13):
                        with self.subTest(size=size), self.assertRaises(ValueError):
                            self.session(max_chunk_length=size)
                    packed.assert_not_called()
                    norm.assert_not_called()
                    allocate.assert_not_called()

    def test_oversized_prefill_and_append_preserve_committed_state_and_allow_decode(self):
        with self.session(max_chunk_length=2) as session:
            session.prefill([1, 3])
            prior, bank = session.report(), session.active_bank
            with patch.object(NexaPackReader, "read_rows_into") as packed:
                with patch.object(ModelBundleReader, "read_f32_into") as norm:
                    for operation in (session.prefill, session.append):
                        with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, "chunk"):
                            operation([5, 7, 9])
                        self.assertEqual(session.token_ids, (1, 3))
                        self.assertEqual(session.active_bank, bank)
                        self.assertEqual(session.report(), prior)
                    packed.assert_not_called()
                    norm.assert_not_called()
            self.assert_rows_close([session.decode(5)], self.baseline([1, 3, 5])[-1:])

    def test_cli_chunked_prefill_includes_every_logit_in_digest_and_prediction(self):
        arguments = ("--tokens", "1,3,5,7,2", "--decode-tokens", "8", "--kv-cache",
                     "--prefill-chunk-size", 2)
        result = self.cli(*arguments, "--include-logits")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        expected = self.baseline([1, 3, 5, 7, 2, 8])
        self.assert_rows_close(report["logits"], expected)
        self.assertEqual(report["token_ids"], [1, 3, 5, 7, 2, 8])
        self.assertEqual(report["logits_shape"], [6, self.config.vocab_size])
        self.assertEqual(report["logits_scope"], "full_context")
        self.assertEqual(report["logits_sha256"], digest(report["logits"]))
        self.assertEqual(report["last_chunk_logits_sha256"], digest(report["logits"][-1:]))
        self.assertEqual(report["next_token_id"], max(range(self.config.vocab_size), key=lambda i: expected[-1][i]))
        self.assertEqual([step["mode"] for step in report["steps"]],
                         ["prefill", "prefill_append", "prefill_append", "decode_incremental"])
        self.assertEqual([step["processed_tokens"] for step in report["steps"]], [2, 2, 1, 1])
        self.assertEqual([step["sequence_length"] for step in report["steps"]], [2, 4, 5, 6])
        self.assertEqual(report["run_totals"]["processed_tokens"], 6)
        compact = self.cli(*arguments)
        self.assertEqual(compact.returncode, 0, compact.stderr)
        without_logits = json.loads(compact.stdout)
        self.assertNotIn("logits", without_logits)
        self.assertEqual(without_logits["logits_sha256"], report["logits_sha256"])
        self.assertEqual(without_logits["next_token_id"], report["next_token_id"])

    def test_default_chunk_limit_and_cli_baseline_are_preserved(self):
        tokens = [1, 3, 5, 7, 2, 8, 4, 6, 10, 9, 11, 12]
        with self.session() as session:
            self.assertEqual(session.max_chunk_length, session.max_sequence_length)
            self.assertEqual(len(session.prefill(tokens)), len(tokens))
        baseline = self.cli("--tokens", "1,3,5,7,2", "--decode-tokens", "8", "--include-logits")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        original = json.loads(baseline.stdout)
        self.assertFalse(original["persistent_kv_cache"])
        self.assertEqual([step["mode"] for step in original["steps"]], ["prefill", "decode_recompute"])
        clamped = self.cli("--tokens", "1,3,5,7,2", "--decode-tokens", "8", "--kv-cache",
                           "--prefill-chunk-size", 100, "--include-logits")
        self.assertEqual(clamped.returncode, 0, clamped.stderr)
        report = json.loads(clamped.stdout)
        self.assertEqual(report["max_chunk_length"], 6)
        self.assertEqual(report["logits_sha256"], original["logits_sha256"])
        self.assert_rows_close(report["logits"], original["logits"])

    def test_cli_requires_kv_and_positive_chunk_size(self):
        for arguments in (("--prefill-chunk-size", 2),
                          ("--kv-cache", "--prefill-chunk-size", 0),
                          ("--kv-cache", "--prefill-chunk-size", -2)):
            with self.subTest(arguments=arguments):
                result = self.cli("--tokens", "1,3", *arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--prefill-chunk-size", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()

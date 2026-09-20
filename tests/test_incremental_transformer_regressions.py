"""Incremental F32 KV execution, atomic cache commits, and numerical proofs."""
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

from compiler.importers.llama import import_llama_checkpoint
from compiler.planner.memory import MemoryBudgetError
from model_checkpoint_fixture import create_checkpoint
from runtime.nexapack.bundle import ModelBundleReader
from runtime.nexapack.format import NexaPackReader
from runtime.nexapack.transformer import TransformerSession
from test_transformer_forward_regressions import random_bundle
from transformer_reference import TorchLlamaReference, compare_logits, load_bundle_weights, torch_available


class _IncrementalFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which('clang') or shutil.which('cc')):
            raise unittest.SkipTest('C compiler unavailable')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nexa-incremental-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / 'bundle'
        self.config, _ = random_bundle(self.path, layers=2, heads=4, kv_heads=2, tied=False, seed=401)

    def session(self, path=None, **kwargs):
        from runtime.nexapack.incremental import IncrementalTransformerSession
        return IncrementalTransformerSession(path or self.path,
                                            memory_budget=kwargs.pop('memory_budget', '1MiB'),
                                            max_sequence_length=kwargs.pop('max_sequence_length', 8), **kwargs)

    def baseline(self, tokens, path=None):
        with TransformerSession(path or self.path, memory_budget='1MiB', max_sequence_length=8, tile_rows=3) as reference:
            return reference.prefill(list(tokens))

    def assert_rows_close(self, actual, expected, atol=1e-5, rtol=1e-4):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertEqual(len(left), len(right))
            for value, reference in zip(left, right):
                self.assertTrue(math.isfinite(value) and math.isfinite(reference))
                self.assertLessEqual(abs(value - reference), atol + rtol * abs(reference))

    def corrupt_second_layer(self):
        with ModelBundleReader(self.path) as bundle:
            entry = bundle.manifest['tensors']['model.layers.1.self_attn.q_proj.weight']
            path = self.path / entry['path']
        original = path.read_bytes()
        with NexaPackReader(path) as reader:
            offset = reader.metadata['blocks'][0]['offset']
        damaged = bytearray(original)
        damaged[offset] ^= 1
        path.write_bytes(damaged)
        return path, original


class IncrementalNativeRegressions(_IncrementalFixture):
    def test_prefill_append_and_decode_match_full_prefix_forward(self):
        tokens = [1, 3, 5, 7, 2, 8, 4]
        expected = self.baseline(tokens)
        with self.session(tile_rows=3) as session:
            actual = session.prefill(tokens[:2])
            actual += session.append(tokens[2:5])
            actual += [session.decode(tokens[5])]
            actual += session.append(tokens[6:])
            self.assertEqual(session.token_ids, tuple(tokens))
            self.assert_rows_close(actual, expected)

    def test_chunk_boundaries_heads_layers_and_weight_aliases(self):
        cases = [(1, 2, 2, False), (2, 4, 1, True), (2, 4, 2, False)]
        tokens = [1, 4, 2, 6, 8, 3]
        for index, (layers, heads, kv_heads, tied) in enumerate(cases):
            path = self.directory / f'case-{index}'
            random_bundle(path, layers=layers, heads=heads, kv_heads=kv_heads, tied=tied, seed=801 + index)
            expected = self.baseline(tokens, path)
            for first, following in ((1, (1, 1, 3)), (2, (2, 2)), (4, (2,))):
                with self.subTest(case=index, first=first, following=following), self.session(path, tile_rows=5) as session:
                    actual = session.prefill(tokens[:first])
                    cursor = first
                    for count in following:
                        actual += session.append(tokens[cursor:cursor + count])
                        cursor += count
                    self.assertEqual(cursor, len(tokens))
                    self.assert_rows_close(actual, expected)

    def test_frozen_pytorch_logits_without_pytorch(self):
        golden = json.loads((ROOT / 'tests/fixtures/transformer_tiny_golden.json').read_text())
        source, bundle_path = self.directory / 'source', self.directory / 'tiny-bundle'
        create_checkpoint(source, include_tied_head=True)
        import_llama_checkpoint(source, bundle_path, **golden['import_settings'])
        with ModelBundleReader(bundle_path) as bundle:
            hashes = {name: hashlib.sha256((bundle_path / entry['path']).read_bytes()).hexdigest()
                      for name, entry in bundle.manifest['tensors'].items()}
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(fingerprint, golden['pack_sha256'])
        with self.session(bundle_path) as session:
            tokens = golden['token_ids']
            actual = session.prefill(tokens[:1])
            actual += session.append(tokens[1:3])
            actual += [session.decode(tokens[3])]
        self.assert_rows_close(actual, golden['logits'], golden['atol'], golden['rtol'])

    def test_failed_append_after_first_layer_write_preserves_cache_and_report(self):
        from runtime.nexapack.transformer import _load_kernels
        with self.session() as session:
            session.prefill([1, 3])
            report = session.report()
            kernels = _load_kernels()
            path, original = self.corrupt_second_layer()
            with patch.object(kernels, 'nexa_causal_gqa_attention_cached',
                              wraps=kernels.nexa_causal_gqa_attention_cached) as attention:
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    session.append([2, 4])
                self.assertEqual(attention.call_count, 1)
                self.assertEqual(attention.call_args.args[6:8], (2, 2))
            self.assertEqual(session.token_ids, (1, 3))
            self.assertEqual(session.report(), report)
            path.write_bytes(original)
            actual = [session.decode(5), session.decode(7)]
            self.assert_rows_close(actual, self.baseline([1, 3, 5, 7])[-2:])

    def test_failed_prefill_replacement_keeps_old_bank_available(self):
        with self.session() as session:
            session.prefill([1, 3, 5])
            report = session.report()
            bank = session.active_bank
            path, original = self.corrupt_second_layer()
            with self.assertRaises(ValueError):
                session.prefill([2, 4, 6, 7])
            self.assertEqual(session.token_ids, (1, 3, 5))
            self.assertEqual(session.report(), report)
            self.assertEqual(session.active_bank, bank)
            path.write_bytes(original)
            resumed = session.append([8, 9])
            self.assert_rows_close(resumed, self.baseline([1, 3, 5, 8, 9])[-2:])

    def test_late_python_failure_never_commits_cache_history_or_bank(self):
        for operation in ('append', 'prefill'):
            with self.subTest(operation=operation), self.session() as session:
                session.prefill([1, 3])
                report, bank = session.report(), session.active_bank
                with patch.object(session, '_report', side_effect=MemoryError('injected report allocation failure')):
                    with self.assertRaises(MemoryError):
                        getattr(session, operation)([2, 4, 6])
                self.assertEqual(session.token_ids, (1, 3))
                self.assertEqual(session.cache_length, 2)
                self.assertEqual(session.active_bank, bank)
                self.assertEqual(session.report(), report)
                self.assert_rows_close([session.decode(5)], self.baseline([1, 3, 5])[-1:])

    def test_decode_uses_one_token_projections_embedding_and_position_offset(self):
        from runtime.nexapack.transformer import _load_kernels
        with self.session(tile_rows=3) as session:
            session.prefill([1, 3, 5])
            kernels = _load_kernels()
            original_open = ModelBundleReader.open_packed
            embedding_reads = []
            def tracked_open(bundle, name):
                reader = original_open(bundle, name)
                if name == 'model.embed_tokens.weight':
                    original_read = reader.read_rows_into
                    def read_rows_into(start, count, destination):
                        embedding_reads.append((start, count))
                        return original_read(start, count, destination)
                    reader.read_rows_into = read_rows_into
                return reader
            with patch.object(ModelBundleReader, 'open_packed', tracked_open):
                with patch.object(kernels, 'nexa_q4_matmul', wraps=kernels.nexa_q4_matmul) as matmul:
                    with patch.object(kernels, 'nexa_rope_offset', wraps=kernels.nexa_rope_offset) as rope:
                        with patch.object(kernels, 'nexa_causal_gqa_attention_cached',
                                          wraps=kernels.nexa_causal_gqa_attention_cached) as attention:
                            actual = session.decode(7)
            self.assertEqual(embedding_reads, [(7, 1)])
            self.assertGreater(matmul.call_count, 0)
            self.assertTrue(all(call.args[2] == 1 for call in matmul.call_args_list))
            self.assertEqual(rope.call_count, 2 * self.config.num_hidden_layers)
            self.assertTrue(all(call.args[2] == 1 and call.args[6] == 3 for call in rope.call_args_list))
            self.assertEqual(attention.call_count, self.config.num_hidden_layers)
            self.assertTrue(all(call.args[6:8] == (3, 1) for call in attention.call_args_list))
            self.assert_rows_close([actual], self.baseline([1, 3, 5, 7])[-1:])
            report = session.report()
            self.assertTrue(report['persistent_kv_cache'])
            self.assertEqual(report['decode_strategy'], 'incremental_kv')
            self.assertEqual(report['sequence_length'], 4)
            self.assertEqual(report['context_length'], 4)
            self.assertEqual(report['processed_tokens'], 1)
            self.assertEqual(report['cache_position_offset'], 3)
            token_desc = next(tensor for tensor in report['model_ir']['tensors'] if tensor['name'] == 'tokens')
            self.assertEqual(token_desc['shape'], [1])
            self.assertEqual(report['io']['embedding_rows_read'], 1)
            self.assertEqual(report['io']['kv_prefix_bytes_copied'], 0)
            self.assertEqual(report['io']['kv_bytes_written'],
                             2 * self.config.num_hidden_layers * self.config.num_key_value_heads * self.config.head_dim * 4)

    def test_two_cache_banks_have_exact_budgeted_payload(self):
        from runtime.nexapack.incremental import IncrementalTransformerSession
        with self.session(max_sequence_length=7, reserve_bytes=321) as session:
            memory = session.report()['memory']
            expected_payload = (2 * 2 * self.config.num_hidden_layers * 7
                                * self.config.num_key_value_heads * self.config.head_dim * 4)
            self.assertEqual(memory['persistent_kv_bytes'], expected_payload)
            self.assertGreaterEqual(memory['kv_arena_extent_bytes'], expected_payload)
            self.assertGreaterEqual(memory['kv_arena_allocation_bytes'], memory['kv_arena_extent_bytes'])
            expected_peak = (memory['arena_allocation_bytes'] + memory['kv_arena_allocation_bytes']
                             + memory['reader_scratch_capacity_bytes'])
            self.assertEqual(memory['managed_buffers_peak_bound_bytes'], expected_peak)
            self.assertLessEqual(expected_peak + 321, memory['budget_bytes'])
            self.assertEqual(session.cache_length, 0)
            arena = session._cache_arena
            session.prefill([1, 3])
            self.assertEqual(session.cache_length, 2)
            session.decode(5)
            self.assertEqual(session.cache_length, 3)
            self.assertEqual(session.report()['memory']['persistent_kv_bytes'], expected_payload)
            self.assertEqual(session.report()['memory']['kv_arena_allocation_bytes'], memory['kv_arena_allocation_bytes'])
            self.assertIs(session._cache_arena, arena)
        required = expected_peak + 321
        with patch.object(IncrementalTransformerSession, '_allocate_cache', side_effect=AssertionError('KV allocated')):
            with self.assertRaises(MemoryBudgetError):
                self.session(max_sequence_length=7, reserve_bytes=321, memory_budget=required - 1)
        with self.session(max_sequence_length=7, reserve_bytes=321, memory_budget=required) as exact:
            self.assertEqual(exact.report()['memory']['managed_buffers_peak_bound_bytes'] + 321, required)

    def test_insufficient_budget_rejected_before_payload_or_native_kernel(self):
        from runtime.nexapack.incremental import IncrementalTransformerSession
        with patch.object(NexaPackReader, 'read_rows_into', side_effect=AssertionError('Q4 payload read')):
            with patch.object(ModelBundleReader, 'read_f32_into', side_effect=AssertionError('norm payload read')):
                with patch('runtime.nexapack.transformer._load_kernels', side_effect=AssertionError('kernel loaded')):
                    with patch.object(IncrementalTransformerSession, '_allocate_cache', side_effect=AssertionError('KV allocated')):
                        with self.assertRaises(MemoryBudgetError):
                            self.session(memory_budget='1KiB')

    def test_replacement_reset_capacity_invalid_ids_and_close(self):
        with self.session(max_sequence_length=4) as session:
            with self.assertRaises(ValueError):
                session.decode(1)
            session.prefill([1, 2, 3])
            self.assert_rows_close(session.prefill([4]), self.baseline([4]))
            self.assertEqual(session.token_ids, (4,))
            self.assert_rows_close(session.append([5, 6]), self.baseline([4, 5, 6])[-2:])
            for values in ([], [True], [-1], [self.config.vocab_size], [1.5], [1, 2]):
                before = session.report()
                with self.subTest(values=values), self.assertRaises(ValueError):
                    session.append(values)
                self.assertEqual(session.token_ids, (4, 5, 6))
                self.assertEqual(session.report(), before)
            session.decode(7)
            with self.assertRaises(ValueError):
                session.decode(8)
            self.assertEqual(session.token_ids, (4, 5, 6, 7))
            session.reset()
            self.assertEqual(session.token_ids, ())
            self.assert_rows_close(session.prefill([1, 2]), self.baseline([1, 2]))
        session.close()
        for operation in (lambda: session.prefill([1]), lambda: session.decode(1), lambda: session.append([1]), session.reset):
            with self.assertRaises(ValueError):
                operation()

    def test_cli_incremental_keeps_native_default_and_rejects_invalid_eos(self):
        command = [sys.executable, str(ROOT / 'tools/nexa_run.py'), str(self.path), '--tokens', '1,3',
                   '--decode-tokens', '5,7', '--memory-budget', '1MiB', '--include-logits']
        baseline = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        incremental = subprocess.run(command + ['--kv-cache'], capture_output=True, text=True, timeout=60)
        self.assertEqual(incremental.returncode, 0, incremental.stderr)
        original, cached = json.loads(baseline.stdout), json.loads(incremental.stdout)
        self.assertFalse(original['persistent_kv_cache'])
        self.assertTrue(cached['persistent_kv_cache'])
        self.assert_rows_close(cached['logits'], original['logits'])
        self.assertEqual(cached['token_ids'], [1, 3, 5, 7])
        self.assertEqual(cached['logits_scope'], 'full_context')
        self.assertEqual(cached['logits_shape'], [4, self.config.vocab_size])
        digest = hashlib.sha256()
        for row in cached['logits']:
            for value in row:
                digest.update(struct.pack('<f', value))
        self.assertEqual(cached['logits_sha256'], digest.hexdigest())
        self.assertEqual(cached['steps'][-1]['processed_tokens'], 1)
        bad = subprocess.run(command + ['--kv-cache', '--eos-token', str(self.config.vocab_size)],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(bad.returncode, 1)
        self.assertIn('EOS token', bad.stderr)


@unittest.skipUnless(torch_available(), 'Optional PyTorch reference is unavailable')
class IncrementalTorchRegressions(_IncrementalFixture):
    def test_native_chunks_and_decode_match_independent_torch_kv(self):
        config, weights = load_bundle_weights(self.path)
        oracle = TorchLlamaReference(config, weights)
        with self.session() as session:
            self.assertTrue(compare_logits(session.prefill([1, 3]), oracle.prefill([1, 3]))['passed'])
            chunk = session.append([5, 7, 2])
            expected = [oracle.decode(token).tolist() for token in (5, 7, 2)]
            self.assertTrue(compare_logits(chunk, expected)['passed'])
            self.assertTrue(compare_logits(session.decode(8), oracle.decode(8))['passed'])
            self.assertEqual(session.token_ids, oracle.token_ids)


if __name__ == '__main__':
    unittest.main()

"""Paged F32 KV: numerical equivalence, bounded residency and atomic failures."""
import ctypes
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

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


class _PagedFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which('clang') or shutil.which('cc')):
            raise unittest.SkipTest('C compiler unavailable')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nexa-paged-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / 'bundle'
        self.config, _ = random_bundle(self.path, layers=2, heads=4, kv_heads=2, tied=False, seed=501)

    def session(self, path=None, **kwargs):
        from runtime.nexapack.paged import PagedTransformerSession
        return PagedTransformerSession(path or self.path,
                                       memory_budget=kwargs.pop('memory_budget', '1MiB'),
                                       max_sequence_length=kwargs.pop('max_sequence_length', 8),
                                       page_tokens=kwargs.pop('page_tokens', 2), **kwargs)

    def baseline(self, tokens, path=None):
        with TransformerSession(path or self.path, memory_budget='1MiB',
                                max_sequence_length=12, tile_rows=3) as reference:
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

    @staticmethod
    def track_page(page, references):
        references.append((weakref.ref(page), weakref.ref(page.arena)))
        return page

    def assert_pages_released(self, references):
        gc.collect()
        for owner_ref, arena_ref in references:
            owner = owner_ref()
            if owner is not None:
                self.assertIsNone(owner.arena)
                self.assertEqual(owner.address, 0)
            self.assertIsNone(arena_ref(), 'failed transaction retained a physical page allocation')

    @staticmethod
    def page_sizes(config, page_tokens):
        buffer_bytes = page_tokens * config.num_key_value_heads * config.head_dim * 4
        aligned = (buffer_bytes + 63) // 64 * 64
        return 2 * config.num_hidden_layers * buffer_bytes, 2 * config.num_hidden_layers * aligned + 63


class PagedNativeRegressions(_PagedFixture):
    def test_chunks_across_page_boundaries_match_full_prefix(self):
        cases = [(1, 2, 2, False), (2, 4, 1, True), (2, 4, 2, False)]
        tokens = [1, 4, 2, 6, 8, 3, 5]
        for index, (layers, heads, kv_heads, tied) in enumerate(cases):
            path = self.directory / f'case-{index}'
            random_bundle(path, layers=layers, heads=heads, kv_heads=kv_heads,
                          tied=tied, seed=1001 + index)
            expected = self.baseline(tokens, path)
            for page_tokens, chunks in ((1, (1, 2, 1, 3)), (2, (1, 3, 2, 1)),
                                        (3, (2, 2, 3)), (16, (3, 1, 3))):
                with self.subTest(case=index, page_tokens=page_tokens), self.session(
                        path, page_tokens=page_tokens, max_chunk_length=3, tile_rows=3) as session:
                    cursor, actual = chunks[0], session.prefill(tokens[:chunks[0]])
                    for count in chunks[1:]:
                        actual += session.append(tokens[cursor:cursor + count])
                        cursor += count
                    self.assertEqual(session.token_ids, tuple(tokens))
                    self.assertEqual(session.resident_page_count, math.ceil(len(tokens) / page_tokens))
                    self.assert_rows_close(actual, expected)

    def test_frozen_pytorch_logits_without_torch(self):
        golden = json.loads((ROOT / 'tests/fixtures/transformer_tiny_golden.json').read_text())
        source, bundle_path = self.directory / 'source', self.directory / 'tiny-bundle'
        create_checkpoint(source, include_tied_head=True)
        import_llama_checkpoint(source, bundle_path, **golden['import_settings'])
        with ModelBundleReader(bundle_path) as bundle:
            hashes = {name: hashlib.sha256((bundle_path / entry['path']).read_bytes()).hexdigest()
                      for name, entry in bundle.manifest['tensors'].items()}
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(fingerprint, golden['pack_sha256'])
        with self.session(bundle_path, page_tokens=2, max_chunk_length=2) as session:
            tokens = golden['token_ids']
            actual = session.prefill(tokens[:1])
            actual += session.append(tokens[1:3])
            actual += [session.decode(tokens[3])]
        self.assert_rows_close(actual, golden['logits'], golden['atol'], golden['rtol'])

    def test_residency_is_lazy_and_existing_page_addresses_stay_stable(self):
        from runtime.nexapack.incremental import IncrementalTransformerSession
        payload, allocation = self.page_sizes(self.config, 2)
        with self.session(max_chunk_length=3) as session:
            initial = session.report()['memory']
            self.assertEqual(session.resident_page_count, 0)
            self.assertEqual(session.resident_kv_bytes, 0)
            self.assertEqual(session._pages, [])
            self.assertEqual(initial['persistent_kv_bytes'], 0)
            self.assertEqual(initial['kv_resident_allocation_bytes'], 0)
            self.assertEqual(initial['kv_page_allocation_bytes'], allocation)
            self.assertEqual(initial['kv_reserved_capacity_bytes'], (4 + 2) * allocation)
            self.assertGreater(initial['capacity_managed_buffers_bound_bytes'],
                               initial['managed_buffers_peak_bound_bytes'])
            original_allocate = session._allocate_page
            with patch.object(session, '_allocate_page', wraps=original_allocate) as allocate:
                session.prefill([1])
                first = session._pages[0]
                address = first.address
                self.assertEqual(address % 64, 0)
                self.assertEqual(allocate.call_count, 1)
                session.decode(3)
                self.assertEqual(allocate.call_count, 1)
                session.append([5, 7, 2])
                self.assertEqual(allocate.call_count, 3)
            self.assertIs(session._pages[0], first)
            self.assertEqual(session._pages[0].address, address)
            self.assertEqual(len({page.address for page in session._pages}), 3)
            self.assertEqual(session.resident_page_count, 3)
            self.assertEqual(session.resident_kv_bytes, 3 * allocation)
            memory = session.report()['memory']
            self.assertEqual(memory['persistent_kv_bytes'], 3 * payload)
            self.assertEqual(memory['kv_resident_page_count'], 3)
            self.assertEqual(memory['kv_resident_allocation_bytes'], 3 * allocation)
            self.assertEqual(memory['kv_transaction_peak_allocation_bytes'], 3 * allocation)
            self.assertEqual(memory['managed_buffers_peak_bound_bytes'],
                             memory['arena_allocation_bytes'] + memory['reader_scratch_capacity_bytes']
                             + 3 * allocation)
            self.assertLessEqual(memory['managed_buffers_peak_bound_bytes'],
                                 memory['capacity_managed_buffers_bound_bytes'])
        with self.session(max_chunk_length=3) as paged, IncrementalTransformerSession(
                self.path, memory_budget='1MiB', max_sequence_length=8, max_chunk_length=3) as two_banks:
            paged.prefill([1])
            self.assertLess(paged.resident_kv_bytes, two_banks.report()['memory']['kv_arena_allocation_bytes'])

    def test_replacement_accounts_for_old_and_staged_pages_then_releases_old(self):
        _, allocation = self.page_sizes(self.config, 2)
        with self.session(max_chunk_length=3) as session:
            session.prefill([1, 3, 5])
            session.append([7, 2, 4])
            old = list(session._pages)
            old_references = []
            for page in old:
                self.track_page(page, old_references)
            actual = session.prefill([6, 8, 9])
            self.assert_rows_close(actual, self.baseline([6, 8, 9]))
            self.assertEqual(session.resident_page_count, 2)
            self.assertTrue(all(page not in old for page in session._pages))
            memory = session.report()['memory']
            self.assertEqual(memory['kv_transaction_peak_allocation_bytes'], 5 * allocation)
            self.assertEqual(memory['kv_resident_allocation_bytes'], 2 * allocation)
            self.assert_pages_released(old_references)

    def test_exact_capacity_budget_and_one_byte_less_reject_before_allocation(self):
        from runtime.nexapack.paged import PagedTransformerSession
        with self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137) as session:
            memory = session.report()['memory']
            required = memory['capacity_managed_buffers_bound_bytes'] + 137
            _, allocation = self.page_sizes(self.config, 3)
            self.assertEqual(memory['kv_reserved_capacity_bytes'], (3 + 1) * allocation)
        with patch.object(NexaPackReader, 'read_rows_into', side_effect=AssertionError('payload read')):
            with patch.object(ModelBundleReader, 'read_f32_into', side_effect=AssertionError('norm read')):
                with patch('runtime.nexapack.transformer._load_kernels', side_effect=AssertionError('kernel loaded')):
                    with patch.object(PagedTransformerSession, '_allocate_page', side_effect=AssertionError('page allocated')):
                        with self.assertRaises(MemoryBudgetError):
                            self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137,
                                         memory_budget=required - 1)
                        with self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137,
                                          memory_budget=required) as exact:
                            self.assertEqual(exact.resident_page_count, 0)
        with self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137,
                          memory_budget=required) as exact:
            exact.prefill([1, 2])
            for chunk in ([3, 4], [5, 6], [7, 8]):
                exact.append(chunk)
            self.assert_rows_close(exact.prefill([2, 4]), self.baseline([2, 4]))
            self.assertLessEqual(exact.report()['memory']['managed_buffers_peak_bound_bytes'] + 137, required)

    def test_decode_computes_one_token_and_reads_one_embedding(self):
        from runtime.nexapack.transformer import _load_kernels
        with self.session(max_chunk_length=3, tile_rows=3) as session:
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
                        with patch.object(kernels, 'nexa_causal_gqa_attention_paged',
                                          wraps=kernels.nexa_causal_gqa_attention_paged) as attention:
                            actual = session.decode(7)
            self.assertEqual(embedding_reads, [(7, 1)])
            self.assertGreater(matmul.call_count, 0)
            self.assertTrue(all(call.args[2] == 1 for call in matmul.call_args_list))
            self.assertEqual(rope.call_count, 2 * self.config.num_hidden_layers)
            self.assertTrue(all(call.args[2] == 1 and call.args[6] == 3 for call in rope.call_args_list))
            self.assertEqual(attention.call_count, self.config.num_hidden_layers)
            self.assertTrue(all(call.args[8:10] == (3, 1) for call in attention.call_args_list))
            self.assertTrue(all(call.args[3] == 2 and call.args[5] == 2 and call.args[6] == 2
                                for call in attention.call_args_list))
            self.assert_rows_close([actual], self.baseline([1, 3, 5, 7])[-1:])
            report = session.report()
            self.assertTrue(report['paged_kv_cache'])
            self.assertTrue(report['persistent_kv_cache'])
            self.assertEqual(report['decode_strategy'], 'paged_incremental_kv')
            self.assertEqual(report['processed_tokens'], 1)
            self.assertEqual(report['context_length'], 4)
            self.assertEqual(report['sequence_length'], 4)
            self.assertEqual(report['cache_position_offset'], 3)
            token_desc = next(item for item in report['model_ir']['tensors'] if item['name'] == 'tokens')
            self.assertEqual(token_desc['shape'], [1])
            self.assertEqual(report['io']['embedding_rows_read'], 1)
            self.assertEqual(report['io']['kv_prefix_bytes_copied'], 0)
            self.assertEqual(report['io']['kv_bytes_written'],
                             2 * self.config.num_hidden_layers * self.config.num_key_value_heads * self.config.head_dim * 4)
            self.assertEqual(report['memory']['kv_page_table_bytes'], 4 * ctypes.sizeof(ctypes.c_void_p))

    def test_checksum_failure_preserves_partial_and_boundary_prefixes_with_traceback(self):
        for prefix in ([1], [1, 3]):
            with self.subTest(prefix=prefix), self.session(max_chunk_length=3) as session:
                session.prefill(prefix)
                before, old = session.report(), list(session._pages)
                addresses = [page.address for page in old]
                references, saved = [], []
                original_allocate = session._allocate_page
                def allocate():
                    return self.track_page(original_allocate(), references)
                path, original = self.corrupt_second_layer()
                original_attention = session._run_attention
                try:
                    with patch.object(session, '_allocate_page', side_effect=allocate):
                        with patch.object(session, '_run_attention', wraps=original_attention) as attention:
                            try:
                                session.append([2, 4, 6])
                            except ValueError as exc:
                                self.assertIn('checksum', str(exc))
                                saved.append(exc)
                            else:
                                self.fail('corrupt second-layer weights were accepted')
                            self.assertEqual(attention.call_count, 1)
                    self.assertEqual(session.token_ids, tuple(prefix))
                    self.assertEqual(session.report(), before)
                    self.assertEqual(session._pages, old)
                    self.assertEqual([page.address for page in session._pages], addresses)
                    self.assertGreater(len(references), 0)
                    self.assert_pages_released(references)
                finally:
                    path.write_bytes(original)
                # Keep the original exception/traceback alive during a retry.
                self.assertIsNotNone(saved[0].__traceback__)
                self.assert_rows_close(session.append([8, 9, 5]), self.baseline(prefix + [8, 9, 5])[-3:])

    def test_failed_replacement_preserves_prior_pages_and_frees_staging(self):
        with self.session(max_chunk_length=4) as session:
            session.prefill([1, 3, 5])
            before, old, references, saved = session.report(), list(session._pages), [], []
            original_allocate = session._allocate_page
            def allocate():
                return self.track_page(original_allocate(), references)
            path, original = self.corrupt_second_layer()
            try:
                with patch.object(session, '_allocate_page', side_effect=allocate):
                    try:
                        session.prefill([2, 4, 6, 8])
                    except ValueError as exc:
                        self.assertIn('checksum', str(exc))
                        saved.append(exc)
                    else:
                        self.fail('corrupt prefill replacement was accepted')
                self.assertEqual(session.token_ids, (1, 3, 5))
                self.assertEqual(session.report(), before)
                self.assertEqual(session._pages, old)
                self.assertEqual(len(references), 2)
                self.assert_pages_released(references)
            finally:
                path.write_bytes(original)
            self.assertIsNotNone(saved[0].__traceback__)
            self.assert_rows_close([session.decode(7)], self.baseline([1, 3, 5, 7])[-1:])

    def test_page_allocation_failure_frees_partial_transaction_before_payload(self):
        for mode in ('append', 'prefill'):
            with self.subTest(mode=mode), self.session(max_chunk_length=4) as session:
                session.prefill([1, 3])
                before, old, references, saved = session.report(), list(session._pages), [], []
                original_allocate = session._allocate_page
                def allocate():
                    if references:
                        raise MemoryError('injected second page allocation failure')
                    return self.track_page(original_allocate(), references)
                with patch.object(session, '_allocate_page', side_effect=allocate):
                    with patch.object(NexaPackReader, 'read_rows_into', side_effect=AssertionError('payload read before allocation')):
                        with patch.object(ModelBundleReader, 'read_f32_into', side_effect=AssertionError('norm read before allocation')):
                            try:
                                getattr(session, mode)([2, 4, 6, 8])
                            except MemoryError as exc:
                                saved.append(exc)
                            else:
                                self.fail('injected allocation failure was not raised')
                self.assertEqual(session._pages, old)
                self.assertEqual(session.token_ids, (1, 3))
                self.assertEqual(session.report(), before)
                self.assertEqual(len(references), 1)
                self.assert_pages_released(references)
                self.assertIsNotNone(saved[0].__traceback__)
                self.assert_rows_close([session.decode(5)], self.baseline([1, 3, 5])[-1:])

    def test_report_failure_rolls_back_and_releases_pages_with_traceback(self):
        for mode in ('append', 'prefill'):
            with self.subTest(mode=mode), self.session(max_chunk_length=3) as session:
                session.prefill([1])
                before, old, references, saved = session.report(), list(session._pages), [], []
                original_allocate = session._allocate_page
                def allocate():
                    return self.track_page(original_allocate(), references)
                with patch.object(session, '_allocate_page', side_effect=allocate):
                    with patch.object(session, '_report', side_effect=MemoryError('injected report failure')):
                        try:
                            getattr(session, mode)([2, 4, 6])
                        except MemoryError as exc:
                            saved.append(exc)
                        else:
                            self.fail('report failure was not raised')
                self.assertEqual(session._pages, old)
                self.assertEqual(session.token_ids, (1,))
                self.assertEqual(session.report(), before)
                self.assertGreater(len(references), 0)
                self.assert_pages_released(references)
                self.assertIsNotNone(saved[0].__traceback__)
                self.assert_rows_close(session.append([8, 9, 5]), self.baseline([1, 8, 9, 5])[-3:])

    def test_reset_is_atomic_and_releases_every_page_on_success_or_close(self):
        with self.session(max_chunk_length=3) as session:
            session.prefill([1, 3, 5])
            before, old = session.report(), list(session._pages)
            references = []
            for page in old:
                self.track_page(page, references)
            with patch.object(session, '_report', side_effect=MemoryError('reset report failed')):
                with self.assertRaises(MemoryError):
                    session.reset()
            self.assertEqual(session._pages, old)
            self.assertEqual(session.token_ids, (1, 3, 5))
            self.assertEqual(session.report(), before)
            session.reset()
            self.assertEqual(session.token_ids, ())
            self.assertEqual(session.cache_length, 0)
            self.assertEqual(session.resident_page_count, 0)
            self.assertEqual(session.resident_kv_bytes, 0)
            self.assertEqual(session.report()['memory']['persistent_kv_bytes'], 0)
            self.assert_pages_released(references)
            self.assert_rows_close(session.prefill([2, 4]), self.baseline([2, 4]))
            references = []
            for page in session._pages:
                self.track_page(page, references)
        session.close()
        self.assertEqual(session.resident_page_count, 0)
        self.assertEqual(session.resident_kv_bytes, 0)
        self.assert_pages_released(references)
        for operation in (lambda: session.prefill([1]), lambda: session.append([1]),
                          lambda: session.decode(1), session.reset):
            with self.assertRaises(ValueError):
                operation()

    def test_invalid_ids_chunk_context_and_page_sizes_do_not_mutate_state(self):
        for page_tokens in (0, -1, True, 1.5):
            with self.subTest(page_tokens=page_tokens), self.assertRaises(ValueError):
                self.session(page_tokens=page_tokens)
        with self.session(max_sequence_length=4, max_chunk_length=2) as session:
            with self.assertRaises(ValueError):
                session.decode(1)
            session.prefill([1, 3])
            before, old = session.report(), list(session._pages)
            for values in ([], [True], [-1], [self.config.vocab_size], [1.5], [1, 2, 3]):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    session.append(values)
                self.assertEqual(session._pages, old)
                self.assertEqual(session.token_ids, (1, 3))
                self.assertEqual(session.report(), before)
            session.append([5, 7])
            with self.assertRaises(ValueError):
                session.decode(9)
            self.assertEqual(session.cache_length, 4)
            self.assert_rows_close(session.prefill([2]), self.baseline([2]))

    def test_cli_paged_chunks_match_baseline_and_reject_invalid_options(self):
        command = [sys.executable, str(ROOT / 'tools/nexa_run.py'), str(self.path),
                   '--tokens', '1,3,5,7,2', '--decode-tokens', '8,4',
                   '--memory-budget', '1MiB', '--include-logits']
        baseline = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        paged = subprocess.run(command + ['--kv-cache', '--kv-page-tokens', '2', '--prefill-chunk-size', '2'],
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(paged.returncode, 0, paged.stderr)
        original, cached = json.loads(baseline.stdout), json.loads(paged.stdout)
        self.assertFalse(original['persistent_kv_cache'])
        self.assertTrue(cached['paged_kv_cache'])
        self.assertEqual(cached['decode_strategy'], 'paged_incremental_kv')
        self.assert_rows_close(cached['logits'], original['logits'])
        self.assertEqual(cached['token_ids'], [1, 3, 5, 7, 2, 8, 4])
        self.assertEqual(cached['run_totals']['processed_tokens'], 7)
        self.assertEqual([step['processed_tokens'] for step in cached['steps']], [2, 2, 1, 1, 1])
        for options in (['--kv-page-tokens', '2'], ['--kv-cache', '--kv-page-tokens', '0'],
                        ['--kv-cache', '--kv-page-tokens', '-1']):
            invalid = subprocess.run(command + options, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn('--kv-page-tokens', invalid.stderr)


@unittest.skipUnless(torch_available(), 'Optional PyTorch reference is unavailable')
class PagedTorchRegressions(_PagedFixture):
    def test_pages_and_chunks_match_independent_torch_incremental_cache(self):
        config, weights = load_bundle_weights(self.path)
        oracle = TorchLlamaReference(config, weights)
        with self.session(page_tokens=3, max_chunk_length=3) as session:
            self.assertTrue(compare_logits(session.prefill([1, 3]), oracle.prefill([1, 3]))['passed'])
            chunk = session.append([5, 7, 2])
            expected = [oracle.decode(token).tolist() for token in (5, 7, 2)]
            self.assertTrue(compare_logits(chunk, expected)['passed'])
            self.assertTrue(compare_logits(session.decode(8), oracle.decode(8))['passed'])
            self.assertEqual(session.token_ids, oracle.token_ids)


if __name__ == '__main__':
    unittest.main()

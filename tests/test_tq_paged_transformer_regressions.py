"""TQ paged KV: independent numerical oracle, bounded buffers and atomicity."""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
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
from test_paged_transformer_regressions import _PagedFixture
from test_q4_paged_transformer_regressions import wide_bundle
from test_transformer_forward_regressions import random_bundle
from tq_reference import TQReference, TRANSFORM_ID
from transformer_reference import (TorchLlamaReference, compare_logits, llama_forward,
                                   load_bundle_weights, torch_available, verify_bundle_forward)


TQ_KV_GOLDEN = json.loads(r'''
{
  "generated_by": "TorchLlamaReference with independent Python TQ and explicit centroids; native codec/executor not used",
  "torch_version": "2.11.0",
  "pack_sha256": "aeabc843eb9a1c4c72368351bb87ae681b5856c1e03bfd1462191c235b699df3",
  "token_ids": [
    1,
    3,
    5,
    7
  ],
  "atol": 1e-05,
  "rtol": 0.0001,
  "kv_codec": "tq",
  "kv_bits": 3,
  "kv_seed": -7,
  "kv_codebook_f32le": "000060bf000020bf0000c0be000000be0000003e0000c03e0000203f0000603f",
  "logits": [
    [
      -2.560523509979248,
      -2.3949270248413086,
      -1.910469889640808,
      -1.6466130018234253,
      -1.378482460975647,
      0.040246278047561646,
      0.27739381790161133,
      0.5861895680427551,
      0.42559918761253357,
      0.7052122950553894,
      0.9264561533927917,
      1.2868643999099731,
      1.6409575939178467,
      1.7086985111236572,
      1.1769839525222778,
      1.3911421298980713
    ],
    [
      -2.6437020301818848,
      -2.6721301078796387,
      -1.0909779071807861,
      -1.1642513275146484,
      -1.2640317678451538,
      -0.18759609758853912,
      -0.3219052851200104,
      -0.45164647698402405,
      0.1000567376613617,
      -0.03546846657991409,
      -0.16283413767814636,
      0.6223146915435791,
      0.6678069233894348,
      0.3501339852809906,
      1.0441826581954956,
      0.9359380602836609
    ],
    [
      1.704504132270813,
      1.6846286058425903,
      1.0087018013000488,
      0.9380342364311218,
      0.7840198874473572,
      0.028314625844359398,
      -0.07117222994565964,
      -0.2763967216014862,
      -1.6704657077789307,
      -1.8737846612930298,
      -1.9620634317398071,
      -2.1683576107025146,
      -2.3168606758117676,
      -2.3542447090148926,
      -1.5987234115600586,
      -1.6544488668441772
    ],
    [
      1.515447974205017,
      1.5251827239990234,
      0.7807809710502625,
      0.7663235068321228,
      0.6478428840637207,
      0.01160422246903181,
      -0.044529400765895844,
      -0.20628909766674042,
      -1.7308369874954224,
      -1.8963879346847534,
      -1.9402474164962769,
      -2.0935068130493164,
      -2.18790864944458,
      -2.1995694637298584,
      -1.637894868850708,
      -1.652206301689148
    ]
  ],
  "errors": {
    "execution_error": {
      "passed": true,
      "atol": 1e-05,
      "rtol": 0.0001,
      "max_abs_error": 0.0,
      "mean_abs_error": 0.0,
      "max_rel_error": 0.0
    },
    "quantization_error": {
      "max_abs_error": 0.44354209303855896,
      "mean_abs_error": 0.16574561974266544,
      "max_rel_error": 10.905645773105643
    },
    "kv_quantization_error": {
      "max_abs_error": 1.1729209870100021,
      "mean_abs_error": 0.2571114568709163,
      "max_rel_error": 1.8405291716709304
    },
    "combined_quantization_error": {
      "max_abs_error": 1.3402653187513351,
      "mean_abs_error": 0.29195758256537374,
      "max_rel_error": 7.0620701460384225
    }
  }
}
''')


def linear_codebook(bits=3):
    levels = 1 << bits
    return struct.pack('<' + 'f' * levels, *[(2 * index + 1 - levels) / levels
                                          for index in range(levels)]).hex()


class TQIndependentCodecRegressions(unittest.TestCase):
    def test_explicit_indices_norm_rotation_and_canonical_zero(self):
        oracle = TQReference(4, 3, -7, linear_codebook())
        self.assertEqual(oracle.encode([0, -0.0, 0, 0]), b'TQ02' + bytes(6))
        self.assertEqual(oracle.decode(b'TQ02' + bytes(6)), [0.0] * 4)
        # With seed -7, D=[-1,-1,+1,+1]. H([-1,0,0,0])/2 gives
        # four exact midpoint coordinates -0.5; ties choose centroid index 1.
        self.assertEqual(oracle.encode([1, 0, 0, 0]), bytes.fromhex('545130320000803f4902'))
        self.assertEqual(oracle.decode(bytes.fromhex('545130320000803f4902')), [1.25, 0, 0, 0])

    def test_strict_parameters_stored_book_and_malformed_records(self):
        for dim, bits, seed, book in ((3, 3, 42, linear_codebook()),
                                      (4, True, 42, linear_codebook()),
                                      (4, 3, 1 << 31, linear_codebook()),
                                      (4, 3, 42, None), (4, 3, 42, '00' * 32),
                                      (4, 3, 42, linear_codebook().upper())):
            with self.subTest(dim=dim, bits=bits, seed=seed), self.assertRaises(ValueError):
                TQReference(dim, bits, seed, book)
        oracle = TQReference(4, 3, 42, linear_codebook())
        for record in (b'', b'TQ01' + bytes(6), b'TQ02' + struct.pack('<f', -0.0) + bytes(2),
                       b'TQ02' + struct.pack('<f', math.nan) + bytes(2),
                       b'TQ02' + bytes(4) + b'\x01\x00',
                       b'TQ02' + struct.pack('<f', 1) + b'\x00\xf0'):
            with self.subTest(record=record), self.assertRaises(ValueError):
                oracle.decode(record)


class _TQFixture(_PagedFixture):
    def session(self, path=None, **kwargs):
        kwargs.setdefault('kv_codec', 'tq')
        return super().session(path, **kwargs)

    def packed_prefix(self, session, length=None):
        length = session.cache_length if length is None else length
        metadata = session.report()['kv_cache_plan']
        row_bytes = 8 + (session.config.head_dim * metadata['bits'] + 7) // 8
        token_bytes = session.config.num_key_value_heads * row_bytes
        stride = (session.page_tokens * token_bytes + 63) // 64 * 64
        records = {}
        for position in range(length):
            page_index, within = divmod(position, session.page_tokens)
            for layer in range(session.config.num_hidden_layers):
                for kind in range(2):
                    address = session._pages[page_index].address + (2 * layer + kind) * stride + within * token_bytes
                    for head in range(session.config.num_key_value_heads):
                        records[layer, kind, position, head] = ctypes.string_at(address + head * row_bytes, row_bytes)
        return records

    def tiny_bundle(self):
        settings = json.loads((ROOT / 'tests/fixtures/transformer_tiny_golden.json').read_text())
        source, path = self.directory / 'source', self.directory / 'tiny'
        create_checkpoint(source, include_tied_head=True)
        import_llama_checkpoint(source, path, **settings['import_settings'])
        return source, path


class TQPagedNativeRegressions(_TQFixture):
    def test_logits_match_across_pages_chunks_layers_heads_bits_and_seeds(self):
        tokens = [1, 4, 2, 6, 8, 3, 5]
        cases = ((1, 2, 2, False, 1, -(1 << 31)), (2, 4, 1, True, 3, 42),
                 (2, 4, 2, False, 8, (1 << 31) - 1))
        for index, (layers, heads, kv_heads, tied, bits, seed) in enumerate(cases):
            path = self.directory / f'case-{index}'
            random_bundle(path, layers=layers, heads=heads, kv_heads=kv_heads, tied=tied, seed=1701 + index)
            options = dict(kv_bits=bits, kv_seed=seed)
            with self.session(path, page_tokens=8, **options) as complete:
                expected = complete.prefill(tokens)
                options['kv_codebook_f32le'] = complete.report()['kv_cache_plan']['codebook_f32le']
            for pages, chunks in ((1, (1, 3, 2, 1)), (2, (3, 1, 3)), (5, (2, 2, 3))):
                with self.subTest(case=index, pages=pages), self.session(
                        path, page_tokens=pages, max_chunk_length=3, **options) as session:
                    actual, cursor = session.prefill(tokens[:chunks[0]]), chunks[0]
                    for count in chunks[1:]:
                        actual += session.append(tokens[cursor:cursor + count])
                        cursor += count
                    self.assert_rows_close(actual, expected)
                    self.assertEqual(session.token_ids, tuple(tokens))

    def test_frozen_oracle_generated_logits_without_torch(self):
        _, path = self.tiny_bundle()
        with ModelBundleReader(path) as bundle:
            hashes = {name: hashlib.sha256((path / entry['path']).read_bytes()).hexdigest()
                      for name, entry in bundle.manifest['tensors'].items()}
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(fingerprint, TQ_KV_GOLDEN['pack_sha256'])
        options = {name: TQ_KV_GOLDEN[name] for name in ('kv_bits', 'kv_seed', 'kv_codebook_f32le')}
        for pages in (1, 3):
            with self.subTest(pages=pages), self.session(path, page_tokens=pages, max_chunk_length=2, **options) as session:
                tokens = TQ_KV_GOLDEN['token_ids']
                actual = session.prefill(tokens[:1]) + session.append(tokens[1:3]) + [session.decode(tokens[3])]
                self.assert_rows_close(actual, TQ_KV_GOLDEN['logits'], TQ_KV_GOLDEN['atol'], TQ_KV_GOLDEN['rtol'])

    def test_prefix_is_immutable_and_new_records_match_independent_encoder(self):
        from runtime.nexapack.tq_kv import _load_tq_kernels
        with self.session(page_tokens=3, max_chunk_length=3, kv_seed=-7) as session:
            session.prefill([1, 3])
            prefix, address = self.packed_prefix(session), session._pages[0].address
            book = session.report()['kv_cache_plan']['codebook_f32le']
            oracle = TQReference(self.config.head_dim, 3, -7, book)
            kernels, counts = _load_tq_kernels(), []
            original = kernels.tq_quantize_tq02
            def quantize(*args):
                _, values, value_count, packed, packed_count, rows, _, _ = args
                expected = b''.join(oracle.encode([values[index] for index in range(start, start + self.config.head_dim)])
                                    for start in range(0, value_count, self.config.head_dim))
                status = original(*args)
                self.assertEqual(status, 0)
                self.assertEqual(ctypes.string_at(packed, packed_count), expected)
                counts.append(rows)
                return status
            with patch.object(kernels, 'tq_quantize_tq02', side_effect=quantize):
                with patch.object(kernels, 'nexa_causal_gqa_attention_paged_tq',
                                  wraps=kernels.nexa_causal_gqa_attention_paged_tq) as attention:
                    session.append([5, 7, 2])
            self.assertEqual(self.packed_prefix(session, 2), prefix)
            self.assertEqual(session._pages[0].address, address)
            self.assertEqual(sum(counts), 2 * self.config.num_hidden_layers * 3 * self.config.num_key_value_heads)
            self.assertEqual(attention.call_count, self.config.num_hidden_layers)
            prefix = self.packed_prefix(session)
            session.decode(8)
            self.assertEqual(self.packed_prefix(session, 5), prefix)

    def test_exported_codebook_is_reused_and_zero_rows_are_canonical(self):
        path = self.directory / 'zero'
        config, _ = random_bundle(path, layers=2, heads=4, kv_heads=1, zero_layers=True)
        with self.session(path, max_chunk_length=2) as first:
            self.assertIsNone(first.report()['kv_cache_plan']['codebook_f32le'])
            expected = first.prefill([1, 3]) + [first.decode(5)]
            book = first.report()['kv_cache_plan']['codebook_f32le']
            records = self.packed_prefix(first)
            self.assertTrue(records)
            for record in records.values():
                self.assertEqual(record, b'TQ02' + bytes(4 + (config.head_dim * 3 + 7) // 8))
            first.reset()
            self.assertEqual(first.resident_page_count, 0)
            self.assertEqual(first.report()['kv_cache_plan']['codebook_f32le'], book)
        with self.session(path, page_tokens=3, max_chunk_length=2, kv_codebook_f32le=book) as imported:
            self.assertEqual(imported.report()['kv_cache_plan']['codebook_f32le'], book)
            self.assert_rows_close(imported.prefill([1, 3]) + [imported.decode(5)], expected)
            self.assertEqual(self.packed_prefix(imported), records)

    def test_context_scratch_and_physical_page_accounting_preserve_other_codecs(self):
        with self.session(page_tokens=3, max_chunk_length=3) as session:
            before = session.report()['memory']
            self.assertEqual(before['kv_tq_context_bytes'], 0)
            self.assertGreater(before['kv_tq_context_reserved_bytes'], 0)
            session.prefill([1, 3])
            session.append([5, 7])
            report, memory = session.report(), session.report()['memory']
            row_bytes = 8 + (self.config.head_dim * 3 + 7) // 8
            token_bytes = 2 * self.config.num_hidden_layers * self.config.num_key_value_heads * row_bytes
            self.assertEqual(report['kv_codec'], 'tq')
            self.assertEqual(report['kv_cache_plan']['codec_id'], 'TQ_MSE_SRHT')
            self.assertEqual(report['kv_cache_plan']['transform_id'], TRANSFORM_ID)
            self.assertEqual(memory['kv_encoded_bytes_per_token'], token_bytes)
            self.assertEqual(memory['kv_valid_prefix_bytes'], 4 * token_bytes)
            self.assertEqual(memory['kv_tq_vector_scratch_bytes'], 4 * self.config.head_dim)
            self.assertEqual(memory['kv_tq_accumulator_bytes'], 8 * self.config.head_dim)
            self.assertGreater(memory['kv_tq_context_bytes'], 0)
            self.assertLessEqual(memory['kv_tq_context_bytes'], memory['kv_tq_context_reserved_bytes'])
            self.assertEqual(memory['kv_full_dequantized_buffer_bytes'], 0)
            self.assertLessEqual(memory['managed_buffers_peak_bound_bytes'], memory['capacity_managed_buffers_bound_bytes'])
        path = self.directory / 'wide'
        wide_bundle(path)
        for codec, expected in (('f32', 8255), ('q4', 1343), ('q3', 1087), ('tq', 1087)):
            options = {'kv_group_size': 32} if codec in ('q4', 'q3') else {}
            with self.subTest(codec=codec), self.session(path, kv_codec=codec, page_tokens=16,
                                                       max_chunk_length=2, tile_rows=32, **options) as session:
                session.prefill([1, 3])
                self.assertEqual(session.resident_kv_bytes, expected)

    def test_exact_budget_rejects_before_native_context_payload_or_pages(self):
        from runtime.nexapack.paged import PagedTransformerSession
        with self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137) as session:
            required = session.report()['memory']['capacity_managed_buffers_bound_bytes'] + 137
        invalid = [{'kv_bits': value} for value in (True, 0, 9, 1.5)]
        invalid += [{'kv_seed': value} for value in (True, -(1 << 31) - 1, 1 << 31)]
        invalid += [{'kv_group_size': 3}, {'kv_codebook_f32le': '00' * 32}, {'memory_budget': required - 1}]
        with patch.object(NexaPackReader, 'read_rows_into', side_effect=AssertionError('payload read')):
            with patch.object(ModelBundleReader, 'read_f32_into', side_effect=AssertionError('norm read')):
                with patch('runtime.nexapack.tq_kv._load_tq_kernels', side_effect=AssertionError('TQ native loaded')):
                    with patch.object(PagedTransformerSession, '_allocate_page', side_effect=AssertionError('page allocated')):
                        for options in invalid:
                            with self.subTest(options=options), self.assertRaises((ValueError, MemoryBudgetError)):
                                self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137, **options)
                        with self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137, memory_budget=required) as exact:
                            self.assertEqual(exact.resident_kv_bytes, 0)
        with self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137, memory_budget=required) as exact:
            exact.prefill([1, 2])
            for chunk in ([3, 4], [5, 6], [7, 8]):
                exact.append(chunk)
            exact.prefill([2, 4])
            self.assertLessEqual(exact.report()['memory']['managed_buffers_peak_bound_bytes'] + 137, required)

    def test_checksum_and_report_failures_release_pages_with_saved_tracebacks(self):
        for mode, failure in (('append', 'checksum'), ('prefill', 'checksum'),
                              ('append', 'report'), ('prefill', 'report')):
            with self.subTest(mode=mode, failure=failure), self.session(max_chunk_length=3) as session:
                session.prefill([1])
                before, prefix, old = session.report(), self.packed_prefix(session), list(session._pages)
                references, saved = [], []
                original_allocate = session._allocate_page
                def allocate():
                    return self.track_page(original_allocate(), references)
                if failure == 'checksum':
                    path, original = self.corrupt_second_layer()
                try:
                    with patch.object(session, '_allocate_page', side_effect=allocate):
                        guard = (patch.object(session, '_report', side_effect=MemoryError('TQ report failed'))
                                 if failure == 'report' else patch.object(session, '_run_attention', wraps=session._run_attention))
                        with guard:
                            try:
                                getattr(session, mode)([2, 4, 6])
                            except (ValueError, MemoryError) as error:
                                saved.append(error)
                            else:
                                self.fail('injected TQ failure did not occur')
                    self.assertEqual(session.report(), before)
                    self.assertEqual(session._pages, old)
                    self.assertEqual(self.packed_prefix(session), prefix)
                    self.assertTrue(references)
                    self.assert_pages_released(references)
                finally:
                    if failure == 'checksum':
                        path.write_bytes(original)
                self.assertIsNotNone(saved[0].__traceback__)
                actual = session.append([8, 9, 5])
                with self.session(page_tokens=4) as fresh:
                    self.assert_rows_close(actual, fresh.prefill([1, 8, 9, 5])[-3:])

    def test_real_nonfinite_norm_after_key_write_rolls_back_and_retries(self):
        from runtime.nexapack.tq_kv import _load_tq_kernels
        with self.session(max_chunk_length=2) as session:
            session.prefill([1, 3])
            before, prefix, old = session.report(), self.packed_prefix(session), list(session._pages)
            references, saved, statuses, key_payloads = [], [], [], []
            original_allocate = session._allocate_page
            native = _load_tq_kernels().tq_quantize_tq02
            def allocate():
                return self.track_page(original_allocate(), references)
            def quantize(*args):
                if len(statuses) == 1:
                    self.assertTrue(key_payloads[0].startswith(b'TQ02'))
                    args[1][0] = math.inf
                result = native(*args)
                if not statuses:
                    key_payloads.append(ctypes.string_at(args[3], args[4]))
                statuses.append(result)
                return result
            with patch.object(session, '_allocate_page', side_effect=allocate):
                with patch.object(_load_tq_kernels(), 'tq_quantize_tq02', side_effect=quantize):
                    try:
                        session.append([5])
                    except ArithmeticError as error:
                        self.assertIn('tq_quantize_tq02', str(error))
                        self.assertIn('status -4', str(error))
                        saved.append(error)
                    else:
                        self.fail('TQ native quantizer accepted a nonfinite norm')
            self.assertEqual(statuses, [0, -4])
            self.assertEqual(session.report(), before)
            self.assertEqual(session._pages, old)
            self.assertEqual(self.packed_prefix(session), prefix)
            self.assertEqual(len(references), 1)
            self.assert_pages_released(references)
            self.assertIsNotNone(saved[0].__traceback__)
            actual = session.append([8, 9])
            with self.session(page_tokens=4) as fresh:
                self.assert_rows_close(actual, fresh.prefill([1, 3, 8, 9])[-2:])

    def test_cli_defaults_seed_greedy_and_option_rejection(self):
        command = [sys.executable, str(ROOT / 'tools/nexa_run.py'), str(self.path), '--tokens', '1,3',
                   '--memory-budget', '1MiB', '--include-logits']
        options = ['--kv-cache', '--kv-page-tokens', '2', '--kv-codec', 'tq', '--prefill-chunk-size', '2']
        completed = subprocess.run(command + options + ['--generate', '3'], capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report['kv_codec'], 'tq')
        self.assertEqual(report['kv_cache_plan']['bits'], 3)
        self.assertEqual(report['kv_cache_plan']['seed'], 42)
        self.assertEqual(report['run_totals']['processed_tokens'], 5)
        with self.session() as session:
            expected, generated = session.prefill([1, 3]), []
            for _ in range(3):
                token = max(range(self.config.vocab_size), key=lambda index: expected[-1][index])
                generated.append(token)
                expected.append(session.decode(token))
        self.assertEqual(report['appended_token_ids'], generated)
        self.assert_rows_close(report['logits'], expected)
        invalid = (['--kv-codec', 'tq'], ['--kv-cache', '--kv-codec', 'tq'],
                   options + ['--kv-group-size', '3'], options + ['--kv-bits', '0'],
                   options + ['--kv-seed', str(1 << 31)], ['--kv-bits', '3'],
                   ['--kv-cache', '--kv-page-tokens', '2', '--kv-codec', 'q3', '--kv-seed', '42'])
        for args in invalid:
            with self.subTest(args=args):
                failed = subprocess.run(command + args, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn('--kv-', failed.stderr)


@unittest.skipUnless(torch_available(), 'Optional PyTorch reference is unavailable')
class TQPagedTorchRegressions(_TQFixture):
    def test_exported_default_codebook_rows_and_logits_match_independent_reference(self):
        config, weights = load_bundle_weights(self.path)
        with self.session(page_tokens=3, max_chunk_length=3, kv_seed=-7) as session:
            actual = session.prefill([1, 3])
            book = session.report()['kv_cache_plan']['codebook_f32le']
            oracle = TorchLlamaReference(config, weights, kv_codec='tq', kv_seed=-7, kv_codebook_f32le=book)
            codec = TQReference(config.head_dim, 3, -7, book)
            self.assertTrue(compare_logits(actual, oracle.prefill([1, 3]))['passed'])
            actual = session.append([5, 7, 2])
            expected = [oracle.decode(token).tolist() for token in (5, 7, 2)]
            self.assertTrue(compare_logits(actual, expected)['passed'])
            for (layer, kind, position, head), packed in self.packed_prefix(session).items():
                self.assertEqual(codec.decode(packed), oracle._cache[layer][kind][position, head].tolist())
            self.assertTrue(compare_logits(session.decode(8), oracle.decode(8))['passed'])

    def test_oracle_preserves_existing_defaults_and_requires_explicit_tq_book(self):
        config, weights = load_bundle_weights(self.path)
        tokens = [1, 3, 5]
        for implicit, explicit in (({}, {'kv_codec': 'f32'}),
                                   ({'kv_group_size': 3}, {'kv_codec': 'q4', 'kv_group_size': 3}),
                                   ({'kv_codec': 'q3'}, {'kv_codec': 'q3', 'kv_group_size': 32})):
            self.assertEqual(llama_forward(config, weights, tokens, **implicit).tolist(),
                             llama_forward(config, weights, tokens, **explicit).tolist())
        for options in ({'kv_codec': 'tq'}, {'kv_codec': 'tq', 'kv_group_size': 3},
                        {'kv_bits': 3}, {'kv_codec': 'q3', 'kv_codebook_f32le': linear_codebook()}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TorchLlamaReference(config, weights, **options)

    def test_verifier_and_cli_separate_execution_weight_kv_and_combined_errors(self):
        source, path = self.tiny_bundle()
        tokens = TQ_KV_GOLDEN['token_ids']
        options = {name: TQ_KV_GOLDEN[name] for name in ('kv_codec', 'kv_bits', 'kv_seed', 'kv_codebook_f32le')}
        with self.session(path, **options) as session:
            actual = session.prefill(tokens)
        report = verify_bundle_forward(path, tokens, actual, source_dir=source, **options)
        self.assertTrue(report['verified'])
        self.assertEqual(report['reference'], 'pytorch_decoded_q4_kv_tq')
        self.assertLessEqual(report['execution_error']['max_abs_error'], 1e-5)
        for field in ('quantization_error', 'kv_quantization_error', 'combined_quantization_error'):
            self.assertNotIn('passed', report[field])
            self.assertAlmostEqual(report[field]['max_abs_error'], TQ_KV_GOLDEN['errors'][field]['max_abs_error'], places=7)
        without_source = verify_bundle_forward(path, tokens, actual, **options)
        self.assertIsNone(without_source['quantization_error'])
        self.assertNotIn('combined_quantization_error', without_source)
        command = [sys.executable, str(ROOT / 'tools/nexa_run.py'), str(path), '--tokens', '1,3,5',
                   '--decode-tokens', '7', '--kv-cache', '--kv-page-tokens', '3', '--kv-codec', 'tq',
                   '--kv-bits', '3', '--kv-seed', '-7', '--prefill-chunk-size', '2', '--verify',
                   '--reference-checkpoint', str(source), '--memory-budget', '1MiB']
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result['validation']['verified'])
        self.assertEqual(result['validation']['reference'], 'pytorch_decoded_q4_kv_tq')
        self.assertEqual(result['validation']['kv_codebook_f32le'], result['kv_cache_plan']['codebook_f32le'])
        for field in ('quantization_error', 'kv_quantization_error', 'combined_quantization_error'):
            self.assertNotIn('passed', result['validation'][field])


if __name__ == '__main__':
    unittest.main()

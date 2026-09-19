"""Independent Q3 KV codec, numerical proofs and transactional cache regressions."""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import random
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
from q3_reference import MAX_GROUP_SIZE, decode_q3_row, quantize_q3_row, row_size
from runtime.nexapack.bundle import ModelBundleReader
from runtime.nexapack.format import NexaPackReader
from test_paged_transformer_regressions import _PagedFixture
from test_q4_paged_transformer_regressions import wide_bundle
from test_transformer_forward_regressions import f32, random_bundle
from transformer_reference import (TorchLlamaReference, compare_logits, llama_forward,
                                   load_bundle_weights, torch_available, verify_bundle_forward)


# Generated exclusively by the independent Python codec and PyTorch equations.
Q3_KV_GOLDEN = json.loads(r'''
{
  "generated_by": "TorchLlamaReference with independent Python Q3; native executor not used",
  "torch_version": "2.11.0",
  "pack_sha256": "aeabc843eb9a1c4c72368351bb87ae681b5856c1e03bfd1462191c235b699df3",
  "kv_codec": "q3",
  "kv_group_size": 3,
  "token_ids": [
    1,
    3,
    5,
    7
  ],
  "atol": 1e-05,
  "rtol": 0.0001,
  "logits": [
    [
      -2.4854228496551514,
      -2.324942111968994,
      -1.8324507474899292,
      -1.5706428289413452,
      -1.2967931032180786,
      0.143454447388649,
      0.3910234272480011,
      0.7035015821456909,
      0.520298421382904,
      0.8077181577682495,
      1.0306648015975952,
      1.3044722080230713,
      1.6558257341384888,
      1.7342743873596191,
      1.0701630115509033,
      1.2868907451629639
    ],
    [
      -2.879885196685791,
      -2.887310743331909,
      -1.252218246459961,
      -1.2899667024612427,
      -1.350604772567749,
      -0.2230464667081833,
      -0.33056581020355225,
      -0.4074232280254364,
      0.16080471873283386,
      0.07282491028308868,
      -0.021347275003790855,
      0.8182641267776489,
      0.907280445098877,
      0.6125878095626831,
      1.313134789466858,
      1.2328358888626099
    ],
    [
      1.912771224975586,
      1.899350643157959,
      1.3507800102233887,
      1.3147892951965332,
      1.210835337638855,
      0.4585179388523102,
      0.42471417784690857,
      0.26725736260414124,
      -1.6917169094085693,
      -1.8335511684417725,
      -1.8865952491760254,
      -2.1895484924316406,
      -2.306654214859009,
      -2.284808874130249,
      -2.0520899295806885,
      -2.0676259994506836
    ],
    [
      1.9603362083435059,
      1.9615099430084229,
      1.6180229187011719,
      1.6395739316940308,
      1.6134394407272339,
      0.9528613090515137,
      1.011400580406189,
      0.935745120048523,
      -1.603981852531433,
      -1.649065375328064,
      -1.645653247833252,
      -1.9828851222991943,
      -2.0399723052978516,
      -1.9427919387817383,
      -2.439650297164917,
      -2.3970446586608887
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
      "max_abs_error": 0.20185807347297668,
      "mean_abs_error": 0.04985728356405161,
      "max_rel_error": 2.739413013591308
    },
    "combined_quantization_error": {
      "max_abs_error": 0.42348912358283997,
      "mean_abs_error": 0.14314948036917485,
      "max_rel_error": 10.364558601431703
    }
  }
}
''')


class Q3CodecReferenceRegressions(unittest.TestCase):
    def test_manual_bit_crossing_halfway_and_padding_vectors(self):
        # Codes [3,1,-1,2,-2] occupy bits 0..14 and cross the first byte.
        packed = quantize_q3_row([3, 0.5, -0.5, 1.5, -1.5], 5)
        self.assertEqual(packed.hex(), '0000803fcb65')
        self.assertEqual(decode_q3_row(packed, 5, 5), [3, 1, -1, 2, -2])
        self.assertEqual(quantize_q3_row([3, -3], 3).hex(), '0000803f2b00')
        self.assertEqual(quantize_q3_row([0.0, -0.0], 8), bytes(7))
        self.assertEqual(row_size(4, 3), 12)
        self.assertEqual(row_size(64, 32), 32)

    def test_reference_rejects_invalid_scales_codes_padding_and_inputs(self):
        invalid = [(bytes.fromhex('0000803f04'), 1, 1),  # reserved -4
                   (bytes.fromhex('0000803f08'), 1, 1),  # high bits
                   (bytes.fromhex('0000803f08'), 1, 2),  # padded coordinate
                   (bytes.fromhex('0000000001'), 1, 1),  # zero scale/nonzero code
                   (struct.pack('<f', -1.0) + b'\x00', 1, 1),
                   (struct.pack('<f', math.nan) + b'\x00', 1, 1),
                   (struct.pack('<f', math.inf) + b'\x00', 1, 1),
                   (b'', 1, 1), (bytes(6), 1, 1)]
        for packed, cols, group in invalid:
            with self.subTest(packed=packed, cols=cols, group=group), self.assertRaises(ValueError):
                decode_q3_row(packed, cols, group)
        for values in ([], [math.nan], [math.inf], [-math.inf], [1e100]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                quantize_q3_row(values, 3)
        for group in (True, 0, -1, 1.5, MAX_GROUP_SIZE + 1):
            with self.subTest(group=group), self.assertRaises(ValueError):
                quantize_q3_row([1.0], group)
        with self.assertRaises(ArithmeticError):
            quantize_q3_row([math.ldexp(1.0, -149)], 3)

    def test_decoder_retains_exact_double_product(self):
        scale = f32(0.1)
        decoded, = decode_q3_row(struct.pack('<f', scale) + b'\x03', 1, 1)
        self.assertEqual(decoded, scale * 3)
        self.assertNotEqual(decoded, f32(scale * 3))


class _Q3Fixture(_PagedFixture):
    def session(self, path=None, **kwargs):
        kwargs.setdefault('kv_codec', 'q3')
        if kwargs['kv_codec'] == 'q3':
            kwargs.setdefault('kv_group_size', 3)
        return super().session(path, **kwargs)

    @staticmethod
    def packed_sizes(config, page_tokens, group_size):
        head_bytes = row_size(config.head_dim, group_size)
        token_bytes = config.num_key_value_heads * head_bytes
        stride = (page_tokens * token_bytes + 63) // 64 * 64
        return head_bytes, token_bytes, 2 * config.num_hidden_layers * page_tokens * token_bytes, 2 * config.num_hidden_layers * stride + 63

    def packed_prefix(self, session, length=None):
        length = session.cache_length if length is None else length
        head_bytes, token_bytes, _, _ = self.packed_sizes(
            session.config, session.page_tokens, session.report()['kv_group_size'])
        stride = (session.page_tokens * token_bytes + 63) // 64 * 64
        rows = {}
        for position in range(length):
            page_index, within_page = divmod(position, session.page_tokens)
            for layer in range(session.config.num_hidden_layers):
                for kind in range(2):
                    address = session._pages[page_index].address + (2 * layer + kind) * stride
                    address += within_page * token_bytes
                    for head in range(session.config.num_key_value_heads):
                        rows[layer, kind, position, head] = ctypes.string_at(address + head * head_bytes, head_bytes)
        return rows


class Q3PagedNativeRegressions(_Q3Fixture):
    def test_native_quantizer_matches_independent_codec_at_bit_boundaries(self):
        from runtime.nexapack.transformer import _load_kernels
        quantize = _load_kernels().nexa_q3_quantize
        rng = random.Random(3003)
        values = [f32(rng.uniform(-3, 3)) for _ in range(21)]
        values[:7] = [3, 0.5, -0.5, 1.5, -1.5, 0, -3]
        source = (ctypes.c_float * len(values))(*values)
        for group in (1, 2, 3, 5, 8, 11, 32):
            with self.subTest(group=group):
                expected = b''.join(quantize_q3_row(values[start:start + 7], group) for start in range(0, 21, 7))
                target = (ctypes.c_uint8 * len(expected))()
                self.assertEqual(quantize(source, len(source), 3, 7, group, target, len(target)), 0)
                self.assertEqual(bytes(target), expected)

    def test_logits_are_invariant_to_page_chunk_layers_heads_and_ties(self):
        tokens = [1, 4, 2, 6, 8, 3, 5]
        for index, (layers, heads, kv_heads, tied, group) in enumerate(
                ((1, 2, 2, False, 1), (2, 4, 1, True, 3), (2, 4, 2, False, 5))):
            path = self.directory / f'case-{index}'
            random_bundle(path, layers=layers, heads=heads, kv_heads=kv_heads, tied=tied, seed=1601 + index)
            with self.session(path, page_tokens=8, kv_group_size=group) as complete:
                expected = complete.prefill(tokens)
            for page_tokens, chunks in ((1, (1, 3, 2, 1)), (2, (3, 1, 3)), (5, (2, 2, 3))):
                with self.subTest(case=index, page=page_tokens), self.session(
                        path, page_tokens=page_tokens, kv_group_size=group, max_chunk_length=3) as session:
                    actual, cursor = session.prefill(tokens[:chunks[0]]), chunks[0]
                    for count in chunks[1:]:
                        actual += session.append(tokens[cursor:cursor + count])
                        cursor += count
                    self.assert_rows_close(actual, expected)
                    self.assertEqual(session.token_ids, tuple(tokens))

    def test_oracle_generated_golden_without_torch(self):
        settings = json.loads((ROOT / 'tests/fixtures/transformer_tiny_golden.json').read_text())
        source, path = self.directory / 'source', self.directory / 'tiny'
        create_checkpoint(source, include_tied_head=True)
        import_llama_checkpoint(source, path, **settings['import_settings'])
        with ModelBundleReader(path) as bundle:
            hashes = {name: hashlib.sha256((path / entry['path']).read_bytes()).hexdigest()
                      for name, entry in bundle.manifest['tensors'].items()}
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(fingerprint, Q3_KV_GOLDEN['pack_sha256'])
        for page_tokens in (1, 3):
            with self.subTest(page_tokens=page_tokens), self.session(
                    path, page_tokens=page_tokens, max_chunk_length=2) as session:
                tokens = Q3_KV_GOLDEN['token_ids']
                actual = session.prefill(tokens[:1])
                actual += session.append(tokens[1:3])
                actual += [session.decode(tokens[3])]
                self.assert_rows_close(actual, Q3_KV_GOLDEN['logits'], Q3_KV_GOLDEN['atol'], Q3_KV_GOLDEN['rtol'])

    def test_prefix_bytes_stay_immutable_and_only_new_heads_are_quantized(self):
        from runtime.nexapack.transformer import _load_kernels
        with self.session(page_tokens=3, max_chunk_length=3) as session:
            session.prefill([1, 3])
            prefix, address = self.packed_prefix(session), session._pages[0].address
            kernels = _load_kernels()
            original = kernels.nexa_q3_quantize
            def quantize(*args):
                rows, cols, group = args[2:5]
                values = list(args[0])
                expected = b''.join(quantize_q3_row(values[start:start + cols], group)
                                    for start in range(0, rows * cols, cols))
                status = original(*args)
                self.assertEqual(status, 0)
                self.assertEqual(ctypes.string_at(args[5], args[6]), expected)
                return status
            with patch.object(kernels, 'nexa_q3_quantize', side_effect=quantize) as quantizer:
                with patch.object(kernels, 'nexa_causal_gqa_attention_paged_q3',
                                  wraps=kernels.nexa_causal_gqa_attention_paged_q3) as attention:
                    with patch.object(kernels, 'nexa_causal_gqa_attention_paged_q4',
                                      side_effect=AssertionError('Q4 attention used for Q3')):
                        session.append([5, 7, 2])
            self.assertEqual(self.packed_prefix(session, 2), prefix)
            self.assertEqual(session._pages[0].address, address)
            self.assertEqual(sum(call.args[2] for call in quantizer.call_args_list),
                             2 * self.config.num_hidden_layers * 3 * self.config.num_key_value_heads)
            self.assertTrue(all(call.args[3:5] == (self.config.head_dim, 3) for call in quantizer.call_args_list))
            self.assertEqual(attention.call_count, self.config.num_hidden_layers)
            self.assertTrue(all(call.args[8:11] == (3, 2, 3) for call in attention.call_args_list))
            prefix = self.packed_prefix(session)
            session.decode(8)
            self.assertEqual(self.packed_prefix(session, 5), prefix)

    def test_packed_accounting_and_three_way_physical_residency(self):
        with self.session(page_tokens=3, max_chunk_length=3) as session:
            session.prefill([1, 3])
            session.append([5, 7])
            report, memory = session.report(), session.report()['memory']
            _, per_layer_token, payload, allocation = self.packed_sizes(self.config, 3, 3)
            token_bytes = 2 * self.config.num_hidden_layers * per_layer_token
            self.assertEqual(report['kv_codec'], 'q3')
            self.assertEqual(report['kv_cache_plan']['codec_id'], 'Q3_GROUPED')
            self.assertEqual(report['kv_cache_plan']['codec_version'], 1)
            self.assertEqual(memory['kv_encoded_bytes_per_token'], token_bytes)
            self.assertEqual(memory['kv_valid_prefix_bytes'], 4 * token_bytes)
            self.assertEqual(memory['kv_valid_prefix_f32_bytes'],
                             4 * 2 * self.config.num_hidden_layers * self.config.num_key_value_heads * self.config.head_dim * 4)
            self.assertEqual(memory['kv_page_payload_bytes'], payload)
            self.assertEqual(memory['kv_page_allocation_bytes'], allocation)
            self.assertEqual(session.resident_kv_bytes, 2 * allocation)
            self.assertEqual(memory['persistent_kv_bytes'], 2 * payload)
            self.assertEqual(memory['kv_page_table_bytes'], 4 * ctypes.sizeof(ctypes.c_void_p))
            self.assertEqual(memory['kv_full_dequantized_buffer_bytes'], 0)
            self.assertEqual(memory['managed_buffers_peak_bound_bytes'],
                             memory['arena_allocation_bytes'] + memory['reader_scratch_capacity_bytes'] + 2 * allocation)
            self.assertEqual(report['io']['kv_bytes_written'], 2 * token_bytes)
            self.assertEqual(report['io']['kv_source_f32_bytes_quantized'],
                             2 * 2 * self.config.num_hidden_layers * self.config.num_key_value_heads * self.config.head_dim * 4)
        path = self.directory / 'wide'
        wide_bundle(path)
        for codec, expected in (('f32', 8255), ('q4', 1343), ('q3', 1087)):
            options = {} if codec == 'f32' else {'kv_group_size': 32}
            with self.subTest(codec=codec), self.session(path, kv_codec=codec, page_tokens=16,
                                                       max_chunk_length=2, tile_rows=32, **options) as session:
                self.assertEqual(session.resident_kv_bytes, 0)
                session.prefill([1, 3])
                self.assertEqual(session.resident_kv_bytes, expected)

    def test_checksum_and_report_failure_rollback_then_retry(self):
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
                        guard = (patch.object(session, '_report', side_effect=MemoryError('Q3 report failed'))
                                 if failure == 'report' else patch.object(session, '_run_attention', wraps=session._run_attention))
                        with guard as attention:
                            try:
                                getattr(session, mode)([2, 4, 6])
                            except (ValueError, MemoryError) as exc:
                                if failure == 'checksum':
                                    self.assertIn('checksum', str(exc))
                                    self.assertEqual(attention.call_count, 1)
                                saved.append(exc)
                            else:
                                self.fail('injected Q3 transaction failure did not occur')
                    self.assertEqual(session.report(), before)
                    self.assertEqual(session.token_ids, (1,))
                    self.assertEqual(session._pages, old)
                    self.assertEqual(self.packed_prefix(session), prefix)
                    self.assertGreater(len(references), 0)
                    self.assert_pages_released(references)
                finally:
                    if failure == 'checksum':
                        path.write_bytes(original)
                self.assertIsNotNone(saved[0].__traceback__)
                actual = session.append([8, 9, 5])
                with self.session(page_tokens=4) as fresh:
                    self.assert_rows_close(actual, fresh.prefill([1, 8, 9, 5])[-3:])

    def test_real_quantizer_underflow_after_key_write_rolls_back(self):
        from runtime.nexapack.transformer import _load_kernels
        with self.session(page_tokens=2, max_chunk_length=2) as session:
            session.prefill([1, 3])
            before, prefix, old = session.report(), self.packed_prefix(session), list(session._pages)
            references, saved, statuses, key_payloads = [], [], [], []
            original_allocate = session._allocate_page
            original_quantize = _load_kernels().nexa_q3_quantize
            def allocate():
                return self.track_page(original_allocate(), references)
            def quantize(*args):
                if len(statuses) == 1:
                    self.assertEqual(statuses, [0])
                    self.assertTrue(any(key_payloads[0]))
                    for index in range(args[1]):
                        args[0][index] = 0.0
                    args[0][0] = math.ldexp(1.0, -149)
                status = original_quantize(*args)
                if not statuses:
                    key_payloads.append(ctypes.string_at(args[5], args[6]))
                statuses.append(status)
                return status
            with patch.object(session, '_allocate_page', side_effect=allocate):
                with patch.object(_load_kernels(), 'nexa_q3_quantize', side_effect=quantize):
                    try:
                        session.append([5])
                    except ArithmeticError as exc:
                        self.assertIn('nexa_q3_quantize', str(exc))
                        self.assertIn('status -5', str(exc))
                        saved.append(exc)
                    else:
                        self.fail('native Q3 quantizer accepted an underflowed scale')
            self.assertEqual(statuses, [0, -5])
            self.assertEqual(session.report(), before)
            self.assertEqual(session.token_ids, (1, 3))
            self.assertEqual(session._pages, old)
            self.assertEqual(self.packed_prefix(session), prefix)
            self.assertEqual(len(references), 1)
            self.assert_pages_released(references)
            self.assertIsNotNone(saved[0].__traceback__)
            actual = session.append([8, 9])
            with self.session(page_tokens=4) as fresh:
                self.assert_rows_close(actual, fresh.prefill([1, 3, 8, 9])[-2:])

    def test_exact_budget_and_invalid_groups_fail_before_payload_pages_or_kernel(self):
        from runtime.nexapack.paged import PagedTransformerSession
        with self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137) as session:
            required = session.report()['memory']['capacity_managed_buffers_bound_bytes'] + 137
        cases = [*({'kv_group_size': group} for group in (True, 0, -1, 1.5, MAX_GROUP_SIZE + 1)),
                 {'memory_budget': required - 1}]
        with patch.object(NexaPackReader, 'read_rows_into', side_effect=AssertionError('payload read')):
            with patch.object(ModelBundleReader, 'read_f32_into', side_effect=AssertionError('norm read')):
                with patch('runtime.nexapack.transformer._load_kernels', side_effect=AssertionError('kernel loaded')):
                    with patch.object(PagedTransformerSession, '_allocate_page', side_effect=AssertionError('page allocated')):
                        for options in cases:
                            with self.subTest(options=options), self.assertRaises((ValueError, MemoryBudgetError)):
                                self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137, **options)
                        with self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137,
                                          memory_budget=required) as exact:
                            self.assertEqual(exact.resident_kv_bytes, 0)
        with self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137, memory_budget=required) as exact:
            exact.prefill([1, 2])
            for chunk in ([3, 4], [5, 6], [7, 8]):
                exact.append(chunk)
            exact.prefill([2, 4])
            self.assertLessEqual(exact.report()['memory']['managed_buffers_peak_bound_bytes'] + 137, required)

    def test_cli_validation_default_group_and_greedy_ids(self):
        command = [sys.executable, str(ROOT / 'tools/nexa_run.py'), str(self.path), '--tokens', '1,3',
                   '--memory-budget', '1MiB', '--include-logits']
        options = ['--kv-cache', '--kv-page-tokens', '2', '--kv-codec', 'q3', '--prefill-chunk-size', '2']
        completed = subprocess.run(command + options + ['--generate', '3'], capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report['kv_codec'], 'q3')
        self.assertEqual(report['kv_group_size'], 32)
        self.assertEqual(report['run_totals']['processed_tokens'], 5)
        with self.session(kv_group_size=32) as session:
            expected = session.prefill([1, 3])
            generated = []
            for _ in range(3):
                token = max(range(self.config.vocab_size), key=lambda index: expected[-1][index])
                generated.append(token)
                expected.append(session.decode(token))
        self.assertEqual(report['appended_token_ids'], generated)
        self.assert_rows_close(report['logits'], expected)
        for invalid in (['--kv-codec', 'q3'], ['--kv-cache', '--kv-codec', 'q3'],
                        ['--kv-cache', '--kv-page-tokens', '2', '--kv-codec', 'q3', '--kv-group-size', '0'],
                        ['--kv-cache', '--kv-page-tokens', '2', '--kv-codec', 'f32', '--kv-group-size', '3']):
            failed = subprocess.run(command + invalid, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('--kv-', failed.stderr)


@unittest.skipUnless(torch_available(), 'Optional PyTorch reference is unavailable')
class Q3PagedTorchRegressions(_Q3Fixture):
    def test_native_cache_rows_and_logits_match_independent_oracle(self):
        config, weights = load_bundle_weights(self.path)
        oracle = TorchLlamaReference(config, weights, kv_codec='q3', kv_group_size=3)
        with self.session(page_tokens=3, max_chunk_length=3) as session:
            self.assertTrue(compare_logits(session.prefill([1, 3]), oracle.prefill([1, 3]))['passed'])
            actual = session.append([5, 7, 2])
            expected = [oracle.decode(token).tolist() for token in (5, 7, 2)]
            self.assertTrue(compare_logits(actual, expected)['passed'])
            for (layer, kind, position, head), packed in self.packed_prefix(session).items():
                self.assertEqual(decode_q3_row(packed, config.head_dim, 3),
                                 oracle._cache[layer][kind][position, head].tolist())
            self.assertTrue(compare_logits(session.decode(8), oracle.decode(8))['passed'])

    def test_oracle_codec_selection_preserves_f32_and_q4(self):
        config, weights = load_bundle_weights(self.path)
        tokens = [1, 3, 5]
        for implicit, explicit in (({}, {'kv_codec': 'f32'}),
                                   ({'kv_group_size': 3}, {'kv_codec': 'q4', 'kv_group_size': 3}),
                                   ({'kv_codec': 'q3'}, {'kv_codec': 'q3', 'kv_group_size': 32})):
            self.assertEqual(llama_forward(config, weights, tokens, **implicit).tolist(),
                             llama_forward(config, weights, tokens, **explicit).tolist())
        for options in ({'kv_codec': 'f32', 'kv_group_size': 3}, {'kv_codec': 'unknown'},
                        {'kv_codec': 'q3', 'kv_group_size': True}, {'kv_codec': 'q3', 'kv_group_size': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TorchLlamaReference(config, weights, **options)
        q3 = TorchLlamaReference(config, weights, kv_codec='q3', kv_group_size=3)
        f32_reference = TorchLlamaReference(config, weights)
        self.assertGreater(compare_logits(q3.prefill([1]), f32_reference.prefill([1]))['max_abs_error'], 0)
        saved = [[value.clone() for value in layer] for layer in q3._cache]
        q3.decode(3)
        for old_layer, new_layer in zip(saved, q3._cache):
            for old, new in zip(old_layer, new_layer):
                self.assertEqual(str(new.dtype), 'torch.float64')
                self.assertTrue((old == new[:1]).all())

    def test_verification_and_cli_separate_execution_weights_kv_and_combined_error(self):
        settings = json.loads((ROOT / 'tests/fixtures/transformer_tiny_golden.json').read_text())
        source, path = self.directory / 'source', self.directory / 'tiny'
        create_checkpoint(source, include_tied_head=True)
        import_llama_checkpoint(source, path, **settings['import_settings'])
        with self.session(path) as session:
            actual = session.prefill(settings['token_ids'])
        report = verify_bundle_forward(path, settings['token_ids'], actual, kv_codec='q3', kv_group_size=3, source_dir=source)
        self.assertTrue(report['verified'])
        self.assertEqual(report['reference'], 'pytorch_decoded_q4_kv_q3')
        self.assertLessEqual(report['execution_error']['max_abs_error'], 1e-5)
        for field in ('quantization_error', 'kv_quantization_error', 'combined_quantization_error'):
            self.assertNotIn('passed', report[field])
            self.assertAlmostEqual(report[field]['max_abs_error'], Q3_KV_GOLDEN['errors'][field]['max_abs_error'], places=7)
        without_source = verify_bundle_forward(path, settings['token_ids'], actual, kv_codec='q3', kv_group_size=3)
        self.assertIsNone(without_source['quantization_error'])
        self.assertNotIn('combined_quantization_error', without_source)
        self.assertGreater(without_source['kv_quantization_error']['max_abs_error'], 0)
        wrong = verify_bundle_forward(path, settings['token_ids'], actual, kv_group_size=3)
        self.assertFalse(wrong['verified'], 'Q3 logits must not select the backwards-compatible Q4 oracle')
        command = [sys.executable, str(ROOT / 'tools/nexa_run.py'), str(path), '--tokens', '1,3,5',
                   '--decode-tokens', '7', '--kv-cache', '--kv-page-tokens', '3', '--kv-codec', 'q3',
                   '--kv-group-size', '3', '--prefill-chunk-size', '2', '--verify',
                   '--reference-checkpoint', str(source), '--memory-budget', '1MiB']
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        validation = json.loads(completed.stdout)['validation']
        self.assertTrue(validation['verified'])
        self.assertEqual(validation['reference'], 'pytorch_decoded_q4_kv_q3')
        for field in ('quantization_error', 'kv_quantization_error', 'combined_quantization_error'):
            self.assertEqual(validation[field], report[field])


if __name__ == '__main__':
    unittest.main()

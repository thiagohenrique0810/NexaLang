"""Headwise Q4 KV pages: independent oracle, immutable prefixes and budgets."""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.importers.llama import import_llama_checkpoint
from compiler.model_config import ModelConfig
from compiler.planner.memory import MemoryBudgetError
from model_checkpoint_fixture import create_checkpoint
from runtime.nexapack.bundle import ModelBundleReader, write_model_bundle
from runtime.nexapack.format import MAX_GROUP_SIZE, NexaPackReader, decode_q4_row
from test_paged_transformer_regressions import _PagedFixture
from test_transformer_forward_regressions import f32, random_bundle
from transformer_reference import (TorchLlamaReference, compare_logits, load_bundle_weights,
                                   torch_available, verify_bundle_forward)


# Filled from the independent PyTorch oracle, never from the native executor.
Q4_KV_GOLDEN = json.loads(r'''
{
  "generated_by": "TorchLlamaReference only; native executor not used",
  "torch_version": "2.11.0",
  "pack_sha256": "aeabc843eb9a1c4c72368351bb87ae681b5856c1e03bfd1462191c235b699df3",
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
      -2.500962495803833,
      -2.339607000350952,
      -1.8460973501205444,
      -1.5834369659423828,
      -1.3104665279388428,
      0.12674349546432495,
      0.3728213906288147,
      0.6849143505096436,
      0.5087310075759888,
      0.795025646686554,
      1.0180398225784302,
      1.3018431663513184,
      1.654175043106079,
      1.7309365272521973,
      1.0867509841918945,
      1.3033336400985718
    ],
    [
      -2.8789451122283936,
      -2.8867619037628174,
      -1.2467464208602905,
      -1.2850492000579834,
      -1.346002459526062,
      -0.22019901871681213,
      -0.3280085623264313,
      -0.4053840935230255,
      0.16035304963588715,
      0.0720444768667221,
      -0.02264440432190895,
      0.8168010115623474,
      0.9053969383239746,
      0.6103574633598328,
      1.310326337814331,
      1.2296497821807861
    ],
    [
      2.017831563949585,
      1.9935240745544434,
      1.5451858043670654,
      1.4994324445724487,
      1.3986009359359741,
      0.6211423873901367,
      0.5926916599273682,
      0.4354475736618042,
      -1.6396055221557617,
      -1.7740262746810913,
      -1.8326994180679321,
      -2.1818954944610596,
      -2.307626962661743,
      -2.2780609130859375,
      -2.176006555557251,
      -2.195142984390259
    ],
    [
      1.7094111442565918,
      1.699354648590088,
      1.6169453859329224,
      1.617714762687683,
      1.5756800174713135,
      1.0574016571044922,
      1.0913108587265015,
      0.9965628385543823,
      -1.6239080429077148,
      -1.686280608177185,
      -1.7084673643112183,
      -1.9802429676055908,
      -2.0419533252716064,
      -1.9866646528244019,
      -2.330575704574585,
      -2.3104312419891357
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
      "max_abs_error": 0.25566303730010986,
      "mean_abs_error": 0.07498438062611967,
      "max_rel_error": 2.7207724682697605
    },
    "combined_quantization_error": {
      "max_abs_error": 0.5265865325927734,
      "mean_abs_error": 0.17100610508350655,
      "max_rel_error": 9.835541418804043
    }
  }
}
''')


def wide_bundle(path):
    """D=64 makes packed-cache savings observable despite page alignment."""
    config = ModelConfig(name='q4_kv_wide_fixture', vocab_size=13, hidden_size=64,
                         intermediate_size=96, num_hidden_layers=1, num_attention_heads=1,
                         num_key_value_heads=1, max_position_embeddings=16,
                         tie_word_embeddings=True, rms_norm_eps=1e-5, rope_theta=10000.0)
    rng = random.Random(411)
    def source(shape):
        if len(shape) == 1:
            return iter([[f32(rng.uniform(0.85, 1.15)) for _ in range(shape[0])]])
        return ([f32(rng.uniform(-0.15, 0.15)) for _ in range(shape[1])] for _ in range(shape[0]))
    sources = {name: lambda shape=shape: source(shape) for name, shape in config.required_tensor_shapes().items()}
    write_model_bundle(path, config, sources, group_size=32, block_rows=16)
    return config


class _Q4Fixture(_PagedFixture):
    def session(self, path=None, **kwargs):
        kwargs.setdefault('kv_codec', 'q4')
        if kwargs['kv_codec'] == 'q4':
            kwargs.setdefault('kv_group_size', 3)
        return super().session(path, **kwargs)

    @staticmethod
    def packed_sizes(config, page_tokens, group_size):
        head_bytes = math.ceil(config.head_dim / group_size) * (4 + math.ceil(group_size / 2))
        token_bytes = config.num_key_value_heads * head_bytes
        stride = (page_tokens * token_bytes + 63) // 64 * 64
        return head_bytes, token_bytes, 2 * config.num_hidden_layers * page_tokens * token_bytes, 2 * config.num_hidden_layers * stride + 63

    def packed_prefix(self, session, length=None):
        length = session.cache_length if length is None else length
        group_size = session.report()['kv_group_size']
        head_bytes, token_bytes, _, _ = self.packed_sizes(session.config, session.page_tokens, group_size)
        stride = (session.page_tokens * token_bytes + 63) // 64 * 64
        rows = {}
        for position in range(length):
            page_index, position_in_page = divmod(position, session.page_tokens)
            for layer in range(session.config.num_hidden_layers):
                for kind in range(2):
                    address = session._pages[page_index].address + (2 * layer + kind) * stride
                    address += position_in_page * token_bytes
                    for head in range(session.config.num_key_value_heads):
                        rows[layer, kind, position, head] = ctypes.string_at(address + head * head_bytes, head_bytes)
        return rows


class Q4PagedNativeRegressions(_Q4Fixture):
    def test_native_outputs_are_invariant_to_chunk_and_page_boundaries(self):
        tokens = [1, 4, 2, 6, 8, 3, 5]
        for index, (layers, heads, kv_heads, tied, group) in enumerate(
                ((1, 2, 2, False, 1), (2, 4, 1, True, 3), (2, 4, 2, False, 5))):
            path = self.directory / f'case-{index}'
            random_bundle(path, layers=layers, heads=heads, kv_heads=kv_heads, tied=tied, seed=1401 + index)
            with self.session(path, page_tokens=8, kv_group_size=group) as complete:
                expected = complete.prefill(tokens)
            for page_tokens, chunks in ((1, (1, 3, 2, 1)), (2, (3, 1, 3)), (3, (2, 2, 3))):
                with self.subTest(case=index, page=page_tokens), self.session(
                        path, page_tokens=page_tokens, kv_group_size=group, max_chunk_length=3) as session:
                    actual, cursor = session.prefill(tokens[:chunks[0]]), chunks[0]
                    for count in chunks[1:]:
                        actual += session.append(tokens[cursor:cursor + count])
                        cursor += count
                    self.assert_rows_close(actual, expected)
                    self.assertEqual(session.token_ids, tuple(tokens))

    def test_oracle_generated_golden_works_without_pytorch(self):
        settings = json.loads((ROOT / 'tests/fixtures/transformer_tiny_golden.json').read_text())
        source, bundle_path = self.directory / 'source', self.directory / 'tiny-bundle'
        create_checkpoint(source, include_tied_head=True)
        import_llama_checkpoint(source, bundle_path, **settings['import_settings'])
        with ModelBundleReader(bundle_path) as bundle:
            hashes = {name: hashlib.sha256((bundle_path / entry['path']).read_bytes()).hexdigest()
                      for name, entry in bundle.manifest['tensors'].items()}
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(fingerprint, Q4_KV_GOLDEN['pack_sha256'])
        self.assertEqual(fingerprint, settings['pack_sha256'])
        for page_tokens in (1, 3):
            with self.subTest(page_tokens=page_tokens), self.session(
                    bundle_path, page_tokens=page_tokens, kv_group_size=Q4_KV_GOLDEN['kv_group_size'],
                    max_chunk_length=2) as session:
                tokens = Q4_KV_GOLDEN['token_ids']
                actual = session.prefill(tokens[:1])
                actual += session.append(tokens[1:3])
                actual += [session.decode(tokens[3])]
                self.assert_rows_close(actual, Q4_KV_GOLDEN['logits'],
                                       Q4_KV_GOLDEN['atol'], Q4_KV_GOLDEN['rtol'])

    def test_append_never_requantizes_or_overwrites_committed_packed_rows(self):
        from runtime.nexapack.transformer import _load_kernels
        with self.session(page_tokens=3, max_chunk_length=3) as session:
            session.prefill([1, 3])
            prefix, addresses = self.packed_prefix(session), [page.address for page in session._pages]
            kernels = _load_kernels()
            with patch.object(kernels, 'nexa_q4_quantize', wraps=kernels.nexa_q4_quantize) as quantize:
                session.append([5, 7, 2])
            self.assertEqual(self.packed_prefix(session, 2), prefix)
            self.assertEqual(session._pages[0].address, addresses[0])
            self.assertEqual(sum(call.args[2] for call in quantize.call_args_list),
                             2 * self.config.num_hidden_layers * 3 * self.config.num_key_value_heads)
            self.assertTrue(all(call.args[3:5] == (self.config.head_dim, 3) for call in quantize.call_args_list))
            prefix = self.packed_prefix(session)
            session.decode(8)
            self.assertEqual(self.packed_prefix(session, 5), prefix)

    def test_decode_uses_headwise_quantizer_and_direct_q4_attention(self):
        from runtime.nexapack.transformer import _load_kernels
        with self.session(page_tokens=2, max_chunk_length=3) as session:
            session.prefill([1, 3, 5])
            kernels = _load_kernels()
            with patch.object(kernels, 'nexa_q4_quantize', wraps=kernels.nexa_q4_quantize) as quantize:
                with patch.object(kernels, 'nexa_causal_gqa_attention_paged_q4',
                                  wraps=kernels.nexa_causal_gqa_attention_paged_q4) as attention:
                    with patch.object(kernels, 'nexa_causal_gqa_attention_paged',
                                      side_effect=AssertionError('F32 attention used for Q4 cache')):
                        session.decode(7)
            self.assertEqual(quantize.call_count, 2 * self.config.num_hidden_layers)
            self.assertTrue(all(call.args[1:5] == (self.config.num_key_value_heads * self.config.head_dim,
                                                 self.config.num_key_value_heads, self.config.head_dim, 3)
                                for call in quantize.call_args_list))
            self.assertEqual(attention.call_count, self.config.num_hidden_layers)
            self.assertTrue(all(call.args[8:11] == (3, 3, 1) for call in attention.call_args_list))
            report = session.report()
            self.assertEqual(report['kv_codec'], 'q4')
            self.assertEqual(report['kv_group_size'], 3)
            self.assertEqual(report['kv_cache_plan']['codec'], 'q4')
            self.assertEqual(report['kv_cache_plan']['codec_id'], 'Q4_GROUPED')
            self.assertEqual(report['kv_cache_plan']['codec_version'], 1)
            self.assertEqual(report['kv_cache_plan']['group_size'], 3)
            self.assertEqual(report['processed_tokens'], 1)
            self.assertEqual(report['io']['kv_prefix_bytes_copied'], 0)
            self.assertEqual(report['memory']['kv_full_dequantized_buffer_bytes'], 0)

    def test_encoded_byte_accounting_padding_and_pointer_tables(self):
        with self.session(page_tokens=3, max_chunk_length=3) as session:
            session.prefill([1, 3])
            session.append([5, 7])
            report = session.report()
            memory = report['memory']
            _, token_bytes, payload, allocation = self.packed_sizes(self.config, 3, 3)
            all_layers_token_bytes = 2 * self.config.num_hidden_layers * token_bytes
            self.assertEqual(memory['kv_encoded_bytes_per_token'], all_layers_token_bytes)
            self.assertEqual(memory['kv_valid_prefix_bytes'], 4 * all_layers_token_bytes)
            self.assertEqual(memory['kv_valid_prefix_f32_bytes'],
                             4 * 2 * self.config.num_hidden_layers * self.config.num_key_value_heads * self.config.head_dim * 4)
            self.assertEqual(memory['kv_page_payload_bytes'], payload)
            self.assertEqual(memory['kv_page_allocation_bytes'], allocation)
            self.assertEqual(memory['persistent_kv_bytes'], 2 * payload)
            self.assertEqual(session.resident_kv_bytes, 2 * allocation)
            self.assertEqual(memory['kv_resident_allocation_bytes'], 2 * allocation)
            self.assertEqual(memory['kv_page_table_bytes'], 4 * ctypes.sizeof(ctypes.c_void_p))
            self.assertEqual(memory['managed_buffers_peak_bound_bytes'],
                             memory['arena_allocation_bytes'] + memory['reader_scratch_capacity_bytes'] + 2 * allocation)
            self.assertEqual(report['io']['kv_bytes_written'], 2 * all_layers_token_bytes)
            self.assertEqual(report['io']['kv_source_f32_bytes_quantized'],
                             2 * 2 * self.config.num_hidden_layers * self.config.num_key_value_heads * self.config.head_dim * 4)

    def test_q4_reduces_physical_residency_for_wide_heads(self):
        path = self.directory / 'wide'
        config = wide_bundle(path)
        with self.session(path, page_tokens=16, kv_group_size=32, max_chunk_length=2, tile_rows=32) as q4:
            initial = q4.report()['memory']
            self.assertEqual(q4.resident_kv_bytes, 0)
            q4.prefill([1, 3])
            memory = q4.report()['memory']
            _, _, payload, allocation = self.packed_sizes(config, 16, 32)
            self.assertEqual(memory['persistent_kv_bytes'], payload)
            self.assertEqual(q4.resident_kv_bytes, allocation)
            self.assertEqual(allocation, 1343)
        with self.session(path, page_tokens=16, kv_codec='f32', max_chunk_length=2, tile_rows=32) as f32_session:
            f32_session.prefill([1, 3])
            self.assertEqual(f32_session.resident_kv_bytes, 8255)
            self.assertLess(allocation, f32_session.resident_kv_bytes)
            self.assertLess(initial['capacity_managed_buffers_bound_bytes'],
                            f32_session.report()['memory']['capacity_managed_buffers_bound_bytes'])

    def test_checksum_and_late_report_failure_preserve_packed_prefix_and_release_new_pages(self):
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
                        if failure == 'report':
                            guard = patch.object(session, '_report', side_effect=MemoryError('Q4 report failed'))
                        else:
                            guard = patch.object(session, '_run_attention', wraps=session._run_attention)
                        with guard as attention:
                            try:
                                getattr(session, mode)([2, 4, 6])
                            except (ValueError, MemoryError) as exc:
                                if failure == 'checksum':
                                    self.assertIn('checksum', str(exc))
                                    self.assertEqual(attention.call_count, 1)
                                saved.append(exc)
                            else:
                                self.fail('injected Q4 transaction failure did not occur')
                    self.assertEqual(session.report(), before)
                    self.assertEqual(session.token_ids, (1,))
                    self.assertEqual(session._pages, old)
                    self.assertEqual(self.packed_prefix(session), prefix)
                    self.assert_pages_released(references)
                finally:
                    if failure == 'checksum':
                        path.write_bytes(original)
                self.assertIsNotNone(saved[0].__traceback__)
                actual = session.append([8, 9, 5])
                with self.session(page_tokens=4) as fresh:
                    self.assert_rows_close(actual, fresh.prefill([1, 8, 9, 5])[-3:])

    def test_native_quantizer_underflow_after_key_write_preserves_prefix_and_retries(self):
        from runtime.nexapack.transformer import _load_kernels
        with self.session(page_tokens=2, max_chunk_length=2) as session:
            # Fill one page so appending one token allocates a partial new page:
            # the first quantizer call writes K and the second writes V.
            session.prefill([1, 3])
            before, prefix, old = session.report(), self.packed_prefix(session), list(session._pages)
            references, saved, statuses, key_payloads = [], [], [], []
            original_allocate = session._allocate_page
            original_quantize = _load_kernels().nexa_q4_quantize
            def allocate():
                return self.track_page(original_allocate(), references)
            def quantize(*args):
                if len(statuses) == 1:
                    self.assertEqual(statuses, [0])
                    self.assertTrue(any(key_payloads[0]), 'K was not written before the V failure')
                    inputs, count = args[:2]
                    for index in range(count):
                        inputs[index] = 0.0
                    # Smallest positive binary32 value divided by seven rounds
                    # to scale zero. Invoke the real kernel to reject it.
                    inputs[0] = math.ldexp(1.0, -149)
                status = original_quantize(*args)
                if not statuses:
                    key_payloads.append(ctypes.string_at(args[5], args[6]))
                statuses.append(status)
                return status
            with patch.object(session, '_allocate_page', side_effect=allocate):
                with patch.object(_load_kernels(), 'nexa_q4_quantize', side_effect=quantize):
                    try:
                        session.append([5])
                    except ArithmeticError as exc:
                        self.assertIn('nexa_q4_quantize', str(exc))
                        self.assertIn('status -5', str(exc))
                        saved.append(exc)
                    else:
                        self.fail('native quantizer accepted an underflowed nonzero scale')
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

    def test_invalid_codecs_groups_and_budget_fail_before_payload_or_allocation(self):
        from runtime.nexapack.paged import PagedTransformerSession
        with self.session(page_tokens=3, max_chunk_length=2, reserve_bytes=137) as session:
            required = session.report()['memory']['capacity_managed_buffers_bound_bytes'] + 137
        cases = ({'kv_codec': 'unknown'}, {'kv_codec': 'f32', 'kv_group_size': 3},
                 *({'kv_group_size': group} for group in (True, 0, -1, 1.5, MAX_GROUP_SIZE + 1)),
                 {'memory_budget': required - 1})
        with patch.object(NexaPackReader, 'read_rows_into', side_effect=AssertionError('Q4 payload read')):
            with patch.object(ModelBundleReader, 'read_f32_into', side_effect=AssertionError('norm read')):
                with patch('runtime.nexapack.transformer._load_kernels', side_effect=AssertionError('kernel loaded')):
                    with patch.object(PagedTransformerSession, '_allocate_page', side_effect=AssertionError('KV allocated')):
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

    def test_default_group_and_f32_default_behavior_remain_explicit(self):
        from runtime.nexapack.paged import PagedTransformerSession
        with PagedTransformerSession(self.path, page_tokens=2, max_sequence_length=4,
                                     memory_budget='1MiB', kv_codec='q4') as q4:
            self.assertEqual(q4.report()['kv_group_size'], 32)
            q4.prefill([1, 3])
        with PagedTransformerSession(self.path, page_tokens=2, max_sequence_length=4,
                                     memory_budget='1MiB') as default:
            expected = default.prefill([1, 3])
            self.assertEqual(default.report()['kv_codec'], 'f32')
        with self.session(page_tokens=2, kv_codec='f32', max_sequence_length=4) as explicit:
            self.assert_rows_close(explicit.prefill([1, 3]), expected)

    def test_cli_q4_codec_requires_paged_kv_and_reports_encoded_work(self):
        command = [sys.executable, str(ROOT / 'tools/nexa_run.py'), str(self.path),
                   '--tokens', '1,3,5', '--decode-tokens', '7', '--memory-budget', '1MiB', '--include-logits']
        options = ['--kv-cache', '--kv-page-tokens', '2', '--kv-codec', 'q4',
                   '--kv-group-size', '3', '--prefill-chunk-size', '2']
        completed = subprocess.run(command + options, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report['kv_codec'], 'q4')
        self.assertEqual(report['kv_group_size'], 3)
        self.assertEqual(report['run_totals']['processed_tokens'], 4)
        with self.session(page_tokens=4) as session:
            self.assert_rows_close(report['logits'], session.prefill([1, 3, 5, 7]))
        for invalid in (['--recompute', '--kv-codec', 'q4'], ['--kv-two-banks', '--kv-codec', 'q4'],
                        ['--kv-cache', '--kv-page-tokens', '2', '--kv-group-size', '3'],
                        ['--kv-cache', '--kv-page-tokens', '2', '--kv-codec', 'q4', '--kv-group-size', '0']):
            with self.subTest(options=invalid):
                failed = subprocess.run(command + invalid, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn('--kv-', failed.stderr)


@unittest.skipUnless(torch_available(), 'Optional PyTorch reference is unavailable')
class Q4PagedTorchRegressions(_Q4Fixture):
    def test_native_and_packed_rows_match_independent_headwise_torch_oracle(self):
        config, weights = load_bundle_weights(self.path)
        oracle = TorchLlamaReference(config, weights, kv_group_size=3)
        with self.session(page_tokens=3, max_chunk_length=3) as session:
            self.assertTrue(compare_logits(session.prefill([1, 3]), oracle.prefill([1, 3]))['passed'])
            actual = session.append([5, 7, 2])
            expected = [oracle.decode(token).tolist() for token in (5, 7, 2)]
            self.assertTrue(compare_logits(actual, expected)['passed'])
            for (layer, kind, position, head), packed in self.packed_prefix(session).items():
                self.assertEqual(decode_q4_row(packed, config.head_dim, 3),
                                 oracle._cache[layer][kind][position, head].tolist())
            self.assertTrue(compare_logits(session.decode(8), oracle.decode(8))['passed'])

    def test_verification_separates_execution_weights_kv_and_combined_error(self):
        settings = json.loads((ROOT / 'tests/fixtures/transformer_tiny_golden.json').read_text())
        source, path = self.directory / 'source', self.directory / 'tiny'
        create_checkpoint(source, include_tied_head=True)
        import_llama_checkpoint(source, path, **settings['import_settings'])
        with self.session(path, kv_group_size=3) as session:
            actual = session.prefill(settings['token_ids'])
        report = verify_bundle_forward(path, settings['token_ids'], actual, source_dir=source, kv_group_size=3)
        self.assertTrue(report['verified'])
        self.assertLessEqual(report['execution_error']['max_abs_error'], 1e-5)
        for field in ('quantization_error', 'kv_quantization_error', 'combined_quantization_error'):
            self.assertNotIn('passed', report[field])
            self.assertAlmostEqual(report[field]['max_abs_error'], Q4_KV_GOLDEN['errors'][field]['max_abs_error'], places=7)
        without_source = verify_bundle_forward(path, settings['token_ids'], actual, kv_group_size=3)
        self.assertIsNone(without_source['quantization_error'])
        self.assertNotIn('combined_quantization_error', without_source)
        self.assertGreater(without_source['kv_quantization_error']['max_abs_error'], 0)
        config, weights = load_bundle_weights(path)
        f32_logits = TorchLlamaReference(config, weights).prefill(settings['token_ids'])
        legacy = verify_bundle_forward(path, settings['token_ids'], f32_logits, source_dir=source)
        self.assertEqual(legacy['reference'], 'pytorch_decoded_q4')
        self.assertNotIn('kv_quantization_error', legacy)
        self.assertNotIn('combined_quantization_error', legacy)
        self.assertEqual(legacy['quantization_error'], report['quantization_error'])
        wrong = verify_bundle_forward(path, settings['token_ids'], actual, source_dir=source)
        self.assertFalse(wrong['verified'], 'Q4 KV must not be checked against a F32 KV execution oracle')
        command = [sys.executable, str(ROOT / 'tools/nexa_run.py'), str(path), '--tokens', '1,3,5',
                   '--decode-tokens', '7', '--kv-cache', '--kv-page-tokens', '3', '--kv-codec', 'q4',
                   '--kv-group-size', '3', '--prefill-chunk-size', '2', '--verify',
                   '--reference-checkpoint', str(source), '--memory-budget', '1MiB']
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        validation = json.loads(completed.stdout)['validation']
        self.assertTrue(validation['verified'])
        self.assertEqual(validation['reference'], 'pytorch_decoded_q4_kv_q4')
        self.assertEqual(validation['kv_group_size'], 3)
        for field in ('quantization_error', 'kv_quantization_error', 'combined_quantization_error'):
            self.assertEqual(validation[field], report[field])

    def test_oracle_validates_groups_and_quantizes_current_token_before_attention(self):
        config, weights = load_bundle_weights(self.path)
        for group in (True, 0, -1, 1.5, MAX_GROUP_SIZE + 1):
            with self.subTest(group=group), self.assertRaises(ValueError):
                TorchLlamaReference(config, weights, kv_group_size=group)
        q4 = TorchLlamaReference(config, weights, kv_group_size=3)
        f32_oracle = TorchLlamaReference(config, weights)
        packed_logits, full_logits = q4.prefill([1]), f32_oracle.prefill([1])
        error = compare_logits(packed_logits, full_logits)
        self.assertGreater(error['max_abs_error'], 0)
        self.assertEqual(str(q4._cache[0][0].dtype), 'torch.float64')
        saved = [[tensor.clone() for tensor in layer] for layer in q4._cache]
        q4.decode(3)
        for old_layer, new_layer in zip(saved, q4._cache):
            for old, new in zip(old_layer, new_layer):
                self.assertTrue((old == new[:1]).all())


if __name__ == '__main__':
    unittest.main()

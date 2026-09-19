"""Native full Llama forward, transactional state, and optional Torch oracle."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.importers.llama import import_llama_checkpoint
from compiler.model_config import ModelConfig
from compiler.planner.memory import MemoryBudgetError
from model_checkpoint_fixture import create_checkpoint
from runtime.nexapack.bundle import ModelBundleReader, write_model_bundle
from runtime.nexapack.format import NexaPackReader, decode_q4_row
from transformer_reference import (
    TorchLlamaReference, compare_logits, llama_forward, load_bundle_weights,
    torch_available, verify_bundle_forward,
)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def random_bundle(path, *, layers=1, heads=2, kv_heads=1, tied=True, seed=728, zero_layers=False):
    config = ModelConfig(name=f'forward_fixture_{seed}', vocab_size=13, hidden_size=heads * 4,
                         intermediate_size=heads * 6, num_hidden_layers=layers,
                         num_attention_heads=heads, num_key_value_heads=kv_heads,
                         max_position_embeddings=12, tie_word_embeddings=tied,
                         rms_norm_eps=1e-5, rope_theta=10000.0)
    rng, values = random.Random(seed), {}
    for name, shape in config.required_tensor_shapes().items():
        if len(shape) == 1:
            values[name] = [1.0 if zero_layers else f32(rng.uniform(0.85, 1.15)) for _ in range(shape[0])]
        else:
            values[name] = [[0.0 if zero_layers and name.startswith('model.layers.') else
                             f32(rng.uniform(-0.45, 0.45)) for _ in range(shape[1])]
                            for _ in range(shape[0])]
    sources = {name: (lambda name=name, shape=shape: iter([values[name]]) if len(shape) == 1 else iter(values[name]))
               for name, shape in config.required_tensor_shapes().items()}
    write_model_bundle(path, config, sources, group_size=3, block_rows=2)
    return config, values


class _TransformerFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which('clang') or shutil.which('cc')):
            raise unittest.SkipTest('C compiler unavailable')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nexa-transformer-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / 'bundle'
        self.config, self.weights = random_bundle(self.path)

    def session(self, path=None, **kwargs):
        from runtime.nexapack.transformer import TransformerSession
        return TransformerSession(path or self.path, memory_budget=kwargs.pop('memory_budget', '1MiB'), **kwargs)

    def assert_rows_close(self, first, second, atol=1e-6):
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertTrue(math.isfinite(a) and math.isfinite(b))
                self.assertLessEqual(abs(a - b), atol + 1e-6 * abs(b))


class NativeTransformerRegressions(_TransformerFixture):
    def test_frozen_pytorch_golden_without_torch_dependency(self):
        golden = json.loads((ROOT / 'tests/fixtures/transformer_tiny_golden.json').read_text())
        source, bundle_path = self.directory / 'golden-source', self.directory / 'golden-bundle'
        create_checkpoint(source, include_tied_head=True)
        import_llama_checkpoint(source, bundle_path, **golden['import_settings'])
        with ModelBundleReader(bundle_path) as bundle:
            tensor_hashes = {name: hashlib.sha256((bundle_path / entry['path']).read_bytes()).hexdigest()
                             for name, entry in bundle.manifest['tensors'].items()}
        pack_hash = hashlib.sha256(json.dumps(tensor_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(pack_hash, golden['pack_sha256'])
        with self.session(bundle_path) as session:
            self.assertEqual(session.config.to_dict(), golden['config'])
            actual = session.prefill(golden['token_ids'])
        self.assertEqual(len(actual), len(golden['logits']))
        for row, expected in zip(actual, golden['logits']):
            self.assertEqual(len(row), len(expected))
            for value, reference in zip(row, expected):
                self.assertLessEqual(abs(value - reference), golden['atol'] + golden['rtol'] * abs(reference))

    def test_causal_prefix_and_decode_match_full_forward_without_torch(self):
        with self.session(tile_rows=3) as session:
            initial = session.prefill([1, 3])
            decoded = [session.decode(token) for token in (2, 7, 4)]
            self.assertEqual(session.token_ids, (1, 3, 2, 7, 4))
            complete = session.prefill([1, 3, 2, 7, 4])
            self.assert_rows_close(initial + decoded, complete)
            different_future = session.prefill([1, 3, 8, 5, 9])
            self.assert_rows_close(different_future[:2], complete[:2])
            report = session.report()
            self.assertTrue(report['executed'])
            self.assertFalse(report['persistent_kv_cache'])
            self.assertEqual(report['decode_strategy'], 'full_prefix_recomputation')
            memory = report['memory']
            self.assertLessEqual(memory['managed_buffers_peak_bound_bytes'], memory['budget_bytes'])
            self.assertEqual(memory['full_dequantized_weight_buffer_bytes'], 0)
            self.assertEqual(memory['persistent_kv_bytes'], 0)

    def test_zero_transformer_layers_match_scalar_embedding_norm_head(self):
        path = self.directory / 'zero-layers'
        config, _ = random_bundle(path, layers=2, zero_layers=True)
        with ModelBundleReader(path) as bundle, bundle.open_q4('model.embed_tokens.weight') as packed:
            embedding = [decode_q4_row(packed.read_rows(row, 1), packed.cols, packed.group_size)
                         for row in range(packed.rows)]
        expected = []
        for token in (1, 5, 2):
            values = [f32(value) for value in embedding[token]]
            scale = 1 / math.sqrt(sum(value * value for value in values) / config.hidden_size + config.rms_norm_eps)
            normalized = [f32(value * scale) for value in values]
            expected.append([f32(sum(value * weight for value, weight in zip(normalized, row))) for row in embedding])
        with self.session(path) as session:
            self.assert_rows_close(session.prefill([1, 5, 2]), expected)

    def test_tied_alias_equals_explicit_identical_head_and_tile_sizes(self):
        separate_path = self.directory / 'untied'
        untied_config = replace(self.config, tie_word_embeddings=False)
        values = dict(self.weights, **{'lm_head.weight': self.weights['model.embed_tokens.weight']})
        sources = {name: (lambda name=name, shape=shape: iter([values[name]]) if len(shape) == 1 else iter(values[name]))
                   for name, shape in untied_config.required_tensor_shapes().items()}
        write_model_bundle(separate_path, untied_config, sources, group_size=3, block_rows=2)
        with self.session(tile_rows=1) as tied, self.session(separate_path, tile_rows=5) as untied:
            self.assert_rows_close(tied.prefill([1, 8, 3]), untied.prefill([1, 8, 3]))

    def test_budget_rejection_precedes_payload_and_kernel_loading(self):
        with patch.object(NexaPackReader, 'read_rows_into', side_effect=AssertionError('Q4 payload read')):
            with patch.object(ModelBundleReader, 'read_f32_into', side_effect=AssertionError('norm payload read')):
                with patch('runtime.nexapack.transformer._load_kernels', side_effect=AssertionError('native loaded')):
                    with self.assertRaises(MemoryBudgetError):
                        self.session(memory_budget='1KiB')

    def test_invalid_ids_capacity_reset_and_failure_preserve_history(self):
        with self.session(max_sequence_length=4) as session:
            with self.assertRaises(ValueError):
                session.decode(1)
            initial = session.prefill([1, 2])
            self.assertEqual(session.token_ids, (1, 2))
            for tokens in ([], [True], [-1], [self.config.vocab_size], [1.5], [1, 2, 3, 4, 5]):
                with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                    session.prefill(tokens)
                self.assertEqual(session.token_ids, (1, 2))
            for token in (True, -1, self.config.vocab_size, 1.5):
                with self.subTest(token=token), self.assertRaises(ValueError):
                    session.decode(token)
                self.assertEqual(session.token_ids, (1, 2))
            session.prefill([1, 2, 3, 4])
            with self.assertRaises(ValueError):
                session.decode(5)
            self.assertEqual(session.token_ids, (1, 2, 3, 4))
            session.reset()
            self.assertEqual(session.token_ids, ())
            self.assertFalse(session.report()['executed'])
            self.assert_rows_close(session.prefill([1, 2]), initial)
        session.close()
        with self.assertRaises(ValueError):
            session.prefill([1])

    def test_payload_error_does_not_commit_history_or_report(self):
        with self.session() as session:
            session.prefill([1, 2])
            previous_report = session.report()
            with ModelBundleReader(self.path) as bundle:
                entry = bundle.manifest['tensors']['model.layers.0.self_attn.q_proj.weight']
                path = self.path / entry['path']
            original = path.read_bytes()
            with NexaPackReader(path) as reader:
                offset = reader.metadata['blocks'][0]['offset']
            damaged = bytearray(original)
            damaged[offset] ^= 1
            path.write_bytes(damaged)
            with self.assertRaises(ValueError):
                session.decode(3)
            self.assertEqual(session.token_ids, (1, 2))
            self.assertEqual(session.report(), previous_report)
            path.write_bytes(original)
            resumed = session.decode(3)
            with self.session() as fresh:
                self.assert_rows_close([resumed], [fresh.prefill([1, 2, 3])[-1]])


@unittest.skipUnless(torch_available(), 'Optional PyTorch reference is unavailable')
class TorchTransformerRegressions(_TransformerFixture):
    def test_tiny_checkpoint_separates_execution_and_quantization_error(self):
        source, bundle_path = self.directory / 'source', self.directory / 'tiny-bundle'
        create_checkpoint(source, include_tied_head=True)
        import_llama_checkpoint(source, bundle_path, group_size=4, block_rows=3)
        tokens = [1, 4, 3, 8, 2]
        with self.session(bundle_path) as session:
            logits = session.prefill(tokens)
        report = verify_bundle_forward(bundle_path, tokens, logits, source_dir=source)
        self.assertTrue(report['verified'], report)
        self.assertLess(report['execution_error']['max_abs_error'], 1e-5)
        self.assertGreater(report['quantization_error']['max_abs_error'], 1e-5)
        self.assertFalse(report['quality_or_perplexity_measured'])

    def test_two_layers_untied_mha_and_gqa_against_independent_oracle(self):
        cases = [(2, 4, 1, True), (2, 2, 2, False), (2, 4, 2, False)]
        for index, (layers, heads, kv_heads, tied) in enumerate(cases):
            with self.subTest(layers=layers, heads=heads, kv_heads=kv_heads, tied=tied):
                path = self.directory / f'case-{index}'
                random_bundle(path, layers=layers, heads=heads, kv_heads=kv_heads, tied=tied, seed=900 + index)
                config, weights = load_bundle_weights(path)
                tokens = [1, 5, 2, 0, 3, 6]
                expected = llama_forward(config, weights, tokens)
                with self.session(path, tile_rows=3) as session:
                    actual = session.prefill(tokens)
                comparison = compare_logits(actual, expected)
                self.assertTrue(comparison['passed'], comparison)
                self.assertLess(comparison['max_abs_error'], 1e-5)

    def test_quantization_comparison_rejects_different_source_weights(self):
        source, bundle_path = self.directory / 'source', self.directory / 'tiny-bundle'
        create_checkpoint(source)
        import_llama_checkpoint(source, bundle_path, group_size=4, block_rows=3)
        tokens = [1, 3]
        with self.session(bundle_path) as session:
            actual = session.prefill(tokens)
        shard = next(source.glob('*.safetensors'))
        content = bytearray(shard.read_bytes())
        header_bytes, = struct.unpack('<Q', content[:8])
        content[8 + header_bytes] ^= 1
        shard.write_bytes(content)
        with self.assertRaisesRegex(ValueError, 'source provenance'):
            verify_bundle_forward(bundle_path, tokens, actual, source_dir=source)

    def test_native_recomputation_matches_independent_incremental_kv(self):
        path = self.directory / 'kv-case'
        random_bundle(path, layers=2, heads=4, kv_heads=2, tied=False, seed=400)
        config, weights = load_bundle_weights(path)
        reference = TorchLlamaReference(config, weights)
        with self.session(path, max_sequence_length=8) as native:
            self.assertTrue(compare_logits(native.prefill([1, 3]), reference.prefill([1, 3]))['passed'])
            tokens = [1, 3]
            for token in (5, 2, 7, 4):
                tokens.append(token)
                cached = reference.decode(token)
                recomputed = native.decode(token)
                expected = llama_forward(config, weights, tokens)[-1]
                self.assertTrue(compare_logits(recomputed, cached)['passed'])
                self.assertTrue(compare_logits(cached, expected)['passed'])
            self.assertEqual(reference.token_ids, native.token_ids)


if __name__ == '__main__':
    unittest.main()

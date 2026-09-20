"""PL.03a: low-rank adapters over a frozen packed base, on the CPU executor.

Three tests carry this file. The null adapter has to be bit-identical to the
baseline, the non-null adapter has to *differ* from it, and the non-null result
has to match a float64 oracle that shares no decoder and no kernel with the
runtime. Drop any one of the three and a delta that is never summed, or a
delta computed by the same code twice, would still pass.
"""
import json
import math
from pathlib import Path
import random
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.model_config import ModelConfig
from compiler.model_plasticity import ModelPlasticityConfig, ParameterRegion, resolve_plasticity_map
from runtime.learning.adapter import (
    ACCUMULATION_ORDER, AdapterError, AdapterSet, AdapterSpec, MAX_ADAPTER_RANK,
    SEMANTICS, matmul_weight_targets, write_adapter_payload,
)
from runtime.nexapack.bundle import write_model_bundle

from adapter_reference import (
    LocalLlamaReference, compare_logits, load_bundle_weights, read_adapter_payload,
)

TOKENS = [1, 4, 2, 7]


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def build_bundle(path, *, tied=True, seed=91):
    config = ModelConfig(name=f'adapter_fixture_{seed}', vocab_size=13, hidden_size=8,
                         intermediate_size=12, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=12,
                         tie_word_embeddings=tied, rms_norm_eps=1e-5, rope_theta=10000.0)
    rng, values = random.Random(seed), {}
    for name, shape in config.required_tensor_shapes().items():
        if len(shape) == 1:
            values[name] = [f32(rng.uniform(0.85, 1.15)) for _ in range(shape[0])]
        else:
            values[name] = [[f32(rng.uniform(-0.45, 0.45)) for _ in range(shape[1])]
                            for _ in range(shape[0])]
    sources = {name: (lambda name=name, shape=shape:
                      iter([values[name]]) if len(shape) == 1 else iter(values[name]))
               for name, shape in config.required_tensor_shapes().items()}
    write_model_bundle(path, config, sources, group_size=4, block_rows=4)
    return config


class _AdapterFixture(unittest.TestCase):
    TARGET = 'model.layers.0.mlp.up_proj.weight'
    RANK = 2

    @classmethod
    def setUpClass(cls):
        if not (shutil.which('clang') or shutil.which('cc')):
            raise unittest.SkipTest('C compiler unavailable')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nexa-adapter-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / 'bundle'
        self.config = build_bundle(self.path)
        self.rows, self.cols = self.config.required_tensor_shapes()[self.TARGET]

    def session(self, **kwargs):
        from runtime.nexapack.transformer import TransformerSession
        return TransformerSession(self.path, memory_budget=kwargs.pop('memory_budget', '4MiB'),
                                  **kwargs)

    def payload(self, name, *, zero_b=False, magnitude=0.3, seed=5):
        rng = random.Random(seed)
        a_rows = [[f32(rng.uniform(-magnitude, magnitude)) for _ in range(self.cols)]
                  for _ in range(self.RANK)]
        b_rows = [[0.0 if zero_b else f32(rng.uniform(-magnitude, magnitude))
                   for _ in range(self.RANK)] for _ in range(self.rows)]
        destination = self.directory / name
        write_adapter_payload(destination, a_rows, b_rows)
        return destination, a_rows, b_rows

    def spec(self, payload_path, *, alpha=1.0, **overrides):
        values = {'id': 'adapter-0', 'target': self.TARGET, 'rank': self.RANK, 'alpha': alpha,
                  'in_features': self.cols, 'out_features': self.rows,
                  'payload_path': str(payload_path), 'order': 0, 'precision': 'f32'}
        values.update(overrides)
        return AdapterSpec(**values)

    def run_prefill(self, adapters=None):
        with self.session(adapters=adapters) as session:
            logits = session.prefill(TOKENS)
            return [list(row) for row in logits], session.report()


class NullAdapterIsByteIdentical(_AdapterFixture):
    def test_zero_b_and_zero_alpha_reproduce_the_baseline_digest(self):
        _, baseline = self.run_prefill()
        zero_b, _, _ = self.payload('zero-b.bin', zero_b=True)
        nonzero, _, _ = self.payload('nonzero.bin')
        for label, path, alpha in (('B = 0', zero_b, 1.0), ('alpha = 0', nonzero, 0.0)):
            with self.subTest(null=label):
                _, report = self.run_prefill(AdapterSet([self.spec(path, alpha=alpha)]))
                self.assertEqual(report['logits_sha256'], baseline['logits_sha256'])
                self.assertEqual(report['io']['adapter_payload_bytes_read'],
                                 self.RANK * (self.cols + self.rows) * 4)

    def test_zero_adapters_keep_the_baseline_plan_and_report_shape(self):
        _, baseline = self.run_prefill()
        _, empty = self.run_prefill(AdapterSet([]))
        self.assertEqual(empty['memory_plan'], baseline['memory_plan'])
        self.assertEqual(empty['io'], baseline['io'])
        self.assertNotIn('adapters', empty)
        self.assertNotIn('adapter_payload_bytes_read', empty['io'])
        self.assertEqual(empty['logits_sha256'], baseline['logits_sha256'])

    def test_a_bound_adapter_does_add_buffers_to_the_plan(self):
        """The previous test would also pass if adapters never reached the plan."""
        _, baseline = self.run_prefill()
        path, _, _ = self.payload('nonzero.bin')
        _, adapted = self.run_prefill(AdapterSet([self.spec(path)]))
        added = set(adapted['memory_plan']['allocations']) - set(baseline['memory_plan']['allocations'])
        self.assertEqual(added, {'__adapter_payload', '__adapter_low', '__adapter_delta'})
        self.assertEqual(adapted['memory_plan']['allocations']['__adapter_payload']['size_bytes'],
                         self.RANK * (self.cols + self.rows) * 4)


class NonNullAdapterChangesTheOutput(_AdapterFixture):
    def test_nonzero_adapter_differs_from_the_baseline(self):
        baseline_logits, baseline = self.run_prefill()
        path, _, _ = self.payload('nonzero.bin')
        adapted_logits, adapted = self.run_prefill(AdapterSet([self.spec(path)]))
        self.assertNotEqual(adapted['logits_sha256'], baseline['logits_sha256'])
        deviation = max(abs(a - b) for first, second in zip(adapted_logits, baseline_logits)
                        for a, b in zip(first, second))
        # Well above the 1e-5 the oracle comparison tolerates, so the oracle
        # agreement below is a real agreement and not tolerance absorbing zero.
        self.assertGreater(deviation, 1e-3)

    def test_runtime_matches_the_independent_float64_oracle(self):
        path, a_rows, b_rows = self.payload('nonzero.bin')
        adapted_logits, _ = self.run_prefill(AdapterSet([self.spec(path)]))
        baseline_logits, _ = self.run_prefill()
        config, weights = load_bundle_weights(self.path)
        self.assertEqual(config.to_dict(), self.config.to_dict())
        read_a, read_b = read_adapter_payload(path, self.RANK, self.cols, self.rows)
        self.assertEqual((read_a, read_b), (a_rows, b_rows))

        plain = LocalLlamaReference(config, weights).prefill(TOKENS)
        adapted = LocalLlamaReference(config, weights, adapters={
            self.TARGET: [(self.RANK, 1.0, read_a, read_b)]}).prefill(TOKENS)

        self.assertTrue(compare_logits(baseline_logits, plain)['passed'])
        agreement = compare_logits(adapted_logits, adapted)
        self.assertTrue(agreement['passed'], agreement)
        # The oracle has to see the same delta the runtime applied: if it did
        # not, the adapted run would still agree with the un-adapted oracle.
        self.assertFalse(compare_logits(adapted_logits, plain)['passed'])
        self.assertFalse(compare_logits(baseline_logits, adapted)['passed'])

    def test_alpha_scales_the_delta_as_declared(self):
        path, a_rows, b_rows = self.payload('nonzero.bin')
        config, weights = load_bundle_weights(self.path)
        for alpha in (0.5, 2.0):
            with self.subTest(alpha=alpha):
                logits, _ = self.run_prefill(AdapterSet([self.spec(path, alpha=alpha)]))
                expected = LocalLlamaReference(config, weights, adapters={
                    self.TARGET: [(self.RANK, alpha, a_rows, b_rows)]}).prefill(TOKENS)
                self.assertTrue(compare_logits(logits, expected)['passed'])

    def test_two_targets_compose_and_each_one_contributes(self):
        other = 'model.layers.0.self_attn.o_proj.weight'
        other_rows, other_cols = self.config.required_tensor_shapes()[other]
        rng = random.Random(17)
        other_path = self.directory / 'other.bin'
        other_a = [[f32(rng.uniform(-0.3, 0.3)) for _ in range(other_cols)]
                   for _ in range(self.RANK)]
        other_b = [[f32(rng.uniform(-0.3, 0.3)) for _ in range(self.RANK)]
                   for _ in range(other_rows)]
        write_adapter_payload(other_path, other_a, other_b)
        first_path, a_rows, b_rows = self.payload('nonzero.bin')
        first = self.spec(first_path)
        second = AdapterSpec(id='adapter-1', target=other, rank=self.RANK, alpha=1.0,
                             in_features=other_cols, out_features=other_rows,
                             payload_path=str(other_path), order=1)
        both_logits, both = self.run_prefill(AdapterSet([second, first]))
        self.assertEqual([entry['id'] for entry in both['adapters']['bound_targets']],
                         ['adapter-0', 'adapter-1'])
        self.assertEqual(both['io']['adapter_payload_bytes_read'],
                         self.RANK * (self.cols + self.rows) * 4
                         + self.RANK * (other_cols + other_rows) * 4)
        config, weights = load_bundle_weights(self.path)
        expected = LocalLlamaReference(config, weights, adapters={
            self.TARGET: [(self.RANK, 1.0, a_rows, b_rows)],
            other: [(self.RANK, 1.0, other_a, other_b)]}).prefill(TOKENS)
        self.assertTrue(compare_logits(both_logits, expected)['passed'])
        only_first, _ = self.run_prefill(AdapterSet([first]))
        self.assertFalse(compare_logits(both_logits, only_first)['passed'])

    def test_decode_after_prefill_keeps_applying_the_adapter(self):
        path, a_rows, b_rows = self.payload('nonzero.bin')
        with self.session(adapters=AdapterSet([self.spec(path)])) as session:
            session.prefill(TOKENS)
            adapted = list(session.decode(3))
            adapter_report = session.report()
        with self.session() as session:
            session.prefill(TOKENS)
            baseline = list(session.decode(3))
        self.assertFalse(compare_logits([adapted], [baseline])['passed'])
        config, weights = load_bundle_weights(self.path)
        expected = LocalLlamaReference(config, weights, adapters={
            self.TARGET: [(self.RANK, 1.0, a_rows, b_rows)]}).prefill(TOKENS + [3])
        self.assertTrue(compare_logits([adapted], [expected[-1]])['passed'])
        self.assertEqual(adapter_report['io']['adapter_payload_bytes_read'],
                         self.RANK * (self.cols + self.rows) * 4)


class AdapterPayloadAccounting(_AdapterFixture):
    def test_payload_bytes_read_is_exactly_rank_times_cols_plus_rows(self):
        path, _, _ = self.payload('nonzero.bin')
        spec = self.spec(path)
        self.assertEqual(spec.payload_bytes, self.RANK * (self.cols + self.rows) * 4)
        self.assertEqual(path.stat().st_size, spec.payload_bytes)
        _, report = self.run_prefill(AdapterSet([spec]))
        self.assertEqual(report['io']['adapter_payload_bytes_read'], spec.payload_bytes)
        self.assertEqual(report['adapters']['payload_bytes_per_pass'], spec.payload_bytes)

    def test_report_states_the_fixed_semantics(self):
        path, _, _ = self.payload('nonzero.bin')
        _, report = self.run_prefill(AdapterSet([self.spec(path)]))
        self.assertEqual(report['adapters']['semantics'], SEMANTICS)
        self.assertEqual(report['adapters']['accumulation_order'], ACCUMULATION_ORDER)
        self.assertIn('alpha / rank', SEMANTICS)
        json.dumps(report, allow_nan=False)


class AdapterValidationRefusesBeforeReading(_AdapterFixture):
    def bind(self, spec):
        from runtime.nexapack.bundle import ModelBundleReader
        with ModelBundleReader(self.path) as bundle:
            return AdapterSet([spec]).bind(bundle)

    def test_missing_payload_never_hides_a_declaration_error(self):
        """Each refusal below names the declaration, not the unreadable file."""
        absent = self.directory / 'does-not-exist.bin'
        with self.assertRaises(AdapterError) as caught:
            self.spec(absent, rank=0)
        self.assertIn('rank', str(caught.exception))
        with self.assertRaises(AdapterError) as caught:
            self.spec(absent, rank=MAX_ADAPTER_RANK + 1)
        self.assertIn('rank', str(caught.exception))
        with self.assertRaises(AdapterError) as caught:
            self.bind(self.spec(absent, in_features=self.cols + 1))
        self.assertIn('stores shape', str(caught.exception))
        with self.assertRaises(AdapterError) as caught:
            self.bind(self.spec(absent, out_features=self.rows + 3))
        self.assertIn('stores shape', str(caught.exception))
        with self.assertRaises(AdapterError) as caught:
            self.bind(self.spec(absent, target='model.layers.0.mlp.nope.weight'))
        self.assertIn('unknown tensor', str(caught.exception))

    def test_nonfinite_alpha_and_wrong_precision_are_refused(self):
        path, _, _ = self.payload('nonzero.bin')
        for alpha in (float('inf'), float('nan')):
            with self.assertRaises(AdapterError):
                self.spec(path, alpha=alpha)
        with self.assertRaises(AdapterError):
            self.spec(path, precision='f16')
        with self.assertRaises(AdapterError):
            self.spec(path, precision='q4')

    def test_embedding_and_output_head_are_refused_in_v1(self):
        path, _, _ = self.payload('nonzero.bin')
        for tied in (True, False):
            with self.subTest(tied=tied):
                bundle_path = self.directory / f'head-{tied}'
                config = build_bundle(bundle_path, tied=tied, seed=23)
                from runtime.nexapack.bundle import ModelBundleReader
                shapes = config.required_tensor_shapes()
                targets = ['model.embed_tokens.weight']
                if not tied:
                    targets.append('lm_head.weight')
                for target in targets:
                    rows, cols = shapes[target]
                    spec = AdapterSpec(id='x', target=target, rank=2, alpha=1.0,
                                       in_features=cols, out_features=rows,
                                       payload_path=str(path), order=0)
                    with ModelBundleReader(bundle_path) as bundle:
                        with self.assertRaises(AdapterError) as caught:
                            AdapterSet([spec]).bind(bundle)
                    self.assertIn('MatMul projection', str(caught.exception))

    def test_tied_alias_name_is_refused_as_a_target(self):
        path, _, _ = self.payload('nonzero.bin')
        spec = AdapterSpec(id='x', target='lm_head.weight', rank=2, alpha=1.0,
                           in_features=8, out_features=13, payload_path=str(path), order=0)
        with self.assertRaises(AdapterError) as caught:
            self.bind(spec)
        self.assertIn('alias', str(caught.exception))

    def test_allowed_targets_come_from_the_lowered_graph(self):
        tied, untied = self.config, build_bundle(self.directory / 'untied', tied=False, seed=31)
        for config in (tied, untied):
            with self.subTest(tied=config.tie_word_embeddings):
                allowed = matmul_weight_targets(config)
                self.assertNotIn('model.embed_tokens.weight', allowed)
                self.assertNotIn('lm_head.weight', allowed)
                self.assertEqual(len(allowed), 7)
                self.assertIn(self.TARGET, allowed)

    def test_payload_size_mismatch_is_refused_before_execution(self):
        path, a_rows, b_rows = self.payload('nonzero.bin')
        truncated = self.directory / 'short.bin'
        truncated.write_bytes(path.read_bytes()[:-8])
        with self.assertRaises(AdapterError) as caught:
            self.bind(self.spec(truncated))
        self.assertIn('bytes', str(caught.exception))

    def test_nonfinite_payload_is_refused_at_read_time(self):
        path, _, _ = self.payload('nonzero.bin')
        broken = self.directory / 'broken.bin'
        data = bytearray(path.read_bytes())
        data[0:4] = struct.pack('<f', float('inf'))
        broken.write_bytes(bytes(data))
        with self.assertRaises(AdapterError):
            self.run_prefill(AdapterSet([self.spec(broken)]))

    def test_duplicate_targets_and_orders_are_refused(self):
        path, _, _ = self.payload('nonzero.bin')
        with self.assertRaises(AdapterError):
            AdapterSet([self.spec(path), self.spec(path, id='adapter-1', order=1)])
        with self.assertRaises(AdapterError):
            AdapterSet([self.spec(path),
                        self.spec(path, id='adapter-1',
                                  target='model.layers.0.mlp.gate_proj.weight')])

    def test_rank_above_the_smaller_dimension_is_refused(self):
        path, _, _ = self.payload('nonzero.bin')
        with self.assertRaises(AdapterError) as caught:
            self.bind(self.spec(path, rank=min(self.rows, self.cols) + 1))
        self.assertIn('exceeds', str(caught.exception))

    def test_a_non_adapter_object_is_refused_by_the_session(self):
        with self.assertRaises(ValueError):
            self.run_prefill([{'target': self.TARGET}])


class PlasticityConfigLeavesTheSessionAlone(_AdapterFixture):
    def test_memory_plan_is_identical_with_and_without_a_plasticity_config(self):
        _, without = self.run_prefill()
        plasticity = ModelPlasticityConfig(self.config.name, (
            ParameterRegion(id='core', domain='base', region_class='stable_core',
                            plasticity=0.0, maturity=1.0, protected=True,
                            tensors=tuple(sorted(self.config.required_tensor_shapes())),
                            provenance={'source': 'test'}),))
        resolved = resolve_plasticity_map(self.config, plasticity)
        self.assertEqual(resolved.covered_scalars, self.config.parameter_count())
        _, with_config = self.run_prefill()
        self.assertEqual(with_config['memory_plan'], without['memory_plan'])
        self.assertEqual(with_config['logits_sha256'], without['logits_sha256'])
        self.assertEqual(with_config['io'], without['io'])


class LocalOracleIsIndependent(unittest.TestCase):
    def test_the_oracle_does_not_import_the_runtime_codec(self):
        source = (ROOT / 'tests/adapter_reference.py').read_text()
        self.assertNotIn('decode_q4_row(', source.replace('decode_q4_row_local(', ''))
        self.assertNotIn('from runtime.nexapack.format', source)
        self.assertNotIn('import torch', source)

    def test_the_local_q4_decoder_rejects_the_reserved_code_and_bad_padding(self):
        from adapter_reference import decode_q4_row_local, q4_row_bytes
        self.assertEqual(q4_row_bytes(5, 4), 2 * (4 + 2))
        row = struct.pack('<f', 0.25) + bytes([0x08, 0x00]) + struct.pack('<f', 0.0) + bytes([0, 0])
        with self.assertRaises(ValueError):
            decode_q4_row_local(row, 5, 4)
        good = struct.pack('<f', 0.25) + bytes([0x21, 0x03]) + struct.pack('<f', 0.5) + bytes([0x01, 0])
        values = decode_q4_row_local(good, 5, 4)
        self.assertEqual(values, [0.25, 0.5, 0.75, 0.0, 0.5])
        zero_scale = struct.pack('<f', 0.0) + bytes([0x01, 0x00]) + struct.pack('<f', 0.5) + bytes([0x01, 0])
        with self.assertRaises(ValueError):
            decode_q4_row_local(zero_scale, 5, 4)

    def test_the_oracle_rounds_to_float32_at_operator_boundaries(self):
        from adapter_reference import f32
        self.assertEqual(f32(0.1), struct.unpack('<f', struct.pack('<f', 0.1))[0])
        self.assertNotEqual(f32(0.1), 0.1)
        with self.assertRaises(ArithmeticError):
            f32(1e39)
        self.assertTrue(math.isfinite(f32(-0.0)))


if __name__ == '__main__':
    unittest.main()

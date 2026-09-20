"""PL.01a: parameter regions resolved against a real physical model.

The load-bearing test here is the partition against ModelConfig.parameter_count(),
an oracle that predates this module, was written for a different purpose and
never calls it. A schema that only agreed with itself would still pass a
round-trip test; it cannot pass that one, nor the tied/untied pair below, where
the same JSON document has to be rejected for one model and accepted for another.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.model_config import ModelConfig
from compiler.model_ir import DType, TensorDesc
from compiler.model_lowering import derive_activation_requests, lower_model
from compiler.model_plasticity import (
    MAX_REGIONS, ModelPlasticityConfig, ParameterRegion, PlasticityMap, SCHEMA_VERSION,
    TensorSpan, protected_tensors, resolve_plasticity_map, validate_learning_target,
)

MODEL_NAME = 'plasticity_fixture'


def tiny_config(*, tied=True):
    """Same shapes as the repository's tiny checkpoint fixture, same name for both."""
    return ModelConfig(name=MODEL_NAME, vocab_size=16, hidden_size=8, intermediate_size=16,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                       max_position_embeddings=8, tie_word_embeddings=tied)


def region(identifier, tensors, **overrides):
    values = {'id': identifier, 'domain': 'fixture', 'region_class': 'plastic_expert',
              'plasticity': 0.5, 'maturity': 0.25, 'protected': False,
              'tensors': tuple(TensorSpan.coerce(span) for span in tensors),
              'provenance': {'source': 'unit test'}}
    values.update(overrides)
    return ParameterRegion(**values)


ALIAS_PAIR_JSON = """
{
  "schema_version": 1,
  "model_name": "plasticity_fixture",
  "regions": [
    {"id": "head", "domain": "output", "region_class": "adapter_delta",
     "plasticity": 0.4, "maturity": 0.1, "protected": false,
     "tensors": [{"tensor": "lm_head.weight", "start_row": null, "row_count": null}],
     "provenance": {"note": "named through the alias"}},
    {"id": "embeddings", "domain": "input", "region_class": "stable_core",
     "plasticity": 0.0, "maturity": 1.0, "protected": true,
     "tensors": [{"tensor": "model.embed_tokens.weight", "start_row": null, "row_count": null}],
     "provenance": {"note": "named physically"}}
  ]
}
"""


class ParameterRegionSchema(unittest.TestCase):
    def test_round_trip_preserves_every_declared_field(self):
        config = ModelPlasticityConfig(MODEL_NAME, (
            region('core', ['model.norm.weight'], region_class='stable_core',
                   plasticity=0.0, protected=True),
            region('tail', [TensorSpan('model.layers.0.mlp.down_proj.weight', 2, 3)]),
        ))
        restored = ModelPlasticityConfig.from_json(config.to_json())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.regions[1].tensors[0], TensorSpan(
            'model.layers.0.mlp.down_proj.weight', 2, 3))
        self.assertEqual(config.to_dict()['schema_version'], SCHEMA_VERSION)

    def test_json_keys_are_sorted_and_nonfinite_numbers_are_refused(self):
        config = ModelPlasticityConfig(MODEL_NAME, (region('a', ['model.norm.weight']),))
        text = config.to_json()
        self.assertLess(text.index('"model_name"'), text.index('"regions"'))
        self.assertLess(text.index('"regions"'), text.index('"schema_version"'))
        for literal in ('NaN', 'Infinity', '-Infinity'):
            broken = text.replace('0.5', literal, 1)
            with self.assertRaises(ValueError):
                ModelPlasticityConfig.from_json(broken)

    def test_duplicate_json_object_keys_are_refused(self):
        text = ('{"schema_version": 1, "model_name": "a", "model_name": "b", "regions": []}')
        with self.assertRaises(ValueError):
            ModelPlasticityConfig.from_json(text)

    def test_booleans_are_not_accepted_where_a_number_is_required(self):
        with self.assertRaises(ValueError):
            region('a', ['model.norm.weight'], plasticity=True)
        with self.assertRaises(ValueError):
            region('a', ['model.norm.weight'], maturity=False)
        with self.assertRaises(ValueError):
            region('a', ['model.norm.weight'], protected=1)

    def test_out_of_range_and_unknown_vocabulary_are_refused(self):
        with self.assertRaises(ValueError):
            region('a', ['model.norm.weight'], plasticity=1.5)
        with self.assertRaises(ValueError):
            region('a', ['model.norm.weight'], maturity=-0.0001)
        with self.assertRaises(ValueError):
            region('a', ['model.norm.weight'], region_class='frozen')
        with self.assertRaises(ValueError):
            region('a', [])

    def test_protected_region_may_not_also_declare_plasticity(self):
        with self.assertRaises(ValueError):
            region('a', ['model.norm.weight'], protected=True, plasticity=0.2)
        self.assertTrue(region('a', ['model.norm.weight'], protected=True,
                               plasticity=0.0).protected)

    def test_unsupported_schema_version_is_refused(self):
        data = ModelPlasticityConfig(MODEL_NAME, ()).to_dict()
        data['schema_version'] = SCHEMA_VERSION + 1
        with self.assertRaises(ValueError):
            ModelPlasticityConfig.from_dict(data)

    def test_omitted_fields_stay_omitted(self):
        """The five unfalsifiable fields are absent on purpose, not forgotten."""
        keys = set(region('a', ['model.norm.weight']).to_dict())
        for absent in ('importance', 'drift_budget', 'update_count', 'last_update', 'residency'):
            self.assertNotIn(absent, keys)

    def test_repeated_ids_and_spans_are_refused(self):
        with self.assertRaises(ValueError):
            ModelPlasticityConfig(MODEL_NAME, (region('a', ['model.norm.weight']),
                                               region('a', ['model.embed_tokens.weight'])))
        with self.assertRaises(ValueError):
            region('a', ['model.norm.weight', 'model.norm.weight'])

    def test_region_count_limit_is_declared(self):
        self.assertEqual(MAX_REGIONS, 4096)
        with self.assertRaises(ValueError):
            ModelPlasticityConfig(MODEL_NAME, tuple(
                region(f'r{index}', ['model.norm.weight']) for index in range(MAX_REGIONS + 1)))


class PlasticityPartition(unittest.TestCase):
    def test_tiny_fixture_inventory_matches_the_parameter_count_oracle(self):
        tied, untied = tiny_config(), tiny_config(tied=False)
        self.assertEqual((tied.parameter_count(), len(tied.required_tensor_shapes())), (728, 11))
        self.assertEqual((untied.parameter_count(), len(untied.required_tensor_shapes())), (856, 12))

    def test_partition_equals_parameter_count_scalar_for_scalar(self):
        config = tiny_config()
        plasticity = ModelPlasticityConfig(MODEL_NAME, (
            region('embeddings', ['model.embed_tokens.weight'], region_class='stable_core',
                   plasticity=0.0, protected=True),
            region('queries', [TensorSpan('model.layers.0.self_attn.q_proj.weight', 0, 4)]),
        ))
        resolved = resolve_plasticity_map(config, plasticity)
        self.assertEqual(resolved.total_scalars, config.parameter_count())
        self.assertEqual(resolved.covered_scalars + resolved.unassigned_scalars,
                         config.parameter_count())
        self.assertEqual(resolved.covered_scalars, 16 * 8 + 4 * 8)
        self.assertEqual(resolved.region('queries').scalars, 32)

    def test_full_physical_cover_leaves_nothing_unassigned(self):
        for tied in (True, False):
            with self.subTest(tied=tied):
                config = tiny_config(tied=tied)
                plasticity = ModelPlasticityConfig(MODEL_NAME, tuple(
                    region(f'r{index}', [name])
                    for index, name in enumerate(sorted(config.required_tensor_shapes()))))
                resolved = resolve_plasticity_map(config, plasticity)
                self.assertEqual(resolved.unassigned_scalars, 0)
                self.assertEqual(resolved.covered_scalars, config.parameter_count())
                self.assertEqual(resolved.total_scalars, config.parameter_count())

    def test_row_ranges_partition_one_tensor_without_gaps(self):
        config = tiny_config()
        plasticity = ModelPlasticityConfig(MODEL_NAME, (
            region('low', [TensorSpan('model.embed_tokens.weight', 0, 6)]),
            region('high', [TensorSpan('model.embed_tokens.weight', 6, 10)]),
        ))
        resolved = resolve_plasticity_map(config, plasticity)
        self.assertEqual(resolved.covered_scalars, 16 * 8)
        self.assertEqual(resolved.covered_scalars + resolved.unassigned_scalars,
                         config.parameter_count())
        self.assertNotIn('model.embed_tokens.weight',
                         {span.tensor for span in resolved.unassigned})

    def test_unassigned_is_the_real_complement_and_not_a_subtraction(self):
        """Derived by walking the gaps; total - covered would be a tautology."""
        config = tiny_config()
        plasticity = ModelPlasticityConfig(MODEL_NAME, (
            region('middle', [TensorSpan('model.embed_tokens.weight', 4, 5)]),))
        resolved = resolve_plasticity_map(config, plasticity)
        embedding = [span for span in resolved.unassigned
                     if span.tensor == 'model.embed_tokens.weight']
        self.assertEqual([(span.start_row, span.row_count, span.scalars) for span in embedding],
                         [(0, 4, 32), (9, 7, 56)])
        self.assertEqual(sum(span.scalars for span in resolved.unassigned),
                         resolved.unassigned_scalars)
        self.assertEqual(resolved.covered_scalars, 40)

    def test_row_range_past_the_end_is_refused(self):
        config = tiny_config()
        plasticity = ModelPlasticityConfig(MODEL_NAME, (
            region('over', [TensorSpan('model.embed_tokens.weight', 12, 8)]),))
        with self.assertRaises(ValueError) as caught:
            resolve_plasticity_map(config, plasticity)
        self.assertIn('exceeds', str(caught.exception))

    def test_unknown_tensor_is_refused(self):
        plasticity = ModelPlasticityConfig(MODEL_NAME, (region('x', ['model.layers.9.mlp.up_proj.weight']),))
        with self.assertRaises(ValueError):
            resolve_plasticity_map(tiny_config(), plasticity)

    def test_config_for_another_model_is_refused(self):
        other = ModelConfig(name='another_model', vocab_size=16, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            max_position_embeddings=8)
        plasticity = ModelPlasticityConfig(MODEL_NAME, (region('x', ['model.norm.weight']),))
        with self.assertRaises(ValueError):
            resolve_plasticity_map(other, plasticity)


class TiedAliasResolution(unittest.TestCase):
    """The same document, two models, opposite answers."""

    def test_one_document_is_refused_when_tied_and_accepted_when_untied(self):
        plasticity = ModelPlasticityConfig.from_json(ALIAS_PAIR_JSON)
        with self.assertRaises(ValueError) as caught:
            resolve_plasticity_map(tiny_config(tied=True), plasticity)
        message = str(caught.exception)
        self.assertIn('overlap', message)
        self.assertIn('model.embed_tokens.weight', message)

        resolved = resolve_plasticity_map(tiny_config(tied=False), plasticity)
        self.assertEqual(resolved.covered_scalars, 2 * 16 * 8)
        self.assertEqual(resolved.total_scalars, 856)
        self.assertEqual(resolved.covered_scalars + resolved.unassigned_scalars, 856)

    def test_alias_and_physical_name_resolve_to_the_same_storage_when_tied(self):
        config = tiny_config(tied=True)
        through_alias = resolve_plasticity_map(config, ModelPlasticityConfig(
            MODEL_NAME, (region('head', ['lm_head.weight']),)))
        through_physical = resolve_plasticity_map(config, ModelPlasticityConfig(
            MODEL_NAME, (region('head', ['model.embed_tokens.weight']),)))
        self.assertEqual(through_alias.to_dict()['regions'], through_physical.to_dict()['regions'])
        self.assertEqual(through_alias.region('head').spans[0].tensor,
                         'model.embed_tokens.weight')

    def test_alias_is_a_distinct_tensor_when_untied(self):
        config = tiny_config(tied=False)
        resolved = resolve_plasticity_map(config, ModelPlasticityConfig(
            MODEL_NAME, (region('head', ['lm_head.weight']),)))
        self.assertEqual(resolved.region('head').spans[0].tensor, 'lm_head.weight')
        self.assertEqual(resolved.covered_scalars, 16 * 8)

    def test_partial_row_overlap_through_the_alias_is_refused(self):
        config = tiny_config(tied=True)
        plasticity = ModelPlasticityConfig(MODEL_NAME, (
            region('a', [TensorSpan('lm_head.weight', 0, 9)]),
            region('b', [TensorSpan('model.embed_tokens.weight', 8, 8)]),
        ))
        with self.assertRaises(ValueError):
            resolve_plasticity_map(config, plasticity)

    def test_adjacent_row_ranges_through_the_alias_are_accepted(self):
        config = tiny_config(tied=True)
        plasticity = ModelPlasticityConfig(MODEL_NAME, (
            region('a', [TensorSpan('lm_head.weight', 0, 8)]),
            region('b', [TensorSpan('model.embed_tokens.weight', 8, 8)]),
        ))
        resolved = resolve_plasticity_map(config, plasticity)
        self.assertEqual(resolved.covered_scalars, 16 * 8)


class ProtectionAndTargets(unittest.TestCase):
    def test_protected_tensors_refuse_a_learning_target(self):
        config = tiny_config()
        plasticity = ModelPlasticityConfig(MODEL_NAME, (
            region('core', ['model.embed_tokens.weight'], region_class='stable_core',
                   plasticity=0.0, protected=True),
            region('mlp', ['model.layers.0.mlp.up_proj.weight']),
        ))
        resolved = resolve_plasticity_map(config, plasticity)
        self.assertEqual(protected_tensors(resolved), frozenset({'model.embed_tokens.weight'}))
        with self.assertRaises(ValueError):
            validate_learning_target(resolved, 'model.embed_tokens.weight')
        self.assertTrue(validate_learning_target(resolved, 'model.layers.0.mlp.up_proj.weight'))

    def test_protection_follows_the_tied_alias(self):
        """A target named through the alias lands on protected physical rows."""
        config = tiny_config(tied=True)
        resolved = resolve_plasticity_map(config, ModelPlasticityConfig(MODEL_NAME, (
            region('core', ['lm_head.weight'], region_class='stable_core',
                   plasticity=0.0, protected=True),)))
        with self.assertRaises(ValueError):
            validate_learning_target(resolved, 'model.embed_tokens.weight')

    def test_row_range_protection_only_refuses_overlapping_rows(self):
        config = tiny_config()
        resolved = resolve_plasticity_map(config, ModelPlasticityConfig(MODEL_NAME, (
            region('core', [TensorSpan('model.embed_tokens.weight', 0, 4)],
                   region_class='stable_core', plasticity=0.0, protected=True),)))
        with self.assertRaises(ValueError):
            validate_learning_target(resolved, 'model.embed_tokens.weight', rows=(3, 2))
        self.assertTrue(validate_learning_target(resolved, 'model.embed_tokens.weight', rows=(4, 4)))


class BaselineIsUntouched(unittest.TestCase):
    def test_resolution_does_not_alter_the_lowered_graph_or_its_requests(self):
        config = tiny_config()
        before_graph = lower_model(config, 4).to_dict()
        before_requests = [request.to_dict() for request in
                           derive_activation_requests(lower_model(config, 4))]
        resolved = resolve_plasticity_map(config, ModelPlasticityConfig(MODEL_NAME, (
            region('all', sorted(config.required_tensor_shapes())),)))
        self.assertIsInstance(resolved, PlasticityMap)
        self.assertEqual(lower_model(config, 4).to_dict(), before_graph)
        self.assertEqual([request.to_dict() for request in
                          derive_activation_requests(lower_model(config, 4))], before_requests)
        self.assertEqual(config.to_dict(), tiny_config().to_dict())

    def test_weight_storage_overrides_still_lower_unchanged(self):
        config = tiny_config()
        storage = {'model.embed_tokens.weight': TensorDesc(
            'model.embed_tokens.weight', (16, 8), DType.F32, DType.Q4, storage_nbytes=128)}
        graph = lower_model(config, 2, weight_storage=storage).to_dict()
        resolve_plasticity_map(config, ModelPlasticityConfig(MODEL_NAME, (
            region('core', ['model.embed_tokens.weight']),)))
        self.assertEqual(lower_model(config, 2, weight_storage=storage).to_dict(), graph)


if __name__ == '__main__':
    unittest.main()

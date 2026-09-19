"""Physical Llama/NexaLM weight contracts and declarative model fixtures."""
from dataclasses import replace
import json
from math import prod
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.model_config import MAX_MODEL_TENSORS, ModelConfig
from compiler.model_definition import ModelDefinitionError, compile_model_definition


TINY_DEFINITION = """
model Tiny {
    vocab_size: 16;
    hidden_size: 8;
    layers: 1;
    attention {
        query_heads: 2;
        kv_heads: 1;
        head_dim: 4;
        position: rope;
    }
    ffn {
        hidden_size: 16;
        activation: swiglu;
    }
    norm: rmsnorm;
    context: 32;
    tied_embeddings: true;
}
"""


def tiny_config(**overrides):
    return ModelConfig(**{"name": "tiny", "vocab_size": 16, "hidden_size": 8,
                          "intermediate_size": 16, "num_hidden_layers": 1,
                          "num_attention_heads": 2, "num_key_value_heads": 1,
                          "max_position_embeddings": 32, **overrides})


def hf_config(**overrides):
    result = tiny_config().to_dict()
    del result["schema_version"]
    result.update({"model_type": "llama", "architectures": ["LlamaForCausalLM"],
                   "hidden_act": "silu", **overrides})
    return result


class ModelConfigRegressions(unittest.TestCase):
    def test_exact_physical_tensor_contract_for_tiny_gqa(self):
        config = tiny_config()
        self.assertEqual(config.required_tensor_shapes(), {
            "model.embed_tokens.weight": (16, 8),
            "model.layers.0.input_layernorm.weight": (8,),
            "model.layers.0.self_attn.q_proj.weight": (8, 8),
            "model.layers.0.self_attn.k_proj.weight": (4, 8),
            "model.layers.0.self_attn.v_proj.weight": (4, 8),
            "model.layers.0.self_attn.o_proj.weight": (8, 8),
            "model.layers.0.post_attention_layernorm.weight": (8,),
            "model.layers.0.mlp.gate_proj.weight": (16, 8),
            "model.layers.0.mlp.up_proj.weight": (16, 8),
            "model.layers.0.mlp.down_proj.weight": (8, 16),
            "model.norm.weight": (8,),
        })
        self.assertEqual(config.parameter_count(), 728)
        self.assertEqual(config.head_dim, 4)
        self.assertEqual(config.architecture, "llama")

    def test_tied_output_projection_has_one_physical_embedding(self):
        tied = tiny_config()
        untied = replace(tied, tie_word_embeddings=False)
        self.assertEqual(tied.tensor_aliases(), {"lm_head.weight": "model.embed_tokens.weight"})
        self.assertNotIn("lm_head.weight", tied.required_tensor_shapes())
        self.assertEqual(untied.required_tensor_shapes()["lm_head.weight"], (16, 8))
        self.assertEqual(untied.tensor_aliases(), {})
        self.assertEqual(untied.parameter_count(), 856)
        self.assertEqual(untied.parameter_count() - tied.parameter_count(), 16 * 8)
        # Returned contracts are fresh maps, so callers cannot change the config.
        tied.required_tensor_shapes().clear()
        tied.tensor_aliases().clear()
        self.assertEqual(tied.parameter_count(), 728)
        self.assertEqual(len(tied.required_tensor_shapes()), 11)

    def test_parameter_count_matches_shapes_for_mha_gqa_and_mqa(self):
        for kv_heads in (1, 2, 4):
            for tied in (False, True):
                config = tiny_config(hidden_size=16, num_attention_heads=4,
                                     num_key_value_heads=kv_heads, num_hidden_layers=3,
                                     tie_word_embeddings=tied)
                with self.subTest(kv_heads=kv_heads, tied=tied):
                    self.assertEqual(config.parameter_count(), sum(map(prod, config.required_tensor_shapes().values())))
                    self.assertEqual(config.required_tensor_shapes()["model.layers.2.self_attn.k_proj.weight"],
                                     (kv_heads * 4, 16))

    def test_tensor_expansion_rejects_oversized_configs_before_allocating(self):
        for tied in (False, True):
            maximum_layers = (MAX_MODEL_TENSORS - 2 - (not tied)) // 9
            config = tiny_config(num_hidden_layers=maximum_layers, tie_word_embeddings=tied)
            self.assertLessEqual(len(config.required_tensor_shapes()), MAX_MODEL_TENSORS)
            with self.assertRaisesRegex(ValueError, "physical tensors; limit"):
                replace(config, num_hidden_layers=maximum_layers + 1).required_tensor_shapes()
        huge = tiny_config(num_hidden_layers=10 ** 20)
        self.assertEqual(huge.parameter_count(), 136 + 592 * 10 ** 20)
        with self.assertRaisesRegex(ValueError, "physical tensors; limit"):
            huge.required_tensor_shapes()

    def test_dimensions_heads_and_numerical_constants_are_validated(self):
        for changes in ({"vocab_size": 0}, {"hidden_size": True}, {"intermediate_size": -2},
                        {"num_hidden_layers": 1.0}, {"max_position_embeddings": 0},
                        {"num_attention_heads": 3}, {"num_key_value_heads": 3},
                        {"hidden_size": 6}, {"tie_word_embeddings": 1}, {"name": " "},
                        {"rope_theta": 0}, {"rope_theta": float("inf")}, {"rope_theta": True},
                        {"rms_norm_eps": -1}, {"rms_norm_eps": float("nan")},
                        {"rms_norm_eps": "0.00001"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tiny_config(**changes)

    def test_json_roundtrip_requires_complete_versioned_schema(self):
        config = tiny_config()
        self.assertEqual(ModelConfig.from_json(config.to_json()), config)
        for key, value in (("schema_version", 2), ("schema_version", True),
                           ("architecture", "llama"), ("unexpected", 1)):
            data = config.to_dict()
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ModelConfig.from_dict(data)
        incomplete = config.to_dict()
        del incomplete["rope_theta"]
        with self.assertRaises(ValueError):
            ModelConfig.from_dict(incomplete)
        for text in ('{"name":"a","name":"b"}', '{"rope_theta": NaN}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                ModelConfig.from_json(text)

    def test_hf_adapter_accepts_standard_metadata_and_keeps_shapes(self):
        config = ModelConfig.from_hf_config(hf_config(
            torch_dtype="bfloat16", transformers_version="4.46.0", bos_token_id=1,
            eos_token_id=[2, 3], pad_token_id=None, use_cache=True,
            attention_dropout=0.1, initializer_range=0.02, attention_bias=False,
            mlp_bias=False, head_dim=4, rope_scaling=None))
        self.assertEqual(config, tiny_config())
        self.assertNotIn("torch_dtype", config.to_dict())

    def test_hf_defaults_preserve_llama_semantics_instead_of_nexalm_defaults(self):
        data = hf_config()
        for name in ("name", "num_key_value_heads", "tie_word_embeddings", "rms_norm_eps"):
            del data[name]
        data["_name_or_path"] = "local/model"
        config = ModelConfig.from_hf_config(data)
        self.assertEqual(config.name, "local/model")
        self.assertEqual(config.num_key_value_heads, config.num_attention_heads)
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual(config.rms_norm_eps, 1e-6)
        self.assertIn("lm_head.weight", config.required_tensor_shapes())
        self.assertEqual(ModelConfig.from_hf_config({**data, "num_key_value_heads": None}), config)

    def test_hf_adapter_rejects_unsupported_numerical_architectures(self):
        cases = ({"model_type": "mistral"}, {"hidden_act": "gelu"}, {"hidden_act": "swiglu"},
                 {"attention_bias": True}, {"attention_bias": 0}, {"mlp_bias": True},
                 {"rope_scaling": {"type": "linear", "factor": 2}}, {"rope_scaling": {}},
                 {"rope_interleaved": True}, {"partial_rotary_factor": 0.5},
                 {"sliding_window": 32}, {"use_sliding_window": True},
                 {"quantization_config": {"bits": 4}}, {"pretraining_tp": 2},
                 {"is_encoder_decoder": True}, {"add_cross_attention": True},
                 {"head_dim": 8}, {"head_dim": True}, {"architectures": ["CustomLlama"]},
                 {"auto_map": {"AutoModel": "custom.Model"}}, {"num_local_experts": 4},
                 {"name": False})
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ModelConfig.from_hf_config(hf_config(**changes))

    def test_hf_rope_parameters_support_only_consistent_unscaled_rope(self):
        data = hf_config(rope_parameters={"rope_type": "default", "rope_theta": 10000})
        self.assertEqual(ModelConfig.from_hf_config(data).rope_theta, 10000)
        del data["rope_theta"]
        data["rope_parameters"]["rope_theta"] = 500000
        self.assertEqual(ModelConfig.from_hf_config(data).rope_theta, 500000)
        for changes in ({"rope_theta": 10000},
                        {"rope_parameters": {"rope_type": "llama3", "factor": 8}},
                        {"rope_parameters": {"rope_type": "default", "factor": 1}},
                        {"rope_parameters": {"rope_theta": True}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ModelConfig.from_hf_config({**data, **changes})

    def test_hf_adapter_does_not_invent_missing_dimensions(self):
        for field in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                      "num_attention_heads", "max_position_embeddings"):
            data = hf_config()
            del data[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                ModelConfig.from_hf_config(data)


class ModelDefinitionRegressions(unittest.TestCase):
    def test_dsl_lowers_to_the_same_config_contract_and_json_roundtrips(self):
        config = compile_model_definition(TINY_DEFINITION)["Tiny"]
        self.assertEqual(config, replace(tiny_config(), name="Tiny"))
        self.assertEqual(ModelConfig.from_json(config.to_json()), config)
        self.assertEqual(config.parameter_count(), 728)

    def test_multiple_variants_and_shared_lexer_comments(self):
        source = "# model Ignored {}\n/* ignored definition */\n" + TINY_DEFINITION
        source += TINY_DEFINITION.replace("model Tiny", "model Untied").replace("tied_embeddings: true", "tied_embeddings: false")
        configs = compile_model_definition(source)
        self.assertEqual(set(configs), {"Tiny", "Untied"})
        self.assertEqual(configs["Untied"].parameter_count(), 856)

    def test_dsl_rejects_unknown_duplicate_missing_and_incompatible_fields(self):
        variants = [TINY_DEFINITION.replace("context: 32;", "context: 32; extra: 1;"),
                    TINY_DEFINITION.replace("context: 32;", "context: 32; context: 64;"),
                    TINY_DEFINITION.replace("context: 32;", ""),
                    TINY_DEFINITION.replace("head_dim: 4;", "head_dim: 8;"),
                    TINY_DEFINITION.replace("kv_heads: 1;", "kv_heads: 3;"),
                    TINY_DEFINITION.replace("position: rope;", "position: alibi;"),
                    TINY_DEFINITION.replace("activation: swiglu;", "activation: gelu;"),
                    TINY_DEFINITION.replace("norm: rmsnorm;", "norm: layernorm;"),
                    TINY_DEFINITION.replace("tied_embeddings: true;", "tied_embeddings: 1;"),
                    TINY_DEFINITION.replace("query_heads: 2;", "query_heads: 2; query_heads: 2;"),
                    TINY_DEFINITION + TINY_DEFINITION,
                    TINY_DEFINITION + "fn main() {}"]
        for source in variants:
            with self.subTest(source=source), self.assertRaises(ModelDefinitionError):
                compile_model_definition(source)

    def test_dsl_rejects_truncated_lexical_constructs_and_empty_source(self):
        for source in ("", "# comment only", TINY_DEFINITION.rstrip()[:-1],
                       TINY_DEFINITION + "/*", TINY_DEFINITION + "/* unclosed comment",
                       TINY_DEFINITION.replace("context: 32;", "context: 32"),
                       TINY_DEFINITION.replace("context: 32;", 'context: "32";'),
                       TINY_DEFINITION + "$"):
            with self.subTest(source=source), self.assertRaises(ModelDefinitionError):
                compile_model_definition(source)

    def test_canonical_r0_and_v1_fixtures_have_exact_physical_counts(self):
        directory = ROOT / "models" / "nexalm512"
        configs = compile_model_definition((directory / "architecture.nxl").read_text())
        expected = {"NexaLM512_R0": ("r0.json", 125_854_464, 146),
                    "NexaLM512_v1": ("v1.json", 394_331_136, 290)}
        self.assertEqual(set(configs), set(expected))
        for name, (filename, count, tensors) in expected.items():
            with self.subTest(name=name):
                config = configs[name]
                fixture = ModelConfig.from_json((directory / "configs" / filename).read_text())
                self.assertEqual(config, fixture)
                self.assertEqual(config.parameter_count(), count)
                self.assertEqual(sum(map(prod, config.required_tensor_shapes().values())), count)
                self.assertEqual(len(config.required_tensor_shapes()), tensors)
                self.assertEqual(config.head_dim, 64)
                self.assertEqual(config.max_position_embeddings, 2048)
                self.assertTrue(config.tie_word_embeddings)


if __name__ == "__main__":
    unittest.main()

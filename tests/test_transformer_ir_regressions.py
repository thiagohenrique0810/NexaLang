"""Transformer graph contracts and serial activation storage behavior."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.model_config import ModelConfig
from compiler.model_ir import DType, ModelGraph, ModelOp, TensorDesc
from compiler.model_lowering import derive_activation_requests, lower_model
from compiler.planner.memory import MemoryPlanner
from tools.nexa_model import compile_definitions


def tiny(**changes):
    return ModelConfig(**{"name": "Tiny", "vocab_size": 16, "hidden_size": 8,
                          "intermediate_size": 12, "num_hidden_layers": 1,
                          "num_attention_heads": 2, "num_key_value_heads": 1,
                          "max_position_embeddings": 16, **changes})


def operation_graph(kind, inputs, output, attributes=None):
    return ModelGraph("one_operation", [*inputs, output],
                      [ModelOp("op", kind, [t.name for t in inputs], [output.name], attributes or {})],
                      [t.name for t in inputs], [output.name])


class TransformerOperatorRegressions(unittest.TestCase):
    def test_each_operator_accepts_its_concrete_shape_contract(self):
        cases = [
            ("Embedding", [TensorDesc("tokens", (3,), "u32", "u32"), TensorDesc("w", (16, 8))],
             TensorDesc("out", (3, 8)), {}),
            ("RMSNorm", [TensorDesc("x", (3, 8)), TensorDesc("w", (8,))],
             TensorDesc("out", (3, 8)), {"epsilon": 1e-5}),
            ("RoPE", [TensorDesc("x", (3, 8))], TensorDesc("out", (3, 8)),
             {"num_heads": 2, "head_dim": 4, "theta": 10000}),
            ("CausalAttention", [TensorDesc("q", (3, 8)), TensorDesc("k", (3, 4)), TensorDesc("v", (3, 4))],
             TensorDesc("out", (3, 8)), {"num_heads": 2, "num_key_value_heads": 1, "head_dim": 4}),
            ("SwiGLU", [TensorDesc("gate", (3, 12)), TensorDesc("up", (3, 12))], TensorDesc("out", (3, 12)), {}),
            ("Add", [TensorDesc("a", (3, 8)), TensorDesc("b", (3, 8))], TensorDesc("out", (3, 8)), {}),
        ]
        for kind, inputs, output, attributes in cases:
            with self.subTest(kind=kind):
                graph = operation_graph(kind, inputs, output, attributes)
                self.assertEqual(ModelGraph.from_json(graph.to_json()).to_dict(), graph.to_dict())

    def test_operator_attributes_and_arities_are_strict(self):
        invalid = [
            ("Embedding", ["tokens", "w"], {"padding_idx": 0}),
            ("RMSNorm", ["x", "w"], {}),
            ("RMSNorm", ["x", "w"], {"epsilon": True}),
            ("RMSNorm", ["x", "w"], {"epsilon": 0}),
            ("RMSNorm", ["x", "w"], {"epsilon": float("nan")}),
            ("RoPE", ["x"], {"num_heads": 2, "head_dim": 3, "theta": 10000}),
            ("RoPE", ["x"], {"num_heads": True, "head_dim": 4, "theta": 10000}),
            ("RoPE", ["x"], {"num_heads": 2, "head_dim": 4, "theta": float("inf")}),
            ("RoPE", ["x"], {"num_heads": 2, "head_dim": 4, "theta": 10000, "offset": 1}),
            ("CausalAttention", ["q", "k", "v"], {"num_heads": 3, "num_key_value_heads": 2, "head_dim": 4}),
            ("CausalAttention", ["q", "k", "v"], {"num_heads": 2, "num_key_value_heads": 0, "head_dim": 4}),
            ("CausalAttention", ["q", "k"], {"num_heads": 2, "num_key_value_heads": 1, "head_dim": 4}),
            ("SwiGLU", ["gate", "up"], {"activation": "relu"}),
            ("Add", ["a", "b"], {"broadcast": True}),
            ("Add", ["a"], {}),
        ]
        for kind, inputs, attributes in invalid:
            with self.subTest(kind=kind, attributes=attributes), self.assertRaises(ValueError):
                ModelOp("bad", kind, inputs, ["out"], attributes)

    def test_embedding_requires_u32_tokens_and_matching_dense_output(self):
        good = [TensorDesc("tokens", (3,), "u32", "u32"), TensorDesc("w", (16, 8))]
        for inputs, output in (([replace(good[0], logical_dtype="i32", storage_dtype="i32"), good[1]], TensorDesc("out", (3, 8))),
                               ([TensorDesc("tokens", (1, 3), "u32", "u32"), good[1]], TensorDesc("out", (3, 8))),
                               (good, TensorDesc("out", (3, 4))),
                               (good, TensorDesc("out", (3, 8), storage_dtype="f16"))):
            with self.subTest(inputs=inputs, output=output), self.assertRaises(ValueError):
                operation_graph("Embedding", inputs, output)

    def test_norm_rope_and_attention_reject_wrong_sequence_width_or_storage(self):
        cases = [
            ("RMSNorm", [TensorDesc("x", (3, 8)), TensorDesc("w", (4,))], TensorDesc("out", (3, 8)), {"epsilon": 1e-5}),
            ("RMSNorm", [TensorDesc("x", (3, 8)), TensorDesc("w", (1, 8))], TensorDesc("out", (3, 8)), {"epsilon": 1e-5}),
            ("RoPE", [TensorDesc("x", (3, 4))], TensorDesc("out", (3, 4)), {"num_heads": 2, "head_dim": 4, "theta": 10000}),
            ("RoPE", [TensorDesc("x", (3, 8), storage_dtype="q4", storage_nbytes=12)], TensorDesc("out", (3, 8)),
             {"num_heads": 2, "head_dim": 4, "theta": 10000}),
            ("CausalAttention", [TensorDesc("q", (3, 8)), TensorDesc("k", (2, 4)), TensorDesc("v", (2, 4))],
             TensorDesc("out", (3, 8)), {"num_heads": 2, "num_key_value_heads": 1, "head_dim": 4}),
            ("CausalAttention", [TensorDesc("q", (3, 8)), TensorDesc("k", (3, 4)), TensorDesc("v", (3, 8))],
             TensorDesc("out", (3, 8)), {"num_heads": 2, "num_key_value_heads": 1, "head_dim": 4}),
        ]
        for case in cases:
            with self.subTest(kind=case[0]), self.assertRaises(ValueError):
                operation_graph(*case)

    def test_elementwise_ops_disallow_broadcasting_and_non_f32_activations(self):
        for kind in ("Add", "SwiGLU"):
            for right in (TensorDesc("b", (1, 8)), TensorDesc("b", (3, 8), logical_dtype="f16")):
                with self.subTest(kind=kind, right=right), self.assertRaises(ValueError):
                    operation_graph(kind, [TensorDesc("a", (3, 8)), right], TensorDesc("out", (3, 8)))


class TransformerLoweringRegressions(unittest.TestCase):
    def test_full_graph_has_complete_operators_shapes_and_residual_dependencies(self):
        config = tiny(num_hidden_layers=2)
        graph = lower_model(config, 3)
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        ops = {op.name: op for op in graph.ops}
        self.assertEqual(Counter(op.kind.value for op in graph.ops),
                         {"Embedding": 1, "RMSNorm": 5, "MatMul": 15, "RoPE": 4,
                          "CausalAttention": 2, "Add": 4, "SwiGLU": 2})
        self.assertEqual(tensors["tokens"].shape, (3,))
        self.assertEqual(tensors["tokens"].logical_dtype, DType.U32)
        self.assertEqual(tensors["logits"].shape, (3, 16))
        self.assertEqual(tensors["layers.0.k"].shape, (3, 4))
        self.assertEqual(tensors["layers.1.swiglu"].shape, (3, 12))
        self.assertEqual(ops["layers.0.attention_residual"].inputs[0], "embedding")
        self.assertEqual(ops["layers.1.input_norm"].inputs[0], "layers.0.output")
        self.assertEqual(ops["layers.1.attention_residual"].inputs[0], "layers.0.output")
        self.assertEqual(ops["layers.1.output"].inputs[0], "layers.1.attention_residual")
        self.assertEqual(ops["layers.0.q_rope"].attributes["num_heads"], 2)
        self.assertEqual(ops["layers.0.k_rope"].attributes["num_heads"], 1)
        self.assertEqual(set(graph.constants), set(config.required_tensor_shapes()))
        self.assertEqual(ModelGraph.from_json(graph.to_json()).to_dict(), graph.to_dict())

    def test_tied_head_reuses_the_embedding_constant_without_alias_storage(self):
        for tied in (False, True):
            graph = lower_model(tiny(tie_word_embeddings=tied), 2)
            weight = "model.embed_tokens.weight" if tied else "lm_head.weight"
            self.assertEqual(graph.ops[-1].inputs[1], weight)
            self.assertEqual("lm_head.weight" in graph.constants, not tied)
            self.assertEqual(sum(t.name == "model.embed_tokens.weight" for t in graph.tensors), 1)

    def test_partial_packed_storage_overrides_preserve_physical_descriptor(self):
        config = tiny()
        weight = TensorDesc("model.embed_tokens.weight", (16, 8), storage_dtype="q4",
                            storage_nbytes=128, tier="disk", alignment=128)
        graph = lower_model(config, 2, weight_storage={weight.name: weight})
        tensors = {tensor.name: tensor for tensor in graph.tensors}
        self.assertEqual(tensors[weight.name], weight)
        self.assertEqual(tensors["model.norm.weight"].storage_dtype, DType.F32)
        self.assertNotIn(weight.name, {r.name for r in derive_activation_requests(graph)})

    def test_lowering_rejects_bad_lengths_and_weight_contracts(self):
        config = tiny()
        for length in (0, -1, True, 1.5, 17):
            with self.subTest(length=length), self.assertRaises(ValueError):
                lower_model(config, length)
        name = "model.embed_tokens.weight"
        invalid = [{"lm_head.weight": TensorDesc("lm_head.weight", (16, 8))},
                   {name: TensorDesc("wrong", (16, 8))}, {name: TensorDesc(name, (8, 16))},
                   {name: TensorDesc(name, (16, 8), logical_dtype="f16")},
                   {name: TensorDesc(name, (16, 8), storage_dtype="f16")},
                   {"model.norm.weight": TensorDesc("model.norm.weight", (8,), storage_dtype="q4", storage_nbytes=8)}]
        for storage in invalid:
            with self.subTest(storage=storage), self.assertRaises(ValueError):
                lower_model(config, 2, weight_storage=storage)
        with self.assertRaises(ValueError):
            lower_model(tiny(num_hidden_layers=10 ** 12), 1)

    def test_graph_edits_cannot_hide_missing_weights_or_reorder_dependencies(self):
        graph = lower_model(tiny(), 2)
        with self.assertRaises(ValueError):
            replace(graph, constants=graph.constants[1:])
        with self.assertRaises(ValueError):
            replace(graph, ops=tuple(reversed(graph.ops)))
        data = graph.to_dict()
        data["ops"][1]["attributes"]["unrecognized"] = 1
        with self.assertRaises(ValueError):
            ModelGraph.from_dict(data)

    def test_cli_preserves_default_and_optionally_serializes_real_graphs(self):
        source = ROOT / "models/nexalm512/architecture.nxl"
        default = compile_definitions(source)
        self.assertNotIn("sequence_length", default)
        self.assertTrue(all("model_ir" not in model for model in default["models"].values()))
        with tempfile.TemporaryDirectory(prefix="nexa-model-ir-") as temp:
            output = Path(temp) / "graphs.json"
            run = subprocess.run([sys.executable, str(ROOT / "tools/nexa_model.py"), str(source),
                                  "--sequence-length", "2", "--out", str(output)],
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            lowered = json.loads(output.read_text())
        self.assertEqual(lowered["sequence_length"], 2)
        for name, entry in lowered["models"].items():
            graph = ModelGraph.from_dict(entry["model_ir"])
            self.assertEqual(entry["activation_requests"], [r.to_dict() for r in derive_activation_requests(graph)])
            self.assertEqual(len(graph.ops), entry["config"]["num_hidden_layers"] * 15 + 3)
            self.assertEqual(entry["parameter_count"], default["models"][name]["parameter_count"])


class ActivationLifetimeRegressions(unittest.TestCase):
    def test_operands_and_results_overlap_in_time_but_dead_storage_is_reused(self):
        tensors = [TensorDesc(name, (1, 4)) for name in ("x", "y", "tmp", "out")]
        graph = ModelGraph("chain", tensors,
                           [ModelOp("first", "Add", ["x", "y"], ["tmp"]),
                            ModelOp("second", "Add", ["tmp", "y"], ["out"])],
                           ["x", "y"], ["out"])
        requests = derive_activation_requests(graph)
        self.assertEqual({r.name: (r.start, r.end) for r in requests},
                         {"x": (0, 2), "y": (0, 3), "tmp": (1, 3), "out": (2, 4)})
        plan = MemoryPlanner().plan(requests, {"host": 1024})
        self.assertEqual(plan.allocations["out"].offset, plan.allocations["x"].offset)
        self.assertEqual(len({plan.allocations[name].offset for name in ("tmp", "y", "out")}), 3)

    def test_returned_values_stay_live_through_the_final_consumer(self):
        tensors = [TensorDesc(name, (1, 4)) for name in ("x", "unused", "a", "b")]
        graph = ModelGraph("retained", tensors,
                           [ModelOp("first", "Add", ["x", "x"], ["a"]),
                            ModelOp("second", "Add", ["a", "a"], ["b"])],
                           ["x", "unused"], ["x", "a", "b"])
        lifetimes = {r.name: r for r in derive_activation_requests(graph)}
        self.assertEqual(lifetimes["unused"].end, 1)
        for output in graph.outputs:
            self.assertEqual(lifetimes[output].end, 4)
        passthrough = ModelGraph("identity", [TensorDesc("x", (1, 4))], [], ["x"], ["x"])
        request, = derive_activation_requests(passthrough)
        self.assertEqual((request.start, request.end), (0, 2))

    def test_multilayer_plan_preserves_live_data_in_an_actual_arena(self):
        graph = lower_model(tiny(num_hidden_layers=3), 3)
        requests = derive_activation_requests(graph)
        self.assertEqual({r.name for r in requests}, {t.name for t in graph.tensors} - set(graph.constants))
        plan = MemoryPlanner().plan(requests, {"host": 1024 * 1024})
        self.assertLess(plan.peak_bytes["host"], sum(r.size_bytes for r in requests))
        arena = bytearray(plan.peak_bytes["host"])
        expected = {}

        def write(name, tag):
            allocation = plan.allocations[name]
            expected[name] = bytes([tag]) * allocation.size_bytes
            arena[allocation.offset:allocation.offset + allocation.size_bytes] = expected[name]

        for index, name in enumerate(graph.inputs, 1):
            write(name, index)
        for index, op in enumerate(graph.ops, 2):
            output = plan.allocations[op.outputs[0]]
            for name in op.inputs:
                if name in graph.constants:
                    continue
                operand = plan.allocations[name]
                self.assertEqual(arena[operand.offset:operand.offset + operand.size_bytes], expected[name])
                self.assertTrue(output.offset + output.size_bytes <= operand.offset
                                or operand.offset + operand.size_bytes <= output.offset)
            write(op.outputs[0], index)
        for name in graph.outputs:
            allocation = plan.allocations[name]
            self.assertEqual(arena[allocation.offset:allocation.offset + allocation.size_bytes], expected[name])


if __name__ == "__main__":
    unittest.main()

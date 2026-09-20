"""Two independent implementations of the same graph, compared bit by bit.

The headline test runs the native C scalar executor (runtime/nexapack) and the
pure-Python operator evaluator (compiler/graph_eval) over the same decoded
weights and the same tokens, and reports the measured disagreement against a
declared tolerance. Neither shares a line of arithmetic with the other: if the
evaluator gets the RoPE half-rotation, the causal GQA mask or the silu wrong,
this fails.
"""
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.graph_eval import (MAX_EVAL_ELEMENTS, evaluate_graph, evaluate_op, f32,
                                 matmul, random_bindings, tensor_bytes)
from compiler.model_config import ModelConfig
from compiler.model_ir import ModelOp
from compiler.model_lowering import lower_model
from test_transformer_forward_regressions import random_bundle

# Declared before any number is measured. The two implementations differ on
# purpose in the softmax: the C kernel rounds its exponentials to float32, this
# evaluator keeps them double. That is the error being bounded here.
ORACLE_ATOL = 1e-5
ORACLE_RTOL = 1e-4


def decode_bundle_bindings(graph, bundle_path, config, tokens):
    """Bind the graph to the bundle's own decoded values, without PyTorch.

    Matrices bind as exact scale*code products, which is what the packed matmul
    kernel accumulates; rounding them to float32 here would compare against a
    third arithmetic that neither implementation uses. A tied lm_head is an
    alias, so it is bound only when the graph declares it as a constant.
    """
    from runtime.nexapack.bundle import ModelBundleReader
    from runtime.nexapack.format import decode_q4_row
    bindings = {"tokens": tuple(tokens)}
    with ModelBundleReader(bundle_path) as bundle:
        for name, shape in config.required_tensor_shapes().items():
            if len(shape) == 1:
                bindings[name] = list(bundle.read_f32(name))
            else:
                with bundle.open_q4(name) as reader:
                    bindings[name] = [decode_q4_row(reader.read_rows(row, 1), reader.cols,
                                                    reader.group_size)
                                      for row in range(reader.rows)]
    for alias, target in config.tensor_aliases().items():
        if alias in graph.constants:
            bindings[alias] = bindings[target]
    return bindings


def compare(actual, expected):
    max_abs = max_rel = 0.0
    for actual_row, expected_row in zip(actual, expected):
        for left, right in zip(actual_row, expected_row):
            if not (math.isfinite(left) and math.isfinite(right)):
                raise AssertionError("logits must be finite")
            error = abs(left - right)
            max_abs = max(max_abs, error)
            max_rel = max(max_rel, error / max(abs(right), 1e-12))
    return {"max_abs_error": max_abs, "max_rel_error": max_rel}


class GraphEvalOracleRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which("clang") or shutil.which("cc")):
            raise unittest.SkipTest("C compiler unavailable")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-graph-eval-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_native_executor_and_python_evaluator_agree_within_declared_tolerance(self):
        from runtime.nexapack.transformer import TransformerSession
        cases = [(1, 2, 1, True, 728), (2, 4, 2, False, 900), (2, 2, 2, True, 901),
                 (2, 4, 1, False, 902)]
        tokens = [1, 5, 2, 0, 3, 6]
        worst = {"max_abs_error": 0.0, "max_rel_error": 0.0}
        for layers, heads, kv_heads, tied, seed in cases:
            with self.subTest(layers=layers, heads=heads, kv_heads=kv_heads, tied=tied):
                path = self.directory / f"bundle-{seed}"
                config, _ = random_bundle(path, layers=layers, heads=heads, kv_heads=kv_heads,
                                          tied=tied, seed=seed)
                with TransformerSession(path, memory_budget="4MiB") as session:
                    native = session.prefill(tokens)
                graph = lower_model(config, len(tokens))
                bindings = decode_bundle_bindings(graph, path, config, tokens)
                evaluated = evaluate_graph(graph, bindings)["logits"]
                self.assertEqual((len(evaluated), len(evaluated[0])),
                                 (len(tokens), config.vocab_size))
                error = compare(native, evaluated)
                self.assertLess(error["max_abs_error"], ORACLE_ATOL, error)
                self.assertLess(error["max_rel_error"], ORACLE_RTOL, error)
                for key in worst:
                    worst[key] = max(worst[key], error[key])
        # A zero here would mean the two implementations share arithmetic they
        # should not; the measurement is reported, not asserted away.
        self.assertGreater(worst["max_abs_error"], 0.0, worst)

    def test_evaluator_disagrees_with_the_native_run_when_the_weights_differ(self):
        from runtime.nexapack.transformer import TransformerSession
        tokens = [1, 5, 2]
        path = self.directory / "bundle"
        config, _ = random_bundle(path, layers=1, heads=2, kv_heads=1, seed=310)
        other = self.directory / "other"
        random_bundle(other, layers=1, heads=2, kv_heads=1, seed=311)
        with TransformerSession(path, memory_budget="4MiB") as session:
            native = session.prefill(tokens)
        graph = lower_model(config, len(tokens))
        evaluated = evaluate_graph(graph, decode_bundle_bindings(graph, other, config, tokens))["logits"]
        self.assertGreater(compare(native, evaluated)["max_abs_error"], ORACLE_ATOL)


class GraphEvalOperatorRegressions(unittest.TestCase):
    def rope(self, values, *, heads=1, head_dim=4, theta=10000.0, row_offset=0):
        op = ModelOp("rope", "RoPE", ["x"], ["y"],
                     {"num_heads": heads, "head_dim": head_dim, "theta": theta})
        return evaluate_op(op, (values,), row_offset=row_offset)

    def test_rope_rotates_halves_and_not_adjacent_lanes(self):
        row = [1.0, 2.0, 3.0, 4.0]
        rotated = self.rope([row, row])[1]
        angles = [1.0 * 10000.0 ** (-2.0 * lane / 4) for lane in range(2)]
        half = [(f32(row[lane] * math.cos(angles[lane]) - row[lane + 2] * math.sin(angles[lane])),
                 f32(row[lane + 2] * math.cos(angles[lane]) + row[lane] * math.sin(angles[lane])))
                for lane in range(2)]
        self.assertEqual(rotated, (half[0][0], half[1][0], half[0][1], half[1][1]))
        # The interleaved convention would pair lanes 0/1 and 2/3 instead.
        interleaved = (f32(row[0] * math.cos(angles[0]) - row[1] * math.sin(angles[0])),)
        self.assertNotEqual(rotated[:1], interleaved)

    def test_rope_position_zero_is_the_identity_and_row_offset_moves_it(self):
        row = [0.5, -1.5, 2.5, 0.25]
        self.assertEqual(self.rope([row])[0], tuple(f32(value) for value in row))
        self.assertEqual(self.rope([row], row_offset=3)[0], self.rope([row] * 4)[3])
        with self.assertRaises(ValueError):
            self.rope([row], row_offset=-1)

    def test_causal_attention_ignores_the_future_and_shares_kv_heads(self):
        op = ModelOp("attention", "CausalAttention", ["q", "k", "v"], ["y"],
                     {"num_heads": 2, "num_key_value_heads": 1, "head_dim": 2})
        query = [[1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 1.0, 0.0]]
        key = [[1.0, 0.0], [0.0, 1.0]]
        value = [[3.0, 4.0], [-5.0, 6.0]]
        result = evaluate_op(op, (query, key, value))
        # Row 0 sees position 0 only, so both query heads return value row 0.
        self.assertEqual(result[0], (f32(3.0), f32(4.0), f32(3.0), f32(4.0)))
        changed = evaluate_op(op, (query, key, [value[0], [100.0, 100.0]]))
        self.assertEqual(changed[0], result[0])
        self.assertNotEqual(changed[1], result[1])

    def test_swiglu_is_silu_of_the_gate_times_the_up_projection(self):
        op = ModelOp("swiglu", "SwiGLU", ["g", "u"], ["y"], {})
        gate, up = [[-40.0, 0.0, 2.0]], [[1.0, 7.0, -3.0]]
        expected = tuple(f32((value / (1.0 + math.exp(-value))) * other)
                         for value, other in zip(gate[0], up[0]))
        self.assertEqual(evaluate_op(op, (gate, up))[0], expected)

    def test_rmsnorm_reduces_inside_its_own_row(self):
        op = ModelOp("norm", "RMSNorm", ["x", "w"], ["y"], {"epsilon": 1e-5})
        rows = [[3.0, 4.0], [30.0, 40.0]]
        weight = [1.0, 2.0]
        result = evaluate_op(op, (rows, weight))
        self.assertNotEqual(result[0], result[1])
        single = evaluate_op(op, ([rows[0]], weight))
        self.assertEqual(single[0], result[0])

    def test_matmul_honors_transpose_b_and_rejects_mismatched_reductions(self):
        left = [[1.0, 2.0]]
        right = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(matmul(left, right, transpose_b=True)[0], (f32(5.0), f32(11.0)))
        self.assertEqual(matmul(left, right)[0], (f32(7.0), f32(10.0)))
        with self.assertRaises(ValueError):
            matmul(left, [[1.0, 2.0, 3.0]], transpose_b=True)

    def test_matmul_accumulates_in_double_and_stores_float32(self):
        # 1 + 1e-8 is representable as a double and rounds away in float32; a
        # result that keeps it means the store was skipped.
        result = matmul([[1.0, 1e-8]], [[1.0, 1.0]], transpose_b=True)[0][0]
        self.assertEqual(result, 1.0)
        self.assertNotEqual(result, 1.0 + 1e-8)
        # Both tiny terms survive the double accumulation and shift the result.
        self.assertGreater(matmul([[1.0, 1e-4, 1e-4]], [[1.0, 1.0, 1.0]],
                                  transpose_b=True)[0][0], 1.0)


class GraphEvalContractRegressions(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(name="tiny", vocab_size=11, hidden_size=8, intermediate_size=12,
                                  num_hidden_layers=1, num_attention_heads=2,
                                  num_key_value_heads=1, max_position_embeddings=8)
        self.graph = lower_model(self.config, 3)
        self.bindings = random_bindings(self.graph, seed=5)

    def test_random_bindings_are_deterministic_and_respect_the_vocabulary(self):
        self.assertEqual(self.bindings, random_bindings(self.graph, seed=5))
        self.assertNotEqual(self.bindings["tokens"], random_bindings(self.graph, seed=6)["tokens"])
        self.assertTrue(all(0 <= token < self.config.vocab_size
                            for token in self.bindings["tokens"]))

    def test_every_produced_value_is_exactly_a_float32(self):
        values = evaluate_graph(self.graph, self.bindings)
        produced = [op.outputs[0] for op in self.graph.ops]
        self.assertTrue(produced)
        for name in produced:
            for row in values[name]:
                for item in row:
                    self.assertEqual(f32(item), item, name)

    def test_evaluation_is_reproducible_byte_for_byte(self):
        first = evaluate_graph(self.graph, self.bindings)["logits"]
        second = evaluate_graph(self.graph, dict(self.bindings))["logits"]
        self.assertEqual(tensor_bytes(first), tensor_bytes(second))
        self.assertEqual(len(tensor_bytes(first)), 3 * self.config.vocab_size * 4)

    def test_tensor_bytes_is_little_endian_float32(self):
        self.assertEqual(tensor_bytes([[1.5, -2.0]]), struct.pack("<ff", 1.5, -2.0))
        self.assertEqual(tensor_bytes([0.25]), struct.pack("<f", 0.25))

    def test_missing_extra_and_malformed_bindings_are_refused(self):
        for mutate in (lambda b: b.pop("tokens"),
                       lambda b: b.update({"not_a_tensor": [[1.0]]}),
                       lambda b: b.update({"tokens": (0, 1)}),
                       lambda b: b.update({"model.norm.weight": [1.0]}),
                       lambda b: b.update({"tokens": (0, 1, self.config.vocab_size)})):
            bindings = {name: value for name, value in self.bindings.items()}
            mutate(bindings)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                evaluate_graph(self.graph, bindings)

    def test_graphs_beyond_the_diagnostic_limit_are_refused(self):
        config = ModelConfig(name="wide", vocab_size=32768, hidden_size=768,
                             intermediate_size=2048, num_hidden_layers=16,
                             num_attention_heads=12, num_key_value_heads=4,
                             max_position_embeddings=2048)
        graph = lower_model(config, 4)
        self.assertGreater(sum(tensor.numel for tensor in graph.tensors), MAX_EVAL_ELEMENTS)
        with self.assertRaises(ValueError):
            evaluate_graph(graph, {})


if __name__ == "__main__":
    unittest.main()

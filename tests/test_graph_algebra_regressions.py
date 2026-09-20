"""Rewrites that must prove themselves, and the zero they score on Llama.

The three passes are exercised on synthetic graphs built to contain the sites
they look for, and then run against the graph the model pipeline actually
lowers, where all three match nothing. That zero is the measurement, not a
failure: the deliverable is the verifier, not a speedup.
"""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.graph_algebra import (EXACT, TOLERANCE, CommonSubexpressionElimination,
                                    ConstantDerivation, DeadOpElimination,
                                    GraphEquivalenceVerifier, GraphRewrite,
                                    MatMulProjectionFold, RewriteError, RewriteReport,
                                    RewriteSite, default_passes, run_passes)
from compiler.graph_eval import evaluate_graph, random_bindings, tensor_bytes
from compiler.model_config import ModelConfig
from compiler.model_definition import compile_model_definition
from compiler.model_ir import DType, ModelGraph, ModelOp, OpKind, TensorDesc
from compiler.model_lowering import lower_model


def activation(name, rows, width):
    return TensorDesc(name, (rows, width))


def projection_graph(*, rows=3, vocab=6, hidden=4, middle=3, columns=2, dead=False):
    """tokens -> Embedding -> x@A^T -> x@A^T@B^T, optionally with a dead branch."""
    tensors = [TensorDesc("tokens", (rows,), DType.U32, DType.U32),
               activation("embed_weight", vocab, hidden),
               activation("left_weight", middle, hidden),
               activation("right_weight", columns, middle),
               activation("embedding", rows, hidden),
               activation("projected", rows, middle),
               activation("logits", rows, columns)]
    ops = [ModelOp("embed", OpKind.EMBEDDING, ["tokens", "embed_weight"], ["embedding"], {}),
           ModelOp("left", OpKind.MATMUL, ["embedding", "left_weight"], ["projected"],
                   {"transpose_b": True}),
           ModelOp("right", OpKind.MATMUL, ["projected", "right_weight"], ["logits"],
                   {"transpose_b": True})]
    constants = ["embed_weight", "left_weight", "right_weight"]
    if dead:
        tensors.append(activation("unused", rows, middle))
        ops.insert(2, ModelOp("dead", OpKind.MATMUL, ["embedding", "left_weight"], ["unused"],
                              {"transpose_b": True}))
    return ModelGraph(name="projections", tensors=tuple(tensors), ops=tuple(ops),
                      inputs=["tokens"], outputs=["logits"], constants=constants)


def duplicate_graph(*, rows=3, vocab=6, hidden=4, middle=3):
    """Two identical projections of the same tensor, summed."""
    tensors = [TensorDesc("tokens", (rows,), DType.U32, DType.U32),
               activation("embed_weight", vocab, hidden),
               activation("weight", middle, hidden),
               activation("embedding", rows, hidden),
               activation("first", rows, middle),
               activation("second", rows, middle),
               activation("logits", rows, middle)]
    ops = [ModelOp("embed", OpKind.EMBEDDING, ["tokens", "embed_weight"], ["embedding"], {}),
           ModelOp("first", OpKind.MATMUL, ["embedding", "weight"], ["first"], {"transpose_b": True}),
           ModelOp("second", OpKind.MATMUL, ["embedding", "weight"], ["second"], {"transpose_b": True}),
           ModelOp("sum", OpKind.ADD, ["first", "second"], ["logits"], {})]
    return ModelGraph(name="duplicates", tensors=tuple(tensors), ops=tuple(ops),
                      inputs=["tokens"], outputs=["logits"],
                      constants=["embed_weight", "weight"])


def normalization_graph(first_epsilon, second_epsilon, *, rows=3, vocab=6, hidden=4):
    """Two RMSNorms over the same rows, differing only in epsilon."""
    tensors = [TensorDesc("tokens", (rows,), DType.U32, DType.U32),
               activation("embed_weight", vocab, hidden),
               TensorDesc("norm_weight", (hidden,)),
               activation("embedding", rows, hidden),
               activation("first", rows, hidden),
               activation("second", rows, hidden),
               activation("logits", rows, hidden)]
    ops = [ModelOp("embed", OpKind.EMBEDDING, ["tokens", "embed_weight"], ["embedding"], {}),
           ModelOp("first", OpKind.RMSNORM, ["embedding", "norm_weight"], ["first"],
                   {"epsilon": first_epsilon}),
           ModelOp("second", OpKind.RMSNORM, ["embedding", "norm_weight"], ["second"],
                   {"epsilon": second_epsilon}),
           ModelOp("sum", OpKind.ADD, ["first", "second"], ["logits"], {})]
    return ModelGraph(name="normalizations", tensors=tuple(tensors), ops=tuple(ops),
                      inputs=["tokens"], outputs=["logits"],
                      constants=["embed_weight", "norm_weight"])


def repeated_chain_graph(*, rows=3, vocab=6, hidden=4):
    """zeroth and first are equal; second and third become equal once rewired."""
    tensors = [TensorDesc("tokens", (rows,), DType.U32, DType.U32),
               activation("embed_weight", vocab, hidden),
               TensorDesc("norm_weight", (hidden,)),
               activation("embedding", rows, hidden)]
    ops = [ModelOp("embed", OpKind.EMBEDDING, ["tokens", "embed_weight"], ["embedding"], {})]
    for name, source in (("zeroth", "embedding"), ("first", "embedding"),
                         ("second", "zeroth"), ("third", "first")):
        tensors.append(activation(name, rows, hidden))
        ops.append(ModelOp(name, OpKind.RMSNORM, [source, "norm_weight"], [name],
                           {"epsilon": 1e-5}))
    tensors.append(activation("logits", rows, hidden))
    ops.append(ModelOp("sum", OpKind.ADD, ["second", "third"], ["logits"], {}))
    return ModelGraph(name="chain", tensors=tuple(tensors), ops=tuple(ops),
                      inputs=["tokens"], outputs=["logits"],
                      constants=["embed_weight", "norm_weight"])


def llama_graph(layers=2, sequence=4):
    config = ModelConfig(name="tiny", vocab_size=13, hidden_size=8, intermediate_size=12,
                         num_hidden_layers=layers, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=12)
    return lower_model(config, sequence)


class _EpsilonDrift(GraphRewrite):
    """A pass that lies: it changes the arithmetic and claims it did not."""
    name = "EpsilonDrift"

    def __init__(self, exactness=EXACT, tolerance=None):
        self.exactness = exactness
        self.tolerance = tolerance

    def rewrite(self, graph):
        ops, sites = [], []
        for op in graph.ops:
            if op.kind == OpKind.RMSNORM and not sites:
                ops.append(replace(op, attributes={"epsilon": 0.5}))
                sites.append(RewriteSite((op.name,), {}))
            else:
                ops.append(op)
        return replace(graph, ops=tuple(ops)), tuple(sites), ()


class _SilentRewrite(GraphRewrite):
    """A pass that returns a different graph while reporting no site at all."""
    name = "SilentRewrite"

    def rewrite(self, graph):
        return replace(graph, name=graph.name + ".silent"), (), ()


class DeadOpEliminationRegressions(unittest.TestCase):
    def test_unreachable_operation_and_its_tensor_are_removed_exactly(self):
        graph = projection_graph(dead=True)
        rewritten, report = run_passes(graph, [DeadOpElimination()],
                                       verifier=GraphEquivalenceVerifier())
        record = report.passes[0]
        self.assertEqual(record.matched, 1)
        self.assertEqual(record.sites[0].ops, ("dead",))
        self.assertEqual(record.exactness, EXACT)
        self.assertTrue(record.verification["identical"])
        self.assertEqual(record.verification["max_abs_error"], 0.0)
        self.assertEqual([op.name for op in rewritten.ops], ["embed", "left", "right"])
        self.assertNotIn("unused", {tensor.name for tensor in rewritten.tensors})
        bindings = random_bindings(graph, seed=3)
        self.assertEqual(tensor_bytes(evaluate_graph(graph, bindings)["logits"]),
                         tensor_bytes(evaluate_graph(rewritten, bindings)["logits"]))

    def test_a_graph_with_nothing_dead_is_returned_untouched(self):
        graph = projection_graph()
        rewritten, report = run_passes(graph, [DeadOpElimination()],
                                       verifier=GraphEquivalenceVerifier())
        self.assertIs(rewritten, graph)
        self.assertEqual(report.passes[0].matched, 0)
        self.assertIsNone(report.passes[0].verification)

    def test_a_chain_of_dead_operations_is_removed_in_one_pass(self):
        graph = projection_graph(dead=True)
        extended = ModelGraph(
            name=graph.name,
            tensors=(*graph.tensors, activation("unused_twice", 3, 3)),
            ops=(*graph.ops, ModelOp("dead_consumer", OpKind.ADD, ["unused", "unused"],
                                     ["unused_twice"], {})),
            inputs=graph.inputs, outputs=graph.outputs, constants=graph.constants)
        rewritten, report = run_passes(extended, [DeadOpElimination()],
                                       verifier=GraphEquivalenceVerifier())
        self.assertEqual(report.passes[0].matched, 2)
        self.assertEqual([op.name for op in rewritten.ops], ["embed", "left", "right"])


class CommonSubexpressionEliminationRegressions(unittest.TestCase):
    def test_duplicate_projection_is_collapsed_without_changing_a_byte(self):
        graph = duplicate_graph()
        rewritten, report = run_passes(graph, [CommonSubexpressionElimination()],
                                       verifier=GraphEquivalenceVerifier())
        record = report.passes[0]
        self.assertEqual(record.matched, 1)
        self.assertEqual(record.sites[0].detail["reuses"], "first")
        self.assertTrue(record.verification["identical"])
        self.assertEqual([op.name for op in rewritten.ops], ["embed", "first", "sum"])
        self.assertEqual(rewritten.ops[-1].inputs, ("first", "first"))
        self.assertNotIn("second", {tensor.name for tensor in rewritten.tensors})

    def test_a_duplicate_the_graph_declares_as_an_output_is_kept(self):
        graph = duplicate_graph()
        declared = ModelGraph(name=graph.name, tensors=graph.tensors, ops=graph.ops,
                              inputs=graph.inputs, outputs=["logits", "second"],
                              constants=graph.constants)
        rewritten, report = run_passes(declared, [CommonSubexpressionElimination()],
                                       verifier=GraphEquivalenceVerifier())
        self.assertEqual(report.passes[0].matched, 0)
        self.assertIs(rewritten, declared)

    def test_operations_that_differ_only_in_an_attribute_are_not_merged(self):
        same = normalization_graph(1e-5, 1e-5)
        _, report = run_passes(same, [CommonSubexpressionElimination()],
                               verifier=GraphEquivalenceVerifier())
        self.assertEqual(report.passes[0].matched, 1)
        different = normalization_graph(1e-5, 1e-3)
        rewritten, report = run_passes(different, [CommonSubexpressionElimination()],
                                       verifier=GraphEquivalenceVerifier())
        self.assertEqual(report.passes[0].matched, 0)
        self.assertIs(rewritten, different)

    def test_a_duplicate_is_only_collapsed_after_its_own_inputs_were_rewired(self):
        # second reads first; once first collapses onto zeroth, second and
        # third become the same expression and collapse in the same pass.
        graph = repeated_chain_graph()
        rewritten, report = run_passes(graph, [CommonSubexpressionElimination()],
                                       verifier=GraphEquivalenceVerifier())
        self.assertEqual(report.passes[0].matched, 2)
        self.assertEqual([op.name for op in rewritten.ops], ["embed", "zeroth", "second", "sum"])
        self.assertTrue(report.passes[0].verification["identical"])


class MatMulProjectionFoldRegressions(unittest.TestCase):
    def test_chained_projections_fold_into_one_derived_constant(self):
        graph = projection_graph()
        rewritten, report = run_passes(graph, [MatMulProjectionFold()],
                                       verifier=GraphEquivalenceVerifier())
        record = report.passes[0]
        self.assertEqual(record.matched, 1)
        self.assertEqual(record.exactness, TOLERANCE)
        self.assertEqual(record.tolerance, MatMulProjectionFold.tolerance)
        self.assertEqual(record.sites[0].ops, ("left", "right"))
        self.assertEqual(record.sites[0].detail["folded_elements"], 8)
        self.assertEqual(record.sites[0].detail["replaced_elements"], 18)
        self.assertEqual(sorted(record.sites[0].detail["unreferenced_constants"]),
                         ["left_weight", "right_weight"])
        self.assertEqual([op.name for op in rewritten.ops], ["embed", "right"])
        derivation, = report.derivations
        self.assertEqual(derivation.sources, ("right_weight", "left_weight"))
        self.assertEqual(derivation.shape, (2, 4))
        self.assertIn(derivation.name, rewritten.constants)
        # A tolerance pass publishes both numbers: the bound and the measurement.
        self.assertGreater(record.verification["max_abs_error"], 0.0)
        self.assertLessEqual(record.verification["max_abs_error"], record.tolerance)
        self.assertFalse(record.verification["identical"])

    def test_the_derived_constant_is_the_product_of_the_two_it_replaces(self):
        graph = projection_graph()
        _, report = run_passes(graph, [MatMulProjectionFold()])
        derivation, = report.derivations
        bindings = random_bindings(graph, seed=11)
        folded = derivation.materialize(bindings)
        left, right = bindings["left_weight"], bindings["right_weight"]
        for row in range(len(folded)):
            for column in range(len(folded[0])):
                total = 0.0
                for index in range(len(right[0])):
                    total += right[row][index] * left[index][column]
                self.assertAlmostEqual(folded[row][column], total, places=6)

    def test_a_fold_that_would_cost_more_than_it_saves_is_declined(self):
        graph = projection_graph(hidden=10, middle=1, columns=10)
        _, report = run_passes(graph, [MatMulProjectionFold()],
                               verifier=GraphEquivalenceVerifier())
        self.assertEqual(report.passes[0].matched, 0)
        self.assertEqual(report.derivations, ())

    def test_an_intermediate_with_another_consumer_is_not_folded(self):
        graph = projection_graph()
        shared = ModelGraph(
            name=graph.name, tensors=(*graph.tensors, activation("kept", 3, 3)),
            ops=(*graph.ops, ModelOp("keep", OpKind.ADD, ["projected", "projected"],
                                     ["kept"], {})),
            inputs=graph.inputs, outputs=["logits", "kept"], constants=graph.constants)
        _, report = run_passes(shared, [MatMulProjectionFold()],
                               verifier=GraphEquivalenceVerifier())
        self.assertEqual(report.passes[0].matched, 0)

    def test_a_projection_by_a_computed_matrix_is_not_a_constant_fold(self):
        graph = projection_graph()
        ops = list(graph.ops)
        # The right operand is now an activation, so no constant can be derived.
        ops[2] = replace(ops[2], inputs=("projected", "projected"))
        tensors = [tensor for tensor in graph.tensors if tensor.name != "logits"]
        tensors.append(activation("logits", 3, 3))
        computed = ModelGraph(name=graph.name, tensors=tuple(tensors), ops=tuple(ops),
                              inputs=graph.inputs, outputs=graph.outputs,
                              constants=graph.constants)
        _, report = run_passes(computed, [MatMulProjectionFold()],
                               verifier=GraphEquivalenceVerifier())
        self.assertEqual(report.passes[0].matched, 0)


class PassManagerRegressions(unittest.TestCase):
    def test_a_pass_that_claims_exactness_and_changes_the_output_is_rejected(self):
        graph = llama_graph(layers=1)
        with self.assertRaisesRegex(RewriteError, "claims exactness"):
            run_passes(graph, [_EpsilonDrift()], verifier=GraphEquivalenceVerifier())

    def test_a_pass_that_exceeds_its_published_tolerance_is_rejected(self):
        graph = llama_graph(layers=1)
        with self.assertRaisesRegex(RewriteError, "exceeded its published tolerance"):
            run_passes(graph, [_EpsilonDrift(TOLERANCE, 1e-12)],
                       verifier=GraphEquivalenceVerifier())

    def test_the_same_drift_passes_when_its_tolerance_is_honest(self):
        graph = llama_graph(layers=1)
        _, report = run_passes(graph, [_EpsilonDrift(TOLERANCE, 10.0)],
                               verifier=GraphEquivalenceVerifier())
        self.assertEqual(report.passes[0].matched, 1)
        self.assertGreater(report.passes[0].verification["max_abs_error"], 0.0)

    def test_an_unverified_rewrite_is_recorded_as_unverified(self):
        graph = projection_graph(dead=True)
        _, report = run_passes(graph, [DeadOpElimination()])
        self.assertEqual(report.passes[0].matched, 1)
        self.assertIsNone(report.passes[0].verification)

    def test_a_rewrite_that_hides_its_sites_is_rejected(self):
        with self.assertRaisesRegex(RewriteError, "without reporting a site"):
            run_passes(projection_graph(), [_SilentRewrite()])

    def test_only_graph_rewrites_run(self):
        with self.assertRaises(ValueError):
            run_passes(projection_graph(), [object()])
        with self.assertRaises(ValueError):
            run_passes("not a graph", default_passes())

    def test_passes_compose_and_the_report_round_trips_through_json(self):
        graph = projection_graph(dead=True)
        rewritten, report = run_passes(graph, default_passes(),
                                       verifier=GraphEquivalenceVerifier())
        matched = {record.name: record.matched for record in report.passes}
        self.assertEqual(matched, {"DeadOpElimination": 1,
                                   "CommonSubexpressionElimination": 0,
                                   "MatMulProjectionFold": 1})
        self.assertEqual([op.name for op in rewritten.ops], ["embed", "right"])
        self.assertEqual(RewriteReport.from_json(report.to_json()).to_dict(), report.to_dict())
        self.assertEqual(report.to_dict()["schema_version"], 1)

    def test_a_report_with_an_unknown_schema_version_is_refused(self):
        data = RewriteReport("g", ()).to_dict()
        data["schema_version"] = 2
        with self.assertRaises(ValueError):
            RewriteReport.from_dict(data)

    def test_a_derivation_only_accepts_the_recipe_it_can_compute(self):
        with self.assertRaises(ValueError):
            ConstantDerivation("c", "convolve", ("a", "b"), (2, 2))
        with self.assertRaises(ValueError):
            ConstantDerivation("c", "matmul", ("a",), (2, 2))
        derivation = ConstantDerivation("c", "matmul", ("a", "b"), (1, 1))
        with self.assertRaises(ValueError):
            derivation.materialize({"a": [[1.0, 2.0]], "b": [[1.0, 1.0], [1.0, 1.0]]})


class LlamaGraphZeroRegressions(unittest.TestCase):
    """The measured zero: on the graph the pipeline lowers, nothing matches."""

    def test_every_pass_matches_zero_sites_on_the_lowered_llama_graph(self):
        graph = llama_graph(layers=2, sequence=4)
        self.assertEqual(len(graph.ops), 33)
        self.assertEqual(sum(1 for op in graph.ops if op.kind == OpKind.MATMUL), 15)
        rewritten, report = run_passes(graph, default_passes(),
                                       verifier=GraphEquivalenceVerifier())
        self.assertEqual(report.matched, 0)
        self.assertEqual([record.matched for record in report.passes], [0, 0, 0])
        self.assertIs(rewritten, graph)
        self.assertEqual(report.derivations, ())

    def test_no_matmul_in_the_lowered_graph_consumes_another_matmul(self):
        graph = llama_graph(layers=2, sequence=4)
        produced = {op.outputs[0]: op.kind for op in graph.ops}
        pairs = [op.name for op in graph.ops if op.kind == OpKind.MATMUL
                 and produced.get(op.inputs[0]) == OpKind.MATMUL]
        self.assertEqual(pairs, [])

    def test_the_zero_survives_the_real_nexalm512_architecture(self):
        definition = ROOT / "models/nexalm512/architecture.nxl"
        models = compile_model_definition(definition.read_text(encoding="utf-8"))
        for name, config in sorted(models.items()):
            with self.subTest(model=name):
                graph = lower_model(config, 4)
                # Too large for the evaluator; the passes still run on it, and
                # a pass that matched nothing has nothing to verify.
                _, report = run_passes(graph, default_passes())
                self.assertEqual(report.matched, 0, report.to_dict())


class GraphToolRegressions(unittest.TestCase):
    """The CLI is the only place these two modules meet a user."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools"))
        self.addCleanup(lambda: sys.path.remove(str(ROOT / "tools")))
        import nexa_graph
        self.tool = nexa_graph
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-graph-tool-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def capture(self, argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(self.tool.main(argv), 0)
        return json.loads(stream.getvalue())

    def test_rewrite_reports_the_zero_and_writes_the_unchanged_graph(self):
        source = self.directory / "graph.json"
        graph = llama_graph(layers=1)
        source.write_text(graph.to_json(), encoding="utf-8")
        target = self.directory / "out" / "graph.json"
        report = self.capture(["rewrite", "--graph", str(source), "--out", str(target)])
        self.assertEqual(report["matched"], 0)
        self.assertEqual(len(report["passes"]), 3)
        self.assertEqual(ModelGraph.from_json(target.read_text(encoding="utf-8")).to_dict(),
                         graph.to_dict())

    def test_rewrite_of_a_foldable_graph_publishes_its_derivation(self):
        source = self.directory / "folds.json"
        source.write_text(projection_graph().to_json(), encoding="utf-8")
        report = self.capture(["rewrite", "--graph", str(source)])
        self.assertEqual(report["matched"], 1)
        self.assertEqual(len(report["derivations"]), 1)
        fold = [record for record in report["passes"] if record["name"] == "MatMulProjectionFold"][0]
        self.assertLessEqual(fold["verification"]["max_abs_error"], fold["tolerance"])

    def test_regions_reports_the_summary_for_a_named_model(self):
        payload = self.capture(["regions", "--definition",
                                str(ROOT / "models/nexalm512/architecture.nxl"),
                                "--model", "NexaLM512_R0", "--sequence-length", "4",
                                "--tile-rows", "1"])
        self.assertEqual(payload["summary"]["regions"], 17)
        self.assertEqual(len(payload["regions"]), 17)
        self.assertLess(payload["summary"]["saved_fraction"], 0.02)

    def test_a_config_source_is_lowered_at_the_requested_length(self):
        source = self.directory / "config.json"
        source.write_text(ModelConfig(name="tiny", vocab_size=13, hidden_size=8,
                                      intermediate_size=12, num_hidden_layers=1,
                                      num_attention_heads=2, num_key_value_heads=1,
                                      max_position_embeddings=12).to_json(), encoding="utf-8")
        payload = self.capture(["regions", "--config", str(source), "--sequence-length", "3"])
        self.assertTrue(all(region["rows"] == 3 for region in payload["regions"]))

    def test_an_ambiguous_or_unknown_model_name_is_an_error(self):
        definition = str(ROOT / "models/nexalm512/architecture.nxl")
        for argv in (["regions", "--definition", definition],
                     ["regions", "--definition", definition, "--model", "absent"],
                     ["regions", "--graph", definition, "--model", "NexaLM512_R0"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.tool.main(argv)


if __name__ == "__main__":
    unittest.main()

"""Streaming regions: the cut rule, byte identity under tiling, and the numbers.

Detection is analysis, so the proof has to be a driver: this file evaluates a
graph whole, then evaluates every detected region tile by tile, and demands
identical float32 bytes for every tensor at several tile heights. Two negative
controls keep that from being vacuous — a driver that forgets the tile's row
offset, and a region drawn across CausalAttention, both of which must diverge.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.graph_eval import evaluate_graph, evaluate_op, random_bindings, tensor_bytes
from compiler.model_config import ModelConfig
from compiler.model_definition import compile_model_definition
from compiler.model_ir import DType, ModelGraph, ModelOp, OpKind, TensorDesc
from compiler.model_lowering import derive_activation_requests, lower_model
from compiler.planner.streaming import (STREAMING_AXIS, StreamingRegion,
                                        derive_streamed_activation_requests,
                                        detect_streaming_regions, row_aligned_operands,
                                        streaming_summary)


def tiny_config(layers=2, heads=2, kv_heads=1):
    return ModelConfig(name="tiny", vocab_size=13, hidden_size=heads * 4,
                       intermediate_size=heads * 6, num_hidden_layers=layers,
                       num_attention_heads=heads, num_key_value_heads=kv_heads,
                       max_position_embeddings=16)


def computed_weight_graph(rows=4, hidden=4, vocab=6):
    """A MatMul whose second operand is an activation, not a constant.

    The lowered Llama graph never does this, so nothing in it exercises the
    whole-operand condition; this graph exists to make that condition fail.
    """
    tensors = (TensorDesc("tokens", (rows,), DType.U32, DType.U32),
               TensorDesc("embed_weight", (vocab, hidden)),
               TensorDesc("norm_weight", (hidden,)),
               TensorDesc("embedding", (rows, hidden)),
               TensorDesc("normalized", (rows, hidden)),
               TensorDesc("logits", (rows, rows)))
    ops = (ModelOp("embed", OpKind.EMBEDDING, ["tokens", "embed_weight"], ["embedding"], {}),
           ModelOp("norm", OpKind.RMSNORM, ["embedding", "norm_weight"], ["normalized"],
                   {"epsilon": 1e-5}),
           ModelOp("pair", OpKind.MATMUL, ["normalized", "embedding"], ["logits"],
                   {"transpose_b": True}))
    return ModelGraph(name="computed", tensors=tensors, ops=ops, inputs=["tokens"],
                      outputs=["logits"], constants=["embed_weight", "norm_weight"])


def drive_tiled(graph, whole, regions, tile_rows, *, row_offsets=True):
    """Evaluate a graph with each region walked one row tile at a time.

    Everything outside a region runs whole, which is what a streaming executor
    would do around an attention barrier. row_offsets=False is the negative
    control: the same driver, with the position each tile starts at thrown away.
    """
    values = {name: whole[name] for name in (*graph.inputs, *graph.constants)}
    by_start = {region.start: region.retile(tile_rows) for region in regions}
    index = 0
    while index < len(graph.ops):
        region = by_start.get(index)
        if region is None:
            op = graph.ops[index]
            values[op.outputs[0]] = evaluate_op(op, [values[name] for name in op.inputs])
            index += 1
            continue
        produced = [graph.ops[position].outputs[0]
                    for position in range(region.start, region.end)]
        collected = {name: [] for name in produced}
        for low, high in region.row_slices():
            tile = {}
            for position in range(region.start, region.end):
                op = graph.ops[position]
                try:
                    aligned = row_aligned_operands(op)
                except ValueError:
                    # The assumption under test when a region is drawn over an
                    # operator that has no row-aligned contract.
                    aligned = tuple(range(len(op.inputs)))
                arguments = []
                for offset, name in enumerate(op.inputs):
                    if name in tile:
                        arguments.append(tile[name])
                    elif offset in aligned:
                        arguments.append(values[name][low:high])
                    else:
                        arguments.append(values[name])
                tile[op.outputs[0]] = evaluate_op(op, arguments,
                                                  row_offset=low if row_offsets else 0)
            for name in produced:
                collected[name].extend(tile[name])
        for name in produced:
            values[name] = tuple(collected[name])
        index = region.end
    return values


class StreamingDetectionRegressions(unittest.TestCase):
    def setUp(self):
        self.graph = lower_model(tiny_config(), 4)
        self.regions = detect_streaming_regions(self.graph)

    def test_attention_ends_a_region_and_never_sits_inside_one(self):
        barriers = [index for index, op in enumerate(self.graph.ops)
                    if op.kind == OpKind.CAUSAL_ATTENTION]
        self.assertEqual(len(barriers), 2)
        covered = {index for region in self.regions
                   for index in range(region.start, region.end)}
        self.assertFalse(covered & set(barriers))
        for barrier in barriers:
            self.assertTrue(any(region.end == barrier for region in self.regions))

    def test_regions_are_disjoint_ordered_and_only_hold_streamable_operators(self):
        previous_end = -1
        streamable = {OpKind.MATMUL, OpKind.EMBEDDING, OpKind.RMSNORM, OpKind.ROPE,
                      OpKind.SWIGLU, OpKind.ADD}
        for region in self.regions:
            self.assertLess(previous_end, region.start)
            self.assertEqual(region.axis, STREAMING_AXIS)
            self.assertEqual(region.rows, 4)
            self.assertEqual(region.tile_rows, region.rows)
            self.assertTrue(region.interior_tensors)
            self.assertLessEqual(region.live_tile_buffers, len(region.interior_tensors))
            for index in range(region.start, region.end):
                self.assertIn(self.graph.ops[index].kind, streamable)
            previous_end = region.end

    def test_a_tensor_read_after_the_region_is_boundary_not_interior(self):
        first = self.regions[0]
        self.assertIn("embedding", first.boundary_tensors)
        self.assertNotIn("embedding", first.interior_tensors)
        # The query projection dies inside the region that produced it.
        self.assertIn("layers.0.q", first.interior_tensors)
        self.assertIn("logits", self.regions[-1].boundary_tensors)

    def test_live_tile_buffer_counts_are_the_peak_of_the_serial_order(self):
        self.assertEqual([region.live_tile_buffers for region in self.regions], [3, 4, 4])
        self.assertEqual([len(region.interior_tensors) for region in self.regions], [3, 10, 9])
        # The MLP region holds ten interior values across its span and never
        # four of them at once; that gap is the whole argument for tiling.
        self.assertLess(self.regions[1].live_tile_buffers, len(self.regions[1].interior_tensors))

    def test_weights_are_constants_of_the_region_and_never_tiled(self):
        for region in self.regions:
            for name in region.constant_tensors:
                self.assertIn(name, self.graph.constants)
            self.assertFalse(set(region.constant_tensors) & set(region.interior_tensors))

    def test_detection_accepts_a_tile_height_and_refuses_an_impossible_one(self):
        regions = detect_streaming_regions(self.graph, tile_rows=2)
        self.assertTrue(all(region.tile_rows == 2 for region in regions))
        self.assertTrue(all(region.tiles == 2 for region in regions))
        for invalid in (0, 5, 1.0, True):
            with self.subTest(tile_rows=invalid), self.assertRaises(ValueError):
                detect_streaming_regions(self.graph, tile_rows=invalid)

    def test_a_matmul_against_an_activation_is_not_streamable(self):
        graph = computed_weight_graph()
        self.assertEqual(detect_streaming_regions(graph), ())

    def test_tiling_that_matmul_anyway_changes_the_result(self):
        graph = computed_weight_graph()
        whole = evaluate_graph(graph, random_bindings(graph, seed=4))
        forced = StreamingRegion(name="forced", start=0, end=3, axis=STREAMING_AXIS,
                                 rows=4, tile_rows=4, boundary_tensors=("logits",),
                                 interior_tensors=("normalized",), constant_tensors=(),
                                 live_tile_buffers=1)
        tiled = drive_tiled(graph, whole, (forced,), 2)
        self.assertNotEqual(tensor_bytes(whole["logits"]), tensor_bytes(tiled["logits"]))

    def test_row_slices_cover_the_extent_once_without_overlap(self):
        for tile_rows in (1, 2, 3, 4):
            region = self.regions[0].retile(tile_rows)
            slices = region.row_slices()
            self.assertEqual(slices[0][0], 0)
            self.assertEqual(slices[-1][1], region.rows)
            self.assertEqual(sum(high - low for low, high in slices), region.rows)
            self.assertEqual(len(slices), region.tiles)


class StreamingByteIdentityRegressions(unittest.TestCase):
    """Whole evaluation against tiled evaluation, with no tolerance allowed."""

    def setUp(self):
        self.graph = lower_model(tiny_config(), 6)
        self.regions = detect_streaming_regions(self.graph)
        self.bindings = random_bindings(self.graph, seed=91)
        self.whole = evaluate_graph(self.graph, self.bindings)

    def assert_identical(self, tiled):
        for tensor in self.graph.tensors:
            self.assertEqual(tensor_bytes(self.whole[tensor.name]),
                             tensor_bytes(tiled[tensor.name]), tensor.name)

    def test_every_tensor_matches_byte_for_byte_at_every_tile_height(self):
        for tile_rows in (1, 2, 3, 4, 5, 6):
            with self.subTest(tile_rows=tile_rows):
                self.assert_identical(drive_tiled(self.graph, self.whole, self.regions, tile_rows))

    def test_a_driver_that_forgets_the_tile_row_offset_diverges(self):
        tiled = drive_tiled(self.graph, self.whole, self.regions, 2, row_offsets=False)
        self.assertNotEqual(tensor_bytes(self.whole["logits"]), tensor_bytes(tiled["logits"]))
        self.assertNotEqual(tensor_bytes(self.whole["layers.0.q_rope"]),
                            tensor_bytes(tiled["layers.0.q_rope"]))
        # Tile zero starts at row zero, so its rows are right either way.
        self.assertEqual(self.whole["layers.0.q_rope"][0], tiled["layers.0.q_rope"][0])

    def test_a_region_drawn_across_attention_produces_different_bytes(self):
        barrier = next(index for index, op in enumerate(self.graph.ops)
                       if op.kind == OpKind.CAUSAL_ATTENTION)
        first = self.regions[0]
        # Same start, extended past the barrier: exactly the region detection
        # refuses to emit, and the reason it refuses.
        invalid = StreamingRegion(name="invalid", start=first.start, end=barrier + 1,
                                  axis=STREAMING_AXIS, rows=first.rows, tile_rows=first.rows,
                                  boundary_tensors=(), interior_tensors=("layers.0.q",),
                                  constant_tensors=(), live_tile_buffers=1)
        tiled = drive_tiled(self.graph, self.whole, (invalid,), 2)
        self.assertNotEqual(tensor_bytes(self.whole["layers.0.attention"]),
                            tensor_bytes(tiled["layers.0.attention"]))
        # The first tile is still right: rows 0..1 only ever attend to rows 0..1.
        self.assertEqual(self.whole["layers.0.attention"][0], tiled["layers.0.attention"][0])

    def test_tiling_a_graph_with_a_single_token_is_the_whole_evaluation(self):
        graph = lower_model(tiny_config(layers=1), 1)
        bindings = random_bindings(graph, seed=7)
        whole = evaluate_graph(graph, bindings)
        tiled = drive_tiled(graph, whole, detect_streaming_regions(graph), 1)
        for tensor in graph.tensors:
            self.assertEqual(tensor_bytes(whole[tensor.name]), tensor_bytes(tiled[tensor.name]))


class StreamingRequestRegressions(unittest.TestCase):
    def setUp(self):
        self.graph = lower_model(tiny_config(), 4)
        self.regions = detect_streaming_regions(self.graph)

    def test_only_interior_tensors_shrink_and_lifetimes_are_preserved(self):
        baseline = {request.name: request for request in derive_activation_requests(self.graph)}
        streamed = {request.name: request
                    for request in derive_streamed_activation_requests(self.graph, self.regions, 1)}
        self.assertEqual(set(baseline), set(streamed))
        interior = {name for region in self.regions for name in region.interior_tensors}
        for name, request in streamed.items():
            self.assertEqual((request.start, request.end),
                             (baseline[name].start, baseline[name].end))
            if name in interior:
                self.assertEqual(request.size_bytes, baseline[name].size_bytes // 4)
            else:
                self.assertEqual(request.size_bytes, baseline[name].size_bytes)

    def test_a_tile_as_tall_as_the_sequence_reproduces_the_baseline_requests(self):
        baseline = derive_activation_requests(self.graph)
        streamed = derive_streamed_activation_requests(self.graph, self.regions, 4)
        self.assertEqual([request.to_dict() for request in baseline],
                         [request.to_dict() for request in streamed])

    def test_impossible_tiles_and_overlapping_regions_are_refused(self):
        with self.assertRaises(ValueError):
            derive_streamed_activation_requests(self.graph, self.regions, 5)
        with self.assertRaises(ValueError):
            derive_streamed_activation_requests(self.graph, self.regions, 0)
        with self.assertRaises(ValueError):
            derive_streamed_activation_requests(self.graph, (self.regions[0], self.regions[0]), 1)
        with self.assertRaises(ValueError):
            derive_streamed_activation_requests(self.graph, ("not a region",), 1)


class StreamingMeasurementRegressions(unittest.TestCase):
    """The honest headline: on a real architecture the saving is about 2%."""

    def summary(self, config, sequence, tile_rows=1):
        graph = lower_model(config, sequence)
        return dict(streaming_summary(graph, detect_streaming_regions(graph), tile_rows))

    def test_the_tiny_fixture_flatters_streaming_and_the_real_model_does_not(self):
        tiny = self.summary(tiny_config(), 4)
        self.assertEqual(tiny["arena_peak_bytes"], 832)
        self.assertEqual(tiny["streamed_arena_peak_bytes"], 512)
        # 38% on a 13-token vocabulary; the same analysis on a real vocabulary
        # is an order of magnitude smaller, which is why both are measured.
        self.assertGreater(tiny["saved_fraction"], 0.38)

    def test_the_real_architecture_saves_about_two_percent_because_logits_dominate(self):
        models = compile_model_definition(
            (ROOT / "models/nexalm512/architecture.nxl").read_text(encoding="utf-8"))
        config = models["NexaLM512_R0"]
        summary = self.summary(config, 512)
        self.assertEqual(summary["regions"], 17)
        self.assertEqual(summary["arena_peak_bytes"], 68681728)
        self.assertEqual(summary["streamed_arena_peak_bytes"], 67111936)
        self.assertEqual(summary["saved_bytes"], 1569792)
        self.assertLess(summary["saved_fraction"], 0.023)
        self.assertGreater(summary["saved_fraction"], 0.022)
        logits_bytes = 512 * config.vocab_size * 4
        self.assertEqual(summary["largest_unstreamable_bytes"], logits_bytes)
        self.assertGreater(logits_bytes / summary["arena_peak_bytes"], 0.97)

    def test_the_saving_stays_small_at_every_sequence_length_and_tile_height(self):
        models = compile_model_definition(
            (ROOT / "models/nexalm512/architecture.nxl").read_text(encoding="utf-8"))
        for name, config in sorted(models.items()):
            for sequence in (4, 128, 512):
                for tile_rows in (1, 4):
                    with self.subTest(model=name, sequence=sequence, tile_rows=tile_rows):
                        summary = self.summary(config, sequence, tile_rows)
                        self.assertLess(summary["saved_fraction"], 0.031)
                        self.assertGreaterEqual(summary["saved_bytes"], 0)

    def test_requested_bytes_and_arena_peak_disagree_on_purpose(self):
        summary = self.summary(tiny_config(), 4)
        self.assertGreater(summary["activation_request_bytes"], summary["arena_peak_bytes"])
        self.assertEqual(summary["ops"], 33)
        self.assertEqual(summary["streamed_ops"], 31)
        self.assertEqual(summary["interior_tensors"], 22)


if __name__ == "__main__":
    unittest.main()

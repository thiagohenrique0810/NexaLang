"""Model graph and arena planning contracts; no model or GPU dependencies."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler import (
    DType, HardwareEstimate, HardwareMeasurement, HardwareProfile,
    MemoryAllocation, MemoryBudgetError, MemoryPlan, MemoryPlanner, MemoryRequest,
    ModelGraph, ModelOp, TensorDesc, parse_memory_size,
)


def matmul_graph(*, transpose=False, packed=False):
    right_shape = (4, 3) if transpose else (3, 4)
    return ModelGraph(
        name="projection",
        tensors=[TensorDesc("x", (2, 3)),
                 TensorDesc("w", right_shape, storage_dtype="q4" if packed else "f32",
                            storage_nbytes=14 if packed else None),
                 TensorDesc("y", (2, 4))],
        ops=[ModelOp("projection", "MatMul", ["x", "w"], ["y"],
                     {"transpose_b": True} if transpose else {})],
        inputs=["x"], constants=["w"], outputs=["y"],
    )


class TensorAndGraphRegressions(unittest.TestCase):
    def test_dense_and_packed_storage_have_independent_logical_sizes(self):
        dense = TensorDesc("dense", (3, 5), logical_dtype="f32", storage_dtype="f16")
        packed = TensorDesc("packed", (3, 5), storage_dtype="q3", storage_nbytes=14)
        self.assertEqual(dense.numel, 15)
        self.assertEqual(dense.storage_nbytes, 30)
        self.assertEqual(packed.storage_nbytes, 14)
        self.assertEqual(packed.logical_dtype, DType.F32)
        self.assertEqual(TensorDesc.from_dict(packed.to_dict()), packed)

    def test_shapes_sizes_and_alignment_are_validated_before_use(self):
        invalid = [dict(shape=()), dict(shape=(0, 2)), dict(shape=(-1, 2)),
                   dict(shape=(True, 2)), dict(shape=(2.0, 2)), dict(shape="2,2"),
                   dict(storage_dtype="q4"), dict(storage_dtype="q4", storage_nbytes=1),
                   dict(storage_nbytes=17), dict(storage_nbytes=True),
                   dict(logical_dtype="q4"), dict(alignment=3), dict(alignment=True),
                   dict(alignment=0), dict(tier="invented")]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TensorDesc(**{"name": "x", "shape": (2, 2), **kwargs})

    def test_graph_roundtrip_preserves_packed_layout_and_transpose(self):
        graph = matmul_graph(transpose=True, packed=True)
        restored = ModelGraph.from_json(graph.to_json())
        self.assertEqual(restored.to_dict(), graph.to_dict())
        self.assertTrue(restored.ops[0].attributes["transpose_b"])
        self.assertEqual(restored.tensors[1].shape, (4, 3))

    def test_json_rejects_unknown_keys_versions_duplicates_and_nonfinite_numbers(self):
        original = matmul_graph().to_dict()
        invalid = []
        for key, value in (("unknown", 1), ("schema_version", 2), ("schema_version", True)):
            data = copy.deepcopy(original)
            data[key] = value
            invalid.append(data)
        data = copy.deepcopy(original)
        data["tensors"][0]["layout"] = "unknown"
        invalid.append(data)
        data = copy.deepcopy(original)
        del data["constants"]
        invalid.append(data)
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                ModelGraph.from_json(json.dumps(data))
        for text in ('{"name":"a","name":"b"}', '{"value": NaN}', '{"value": Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                ModelGraph.from_json(text)

    def test_matmul_checks_dimensions_rank_dtype_and_transpose_type(self):
        graph = matmul_graph()
        for replacement in (TensorDesc("w", (4, 3)), TensorDesc("w", (12,)),
                            TensorDesc("w", (3, 4), logical_dtype="i32")):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                replace(graph, tensors=(graph.tensors[0], replacement, graph.tensors[2]))
        with self.assertRaises(ValueError):
            replace(graph, tensors=(*graph.tensors[:2], TensorDesc("y", (2, 5))))
        for attributes in ({"transpose_b": 1}, {"transpose_b": "true"}, {"transpose_a": True}):
            with self.subTest(attributes=attributes), self.assertRaises(ValueError):
                ModelOp("bad", "MatMul", ["x", "w"], ["y"], attributes)
        with self.assertRaises(ValueError):
            ModelOp("unknown", "Attention", ["x", "w"], ["y"])

    def test_topology_requires_declared_sources_before_use(self):
        tensors = [TensorDesc(name, (2, 2)) for name in ("x", "w", "tmp", "y")]
        first = ModelOp("first", "MatMul", ["x", "w"], ["tmp"])
        second = ModelOp("second", "MatMul", ["tmp", "w"], ["y"])
        graph = ModelGraph("chain", tensors, [first, second], ["x"], ["y"], ["w"])
        self.assertIs(graph.validate(), graph)
        for ops in ([second, first], [first, replace(second, inputs=("missing", "w"))],
                    [first, replace(second, outputs=("tmp",))],
                    [replace(first, outputs=("x",)), second]):
            with self.subTest(ops=ops), self.assertRaises(ValueError):
                replace(graph, ops=ops)
        with self.assertRaises(ValueError):
            replace(graph, constants=[])
        with self.assertRaises(ValueError):
            replace(graph, constants=["w", "x"])

    def test_graph_rejects_duplicate_names_and_unproduced_tensors(self):
        graph = matmul_graph()
        for change in (dict(tensors=(*graph.tensors, graph.tensors[0])),
                       dict(tensors=(*graph.tensors, TensorDesc("unused", (2, 2)))),
                       dict(ops=(*graph.ops, graph.ops[0])), dict(inputs=["x", "x"]),
                       dict(outputs=[]), dict(ops=[])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(graph, **change)

    def test_repeated_input_is_legal_and_graph_metadata_is_immutable(self):
        graph = ModelGraph("square", [TensorDesc("x", (2, 2)), TensorDesc("y", (2, 2))],
                           [ModelOp("square", "MatMul", ["x", "x"], ["y"])], ["x"], ["y"])
        self.assertEqual(graph.ops[0].inputs, ("x", "x"))
        with self.assertRaises(TypeError):
            graph.ops[0].attributes["transpose_b"] = True


class MemoryPlanningRegressions(unittest.TestCase):
    def test_parse_sizes_distinguishes_decimal_and_binary_units(self):
        cases = {"512MB": 512_000_000, "512MiB": 536_870_912, "1KB": 1000,
                 "1KiB": 1024, "2GB": 2_000_000_000, "2GiB": 2_147_483_648,
                 " 1.5 MiB ": 1_572_864, "0.001KB": 1, "0B": 0, 0: 0, 19: 19}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(parse_memory_size(value), expected)
        self.assertEqual(parse_memory_size("123456789012345678901234567890B"),
                         123456789012345678901234567890)

    def test_parse_sizes_rejects_ambiguous_invalid_or_fractional_bytes(self):
        for value in (True, False, -1, 1.5, None, "512", "512M", "512Mb", "512mb",
                      "512Mi", "1e3B", "-1B", "0.1B", "NaNB", "+1B", "1TB"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_memory_size(value)

    def test_half_open_lifetimes_reuse_the_same_buffer_at_the_boundary(self):
        requests = [MemoryRequest("first", 65, 0, 2), MemoryRequest("next", 65, 2, 4)]
        plan = MemoryPlanner().plan(requests, {"host": 65})
        self.assertEqual(plan.allocations["first"].offset, 0)
        self.assertEqual(plan.allocations["next"].offset, 0)
        self.assertEqual(plan.peak_bytes, {"host": 65})
        self.assertIs(plan.validate(requests), plan)

    def test_live_buffers_are_disjoint_and_alignment_holes_count_against_budget(self):
        requests = [MemoryRequest("a", 65, 0, 3), MemoryRequest("b", 1, 1, 2)]
        plan = MemoryPlanner().plan(requests, {"host": 129})
        self.assertEqual(plan.allocations["b"].offset, 128)
        self.assertEqual(plan.peak_bytes["host"], 129)
        with self.assertRaises(MemoryBudgetError):
            MemoryPlanner().plan(requests, {"host": 128})

    def test_first_fit_reclaims_a_hole_between_live_allocations(self):
        requests = [MemoryRequest("a", 64, 0, 4), MemoryRequest("b", 64, 1, 3),
                    MemoryRequest("c", 64, 2, 5), MemoryRequest("d", 32, 3, 4)]
        plan = MemoryPlanner().plan(requests, {"host": 192})
        self.assertEqual(plan.allocations["d"].offset, plan.allocations["b"].offset)
        self.assertEqual(plan.peak_bytes["host"], 192)

    def test_tiers_have_independent_arenas_and_reserves_reduce_the_budget(self):
        requests = [MemoryRequest("cpu", 64, 0, 2),
                    MemoryRequest("gpu", 64, 0, 2, tier="device")]
        plan = MemoryPlanner().plan(requests, {"host": 96, "device": 80}, {"host": 32, "device": 16})
        self.assertEqual([a.offset for a in plan.allocations.values()], [0, 0])
        self.assertEqual(plan.peak_bytes, {"host": 64, "device": 64})
        with self.assertRaises(MemoryBudgetError) as caught:
            MemoryPlanner().plan(requests, {"host": 95, "device": 80}, {"host": 32, "device": 16})
        self.assertEqual(caught.exception.tier, "host")
        self.assertEqual(caught.exception.required_bytes, 64)
        self.assertEqual(caught.exception.reserve_bytes, 32)

    def test_budget_rejection_needs_no_model_sized_allocation(self):
        huge = MemoryRequest("too_large", 1024 ** 5, 0, 1)
        with self.assertRaises(MemoryBudgetError):
            MemoryPlanner().plan([huge], {"host": parse_memory_size("512MiB")})
        empty = MemoryPlanner().plan([], {"host": 0})
        self.assertEqual(empty.peak_bytes, {"host": 0})

    def test_invalid_requests_and_budget_contracts_are_rejected(self):
        for change in (dict(size_bytes=0), dict(size_bytes=True), dict(start=-1), dict(start=True),
                       dict(end=0), dict(end=0.5), dict(start=1, end=1), dict(alignment=3),
                       dict(alignment=0), dict(name=""), dict(tier="")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                MemoryRequest(**{"name": "x", "size_bytes": 4, "start": 0, "end": 1, **change})
        request = MemoryRequest("x", 4, 0, 1)
        for requests, budgets, reserves in (([request, request], {"host": 8}, None),
                                             ([request], {}, None),
                                             ([request], {"host": True}, None),
                                             ([], {"host": -1}, None),
                                             ([], {"host": 8}, {"device": 0}),
                                             ([], {"host": 8}, {"host": 9})):
            with self.subTest(budgets=budgets, reserves=reserves), self.assertRaises(ValueError):
                MemoryPlanner().plan(requests, budgets, reserves)

    def test_plans_are_deterministic_independent_of_request_order(self):
        requests = [MemoryRequest("c", 41, 0, 2), MemoryRequest("b", 50, 0, 3),
                    MemoryRequest("a", 65, 0, 2), MemoryRequest("d", 32, 2, 4)]
        planner = MemoryPlanner()
        expected = planner.plan(requests, {"host": 1024}).to_json()
        for seed in range(10):
            random.Random(seed).shuffle(requests)
            self.assertEqual(planner.plan(requests, {"host": 1024}).to_json(), expected)

    def test_live_data_survives_reuse_in_actual_arena_buffers(self):
        rng = random.Random(713)
        requests = []
        for index in range(80):
            start = rng.randrange(12)
            requests.append(MemoryRequest(f"buffer{index}", rng.randrange(1, 81), start,
                                          start + rng.randrange(1, 5), 2 ** rng.randrange(7),
                                          "host" if index % 2 else "device"))
        plan = MemoryPlanner().plan(requests, {"host": 8192, "device": 8192})
        arenas = {tier: bytearray(size) for tier, size in plan.peak_bytes.items()}
        tags = {request.name: index + 1 for index, request in enumerate(requests)}
        for time in range(16):
            for allocation in plan.allocations.values():
                if allocation.start == time:
                    arena = arenas[allocation.tier]
                    arena[allocation.offset:allocation.offset + allocation.size_bytes] = bytes([tags[allocation.name]]) * allocation.size_bytes
            for allocation in plan.allocations.values():
                if allocation.start <= time < allocation.end:
                    stored = arenas[allocation.tier][allocation.offset:allocation.offset + allocation.size_bytes]
                    self.assertEqual(stored, bytes([tags[allocation.name]]) * allocation.size_bytes)

    def test_json_roundtrip_and_verification_reject_forged_allocations(self):
        requests = [MemoryRequest("a", 64, 0, 2), MemoryRequest("b", 64, 1, 3)]
        plan = MemoryPlanner().plan(requests, {"host": 256}, {"host": 64})
        restored = MemoryPlan.from_json(plan.to_json())
        self.assertEqual(restored.to_dict(), plan.to_dict())
        restored.validate(requests)
        corruptions = []
        overlap = plan.to_dict()
        overlap["allocations"]["b"]["offset"] = 0
        overlap["peak_bytes"]["host"] = 64
        corruptions.append(overlap)
        for field, value in (("offset", 1), ("size_bytes", 0), ("name", "wrong")):
            data = plan.to_dict()
            data["allocations"]["a"][field] = value
            corruptions.append(data)
        data = plan.to_dict()
        data["peak_bytes"]["host"] = 127
        corruptions.append(data)
        data = plan.to_dict()
        data["unexpected"] = 1
        corruptions.append(data)
        for data in corruptions:
            with self.subTest(data=data), self.assertRaises(ValueError):
                MemoryPlan.from_dict(data)
        with self.assertRaises(ValueError):
            restored.validate([replace(requests[0], size_bytes=32), requests[1]])
        with self.assertRaises(ValueError):
            MemoryPlan.from_json('{"schema_version":1,"schema_version":1}')


class HardwareProfileRegressions(unittest.TestCase):
    def test_cpu_detection_reports_only_observable_properties(self):
        with patch("compiler.hardware_profile.os.cpu_count", return_value=8), \
             patch("compiler.hardware_profile.platform.machine", return_value="test-cpu"), \
             patch("compiler.hardware_profile.platform.processor", return_value=""):
            profile = HardwareProfile.detect_cpu().to_dict()
        self.assertEqual(profile["device_type"], "cpu")
        self.assertEqual(profile["architecture"], "test-cpu")
        self.assertEqual(profile["logical_cpu_count"], 8)
        for key in ("processor", "physical_cpu_count", "total_memory_bytes", "cache_bytes",
                    "memory_bandwidth_bytes_per_second", "simd_width_bits"):
            self.assertIsNone(profile[key])
        self.assertIn(profile["capabilities"]["byte_order"], ("little", "big"))
        self.assertEqual(profile["measurements"], {})
        self.assertEqual(profile["estimates"], {})
        json.dumps(profile, allow_nan=False)

    def test_measurements_and_estimates_keep_their_provenance_separate(self):
        profile = replace(HardwareProfile.detect_cpu(),
                          measurements={"bandwidth": HardwareMeasurement(100, "B/s", "copy benchmark")},
                          estimates={"bandwidth": HardwareEstimate(200, "B/s", "vendor rating")})
        data = profile.to_dict()
        self.assertEqual(data["measurements"]["bandwidth"]["method"], "copy benchmark")
        self.assertEqual(data["estimates"]["bandwidth"]["basis"], "vendor rating")
        self.assertIsNone(data["memory_bandwidth_bytes_per_second"])
        for item in (HardwareMeasurement(float("nan"), "B/s", "test"),
                     HardwareEstimate(20, "B/s", "")):
            with self.assertRaises(ValueError):
                item.to_dict()


if __name__ == "__main__":
    unittest.main()

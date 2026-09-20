"""Selection by measured physical cost: what the files hold, not what codecs encode."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.precision_map import COSTS, POLICY_IDS, PrecisionMap, select_precision


def calibration(entries, *, measured=("q4", "f16"), **overrides):
    """Report with both byte counts: {tensor: (dense, dense_physical, {codec: ...})}."""
    tensors = []
    for name, dense, dense_physical, codecs in entries:
        tensors.append({
            "name": name, "dense_bytes": dense, "dense_physical_bytes": dense_physical,
            "codecs": {codec: {"packed_bytes": payload, "physical_bytes": physical,
                               "container_overhead_bytes": physical - payload,
                               "sensitivity": {"rmse": rmse}}
                       for codec, (payload, physical, rmse) in codecs.items()}})
    report = {"checkpoint": "/tmp/checkpoint", "tokens": [1, 3], "group_size": 32,
              "measured_codecs": list(measured), "sensitivity_measured": True,
              "quality_measured": False, "tensors": tensors}
    report.update(overrides)
    return report


# A small tensor, with the container overhead the fixture actually measures:
# 4096 bytes of header and block metadata for a packed file, none for a dense one.
SMALL = calibration([
    ("alpha", 512, 512, {"q4": (192, 4288, 0.30), "f16": (256, 256, 0.02)}),
    ("beta", 512, 512, {"q4": (192, 4288, 0.10), "f16": (256, 256, 0.01)}),
])

# A large tensor, where the same fixed overhead is noise.
LARGE = calibration([
    ("alpha", 2097152, 2097152, {"q4": (327680, 331776, 0.30), "f16": (1048576, 1048576, 0.02)}),
    ("beta", 2097152, 2097152, {"q4": (327680, 331776, 0.10), "f16": (1048576, 1048576, 0.01)}),
])


class PhysicalCostSelectionRegressions(unittest.TestCase):
    def test_payload_and_physical_disagree_on_a_small_tensor(self):
        # By payload, packing is the cheapest thing available.
        by_payload = select_precision(SMALL, max_rmse=1.0, cost="payload")
        self.assertEqual(set(by_payload.codecs.values()), {"q4"})
        self.assertEqual(by_payload.provenance["planned_bytes"], 384)
        # By what the files hold, packing this tensor makes the bundle bigger
        # than storing it dense, so the packed option never survives.
        by_physical = select_precision(SMALL, max_rmse=1.0, cost="physical")
        self.assertEqual(set(by_physical.codecs.values()), {"f16"})
        self.assertEqual(by_physical.provenance["planned_bytes"], 512)

    def test_on_a_large_tensor_the_overhead_stops_mattering(self):
        for cost in COSTS:
            precision = select_precision(LARGE, max_rmse=1.0, cost=cost)
            self.assertEqual(set(precision.codecs.values()), {"q4"}, cost)

    def test_a_dominated_packed_codec_disappears_under_physical_cost(self):
        # q4 costs more physically than f16 and errs more, so it is dominated:
        # no budget can make it the answer, and the cheapest plan is already f16.
        for budget in (512, 1024, 10_000):
            precision = select_precision(SMALL, budget_bytes=budget, cost="physical")
            self.assertNotIn("q4", set(precision.codecs.values()), budget)
            self.assertEqual(precision.provenance["cheapest_baseline_bytes"], 512)
        # Spare budget is spent climbing to dense, never on the dominated codec.
        self.assertEqual(set(select_precision(SMALL, budget_bytes=10_000,
                                              cost="physical").codecs.values()), {"f32"})

    def test_the_policy_id_records_which_cost_was_optimized(self):
        payload = select_precision(SMALL, max_rmse=1.0, cost="payload")
        physical = select_precision(SMALL, max_rmse=1.0, cost="physical")
        self.assertEqual(payload.policy_id, POLICY_IDS["payload"])
        self.assertEqual(physical.policy_id, POLICY_IDS["physical"])
        self.assertEqual((payload.cost_basis, physical.cost_basis), ("payload", "physical"))
        for precision in (payload, physical):
            restored = PrecisionMap.from_json(precision.to_json())
            self.assertEqual(restored.policy_id, precision.policy_id)
            self.assertEqual(dict(restored.codecs), dict(precision.codecs))

    def test_the_provenance_states_the_cost_basis_and_its_scope(self):
        precision = select_precision(SMALL, max_rmse=1.0, cost="physical")
        provenance = precision.provenance
        self.assertEqual(provenance["cost_basis"], "physical")
        self.assertIn("checksums", provenance["cost_scope"])
        self.assertEqual(provenance["policy"], POLICY_IDS["physical"])

    def test_a_report_without_physical_bytes_is_refused_by_name(self):
        report = calibration([("alpha", 512, 512, {"q4": (192, 4288, 0.30)})])
        del report["tensors"][0]["dense_physical_bytes"]
        with self.assertRaises(ValueError) as raised:
            select_precision(report, max_rmse=1.0, cost="physical")
        self.assertIn("dense_physical_bytes", str(raised.exception))
        # The same report still plans by payload; nothing regressed.
        self.assertTrue(select_precision(report, max_rmse=1.0, cost="payload").codecs)
        report = calibration([("alpha", 512, 512, {"q4": (192, 4288, 0.30)})])
        del report["tensors"][0]["codecs"]["q4"]["physical_bytes"]
        with self.assertRaises(ValueError) as raised:
            select_precision(report, max_rmse=1.0, cost="physical")
        self.assertIn("physical_bytes", str(raised.exception))

    def test_an_unknown_cost_basis_is_refused(self):
        for bad in ("bytes", "", None, "PHYSICAL"):
            with self.assertRaises(ValueError):
                select_precision(SMALL, max_rmse=1.0, cost=bad)

    def test_a_map_from_an_unknown_policy_is_refused(self):
        data = json.loads(select_precision(SMALL, max_rmse=1.0, cost="physical").to_json())
        data["policy_id"] = "GREEDY_SOMETHING_ELSE_V9"
        with self.assertRaises(ValueError):
            PrecisionMap.from_json(json.dumps(data))

    def test_a_physical_budget_below_the_cheapest_plan_is_refused_with_the_number(self):
        with self.assertRaises(ValueError) as raised:
            select_precision(SMALL, budget_bytes=100, cost="physical")
        self.assertIn("512", str(raised.exception))


class PhysicalCostMeasurementRegressions(unittest.TestCase):
    """The overhead the planner reasons about is the one the writer produces."""

    def sizes(self, rows, cols, group_size=32):
        from runtime.nexapack.bundle import PACKED_CODECS, _write_raw_matrix
        from runtime.nexapack.format import write_grouped_matrix
        work = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: [path.unlink() for path in work.iterdir() if path.is_file()])

        def source():
            return ([float((index * 7 % 13) - 6) for index in range(cols)] for _ in range(rows))

        packed = work / "packed.nxp"
        write_grouped_matrix(packed, rows, cols, group_size, source(),
                             block_rows=64, codec=PACKED_CODECS["q4"])
        _, f16 = _write_raw_matrix(work / "dense.f16", rows, cols, source(), 64, 2)
        _, f32 = _write_raw_matrix(work / "dense.f32", rows, cols, source(), 64, 4)
        return {"q4": packed.stat().st_size, "f16": f16, "f32": f32}

    def test_packing_a_small_matrix_produces_a_larger_file_than_not_packing_it(self):
        small = self.sizes(8, 64)
        self.assertGreater(small["q4"], small["f32"])
        self.assertGreater(small["q4"], small["f16"])

    def test_packing_a_large_matrix_produces_a_much_smaller_file(self):
        large = self.sizes(8, 16384)
        self.assertLess(large["q4"], large["f16"])
        self.assertLess(large["q4"] * 5, large["f32"])

    def test_the_container_overhead_is_fixed_and_not_proportional(self):
        overheads = []
        for cols in (64, 1024, 16384):
            values = 8 * cols
            # Q4 payload: four bits a value plus one F32 scale a group.
            payload = values // 2 + (values // 32) * 4
            overheads.append(self.sizes(8, cols)["q4"] - payload)
        # The same header and block metadata whatever the matrix size, which is
        # exactly why it only decides the choice for small tensors.
        self.assertEqual(len(set(overheads)), 1, overheads)
        self.assertGreater(overheads[0], 0)

    def test_the_turning_point_sits_between_two_and_eight_thousand_values(self):
        self.assertGreater(self.sizes(8, 256)["q4"], self.sizes(8, 256)["f16"])
        self.assertLess(self.sizes(8, 1024)["q4"], self.sizes(8, 1024)["f16"])


class PhysicalCostCliRegressions(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.report = self.directory / "calibration.json"
        self.report.write_text(json.dumps(SMALL))

    def plan(self, *extra):
        command = [sys.executable, str(ROOT / "tools/nexa_precision.py"), "plan",
                   "--calibration", str(self.report), "--max-rmse", "1.0", *extra]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_the_cli_plans_against_either_cost(self):
        self.assertEqual(set(self.plan()["codecs"].values()), {"q4"})
        self.assertEqual(set(self.plan("--cost", "payload")["codecs"].values()), {"q4"})
        physical = self.plan("--cost", "physical")
        self.assertEqual(set(physical["codecs"].values()), {"f16"})
        self.assertEqual(physical["policy_id"], POLICY_IDS["physical"])

    def test_the_cli_refuses_an_unknown_cost(self):
        command = [sys.executable, str(ROOT / "tools/nexa_precision.py"), "plan",
                   "--calibration", str(self.report), "--max-rmse", "1.0", "--cost", "nominal"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 2)

    def test_show_reads_back_a_physical_map(self):
        written = self.directory / "map.json"
        self.plan("--cost", "physical", "--out", str(written))
        command = [sys.executable, str(ROOT / "tools/nexa_precision.py"), "show", str(written)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("f16", result.stdout)


class PhysicalCostPipelineRegressions(unittest.TestCase):
    """The whole path: calibrate, plan against each cost, convert, compare disks."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from model_checkpoint_fixture import create_checkpoint
        self.directory = Path(tempfile.mkdtemp())
        self.checkpoint = self.directory / "checkpoint"
        create_checkpoint(self.checkpoint)

    def run_tool(self, tool, *arguments):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / tool), *arguments],
                                capture_output=True, text=True, timeout=900)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    @staticmethod
    def disk_bytes(bundle):
        return sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())

    def test_planning_by_physical_cost_produces_the_smaller_bundle(self):
        report = self.directory / "calibration.json"
        self.run_tool("nexa_calibrate.py", "--checkpoint", str(self.checkpoint), "--tokens", "1,3",
                      "--group-size", "4", "--block-rows", "3", "--tile-rows", "3",
                      "--memory-budget", "8MiB", "--codec", "q4", "--codec", "f16",
                      "--report", str(report))
        bundles = {}
        for cost in COSTS:
            plan = self.directory / f"plan-{cost}.json"
            planned = self.run_tool("nexa_precision.py", "plan", "--calibration", str(report),
                                    "--max-rmse", "10", "--cost", cost, "--out", str(plan))
            self.assertEqual(planned["policy_id"], POLICY_IDS[cost])
            bundle = self.directory / f"bundle-{cost}"
            self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint), "--out",
                          str(bundle), "--precision-map", str(plan), "--group-size", "4",
                          "--block-rows", "3")
            self.run_tool("nexa_inspect.py", str(bundle), "--verify")
            executed = self.run_tool("nexa_run.py", str(bundle), "--tokens", "1,3",
                                     "--tile-rows", "3", "--memory-budget", "8MiB")
            self.assertEqual(executed["token_ids"], [1, 3])
            bundles[cost] = (planned, self.disk_bytes(bundle))
        payload_plan, payload_disk = bundles["payload"]
        physical_plan, physical_disk = bundles["physical"]
        # The payload plan claims to be the cheaper one and is not: on this
        # checkpoint its packed containers cost more than the dense files.
        self.assertLess(payload_plan["provenance"]["planned_bytes"],
                        physical_plan["provenance"]["planned_bytes"])
        self.assertLess(physical_disk, payload_disk)


if __name__ == "__main__":
    unittest.main()

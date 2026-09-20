"""Precision map: budget-bounded selection, contract and application."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.precision_map import POLICY_ID, PrecisionMap, select_precision
from test_dense_weights_regressions import _DenseFixture


def calibration(entries, *, measured=("q4",), **overrides):
    """Report in the multi-codec shape: {tensor: {codec: (bytes, rmse)}}."""
    tensors = []
    for name, dense, codecs in entries:
        tensors.append({"name": name, "dense_bytes": dense,
                        "codecs": {codec: {"packed_bytes": size, "sensitivity": {"rmse": rmse}}
                                   for codec, (size, rmse) in codecs.items()}})
    report = {"checkpoint": "/tmp/checkpoint", "tokens": [1, 3], "group_size": 4,
              "measured_codecs": list(measured), "sensitivity_measured": True,
              "quality_measured": False, "tensors": tensors}
    report.update(overrides)
    return report


def single_codec(entries, **overrides):
    """The older single-codec report shape, which must still plan correctly."""
    report = {"checkpoint": "/tmp/checkpoint", "tokens": [1, 3], "group_size": 4,
              "sensitivity_measured": True, "quality_measured": False,
              "tensors": [{"name": name, "dense_bytes": dense, "packed_bytes": packed,
                           "sensitivity": {"rmse": rmse}} for name, dense, packed, rmse in entries]}
    report.update(overrides)
    return report


# One codec only: promoting beta costs the fewest bytes per avoided error.
REPORT = single_codec([
    ("alpha", 800, 100, 0.40),   # 0.40 / 700 bytes
    ("beta", 400, 100, 0.20),    # 0.20 / 300 bytes, the best ratio
    ("gamma", 900, 100, 0.10),   # 0.10 / 800 bytes, the worst ratio
])

# Two packed codecs plus dense: a ladder, not a binary choice.
LADDER = calibration([
    # q4 100B/0.40 -> q8 200B/0.05 -> f32 800B/0: the first step is the bargain.
    ("alpha", 800, {"q4": (100, 0.40), "q8": (200, 0.05)}),
    # q8 buys almost nothing here; dense is the only real improvement.
    ("beta", 400, {"q4": (100, 0.12), "q8": (200, 0.11)}),
], measured=("q4", "q8"))


class PrecisionSelectionRegressions(unittest.TestCase):
    def test_a_ladder_climbs_the_best_step_first(self):
        # 200 at the cheapest codecs, 100 spare: only alpha's q8 step fits.
        precision = select_precision(LADDER, 300)
        self.assertEqual(dict(precision.codecs), {"alpha": "q8", "beta": "q4"})
        self.assertEqual(precision.provenance["codec_counts"],
                         {"q3": 0, "q4": 1, "q8": 1, "f32": 0})
        self.assertEqual([item["codec"] for item in precision.provenance["upgrades"]], ["q8"])
        # Room for alpha's q8 step and beta's jump straight to dense.
        precision = select_precision(LADDER, 600)
        self.assertEqual(dict(precision.codecs), {"alpha": "q8", "beta": "f32"})
        self.assertEqual(precision.provenance["unused_budget_bytes"], 0)
        # Everything dense once the budget covers it.
        precision = select_precision(LADDER, 1 << 20)
        self.assertEqual(set(precision.codecs.values()), {"f32"})

    def test_a_tensor_may_climb_more_than_one_step(self):
        # 200 baseline + 100 (alpha q8) + 300 (beta dense) + 600 (alpha dense).
        precision = select_precision(LADDER, 1200)
        self.assertEqual(precision.codecs["alpha"], "f32")
        self.assertEqual([item["tensor"] for item in precision.provenance["upgrades"]].count("alpha"), 2)
        self.assertEqual(precision.provenance["unused_budget_bytes"], 0)

    def test_a_step_that_buys_no_accuracy_is_never_taken(self):
        # q8 costs more and is not better: the frontier drops it entirely.
        report = calibration([("solo", 800, {"q4": (100, 0.20), "q8": (400, 0.25)})],
                             measured=("q4", "q8"))
        precision = select_precision(report, 500)
        self.assertEqual(precision.codecs["solo"], "q4")
        self.assertEqual(precision.provenance["upgrades"], [])
        precision = select_precision(report, 900)
        self.assertEqual(precision.codecs["solo"], "f32")

    def test_a_tight_budget_keeps_everything_packed(self):
        precision = select_precision(REPORT, 300)
        self.assertEqual(set(precision.codecs.values()), {"q4"})
        self.assertEqual(precision.dense_tensors, ())
        self.assertEqual(precision.provenance["planned_bytes"], 300)
        self.assertEqual(precision.provenance["unused_budget_bytes"], 0)
        self.assertEqual(precision.provenance["estimated_avoided_rmse_sum"], 0.0)
        self.assertEqual(precision.provenance["cheapest_baseline_bytes"], 300)

    def test_a_large_budget_keeps_everything_dense(self):
        precision = select_precision(REPORT, 1 << 20)
        self.assertEqual(set(precision.codecs.values()), {"f32"})
        self.assertEqual(precision.provenance["planned_bytes"], 800 + 400 + 900)
        self.assertAlmostEqual(precision.provenance["estimated_avoided_rmse_sum"], 0.70)

    def test_a_partial_budget_buys_the_best_ratio_first(self):
        # 300 packed + 300 spare: only beta's promotion fits.
        precision = select_precision(REPORT, 600)
        self.assertEqual(precision.dense_tensors, ("beta",))
        self.assertEqual(precision.provenance["unused_budget_bytes"], 0)
        # 300 packed + 300 for beta + 700 for alpha; gamma's 800 never fit.
        precision = select_precision(REPORT, 1300)
        self.assertEqual(precision.dense_tensors, ("alpha", "beta"))
        self.assertEqual(precision.provenance["unused_budget_bytes"], 0)
        self.assertEqual([item["tensor"] for item in precision.provenance["upgrades"]],
                         ["beta", "alpha"])

    def test_selection_is_deterministic_including_ties(self):
        tied = single_codec([("a", 500, 100, 0.25), ("b", 500, 100, 0.25), ("c", 500, 100, 0.25)])
        # 300 packed plus room for two promotions of 400 bytes each.
        first = select_precision(tied, 1100)
        second = select_precision(tied, 1100)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.dense_tensors, ("a", "b"))  # Name breaks the tie.

    def test_a_tensor_the_codec_does_not_shrink_is_always_dense(self):
        report = single_codec([("wide", 100, 120, 0.05), ("narrow", 400, 100, 0.30)])
        precision = select_precision(report, 220)
        self.assertEqual(precision.codecs["wide"], "f32")
        self.assertEqual(precision.codecs["narrow"], "q4")

    def test_impossible_budgets_and_unmeasured_reports_are_rejected(self):
        with self.assertRaises(ValueError) as failure:
            select_precision(REPORT, 299)
        self.assertIn("below", str(failure.exception))
        for budget in (-1, 1.5, True, None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                select_precision(REPORT, budget)
        with self.assertRaises(ValueError):
            select_precision(single_codec([("a", 400, 100, 0.1)], sensitivity_measured=False), 1000)
        with self.assertRaises(ValueError):
            select_precision({"tensors": []}, 1000)
        broken = single_codec([("a", 400, 100, 0.1)])
        broken["tensors"][0]["sensitivity"] = {"rmse": float("nan")}
        with self.assertRaises(ValueError):
            select_precision(broken, 1000)
        missing = single_codec([("a", 400, 100, 0.1)])
        del missing["tensors"][0]["packed_bytes"]
        with self.assertRaises(ValueError):
            select_precision(missing, 1000)

    def test_reports_state_that_the_estimate_is_not_a_quality_claim(self):
        precision = select_precision(REPORT, 1300)
        self.assertFalse(precision.provenance["quality_measured"])
        self.assertIn("errors do not add", precision.provenance["estimate_scope"])
        self.assertEqual(precision.provenance["policy"], POLICY_ID)
        self.assertEqual(precision.provenance["calibration_tokens"], [1, 3])


class PrecisionMapContractRegressions(unittest.TestCase):
    def test_round_trip_through_json_preserves_the_map(self):
        precision = select_precision(REPORT, 1300)
        restored = PrecisionMap.from_json(precision.to_json())
        self.assertEqual(restored.to_dict(), precision.to_dict())
        self.assertEqual(restored.dense_tensors, precision.dense_tensors)

    def test_malformed_maps_are_rejected(self):
        valid = select_precision(REPORT, 1300).to_dict()
        for mutate in (lambda data: data.pop("codecs"),
                       lambda data: data.__setitem__("schema_version", 2),
                       lambda data: data.__setitem__("policy_id", "other"),
                       lambda data: data.__setitem__("codecs", []),
                       lambda data: data.__setitem__("extra", 1)):
            data = json.loads(json.dumps(valid))
            mutate(data)
            with self.subTest(data=sorted(data)), self.assertRaises(ValueError):
                PrecisionMap.from_dict(data)
        for codecs in ({}, {"a": "q2"}, {"": "q4"}, {"a": None}):
            with self.subTest(codecs=codecs), self.assertRaises(ValueError):
                PrecisionMap(codecs)
        with self.assertRaises(ValueError):
            PrecisionMap({"a": "q4"}, provenance=["not", "an", "object"])

    def test_bytes_for_uses_the_measured_sizes(self):
        precision = PrecisionMap({"a": "f32", "b": "q4"})
        sizes = {"a": {"q4": 100, "f32": 800}, "b": {"q4": 50, "f32": 400}}
        self.assertEqual(precision.bytes_for(sizes), 850)
        with self.assertRaises(ValueError):
            precision.bytes_for({"a": {"q4": 100, "f32": 800}})


class PrecisionMapCLIRegressions(_DenseFixture):
    def setUp(self):
        super().setUp()
        from model_checkpoint_fixture import create_checkpoint
        self.checkpoint = self.directory / "checkpoint"
        create_checkpoint(self.checkpoint)

    def run_tool(self, tool, *arguments, expect=0):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / tool), *arguments],
                                capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, expect, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout) if expect == 0 else result.stderr

    def test_calibrate_plan_convert_and_run(self):
        report = self.directory / "calibration.json"
        self.run_tool("nexa_calibrate.py", "--checkpoint", str(self.checkpoint), "--tokens", "1,3",
                      "--group-size", "4", "--block-rows", "3", "--tile-rows", "3",
                      "--memory-budget", "8MiB", "--report", str(report))
        plan = self.directory / "precision.json"
        planned = self.run_tool("nexa_precision.py", "plan", "--calibration", str(report),
                                "--budget", "2KiB", "--out", str(plan))
        self.assertEqual(planned["policy_id"], POLICY_ID)
        self.assertLessEqual(planned["provenance"]["planned_bytes"], 2048)
        dense = {name for name, codec in planned["codecs"].items() if codec == "f32"}
        shown = self.run_tool("nexa_precision.py", "show", str(plan))
        self.assertEqual(set(shown["dense_tensors"]), dense)
        bundle = self.directory / "planned-bundle"
        converted = self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint),
                                  "--out", str(bundle), "--precision-map", str(plan),
                                  "--group-size", "4", "--block-rows", "3")
        self.assertEqual(set(converted["dense_tensors"]), dense)
        inspected = self.run_tool("nexa_inspect.py", str(bundle), "--verify")
        codecs = {item["name"]: item["codec"] for item in inspected["tensors"]}
        stored = {"f32": "RAW_F32_MATRIX", "q4": "Q4_GROUPED", "q8": "Q8_GROUPED"}
        for name, codec in planned["codecs"].items():
            self.assertEqual(codecs[name], stored[codec])
        self.assertTrue(set(planned["codecs"].values()) <= set(stored))
        executed = self.run_tool("nexa_run.py", str(bundle), "--tokens", "1,3",
                                 "--tile-rows", "3", "--memory-budget", "8MiB")
        self.assertEqual(executed["token_ids"], [1, 3])

    def test_cli_rejects_conflicting_options_and_broken_maps(self):
        plan = self.directory / "broken.json"
        plan.write_text(json.dumps({"schema_version": 9, "policy_id": "x", "codecs": {}, "provenance": {}}))
        message = self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint), "--out",
                                str(self.directory / "never"), "--precision-map", str(plan), expect=1)
        self.assertTrue(message.strip())
        self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint), "--out",
                      str(self.directory / "never-two"), "--precision-map", str(plan),
                      "--dense-all", expect=2)
        static = self.directory / "static.json"
        self.run_tool("nexa_calibrate.py", "--checkpoint", str(self.checkpoint), "--tokens", "1",
                      "--group-size", "4", "--static-only", "--report", str(static))
        message = self.run_tool("nexa_precision.py", "plan", "--calibration", str(static),
                                "--budget", "2KiB", expect=1)
        self.assertIn("sensitivity", message)


if __name__ == "__main__":
    unittest.main()

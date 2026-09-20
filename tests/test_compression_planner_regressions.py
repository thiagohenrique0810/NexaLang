"""CompressionPlanner: the codec decision unchanged, the axes it refuses named, decode time measured.

Three things are under test and they are deliberately different in kind.

The first is an identity: putting a planner in front of `select_precision` must
not change a single byte of what it decides. That is checked against plans
frozen before this module existed, in `tests/fixtures/compression_planner_golden.json`.

The second is a refusal: sparsity and low rank do not exist in this repository,
and the planner has to say so by name instead of accepting the argument and
planning something else. The reason it gives is itself checked against the
runtime, so the claim cannot rot.

The third is the only one that is a measurement. Decode time per stored byte is
timed here, on this host, on the kernels the executor runs, and the planner's
prediction is compared against a timing loop written in this file rather than
against the module that produced the prediction.

The tensor sizes in the report below are not invented: 128 rows of 512 values
at group size 32 is exactly 24576 bytes under Q2, 32768 under Q3, 40960 under
Q4, 73728 under Q8, 131072 under F16 and 262144 dense, which is what the pack
format writes and what the timing test encodes.
"""
import ctypes
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.codec_speed import (
    DECODE_RATE_POLICY_ID, SPEED_CODECS, decode_ns, encode_row, measure_decode_rates, rate_table,
)
from compiler.planner.compression import CompressionPlanner, UNSUPPORTED_AXES
from compiler.precision_map import POLICY_IDS, select_precision
from runtime.nexapack.bundle import PACKED_CODECS
from runtime.nexapack.transformer import _PACKED_KERNELS, _load_kernels

GOLDEN = json.loads((ROOT / "tests" / "fixtures" / "compression_planner_golden.json")
                    .read_text(encoding="utf-8"))
ROWS, COLS, GROUP = 128, 512, 32
VALUES = ROWS * COLS
STORED_BYTES = {"q2": 24576, "q3": 32768, "q4": 40960, "q8": 73728, "f16": 131072, "f32": 262144}
# The container writes one fixed header and block table per packed tensor.
OVERHEAD = 4096
SENSITIVITY = {
    "alpha": {"q2": 0.90, "q3": 0.55, "q4": 0.30, "q8": 0.08, "f16": 0.02},
    "beta": {"q2": 0.40, "q3": 0.24, "q4": 0.12, "q8": 0.03, "f16": 0.01},
    "gamma": {"q2": 0.12, "q3": 0.07, "q4": 0.04, "q8": 0.01, "f16": 0.002},
}
PAYLOAD_BUDGET = 720896
PHYSICAL_CEILING = 0.5


def ladder_report(**overrides):
    tensors = []
    for name, errors in SENSITIVITY.items():
        codecs = {codec: {"packed_bytes": STORED_BYTES[codec],
                          "physical_bytes": STORED_BYTES[codec] + OVERHEAD,
                          "container_overhead_bytes": OVERHEAD,
                          "sensitivity": {"rmse": rmse}}
                  for codec, rmse in errors.items()}
        tensors.append({"name": name, "dense_bytes": STORED_BYTES["f32"],
                        "dense_physical_bytes": STORED_BYTES["f32"], "codecs": codecs})
    report = {"checkpoint": "/models/tiny-llama", "tokens": [1, 3, 5], "group_size": GROUP,
              "measured_codecs": ["q2", "q3", "q4", "q8", "f16"], "sensitivity_measured": True,
              "quality_measured": False, "tensors": tensors}
    report.update(overrides)
    return report


LADDER_REPORT = ladder_report()


class _MeasuredDecodeRates(unittest.TestCase):
    """Times the ladder once for the whole class; every rate below is from this host."""
    @classmethod
    def setUpClass(cls):
        cls.block = measure_decode_rates(trials=11, min_trial_ns=5_000_000)
        cls.rates = rate_table({"decode_time": cls.block})
        cls.timed_report = ladder_report(decode_time=cls.block)

    def plan_decode_ns(self, codecs):
        """What the measured rates say this codec assignment costs to decode once."""
        return sum(decode_ns(self.rates, codec, STORED_BYTES[codec]) for codec in codecs.values())


class CodecOnlyIdentityRegressions(unittest.TestCase):
    def test_the_codec_only_plan_is_byte_identical_to_the_plan_frozen_before_the_planner(self):
        payload = CompressionPlanner(LADDER_REPORT, max_bytes=PAYLOAD_BUDGET, cost="payload").plan()
        physical = CompressionPlanner(LADDER_REPORT, max_rmse=PHYSICAL_CEILING,
                                      cost="physical").plan()
        self.assertEqual(json.loads(payload.to_json()), GOLDEN["payload"])
        self.assertEqual(json.loads(physical.to_json()), GOLDEN["physical"])
        # Byte for byte, not merely equal as objects: key order and float
        # formatting are part of what a converter reads back.
        self.assertEqual(payload.to_json(),
                         json.dumps(GOLDEN["payload"], indent=2, sort_keys=True, allow_nan=False))
        self.assertEqual(physical.to_json(),
                         json.dumps(GOLDEN["physical"], indent=2, sort_keys=True, allow_nan=False))

    def test_the_planner_and_select_precision_agree_on_both_cost_bases(self):
        for cost, kwargs in (("payload", {"max_bytes": PAYLOAD_BUDGET}),
                             ("payload", {"max_rmse": 0.2}),
                             ("physical", {"max_bytes": 900000}),
                             ("physical", {"max_rmse": PHYSICAL_CEILING})):
            with self.subTest(cost=cost, **kwargs):
                planned = CompressionPlanner(LADDER_REPORT, cost=cost, **kwargs).plan()
                direct = select_precision(LADDER_REPORT, kwargs.get("max_bytes"),
                                          max_rmse=kwargs.get("max_rmse"), cost=cost)
                self.assertEqual(planned.to_json(), direct.to_json())

    def test_the_two_policy_ids_did_not_change(self):
        self.assertEqual(POLICY_IDS, {"payload": "GREEDY_SENSITIVITY_PER_BYTE_V2",
                                      "physical": "GREEDY_SENSITIVITY_PER_PHYSICAL_BYTE_V3"})
        self.assertEqual(GOLDEN["payload"]["policy_id"], POLICY_IDS["payload"])
        self.assertEqual(GOLDEN["physical"]["policy_id"], POLICY_IDS["physical"])

    def test_a_codec_only_plan_carries_no_decode_time_provenance(self):
        # Silence is the honest record when no time bound was declared.
        plan = CompressionPlanner(LADDER_REPORT, max_bytes=PAYLOAD_BUDGET).plan()
        self.assertNotIn("decode_time", plan.provenance)
        self.assertIn("not bounded", plan.axes["considered"]["decode_time"])


class RefusedAxisRegressions(unittest.TestCase):
    def test_every_unsupported_axis_is_refused_by_name(self):
        for axis, value in (("sparsity", 0.5), ("low_rank", 16), ("peak_vram", 1 << 30),
                            ("transfer_bytes", 1 << 20), ("energy", 12.0)):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError) as caught:
                    CompressionPlanner(LADDER_REPORT, max_bytes=PAYLOAD_BUDGET, **{axis: value})
                self.assertIn(axis, str(caught.exception))
                self.assertIn(UNSUPPORTED_AXES[axis], str(caught.exception))

    def test_an_axis_passed_as_none_is_still_refused(self):
        # Naming the axis is the claim that it was planned; None does not undo it.
        with self.assertRaises(ValueError) as caught:
            CompressionPlanner(LADDER_REPORT, max_bytes=PAYLOAD_BUDGET, sparsity=None)
        self.assertIn("sparsity", str(caught.exception))

    def test_an_axis_nobody_ever_defined_is_refused_as_unknown(self):
        with self.assertRaises(ValueError) as caught:
            CompressionPlanner(LADDER_REPORT, max_bytes=PAYLOAD_BUDGET, huffman=True)
        self.assertIn("huffman", str(caught.exception))
        self.assertIn("unknown axis", str(caught.exception))

    def test_the_refusal_reasons_still_describe_the_runtime(self):
        """The provenance claims there is no sparse codec and no sparse kernel. Check it."""
        for codec in PACKED_CODECS:
            self.assertNotIn("sparse", codec.lower())
        for name, stored in PACKED_CODECS.items():
            self.assertNotIn("sparse", stored.lower())
            self.assertIn(stored, _PACKED_KERNELS)
        for kernels in _PACKED_KERNELS.values():
            for kernel in kernels:
                self.assertNotIn("sparse", kernel)
                self.assertNotIn("low_rank", kernel)
        self.assertIn("PACKED_CODECS", UNSUPPORTED_AXES["sparsity"])
        self.assertIn("M6.06", UNSUPPORTED_AXES["sparsity"])
        self.assertIn("M6.07", UNSUPPORTED_AXES["low_rank"])

    def test_the_axis_provenance_names_everything_it_left_out(self):
        plan = CompressionPlanner(LADDER_REPORT, max_bytes=PAYLOAD_BUDGET).plan()
        self.assertEqual(set(plan.axes["not_considered"]), set(UNSUPPORTED_AXES))
        self.assertEqual(set(plan.axes["considered"]), {"codec", "decode_time"})
        self.assertEqual(plan.axes["decode_rate_policy_id"], DECODE_RATE_POLICY_ID)
        self.assertFalse(plan.axes["decode_rate_available"])
        report = json.loads(plan.to_report_json())
        self.assertEqual(report["precision_map"], json.loads(plan.to_json()))
        self.assertEqual(report["constraints"]["max_decode_ns"], None)

    def test_the_two_bounds_of_the_codec_axis_stay_mutually_exclusive(self):
        for kwargs in ({}, {"max_bytes": 1, "max_rmse": 1.0}):
            with self.assertRaises(ValueError):
                CompressionPlanner(LADDER_REPORT, **kwargs)


class DecodeRateMeasurementRegressions(_MeasuredDecodeRates):
    def test_every_codec_of_the_ladder_is_timed_with_its_spread(self):
        self.assertEqual(set(self.block["codecs"]), set(SPEED_CODECS))
        for codec, entry in sorted(self.block["codecs"].items()):
            with self.subTest(codec=codec):
                self.assertGreater(entry["ns_per_byte"], 0.0)
                self.assertGreaterEqual(entry["repeats"], 1)
                self.assertLessEqual(entry["ns_per_block_min"], entry["ns_per_block_median"])
                self.assertLessEqual(entry["ns_per_block_median"], entry["ns_per_block_max"])
                # Loose on purpose: on a busy machine one trial in eleven can
                # land several times above the median. That is exactly why the
                # median is the published figure and the spread is published
                # beside it instead of being asserted tightly.
                self.assertLess(entry["relative_spread"], 4.0)
        # A rate that travelled from another machine has to be recognizable.
        self.assertEqual(self.block["host"]["python"], platform.python_version())
        self.assertTrue(self.block["host"]["platform"])

    def test_a_smaller_codec_is_not_automatically_a_faster_one(self):
        """The claim the whole increment rests on, asserted against the kernels.

        Ordering the ladder by stored bytes and by measured decode time must
        not produce the same ordering. If a host ever makes bit unpacking as
        cheap as copying floats, this fails, and the guide has to be rewritten
        rather than quietly left wrong.
        """
        by_bytes = sorted(SPEED_CODECS, key=lambda codec: STORED_BYTES[codec])
        by_time = sorted(SPEED_CODECS,
                         key=lambda codec: self.block["codecs"][codec]["ns_per_value"])
        inversions = [(small, large) for small in SPEED_CODECS for large in SPEED_CODECS
                      if STORED_BYTES[small] < STORED_BYTES[large]
                      and (self.block["codecs"][small]["ns_per_value"] >
                           self.block["codecs"][large]["ns_per_value"])]
        print(f"\n  ladder by stored bytes: {' < '.join(by_bytes)}")
        print("  ladder by decode time : " + " < ".join(
            f"{codec} ({self.block['codecs'][codec]['ns_per_value']:.2f} ns/value)"
            for codec in by_time))
        print(f"  smaller-but-slower pairs: {inversions}")
        self.assertNotEqual(by_bytes, by_time)
        self.assertTrue(inversions)

    def test_the_rate_table_refuses_a_report_that_never_measured_time(self):
        for report in ({}, {"decode_time": {"policy_id": "OTHER", "codecs": {}}},
                       {"decode_time": {"policy_id": DECODE_RATE_POLICY_ID, "codecs": {}}},
                       {"decode_time": {"policy_id": DECODE_RATE_POLICY_ID,
                                        "codecs": {"q4": {"ns_per_byte": 0.0}}}}):
            with self.subTest(report=report):
                with self.assertRaises(ValueError):
                    rate_table(report)

    def test_a_time_bound_without_a_measurement_is_refused_naming_the_block(self):
        with self.assertRaises(ValueError) as caught:
            CompressionPlanner(LADDER_REPORT, max_bytes=PAYLOAD_BUDGET,
                               max_decode_ns=1_000_000).plan()
        self.assertIn("decode_time", str(caught.exception))


class TimeBoundedPlanRegressions(_MeasuredDecodeRates):
    def test_the_time_ceiling_changes_the_plan_the_byte_budget_produced(self):
        """The disagreement oracle: both outcomes are results, and both are asserted."""
        by_bytes = CompressionPlanner(self.timed_report, max_rmse=10.0, cost="payload").plan()
        baseline_ns = self.plan_decode_ns(by_bytes.codecs)
        ceiling = int(baseline_ns * 0.9)
        print(f"\n  plan bounded by bytes: {dict(by_bytes.codecs)} "
              f"-> {by_bytes.provenance['planned_bytes']} bytes, {baseline_ns} ns to decode")
        try:
            by_time = CompressionPlanner(self.timed_report, max_rmse=10.0, cost="payload",
                                         max_decode_ns=ceiling).plan()
        except ValueError as error:
            # Recorded as a result, not swept up: on a host where the smallest
            # codec is also the fastest, no faster plan exists to find.
            print(f"  no plan under {ceiling} ns on this host: {error}")
            self.assertIn(str(ceiling), str(error))
            self.assertIn(str(baseline_ns), str(error))
            return
        timed_ns = by_time.provenance["decode_time"]["planned_decode_ns"]
        print(f"  plan bounded by time : {dict(by_time.codecs)} "
              f"-> {by_time.provenance['planned_bytes']} bytes, {timed_ns} ns to decode "
              f"(ceiling {ceiling})")
        self.assertNotEqual(dict(by_time.codecs), dict(by_bytes.codecs))
        self.assertLessEqual(timed_ns, ceiling)
        self.assertLess(timed_ns, baseline_ns)
        # Buying time costs bytes; a plan that got faster for free would mean
        # the byte-bounded plan was never the cheapest one.
        self.assertGreater(by_time.provenance["planned_bytes"],
                           by_bytes.provenance["planned_bytes"])

    def test_the_time_provenance_records_the_rate_it_used_and_the_headroom_left(self):
        by_bytes = CompressionPlanner(self.timed_report, max_rmse=10.0).plan()
        ceiling = int(self.plan_decode_ns(by_bytes.codecs) * 0.9)
        plan = CompressionPlanner(self.timed_report, max_rmse=10.0, max_decode_ns=ceiling).plan()
        recorded = plan.provenance["decode_time"]
        self.assertEqual(recorded["policy"], DECODE_RATE_POLICY_ID)
        self.assertEqual(recorded["max_decode_ns"], ceiling)
        self.assertTrue(recorded["meets_max_decode_ns"])
        self.assertEqual(recorded["unused_decode_ns"], ceiling - recorded["planned_decode_ns"])
        self.assertEqual(set(recorded["ns_per_byte"]), set(self.rates))
        self.assertTrue(recorded["feasibility_upgrades"])
        self.assertEqual(plan.axes["considered"]["decode_time"],
                         "bounded by measured decode nanoseconds per stored byte")

    def test_a_quality_ceiling_and_a_time_ceiling_are_both_honoured(self):
        """Where the time bound earns its keep: F16 is small, accurate and slow."""
        quality = CompressionPlanner(self.timed_report, max_rmse=0.05).plan()
        total = self.plan_decode_ns(quality.codecs)
        ceiling = int(total * 0.5)
        bounded = CompressionPlanner(self.timed_report, max_rmse=0.05,
                                     max_decode_ns=ceiling).plan()
        recorded = bounded.provenance["decode_time"]
        print(f"\n  rmse only : {dict(quality.codecs)} -> "
              f"{quality.provenance['planned_bytes']} bytes, {total} ns")
        print(f"  rmse+time : {dict(bounded.codecs)} -> "
              f"{bounded.provenance['planned_bytes']} bytes, "
              f"{recorded['planned_decode_ns']} ns (ceiling {ceiling})")
        self.assertLessEqual(recorded["planned_decode_ns"], ceiling)
        self.assertTrue(recorded["meets_max_decode_ns"])
        self.assertNotEqual(dict(bounded.codecs), dict(quality.codecs))
        # The quality ceiling is still met: buying speed here also bought
        # accuracy, because the slow codec was not the accurate one.
        self.assertTrue(bounded.provenance["meets_max_rmse"])
        self.assertGreater(bounded.provenance["planned_bytes"],
                           quality.provenance["planned_bytes"])

    def test_an_unreachable_ceiling_is_refused_with_both_numbers(self):
        with self.assertRaises(ValueError) as caught:
            CompressionPlanner(self.timed_report, max_rmse=10.0, max_decode_ns=1).plan()
        self.assertIn("1 ns", str(caught.exception))

    def test_a_ceiling_the_cheapest_plan_already_meets_changes_nothing(self):
        loose = CompressionPlanner(self.timed_report, max_rmse=10.0).plan()
        generous = int(self.plan_decode_ns(loose.codecs) * 4)
        bounded = CompressionPlanner(self.timed_report, max_rmse=10.0,
                                     max_decode_ns=generous).plan()
        self.assertEqual(dict(bounded.codecs), dict(loose.codecs))
        self.assertEqual(bounded.provenance["decode_time"]["feasibility_upgrades"], [])

    def test_a_byte_budget_too_small_to_buy_the_time_ceiling_is_refused(self):
        cheapest = sum(STORED_BYTES["q2"] for _ in SENSITIVITY)
        with self.assertRaises(ValueError):
            # Exactly the cheapest plan fits, so no upgrade can be bought, and
            # the cheapest plan is not the fastest one.
            CompressionPlanner(self.timed_report, max_bytes=cheapest, max_decode_ns=1).plan()


def handmade_rates(**ns_per_byte):
    """A decode_time block with rates chosen by hand.

    The measurement classes cover what the clock says. This one covers what the
    planner does with a rate, and that has to be decided by the numbers in the
    fixture rather than by how fast this machine happens to be today: two of
    the branches below only appear when a specific codec is the slow one.
    """
    return {"policy_id": DECODE_RATE_POLICY_ID, "unit": "nanoseconds per stored byte",
            "codecs": {codec: {"ns_per_byte": rate} for codec, rate in ns_per_byte.items()}}


def one_tensor_report(sensitivity, rates):
    codecs = {codec: {"packed_bytes": STORED_BYTES[codec],
                      "physical_bytes": STORED_BYTES[codec] + OVERHEAD,
                      "sensitivity": {"rmse": rmse}}
              for codec, rmse in sensitivity.items()}
    return {"checkpoint": "/models/one", "tokens": [1], "group_size": GROUP,
            "measured_codecs": sorted(sensitivity), "sensitivity_measured": True,
            "decode_time": rates,
            "tensors": [{"name": "alpha", "dense_bytes": STORED_BYTES["f32"],
                         "dense_physical_bytes": STORED_BYTES["f32"], "codecs": codecs}]}


class TimeBoundDecisionRegressions(unittest.TestCase):
    """The branches of the time bound, over rates chosen so the branch is reached."""

    def test_a_codec_dominated_on_error_survives_when_it_is_the_only_fast_one(self):
        # Q3 is both smaller and more accurate than Q4 here, so the old
        # one-dimensional frontier dropped Q4 outright. Q4 is also the only
        # option under the budget that meets the time ceiling.
        report = one_tensor_report({"q2": 0.90, "q3": 0.20, "q4": 0.30, "q8": 0.08},
                                   handmade_rates(q2=8.0, q3=6.0, q4=1.0, q8=0.5, f32=0.1))
        plan = CompressionPlanner(report, max_bytes=45000, max_decode_ns=50_000).plan()
        self.assertEqual(dict(plan.codecs), {"alpha": "q4"})
        self.assertEqual(plan.provenance["decode_time"]["planned_decode_ns"], 40960)
        # Without the time bound the same report drops Q4 as dominated.
        blind = CompressionPlanner(report, max_bytes=45000).plan()
        self.assertEqual(dict(blind.codecs), {"alpha": "q3"})

    def test_the_time_ceiling_can_put_the_quality_ceiling_out_of_reach_and_says_so(self):
        # Only F16 and F32 are accurate enough, and both are slower than the
        # ceiling allows. The plan stops short of the quality ceiling instead
        # of breaking the time one, and both facts are in the provenance.
        report = one_tensor_report({"q2": 0.90, "q3": 0.55, "q4": 0.30, "q8": 0.25, "f16": 0.02},
                                   handmade_rates(q2=8.0, q3=6.0, q4=1.0, q8=0.5, f16=4.0, f32=0.3))
        plan = CompressionPlanner(report, max_rmse=0.10, max_decode_ns=60_000).plan()
        self.assertEqual(dict(plan.codecs), {"alpha": "q8"})
        self.assertFalse(plan.provenance["meets_max_rmse"])
        self.assertTrue(plan.provenance["decode_time"]["meets_max_decode_ns"])
        self.assertLessEqual(plan.provenance["decode_time"]["planned_decode_ns"], 60_000)
        # Lift the time ceiling and the accurate, slow codec is taken again.
        self.assertEqual(dict(CompressionPlanner(report, max_rmse=0.10).plan().codecs),
                         {"alpha": "f16"})

    def test_the_feasibility_pass_buys_time_with_the_fewest_extra_bytes(self):
        report = one_tensor_report({"q2": 0.90, "q3": 0.55, "q4": 0.30, "q8": 0.08},
                                   handmade_rates(q2=8.0, q3=6.0, q4=1.0, q8=0.5, f32=0.1))
        plan = CompressionPlanner(report, max_rmse=10.0, max_decode_ns=50_000).plan()
        steps = plan.provenance["decode_time"]["feasibility_upgrades"]
        # Q8 saves marginally more time but costs three times the bytes.
        self.assertEqual([step["codec"] for step in steps], ["q4"])
        self.assertEqual(steps[0]["extra_bytes"], STORED_BYTES["q4"] - STORED_BYTES["q2"])
        self.assertEqual(steps[0]["saved_decode_ns"], 196608 - 40960)

    def test_a_bad_rate_block_is_refused_before_any_planning(self):
        report = one_tensor_report({"q2": 0.90, "q4": 0.30},
                                   handmade_rates(q2=8.0, q4=1.0, f32=0.1))
        report["decode_time"]["codecs"]["q4"]["ns_per_byte"] = -1.0
        with self.assertRaises(ValueError) as caught:
            CompressionPlanner(report, max_rmse=10.0, max_decode_ns=50_000).plan()
        self.assertIn("q4", str(caught.exception))


class PredictionAgainstAnIndependentMeasurementRegressions(_MeasuredDecodeRates):
    """The test that a forwarding facade cannot pass.

    The planner predicts how long its plan takes to decode. This class encodes
    the very matrices the plan describes, runs the executor's kernels over them
    with a timing loop written here, and compares. A planner that only renamed
    `select_precision` would have no prediction to compare; a planner whose
    prediction agreed only with its own arithmetic would drift away from the
    clock as soon as a rate were attached to the wrong codec.
    """
    def measure_plan(self, codecs):
        lib = _load_kernels()
        generator = random.Random(4242)
        block = [[generator.uniform(-1.0, 1.0) for _ in range(COLS)] for _ in range(ROWS)]
        left = (ctypes.c_float * COLS)(*[generator.uniform(-1.0, 1.0) for _ in range(COLS)])
        out = (ctypes.c_float * ROWS)()
        observed = {}
        for codec in sorted(set(codecs.values())):
            payload = b"".join(encode_row(codec, row, GROUP) for row in block)
            self.assertEqual(len(payload), STORED_BYTES[codec])
            buffer = ((ctypes.c_float * VALUES) if codec == "f32"
                      else (ctypes.c_uint8 * len(payload))).from_buffer_copy(payload)
            if codec == "f32":
                def call():
                    return lib.nexa_f32_matmul(left, COLS, 1, buffer, VALUES, ROWS, COLS, out, ROWS)
            elif codec == "f16":
                def call(size=len(payload)):
                    return lib.nexa_f16_matmul(left, COLS, 1, buffer, size, ROWS, COLS, out, ROWS)
            else:
                def call(kernel=getattr(lib, f"nexa_{codec}_matmul"), size=len(payload)):
                    return kernel(left, COLS, 1, buffer, size, ROWS, COLS, GROUP, out, ROWS)
            samples = []
            for _ in range(9):
                start = time.perf_counter_ns()
                for _ in range(16):
                    call()
                samples.append((time.perf_counter_ns() - start) / 16)
            observed[codec] = statistics.median(samples)
        return sum(observed[codec] for codec in codecs.values()), observed

    def test_the_predicted_decode_time_of_a_plan_matches_a_measurement_of_that_plan(self):
        by_bytes = CompressionPlanner(self.timed_report, max_rmse=10.0).plan()
        ceiling = int(self.plan_decode_ns(by_bytes.codecs) * 0.9)
        for label, plan in (("bytes", by_bytes),
                            ("time", CompressionPlanner(self.timed_report, max_rmse=10.0,
                                                        max_decode_ns=ceiling).plan())):
            with self.subTest(plan=label):
                predicted = self.plan_decode_ns(plan.codecs)
                observed, per_codec = self.measure_plan(plan.codecs)
                ratio = observed / predicted
                print(f"\n  {label}-bounded plan {dict(plan.codecs)}: predicted {predicted} ns, "
                      f"clock {observed:.0f} ns, ratio {ratio:.3f}, per codec "
                      + ", ".join(f"{codec} {value:.0f} ns" for codec, value in per_codec.items()))
                # The rate was timed on a 32x1024 block and is applied here to a
                # 128x512 one, so the two numbers are not the same measurement
                # and the absolute band has to be wide.
                self.assertGreater(ratio, 0.5)
                self.assertLess(ratio, 2.0)
                # The band between codecs is where the band above is blunt: the
                # constant offset between the two block shapes cancels, so a
                # rate attached to the wrong codec has nowhere to hide.
                for left in sorted(per_codec):
                    for right in sorted(per_codec):
                        if left >= right:
                            continue
                        clocked = per_codec[left] / per_codec[right]
                        expected = (decode_ns(self.rates, left, STORED_BYTES[left]) /
                                    decode_ns(self.rates, right, STORED_BYTES[right]))
                        print(f"    {left}/{right}: predicted {expected:.3f}, "
                              f"clock {clocked:.3f}")
                        self.assertLess(abs(clocked / expected - 1.0), 0.30)

    def test_the_measured_rate_predicts_the_relative_cost_of_every_codec_pair(self):
        """The whole ladder, not only the codecs a plan happened to choose.

        This is the assertion a schema that agrees with itself cannot pass.
        The prediction is rate x bytes; the comparison is a clock on the
        executor's kernels, over a block of a different shape than the one the
        rate was timed on. Attaching any rate to the wrong codec moves a pair
        by several times, while the shape difference moves it by under a fifth.
        """
        observed, per_codec = self.measure_plan({codec: codec for codec in SPEED_CODECS})
        predicted = {codec: decode_ns(self.rates, codec, STORED_BYTES[codec])
                     for codec in SPEED_CODECS}
        print("\n  codec: predicted ns / clocked ns")
        for codec in SPEED_CODECS:
            print(f"    {codec}: {predicted[codec]} / {per_codec[codec]:.0f} "
                  f"({per_codec[codec] / predicted[codec]:.3f})")
        self.assertEqual(observed, sum(per_codec.values()))
        for left in SPEED_CODECS:
            for right in SPEED_CODECS:
                if left >= right:
                    continue
                with self.subTest(pair=(left, right)):
                    clocked = per_codec[left] / per_codec[right]
                    expected = predicted[left] / predicted[right]
                    self.assertLess(abs(clocked / expected - 1.0), 0.30)

    def test_the_planner_prediction_is_the_one_written_into_the_provenance(self):
        by_bytes = CompressionPlanner(self.timed_report, max_rmse=10.0).plan()
        ceiling = int(self.plan_decode_ns(by_bytes.codecs) * 0.9)
        plan = CompressionPlanner(self.timed_report, max_rmse=10.0, max_decode_ns=ceiling).plan()
        self.assertEqual(plan.provenance["decode_time"]["planned_decode_ns"],
                         self.plan_decode_ns(plan.codecs))


class CompressionCLIRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.holder = tempfile.TemporaryDirectory(prefix="nexa-compression-cli-")
        cls.directory = Path(cls.holder.name)
        cls.report = cls.directory / "calibration.json"
        block = measure_decode_rates(trials=5)
        cls.report.write_text(json.dumps(ladder_report(decode_time=block)), encoding="utf-8")
        cls.rates = rate_table({"decode_time": block})

    @classmethod
    def tearDownClass(cls):
        cls.holder.cleanup()

    def run_tool(self, *arguments, expect=0):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "nexa_precision.py"),
                                 *arguments], capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, expect, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout) if expect == 0 else result.stderr

    def test_the_cli_plans_under_a_decode_time_ceiling_and_writes_the_axes(self):
        loose = self.run_tool("plan", "--calibration", str(self.report), "--max-rmse", "10")
        baseline = sum(decode_ns(self.rates, codec, STORED_BYTES[codec])
                       for codec in loose["codecs"].values())
        axes = self.directory / "axes.json"
        bounded = self.run_tool("plan", "--calibration", str(self.report), "--max-rmse", "10",
                                "--max-decode-ns", str(int(baseline * 0.9)),
                                "--axes", str(axes))
        self.assertLessEqual(bounded["provenance"]["decode_time"]["planned_decode_ns"],
                             int(baseline * 0.9))
        recorded = json.loads(axes.read_text())
        self.assertEqual(set(recorded["not_considered"]), set(UNSUPPORTED_AXES))
        self.assertTrue(recorded["decode_rate_available"])
        self.assertNotIn("decode_time", loose["provenance"])

    def test_the_cli_refuses_a_time_ceiling_no_plan_can_meet(self):
        message = self.run_tool("plan", "--calibration", str(self.report), "--max-rmse", "10",
                                "--max-decode-ns", "1", expect=1)
        self.assertIn("ns", message)


if __name__ == "__main__":
    unittest.main()

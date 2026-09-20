"""Per-tensor calibration: statistics, codec error, variants and sensitivity."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.calibration import (
    build_variant, logit_delta, quantization_error, row_statistics,
)
from test_dense_weights_regressions import _DenseFixture, exact_q4_value


class CalibrationMetricRegressions(unittest.TestCase):
    def test_statistics_describe_the_distribution_and_its_outliers(self):
        rows = [[1.0, -1.0, 0.5, -0.5], [0.25, -0.25, 0.75, -0.75], [8.0, 0.0, 0.0, 0.0]]
        report = row_statistics(rows)
        self.assertEqual((report["values"], report["rows"]), (12, 3))
        self.assertEqual((report["min"], report["max"]), (-1.0, 8.0))
        self.assertAlmostEqual(report["mean"], sum(sum(row) for row in rows) / 12)
        self.assertAlmostEqual(report["rms"], math.sqrt(sum(v * v for row in rows for v in row) / 12))
        self.assertEqual(report["max_abs"], 8.0)
        # The median row peaks at 1.0, so the worst row sticks out eight times.
        self.assertEqual(report["median_row_max_abs"], 1.0)
        self.assertEqual(report["outlier_ratio"], 8.0)

    def test_quantization_error_is_zero_for_values_the_codec_stores_exactly(self):
        exact = [[exact_q4_value(index + offset) for index in range(4)] for offset in range(3)]
        for row in exact:
            row[0] = 3.5
        report = quantization_error(exact, 4)
        self.assertEqual(report["codec"], "Q4_GROUPED")
        self.assertEqual(report["max_abs_error"], 0.0)
        self.assertEqual(report["rmse"], 0.0)
        self.assertEqual(report["snr_db"], math.inf)
        self.assertEqual(report["relative_rmse"], 0.0)
        noisy = quantization_error([[1.0, 0.31, -0.17, 5.0]], 4)
        self.assertGreater(noisy["max_abs_error"], 0.0)
        self.assertLess(noisy["snr_db"], math.inf)
        self.assertAlmostEqual(noisy["relative_rmse"], noisy["rmse"] / noisy["reference_rms"])

    def test_metrics_reject_malformed_input(self):
        for rows in ([], [[]], [[float("nan")]], [["text"]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                row_statistics(rows)
        for group_size in (0, -1, 1.0, True):
            with self.assertRaises(ValueError):
                quantization_error([[1.0, 2.0]], group_size)
        for codec in ("q2", "f32", None):
            with self.subTest(codec=codec), self.assertRaises(ValueError):
                quantization_error([[1.0, 2.0]], 2, codec)
        with self.assertRaises(ValueError):
            logit_delta([[1.0, 2.0]], [[1.0]])
        with self.assertRaises(ValueError):
            logit_delta([[1.0]], [[float("inf")]])
        with self.assertRaises(ValueError):
            logit_delta([], [])

    def test_logit_delta_is_zero_for_identical_matrices(self):
        matrix = [[1.0, -2.0, 3.0], [0.5, 0.25, -0.125]]
        report = logit_delta(matrix, [list(row) for row in matrix])
        self.assertEqual((report["max_abs_delta"], report["rmse"], report["relative_rmse"]), (0.0, 0.0, 0.0))
        moved = logit_delta(matrix, [[1.0, -2.0, 3.5], [0.5, 0.25, -0.125]])
        self.assertAlmostEqual(moved["max_abs_delta"], 0.5)
        self.assertGreater(moved["relative_rmse"], 0.0)


class CalibrationVariantRegressions(_DenseFixture):
    def bundles(self, values=None, group_size=4):
        config = self.config()
        values = values if values is not None else self.exact_weights(config, group_size)
        dense = self.write(self.directory / "dense", config, values,
                           codecs=self.dense_codecs(config), group_size=group_size)
        packed = self.write(self.directory / "packed", config, values, group_size=group_size)
        return config, dense, packed

    def test_a_variant_packs_exactly_one_tensor(self):
        _, dense, packed = self.bundles()
        name = "model.layers.0.mlp.down_proj.weight"
        variant = build_variant(dense, packed, name, self.directory / "variant")
        manifest = json.loads((variant / "manifest.json").read_text())
        codecs = {tensor: entry["codec"] for tensor, entry in manifest["tensors"].items()}
        self.assertEqual(codecs[name], "Q4_GROUPED")
        self.assertEqual({codec for tensor, codec in codecs.items() if tensor != name},
                         {"RAW_F32_MATRIX", "RAW_F32"})
        logits, _ = self.run_model(variant)
        # Exactly representable weights: packing one tensor changes nothing.
        self.assertEqual(logits, self.run_model(dense)[0])

    def test_sensitivity_appears_only_where_quantization_is_lossy(self):
        config = self.config()
        values = self.exact_weights(config, 4)
        lossy = "model.layers.1.mlp.down_proj.weight"
        rows, cols = config.required_tensor_shapes()[lossy]
        values[lossy] = [[0.37 * (row + 1) - 0.21 * column for column in range(cols)]
                         for row in range(rows)]
        _, dense, packed = self.bundles(values)
        reference, _ = self.run_model(dense)
        exact_variant = build_variant(dense, packed, "model.layers.0.mlp.up_proj.weight",
                                      self.directory / "exact-variant")
        lossy_variant = build_variant(dense, packed, lossy, self.directory / "lossy-variant")
        exact_delta = logit_delta(reference, self.run_model(exact_variant)[0])
        lossy_delta = logit_delta(reference, self.run_model(lossy_variant)[0])
        self.assertEqual(exact_delta["rmse"], 0.0)
        self.assertGreater(lossy_delta["rmse"], 0.0)
        self.assertGreater(lossy_delta["max_abs_delta"], 0.0)

    def test_variants_reject_mismatched_bundles_and_existing_destinations(self):
        _, dense, packed = self.bundles()
        name = "model.layers.0.mlp.down_proj.weight"
        destination = self.directory / "first"
        build_variant(dense, packed, name, destination)
        with self.assertRaises(FileExistsError):
            build_variant(dense, packed, name, destination)
        with self.assertRaises(ValueError):
            build_variant(dense, packed, "model.missing.weight", self.directory / "unknown")
        with self.assertRaises(ValueError):
            build_variant(packed, packed, name, self.directory / "not-dense")
        with self.assertRaises(ValueError):
            build_variant(dense, dense, name, self.directory / "not-packed")
        other = self.write(self.directory / "other", self.config(num_hidden_layers=1),
                           self.exact_weights(self.config(num_hidden_layers=1), 4),
                           codecs=self.dense_codecs(self.config(num_hidden_layers=1)))
        with self.assertRaises(ValueError):
            build_variant(other, packed, name, self.directory / "other-config")

    def test_a_failed_variant_leaves_no_partial_directory(self):
        _, dense, packed = self.bundles()
        name = "model.layers.0.mlp.down_proj.weight"
        destination = self.directory / "incomplete"
        from unittest.mock import patch
        with patch("compiler.calibration.shutil.copyfile", side_effect=OSError("injected copy failure")), \
             patch("compiler.calibration.os.link", side_effect=OSError("injected link failure")):
            with self.assertRaises(OSError):
                build_variant(dense, packed, name, destination)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_name(destination.name + ".partial").exists())


class CalibrationCLIRegressions(_DenseFixture):
    def setUp(self):
        super().setUp()
        from model_checkpoint_fixture import create_checkpoint
        self.checkpoint = self.directory / "checkpoint"
        create_checkpoint(self.checkpoint)

    def run_cli(self, *arguments, expect=0):
        result = subprocess.run([sys.executable, str(ROOT / "tools/nexa_calibrate.py"), *arguments],
                                capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, expect, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout) if expect == 0 else result.stderr

    def test_static_pass_ranks_tensors_without_executing(self):
        report = self.run_cli("--checkpoint", str(self.checkpoint), "--tokens", "1,3",
                              "--group-size", "4", "--static-only")
        self.assertFalse(report["sensitivity_measured"])
        self.assertFalse(report["quality_measured"])
        self.assertNotIn("reference", report)
        self.assertEqual(report["measured_codecs"], ["q3", "q4", "q8"])
        errors = [max(codec["quantization"]["relative_rmse"] for codec in tensor["codecs"].values())
                  for tensor in report["tensors"]]
        self.assertEqual(errors, sorted(errors, reverse=True))
        for tensor in report["tensors"]:
            self.assertGreater(tensor["dense_bytes"], 0)
            self.assertIn("outlier_ratio", tensor["statistics"])
            self.assertEqual(set(tensor["codecs"]), {"q3", "q4", "q8"})
            for codec, measured in tensor["codecs"].items():
                self.assertNotIn("sensitivity", measured)
            # More levels always mean a smaller round-trip error.
            self.assertLess(tensor["codecs"]["q8"]["quantization"]["rmse"],
                            tensor["codecs"]["q4"]["quantization"]["rmse"])
            self.assertLess(tensor["codecs"]["q4"]["quantization"]["rmse"],
                            tensor["codecs"]["q3"]["quantization"]["rmse"])

    def test_sensitivity_pass_measures_each_tensor_and_codec(self):
        names = ["model.embed_tokens.weight", "model.layers.0.mlp.down_proj.weight"]
        report = self.run_cli("--checkpoint", str(self.checkpoint), "--tokens", "1,3",
                              "--group-size", "4", "--block-rows", "3", "--tile-rows", "3",
                              "--memory-budget", "8MiB", "--tensor", names[0], "--tensor", names[1])
        self.assertTrue(report["sensitivity_measured"])
        self.assertFalse(report["quality_measured"])
        self.assertEqual({tensor["name"] for tensor in report["tensors"]}, set(names))
        worst = [max(codec["sensitivity"]["rmse"] for codec in tensor["codecs"].values())
                 for tensor in report["tensors"]]
        self.assertEqual(worst, sorted(worst, reverse=True))
        self.assertEqual(set(report["all_packed"]), {"q3", "q4", "q8"})
        for codec, summary in report["all_packed"].items():
            self.assertNotEqual(report["reference"]["logits_sha256"], summary["logits_sha256"])
            self.assertGreater(summary["sensitivity"]["rmse"], 0.0)
        # The whole model in Q8 moves the logits less than the whole in Q4.
        self.assertLess(report["all_packed"]["q8"]["sensitivity"]["rmse"],
                        report["all_packed"]["q4"]["sensitivity"]["rmse"])
        for tensor in report["tensors"]:
            for codec, measured in tensor["codecs"].items():
                self.assertGreater(measured["packed_bytes"], 0)
                self.assertEqual(measured["saved_bytes"],
                                 tensor["dense_bytes"] - measured["packed_bytes"])
                self.assertGreater(measured["cost_per_saved_kib"], 0.0)
            self.assertLess(tensor["codecs"]["q8"]["sensitivity"]["rmse"],
                            tensor["codecs"]["q4"]["sensitivity"]["rmse"])

    def test_a_single_codec_run_measures_only_what_it_was_asked_for(self):
        report = self.run_cli("--checkpoint", str(self.checkpoint), "--tokens", "1,3",
                              "--group-size", "4", "--block-rows", "3", "--tile-rows", "3",
                              "--memory-budget", "8MiB", "--codec", "q8",
                              "--tensor", "model.embed_tokens.weight")
        self.assertEqual(report["measured_codecs"], ["q8"])
        self.assertEqual(set(report["all_packed"]), {"q8"})
        self.assertEqual(set(report["tensors"][0]["codecs"]), {"q8"})

    def test_unknown_tensors_and_missing_checkpoints_are_rejected(self):
        for arguments in (("--checkpoint", str(self.checkpoint), "--tokens", "1", "--tensor", "model.norm.weight"),
                          ("--checkpoint", str(self.directory / "missing"), "--tokens", "1"),
                          ("--checkpoint", str(self.checkpoint), "--tokens", "1", "--tensor", "nope")):
            with self.subTest(arguments=arguments[-1]):
                message = self.run_cli(*arguments, "--static-only", expect=1)
                self.assertTrue(message.strip())


if __name__ == "__main__":
    unittest.main()

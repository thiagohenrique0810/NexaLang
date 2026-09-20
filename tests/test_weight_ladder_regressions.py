"""Q2 and F16 weights, and the whole codec ladder they complete."""
import ctypes
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.calibration import quantization_error
from compiler.precision_map import _options, select_precision
from runtime.nexapack.bundle import ModelBundleError, ModelBundleReader
from runtime.nexapack.format import (
    NexaPackError, NexaPackReader, Q2_CODEC_ID, decode_q2_row, quantize_q2_row, write_q2_matrix,
)
from runtime.nexapack.transformer import _load_kernels
from test_dense_weights_regressions import _DenseFixture

ROWS = [[0.5, -0.25, 3.0, 1.0, -0.125, 0.75, 2.5, -1.5],
        [0.03, -0.44, 0.91, -0.02, 0.17, -0.68, 0.33, 0.05]]
LADDER = ("q2", "q3", "q4", "q8", "f16", "f32")


class Q2CodecRegressions(unittest.TestCase):
    def test_two_bits_hold_a_ternary_code(self):
        encoded = quantize_q2_row(ROWS[0], 8)
        self.assertEqual(len(encoded), 4 + 2)
        decoded = decode_q2_row(encoded, 8, 8)
        # Scale is the group maximum, so every value lands on -max, 0 or max.
        self.assertEqual(set(decoded) - {0.0}, {3.0, -3.0})
        self.assertEqual(decoded[2], 3.0)

    def test_round_trip_is_exact_for_ternary_inputs(self):
        row = [2.0, -2.0, 0.0, 2.0]
        self.assertEqual(decode_q2_row(quantize_q2_row(row, 4), 4, 4), row)
        zeros = [0.0, 0.0, 0.0, 0.0]
        self.assertEqual(decode_q2_row(quantize_q2_row(zeros, 4), 4, 4), zeros)

    def test_reserved_code_padding_and_scale_are_validated(self):
        encoded = bytearray(quantize_q2_row([1.0, 2.0, 3.0], 4))
        reserved = bytearray(encoded)
        reserved[4] = (reserved[4] & ~0b11) | 0b10  # code -2
        with self.assertRaises(NexaPackError):
            decode_q2_row(bytes(reserved), 3, 4)
        padding = bytearray(encoded)
        padding[4] |= 0b1100_0000  # the fourth lane of a three-column row
        with self.assertRaises(NexaPackError):
            decode_q2_row(bytes(padding), 3, 4)
        negative = bytearray(encoded)
        negative[3] |= 0x80
        with self.assertRaises(NexaPackError):
            decode_q2_row(bytes(negative), 3, 4)

    def test_the_container_reports_the_codec(self):
        import tempfile
        path = Path(tempfile.mkdtemp()) / "matrix.nxp"
        write_q2_matrix(path, 2, 8, 8, iter(ROWS), block_rows=1)
        with NexaPackReader(path) as reader:
            self.assertEqual(reader.codec_id, Q2_CODEC_ID)
            self.assertEqual(reader.row_bytes, 4 + 2)


class Q2AndF16KernelRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import shutil
        if not (shutil.which("clang") or shutil.which("cc")):
            raise unittest.SkipTest("C compiler unavailable")
        cls.kernels = _load_kernels()

    def test_q2_kernel_matches_the_python_reference(self):
        inputs = [0.25, -0.5, 1.0, 0.125, -0.75, 0.5, 0.0, 2.0]
        for group_size in (4, 8):
            with self.subTest(group_size=group_size):
                packed = b"".join(quantize_q2_row(row, group_size) for row in ROWS)
                source = (ctypes.c_float * 8)(*inputs)
                weights = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
                output = (ctypes.c_float * 2)()
                status = self.kernels.nexa_q2_matmul(source, 8, 1, weights, len(weights),
                                                     2, 8, group_size, output, 2)
                self.assertEqual(status, 0)
                for index, row in enumerate(ROWS):
                    decoded = decode_q2_row(quantize_q2_row(row, group_size), 8, group_size)
                    expected = math.fsum(a * b for a, b in zip(inputs, decoded))
                    self.assertAlmostEqual(output[index], expected, places=5)

    def test_f16_kernel_matches_python_half_precision(self):
        half = struct.Struct("<e")
        weights_f16 = b"".join(half.pack(value) for row in ROWS for value in row)
        inputs = [0.25, -0.5, 1.0, 0.125, -0.75, 0.5, 0.0, 2.0]
        source = (ctypes.c_float * 8)(*inputs)
        weights = (ctypes.c_uint8 * len(weights_f16)).from_buffer_copy(weights_f16)
        output = (ctypes.c_float * 2)()
        status = self.kernels.nexa_f16_matmul(source, 8, 1, weights, len(weights), 2, 8, output, 2)
        self.assertEqual(status, 0)
        for index, row in enumerate(ROWS):
            stored = [half.unpack(half.pack(value))[0] for value in row]
            expected = math.fsum(a * b for a, b in zip(inputs, stored))
            self.assertAlmostEqual(output[index], expected, places=5)

    def test_f16_decode_row_and_subnormals_round_trip(self):
        half = struct.Struct("<e")
        # Smallest normal, a subnormal and zero all survive the conversion.
        values = [6.103515625e-05, 5.960464477539063e-08, 0.0, -1.0]
        encoded = b"".join(half.pack(value) for value in values)
        weights = (ctypes.c_uint8 * len(encoded)).from_buffer_copy(encoded)
        output = (ctypes.c_float * 4)()
        status = self.kernels.nexa_f16_decode_row(weights, len(weights), 4, output, 4)
        self.assertEqual(status, 0)
        self.assertEqual(list(output), values)

    def test_f16_rejects_nonfinite_stored_values(self):
        encoded = struct.pack("<HH", 0x7C00, 0)  # +inf then zero
        weights = (ctypes.c_uint8 * len(encoded)).from_buffer_copy(encoded)
        output = (ctypes.c_float * 2)()
        self.assertNotEqual(self.kernels.nexa_f16_decode_row(weights, len(weights), 2, output, 2), 0)
        source = (ctypes.c_float * 2)(1.0, 1.0)
        self.assertNotEqual(
            self.kernels.nexa_f16_matmul(source, 2, 1, weights, len(weights), 1, 2, output, 1), 0)


class WeightLadderRegressions(_DenseFixture):
    def test_bytes_and_error_are_monotonic_across_the_ladder(self):
        from test_transformer_forward_regressions import random_bundle
        config, values = random_bundle(self.directory / "seed", layers=2, heads=4,
                                       kv_heads=2, tied=False, seed=77)
        matrices = [name for name, shape in config.required_tensor_shapes().items() if len(shape) == 2]
        logits, payload = {}, {}
        for codec in LADDER:
            bundle = self.write(self.directory / codec, config, values,
                                codecs={name: codec for name in matrices},
                                group_size=8, block_rows=3)
            logits[codec], _ = self.run_model(bundle)
            with ModelBundleReader(bundle) as reader:
                payload[codec] = sum(item["packed_payload_bytes"] for item in reader.inspect()["tensors"])
        error = {codec: max(abs(a - b) for left, right in zip(logits[codec], logits["f32"])
                            for a, b in zip(left, right)) for codec in LADDER}
        self.assertEqual(error["f32"], 0.0)
        for cheaper, dearer in zip(LADDER, LADDER[1:]):
            with self.subTest(step=f"{cheaper}->{dearer}"):
                self.assertLess(payload[cheaper], payload[dearer])
                self.assertGreater(error[cheaper], error[dearer])

    def test_one_bundle_holds_every_codec_at_once(self):
        config = self.config(num_hidden_layers=3)
        values = self.exact_weights(config, 4)
        names = sorted(name for name, shape in config.required_tensor_shapes().items()
                       if len(shape) == 2)
        codecs = {name: LADDER[index % len(LADDER)] for index, name in enumerate(names)}
        bundle = self.write(self.directory / "every-codec", config, values, codecs=codecs)
        with ModelBundleReader(bundle) as reader:
            stored = {item["name"]: item["codec"] for item in reader.inspect()["tensors"]}
        expected = {"q2": "Q2_GROUPED", "q3": "Q3_GROUPED", "q4": "Q4_GROUPED",
                    "q8": "Q8_GROUPED", "f16": "RAW_F16_MATRIX", "f32": "RAW_F32_MATRIX"}
        for name, codec in codecs.items():
            self.assertEqual(stored[name], expected[codec])
        logits, report = self.run_model(bundle)
        self.assertTrue(all(math.isfinite(value) for row in logits for value in row))
        self.assertGreater(report["io"]["q4_payload_bytes_read"], 0)
        self.assertGreater(report["io"]["raw_payload_bytes_read"], 0)

    def test_calibration_measures_f16_without_a_group_size(self):
        rows = [[0.5, -0.25, 3.0, 1.0]]
        half = quantization_error(rows, 8, "f16")
        self.assertEqual(half["codec"], "RAW_F16_MATRIX")
        self.assertIsNone(half["group_size"])
        coarse = quantization_error(rows, 4, "q2")
        self.assertEqual(coarse["codec"], "Q2_GROUPED")
        # Half precision is far closer to the original than a ternary code.
        self.assertLess(half["rmse"], coarse["rmse"])
        for codec in ("q16", "f64", None):
            with self.subTest(codec=codec), self.assertRaises(ValueError):
                quantization_error(rows, 4, codec)

    def test_a_plan_can_use_every_rung_of_the_ladder(self):
        entry = {"name": "tensor", "dense_bytes": 1600,
                 "codecs": {"q2": {"packed_bytes": 200, "sensitivity": {"rmse": 1.40}},
                            "q3": {"packed_bytes": 300, "sensitivity": {"rmse": 0.55}},
                            "q4": {"packed_bytes": 400, "sensitivity": {"rmse": 0.18}},
                            "q8": {"packed_bytes": 700, "sensitivity": {"rmse": 0.012}},
                            "f16": {"packed_bytes": 800, "sensitivity": {"rmse": 0.002}}}}
        self.assertEqual([option["codec"] for option in _options("tensor", entry)],
                         ["q2", "q3", "q4", "q8", "f16", "f32"])
        report = {"checkpoint": "/tmp/c", "tokens": [1], "group_size": 8, "sensitivity_measured": True,
                  "measured_codecs": ["q2", "q3", "q4", "q8", "f16"], "tensors": [entry]}
        self.assertEqual(select_precision(report, 200).codecs["tensor"], "q2")
        self.assertEqual(select_precision(report, 450).codecs["tensor"], "q4")
        self.assertEqual(select_precision(report, 1599).codecs["tensor"], "f16")
        self.assertEqual(select_precision(report, 1600).codecs["tensor"], "f32")


class LadderCLIRegressions(_DenseFixture):
    def setUp(self):
        super().setUp()
        from model_checkpoint_fixture import create_checkpoint
        self.checkpoint = self.directory / "checkpoint"
        create_checkpoint(self.checkpoint)

    def run_tool(self, tool, *arguments, expect=0):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / tool), *arguments],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, expect, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout) if expect == 0 else result.stderr

    def test_convert_and_run_the_two_new_codecs(self):
        for codec, stored, bits in (("q2", "Q2_GROUPED", 2), ("f16", "RAW_F16_MATRIX", 16)):
            with self.subTest(codec=codec):
                bundle = self.directory / f"{codec}-bundle"
                self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint),
                              "--out", str(bundle), "--matrix-codec", codec,
                              "--group-size", "8", "--block-rows", "3")
                inspected = self.run_tool("nexa_inspect.py", str(bundle), "--verify")
                widths = {item["codec"]: item["storage_bits"] for item in inspected["tensors"]}
                self.assertEqual(widths[stored], bits)
                report = self.run_tool("nexa_run.py", str(bundle), "--tokens", "1,3",
                                       "--tile-rows", "3", "--memory-budget", "4MiB")
                self.assertEqual(report["token_ids"], [1, 3])

    def test_a_manifest_that_claims_the_wrong_dense_width_is_rejected(self):
        import shutil
        bundle = self.directory / "f16-honest"
        self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint), "--out", str(bundle),
                      "--matrix-codec", "f16", "--group-size", "8", "--block-rows", "3")
        damaged = self.directory / "f16-mislabelled"
        shutil.copytree(bundle, damaged)
        manifest = json.loads((damaged / "manifest.json").read_text())
        name = next(key for key, value in manifest["tensors"].items()
                    if value["codec"] == "RAW_F16_MATRIX")
        manifest["tensors"][name]["codec"] = "RAW_F32_MATRIX"
        (damaged / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ModelBundleError):
            ModelBundleReader(damaged).close()


if __name__ == "__main__":
    unittest.main()

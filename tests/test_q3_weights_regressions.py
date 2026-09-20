"""Q3_GROUPED weights: shared layout with the KV codec, kernel and dominance."""
import ctypes
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import q3_reference
from compiler.precision_map import _options
from runtime.nexapack.bundle import ModelBundleReader
from runtime.nexapack.format import (
    NexaPackError, NexaPackReader, Q3_CODEC_ID, decode_q3_row, decode_q4_row,
    quantize_q3_row, quantize_q4_row, write_q3_matrix,
)
from runtime.nexapack.transformer import _load_kernels
from test_dense_weights_regressions import _DenseFixture

ROWS = [[0.5, -0.25, 3.0, 1.0, -0.125, 0.75, 2.5, -1.5],
        [0.03, -0.44, 0.91, -0.02, 0.17, -0.68, 0.33, 0.05]]


class Q3CodecRegressions(unittest.TestCase):
    def test_weight_bytes_match_the_independent_kv_reference(self):
        # The KV cache already stores this codec; a second layout would make
        # the same logical page mean two different things.
        for group_size in (1, 3, 4, 8):
            for row in ROWS:
                with self.subTest(group_size=group_size, row=row[:2]):
                    mine = quantize_q3_row(row, group_size)
                    theirs = q3_reference.quantize_q3_row(row, group_size)
                    self.assertEqual(mine, theirs)
                    self.assertEqual(decode_q3_row(mine, len(row), group_size),
                                     q3_reference.decode_q3_row(theirs, len(row), group_size))

    def test_three_bits_per_value_sit_between_nothing_and_a_nibble(self):
        self.assertEqual(len(quantize_q3_row(ROWS[0], 8)), 4 + 3)
        self.assertEqual(len(quantize_q3_row(ROWS[0], 4)), 2 * (4 + 2))
        # With a small group the scale dominates and Q3 saves nothing at all.
        self.assertEqual(len(quantize_q3_row(ROWS[0], 4)), len(quantize_q4_row(ROWS[0], 4)))
        self.assertLess(len(quantize_q3_row(ROWS[0], 8)), len(quantize_q4_row(ROWS[0], 8)))

    def test_q3_is_coarser_than_q4_on_the_same_group(self):
        for row in ROWS:
            with self.subTest(row=row[:2]):
                q3 = decode_q3_row(quantize_q3_row(row, 8), len(row), 8)
                q4 = decode_q4_row(quantize_q4_row(row, 8), len(row), 8)
                self.assertGreater(max(abs(a - b) for a, b in zip(row, q3)),
                                   max(abs(a - b) for a, b in zip(row, q4)))

    def test_reserved_codes_padding_bits_and_bad_scales_are_rejected(self):
        encoded = bytearray(quantize_q3_row([1.0, 2.0, 3.0], 4))
        reserved = bytearray(encoded)
        reserved[4] = (reserved[4] & ~0b111) | 0b100  # code -4
        with self.assertRaises(NexaPackError):
            decode_q3_row(bytes(reserved), 3, 4)
        padding = bytearray(encoded)
        padding[4 + 1] |= 0b1000_0000  # a bit past the last code
        with self.assertRaises(NexaPackError):
            decode_q3_row(bytes(padding), 3, 4)
        negative = bytearray(encoded)
        negative[3] |= 0x80
        with self.assertRaises(NexaPackError):
            decode_q3_row(bytes(negative), 3, 4)
        with self.assertRaises(NexaPackError):
            decode_q3_row(bytes(encoded[:-1]), 3, 4)

    def test_the_container_reports_the_codec_it_stores(self):
        import tempfile
        path = Path(tempfile.mkdtemp()) / "matrix.nxp"
        write_q3_matrix(path, 2, 8, 8, iter(ROWS), block_rows=1)
        with NexaPackReader(path) as reader:
            self.assertEqual(reader.codec_id, Q3_CODEC_ID)
            self.assertEqual(reader.row_bytes, 4 + 3)
            buffer = bytearray(reader.row_bytes)
            reader.read_rows_into(0, 1, buffer)
            self.assertEqual(decode_q3_row(bytes(buffer), 8, 8),
                             decode_q3_row(quantize_q3_row(ROWS[0], 8), 8, 8))


class Q3KernelRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import shutil
        if not (shutil.which("clang") or shutil.which("cc")):
            raise unittest.SkipTest("C compiler unavailable")
        cls.kernels = _load_kernels()

    def matmul(self, inputs, packed, rows, cols, group_size, *, output_count=None):
        source = (ctypes.c_float * len(inputs))(*inputs)
        weights = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
        batch = len(inputs) // cols
        count = rows * batch if output_count is None else output_count
        output = (ctypes.c_float * max(count, 1))()
        status = self.kernels.nexa_q3_matmul(source, len(source), batch, weights, len(weights),
                                             rows, cols, group_size, output, count)
        return status, list(output)

    def test_kernel_matches_the_python_reference_across_group_sizes(self):
        inputs = [0.25, -0.5, 1.0, 0.125, -0.75, 0.5, 0.0, 2.0]
        for group_size in (3, 4, 8):
            with self.subTest(group_size=group_size):
                packed = b"".join(quantize_q3_row(row, group_size) for row in ROWS)
                status, output = self.matmul(inputs, packed, 2, 8, group_size)
                self.assertEqual(status, 0)
                for index, row in enumerate(ROWS):
                    decoded = decode_q3_row(quantize_q3_row(row, group_size), 8, group_size)
                    expected = math.fsum(a * b for a, b in zip(inputs, decoded))
                    self.assertAlmostEqual(output[index], expected, places=5)

    def test_kernel_rejects_reserved_codes_padding_and_small_buffers(self):
        packed = b"".join(quantize_q3_row(row, 4) for row in ROWS)
        status, _ = self.matmul([0.5] * 8, packed, 2, 8, 4, output_count=1)
        self.assertNotEqual(status, 0)
        reserved = bytearray(packed)
        reserved[4] = (reserved[4] & ~0b111) | 0b100
        status, _ = self.matmul([0.5] * 8, bytes(reserved), 2, 8, 4)
        self.assertNotEqual(status, 0)
        padded = bytearray(packed)
        padded[5] |= 0b0100_0000  # bit past the twelfth code of a group of four
        status, _ = self.matmul([0.5] * 8, bytes(padded), 2, 8, 4)
        self.assertNotEqual(status, 0)

    def test_decode_row_matches_the_reference(self):
        packed = quantize_q3_row(ROWS[1], 8)
        weights = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
        output = (ctypes.c_float * 8)()
        status = self.kernels.nexa_q3_decode_row(weights, len(weights), 8, 8, output, 8)
        self.assertEqual(status, 0)
        for produced, expected in zip(output, decode_q3_row(packed, 8, 8)):
            self.assertAlmostEqual(produced, expected, places=6)


class Q3ExecutionRegressions(_DenseFixture):
    def test_four_codecs_form_a_ladder_in_bytes_and_error(self):
        from test_transformer_forward_regressions import random_bundle
        path = self.directory / "reference"
        config, values = random_bundle(path, layers=2, heads=4, kv_heads=2, tied=False, seed=77)
        matrices = [name for name, shape in config.required_tensor_shapes().items() if len(shape) == 2]
        logits, payload = {}, {}
        for codec in ("q3", "q4", "q8", "f32"):
            bundle = self.write(self.directory / codec, config, values,
                                codecs={name: codec for name in matrices},
                                group_size=8, block_rows=3)
            logits[codec], _ = self.run_model(bundle)
            with ModelBundleReader(bundle) as reader:
                payload[codec] = sum(item["packed_payload_bytes"] for item in reader.inspect()["tensors"])
        error = {codec: max(abs(a - b) for left, right in zip(logits[codec], logits["f32"])
                            for a, b in zip(left, right)) for codec in ("q3", "q4", "q8")}
        self.assertGreater(error["q3"], error["q4"])
        self.assertGreater(error["q4"], error["q8"])
        self.assertLess(payload["q3"], payload["q4"])
        self.assertLess(payload["q4"], payload["q8"])
        self.assertLess(payload["q8"], payload["f32"])

    def test_a_small_group_makes_q3_dominated_and_the_plan_drops_it(self):
        # Four values per group: Q3 and Q4 both spend six bytes, so the coarser
        # codec buys nothing and the frontier must exclude it.
        entry = {"name": "tensor", "dense_bytes": 512,
                 "codecs": {"q3": {"packed_bytes": 192, "sensitivity": {"rmse": 0.28}},
                            "q4": {"packed_bytes": 192, "sensitivity": {"rmse": 0.08}},
                            "q8": {"packed_bytes": 256, "sensitivity": {"rmse": 0.004}}}}
        self.assertEqual([option["codec"] for option in _options("tensor", entry)],
                         ["q4", "q8", "f32"])
        # With a larger group Q3 is genuinely cheaper and stays in the ladder.
        entry["codecs"]["q3"]["packed_bytes"] = 112
        entry["codecs"]["q4"]["packed_bytes"] = 128
        self.assertEqual([option["codec"] for option in _options("tensor", entry)],
                         ["q3", "q4", "q8", "f32"])

    def test_a_bundle_can_mix_every_weight_codec(self):
        config = self.config()
        values = self.exact_weights(config, 4)
        names = sorted(name for name, shape in config.required_tensor_shapes().items() if len(shape) == 2)
        codecs = dict(zip(names, ("q3", "q4", "q8", "f32")))
        bundle = self.write(self.directory / "all-codecs", config, values, codecs=codecs)
        with ModelBundleReader(bundle) as reader:
            stored = {item["name"]: item["codec"] for item in reader.inspect()["tensors"]}
        expected = {"q3": "Q3_GROUPED", "q4": "Q4_GROUPED", "q8": "Q8_GROUPED", "f32": "RAW_F32_MATRIX"}
        for name, codec in codecs.items():
            self.assertEqual(stored[name], expected[codec])
        logits, report = self.run_model(bundle)
        self.assertTrue(all(math.isfinite(value) for row in logits for value in row))
        self.assertGreater(report["io"]["q4_payload_bytes_read"], 0)
        self.assertGreater(report["io"]["raw_payload_bytes_read"], 0)


class Q3CLIRegressions(_DenseFixture):
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

    def test_convert_inspect_and_run_a_q3_bundle(self):
        bundle = self.directory / "q3-bundle"
        self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint), "--out", str(bundle),
                      "--matrix-codec", "q3", "--group-size", "8", "--block-rows", "3")
        inspected = self.run_tool("nexa_inspect.py", str(bundle), "--verify")
        bits = {item["codec"]: item["storage_bits"] for item in inspected["tensors"]}
        self.assertEqual(bits["Q3_GROUPED"], 3)
        report = self.run_tool("nexa_run.py", str(bundle), "--tokens", "1,3", "--decode-tokens", "5",
                               "--tile-rows", "3", "--memory-budget", "4MiB")
        self.assertEqual(report["token_ids"], [1, 3, 5])

    def test_calibration_and_plan_cover_the_whole_ladder(self):
        report = self.directory / "calibration.json"
        self.run_tool("nexa_calibrate.py", "--checkpoint", str(self.checkpoint), "--tokens", "1,3",
                      "--group-size", "8", "--block-rows", "3", "--tile-rows", "3",
                      "--memory-budget", "8MiB", "--report", str(report))
        measured = json.loads(report.read_text())
        self.assertEqual(measured["measured_codecs"], ["q3", "q4", "q8"])
        # Coarser codecs move the logits more, whole model at a time.
        self.assertGreater(measured["all_packed"]["q3"]["sensitivity"]["rmse"],
                           measured["all_packed"]["q4"]["sensitivity"]["rmse"])
        self.assertGreater(measured["all_packed"]["q4"]["sensitivity"]["rmse"],
                           measured["all_packed"]["q8"]["sensitivity"]["rmse"])
        plan = self.directory / "plan.json"
        planned = self.run_tool("nexa_precision.py", "plan", "--calibration", str(report),
                                "--budget", "900B", "--out", str(plan))
        self.assertLessEqual(planned["provenance"]["planned_bytes"], 900)
        self.assertTrue(set(planned["codecs"].values()) <= {"q3", "q4", "q8", "f32"})
        bundle = self.directory / "planned"
        converted = self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint), "--out",
                                  str(bundle), "--precision-map", str(plan),
                                  "--group-size", "8", "--block-rows", "3")
        self.assertEqual(set(converted["dense_tensors"]),
                         {name for name, codec in planned["codecs"].items() if codec == "f32"})
        executed = self.run_tool("nexa_run.py", str(bundle), "--tokens", "1,3",
                                 "--tile-rows", "3", "--memory-budget", "8MiB")
        self.assertEqual(executed["token_ids"], [1, 3])


if __name__ == "__main__":
    unittest.main()

"""Q8_GROUPED weights: codec contract, kernel, dispatch and mixed bundles."""
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

from runtime.nexapack.bundle import ModelBundleError, ModelBundleReader
from runtime.nexapack.format import (
    NexaPackError, NexaPackReader, Q8_CODEC_ID, decode_q4_row, decode_q8_row,
    quantize_q4_row, quantize_q8_row, write_q8_matrix,
)
from runtime.nexapack.transformer import _load_kernels
from test_dense_weights_regressions import _DenseFixture
from test_transformer_forward_regressions import f32

ROWS = [[0.5, -0.25, 3.0, 1.0, -0.125, 0.75, 2.5, -1.5],
        [0.03, -0.44, 0.91, -0.02, 0.17, -0.68, 0.33, 0.05]]


class Q8CodecRegressions(unittest.TestCase):
    def test_q8_round_trip_is_closer_than_q4_for_the_same_group(self):
        for row in ROWS:
            with self.subTest(row=row[:2]):
                q8 = decode_q8_row(quantize_q8_row(row, 4), len(row), 4)
                q4 = decode_q4_row(quantize_q4_row(row, 4), len(row), 4)
                error8 = max(abs(a - b) for a, b in zip(row, q8))
                error4 = max(abs(a - b) for a, b in zip(row, q4))
                self.assertLess(error8, error4)
                # 127 levels instead of 7: the error shrinks with the step.
                self.assertLess(error8, error4 / 8)

    def test_group_layout_costs_one_byte_per_value_plus_a_scale(self):
        encoded = quantize_q8_row(ROWS[0], 4)
        self.assertEqual(len(encoded), 2 * (4 + 4))
        tail = quantize_q8_row([1.0, 2.0, 3.0], 4)
        self.assertEqual(len(tail), 4 + 4)  # A partial group still pads to the group.
        decoded = decode_q8_row(tail, 3, 4)
        self.assertEqual(len(decoded), 3)
        for original, restored in zip([1.0, 2.0, 3.0], decoded):
            self.assertLess(abs(original - restored), 3.0 / 127)

    def test_zero_rows_and_exact_values_survive_the_round_trip(self):
        zeros = quantize_q8_row([0.0, 0.0, 0.0, 0.0], 4)
        self.assertEqual(decode_q8_row(zeros, 4, 4), [0.0, 0.0, 0.0, 0.0])
        exact = [127 * 0.5, -64 * 0.5, 0.0, 32 * 0.5]
        self.assertEqual(decode_q8_row(quantize_q8_row(exact, 4), 4, 4), exact)

    def test_reserved_codes_padding_and_bad_scales_are_rejected(self):
        encoded = bytearray(quantize_q8_row([1.0, 2.0, 3.0], 4))
        reserved = bytearray(encoded)
        reserved[4] = 0x80  # -128
        with self.assertRaises(NexaPackError):
            decode_q8_row(bytes(reserved), 3, 4)
        padded = bytearray(encoded)
        padded[4 + 3] = 5  # The padding lane of a partial group must stay zero.
        with self.assertRaises(NexaPackError):
            decode_q8_row(bytes(padded), 3, 4)
        negative = bytearray(encoded)
        negative[3] |= 0x80  # Flip the scale's sign bit.
        with self.assertRaises(NexaPackError):
            decode_q8_row(bytes(negative), 3, 4)
        with self.assertRaises(NexaPackError):
            decode_q8_row(bytes(encoded[:-1]), 3, 4)

    def test_the_container_reports_the_codec_it_stores(self):
        import tempfile
        directory = Path(tempfile.mkdtemp())
        path = directory / "matrix.nxp"
        write_q8_matrix(path, 2, 8, 4, iter(ROWS), block_rows=1)
        with NexaPackReader(path) as reader:
            self.assertEqual(reader.codec_id, Q8_CODEC_ID)
            self.assertEqual(reader.group_size, 4)
            self.assertEqual(reader.row_bytes, 2 * (4 + 4))
            buffer = bytearray(reader.row_bytes)
            reader.read_rows_into(1, 1, buffer)
            decoded = decode_q8_row(bytes(buffer), 8, 4)
        for original, restored in zip(ROWS[1], decoded):
            self.assertLess(abs(original - restored), 0.01)


class Q8KernelRegressions(unittest.TestCase):
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
        status = self.kernels.nexa_q8_matmul(source, len(source), batch, weights, len(weights),
                                             rows, cols, group_size, output, count)
        return status, list(output)

    def test_kernel_matches_the_python_reference(self):
        packed = b"".join(quantize_q8_row(row, 4) for row in ROWS)
        inputs = [0.25, -0.5, 1.0, 0.125, -0.75, 0.5, 0.0, 2.0]
        status, output = self.matmul(inputs, packed, 2, 8, 4)
        self.assertEqual(status, 0)
        for index, row in enumerate(ROWS):
            decoded = decode_q8_row(quantize_q8_row(row, 4), 8, 4)
            expected = math.fsum(a * b for a, b in zip(inputs, decoded))
            self.assertAlmostEqual(output[index], expected, places=5)

    def test_kernel_rejects_small_buffers_and_invalid_codes(self):
        packed = b"".join(quantize_q8_row(row, 4) for row in ROWS)
        status, _ = self.matmul([0.5] * 8, packed, 2, 8, 4, output_count=1)
        self.assertNotEqual(status, 0)
        status, _ = self.matmul([0.5] * 4, packed, 2, 8, 4)
        self.assertNotEqual(status, 0)
        reserved = bytearray(packed)
        reserved[4] = 0x80
        status, _ = self.matmul([0.5] * 8, bytes(reserved), 2, 8, 4)
        self.assertNotEqual(status, 0)
        status, _ = self.matmul([float("inf")] * 8, packed, 2, 8, 4)
        self.assertNotEqual(status, 0)


class Q8ExecutionRegressions(_DenseFixture):
    def test_q8_lands_between_q4_and_dense_in_error_and_in_bytes(self):
        from test_transformer_forward_regressions import random_bundle
        path = self.directory / "reference"
        config, values = random_bundle(path, layers=2, heads=4, kv_heads=2, tied=False, seed=77)
        matrices = [name for name, shape in config.required_tensor_shapes().items() if len(shape) == 2]
        bundles = {}
        for codec in ("q4", "q8", "f32"):
            bundles[codec] = self.write(self.directory / codec, config, values,
                                        codecs={name: codec for name in matrices},
                                        group_size=8, block_rows=3)
        logits, payload = {}, {}
        for codec, bundle in bundles.items():
            output, report = self.run_model(bundle)
            logits[codec] = output
            with ModelBundleReader(bundle) as reader:
                payload[codec] = sum(item["packed_payload_bytes"] for item in reader.inspect()["tensors"])
        error = {codec: max(abs(a - b) for left, right in zip(logits[codec], logits["f32"])
                            for a, b in zip(left, right)) for codec in ("q4", "q8")}
        self.assertLess(error["q8"], error["q4"])
        self.assertLess(payload["q4"], payload["q8"])
        self.assertLess(payload["q8"], payload["f32"])

    def boundary_weights(self, config, group_size=4):
        """Values every codec stores exactly: zero and the group's own maximum.

        Q4 scales by max/7 and Q8 by max/127, so no single set of fractions is
        exact for both. Zero and +-max are, which isolates the dispatch path
        from any quantization difference.
        """
        values = {}
        for name, shape in config.required_tensor_shapes().items():
            if len(shape) == 1:
                values[name] = [f32(1.0 + (index % 3) * 0.25) for index in range(shape[0])]
                continue
            rows = []
            for row in range(shape[0]):
                line = [f32(3.5 if (row + column) % 3 == 0 else
                            (-3.5 if (row + column) % 3 == 1 else 0.0))
                        for column in range(shape[1])]
                for start in range(0, shape[1], group_size):
                    line[start] = f32(3.5)
                rows.append(line)
            values[name] = rows
        return values

    def test_one_bundle_can_mix_three_weight_codecs(self):
        config = self.config()
        values = self.boundary_weights(config, 4)
        codecs = {"model.embed_tokens.weight": "f32",
                  "model.layers.0.mlp.down_proj.weight": "q8",
                  "model.layers.1.mlp.down_proj.weight": "q4"}
        bundle = self.write(self.directory / "mixed", config, values, codecs=codecs)
        with ModelBundleReader(bundle) as reader:
            stored = {item["name"]: (item["codec"], item["storage_bits"])
                      for item in reader.inspect()["tensors"]}
        self.assertEqual(stored["model.embed_tokens.weight"], ("RAW_F32_MATRIX", 32))
        self.assertEqual(stored["model.layers.0.mlp.down_proj.weight"], ("Q8_GROUPED", 8))
        self.assertEqual(stored["model.layers.1.mlp.down_proj.weight"], ("Q4_GROUPED", 4))
        logits, report = self.run_model(bundle)
        # Exactly representable weights: every codec agrees with the others.
        packed, _ = self.run_model(self.write(self.directory / "all-q4", config, values))
        self.assertEqual(logits, packed)
        self.assertGreater(report["io"]["q4_payload_bytes_read"], 0)
        self.assertGreater(report["io"]["raw_payload_bytes_read"], 0)

    def test_a_manifest_codec_that_contradicts_its_file_is_rejected(self):
        import shutil
        config = self.config()
        values = self.boundary_weights(config, 4)
        source = self.write(self.directory / "honest", config, values,
                            codecs={"model.layers.0.mlp.down_proj.weight": "q8"})
        damaged = self.directory / "mislabelled"
        shutil.copytree(source, damaged)
        manifest = json.loads((damaged / "manifest.json").read_text())
        manifest["tensors"]["model.layers.0.mlp.down_proj.weight"]["codec"] = "Q4_GROUPED"
        (damaged / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ModelBundleError):
            ModelBundleReader(damaged).close()


class Q8CLIRegressions(_DenseFixture):
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

    def test_convert_inspect_and_run_a_q8_bundle(self):
        bundle = self.directory / "q8-bundle"
        self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint), "--out", str(bundle),
                      "--matrix-codec", "q8", "--group-size", "4", "--block-rows", "3")
        inspected = self.run_tool("nexa_inspect.py", str(bundle), "--verify")
        codecs = {item["codec"] for item in inspected["tensors"]}
        self.assertEqual(codecs, {"Q8_GROUPED", "RAW_F32"})
        self.assertGreater(inspected["validation"]["q4_payload_bytes_read"], 0)
        report = self.run_tool("nexa_run.py", str(bundle), "--tokens", "1,3", "--decode-tokens", "5",
                               "--tile-rows", "3", "--memory-budget", "4MiB")
        self.assertEqual(report["token_ids"], [1, 3, 5])

    def test_tensor_codec_builds_a_mixed_bundle_and_rejects_bad_names(self):
        bundle = self.directory / "cli-mixed"
        self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint), "--out", str(bundle),
                      "--tensor-codec", "model.embed_tokens.weight=q8",
                      "--tensor-codec", "model.layers.0.mlp.down_proj.weight=f32",
                      "--group-size", "4", "--block-rows", "3")
        inspected = self.run_tool("nexa_inspect.py", str(bundle), "--verify")
        codecs = {item["name"]: item["codec"] for item in inspected["tensors"]}
        self.assertEqual(codecs["model.embed_tokens.weight"], "Q8_GROUPED")
        self.assertEqual(codecs["model.layers.0.mlp.down_proj.weight"], "RAW_F32_MATRIX")
        self.assertEqual(codecs["model.layers.0.mlp.up_proj.weight"], "Q4_GROUPED")
        for arguments in (("--tensor-codec", "model.embed_tokens.weight=q2"),
                          ("--tensor-codec", "model.norm.weight=q8"),
                          ("--tensor-codec", "no-equals-sign"),
                          ("--tensor-codec", "model.embed_tokens.weight=q8", "--matrix-codec", "q4")):
            with self.subTest(arguments=arguments[1]):
                self.run_tool("nexa_convert.py", "--checkpoint", str(self.checkpoint), "--out",
                              str(self.directory / "never"), *arguments, expect=2)


if __name__ == "__main__":
    unittest.main()

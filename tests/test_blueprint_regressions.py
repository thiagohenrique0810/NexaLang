"""Integrated CPU milestone: real packed execution, budgeting and legacy guard."""
import json
import io
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "projeto-llm"))

from compiler.planner.memory import MemoryBudgetError
from runtime.build_runtime import build_runtime
from runtime.nexapack.executor import make_plan, run_packed_matmul
from runtime.nexapack.format import NexaPackReader, write_q4_matrix
from tools.nexa_convert import convert_matrix, _read_f32_rows
from turboir.ir.nodes import IRModelConfig, IRProgram, DeviceTarget
from turboir.planner.planner import ExecutionPlanner
from turboruntime.core.engine import EngineConfig, InferenceEngine


class BlueprintRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which("clang") or shutil.which("cc")):
            raise unittest.SkipTest("C compiler unavailable")
        cls.build_dir = tempfile.TemporaryDirectory(prefix="nexa-q4-integration-")
        cls.library = build_runtime("nexa_q4", cls.build_dir.name)

    @classmethod
    def tearDownClass(cls):
        # A loaded DLL cannot be removed on Windows until process exit.
        try:
            cls.build_dir.cleanup()
        except PermissionError:
            pass

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-blueprint-")
        self.root = Path(self.temp.name)
        self.pack = self.root / "weights.nxp"

    def tearDown(self):
        self.temp.cleanup()

    def write_weights(self, rows=2000, cols=73, group_size=17):
        def data():
            for r in range(rows):
                yield [((r * 13 + c * 7) % 29 - 14) / 16.0 for c in range(cols)]
        write_q4_matrix(self.pack, rows, cols, group_size, data(), block_rows=31)

    def test_weights_larger_than_budget_execute_without_expansion(self):
        self.write_weights()
        result = run_packed_matmul(self.pack, batch=2, tile_rows=13,
                                   memory_budget="80KiB", verify=True, library_path=self.library)
        memory = result["memory"]
        self.assertGreater(memory["weights_packed_bytes"], memory["budget_bytes"])
        self.assertLessEqual(memory["managed_buffers_peak_bound_bytes"], memory["budget_bytes"])
        self.assertEqual(memory["full_dequantized_weight_buffer_bytes"], 0)
        self.assertIsNone(memory["peak_vram_bytes"])
        self.assertTrue(result["validation"]["verified"])
        self.assertGreater(result["tiles_executed"], 1)
        self.assertGreaterEqual(result["io"]["file_payload_bytes_read"], memory["weights_packed_bytes"])
        self.assertEqual(len(result["output_sha256_by_batch"]), 2)

    def test_tiling_does_not_change_outputs(self):
        self.write_weights(rows=31, cols=19, group_size=5)
        results = []
        for rows in (1, 7, 31):
            results.append(run_packed_matmul(self.pack, batch=3, tile_rows=rows,
                                            memory_budget="96KiB", verify=True,
                                            library_path=self.library)["output_sha256_by_batch"])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_budget_rejects_before_weights_and_native_code_are_loaded(self):
        self.write_weights(rows=3, cols=7, group_size=4)
        with patch.object(NexaPackReader, "read_rows_into", side_effect=AssertionError("weights loaded")):
            with patch("runtime.nexapack.executor._load_kernel", side_effect=AssertionError("kernel loaded")):
                with self.assertRaises(MemoryBudgetError):
                    run_packed_matmul(self.pack, memory_budget="1KiB")
        with NexaPackReader(self.pack) as reader:
            plan, _, _, _ = make_plan(reader, 1, 2, "96KiB", "4KiB")
            self.assertEqual(reader.payload_bytes_read, 0)
            self.assertGreaterEqual(plan.reserves["host"], 4096)

    def test_tile_is_capped_to_reader_limit_during_preflight(self):
        reader = SimpleNamespace(rows=20, cols=64, row_bytes=33 * 1024 * 1024,
                                 codec_id='Q4_GROUPED', codec_version=1)
        plan, rows, _, _ = make_plan(reader, 1, 20, "512MiB")
        self.assertEqual(rows, 1)
        self.assertEqual(plan.allocations["packed_tile"].size_bytes, reader.row_bytes)

    def test_float32_conversion_is_streamed_and_atomic_on_invalid_input(self):
        source = self.root / "weights.f32"
        source.write_bytes(struct.pack("<15f", *range(-7, 8)))
        result = convert_matrix(source, self.pack, 3, 5, group_size=4, block_rows=2)
        self.assertEqual(result["source_bytes"], 60)
        original = self.pack.read_bytes()
        with self.assertRaises(ValueError):
            convert_matrix(source, self.pack, 4, 5)
        source.write_bytes(struct.pack("<15f", *([0.0] * 14 + [float("nan")])))
        with self.assertRaises(ValueError):
            convert_matrix(source, self.pack, 3, 5)
        self.assertEqual(self.pack.read_bytes(), original)
        with self.assertRaises(ValueError):
            convert_matrix(source, source, 3, 5)

    def test_conversion_handles_large_rows_and_short_reads_with_bounded_io(self):
        class ShortStream(io.BytesIO):
            largest_request = 0

            def readinto(self, buffer):
                self.largest_request = max(self.largest_request, len(buffer))
                return super().readinto(buffer[:101])

        source = ShortStream(struct.pack("<f", 1.25) * 40000)
        write_q4_matrix(self.pack, 2, 20000, 32, _read_f32_rows(source, 2, 20000))
        self.assertLessEqual(source.largest_request, 65536)
        with NexaPackReader(self.pack) as reader:
            self.assertEqual(reader.rows, 2)
            self.assertEqual(reader.read_rows(0, 1), reader.read_rows(1, 1))

    def test_benchmark_cli_reports_failure_and_preserves_weights(self):
        self.write_weights(rows=9, cols=7, group_size=3)
        command = [sys.executable, str(ROOT / "tools/nexa_bench.py"), "--pack", str(self.pack)]
        report = self.root / "report.json"
        failed = subprocess.run(command + ["--memory-budget", "1KiB", "--report", str(report)],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("budget exceeded", failed.stderr)
        self.assertFalse(report.exists())
        original = self.pack.read_bytes()
        alias = subprocess.run(command + ["--report", str(self.pack)],
                               capture_output=True, text=True, timeout=30)
        self.assertNotEqual(alias.returncode, 0)
        self.assertEqual(self.pack.read_bytes(), original)
        passed = subprocess.run(command + ["--verify", "--report", str(report),
                                          "--csv", str(self.root / "report.csv")],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        record = json.loads(report.read_text())
        self.assertTrue(record["validation"]["verified"])
        self.assertEqual(record, json.loads(passed.stdout))
        self.assertIn("managed_buffers_peak_bound_bytes", (self.root / "report.csv").read_text())


class LegacyBudgetRegressions(unittest.TestCase):
    def plan(self, limit):
        model = IRModelConfig(name="test-model")
        model.target.device = DeviceTarget.CPU
        model.target.memory_limit = limit
        return ExecutionPlanner().plan(IRProgram(models=[model]))[0]

    def test_legacy_budget_survives_plan_serialization(self):
        small, large = self.plan("512MB"), self.plan("8GB")
        self.assertNotEqual(small.config, large.config)
        self.assertEqual(EngineConfig.from_plan(small).memory_limit, "512MB")

    def test_legacy_executor_rejects_unsupported_budget_before_loading(self):
        engine = InferenceEngine()
        with patch("turboruntime.core.engine.ModelLoader.load") as load:
            with self.assertRaisesRegex(NotImplementedError, "cannot enforce memory_limit"):
                engine.init_from_plan(self.plan("512MB"))
            self.assertFalse(engine.ready)
            load.assert_not_called()

    def test_legacy_executor_rejects_invalid_budget_type(self):
        with self.assertRaises(ValueError):
            InferenceEngine().init(EngineConfig(memory_limit=0))


if __name__ == "__main__":
    unittest.main()

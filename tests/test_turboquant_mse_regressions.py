"""MSE-only TurboQuant ownership, legacy determinism, and allocation failures.

The C harness intercepts runtime heap calls, never caller buffers. It is compiled
with sanitizers where supported; no NumPy, Torch or pre-change artifact is needed.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/turboquant_mse_regressions.c"


class TurboQuantMSERegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang") or shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="nexa-tq-mse-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = Path(cls.directory.name) / ("tq_mse.exe" if os.name == "nt" else "tq_mse")
        command = [compiler, "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                   str(SOURCE), "-o", str(cls.binary)]
        if os.name != "nt":
            command += ["-pthread", "-lm", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.environment = dict(os.environ)
        # macOS ASan lacks LeakSanitizer; the independent ledger checks every free.
        cls.environment["ASAN_OPTIONS"] = "detect_leaks=0" if sys.platform == "darwin" else "detect_leaks=1"
        cls.environment["UBSAN_OPTIONS"] = "halt_on_error=1"

    def run_mode(self, mode):
        result = subprocess.run([str(self.binary), mode], capture_output=True, text=True,
                                env=self.environment, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_raw_packed_mse_exact_legacy_equivalence(self):
        self.run_mode("equivalence")

    def test_parallel_context_reuse_and_scratch(self):
        self.run_mode("parallel")

    def test_every_creation_allocation_failure_releases_owned_memory(self):
        self.run_mode("creation_failures")

    def test_operation_allocation_failures_preserve_outputs_and_context(self):
        self.run_mode("operation_failures")

    def test_prod_rejection_never_writes_or_allocates_for_mse_context(self):
        self.run_mode("unsupported_prod")

    def test_invalid_and_large_dimensions_do_not_allocate_unchecked(self):
        self.run_mode("invalid")

    def test_pre_change_golden_preserves_prod_rng_and_mse_bytes(self):
        self.run_mode("golden")

    def test_reported_state_matches_allocator_and_grows_linearly(self):
        report = json.loads(self.run_mode("memory"))
        self.assertEqual(report["bits"], 3)
        self.assertEqual(report["seed"], 42)
        small, large = report["measurements"]
        self.assertEqual((small["dimension"], large["dimension"]), (64, 1024))
        self.assertEqual(large["mse_state_bytes"] - small["mse_state_bytes"], (1024 - 64) * 4)
        for sample in report["measurements"]:
            self.assertEqual(sample["mse_quantize_scratch_bytes"], sample["dimension"] * 4)
            self.assertEqual(sample["legacy_quantize_scratch_bytes"], sample["dimension"] * 4)
            self.assertGreater(sample["legacy_state_bytes"], sample["mse_state_bytes"])
            self.assertGreaterEqual(sample["mse_creation_peak_bytes"], sample["mse_state_bytes"])


if __name__ == "__main__":
    unittest.main()

"""Integration checks for the installed CLI and failed compilation boundaries."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CliRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nexa-cli-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def source(self, text, name="main.nxl"):
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "nx.py"), *map(str, args)],
                              cwd=self.directory, capture_output=True, text=True, timeout=60)

    @unittest.skipIf(os.name == "nt", "Creating symlinks needs Windows developer mode")
    def test_installed_symlink(self):
        link = self.directory / "nxc"
        link.symlink_to(ROOT / "nxc")
        result = subprocess.run([str(link), "--version"], cwd=self.directory,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NexaLang", result.stdout)

    def test_failed_compile_never_runs_previous_program(self):
        source = self.source('fn main() -> i32 { print("OLD_BINARY_EXECUTED"); return 0; }')
        good = self.cli("run", source)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertIn("OLD_BINARY_EXECUTED", good.stdout)
        source.write_text('fn main() -> i32 { while (1) { } return 0; }')
        bad = self.cli("run", source)
        self.assertNotEqual(bad.returncode, 0)
        self.assertNotIn("OLD_BINARY_EXECUTED", bad.stdout)

    def test_invalid_source_does_not_report_success(self):
        source = self.source('fn main() { let x = ; }')
        result = self.cli("build", source, "--no-link")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.directory / "artifacts/build/output.ll").exists())

    def test_missing_std_module_fails(self):
        source = self.source('use std::missing_module::Thing; fn main() -> i32 { return 0; }')
        result = self.cli("build", source, "--no-link")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Module not found", result.stdout + result.stderr)

    def test_std_resolves_from_installed_toolchain(self):
        source = self.source('use std::compress::Quantizer; fn main() -> i32 { return 0; }')
        result = self.cli("build", source, "--no-link")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.directory / "artifacts/build/output.ll").is_file())

    def test_cyclic_modules_produce_error(self):
        source = self.source('mod a; fn main() -> i32 { return 0; }')
        self.source('mod b;', 'a.nxl')
        self.source('mod a;', 'b.nxl')
        result = self.cli("build", source, "--no-link")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cyclic module", result.stdout + result.stderr)

    def test_package_entry_point_resolves_from_src(self):
        (self.directory / "nexa.json").write_text(json.dumps({"main": "src/main.nxl"}))
        (self.directory / "src").mkdir()
        source = self.source('mod demo; fn main() -> i32 { return demo::answer(); }', 'src/main.nxl')
        package = self.directory / "deps/demo"
        (package / "src").mkdir(parents=True)
        (package / "nexa.json").write_text(json.dumps({"main": "src/lib.nxl"}))
        (package / "src/lib.nxl").write_text('pub fn answer() -> i32 { return 0; }')
        result = self.cli("run", source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_and_jit_preserve_exit_status(self):
        source = self.source('fn main() -> i32 { return 7; }')
        for extra in ([], ["--jit"]):
            result = self.cli("run", source, *extra)
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)

    def test_relative_executable_path_and_optimization(self):
        source = self.source('fn main() -> i32 { return 0; }')
        result = self.cli("run", source, "--exe", "output/bin/program", "--opt", "O3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.directory / "output/bin/program").exists())
        self.assertIn("LLVM O3 passes applied", result.stdout)

    def test_missing_spirv_toolchain_is_a_failure(self):
        output = self.directory / "kernel.spv"
        environment = dict(os.environ, PATH="")
        result = subprocess.run(
            [sys.executable, str(ROOT / "bootstrap/main.py"),
             str(ROOT / "examples/gpu_kernel_spirv.nxl"), "--target", "spirv",
             "--emit", "spv", "--out", str(output)],
            env=environment, cwd=self.directory, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())
        self.assertIn("SPIR-V EMIT ERROR", result.stdout + result.stderr)

    def test_native_test_runner_preserves_assertion_status(self):
        source = self.source('@[test] fn example() { assert!(true, "passing"); }')
        good = self.cli("test", source)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertIn("1 test(s) passed", good.stdout)
        source.write_text('@[test] fn example() { assert!(false, "expected failure"); }')
        bad = self.cli("test", source)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("expected failure", bad.stdout + bad.stderr)


if __name__ == "__main__":
    unittest.main()

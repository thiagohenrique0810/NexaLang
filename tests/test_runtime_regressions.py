"""Build native runtime regressions from source, without cached binaries."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimeRegressions(unittest.TestCase):
    def test_native_smoke_and_regressions(self):
        compiler = shutil.which("clang") or shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="nexa-runtime-tests-") as temp:
            for source in ("turboquant_test.c", "turboquant_regressions.c", "nexa_async_regressions.c"):
                binary = Path(temp) / (Path(source).stem + (".exe" if os.name == "nt" else ""))
                command = [compiler, "-std=c11", "-O1", "-g", "-Wall", "-Wextra",
                           str(ROOT / "runtime" / source),
                           str(ROOT / "runtime" / ("nexa_async.c" if source.startswith("nexa_async") else "turboquant.c")),
                           "-o", str(binary)]
                if os.name != "nt":
                    command.extend(["-lm", "-pthread", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])
                elif source.startswith("nexa_async"):
                    command.append("-lws2_32")
                built = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                env = dict(os.environ)
                env["ASAN_OPTIONS"] = "detect_leaks=0" if sys.platform == "darwin" else "detect_leaks=1"
                env["UBSAN_OPTIONS"] = "halt_on_error=1"
                run = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=60)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()

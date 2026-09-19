"""Compile every example, checking explicit success/error expectations.

The manifest must account for every .nxl file. Error cases must fail for the
expected reason; a parser crash is not accepted as a successful ownership test.
Selected standalone examples are also linked and executed.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nx import _native_link_cmd


def main():
    manifest = json.loads((Path(__file__).with_name("examples_manifest.json")).read_text())
    entries = manifest["examples"]
    actual = {path.relative_to(ROOT / "examples").as_posix()
              for path in (ROOT / "examples").rglob("*.nxl")}
    if set(entries) != actual:
        print(f"Manifest missing: {sorted(actual - set(entries))}")
        print(f"Manifest stale: {sorted(set(entries) - actual)}")
        return 1
    failed = 0
    compiled = rejected = executed = experimental = 0
    with tempfile.TemporaryDirectory(prefix="nexa-examples-") as directory:
        for index, (name, expectation) in enumerate(sorted(entries.items())):
            out = Path(directory) / f"{index}.ll"
            command = [sys.executable, str(ROOT / "bootstrap/main.py"),
                       str(ROOT / "examples" / name), "--opt", "0", "--out", str(out)]
            try:
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
                output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
                if expectation["status"] in {"reject", "experimental"}:
                    if result.returncode == 0 or out.exists() or not re.search(expectation["diagnostic"], output):
                        raise AssertionError(f"Expected rejection /{expectation['diagnostic']}/, got:\n{output[-2500:]}")
                    if expectation["status"] == "experimental":
                        experimental += 1
                        print(f"EXPERIMENTAL {name}: {expectation['reason']}")
                    else:
                        rejected += 1
                    continue
                if expectation["status"] != "compile":
                    raise AssertionError(f"Unknown manifest status: {expectation['status']}")
                if result.returncode or not out.is_file():
                    raise AssertionError(output[-2500:])
                compiled += 1
                if "run" in expectation:
                    executable = Path(directory) / (f"{index}.exe" if os.name == "nt" else str(index))
                    link = subprocess.run(_native_link_cmd(str(out), str(executable), "O0"),
                                          capture_output=True, text=True, timeout=60)
                    if link.returncode:
                        raise AssertionError(link.stderr[-2500:])
                    run = subprocess.run([str(executable)], cwd=directory,
                                         capture_output=True, text=True, timeout=15)
                    required = expectation["run"]
                    if run.returncode != required.get("exit", 0):
                        raise AssertionError(f"Exit {run.returncode}: {run.stdout[-1500:]}\n{run.stderr[-1500:]}")
                    for fragment in required.get("contains", []):
                        if fragment not in run.stdout:
                            raise AssertionError(f"Missing {fragment!r} in:\n{run.stdout[-2500:]}")
                    executed += 1
            except (AssertionError, OSError, subprocess.TimeoutExpired) as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    print(f"Examples: {compiled} compiled, {rejected} expected errors, "
          f"{executed} executed, {experimental} experimental, {failed} failures")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

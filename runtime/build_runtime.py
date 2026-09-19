"""Build host runtime libraries from source into an ignored artifact directory."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent


def build_runtime(name="turboquant", output_dir=None):
    sources = {"turboquant": [ROOT / "turboquant.c"],
               "nexa_async": [ROOT / "nexa_async.c"],
               "nexa_q4": [ROOT / "nexapack" / "q4.c"],
               "nexa_transformer": [ROOT / "nexapack" / "q4.c", ROOT / "nexapack" / "transformer.c"],
               "nexa_tq_attention": [ROOT / "turboquant.c", ROOT / "nexapack" / "tq_attention.c"]}
    if name not in sources:
        raise ValueError(f"Unknown runtime: {name}")
    output_dir = Path(output_dir or ROOT.parent / "artifacts" / "build" / "runtime").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    filename = f"{name}.dll" if system == "Windows" else f"lib{name}.{'dylib' if system == 'Darwin' else 'so'}"
    destination = output_dir / filename
    compiler = shlex.split(os.environ["CC"]) if os.environ.get("CC") else [shutil.which("clang") or shutil.which("cc") or "cc"]
    source = sources[name]
    with tempfile.TemporaryDirectory(prefix="nexa-runtime-", dir=output_dir) as tmp:
        candidate = Path(tmp) / filename
        cmd = compiler + ["-std=c11", "-O2"]
        if system != "Windows":
            cmd.append("-fPIC")
        cmd += ["-dynamiclib" if system == "Darwin" else "-shared", *map(str, source), "-o", str(candidate)]
        if system != "Windows" and name in ("turboquant", "nexa_tq_attention"):
            cmd += ["-lm", "-pthread"]
        elif system != "Windows" and name in ("nexa_q4", "nexa_transformer"):
            cmd.append("-lm")
        if system == "Windows" and name == "nexa_async":
            cmd.append("-lws2_32")
        subprocess.run(cmd, check=True)
        try:
            os.replace(candidate, destination)
        except PermissionError:
            if system != "Windows":
                raise
            # A DLL loaded by this or another process cannot be replaced on
            # Windows. Publish the newly compiled library at a distinct stable
            # path; the build's TemporaryDirectory is removed before return.
            destination = output_dir / f"{name}-{uuid.uuid4().hex}.dll"
            os.replace(candidate, destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for library in ("turboquant", "nexa_async", "nexa_q4", "nexa_transformer", "nexa_tq_attention"):
        print(build_runtime(library, args.output_dir))

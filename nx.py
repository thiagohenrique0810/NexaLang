import argparse
import os
import subprocess
import sys


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP_MAIN = os.path.join(REPO_ROOT, "bootstrap", "main.py")
DEV_ARTIFACTS = os.path.join(REPO_ROOT, "dev", "artifacts")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    p = subprocess.run(cmd)
    return int(p.returncode)


def _python() -> list[str]:
    return [sys.executable]


def cmd_build(args: argparse.Namespace) -> int:
    _ensure_dir(DEV_ARTIFACTS)
    out = args.out or (os.path.join(DEV_ARTIFACTS, "output.exe") if args.target == "native" else None)
    ll_out = args.ll_out or (os.path.join(DEV_ARTIFACTS, "output.ll") if args.emit == "ll" else os.path.join(DEV_ARTIFACTS, "output.ll"))
    spv_out = args.spv_out or (os.path.join(DEV_ARTIFACTS, "output.spv") if args.emit == "spv" else None)

    if args.target == "native":
        # Emit LLVM IR
        rc = _run(_python() + [BOOTSTRAP_MAIN, args.file, "--target", "native", "--emit", "ll", "--out", ll_out])
        if rc != 0:
            return rc
        # Link
        if not args.no_link:
            return _run(["clang", ll_out, "-o", out])
        return 0

    # SPIR-V
    if args.emit == "ll":
        return _run(
            _python()
            + [
                BOOTSTRAP_MAIN,
                args.file,
                "--target",
                "spirv",
                "--emit",
                "ll",
                "--out",
                ll_out,
                "--spirv-env",
                args.spirv_env,
                "--spirv-local-size",
                args.spirv_local_size,
            ]
        )

    return _run(
        _python()
        + [
            BOOTSTRAP_MAIN,
            args.file,
            "--target",
            "spirv",
            "--emit",
            "spv",
            "--out",
            spv_out,
            "--spirv-env",
            args.spirv_env,
            "--spirv-local-size",
            args.spirv_local_size,
            "--spirv-vulkan-var-pointers",
            args.spirv_vulkan_var_pointers,
            "--spirv-vulkan-descriptors",
            args.spirv_vulkan_descriptors,
            "--spirv-vulkan-descriptor-set",
            str(args.spirv_vulkan_descriptor_set),
            "--spirv-vulkan-binding-base",
            str(args.spirv_vulkan_binding_base),
        ]
    )


def cmd_run(args: argparse.Namespace) -> int:
    # Build native and run
    _ensure_dir(DEV_ARTIFACTS)
    
    cmd = _python() + [BOOTSTRAP_MAIN, args.file, "--target", "native"]
    
    if args.jit:
        cmd.append("--run-jit")
        return _run(cmd)

    exe = args.exe or os.path.join(DEV_ARTIFACTS, "output.exe")
    ll_out = args.ll_out or os.path.join(DEV_ARTIFACTS, "output.ll")
    
    cmd.extend(["--emit", "ll", "--out", ll_out])
    
    rc = _run(cmd)
    if rc != 0:
        return rc
    rc = _run(["clang", ll_out, "-o", exe])
    if rc != 0:
        return rc
    return _run([exe])


def cmd_val(args: argparse.Namespace) -> int:
    if args.kind == "spirv":
        return _run(["spirv-val", args.file])
    return 2


def cmd_examples(args: argparse.Namespace) -> int:
    examples_dir = os.path.join(REPO_ROOT, "examples")
    files = sorted([f for f in os.listdir(examples_dir) if f.endswith(".nxl")])
    for f in files:
        print(f)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="nx", description="NexaLang helper CLI (bootstrap)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build a .nxl file (native or SPIR-V)")
    p_build.add_argument("file", help="Input .nxl")
    p_build.add_argument("--target", choices=["native", "spirv"], default="native")
    p_build.add_argument("--emit", choices=["ll", "spv"], default="spv", help="(spirv) emit format")
    p_build.add_argument("--no-link", action="store_true", help="(native) only emit .ll, don't run clang")
    p_build.add_argument("--out", default=None, help="(native) output exe path")
    p_build.add_argument("--ll-out", default=None, help="output .ll path")
    p_build.add_argument("--spv-out", default=None, help="output .spv path")

    # SPIR-V flags
    p_build.add_argument("--spirv-env", choices=["opencl", "vulkan"], default="opencl")
    p_build.add_argument("--spirv-local-size", default="1,1,1")
    p_build.add_argument("--spirv-vulkan-var-pointers", choices=["on", "off"], default="on")
    p_build.add_argument("--spirv-vulkan-descriptors", choices=["on", "off"], default="on")
    p_build.add_argument("--spirv-vulkan-descriptor-set", type=int, default=0)
    p_build.add_argument("--spirv-vulkan-binding-base", type=int, default=0)
    p_build.set_defaults(func=cmd_build)

    p_run = sub.add_parser("run", help="Build native and run")
    p_run.add_argument("file", help="Input .nxl")
    p_run.add_argument("--exe", default=None, help="Output exe path")
    p_run.add_argument("--ll-out", default=None, help="Output .ll path")
    p_run.add_argument("--jit", action="store_true", help="Run using JIT (no clang required)")
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("val", help="Validate artifacts (SPIR-V)")
    p_val.add_argument("kind", choices=["spirv"])
    p_val.add_argument("file", help="Path to artifact (e.g. .spv)")
    p_val.set_defaults(func=cmd_val)

    p_ex = sub.add_parser("examples", help="List examples")
    p_ex.set_defaults(func=cmd_examples)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())



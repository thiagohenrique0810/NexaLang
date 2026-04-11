import argparse
import os
import platform
import subprocess
import sys
import json


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP_MAIN = os.path.join(REPO_ROOT, "bootstrap", "main.py")
DEV_ARTIFACTS = os.path.join(REPO_ROOT, "dev", "artifacts")

_EXE_EXT = ".exe" if platform.system() == "Windows" else ""

__version__ = "0.5.0"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_project_config():
    if os.path.exists("nexa.json"):
        try:
            with open("nexa.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading nexa.json: {e}")
    return None


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    p = subprocess.run(cmd)
    return int(p.returncode)


def _python() -> list[str]:
    return [sys.executable]


def _clang():
    # Try to find clang across platforms
    import shutil
    import platform

    # 1. Check if clang is on PATH (works on all platforms)
    clang_path = shutil.which("clang")
    if clang_path:
        return clang_path

    # 2. Platform-specific fallback paths
    system = platform.system()
    candidates = []

    if system == "Windows":
        candidates = [
            os.path.join(os.environ.get("USERPROFILE", ""), "scoop", "apps", "llvm", "current", "bin", "clang.exe"),
            "C:\\Program Files\\LLVM\\bin\\clang.exe",
            "C:\\Program Files (x86)\\LLVM\\bin\\clang.exe",
        ]
    elif system == "Darwin":
        candidates = [
            "/usr/local/opt/llvm/bin/clang",  # Homebrew Intel
            "/opt/homebrew/opt/llvm/bin/clang",  # Homebrew Apple Silicon
            "/Library/Developer/CommandLineTools/usr/bin/clang",
            "/usr/bin/clang",
        ]
    else:  # Linux
        # Try versioned clang (clang-15, clang-16, etc.)
        for ver in range(20, 13, -1):
            versioned = shutil.which(f"clang-{ver}")
            if versioned:
                return versioned
        candidates = [
            "/usr/bin/clang",
            "/usr/local/bin/clang",
        ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    print("Warning: clang not found. Install LLVM/Clang for native compilation.")
    return "clang"  # Fallback — will fail with a clear error


def _uses_turboquant(ll_path: str) -> bool:
    """Return True if generated LLVM IR references TurboQuant runtime symbols."""
    if not os.path.exists(ll_path):
        return False
    try:
        with open(ll_path, "r", encoding="utf-8", errors="ignore") as f:
            ir_text = f.read()
        return (
            "tq_create" in ir_text
            or "tq_quantize" in ir_text
            or "tq_dequantize" in ir_text
            or "tq_mse" in ir_text
            or "tq_destroy" in ir_text
        )
    except OSError:
        return False


def _turboquant_link_args() -> list[str]:
    """Return linker flags for TurboQuant, preferring static linking when available."""
    runtime_dir = os.path.join(REPO_ROOT, "runtime")
    static_lib = os.path.join(runtime_dir, "libturboquant.a")
    dyn_lib = os.path.join(runtime_dir, "libturboquant.dylib")

    if os.path.exists(static_lib):
        args = [static_lib]
    elif os.path.exists(dyn_lib):
        args = ["-L", runtime_dir, "-lturboquant"]
    else:
        args = ["-L", runtime_dir, "-lturboquant"]

    # TurboQuant uses math + pthread APIs.
    if platform.system() != "Windows":
        args.extend(["-lm", "-lpthread"])
    return args


def _native_link_cmd(ll_out: str, exe_out: str, opt: str) -> list[str]:
    cmd = [_clang(), ll_out, f"-{opt}", "-o", exe_out]
    if _uses_turboquant(ll_out):
        cmd.extend(_turboquant_link_args())
    return cmd


def cmd_build(args: argparse.Namespace) -> int:
    _ensure_dir(DEV_ARTIFACTS)
    
    file = args.file
    if not file:
        config = load_project_config()
        if config and "main" in config:
            file = config["main"]
        else:
            print("Error: No input file specified and no nexa.json project file found.")
            return 1

    out = args.out or (os.path.join(DEV_ARTIFACTS, f"output{_EXE_EXT}") if args.target == "native" else None)
    ll_out = args.ll_out or (os.path.join(DEV_ARTIFACTS, "output.ll") if args.emit == "ll" else os.path.join(DEV_ARTIFACTS, "output.ll"))
    spv_out = args.spv_out or (os.path.join(DEV_ARTIFACTS, "output.spv") if args.emit == "spv" else None)

    if args.target == "native":
        # Emit LLVM IR
        rc = _run(_python() + [BOOTSTRAP_MAIN, file, "--target", "native", "--emit", "ll", "--out", ll_out])
        if rc != 0:
            return rc
        # Link
        if not args.no_link:
            return _run(_native_link_cmd(ll_out, out, args.opt))
        return 0

    # SPIR-V
    if args.emit == "ll":
        return _run(
            _python()
            + [
                BOOTSTRAP_MAIN,
                file,
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
            file,
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
    
    file = args.file
    if not file:
        config = load_project_config()
        if config and "main" in config:
            file = config["main"]
        else:
            print("Error: No input file specified and no nexa.json project file found.")
            return 1

    cmd = _python() + [BOOTSTRAP_MAIN, file, "--target", "native"]
    
    if args.jit:
        cmd.append("--run-jit")
        return _run(cmd)

    exe = args.exe or os.path.join(DEV_ARTIFACTS, f"output{_EXE_EXT}")
    ll_out = args.ll_out or os.path.join(DEV_ARTIFACTS, "output.ll")
    
    cmd.extend(["--emit", "ll", "--out", ll_out])
    
    rc = _run(cmd)
    if rc != 0:
        return rc
    rc = _run(_native_link_cmd(ll_out, exe, getattr(args, "opt", "O0")))
    if rc != 0:
        return rc
    return _run([exe])


def cmd_val(args: argparse.Namespace) -> int:
    if args.kind == "spirv":
        return _run(["spirv-val", args.file])
    return 2


def cmd_test(args: argparse.Namespace) -> int:
    _ensure_dir(DEV_ARTIFACTS)
    
    file = args.file
    if not file:
        config = load_project_config()
        if config and "main" in config:
            file = config["main"]
        else:
            print("Error: No input file specified.")
            return 1

    ll_out = os.path.join(DEV_ARTIFACTS, "test.ll")
    exe_out = os.path.join(DEV_ARTIFACTS, f"test{_EXE_EXT}")

    # We need to tell the compiler to run tests
    rc = _run(_python() + [BOOTSTRAP_MAIN, file, "--target", "native", "--run-tests", "--out", ll_out])
    if rc != 0: return rc
    
    # Link
    rc = _run(_native_link_cmd(ll_out, exe_out, getattr(args, "opt", "O0")))
    if rc != 0: return rc
    
    # Run
    print("\n[RUNNING TESTS]")
    return _run([exe_out])


def cmd_examples(args: argparse.Namespace) -> int:
    examples_dir = os.path.join(REPO_ROOT, "examples")
    files = sorted([f for f in os.listdir(examples_dir) if f.endswith(".nxl")])
    for f in files:
        print(f)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="nxc", description="NexaLang compiler CLI")
    ap.add_argument("--version", "-v", action="version", version=f"NexaLang {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build a .nxl file (native or SPIR-V)")
    p_build.add_argument("file", nargs="?", help="Input .nxl (optional if nexa.json exists)")
    p_build.add_argument("--target", choices=["native", "spirv"], default="native")
    p_build.add_argument("--emit", choices=["ll", "spv"], default="spv", help="(spirv) emit format")
    p_build.add_argument("--no-link", action="store_true", help="(native) only emit .ll, don't run clang")
    p_build.add_argument("--out", default=None, help="(native) output exe path")
    p_build.add_argument("--ll-out", default=None, help="output .ll path")
    p_build.add_argument("--spv-out", default=None, help="output .spv path")
    p_build.add_argument("--opt", choices=["O0", "O1", "O2", "O3"], default="O0", help="Optimization level")

    # SPIR-V flags
    p_build.add_argument("--spirv-env", choices=["opencl", "vulkan"], default="opencl")
    p_build.add_argument("--spirv-local-size", default="1,1,1")
    p_build.add_argument("--spirv-vulkan-var-pointers", choices=["on", "off"], default="on")
    p_build.add_argument("--spirv-vulkan-descriptors", choices=["on", "off"], default="on")
    p_build.add_argument("--spirv-vulkan-descriptor-set", type=int, default=0)
    p_build.add_argument("--spirv-vulkan-binding-base", type=int, default=0)
    p_build.set_defaults(func=cmd_build)

    p_run = sub.add_parser("run", help="Build native and run")
    p_run.add_argument("file", nargs="?", help="Input .nxl (optional if nexa.json exists)")
    p_run.add_argument("--exe", default=None, help="Output exe path")
    p_run.add_argument("--ll-out", default=None, help="Output .ll path")
    p_run.add_argument("--jit", action="store_true", help="Run using JIT (no clang required)")
    p_run.add_argument("--opt", choices=["O0", "O1", "O2", "O3"], default="O0", help="Optimization level")
    p_run.set_defaults(func=cmd_run)

    p_test = sub.add_parser("test", help="Run tests (functions marked with @[test])")
    p_test.add_argument("file", nargs="?", help="Input .nxl")
    p_test.add_argument("--opt", choices=["O0", "O1", "O2", "O3"], default="O0", help="Optimization level")
    p_test.set_defaults(func=cmd_test)

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



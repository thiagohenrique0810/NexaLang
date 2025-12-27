import os
import platform
import shutil
import sys


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> int:
    _print_header("NexaLang SPIR-V Toolchain Check")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"cwd: {os.getcwd()}")

    _print_header("PATH checks")

    checks = [
        ("llc", "LLVM backend driver (preferred for .ll -> .spv if SPIR-V targets are enabled)"),
        ("llvm-as", "LLVM tools (turn .ll into .bc)"),
        ("llvm-spirv", "SPIRV-LLVM-Translator (turn .bc into .spv)"),
        ("spirv-val", "SPIR-V Tools (validate .spv)"),
        ("spirv-dis", "SPIR-V Tools (disassemble .spv)"),
        ("spirv-as", "SPIR-V Tools (assemble .spvasm -> .spv; used for Vulkan variable-pointers patching)"),
        ("clang", "LLVM/Clang (optional; useful for native builds and some SPIR-V workflows)"),
    ]

    found = {}
    for cmd, desc in checks:
        path = which(cmd)
        found[cmd] = path
        status = "OK" if path else "MISSING"
        print(f"{cmd:10} {status:8}  {desc}")
        if path:
            print(f"{'':10} {'':8}  -> {path}")

    _print_header("What you need to emit .spv from this repo")
    has_llc = bool(found.get("llc"))
    has_translator = bool(found.get("llvm-as")) and bool(found.get("llvm-spirv"))

    if has_llc or has_translator:
        print("OK: Required tools found. You can run:")
        print("  python bootstrap\\main.py examples\\gpu_dispatch.nxl --target spirv --emit spv --out output.spv")
        if found.get("spirv-val"):
            print("  spirv-val output.spv")
        if found.get("spirv-as"):
            print("\nNote: For `--spirv-env vulkan`, `spirv-as` enables an optional patch to add VariablePointers capabilities when needed.")
        return 0

    print("Missing required tools: need either `llc` OR (`llvm-as` + `llvm-spirv`).")
    print("\nInstall guidance (high level):")
    print("- Preferred: Install LLVM with SPIR-V targets enabled so `llc` can emit `.spv`.")
    print("- Fallback: Install LLVM tools so `llvm-as` is available in PATH, and SPIRV-LLVM-Translator so `llvm-spirv` is available.")
    print("- (Optional) Install SPIR-V Tools for `spirv-val`/`spirv-dis`.")
    print("\nAfter installing, re-run this script and then run:")
    print("  python bootstrap\\main.py examples\\gpu_dispatch.nxl --target spirv --emit spv --out output.spv")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())



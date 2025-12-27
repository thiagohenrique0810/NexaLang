import os
import shutil
import subprocess
import tempfile


def _which(name: str) -> str | None:
    return shutil.which(name)


def emit_spirv_from_llvm_ir(llvm_ir: str, out_spv_path: str) -> None:
    """
    Convert LLVM IR (text) -> SPIR-V binary using external tools when available.

    Preferred (LLVM native SPIR-V backend):
    - llc (with spirv64 target)

    Fallback (SPIRV-LLVM-Translator):
    - llvm-as   (to turn .ll into .bc)
    - llvm-spirv (to turn .bc into .spv)
    """
    llc = _which("llc")
    llvm_as = _which("llvm-as")
    llvm_spirv = _which("llvm-spirv")

    if not llc and (not llvm_as or not llvm_spirv):
        missing = []
        if not llc:
            missing.append("llc (with spirv64 target)")
        if not llvm_as:
            missing.append("llvm-as")
        if not llvm_spirv:
            missing.append("llvm-spirv")
        raise RuntimeError(
            "SPIR-V emission requires tools not found in PATH: "
            + ", ".join(missing)
            + ".\n"
            + "Install options:\n"
            + "- LLVM (preferred): llc with SPIR-V targets enabled\n"
            + "- Or SPIRV-LLVM-Translator: llvm-as + llvm-spirv\n"
        )

    out_spv_path = os.path.abspath(out_spv_path)
    os.makedirs(os.path.dirname(out_spv_path) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        ll_path = os.path.join(td, "module.ll")
        bc_path = os.path.join(td, "module.bc")
        with open(ll_path, "w", encoding="utf-8") as f:
            f.write(llvm_ir)

        # Preferred: llc module.ll -> out.spv
        if llc:
            subprocess.check_call(
                [llc, "-mtriple=spirv64-unknown-unknown", ll_path, "-filetype=obj", "-o", out_spv_path]
            )
            return

        # Fallback: llvm-as + llvm-spirv
        subprocess.check_call([llvm_as, ll_path, "-o", bc_path])
        subprocess.check_call([llvm_spirv, bc_path, "-o", out_spv_path])



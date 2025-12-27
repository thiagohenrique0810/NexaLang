import os
import shutil
import subprocess
import tempfile


def _which(name: str) -> str | None:
    return shutil.which(name)


def emit_spirv_from_llvm_ir(llvm_ir: str, out_spv_path: str) -> None:
    """
    Convert LLVM IR (text) -> SPIR-V binary using external tools when available.

    Requires:
    - llvm-as   (to turn .ll into .bc)
    - llvm-spirv (SPIRV-LLVM-Translator)
    """
    llvm_as = _which("llvm-as")
    llvm_spirv = _which("llvm-spirv")

    missing = []
    if not llvm_as:
        missing.append("llvm-as")
    if not llvm_spirv:
        missing.append("llvm-spirv")

    if missing:
        raise RuntimeError(
            "SPIR-V emission requires external tools not found in PATH: "
            + ", ".join(missing)
            + ".\n"
            + "Install options:\n"
            + "- SPIRV-LLVM-Translator (provides llvm-spirv)\n"
            + "- LLVM tools (provides llvm-as)\n"
        )

    out_spv_path = os.path.abspath(out_spv_path)
    os.makedirs(os.path.dirname(out_spv_path) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        ll_path = os.path.join(td, "module.ll")
        bc_path = os.path.join(td, "module.bc")
        with open(ll_path, "w", encoding="utf-8") as f:
            f.write(llvm_ir)

        # llvm-as module.ll -o module.bc
        subprocess.check_call([llvm_as, ll_path, "-o", bc_path])

        # llvm-spirv module.bc -o out.spv
        subprocess.check_call([llvm_spirv, bc_path, "-o", out_spv_path])



import os
import shutil
import subprocess
import tempfile


def _which(name: str) -> str | None:
    return shutil.which(name)

def _patch_vulkan_variable_pointers(spv_path: str) -> bool:
    """
    Best-effort patch for Vulkan env when LLVM SPIR-V backend emits OpPtrAccessChain
    but does not add required capabilities/extensions for variable pointers.

    Requires: spirv-dis + spirv-as (SPIRV-Tools).
    """
    spirv_dis = _which("spirv-dis")
    spirv_as = _which("spirv-as")
    if not spirv_dis or not spirv_as:
        return False

    with tempfile.TemporaryDirectory() as td:
        asm_path = os.path.join(td, "module.spvasm")
        patched_path = os.path.join(td, "module_patched.spv")

        subprocess.check_call([spirv_dis, spv_path, "-o", asm_path])

        with open(asm_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        # Only patch if PtrAccessChain is present.
        if not any("OpPtrAccessChain" in ln for ln in lines):
            return False

        # Insert extension/capabilities if not present.
        want_ext = 'OpExtension "SPV_KHR_variable_pointers"'
        want_caps = [
            "OpCapability VariablePointers",
            "OpCapability VariablePointersStorageBuffer",
        ]

        has_ext = any(ln.strip() == want_ext for ln in lines)
        has_caps = {cap: any(ln.strip() == cap for ln in lines) for cap in want_caps}

        caps_to_insert = [cap for cap in want_caps if not has_caps[cap]]
        ext_to_insert = [] if has_ext else [want_ext]

        if not caps_to_insert and not ext_to_insert:
            return False

        # SPIR-V layout rules:
        # - OpCapability must appear before OpExtension.
        # - Both must appear before OpMemoryModel.
        first_ext = None
        mm = None
        last_cap = None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith("OpCapability "):
                last_cap = i
            elif first_ext is None and (s.startswith("OpExtension ") or s.startswith("OpExtInstImport ") or s.startswith("OpMemoryModel ")):
                first_ext = i
            if s.startswith("OpMemoryModel "):
                mm = i
                break

        # Insert capabilities after last capability, but before first extension/memory model.
        cap_insert_at = None
        if last_cap is not None:
            cap_insert_at = last_cap + 1
        elif first_ext is not None:
            cap_insert_at = first_ext
        elif mm is not None:
            cap_insert_at = mm
        else:
            cap_insert_at = 0
            while cap_insert_at < len(lines) and lines[cap_insert_at].lstrip().startswith(";"):
                cap_insert_at += 1

        patched_lines = lines[:cap_insert_at] + caps_to_insert + lines[cap_insert_at:]

        # After inserting capabilities, place the extension right after the (new) capability block.
        if ext_to_insert:
            # Find new last capability index
            new_last_cap = None
            for i, ln in enumerate(patched_lines):
                if ln.strip().startswith("OpCapability "):
                    new_last_cap = i
            ext_insert_at = (new_last_cap + 1) if new_last_cap is not None else cap_insert_at
            patched_lines = patched_lines[:ext_insert_at] + ext_to_insert + patched_lines[ext_insert_at:]
        with open(asm_path, "w", encoding="utf-8") as f:
            f.write("\n".join(patched_lines) + "\n")

        subprocess.check_call([spirv_as, asm_path, "-o", patched_path])

        # Overwrite original
        with open(patched_path, "rb") as src, open(spv_path, "wb") as dst:
            dst.write(src.read())
        return True

def _patch_vulkan_storagebuffer_access_chains(spv_path: str) -> bool:
    """
    Workaround for LLVM SPIR-V backend Vulkan path: it tends to emit OpPtrAccessChain
    for StorageBuffer runtime-array indexing with an extra leading index (reg2mem_alloca_point),
    which triggers spirv-val errors about ArrayStride.

    This patch rewrites:
      OpPtrAccessChain ... %__nexa_*_data %reg2mem_alloca_point %idx
    into:
      OpAccessChain ... %__nexa_*_data %idx

    Requires: spirv-dis + spirv-as.
    """
    spirv_dis = _which("spirv-dis")
    spirv_as = _which("spirv-as")
    if not spirv_dis or not spirv_as:
        return False

    with tempfile.TemporaryDirectory() as td:
        asm_path = os.path.join(td, "module.spvasm")
        patched_path = os.path.join(td, "module_patched.spv")

        subprocess.check_call([spirv_dis, spv_path, "-o", asm_path])
        with open(asm_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        changed = False
        out_lines: list[str] = []
        for ln in lines:
            if (
                "OpPtrAccessChain" in ln
                and "__nexa_" in ln
                and "_data" in ln
                and "%reg2mem_alloca_point" in ln
            ):
                ln2 = ln.replace("OpPtrAccessChain", "OpAccessChain")
                # drop the extra leading 0 index
                ln2 = ln2.replace(" %reg2mem_alloca_point ", " ")
                out_lines.append(ln2)
                changed = True
            else:
                out_lines.append(ln)

        if not changed:
            return False

        # If no PtrAccessChain remains, we can also drop variable pointers caps/ext (optional cleanup).
        if not any("OpPtrAccessChain" in ln for ln in out_lines):
            out_lines = [
                ln for ln in out_lines
                if ("OpCapability VariablePointers" not in ln)
                and ("OpCapability VariablePointersStorageBuffer" not in ln)
                and ('OpExtension "SPV_KHR_variable_pointers"' not in ln)
            ]

        with open(asm_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")

        subprocess.check_call([spirv_as, asm_path, "-o", patched_path])
        with open(patched_path, "rb") as src, open(spv_path, "wb") as dst:
            dst.write(src.read())
        return True

def _patch_vulkan_descriptor_bindings(spv_path: str, descriptor_set: int, binding_base: int) -> bool:
    """
    Best-effort patch: add DescriptorSet/Binding decorations to interface variables
    with names starting with "__nexa_" (kernel args lowered to globals).

    Requires: spirv-dis + spirv-as.
    """
    spirv_dis = _which("spirv-dis")
    spirv_as = _which("spirv-as")
    if not spirv_dis or not spirv_as:
        return False

    with tempfile.TemporaryDirectory() as td:
        asm_path = os.path.join(td, "module.spvasm")
        patched_path = os.path.join(td, "module_patched.spv")

        subprocess.check_call([spirv_dis, spv_path, "-o", asm_path])
        with open(asm_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        # Collect ids for __nexa_* interface vars based on OpName.
        # Example: OpName %__nexa_fill_buf_data "__nexa_fill_buf_data"
        import re
        name_re = re.compile(r'^\s*OpName\s+(%[A-Za-z0-9_\.]+)\s+"(__nexa_[^"]+)"\s*$')
        nexa: list[tuple[str, str]] = []  # (id, name)
        for ln in lines:
            m = name_re.match(ln)
            if m:
                nexa.append((m.group(1), m.group(2)))

        # Deduplicate, stable order.
        seen = set()
        nexa = [(sid, sname) for (sid, sname) in nexa if not (sid in seen or seen.add(sid))]
        if not nexa:
            return False

        # Deterministic binding order:
        # Group by kernel/arg, and order suffixes as: len, data, other.
        # Names we generate:
        #   __nexa_{kernel}_{arg}_len
        #   __nexa_{kernel}_{arg}_data
        #   __nexa_{kernel}_{arg}        (fallback)
        def sort_key(item: tuple[str, str]) -> tuple[str, str, int, str]:
            _sid, sname = item
            # strip prefix
            rest = sname[len("__nexa_"):] if sname.startswith("__nexa_") else sname
            kind = ""
            base = rest
            if rest.endswith("_len"):
                kind = "len"
                base = rest[:-4]
            elif rest.endswith("_data"):
                kind = "data"
                base = rest[:-5]
            # base is "{kernel}_{arg}" (best-effort)
            # split first '_' as kernel name; if not present keep whole as kernel.
            if "_" in base:
                kernel, arg = base.split("_", 1)
            else:
                kernel, arg = base, ""
            kind_order = 2
            if kind == "len":
                kind_order = 0
            elif kind == "data":
                kind_order = 1
            return (kernel, arg, kind_order, sname)

        nexa.sort(key=sort_key)

        # Determine insertion point: before first type declaration.
        insert_at = None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith("%") and " = OpType" in s:
                insert_at = i
                break
        if insert_at is None:
            return False

        indent = "               "
        new_decorations: list[str] = []
        binding = binding_base
        for sid, _name in nexa:
            new_decorations.append(f"{indent}OpDecorate {sid} DescriptorSet {descriptor_set}")
            new_decorations.append(f"{indent}OpDecorate {sid} Binding {binding}")
            binding += 1

        # Avoid duplicates if already decorated.
        existing = set(ln.strip() for ln in lines if ln.strip().startswith("OpDecorate "))
        filtered = []
        for ln in new_decorations:
            if ln.strip() not in existing:
                filtered.append(ln)

        if not filtered:
            return False

        patched_lines = lines[:insert_at] + filtered + lines[insert_at:]
        with open(asm_path, "w", encoding="utf-8") as f:
            f.write("\n".join(patched_lines) + "\n")

        subprocess.check_call([spirv_as, asm_path, "-o", patched_path])
        with open(patched_path, "rb") as src, open(spv_path, "wb") as dst:
            dst.write(src.read())
        return True


def emit_spirv_from_llvm_ir(
    llvm_ir: str,
    out_spv_path: str,
    spirv_env: str = "opencl",
    vulkan_variable_pointers: bool = True,
    vulkan_descriptors: bool = True,
    vulkan_descriptor_set: int = 0,
    vulkan_binding_base: int = 0,
) -> None:
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

        # Preferred: llc module.ll -> out.spv (respect module triple; don't override -mtriple)
        if llc:
            subprocess.check_call(
                [llc, ll_path, "-filetype=obj", "-o", out_spv_path]
            )
            if spirv_env == "vulkan":
                # Apply StorageBuffer access-chain workaround first (it may remove the need for variable pointers).
                _patch_vulkan_storagebuffer_access_chains(out_spv_path)
                if vulkan_variable_pointers:
                    _patch_vulkan_variable_pointers(out_spv_path)
                if vulkan_descriptors:
                    _patch_vulkan_descriptor_bindings(out_spv_path, vulkan_descriptor_set, vulkan_binding_base)
            return

        # Fallback: llvm-as + llvm-spirv
        subprocess.check_call([llvm_as, ll_path, "-o", bc_path])
        subprocess.check_call([llvm_spirv, bc_path, "-o", out_spv_path])



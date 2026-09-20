"""LLVM JIT with explicit symbol resolution before executable code is finalized."""
import ctypes
import ctypes.util
import os
from pathlib import Path
import re
import sys

import llvmlite.binding as llvm


def run_jit(llvm_ir):
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    defined = {function.name for function in mod.functions if not function.is_declaration}
    calls = re.findall(r'\b(?:call|invoke)\b[^\n@]*@(?:"([^"\n]+)"|([\w.$]+))', llvm_ir)
    required = {quoted or plain for quoted, plain in calls} - defined
    if "__nexa_gpu_dispatch" in required:
        raise RuntimeError("GPU dispatch is unavailable in JIT mode; use a native GPU backend")

    libraries = [ctypes.CDLL("msvcrt" if os.name == "nt" else None)]
    runtime_names = []
    if any(name.startswith("tq_") for name in required):
        runtime_names.append("turboquant")
    if any(name.startswith("nexa_qpack_") for name in required):
        runtime_names.append("nexa_q4")
    if any(name.startswith("__nexa_") for name in required) or (os.name == "nt" and "sched_yield" in required):
        runtime_names.append("nexa_async")
    if runtime_names:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
        from build_runtime import build_runtime
        libraries.extend(ctypes.CDLL(str(build_runtime(name))) for name in runtime_names)
    for prefix, library_name in (("curl_", "curl"), ("sqlite3_", "sqlite3")):
        if any(name.startswith(prefix) for name in required):
            path = ctypes.util.find_library(library_name)
            if not path:
                raise RuntimeError(f"Required JIT library not found: {library_name}")
            libraries.append(ctypes.CDLL(path))

    for name in required:
        if name.startswith("llvm."):
            continue
        for library in libraries:
            symbol = getattr(library, name, None)
            if symbol is not None:
                llvm.add_symbol(name, ctypes.cast(symbol, ctypes.c_void_p).value)
                break
        else:
            raise RuntimeError(f"Unresolved JIT symbol: {name}")

    if "main" not in defined:
        raise RuntimeError("'main' function not found in JIT module")
    main = mod.get_function("main")
    argc = len(list(main.arguments))
    if argc not in (0, 2):
        raise RuntimeError("JIT main must take no arguments or (argc, argv)")
    result_type = None if str(main.global_value_type.get_function_return()) == "void" else ctypes.c_int32
    machine = llvm.Target.from_default_triple().create_target_machine()
    with llvm.create_mcjit_compiler(mod, machine) as engine:
        engine.finalize_object()
        engine.run_static_constructors()
        try:
            address = engine.get_function_address("main")
            if argc == 2:
                argv = (ctypes.c_char_p * 2)(b"nxc", None)
                run = ctypes.CFUNCTYPE(result_type, ctypes.c_int32, ctypes.POINTER(ctypes.c_char_p))(address)
                return run(1, argv) or 0
            return ctypes.CFUNCTYPE(result_type)(address)() or 0
        finally:
            # The program wrote to descriptor 1 through the C runtime's buffer,
            # which this process does not otherwise touch. Draining it here is
            # what keeps a later Python line from landing inside its output.
            libraries[0].fflush(None)
            engine.run_static_destructors()

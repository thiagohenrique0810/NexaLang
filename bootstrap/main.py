import sys
import os
import argparse
import tempfile
from lexer import Lexer
import n_parser
from n_parser import ModDecl, FunctionDef, StructDef, EnumDef, ImplDef, CallExpr, IntegerLiteral, StringLiteral, ReturnStmt, UseStmt
from codegen import CodeGen
from errors import CompilerError
from modules import resolve_modules


DEFAULT_BUILD_DIR = os.path.join("artifacts", "build")
DEFAULT_LL_PATH = os.path.join(DEFAULT_BUILD_DIR, "output.ll")
DEFAULT_SPV_PATH = os.path.join(DEFAULT_BUILD_DIR, "output.spv")


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_artifact(path, content):
    """Publish a complete artifact only after compilation and validation succeed."""
    _ensure_parent_dir(path)
    directory = os.path.dirname(os.path.abspath(path))
    candidate = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as f:
            candidate = f.name
            f.write(str(content))
        os.replace(candidate, path)
    finally:
        if candidate and os.path.exists(candidate):
            os.unlink(candidate)


def _build_test_runner(test_names):
    """Build a proper test main function body with error handling."""
    body = []
    total = len(test_names)
    body.append(CallExpr("print", [StringLiteral(f"Running {total} test(s)...\n")]))
    
    for test_name in test_names:
        body.append(CallExpr("print", [StringLiteral(f"  test {test_name} ... ")]))
        body.append(CallExpr(test_name, []))
        body.append(CallExpr("print", [StringLiteral("PASSED\n")]))
    
    body.append(CallExpr("print", [StringLiteral(f"\n{total} test(s) passed.\n")]))
    body.append(ReturnStmt(IntegerLiteral(0)))
    
    # Set required attributes on each node
    for node in body:
        node.line = 0
        node.column = 0
    
    return body

def _compile():
    ap = argparse.ArgumentParser(prog="nxc (bootstrap)", add_help=True)
    ap.add_argument("file", help="Input .nxl file")
    ap.add_argument("--target", choices=["native", "spirv"], default="native", help="Compilation target")
    ap.add_argument("--emit", choices=["ll", "spv"], default="ll", help="Emit format (spv requires external tools)")
    ap.add_argument("--spirv-env", choices=["opencl", "vulkan"], default="opencl", help="SPIR-V environment (only for --target spirv)")
    ap.add_argument("--spirv-local-size", default="1,1,1", help="Vulkan compute local size (x,y,z) when --spirv-env vulkan")
    ap.add_argument("--spirv-vulkan-var-pointers", choices=["on", "off"], default="on", help="(vulkan) Try to enable variable pointers (requires spirv-as for patching).")
    ap.add_argument("--spirv-vulkan-descriptors", choices=["on", "off"], default="on", help="(vulkan) Patch DescriptorSet/Binding decorations for __nexa_* interface vars (requires spirv-as).")
    ap.add_argument("--spirv-vulkan-descriptor-set", type=int, default=0, help="(vulkan) DescriptorSet number to use for kernel args")
    ap.add_argument("--spirv-vulkan-binding-base", type=int, default=0, help="(vulkan) First binding number to assign to kernel args")
    ap.add_argument("--run-jit", action="store_true", help="Run the generated code immediately using JIT (no external compiler required)")
    ap.add_argument("--run-tests", action="store_true", help="Find and run all functions marked with @[test]")
    ap.add_argument("--out", default=None, help="Output path (default: artifacts/build/output.ll or artifacts/build/output.spv)")
    ap.add_argument("--emit-mir", action="store_true", help="Emit MIR (Mid-level IR) for debugging/optimization analysis")
    ap.add_argument("--opt", choices=["0", "1", "2", "3"], default="1", help="Optimization level (0=none, 1=basic, 2=aggressive, 3=max)")
    ap.add_argument("--quantize-gpu", type=int, default=0, choices=[0, 1, 2, 3, 4], help="Experimental automatic GPU quantization is disabled; only 0 is supported")
    args = ap.parse_args()
    if args.emit == "spv" and args.target != "spirv":
        ap.error("--emit spv requires --target spirv")
    if args.run_jit and args.target != "native":
        ap.error("--run-jit requires --target native")

    filepath = args.file
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    # 1. Lexing
    lexer = Lexer(source)
    tokens = lexer.tokenize()

    # 2. Parsing
    p = n_parser.Parser(tokens)
    ast = p.parse()
    
    # 2.5 Resolve Modules
    ast = resolve_modules(ast, os.path.dirname(os.path.abspath(filepath)))

    # 3. Semantic Analysis
    from semantic import SemanticAnalyzer
    analyzer = SemanticAnalyzer()
    analyzer.current_dir = os.path.dirname(os.path.abspath(filepath))
    analyzer.current_file_path = os.path.basename(filepath)
    
    try:
        analyzer.analyze(ast)
    except CompilerError as e:
        code_str = f" [{e.error_code}]" if e.error_code else ""
        print(f"\033[31mError{code_str}: {e.message}\033[0m")
        if e.line:
            lines = source.splitlines()
            if 0 <= e.line - 1 < len(lines):
                 print(f"  --> {filepath}:{e.line}:{e.column}")
                 print(f"   |")
                 print(f"{e.line:3} | {lines[e.line-1]}")
                 print(f"   | {' ' * (e.column-1)}^")
        if getattr(e, 'hint', None):
             print(f"  = help: {e.hint}")
        if e.error_code:
             print(f"  = docs: https://nexalang.org/errors/{e.error_code}")
        sys.exit(1)
    except Exception as e:
        print(f"[SEMANTIC ERROR] {e}")
        return 1
    # 3.5 Print Warnings
    if analyzer.warnings:
         lines = source.splitlines()
         for (msg, line, col) in analyzer.warnings:
              print(f"\033[33m[WARNING] {msg}\033[0m")
              if line and 0 <= line - 1 < len(lines):
                   print(f"  --> {filepath}:{line}:{col}")
                   print(f"   |")
                   print(f"{line:3} | {lines[line-1]}")
                   print(f"   | {' ' * (col-1)}^")

    # 4. Code Generation
    if args.run_tests:
        # Build test runner with proper AST construction
        test_body = _build_test_runner(analyzer.tests)
        
        test_main = FunctionDef("main", [], "i32", test_body)
        test_main.generics = []
        test_main.is_kernel = False
        test_main.is_async = False
        test_main.is_pub = False
        test_main.attributes = []
        test_main.is_vararg = False
        test_main.module = ""
        test_main.line = 0
        test_main.column = 0
        
        # Remove existing main if any
        ast = [n for n in ast if not (isinstance(n, FunctionDef) and n.name == "main")]
        ast.append(test_main)
        
        # Register in semantic analyzer
        analyzer.functions.add("main")
        analyzer.function_defs["main"] = [test_main]
        test_main.return_type = "i32"
        test_main.used = True
        test_main.mangled_name = "main"

    # 4.5 MIR (optional optimization layer)
    if args.emit_mir:
        from mir import MIRLowering, MIROptimizer, MIRPrinter
        lowering = MIRLowering()
        mir_module = lowering.lower(ast, struct_info=analyzer.structs, enum_info=analyzer.enums)
        
        optimizer = MIROptimizer()
        mir_module = optimizer.optimize(mir_module, level=min(int(args.opt), 2))
        
        printer = MIRPrinter()
        mir_text = printer.print_module(mir_module)
        
        mir_base = args.out or DEFAULT_LL_PATH
        mir_path = os.path.splitext(mir_base)[0] + ".mir"
        _ensure_parent_dir(mir_path)
        with open(mir_path, "w", encoding="utf-8") as f:
            f.write(mir_text)
        print(f"[MIR] Emitted to '{mir_path}' (opt={args.opt}, folded={optimizer.stats['constant_folded']}, eliminated={optimizer.stats['dead_eliminated']}, propagated={optimizer.stats['copies_propagated']})")

    spirv_env = args.spirv_env if args.target == "spirv" else "opencl"
    emit_kernels_only = args.target == "spirv" and args.emit == "spv"
    codegen = CodeGen(
        target=args.target,
        emit_kernels_only=emit_kernels_only,
        spirv_env=spirv_env,
        spirv_local_size=args.spirv_local_size,
        quantize_gpu=args.quantize_gpu,
    )
    llvm_ir = codegen.generate(ast)

    # Reject malformed IR before reporting a successful compilation.
    if args.target == "native":
        import llvmlite.binding as llvm_binding
        verified_module = llvm_binding.parse_assembly(str(llvm_ir))
        verified_module.verify()

    # ── LLVM Optimization Passes ──
    opt_level = int(args.opt)
    if opt_level >= 2 and args.target == "native":
        try:
            import llvmlite.binding as llvm_binding
            llvm_binding.initialize_native_target()
            llvm_binding.initialize_native_asmprinter()

            mod = llvm_binding.parse_assembly(str(llvm_ir))
            mod.verify()

            pto = llvm_binding.PipelineTuningOptions()
            pto.speed_level = min(opt_level, 3)
            pto.loop_vectorization = True
            pto.loop_unrolling = True
            pto.slp_vectorization = (opt_level >= 3)
            pto.loop_interleaving = True

            target = llvm_binding.Target.from_default_triple()
            tm = target.create_target_machine(opt=min(opt_level, 3))
            pb = llvm_binding.create_pass_builder(tm, pto)
            mpm = pb.getModulePassManager()
            mpm.run(mod, pb)

            llvm_ir = str(mod)
            print(f"[OPT] LLVM O{min(opt_level, 3)} passes applied")
        except Exception as e:
            raise RuntimeError(f"LLVM optimization failed: {e}") from e

    if args.run_jit and args.target == "native":
        from jit import run_jit
        print("[JIT] Starting JIT...")
        ret = run_jit(str(llvm_ir))
        print(f"[JIT] Finished with code {ret}")
        return ret

    if args.emit == "ll":
        out_path = args.out or DEFAULT_LL_PATH
        _ensure_parent_dir(out_path)
        _write_artifact(out_path, llvm_ir)
        print(f"\n[SUCCESS] LLVM IR compiled to '{out_path}'")
        if args.target == "native" and not args.run_jit:
            print(f"To build and run with runtime dependencies: nxc run {filepath}")
        return

    # emit spv
    out_path = args.out or DEFAULT_SPV_PATH
    _ensure_parent_dir(out_path)
    try:
        from spirv_backend import emit_spirv_from_llvm_ir
        emit_spirv_from_llvm_ir(
            llvm_ir,
            out_path,
            spirv_env=args.spirv_env,
            vulkan_variable_pointers=(args.spirv_vulkan_var_pointers == "on"),
            vulkan_descriptors=(args.spirv_vulkan_descriptors == "on"),
            vulkan_descriptor_set=args.spirv_vulkan_descriptor_set,
            vulkan_binding_base=args.spirv_vulkan_binding_base,
        )
        print(f"\n[SUCCESS] SPIR-V emitted to '{out_path}'")
    except Exception as e:
        ll_fallback = os.path.splitext(out_path)[0] + ".ll"
        _ensure_parent_dir(ll_fallback)
        with open(ll_fallback, "w", encoding="utf-8") as f:
             f.write(llvm_ir)
        print(f"[SPIR-V EMIT ERROR] {e}")
        print(f"[FALLBACK] Wrote LLVM IR to '{ll_fallback}' (use llvm-as + llvm-spirv to convert).")
        return 1

def main():
    try:
        return _compile() or 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"[COMPILE ERROR] {exc or 'Interrupted'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

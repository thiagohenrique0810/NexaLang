import sys
import os
import argparse
from lexer import Lexer
import n_parser
from n_parser import ModDecl, FunctionDef, StructDef, EnumDef, ImplDef
from codegen import CodeGen
from errors import CompilerError
import semantic

def mangle_ast(nodes, prefix):
    for node in nodes:
        if isinstance(node, (FunctionDef, StructDef, EnumDef)):
            if getattr(node, 'module', None):
                node.module = f"{prefix}::{node.module}"
            else:
                node.module = prefix
            
            node.name = f"{prefix}_{node.name}"
        elif isinstance(node, ImplDef):
            if getattr(node, 'module', None):
                node.module = f"{prefix}::{node.module}"
            else:
                node.module = prefix
            node.struct_name = f"{prefix}_{node.struct_name}"
            for method in node.methods:
                method.module = node.module

def resolve_modules(ast, base_dir):
    new_ast = []
    for node in ast:
        if isinstance(node, ModDecl):
            if node.body is not None:
                # Nested module block
                inner_ast = resolve_modules(node.body, base_dir)
                mangle_ast(inner_ast, node.name)
                new_ast.extend(inner_ast)
            else:
                # File-based module
                mod_path = os.path.join(base_dir, node.name + ".nxl")
                if not os.path.exists(mod_path):
                     mod_path = os.path.join(base_dir, node.name, "mod.nxl")
                
                # Fallback to CWD/Project Root for std lib
                if not os.path.exists(mod_path):
                     mod_path = os.path.join(os.getcwd(), node.name + ".nxl")
                     if not os.path.exists(mod_path):
                          mod_path = os.path.join(os.getcwd(), node.name, "mod.nxl")

                if not os.path.exists(mod_path):
                     raise Exception(f"Module file not found: {node.name}.nxl or {node.name}/mod.nxl in {base_dir} or {os.getcwd()}")
                
                with open(mod_path, 'r') as f:
                    mod_src = f.read()
                
                lx = Lexer(mod_src)
                tokens = lx.tokenize()
                p = n_parser.Parser(tokens)
                mod_ast = p.parse()
                
                # Recurse
                mod_ast = resolve_modules(mod_ast, os.path.dirname(mod_path))
                
                # Mangle
                mangle_ast(mod_ast, node.name)
                
                new_ast.extend(mod_ast)
        else:
            new_ast.append(node)
    return new_ast

def main():
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
    ap.add_argument("--out", default=None, help="Output path (default: output.ll or output.spv)")
    args = ap.parse_args()

    filepath = args.file
    with open(filepath, 'r') as f:
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
    try:
        analyzer.analyze(ast)
    except CompilerError as e:
        print(f"Error: {e.message}")
        if e.line:
            lines = source.splitlines()
            if 0 <= e.line - 1 < len(lines):
                 print(f"  --> {filepath}:{e.line}:{e.column}")
                 print(f"   |")
                 print(f"{e.line:3} | {lines[e.line-1]}")
                 print(f"   | {' ' * (e.column-1)}^")
        if getattr(e, 'hint', None):
             print(f"  = help: {e.hint}")
        sys.exit(1)
    except Exception as e:
        print(f"[SEMANTIC ERROR] {e}")
        import traceback; traceback.print_exc()
        return

    # 4. Code Generation
    spirv_env = args.spirv_env if args.target == "spirv" else "opencl"
    emit_kernels_only = args.target == "spirv" and args.emit == "spv"
    codegen = CodeGen(
        target=args.target,
        emit_kernels_only=emit_kernels_only,
        spirv_env=spirv_env,
        spirv_local_size=args.spirv_local_size,
    )
    llvm_ir = codegen.generate(ast)

    if args.run_jit and args.target == "native":
        from jit import run_jit
        print("[JIT] Starting JIT...")
        ret = run_jit(str(llvm_ir))
        print(f"[JIT] Finished with code {ret}")
        return

    if args.emit == "ll":
        out_path = args.out or "output.ll"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
        print(f"\n[SUCCESS] LLVM IR compiled to '{out_path}'")
        if args.target == "native" and not args.run_jit:
            print("To run (requires clang): clang output.ll -o output.exe && ./output.exe")
        return

    # emit spv
    out_path = args.out or "output.spv"
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
        with open(ll_fallback, "w", encoding="utf-8") as f:
             f.write(llvm_ir)
        print(f"[SPIR-V EMIT ERROR] {e}")
        print(f"[FALLBACK] Wrote LLVM IR to '{ll_fallback}' (use llvm-as + llvm-spirv to convert).")

if __name__ == "__main__":
    main()

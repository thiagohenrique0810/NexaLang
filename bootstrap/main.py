import sys
import os
import argparse
from lexer import Lexer
from parser import Parser
from codegen import CodeGen

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
    ap.add_argument("--out", default=None, help="Output path (default: output.ll or output.spv)")
    args = ap.parse_args()

    filepath = args.file
    with open(filepath, 'r') as f:
        source = f.read()

    # 1. Lexing
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    # print("Tokens:", tokens)

    # 2. Parsing
    parser = Parser(tokens)
    ast = parser.parse()
    # print("AST:", ast)

    # 3. Semantic Analysis
    from semantic import SemanticAnalyzer
    analyzer = SemanticAnalyzer()
    try:
        analyzer.analyze(ast)
    except Exception as e:
        print(f"[SEMANTIC ERROR] {e}")
        return

    # 4. Code Generation
    # For SPIR-V emission we emit kernels-only to avoid illegal calls into kernels.
    emit_kernels_only = args.target == "spirv" and args.emit == "spv"
    spirv_env = args.spirv_env if args.target == "spirv" else "opencl"
    codegen = CodeGen(
        target=args.target,
        emit_kernels_only=emit_kernels_only,
        spirv_env=spirv_env,
        spirv_local_size=args.spirv_local_size,
    )
    llvm_ir = codegen.generate(ast)

    if args.emit == "ll":
        out_path = args.out or "output.ll"
        print(llvm_ir)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
        print(f"\n[SUCCESS] LLVM IR compiled to '{out_path}'")
        if args.target == "native":
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
        # Still write the LLVM IR so the user can translate externally.
        ll_fallback = os.path.splitext(out_path)[0] + ".ll"
        with open(ll_fallback, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
        print(f"[SPIR-V EMIT ERROR] {e}")
        print(f"[FALLBACK] Wrote LLVM IR to '{ll_fallback}' (use llvm-as + llvm-spirv to convert).")

if __name__ == "__main__":
    main()

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
    codegen = CodeGen(target=args.target, emit_kernels_only=emit_kernels_only)
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
        emit_spirv_from_llvm_ir(llvm_ir, out_path)
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

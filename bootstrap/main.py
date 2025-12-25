import sys
import os
from lexer import Lexer
from parser import Parser
from codegen import CodeGen

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <file.nxl>")
        return

    filepath = sys.argv[1]
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

    # 3. Code Generation
    codegen = CodeGen()
    llvm_ir = codegen.generate(ast)

    # 4. Output
    print(llvm_ir)
    
    # Save to file
    with open("output.ll", "w") as f:
        f.write(llvm_ir)
    
    print("\n[SUCCESS] LLVM IR compiled to 'output.ll'")
    print("To run (requires clang): clang output.ll -o output.exe && ./output.exe")

if __name__ == "__main__":
    main()

# NexaLang Development Roadmap

This document serves as the master checklist for the development of the NexaLang compiler and ecosystem.

## Phase 1: Bootstrap & Foundation 🏗️
The goal is to create a minimal working compiler in a host language (e.g., Python, C++ or Rust) that can emit LLVM IR.

- [ ] **Lexer (Tokenizer)**
    - [ ] Define tokens (Keywords, Identifiers, Literals, Operators).
    - [ ] Implement `Lexer` class to convert source to token stream.
- [ ] **Parser (AST)**
    - [ ] Define AST nodes (Program, Function, Statement, Expression).
    - [ ] Implement recursive descent parser.
    - [ ] Error handling for syntax errors.
- [ ] **Code Generation (LLVM)**
    - [ ] Setup LLVM bindings (e.g., `llvmlite` if Python, `llvm-sys` if Rust).
    - [ ] Implement IR generation for:
        - [ ] Functions (`fn main()`).
        - [ ] Basic types (`i32`, `f32`).
        - [ ] Arithmetic operations.
        - [ ] Return statements.
- [ ] **Hello World**
    - [ ] Link simple `libc` print function.
    - [ ] Compile and run a "Hello World" binary.

## Phase 2: Core Language Features 🧬

- [ ] **Data Types**
    - [ ] Primitive types (`u8`, `i64`, `bool`, `char`).
    - [ ] Structs (Aggregate types).
    - [ ] Tuples.
    - [ ] Arrays (Fixed size).
- [ ] **Control Flow**
    - [ ] `if` / `else` expressions.
    - [ ] `while` loops.
    - [ ] `match` pattern matching (basic).
- [ ] **Functions**
    - [ ] Arguments and Return values.
    - [ ] Function overloading (optional, or distinct names initially).

## Phase 3: Memory Safety System 🛡️

- [ ] **Semantic Analysis**
    - [ ] Symbol Table (Scope resolution).
    - [ ] Type Checking (Strong static typing).
- [ ] **Ownership Model (Affine Types)**
    - [ ] Track ownership moves.
    - [ ] Detect use-after-move errors.
    - [ ] Implement `drop` logic (cleanup at end of scope).
- [ ] **Borrow Checker**
    - [ ] Immutable borrowings (`&T`).
    - [ ] Mutable borrowings (`&mut T`).
    - [ ] Enforce "One writer OR many readers" rule.
- [ ] **Region Management**
    - [ ] Implement `region` keyword syntax.
    - [ ] Arena allocator implementation in runtime.

## Phase 4: Native GPU Support 🎮

- [ ] **GPU Syntax**
    - [ ] `kernel` keyword parsing.
    - [ ] `buffer<T>` generic type.
- [ ] **SPIR-V Backend**
    - [ ] Add SPIR-V target to LLVM pipeline.
    - [ ] Map `gpu::global_id()` to intrinsics.
- [ ] **Runtime Dispatch**
    - [ ] Implement `gpu::dispatch` to interface with Vulkan/Compute APIs.

## Phase 5: Self-Hosting & Ecosystem 🚀

- [ ] **Standard Library (`std`)**
    - [ ] `std::io` (File/Console).
    - [ ] `std::mem` (Allocators).
    - [ ] `std::vec` (Dynamic arrays).
- [ ] **Self-Hosting**
    - [ ] Rewrite the compiler using NexaLang itself.
    - [ ] Verify `nxc` can compile `nxc`.
- [ ] **Tooling**
    - [ ] `nx` CLI build tool.
    - [ ] Syntax highlighter extension (VSCode).

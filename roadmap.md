# NexaLang Development Roadmap

This document serves as the master checklist for the development of the NexaLang compiler and ecosystem.

## Phase 1: Bootstrap & Foundation 🏗️
The goal is to create a minimal working compiler in a host language (e.g., Python, C++ or Rust) that can emit LLVM IR.

- [x] **Lexer (Tokenizer)**
    - [x] Define tokens (Keywords, Identifiers, Literals, Operators).
    - [x] Implement `Lexer` class to convert source to token stream.
- [x] **Parser (AST)**
    - [x] Define AST nodes (Program, Function, Statement, Expression).
    - [x] Implement recursive descent parser.
    - [x] Error handling for syntax errors.
- [x] **Code Generation (LLVM)**
    - [x] Setup LLVM bindings (e.g., `llvmlite` if Python, `llvm-sys` if Rust).
    - [x] Implement IR generation for:
        - [x] Functions (`fn main()`).
        - [x] Basic types (`i32`, `i64`, `u8`, `bool`, pointers, strings-as-`i8*`).
        - [x] Floating point (`f32`).
        - [x] Arithmetic operations.
        - [x] Return statements.
- [x] **Hello World**
    - [x] Link simple `libc` print function.
    - [x] Compile and run a "Hello World" binary.

## Phase 2: Core Language Features 🧬

- [x] **Data Types**
    - [x] Primitive types (`u8`, `i64`, `bool`, `char`).
- [x] **Type System 2.0**
  - [x] Structs & Methods
  - [x] Arrays & Slices
  - [x] Enums & Pattern Matching
  - [x] Generics (Basic Monomorphization)
  - [x] Type Inference (Local)
- [ ] **Control Flow**
    - [x] `if` / `else` expressions.
    - [x] `while` loops.
    - [x] `match` pattern matching (basic).
- [x] **Functions**
    - [x] Arguments and Return values.
    - [ ] Function overloading (optional, or distinct names initially).

## Phase 3: Memory Safety System 🛡️

- [x] **Semantic Analysis**
    - [x] Symbol Table (Scope resolution).
    - [x] Type Checking (Strong static typing).
- [x] **Ownership Model (Affine Types)**
    - [x] Track ownership moves (bootstrap rules: non-Copy types move on by-value assignment/calls/return).
    - [x] Detect use-after-move errors.
    - [x] Implement `drop` logic (cleanup at end of scope if `{Type}_drop` exists in module).
- [x] **Borrow Checker**
    - [x] Immutable borrowings (`&T`).
    - [x] Mutable borrowings (`&mut T`).
    - [x] Enforce "One writer OR many readers" rule.
- [ ] **Region Management**
    - [x] Implement `region` keyword syntax.
    - [x] Arena allocator implementation in runtime (bootstrap: `Arena_new/drop/alloc`).

## Phase 4: Native GPU Support 🎮

- [x] **GPU Syntax**
    - [x] `kernel` keyword parsing.
    - [x] `Buffer<T>` generic type (bootstrap builtin).
- [ ] **SPIR-V Backend**
    - [x] Add SPIR-V target to LLVM pipeline (bootstrap: emits LLVM IR with `spir64-unknown-unknown`; `.spv` needs external tools).
    - [~] Map `gpu::global_id()` to intrinsics (bootstrap: uses placeholder global `__spirv_BuiltInGlobalInvocationId_x`; real mapping depends on translator/toolchain).
- [ ] **Runtime Dispatch**
    - [x] Implement `gpu::dispatch` (bootstrap/mock: CPU loop calling kernel + sets `gpu::global_id()`).

## Phase 5: Self-Hosting & Ecosystem 🚀

- [x] **Standard Library Basics**
  - [x] `print` function (intrinsic).
  - [x] `Option<T>` and `Result<T, E>` (via Enums).
    - [x] `Vec<T>` (Dynamic array).
    - [x] String manipulation.
- [ ] **Self-Hosting**
    - [ ] Rewrite the compiler using NexaLang itself.
    - [ ] Verify `nxc` can compile `nxc`.
- [ ] **Tooling**
    - [ ] `nx` CLI build tool.
    - [ ] Syntax highlighter extension (VSCode).

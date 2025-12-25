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
        - [x] Basic types (`i32`, `f32`).
        - [x] Arithmetic operations.
        - [x] Return statements.
- [x] **Hello World**
    - [x] Link simple `libc` print function.
    - [x] Compile and run a "Hello World" binary.

## Phase 2: Core Language Features 🧬

- [ ] **Data Types**
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
    - [ ] `match` pattern matching (basic).
- [x] **Functions**
    - [x] Arguments and Return values.
    - [ ] Function overloading (optional, or distinct names initially).

## Phase 3: Memory Safety System 🛡️

- [x] **Semantic Analysis**
    - [x] Symbol Table (Scope resolution).
    - [x] Type Checking (Strong static typing).
- [ ] **Ownership Model (Affine Types)**
    - [x] Track ownership moves.
    - [x] Detect use-after-move errors.
    - [ ] Implement `drop` logic (cleanup at end of scope).
- [ ] **Borrow Checker**
    - [ ] Immutable borrowings (`&T`).
    - [ ] Mutable borrowings (`&mut T`).
    - [ ] Enforce "One writer OR many readers" rule.
- [ ] **Region Management**
    - [ ] Implement `region` keyword syntax.
    - [ ] Arena allocator implementation in runtime.

## Phase 4: Native GPU Support 🎮

- [x] **GPU Syntax**
    - [x] `kernel` keyword parsing.
    - [ ] `buffer<T>` generic type.
- [ ] **SPIR-V Backend**
    - [ ] Add SPIR-V target to LLVM pipeline.
    - [x] Map `gpu::global_id()` to intrinsics.
- [ ] **Runtime Dispatch**
    - [ ] Implement `gpu::dispatch` to interface with Vulkan/Compute APIs.

## Phase 5: Self-Hosting & Ecosystem 🚀

- [x] **Standard Library Basics**
  - [x] `print` function (intrinsic).
  - [x] `Option<T>` and `Result<T, E>` (via Enums).
  - [ ] `Vec<T>` (Dynamic array).
  - [ ] String manipulation.
- [ ] **Self-Hosting**
    - [ ] Rewrite the compiler using NexaLang itself.
    - [ ] Verify `nxc` can compile `nxc`.
- [ ] **Tooling**
    - [ ] `nx` CLI build tool.
    - [ ] Syntax highlighter extension (VSCode).

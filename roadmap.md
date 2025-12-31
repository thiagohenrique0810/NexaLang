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
    - [~] Primitive types (`u8`, `i64`, `bool`).
    - [x] `char`.
- [x] **Type System 2.0**
  - [x] Structs & Methods
    - [x] Struct Definitions
    - [x] `impl` blocks (Associating functions with types).
    - [x] Methods (`self`, `&self` receivers).
    - [x] Static Methods (`Type::new()`) vs Instance Methods (`obj.method()`).
  - [x] Arrays & Slices
    - [x] Arrays
    - [x] Slices (bootstrap: `[]T` lowered to `Slice<T>`, `slice_from_array(&arr)`, `s.len`, `s[i]`).
  - [x] Enums & Pattern Matching
  - [x] Generics (Basic Monomorphization)
  - [x] Type Inference (Local)
- [x] **Control Flow**
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
- [x] **Region Management**
    - [x] Implement `region` keyword syntax.
    - [x] Arena allocator implementation in runtime (bootstrap: `Arena_new/drop/alloc`).

## Phase 4: Native GPU Support 🎮

- [x] **GPU Syntax**
    - [x] `kernel` keyword parsing.
    - [x] `Buffer<T>` generic type (bootstrap builtin).
- [x] **SPIR-V Backend**
    - [x] Add SPIR-V target to LLVM pipeline (bootstrap: module triple `spirv64-unknown-unknown`; `.spv` needs external tools).
    - [x] Map `gpu::global_id()` to SPIR-V BuiltIn `GlobalInvocationId` (bootstrap: loads from `__spirv_BuiltInGlobalInvocationId` as `<3 x i32>` in `addrspace(5)` and extracts `.x`).
    - [x] Vulkan env (bootstrap): `--spirv-env vulkan` emits `OpCapability Shader` + `OpMemoryModel Logical GLSL450`, supports `--spirv-local-size`, maps kernel args to interface globals (no params), and auto-patches access chains + descriptor bindings using `spirv-dis`/`spirv-as` so `spirv-val` passes for `Buffer<T>` indexing.
- [x] **Runtime Dispatch**
    - [x] Implement `gpu::dispatch` (bootstrap/mock: CPU loop calling kernel + sets `gpu::global_id()`).

## Phase 5: Self-Hosting & Ecosystem 🚀

- [x] **Standard Library Basics**
  - [x] `print` function (intrinsic).
  - [x] `Option<T>` and `Result<T, E>` (via Enums).
    - [x] `Vec<T>` (Dynamic array).
    - [x] String manipulation.
- [ ] **Self-Hosting**
    - [~] Rewrite the compiler using NexaLang itself.
      - [x] Stage 1: file IO (`fs::read_file`) + Buffer<u8> sample.
      - [x] Stage 2: minimal lexer in NexaLang (token counting).
      - [x] Stage 3: tokenize into a token stream data structure (Token {kind,start,len}).
      - [x] Stage 4: parser subset (parse `fn` blocks + count `let`/`return` and basic block structure).
      - [ ] Stage 5: emit LLVM IR (subset) or transpile to bootstrap IR.
    - [ ] Verify `nxc` can compile `nxc`.
- [ ] **Tooling**
    - [x] `nx` CLI build tool (bootstrap: `python nx.py ...`).
    - [x] Syntax highlighter extension (VSCode) (TextMate grammar under `vscode-nexalang/`).

## Phase 6: Advanced Language Features 🎯

### 6.1 Complete OOP Support ✅ COMPLETE
- [x] **Struct Methods - Foundation**
  - [x] Parse `impl Type { }` blocks
  - [x] Basic method declarations in codegen
- [x] **Instance Methods**
  - [x] Parser: Handle `self`, `&self`, `&mut self` parameters
  - [x] Semantic: Type-check self receivers
  - [x] Codegen: Pass struct as first argument (by value, reference, or mutable reference)
  - [x] Method call syntax: `obj.method()` instead of `Type::method(obj)`
- [x] **Static Methods**
  - [x] Call syntax: `Type::method()`
  - [x] Ensure no `self` parameter for static methods
- [x] **Method Resolution**
  - [x] Resolve method calls in semantic analysis
  - [x] Handle method overloading (struct-mangled names)
  - [x] Support method chaining: `obj.method1().method2()`

### 6.2 Enhanced Standard Library (CURRENT PRIORITY)
- [x] **Vec<T> Complete Implementation**
  - [x] `Vec::new()` - constructor
  - [x] `push(&mut self, item: T)` - add element
  - [x] `pop(&mut self) -> Option<T>` (implemented as `pop(&mut self) -> T` for bootstrap)
  - [x] `len(&self) -> i32` - get length
  - [x] `get(&self, idx: i32) -> Option<&T>` (implemented as `get(&self, index: i32) -> T` for bootstrap)
  - [x] `clear(&mut self)` - remove all
  - [ ] Iterator support (for future for-loops)
- [x] **String Manipulation**
  - [x] `String::from(s: &str)` - convert from string literal
  - [x] `len(&self) -> i32`
  - [x] `concat(&self, other: &str) -> String`
  - [x] `substring(&self, start: i32, len: i32) -> String`
  - [x] `contains(&self, needle: &str) -> bool`
  - [x] `split(&self, delimiter: char) -> Vec<String>`

### 6.3 Module System & Code Organization
- [x] **Basic Modules**
  - [x] `mod module_name;` syntax
  - [x] File-based modules (one file = one module)
  - [x] Multi-file compilation in `nx.py` (via `main.py` resolution)
- [ ] **Visibility & Privacy**
- [x] **Visibility & Privacy**
  - [x] `pub` keyword for public items
  - [x] Default private visibility
  - [x] Privacy checking in semantic analysis
- [ ] **Import System**
  - [ ] `use module::item;` syntax
  - [ ] `use module::*;` glob imports
  - [ ] Path resolution across modules
- [ ] **Module Hierarchy**
  - [ ] Nested modules (`mod parent { mod child { } }`)
  - [ ] Directory-based modules (`mod.nxl` or `mod/mod.nxl`)

### 6.4 Control Flow Enhancements
- [~] **For Loops**
  - [x] Range syntax: `0..10`, `0..=10` (inclusive)
  - [ ] `for item in collection` syntax
  - [ ] Iterator trait (simple version)
  - [x] Native Codegen implementation (direct LLVM IR)
- [x] **Loop Control**
  - [x] `break` statement
  - [x] `continue` statement
  - [ ] Labeled for circles (`'label: loop`) { }`

### 6.5 Developer Experience
- [ ] **Better Error Messages**
  - [ ] Show file, line, and column numbers
  - [ ] Pretty-print error context with caret (^) pointing to error
  - [ ] Suggestion system ("did you mean X?")
  - [ ] Error codes and documentation links
- [ ] **Warnings System**
  - [ ] Unused variable warnings
  - [ ] Dead code detection
  - [ ] Type coercion warnings

## Phase 7: Advanced Type System 🔧
- [ ] **Traits (Interfaces)**
  - [ ] `trait Name { }` syntax
  - [ ] Trait methods (required and provided)
  - [ ] `impl Trait for Type` syntax
  - [ ] Trait bounds in generics: `fn foo<T: Display>(x: T)`
- [ ] **Advanced Generics**
  - [ ] Multiple type parameters
  - [ ] Const generics: `Array<T, const N: usize>`
  - [ ] Associated types in traits
- [ ] **Type Aliases**
  - [ ] `type Name = ExistingType;`

## Phase 8: Future Directions 🚀
- [ ] Closures and lambdas
- [ ] Async/await (future consideration)
- [ ] Procedural macros
- [ ] Foreign Function Interface (FFI) for C interop
- [ ] Package manager (`nxpkg`)
- [ ] Build system improvements
- [ ] Standard library expansion (HashMap, File I/O, etc.)


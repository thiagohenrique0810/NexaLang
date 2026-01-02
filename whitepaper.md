# NexaLang: The Universal Systems Language
**Technical Specification & Whitepaper v0.5**

## 1. Introduction

**NexaLang** is a modern, high-performance systems programming language designed to unify CPU and GPU computing under a single, expressive syntax. It aims to replace C++ and Rust in critical domains by offering **memory safety without garbage collection**, **seamless manual control**, and **native heterogeneity**.

**Extension**: `.nxl`
**Philosophy**: *Control by Choice, Safety by Default.*

---

## 2. Architecture & Execution

### 2.1 Compilation Model
NexaLang uses an **AOT (Ahead-of-Time)** compilation model built on top of **LLVM**, ensuring industry-standard optimization and portability.

*   **Compiler**: `nxc` (Nexa Compile)
*   **Backends**: Native (X86/ARM), SPIR-V (GPU/Vulkan).
*   **Optimization**: Built-in support for LLVM optimization levels (`-O1` to `-O3`).

### 2.2 The "Dual-Mode" Runtime
NexaLang features a **minimal runtime** (smaller than C runtime) by default.
*   **Zero-Overhead abstractions**: Most features (like Generics and Traits) compile away.
*   **Standard Library**: A modular `std/` library providing `Vec`, `HashMap`, `String`, and `File IO`.

---

## 3. Memory Management

NexaLang introduces a **Hybrid Safety Model**.

### 3.1 Affine Ownership & RAII
By default, the language enforces ownership rules. Non-copyable types are moved on assignment.
```nexalang
fn main() {
    let message = String::from("Hello")
    process(message) # Ownership moved
    # print(message) # Compile Error: Use after move
}
```
Resources are automatically cleaned up via the `drop` method (RAII) at the end of their scope.

### 3.2 Explicit Regions (Arenas)
NexaLang treats **Region-based Memory Management** as a first-class citizen.
```nexalang
fn process() {
    let mut a = Arena::new(1024)
    let ptr = Arena::alloc(&a, 64)
    # 'a' drops here, freeing all allocated memory in O(1).
}
```

---

## 4. Parallelism & GPU Computing

### 4.1 Async / Await
Foundation for non-blocking IO and concurrency using the `async` and `await` keywords.
```nexalang
async fn fetch_data() -> String { ... }

fn main() {
    let data = await fetch_data()
}
```

### 4.2 Native GPU Kernels (Silicon Mode)
Write GPU code directly in NexaLang using the `kernel` keyword. NexaLang supports **Silicon Mode**, allowing direct execution on AMD and NVIDIA hardware via a high-performance OpenCL bridge.
```nexalang
kernel fn fill(buf: Buffer<i32>, val: i32) {
    let i = gpu::global_id()
    buf[i] = val
}
```
*   **Zero-Copy Memory**: Direct mapping of host buffers to GPU address space.
*   **Hardware Agnostic**: Single binary works across different GPU vendors.

---

## 5. Metaprogramming & Attributes

### 5.1 Attributes
Custom behavior via `@[attr]` syntax.
```nexalang
@[test]
fn test_logic() {
    assert(1 + 1 == 2, "Error")
}
```

### 5.2 Derivation
Automatically generate implementation for common traits.
```nexalang
@[derive(Debug)]
struct Point { x: i32, y: i32 }
# Generates p.debug_print()
```

---

## 6. Interoperability (FFI)

Direct C linking with no overhead.
```nexalang
extern "C" {
    fn printf(fmt: *u8, ...);
}
```

---

## 7. Ecosystem

### The `nx` toolchain
- `nxc build`: Compile projects using `nexa.json`.
- `nxc run`: Build and execute.
- `nxc test`: Run functions marked with `@[test]`.
- `nxpkg`: Package manager for dependencies.

### Identity
*   **Name**: NexaLang
*   **Extensions**: `.nxl` (Source), `.ll` (Low-level IR)
*   **Mascot**: A crystalline fox (representing agility and structure).
*   **Use Cases**: OS Dev, Game Engines, AI Infrastructure, Embedded Systems.

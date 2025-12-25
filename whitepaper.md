# NexaLang: The Universal Systems Language
**Technical Specification & Whitepaper v0.1**

## 1. Introduction

**NexaLang** is a modern, high-performance systems programming language designed to unify CPU and GPU computing under a single, expressive syntax. It aims to replace C++ and Rust in critical domains (Game Engines, OS, AI, Embedded) by offering **memory safety without garbage collection**, **seamless manual control**, and **native heterogeneity**.

**Extension**: `.nxl`
**Philosophy**: *Control by Choice, Safety by Default.*

---

## 2. Architecture & Execution

### 2.1 Compilation Model
NexaLang uses an **AOT (Ahead-of-Time)** compilation model built on top of a modified **LLVM** backend, ensuring industry-standard optimization and portability.

*   **Compiler**: `nxc` (Nexa Compile)
*   **Intermediate Representation**: NexaIR (High-level) -> LLVM IR (Low-level) -> Machine Code.
*   **Targets**: Windows, Linux, macOS, Android, iOS, Bare Metal, WebAssembly (WASM).

### 2.2 The "Dual-Mode" Runtime
NexaLang features a **minimal runtime** (smaller than C runtime) by default.
*   **Zero-Overhead abstractions**: Most features compile away to raw machine instructions.
*   **Optional Standard Library**: Can be completely disabled (`@no_std`) for embedded/kernels.

---

## 3. Memory Management

NexaLang introduces a **Hybrid Safety Model** that acts as a bridge between Rust's strictness and Zig's freedom.

### 3.1 Default: Affine Ownership (Auto-Safety)
By default, the language enforces ownership and borrowing rules similar to Rust to prevent use-after-free and data races at compile time.

```nexalang
fn main() {
    let message = String.from("Hello")
    process(message) // Ownership moved here
    // print(message) // Compile Error: Use after move
}
```

### 3.2 Explicit Regions (Arenas)
Unlike Rust, NexaLang treats **Region-based Memory Management** as a first-class citizen. You can allocate entire scopes to an arena and free them instantly.

```nexalang
fn process_transaction() {
    region temp_arena {
        let user = User.new(in temp_arena)
        let cart = Cart.new(in temp_arena)
        // ... heavy processing ...
    } // 'temp_arena' dropped here. Instant O(1) deallocation.
}
```

### 3.3 Manual Control (`unsafe`)
For hardware interaction, you can opt-out of safety checks explicitly.

```nexalang
unsafe {
    let ptr: *mut u8 = 0xB8000 as *mut u8
    *ptr = 0xFF // Direct hardware write
}
```

---

## 4. Parallelism & GPU Computing (Native Heterogeneity)

One of NexaLang's pillar features is treating the GPU as a standard execution context, not an external device requiring foreign APIs (CUDA/OpenCL).

### 4.1 CPU Parallelism
Lightweight threads (fibers) managed by a work-stealing scheduler.

```nexalang
async fn fetch_data() -> Data { ... }

fn main() {
    let result = await spawn fetch_data()
}
```

### 4.2 GPU Kernels (The "Killer Feature")
Write shader/kernel code directly in NexaLang using the `kernel` keyword. The compiler targets SPIR-V or PTX depending on the hardware.

```nexalang
// Functions marked 'kernel' are compiled to GPU bytecode
kernel fn gaussian_blur(image: Buffer<f32>, width: u32) {
    let idx = gpu::global_id().x
    if idx >= width { return }
    
    // Efficient local memory usage
    let pixel = image[idx]
    image[idx] = pixel * 0.5
}

fn main() {
    let img = load_image("texture.png")
    // Dispatch kernel directly
    gpu::dispatch(gaussian_blur, args: (img, 1024), threads: 1024)
}
```

---

## 5. Interoperability

### 5.1 C / C++
Zero-cost, two-way interoperability. No wrapper generation needed for standard headers.

```nexalang
@import_c("stdio.h")

fn main() {
    c::printf(c"Hello from C world!\n")
}
```

### 5.2 Python / JS
Automatic binding generation via the `nx bind` tool, allowing NexaLang libraries to be imported as native Python modules or Node.js addons.

---

## 6. Syntax Overview

### Hello World
```nexalang
import std::io

fn main() {
    io::print("Hello, World!")
}
```

### Structs & Tuples
```nexalang
struct Player {
    name: String
    health: i32
    pub score: i32 // Public field
}

fn update(p: &mut Player) {
    p.health -= 10
}
```

### Pattern Matching
```nexalang
match status {
    .ok(val) => print("Success: ${val}"),
    .err(e)  => print("Error: ${e}"),
    _        => print("Unknown")
}
```

---

## 7. Ecosystem

### The `nx` Tool
A single binary that handles everything.

*   `nx new <project>`: Create project.
*   `nx build`: Compile (Debug/Release).
*   `nx run`: Build and run.
*   `nx test`: Run unit tests.
*   `nx pkg`: Package manager.
*   `nx fmt`: Auto-formatter.

### Identity
*   **Name**: NexaLang
*   **Extensions**: `.nxl` (Source), `.nxb` (Binary/Lib)
*   **Mascot**: A crystalline fox (representing agility and structure).
*   **Primary Use Cases**: Operating Systems, High-Performance Game Engines, Real-time Simulation, Embedded Safety-Critical Systems.

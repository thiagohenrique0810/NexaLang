# NexaLang

**The Universal Systems Language**

![NexaLang Badge](https://img.shields.io/badge/NexaLang-v0.5-blueviolet?style=for-the-badge) ![Status](https://img.shields.io/badge/Status-Self--Hosted-green?style=for-the-badge) ![Mascot](https://img.shields.io/badge/Mascot-Crystalline_Fox-orange?style=for-the-badge)

NexaLang is a modern, high-performance systems programming language designed to unify CPU and GPU computing. It combines the safety of modern languages with the raw power of low-level control, featuring a **self-hosted compiler** and native binary generation.

---

## 🚀 Key Features

### 🧠 Hybrid Memory Management
Safe by default, manual when you need it. NexaLang uses **Affine Ownership** and **RAII (Automatic Drop)** to prevent leaks at compile time, while offering **Region-based Arenas** for high-performance allocations.

### ⚡ Native GPU Kernels (Silicon Mode)
Write kernels directly in NexaLang. Treat the GPU as a first-class citizen with **Hardware Silicon Mode**. Support for **AMD Radeon (RX 580+)** and **NVIDIA** via high-performance OpenCL hardware dispatch.

```nexalang
kernel fn compute_physics(particles: Buffer<Particle>) {
    let i = gpu::global_id()
    # ... logic runs directly on GPU Stream Processors ...
}
```

### 🔌 C Interop & FFI
Zero-overhead linking with C libraries. Call `malloc`, `printf`, or any system API directly with the `extern "C"` block.

### 🛠️ Professional Tooling
- **Integrated Test Framework**: Mark functions with `@[test]` and run them with `nxc test`.
- **Automatic Derivation**: Generate boilerplate like `debug_print()` automatically with `@[derive(Debug)]`.
- **Function Overloading**: Multiple functions with the same name, resolved by parameter types.
- **Package Manager (`nxpkg`)**: Manage dependencies and project structure seamlessly.

---

## 📦 Standard Library (`std`)
NexaLang comes with a growing modular standard library:
- `std::vec::Vec<T>`: Dynamic arrays with `map`, `filter`, and `fold`.
- `std::map::HashMap<K, V>`: High-performance hash maps.
- `std::string::String`: Managed string type with full manipulation support.
- `std::fs`: Object-oriented file I/O.
- `std::option` & `std::result`: Modern error handling.
- `std::compress`: Near-optimal vector compression powered by **TurboQuant** (1–4 bit quantization, ~2.7× of Shannon's distortion-rate limit).

### 🗜️ TurboQuant Compression
NexaLang integrates a C port of the [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm (Zandieh et al. 2025) for ultra-low-bitwidth vector quantization. Use cases include KV-cache compression for transformer inference, vector database compaction, and GPU buffer optimization.

```nexalang
use std::compress::Quantizer;

let q = Quantizer::new(128, 3);      # 3-bit, ~10.7× compression vs f32
q.quantize(input, compressed, 1);
q.dequantize(compressed, output, 1);
q.drop();
```

The runtime library (`runtime/libturboquant`) is data-oblivious — no calibration data or codebook training required.

---

## 🛠️ Getting Started

### Requirements
- **LLVM / Clang**: Essential for compilation and linking (version 15+ recommended).
- **Python 3.8+**: Required for the bootstrap CLI.

### Global Installation

#### Windows
1. Add the NexaLang root directory to your **User PATH** environment variable.
2. You can now use the `nxc` command from any terminal.

#### Linux / macOS
1. Create a symbolic link:
   ```bash
   chmod +x nxc
   sudo ln -s $(pwd)/nxc /usr/local/bin/nxc
   ```

### Basic Commands
```bash
# Initialize a new project
nxpkg init my_project

# Compile and run a file immediately
nxc run main.nxl

# Build a standalone executable with O3 optimization
nxc build main.nxl --opt O3 --out artifacts/build/bin/my_app

# Bootstrap compiler default output
python3 bootstrap/main.py examples/hello.nxl --emit ll --opt 3
# writes LLVM IR to artifacts/build/output.ll

# Run integrated unit tests
nxc test my_logic.nxl
```

---

## 🎨 VSCode Integration
The official extension provides syntax highlighting, snippets, and integrated error reporting.
- **Snippets**: Supports `fn`, `struct`, `test`, `main`, `if`, `for`, and `while`.
- **Installation**: Copy `vscode-nexalang/` to your extensions folder.

---

## 🧬 Self-Hosting Milestone
NexaLang has successfully achieved **self-hosting**. The compiler source code in `selfhost/stage5_full.nxl` is capable of compiling itself into the native `bin/nxc.exe` binary, proving the language's maturity.

---

*Designed by **Thiago Henrique**.*

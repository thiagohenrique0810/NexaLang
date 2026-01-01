# NexaLang

**The Universal Systems Language**

![NexaLang Badge](https://img.shields.io/badge/NexaLang-v0.5-blueviolet?style=for-the-badge) ![Status](https://img.shields.io/badge/Status-Self--Hosted-green?style=for-the-badge)

NexaLang is a modern, high-performance systems programming language designed to unify CPU and GPU computing. It combines the safety of modern languages with the raw power of low-level control, featuring a **self-hosted compiler** and native binary generation.

---

## 🚀 Key Features

### 🧠 Hybrid Memory Management
Safe by default, manual when you need it. NexaLang uses **Affine Ownership** and **RAII (Automatic Drop)** to prevent leaks at compile time, while offering **Region-based Arenas** for high-performance allocations.

### ⚡ Native GPU Kernels
Write kernels directly in NexaLang. Treat the GPU as a first-class citizen with seamless dispatch and SPIR-V/Vulkan support.

```nexalang
kernel fn compute_physics(particles: Buffer<Particle>) {
    let i = gpu::global_id()
    # ... logic runs on GPU ...
}
```

### 🔌 C Interop & FFI
Zero-overhead linking with C libraries. Call `malloc`, `printf`, or any system API directly with the `extern "C"` block.

### 🛠️ Professional Tooling
- **Integrated Test Framework**: Mark functions with `@[test]` and run them with `nxc test`.
- **Automatic Derivation**: Generate boilerplate like `debug_print()` automatically with `@[derive(Debug)]`.
- **Function Overloading**: Multiple functions with the same name, resolved by parameter types.

---

## 🛠️ Getting Started

### Requirements
- **LLVM / Clang**: Essential for compilation and linking. (Recommended: version 15+).
- **Python 3.8+**: Required for the bootstrap CLI and build tools.

### Global Installation

#### Windows
1. Add the NexaLang root directory (`Y:\WWW\NexaLang`) to your **User PATH** environment variable.
2. You can now use the `nxc` command from any CMD or PowerShell window.

#### Linux / macOS
1. Create a symbolic link to the `nxc` wrapper:
   ```bash
   chmod +x nxc
   sudo ln -s $(pwd)/nxc /usr/local/bin/nxc
   ```

### Basic Commands
```bash
# Compile and run a file immediately
nxc run examples/hello.nxl

# Build a standalone executable with O3 optimization
nxc build main.nxl --opt O3 --out my_app.exe

# Run integrated unit tests
nxc test examples/test_example.nxl
```

---

## 🎨 VSCode Integration
The official extension provides syntax highlighting, snippets, and integrated error reporting.
- Install from `vscode-nexalang/` folder.
- Supports `fn`, `struct`, `test`, and `main` snippets.

---

## 🧬 Self-Hosting Milestone
NexaLang has successfully achieved **self-hosting**. The compiler source code in `selfhost/stage5_full.nxl` is capable of compiling itself into the native `bin/nxc.exe` binary.

---

*Designed by **Thiago Henrique**.*

# NexaLang

**The Universal Systems Language**

![NexaLang Badge](https://img.shields.io/badge/NexaLang-v0.1-blueviolet?style=for-the-badge) ![Status](https://img.shields.io/badge/Status-In_Development-orange?style=for-the-badge)

NexaLang is a modern, high-performance systems programming language designed to unify CPU and GPU computing. It combines the safety of modern languages with the raw power of low-level control, all without a garbage collector.

---

## 🚀 Key Features

### 🧠 Hybrid Memory Management
Safe by default, manual when you need it. NexaLang uses **Affine Ownership** to prevent leaks and data races at compile time, but allows explicit **Region-based** management and `unsafe` blocks for hardware drivers.

### ⚡ Native GPU Kernels
Stop writing CUDA C++ strings in your code. NexaLang treats the GPU as a first-class citizen. Write kernels directly in standard NexaLang syntax and dispatch them seamlessly.

```nexalang
kernel fn compute_physics(particles: Buffer<Particle>) {
    let i = gpu::global_id().x
    // ... logic runs on GPU ...
}
```

### 🔌 Zero-Cost Interop
Link directly with C and C++ libraries with no overhead. Generate Python and JavaScript bindings automatically with the `nx` toolchain.

---

## 📄 Technical Specification

For a deep dive into the architecture, memory model, and syntax, please read the **[Official Whitepaper](whitepaper.md)**.

---

## 🛠️ Getting Started

*Current Status: **Design & Prototyping Phase***

We are currently implementing the `nxc` bootstrap compiler. Stay tuned for version 0.1!

```bash
# Future usage
nx new my-project
nx run
```

---

## 🧩 SPIR-V Backend (Bootstrap)

The bootstrap compiler can **emit LLVM IR prepared for SPIR-V** and can optionally **emit `.spv`** if you have external tools installed.

### Generate SPIR-V-flavored LLVM IR

```bash
python bootstrap/main.py examples/gpu_dispatch.nxl --target spirv --emit ll --out output.spirv.ll
```

### Emit `.spv` (requires external tools)

Preferred: `llc` with SPIR-V targets enabled (LLVM 20+).

Fallback: `llvm-as` + `llvm-spirv` (SPIRV-LLVM-Translator).

```bash
python bootstrap/main.py examples/gpu_dispatch.nxl --target spirv --emit spv --out output.spv
```

### Check toolchain

```bash
python tools/check_spirv_toolchain.py
```

### Activate toolchain (this repo)

If you created the environment under `Y:\tools\nexalang-spirv`, you can activate it for the current PowerShell session:

```powershell
.\tools\activate_spirv_env.ps1
```

### Vulkan environment (experimental)

You can switch the SPIR-V environment to Vulkan compute:

```powershell
python bootstrap\main.py examples\gpu_kernel_vulkan_spirv.nxl --target spirv --spirv-env vulkan --spirv-local-size 8,1,1 --emit spv --out fill_vulkan.spv
spirv-dis fill_vulkan.spv | Select-String -Pattern "OpCapability|OpMemoryModel|OpEntryPoint|OpExecutionMode"
spirv-val fill_vulkan.spv
```

Notes:
- The LLVM SPIR-V backend may emit `OpPtrAccessChain` for buffer indexing. The bootstrap emitter will patch this to `OpAccessChain` for `__nexa_*_data` StorageBuffer variables when `spirv-as` is available (required for Vulkan validation).
- When `--spirv-vulkan-var-pointers on` (default) and `spirv-as` is available, the bootstrap emitter may also inject `SPV_KHR_variable_pointers` + `VariablePointers*` capabilities if any remaining `OpPtrAccessChain` exists.
- If you don't have `spirv-as`, install SPIRV-Tools and ensure it is in PATH.
- Vulkan bindings are assigned deterministically by name: `*_len` first, then `*_data`, then others (per kernel/arg).

---

## 🛠️ `nx` CLI (Bootstrap)

A small helper CLI to run common tasks:

```bash
python nx.py examples
python nx.py run examples/hello.nxl
python nx.py build examples/gpu_kernel_vulkan_spirv.nxl --target spirv --spirv-env vulkan --emit spv --spv-out fill.spv
python nx.py val spirv fill.spv
```

By default, `nx` writes build outputs to `dev/artifacts/` to keep the repo root clean.

---

## 🎨 VSCode Syntax Highlighting

This repo includes a local VSCode extension under `vscode-nexalang/` for `.nxl` files.

- Open VSCode
- `Ctrl+Shift+P` → **Developer: Install Extension from Location...**
- Select the folder `vscode-nexalang`

---

## 🧬 Self-hosting (Stage 1)

The first self-hosting milestone is making it possible to write compiler tooling in NexaLang that can **read real source files**.

Bootstrap now provides:
- `fs::read_file(path: string) -> Buffer<u8>`

Try:

```bash
python nx.py run selfhost/stage1_read_file.nxl
```

---

## 🧬 Self-hosting (Stage 2)

A minimal lexer written in NexaLang (currently just counts token categories):

```bash
python nx.py run selfhost/stage2_lexer.nxl
```

---

## 🧬 Self-hosting (Stage 3)

Token stream milestone: a lexer written in NexaLang that produces `Token { kind, start, len }` records.

```bash
python nx.py run selfhost/stage3_tokens.nxl
```

---

## 🧬 Self-hosting (Stage 4)

Parser milestone (subset): parse `fn` blocks and count statements (`let` / `return` / basic `if`/`while` blocks).

```bash
python nx.py run selfhost/stage4_parser.nxl
```

---

*Designed by **Thiago Henrique**.*

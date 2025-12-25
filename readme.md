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

*Designed by **Thiago Henrique**.*

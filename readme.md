# NexaLang

NexaLang is an **experimental systems language** with a Python/LLVM bootstrap compiler, native builds, a standard library, a C compression runtime, and editor tooling.

The supported compiler entry point is `nxc` → `nx.py` → `bootstrap/`. The sources in `selfhost/` are prototypes; a reproducible self-hosting bootstrap has **not** been demonstrated by the current source tree. Ownership checks and automatic destruction are implemented, but the language does not yet provide a complete memory-safety guarantee. Raw pointers and C FFI require manual care.

## Install

Requirements: **Python 3.10+**, **Clang**, and the Python dependency pinned in `requirements.txt`. Development CI targets Python 3.11 and 3.12 on Linux, macOS, and Windows.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
./nxc run examples/hello.nxl
```

Keep the virtual environment active when invoking the launchers. On Windows, activate `.venv\Scripts\activate`, use `python`, and put the repository directory on PATH for `nxc.bat` and `nxpkg.bat`.

On Linux/macOS, optional user-local symlinks work from other project directories:

```sh
mkdir -p "$HOME/.local/bin"
ln -s "$(pwd)/nxc" "$HOME/.local/bin/nxc"
ln -s "$(pwd)/nxpkg" "$HOME/.local/bin/nxpkg"
# Add $HOME/.local/bin to PATH if necessary.
```

Native builds compile required C runtime sources for the current host. They do not link the historical ARM/Windows binaries from earlier revisions. Programs using SQLite or libcurl also need those libraries and their linker dependencies installed.

## Commands

```sh
nxpkg init my_project
nxc run main.nxl
nxc build main.nxl --opt O3 --out artifacts/build/my_app
nxc build main.nxl --no-link
nxc run main.nxl --jit
nxc test my_logic.nxl
```

Build artifacts default to `artifacts/build/` in the current project. `--opt` applies to both LLVM generation and native linking. Compilation errors return nonzero status and stop the build/run pipeline. Standard modules resolve relative to the installed toolchain.

`nxpkg` supports local directory dependencies and a local versioned registry cache. Install preserves the lockfile's resolved versions and verifies cached package integrity; `nxpkg update` explicitly resolves newer compatible versions. It does not implement remote fetching or a transitive dependency solver.

Republish packages created with the old cache format: integrity now covers file names and all package contents, not just `.nxl` source files.

## Language and runtime status

- Lexer, parser, semantic analysis, LLVM generation, native compilation, and JIT form the working bootstrap pipeline.
- Structs, methods, generic types, closures, slices, and ownership have regression tests. These tests cover specific supported cases, not a complete language specification.
- Async functions currently execute **eagerly**; `await` consumes the task result. This is not a suspendable LLVM coroutine scheduler or asynchronous I/O implementation.
- SPIR-V/OpenCL support is experimental and requires external tooling and suitable hardware. JIT does not simulate successful GPU execution.
- Automatic GPU quantization (`--quantize-gpu` and quantized dispatch attributes) is disabled with an explicit error until the dispatch contract preserves input scales and kernel writes. CPU packed compression remains available.
- `std/` includes collections, strings, files, JSON, networking, SQLite, tasks, and compression. APIs and low-level FFI remain experimental.
- MIR can be emitted for analysis; it is not the production LLVM lowering pipeline.

## TurboQuant buffers

The C runtime implements SRHT-based quantization. Dimensions must be powers of two. The low-level `uint16_t` index API expects unit-normalized vectors; the packed API preserves each vector's norm.

Packed buffers now use **TQ01** records: a 4-byte format marker, a 4-byte float norm, and packed indices for each vector. The norm uses host byte order; decoding requires the same dimension, bit width, and seed. **This format is incompatible with older raw packed buffers.** Recreate old buffers from their original vectors. Always obtain capacity through `tq_packed_size`, or `Quantizer.compressed_size`; `(dimension * bits + 7) / 8` alone is too small. The packed C calls return `0` on success and a negative status on failure.

```nexalang
use std::compress::Quantizer;

fn main() -> i32 {
    let q = Quantizer::new(128, 3);
    print(q.compressed_size(1));
    return 0; # Automatic resource cleanup.
}
```

Compression quality depends on the input and bit width. The research paper's theoretical guarantees are not a validation of this implementation or of end-to-end model quality.

## Verification

```sh
python3 tests/run_tests.py
python3 -m unittest discover -s tests -p 'test_*regressions.py' -v
python3 tests/check_examples.py
```

Tests build the C runtime from source instead of loading a platform-specific checked-in library. The regression suite exercises native binaries, ownership/destruction, numeric semantics, CLI failures, package confinement, LSP lifecycle, and packed-buffer behavior. Runtime tests also use AddressSanitizer and UndefinedBehaviorSanitizer where supported.

The example manifest records successful compilations, intentional errors, and explicitly experimental cases. Model training, network servers, and GPU examples are not automatically executed as application workloads.

## VS Code

See [vscode-nexalang/README.md](vscode-nexalang/README.md) for dependency installation and packaging. The extension bundles the language-server frontend and standard library during preparation. Build/run/test commands remain available when the LSP is disabled.

See [roadmap.md](roadmap.md) for outstanding work and [selfhost/README.md](selfhost/README.md) for self-hosting limitations.

## 512 MB model execution work

The [development checklist](docs/BLUEPRINT_512MB_CHECKLIST.md) tracks the complete
blueprint, technical amendments, validation, and the next task for resuming work.
The first implementation provides a validated model/tensor IR, static arena
planning, a versioned NexaPack Q4 matrix container, and a source-built C kernel.
The kernel consumes packed weights directly; Python schedules bounded row tiles.

```sh
# Use a new --pack path when generating a demo.
python3 tools/nexa_bench.py --generate-demo --pack artifacts/models/demo-q4.nxp --rows 4096 --cols 128 --tile-rows 64 --memory-budget 96KiB --verify --report artifacts/reports/demo-q4.json --csv artifacts/reports/demo-q4.csv

# Rerun the same packed matrix without regenerating it.
python3 tools/nexa_bench.py --pack artifacts/models/demo-q4.nxp --memory-budget 96KiB --verify

# Convert an existing row-major float32 little-endian matrix.
python3 tools/nexa_convert.py weights.f32 --rows 4096 --cols 128 --group-size 32 --out artifacts/models/weights.nxp
```

`--memory-budget` covers the CPU data arena, alignment padding, and bounded reader
scratch, plus `--reserve`. Reports identify excluded Python/metadata/OS overhead;
the limit is **not a process RSS or GPU VRAM cap**. Verification compares the C
result with scalar arithmetic over decoded Q4 values, not language-model quality.
`MB` and `MiB` are parsed separately. No model download or PyTorch is required.

Trained-model inference with tokenization, GPU execution for this pipeline,
additional KV codecs/residency tiers and backends remain pending gates. The legacy
PyTorch executor now rejects a requested memory limit it cannot enforce.
See [NexaPack V1](docs/NEXAPACK_V1.md) for the file and kernel contracts.

The model pipeline includes canonical NexaLM R0/v1 definitions and local Safetensors
import into a multi-tensor bundle. Matrices use Q4, norm vectors retain F32, and
tied embeddings use explicit aliases. Configuration, source hashes and tokenizer
assets are preserved. Tokenizer execution remains pending.

```sh
python3 tools/nexa_model.py models/nexalm512/architecture.nxl --out artifacts/reports/nexalm-architecture.json
# The destination must not already exist.
python3 tools/nexa_convert.py --checkpoint /path/to/local-checkpoint --out artifacts/models/local-model
python3 tools/nexa_inspect.py artifacts/models/local-model --verify
python3 tools/nexa_bench.py --bundle artifacts/models/local-model --tensor lm_head.weight --memory-budget 96KiB --verify
```

See the [architecture and import guide](docs/NEXALM_IMPORTACAO.md) for the supported
Llama subset, format limits and an offline fixture that runs without model downloads.

The native CPU executor now consumes the complete Transformer graph, including
RMSNorm, half-rotation RoPE, causal GQA, SwiGLU and residuals. It produces logits
from token IDs and supports greedy ID generation. By default, decode recomputes
the prefix. `--kv-cache` enables incremental CPU decode with two persistent F32 KV
banks; `--prefill-chunk-size` splits the prompt and requires `--kv-cache`.

```sh
# After creating/importing the tiny fixture described in the guide:
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-forward.json

# Optional incremental KV; prefill activations are limited to two tokens per chunk.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --prefill-chunk-size 2 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-chunks.json
```

Both KV banks count toward the managed CPU buffer budget alongside activations,
staging and reader scratch. Chunking limits temporary activation/logit buffers;
the cache still reserves the full declared context capacity. Commits are atomic
per prefill/append/decode call. Additional KV codecs and tiers, GPU execution, text
tokenization and trained-model quality remain pending gates.

`--kv-cache --kv-page-tokens 16` selects on-demand F32 KV pages. Attention reads
the pages directly, without concatenating the prefix; reset and close free them.
The budget distinguishes reserved capacity, resident pages and replacement staging.
See [paged KV execution](docs/NEXALM_KV_PAGINADO_CPU.md) for layout, transactions,
memory accounting and reproducible comparisons with the two-bank baseline.

Add `--kv-codec q4 --kv-group-size 32` to the paged mode to quantize each new KV
token/head independently. Attention consumes packed groups directly with no full
F32 cache copy; partial-page appends preserve the committed prefix. See
[Q4 KV execution](docs/NEXALM_KV_Q4_CPU.md) for the byte format, measured physical
memory savings and separate execution, weight-quantization and KV-quantization errors.

`--kv-codec q3` selects three-bit KV with the same paging and transaction API.
[Q3 KV execution](docs/NEXALM_KV_Q3_CPU.md) specifies the versioned bit layout and
compares physical F32/Q4/Q3 memory and numerical error under matching capacities.
Model weights remain Q4; Q3 weight import and matrix kernels are still pending.

The standalone TurboQuant runtime offers `tq_create_mse` for a linear-size MSE
context without the quadratic Prod/QJL state. Existing constructors keep Prod
support. See [MSE context memory and compatibility](docs/TURBOQUANT_MSE_CPU.md);
[Portable TQ storage](docs/NEXAPACK_TQ_V1.md) now persists explicit centroids and
little-endian TQ02 records in NexaPack, with bounded conversion and explicit
legacy TQ01 migration. [Paged TQ KV](docs/NEXALM_KV_TQ_CPU.md) adds native CPU
attention with one-head reconstruction scratch, explicit context memory and
transactional writes. Select `--kv-codec tq --kv-bits 3 --kv-seed 42` with paged
KV. [CPU page aging](docs/NEXALM_KV_TIERS_CPU.md) adds optional hot F32 / warm Q4 /
cold Q3 pages with mixed attention, bounded re-encoding and atomic publication.
Select `--kv-policy age --kv-hot-pages 1 --kv-warm-pages 1` with paged KV.
TQ model-weight matrix kernels, eviction/reload and GPU execution remain pending.

See [CPU Transformer execution](docs/NEXALM_EXECUCAO_CPU.md) for the graph contract,
historical baseline measurements and optional PyTorch oracle, and
[incremental KV execution](docs/NEXALM_KV_CPU.md) for the cache API, memory accounting
and chunked prefill examples.

The [Nexa Omni plan](docs/NexaLang_Plano_Implementacao_Nexa_Omni.pdf) proposes
multi-model orchestration through Packs, routing, shared memory and scheduling.
Its [integration notes and gates](docs/NEXA_OMNI_AJUSTES.md) preserve `nxpkg` as the
package manager and distinguish future Omni work from the implemented CPU runtime.

The [training integration notes](docs/NEXALM_TREINAMENTO_AJUSTES.md) map the new
NexaData/training plan onto the existing checklist, with separate contracts for
data, tokenizer, shards, checkpoint/resume and QAT. Training remains pending.

Designed by Thiago Henrique.

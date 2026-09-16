# FP16 GEMM with CuTe on SM100

A persistent, two-CTA GEMM expressed with Rust layouts, tiled copies/MMA,
tensor memory, and asynchronous pipelines.

```text
C[M, N] = A[M, K] @ B[N, K].T + bias[M, 1]  # bias is optional
```

- A, B, C, and bias are FP16; multiplication accumulates in FP32.
- `--has-bias` adds one value per output row before conversion to FP16.
- Layouts and pipeline contracts stay visible through the compiler:

![CuTe compiler flow from Rust layouts through verified MLIR and CUTLASS to a CUDA cubin](../../docs/assets/cutlass-translation-flow.svg)

## Tile and pipeline

```text
                 one 256 × 352 output tile
                 192 columns       160 columns
               ┌─────────────────┬────────────────┐
CTA 0: 128 rows│                 │                │
               ├─────────────────┼────────────────┤
CTA 1: 128 rows│                 │                │
               └─────────────────┴────────────────┘
                 two-CTA MMA pair; A collector reused

A/B global ──cluster TMA──▶ shared [5 stages] ──MMA──▶ TMEM
C global   ◀─────TMA────── shared [2 stages] ◀──FP16 + bias──┘
```

- **Cooperative MMA:** 256×192 and 256×160 outputs share A for each K16
  instruction pair; each mainloop stage covers K64.
- **Warp roles:** each CTA has 192 threads: four epilogue warps, one MMA
  warp, and one TMA warp.
- **Input pipeline:** five shared stages overlap cluster TMA and MMA.
  Every lane of the producer warp participates in each native CuTe copy.
- **Accumulator and epilogue:** 512 TMEM columns, then eleven 128×32 slices
  per CTA; bias/conversion and vector stores feed two shared C stages.
- **Persistent schedule:** clusters visit tiles in M-first order. The host
  uses the smaller of tile count and resident capacity; `--clusters` overrides it.
- **Shared memory:** 211,072 bytes per CTA, aligned to 1,024 bytes. Host
  tensor maps and device tiles use the same Rust layout types:

| Layout | Shared tile (rows × columns) | TMA swizzle |
| --- | --- | --- |
| `ASmem` | 128 × 64 | 128 bytes |
| `B0Smem` | 96 × 64 | 128 bytes |
| `B1Smem` | 80 × 64 | 128 bytes |
| `CSmem` | 128 × 32 | 64 bytes |

## Build and run

Requirements: an SM100 GPU, a compatible CUDA driver, and the repository's
configured cuda-oxide toolchain. From the repository root:

```bash
cargo oxide toolchain install cutlass
export CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir
cargo oxide build fp16_gemm_256x352_cute --arch sm_100a

# Exactly one kernel launch; verify its output afterward.
cargo oxide run fp16_gemm_256x352_cute --arch sm_100a -- \
  --mnk 256,352,64 --timing-mode direct --warmup 0 --iters 1

# N/K tails, row bias, persistent tiles, and five-stage ring reuse.
cargo oxide run fp16_gemm_256x352_cute --arch sm_100a -- \
  --mnk 512,712,392 --has-bias --clusters 1 \
  --timing-mode direct --warmup 0 --iters 1
```

- M must be a positive multiple of **256**; N and K, of **8**.
  TMA handles partial N352 and K64 tiles.
- The runner checks every output against CPU FP32 accumulation rounded to
  FP16 (`atol=0.1`, `rtol=1e-5`). Large shapes make this check expensive.
- Standalone timing defaults to **graph**. The commands above explicitly use
  **direct**: one kernel per CUDA-event pair, with no graph capture.
- `--warmup 0 --iters 1` in direct mode launches once total and verifies that
  timed output. Setup and verification are outside the event interval.
- `--json result.json` saves configuration, verification, and timing metadata;
  `--help` lists all options. Graph mode uses `--graph-launches` launches per sample.

## Rust API example

These declarations are taken from [the kernel](src/kernel.rs). The layout
parameters bind shared operands, tensor maps, and the paired MMA dimensions:

```rust
use cute_rs::{Composed, RowMajor, Sm100TiledMma, Sm100TmaMmaPipeline, Swizzle};

type ASmem = Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
type B0Smem = Composed<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
type B1Smem = Composed<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
type TiledMma = Sm100TiledMma<ASmem, B0Smem, B1Smem, 192, 160>;
type MainloopPipeline = Sm100TmaMmaPipeline<5, 77_824>;
```

- [SM100 APIs](../../cute-rs/src/sm100.rs): `Sm100SharedTile`,
  `Sm100ClusterTmaCopy`, `Sm100TiledMma`, `Sm100TmemEpilogue`, and pipelines.
- [Host setup](src/main.rs): typed TMA descriptors, cluster launch, and validation.
- [Dialect operations](../../dialect-cute/src/sm100_ops.rs): layout, collector,
  and pipeline verification before lowering.

## Regression checks

[validate.py](validate.py) uses only Python's standard library and runs six
cases covering full tiles, bias, tails, persistent scheduling, and pipeline
phase changes. It checks CPU results and graph replay output on Linux/SM100.

```bash
python cuteir/examples/fp16_gemm_256x352_cute/validate.py --sanitizer
```

- Build first; use `--binary PATH` if Cargo has a custom target directory.
- `--sanitizer` adds Compute Sanitizer memcheck/TMA and synccheck checks;
  omit it to run only the six cases.

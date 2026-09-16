# Official CUTLASS translation backend

Translate Rust CuTe contracts into NVIDIA's pinned CUTLASS 4.7 MLIR profile:

![CuTe compiler flow from Rust layouts through verified MLIR and CUTLASS to a CUDA cubin](assets/cutlass-translation-flow.svg)

- High-level CuTe survives preparation; reviewed mapping packs translate it
  directly, without native CuTe expansion or prior MIR/NVVM leaf lowering.
- Ordinary non-CuTe kernels retain the default MIR/NVVM/LLVM-to-PTX backend.

## Install and build

From the repository root after normal CUDA Oxide setup:

```bash
cargo oxide toolchain install cutlass
export CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir

cargo oxide build elementwise_cute --arch sm_120a
cargo oxide build nvfp4_gemv_cute --arch sm_120a
cargo oxide build blockscale_gemm_cute --arch sm_120a
cargo oxide build fp16_gemm_256x352_cute --arch sm_100a
```

- Installation verifies both archive and library digests. `cargo oxide`
  resolves the managed compiler and fingerprints its library digest.
- An explicit absolute path overrides discovery; library loading still
  enforces the pinned digest:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
CUDA_OXIDE_CUTLASS_COMPILER=/absolute/path/to/libCutlassCompiler.so \
  cargo oxide build elementwise_cute --arch sm_120a
```

## Supported mappings

| Example | Compiler-visible operations |
| --- | --- |
| [Elementwise](../examples/elementwise_cute) | Tensor tiling and copying |
| [NVFP4 GEMV](../examples/nvfp4_gemv_cute) | Scaled tensor views and GEMV |
| [Block-scaled GEMM](../examples/blockscale_gemm_cute) | Scheduler, work tiles, TMA pipelines, shared-memory MMA, epilogue stores |
| [SM100 FP16 GEMM](../examples/fp16_gemm_256x352_cute) | Typed shared tiles, TMEM, paired two-CTA MMA, cluster TMA, producer/consumer pipelines |

Block-scaled epilogue hand-offs: `ReadyForTma` emits an async-shared proxy
publication fence, then a counted CTA barrier; `Reusable` emits only the barrier.

## SM100 contracts

The [Rust APIs](../cute-rs/src/sm100.rs) become verified `cute.sm100_*` plans:

| API | Contract |
| --- | --- |
| `Sm100SharedTile`, `TmaDesc<f16, Layout>` | Operand role and matching host/device layout |
| `Sm100TiledMma` | Two N partitions sharing A's collector lifetime |
| `Sm100TmaMmaPipeline`, `Sm100AccumulatorPipeline` | Buffer ownership and completion |
| `Sm100Tmem`, `Sm100TmemEpilogue` | TMEM allocation and FP32 → FP16 epilogue |

- **Verified:** layout/partition compatibility, cluster transaction bytes,
  barrier arrivals, and allocation/MMA/copy/release lifecycle presence.
- **Caller-owned unsafe preconditions:** dynamic phases, lane participation,
  and pointer lifetimes.
- **Profile:** two CTAs; FP16 inputs, FP32 accumulation; M256/K64;
  128-byte input swizzle; 512 TMEM columns; 128×32 epilogue with 64-byte swizzle.
  The example uses N192+160 and checks CPU FP32 accumulation rounded to FP16.

### CuTe and the collector leaf

- Native CuTe builds shared descriptors, TMEM operations, and cluster TMA.
- CUTLASS 4.7's native MMA atom lacks A-collector fill/last-use control.
  The backend therefore emits an **NVVM TCGen05 MMA leaf** from the verified plan:

```text
Each K16:  first N partition → fill A collector
           second partition → last use of A collector
```

- Translation owns descriptor bits, instruction selection, barrier arrivals,
  completion multicast, and TMEM fences.

### Collective TMA loads

- `Sm100ClusterTmaCopy::copy`: all 32 producer lanes per CTA participate,
  with identical arguments within each warp; native CuTe elects internally.
- An outer elected-lane branch can deadlock on stage reuse. Regression cases
  cover more than five K stages and multiple persistent tiles.
- TMA **stores** have one issuing thread; the collective load rule does not apply.

## Diagnostics and limits

- `CUDA_OXIDE_DEVICE_BACKEND` accepts `native` or `cutlass-mlir`; unknown values
  fail. `CUDA_OXIDE_BACKEND` selects the rustc backend library, not this path.
- Unsupported semantics/profiles, verification failures, or compiler failures
  stop the build; translation does not fall back to another backend.
- To inspect prepared MLIR, add `CUDA_OXIDE_MLIR_OUTPUT="$PWD/module.mlir"`
  to a CUTLASS build. MLIR dumps and generated cubins are build artifacts;
  do not commit them.

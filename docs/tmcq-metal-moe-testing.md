# TMCQ Metal MoE (segmented) paths: notes for the Mac test run

Branch `wt/tmcq-metal-moe`. Written on Linux without Metal, so none of this has been compiled by
the Metal compiler or run on a GPU. Before writing this, the generated MSL for the segmented GEMV,
the segmented SwiGLU and the prefill block decode was type-checked with g++ against a mock MSL
header. It was also run serially (lane by lane, with emulated `simd_shuffle_xor` and threadgroup
variables) against an independent C decode of the TMCQ spec, and matched exactly in the integer
codes and to 1e-6 in the outputs. The Mac run is the first real test.

## What changed

| Path | Kernel | TMCQ handling |
| --- | --- | --- |
| MoE decode, `SEGMENTED_GEMM_FORWARD` with one row per bin | `SegmentedInt8GemvKernel` (`segmented_int8_gemv`) | New TMCQ kernel. 4 SIMD groups per threadgroup; each SIMD group owns one 16-row tile row of the routed expert (64 rows per threadgroup). Lanes 0-15 take rows 0-15 of the even tiles along K, lanes 16-31 the same rows of the odd tiles, so the 16 lanes of a half share each activation load and each tile's words. One `simd_shuffle_xor(16)`, then the row scale (and bias) is applied. |
| MoE decode, `SEGMENTED_SWIGLU_FORWARD` (vector path) | `SegmentedInt8SwiGLUKernel` (`segmented_int8_swiglu`, `_shared`) | Same mapping for gate and up. Each projection's full dot product times its row scale comes first, then the clamp and SiLU. The route weight is loaded before the projections, as the 257-expert workaround requires. The fused shared-expert kernels (Q8_0 / Q5_K / Q6_K shared rows) keep their 4-SIMD-group layout and work unchanged. |
| MoE prefill, `SEGMENTED_GEMM_FORWARD` / `SEGMENTED_SWIGLU_FORWARD` with neural accelerators | `SegmentedScaledGEMMKernel` threadgroup path (BLOCK 16 or 64 x 128 x 128) | Fused. The 128 x 128 weight block is 8 x 8 tiles. Each step, every thread decodes one 16-weight tile row (8 per thread per K block) into 16 bytes of the `[N][K]` threadgroup block, using `tmcq_decode` from `TMCQDecodeSource.hpp`. The register path (32 x 64 x 64) is not used for TMCQ. |
| Longer prefill (more than 64 rows per bin), or no neural accelerators | unfused | Expand the selected experts to int8 (`Dequantize8iRowwiseXSelected`), then Int8MatMul / NA GEMM. This is E1's decoder. |
| Shared expert / dense `SWIGLU_FORWARD` | unchanged | Decode to int8 or FP (E1's `Dequantize8iRowwiseX*`), then GEMM. |
| MoE weights streaming | host only | Each expert slot must be whole tile rows (`dim[1] % 16`, `dim[2] % 256`, no modifier bits), otherwise the layout is rejected. |

The fused-versus-expanded cutoff for TMCQ (`SegmentedScaledGEMMKernel::fusesPackedFormat`) uses the
same rule as the shared group decoder: fuse while `M <= 64 * bincount`. That cutoff has not been
measured for TMCQ; see the benchmark section.

### Dependencies on E1 (Dequantize\*)

These paths call code that E1 owns. They fail a `CCV_NNC_MFA_PRECONDITION` (or decode wrongly) until
E1's TMCQ support is merged:

- `Dequantize8iRowwiseXSelectedDescriptor::inputScaleOffset()` / `groupSize()` / `groupBits()`, which
  `ccv_nnc_mfa_segmented_scaled_gemm.cpp` and `ccv_nnc_mfa_segmented_scaled_swiglu.cpp` use to find
  the row scales of a TMCQ expert bank. This affects the **fused prefill path too**, not only the
  unfused one.
- `Dequantize8iRowwiseX` / `...Selected` / `...FP` TMCQ decode: the unfused prefill, the
  no-neural-accelerator paths, `SWIGLU_FORWARD` and the generic segmented GEMM fallback.

The decode paths (GEMV and the SwiGLU vector path) depend only on this branch.

E1 committed this in `wt/tmcq-metal-dense` 9861d630. Its descriptors use a 256-weight group of
16n bits, which gives the same scale offset as this branch. The two branches change disjoint files
(checked with `git merge-tree`). **Build this branch merged with 9861d630** before running the
prefill tests. E1's own cases are in `test/int/nnc/tmcq_mps.tests.c`.

All TMCQ paths require the weight's offset to be a multiple of 4 bytes, because the kernels read
tiles as 32-bit words. The segmented GEMM / GEMV / SwiGLU hosts reject or reroute a misaligned
weight; the MFA encoders and descriptors also assert the offset and the expert stride.

## Build (commands M100 has confirmed on the Mac)

```sh
./configure --enable-mps
cd lib && make -j 4 lib            # not `all`: it downloads models and there is no wget
cd ../test/int/nnc
make -j 4 -o libccv.a segmented_swiglu.tests mpsblas.tests moe_weights_streaming.tests
```

(If `make lib` must run from the repo root on your setup, run it there, as before.)

Baseline before these changes: segmented_swiglu 24/24, mpsblas 322/322 (277 s).

## Tests

Every new case is guarded by `ccv_nnc_cmd_ok(..., CCV_NNC_BACKEND_MPS)`, like the existing MPS
cases, so none is skipped silently on the Mac. Each case has "TMCQ" in its name, and the runner
matches its argument as a substring:

```sh
cd test/int/nnc
./segmented_swiglu.tests TMCQ
./mpsblas.tests TMCQ
./moe_weights_streaming.tests TMCQ
# then the full files, to check nothing else regressed
./segmented_swiglu.tests
./mpsblas.tests
./moe_weights_streaming.tests
```

| File | Case | Covers |
| --- | --- | --- |
| segmented_swiglu.tests.c | `MPS segmented SwiGLU decodes TMCQ tiles for one row and short routed batches` | Vector path. All 9 rates in FP32 at M = 1 and M = 6; N16 / N18 / N32 in FP16 at M = 1, 6 and 18, plus broadcast input with clamp 0.25. Checked per row against the spec C decode (relative L2 below 1e-4 in FP32, 5e-3 in FP16). |
| segmented_swiglu.tests.c | `MPS segmented SwiGLU TMCQ prefill matches the spec reference at 64 and 512 rows` | Matrix paths, FP32, with and without neural accelerators. With them, M = 64 is the fused tile decode and M = 512 the expanded experts. The reference uses absmax-quantized activations, as the existing packed prefill cases do. |
| mpsblas.tests.c | `mps segmented int8 gemv with TMCQ weights` | GEMV, M = 1, all 9 rates, with and without bias; 8 routes out of 11 experts (N = 512, K = 768) in FP32 / FP16 / BF16. |
| mpsblas.tests.c | `mps segmented gemm with TMCQ weights at 64 and 512 rows` | Segmented GEMM prefill, N16 / N18 / N24 / N32, FP16 / FP32 / BF16, with and without neural accelerators, with guard rows. With them, both sizes fuse (16-row tiles at 64 rows, 64-row tiles at 512). |
| moe_weights_streaming.tests.c | `MPS MoE weights streaming stages TMCQ experts by whole tile rows` | Streamed weights against the same resident weights: decode across misses, hits and eviction; a resident 6-row call; prefill at 64, 512 and 3072 rows (staged, then full load). |

The CPU reference is a C function in each test file, written from section 3 of the spec. It does
not call `ccv_nnc_dequantize_8i_rowwise_x`, because the library's CPU TMCQ codec (agent A,
`.worktrees/cpu`) is not merged yet. Once it lands, the decode can be cross-checked against it, and
the shared-expert fusion test (`_segmented_swiglu_shared_case`, which quantizes on the CPU) can add
TMCQ routed weights.

Read failures this way: a GEMV or SwiGLU **decode** failure is in this branch. A prefill failure
with `fallback=1` / no neural accelerators, or at M = 512 in segmented SwiGLU, most likely comes
from the E1 decoder. A failure at `fallback=0` and 64 rows comes from the fused block decode in
`SegmentedScaledGEMMKernel.cpp`, or from E1's scale offset.

## Benchmarks

`bin/nnc/segmented_swiglu_bench.c` and `bin/nnc/routed_segmented_gemm_bench.c` take `iq1_s`,
`iq2_xxs`, `tmcq_n18`, `tmcq_n22` and `tmcq_n26` (n18 / n22 / n26 are 1.125 / 1.375 / 1.625 bits per
weight). Every TMCQ and IQ bit string is a valid payload, so the benches time their random or
uninitialized weights directly and need no quantization pass.

```sh
cd bin/nnc && make -j 4 -o ../../lib/libccv.a segmented_swiglu_bench routed_segmented_gemm_bench

# Routed segmented GEMM (gate 2048x4096, up, down 4096x2048, 256 experts, FP16 unless "fp32").
# "tmcq" runs iq1_s, iq2_xxs, tmcq_n18, tmcq_n22 and tmcq_n26 in turn.
# Arguments: routing format warmup iterations tokens [fp16|fp32].
./routed_segmented_gemm_bench sparse6 tmcq 3 12 1          # decode: 6 rows, one per bin -> segmented GEMV
./routed_segmented_gemm_bench sparse6 tmcq 3 12 6          # 6-token verify
./routed_segmented_gemm_bench balanced tmcq 3 12 64        # short prefill: fused TMCQ tile decode
./routed_segmented_gemm_bench balanced tmcq 3 12 512       # fused while M <= 64 x bins
./routed_segmented_gemm_bench balanced tmcq 3 12 2048      # long prefill: expands the selected experts
./routed_segmented_gemm_bench balanced tmcq 3 12 512 fp32  # FP32 (Int8MatMul without neural accelerators)

# Segmented SwiGLU decode (vector path), FP16, n = 2048, k = 4096.
# Arguments: format warmup iterations batch experts rows segments.
for f in iq1_s iq2_xxs tmcq_n18 tmcq_n22 tmcq_n26; do ./segmented_swiglu_bench $f 5 30 8 6 6 6; done
for f in iq1_s iq2_xxs tmcq_n18 tmcq_n22 tmcq_n26; do ./segmented_swiglu_bench $f 5 30 8 64 1 1; done   # one row

# Segmented SwiGLU FP32 prefill (256 experts), now with a format: warmup iterations tokens format.
for f in rowwise iq1_s iq2_xxs tmcq_n18 tmcq_n22 tmcq_n26; do ./segmented_swiglu_bench prefill_fp32 2 7 2048 $f; done
for f in iq1_s iq2_xxs tmcq_n18 tmcq_n22 tmcq_n26; do ./segmented_swiglu_bench prefill_fp32 2 7 64 $f; done
```

Each SwiGLU decode run also prints a `validation,...` line comparing the fused kernel with the
composed segmented GEMM path. For TMCQ, that line also cross-checks the new SwiGLU kernel against
the new GEMV kernel.

What to look for:

- **Decode (T = 1 / 6):** TMCQ against IQ2_XXS and IQ1_S. TMCQ moves fewer bytes but decodes with
  more ALU work (two loads, a multiply and a byte sum per weight).
- **Prefill (T = 64, 512, 2048):** whether TMCQ's fused / expanded cutoff (`M <= 64 x bins` in
  `SegmentedScaledGEMMKernel::fusesPackedFormat`) sits in the right place. To measure both sides at
  one size, temporarily make the TMCQ branch return `true` or `false`.

## Where compile or numeric failures are most likely

1. `char4(int4(...))` in the fused prefill decode (`SegmentedScaledGEMMKernel.cpp`, the TMCQ
   branch of the K loop). If the Metal compiler rejects the vector conversion, use
   `char4((char)a, (char)b, (char)c, (char)d)` as the group decoder does.
2. `float4(int4(...))` in `tmcq_codes4` (GEMV and SwiGLU kernels). This is the same kind of
   conversion; the fallback is `float4((float)a, ...)`.
3. Program-scope `constant constexpr uint TMCQ_N` / `TMCQ_TILE_WORDS` / `S`, passed as the `uint n`
   argument of `tmcq_decode`. The existing Q2_K kernel uses `constant constexpr` the same way.
4. Prefill code size: the TMCQ block decode is fully unrolled (8 tile rows x 16 `tmcq_decode` per
   thread per K block). If compilation is slow or registers spill, drop `unroll(full)` on the outer
   loop.
5. Scale offsets in the prefill path come from E1's `Dequantize8iRowwiseXSelectedDescriptor`. If
   E1's TMCQ group bits differ from 16n, results are wrong while the fused decode itself is right.
6. `segmentedInt8GemvSIMDGroupsPerThreadgroup` now returns 4 for TMCQ. The host grid is
   `ceil(N / 64) * M` threadgroups and the kernel derives the same count from `nrows / 16`; the two
   must stay in step.
7. Performance rather than correctness: each `tmcq_decode` does two device loads and a modulo
   (not a power of two for odd n / 2). It relies on the tile words staying in L1. If the decode is
   slow, keep the tile's words in registers per lane (a tile row's 16 windows span at most n + 15
   bits). That needs a helper over thread memory in `TMCQDecodeSource.hpp`.

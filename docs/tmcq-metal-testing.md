# TMCQ Metal (dense path) test notes for M100

Branch `wt/tmcq-metal-dense`. This code was written on Linux and has never been compiled. Expect the
first build to surface compile errors; the list at the end names the most likely spots.

## Build

From the repo root:

```sh
cd lib
./configure --enable-mps
make -j 4 lib          # not `make all`: it downloads models and the Mac has no wget
cd ../test/int/nnc
make -j 4 -o libccv.a tmcq_mps.tests mpsblas.tests   # -o libccv.a: do not rebuild the library each time
```

After editing library sources, rerun `make -j 4 lib` in `lib/` and then the test `make` line. If a
build fails oddly, delete stale `.dep.mk` files (`find . -name .dep.mk -delete`) and retry.

Benchmark:

```sh
cd bin/mfa
make -j 4 scaled_gemv_bench
```

## Tests to run

```sh
cd test/int/nnc
./tmcq_mps.tests
./mpsblas.tests        # regression: 326/326 (see the regression record below)
./segmented_swiglu.tests   # regression: 27/27
```

Every GPU case in `tmcq_mps.tests` is guarded with
`GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(..., CCV_NNC_BACKEND_MPS))` and compiled under `HAVE_MPS`, so on the
Mac none of them should be skipped. Expected: `all test case(s) passed, congratulations!` with 10 cases:

| Case | What it checks |
| --- | --- |
| `tmcq reference decode fixed vectors` | The test's own C reference decode (spec section 3) on hand-derived vectors: zero payload -> -127, newest bit is codeword bit 14, tail-biting wrap. CPU only. |
| `mps tmcq int8 decode matches the reference bit for bit` | `Dequantize8iRowwiseX` (prefill decode, int8 rows + scale copy) for all 9 rates x 32F/16F/16BF, 48 x 768 random payload. Exact int8 match and exact scale bytes. |
| `mps tmcq int8 decode of 7168-wide rows matches the reference bit for bit at every rate` | Same int8 decode for all 9 rates at 32 x 7168 (fp16 scales). |
| `mps tmcq fp decode matches the reference bit for bit` | `Dequantize8iRowwiseXFP` for all 9 rates x 3 dtypes, random scales, compared bit for bit with `round_to_nearest_even(scale * q8)`. |
| `mps tmcq selected expert int8 decode matches the reference` | `dequantize_8i_rowwise_x_selected` (MoE expert decode), 3 experts x 32 rows; expert 0 has an empty bin and must keep the 0x5a sentinel. |
| `mps tmcq gemv for 1 to 3 rows matches the reference` | GEMM command with M = 1, 2, 3 (TMCQ GEMV kernel), N = 1040 (N % 16 but not % 256), K = 2048, n = 16/22/32, 3 dtypes, with and without bias. Compared to a CPU double reference and to the dense GPU GEMM on the reference-decoded weights. Tolerance (mpsblas max_rel metric): 1e-4 fp32, 4e-3 fp16, 2e-2 bf16. |
| `mps tmcq gemv covers K from 256 to 7168 and N from 16 to 4096` | GEMV (M = 1, 2, 3) for K = 256, 512, 4096, 7168 x N = 16, 1040, 4096 x n = 16/22/32 x fp16/fp32, bias alternating; then all 9 rates at N = 1040, K = 7168. Same references and tolerances as the case above. |
| `mps tmcq gemm prefill covers K 256 and 7168 with partial row blocks` | GEMM with M = 16, 64, 65 (65 leaves one row in the last block), N = 528, K = 256 and 7168, n = 16/22/32, fp16, the three routing passes. Passes 0 / 1 (int8 scaled GEMM, activations quantized to int8) are bounded by 2e-3 x max(1, sqrt(K / 1024)) against the dense GPU GEMM: that error grows like sqrt(K) (measured about 6e-4 at K = 256 and 4e-3 at K = 7168, the same for every rate). Pass 2 (fp decode) must equal the dense GEMM exactly. vs CPU < 1e-2. |
| `mps tmcq gemm decodes then multiplies for prefill rows` | GEMM with M = 64 and 512, N = 528, K = 1024, n = 22, fp16/fp32, three routing passes: default (ANE / ANE + GPU / NA), ANE disabled, ANE and neural accelerators disabled. vs dense GPU < 2e-3, vs CPU < 1e-2. |
| `mps tmcq index select is rejected` | IndexSelect on a TMCQ table returns `CCV_NNC_EXEC_INVALID`. |

Shape coverage of the routed (MoE) kernels lives in the other two suites:

| Suite / case | What it checks |
| --- | --- |
| `mpsblas.tests` `mps segmented int8 gemv with TMCQ weights at K 256 and 7168 over 1 and 64 experts` | Segmented GEMV, N = 256, K = 256 and 7168, n = 16/22/32, fp16: one route on 1 or 64 experts, and several routes (4 rows on the only expert, 6 rows over 6 of 64 experts, with bias). max_rel < 3e-3 against the spec reference. |
| `mpsblas.tests` `mps segmented gemm with TMCQ weights at K 256 and 7168 with partial row blocks` | Segmented GEMM with 16, 64 and 65 rows over 6 of 8 experts with guard rows, K = 256 and 7168, with and without neural accelerators. |
| `segmented_swiglu.tests` `MPS segmented SwiGLU TMCQ decode covers K 256 and 7168 over 1 and 64 experts` | SwiGLU decode, N = 256, K = 256 and 7168, n = 16/22/32: one route (fp32), and 6 rows over 6 of 64 experts or 3 rows on the only expert (fp16, and fp32 with a broadcast input and clamp 0.25). The decode path takes at most 3 rows per bin (M <= 3 x bincount); more rows go to prefill, where fp16 activations are quantized to int8 and a broadcast input is not supported. |

K split remainder: every TMCQ GEMM / GEMV / SwiGLU path requires K % 256 == 0, so K / 16 is a multiple
of 16 and each of the 2 x 4 streams over a tile row runs the same number of steps. A K / 16 that the
4 SIMD groups do not divide cannot reach these kernels; K = 256 (2 steps per stream) is the shortest
split and is covered above.

Once the CPU TMCQ decode from the `wt/tmcq-cpu` branch is merged in, also run

```sh
TMCQ_CPU_REFERENCE=1 ./tmcq_mps.tests
```

which additionally requires the GPU fp decode to equal `ccv_nnc_dequantize_8i_rowwise_x` on CPU
memory byte for byte. Without that branch the CPU decode does not know TMCQ, so leave the variable
unset.

Debug knob if something fails: `CCV_METAL_LOG_LEVEL=3 ./tmcq_mps.tests` logs which GEMM route ran.

## Benchmark

`kPackedFormats` now includes `iq1_s` and `tmcq_n22`. Compare TMCQ against IQ1_S and IQ2_XXS:

```sh
cd bin/mfa
for f in tmcq_n22 iq1_s iq2_xxs; do
  for m in 1 2 3; do
    ./scaled_gemv_bench --dtype fp16 --format $f --mrows $m --shape 4096 4096 --shape 5120 17408 --shape 17408 5120
  done
done
```

Each shape prints one block, e.g.:

```text
shape mrows=1 rows=4096 cols=4096 dtype=fp16 format=tmcq_n22 bias=1 dispatches=5 runs=20 sleep_ms=200 rowwise_bytes=16785408 packed_bytes=11542528 compression=...
  dequant_vs_raw max_abs=... max_rel=...
  scaled_vs_raw max_abs=... max_rel=...
  packed_direct_vs_packed_dequant max_abs=... max_rel=...
  raw_gemv avg_ms=... best3_avg_ms=... median_ms=... min_ms=... max_ms=... tflops=... effective_tb_s=...
  dequant_gemv ...
  rowwise_int8_gemv ...
  dequant_x_only ...
  packed_dequant_int8_gemv ...
  packed_direct_int8_gemv ...
  packed_direct_vs_rowwise_best3=... packed_direct_vs_dequant_best3=... packed_direct_time_over_rowwise=...
```

Please report, per format, shape and mrows: `packed_direct_int8_gemv best3_avg_ms` and
`effective_tb_s`, `dequant_x_only best3_avg_ms`, and `packed_direct_vs_packed_dequant max_rel`
(TMCQ GEMV against TMCQ int8 decode + rowwise GEMV; should be at fp16 rounding level, about 1e-3 or
below). The bench payload is random bytes, which is valid for both TMCQ and IQ1_S.

## Thread mapping (for reading the kernels)

- `Dequantize8iRowwiseX` / `XFP` / `_selected` (TMCQ): one thread per row segment, i.e. the 16
  weights of one output row inside one tile. Thread x -> row x / (K/16), tile column x % (K/16), so a
  SIMD group writes 32 consecutive 16-element pieces of one row (contiguous stores; the output is the
  heavy side) and the 16 threads of a tile reread the same n/2 words from cache. Dispatch items are
  rows * K / 16 (max with the scale bytes for the int8 kernel).
- `Int8Gemv` (TMCQ): one threadgroup (4 SIMD groups) per 16-row tile row. Each half SIMD group is a
  stream over tile columns (stream s takes columns s, s + 8, ...); lane r of the half computes row r
  of each tile. The 16 activations of a tile column are a broadcast load shared by the 16 rows.
  Reduce: xor-16 shuffle within the SIMD group, threadgroup memory across the 4 SIMD groups, then
  multiply the row scale (and add bias).

## Most likely compile failures

1. `Int8GemvKernel.cpp` TMCQ kernel: `threadgroup float partials[S * M * 16];` declared inside the
   kernel with `constexpr` sizes, and `float sum[M]` loops; `simd_shuffle_xor(sum[m], 16)`.
2. `Dequantize8iRowwiseXFPKernel.cpp`: `vec<real, 4>(float4(...) * scale)` for `real = bfloat`
   (needs Metal 3.1 bfloat vectors; the H256 kernel in the same file already uses `vec<real, 4>`).
3. `reinterpret_cast<device const uint*>(source)` from `device const uchar*` in all three TMCQ kernels.
4. Unused function constants (`group_size`, `groups_per_row`) in the TMCQ decode libraries: allowed in
   MSL, but a pipeline creation error about unspecified constants would point here.
5. Host C++: the `is_tmcq` helpers live in anonymous namespaces of each .cpp; `CCV_NNC_QX_8I_ROWWISE_TMCQ_*`
   must be visible there (they come from `nnc/ccv_nnc.h` via `ccv_nnc_mfa.hpp`).
6. `tmcq_mps.tests.c`: direct calls to `ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_selected` and
   `ccv_nnc_mfa_request_scratch` from C, and `GPU_TENSOR_NHWC(000, 8U, ...)` byte tensors.

## Regression record

`rowwise-x-trellis` at `c654c644`, Mac Studio M5 Ultra, 2026-11-12. Each binary rebuilt against the
current library, then run from `test/int/nnc`:

```sh
cd test/int/nnc
make -j 4 -o libccv.a mpsblas.tests segmented_swiglu.tests palettize.tests moe_weights_streaming.tests tmcq.tests tmcq_mps.tests
./mpsblas.tests
./segmented_swiglu.tests
./palettize.tests
./moe_weights_streaming.tests
./tmcq.tests
./tmcq_mps.tests
```

| Suite | c654c644 | With the shape coverage above |
| --- | --- | --- |
| `mpsblas.tests` | 324 / 324 | 326 / 326 |
| `segmented_swiglu.tests` | 26 / 26 | 27 / 27 |
| `palettize.tests` | 20 passed, 29 skipped | 20 passed, 29 skipped |
| `moe_weights_streaming.tests` | 3 / 3 | 3 / 3 |
| `tmcq.tests` | 11 / 11 | 11 / 11 |
| `tmcq_mps.tests` | 7 / 7 | 10 / 10 |

The 29 skipped `palettize` cases are guarded on `CCV_NNC_BACKEND_GPU_REF` (CUDA only).

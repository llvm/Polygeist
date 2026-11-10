*MFEM 25x4 by 4x5 cuBLASDx experiment*

This experiment evaluates one batch of 1,024 independent MFEM-shaped
contractions, `C[25,5] = A[25,4] * B[4,5]`, on Jetson AGX Orin (SM87).
Timing is device kernel time only: 20 warm-up launches followed by 500 timed
launches per process, repeated in 10 processes. The values below are medians
of those 10 process-level means.

Toolchain:

- CUDA 12.6, cross-compiled for AArch64 and `sm_87` on the x86 host
- NVIDIA MathDx 25.12.0, cuBLASDx 0.5.0 CUDA 12 package
- cuBLASDx specification: TF32 A/B, FP32 accumulation, row-major A/B/C,
  `Size<25,5,4>`, `Block`, `SM<870>`
- cuBLASDx-selected launch: 128 threads and 480 bytes dynamic shared memory

Median results:

```
implementation                    median us   versus scalar
scalar FP32 CUDA                       6.409       1.000x
direct TF32 mma.m16n8k4                6.254       1.025x faster
cuBLASDx 0.5.0 TF32                    9.509       1.484x slower
padded WMMA 16x16x8                   11.772       1.837x slower
```

The cuBLASDx result passed the complete 128,000-element output comparison
against the scalar FP32 reference: maximum absolute error `3.0875206e-4` and
relative L2 error `4.44360378e-4`. The high maximum elementwise relative error
(`0.3069`) occurs at a reference value close to zero and is not used alone as
the acceptance criterion.

SASS inspection of the final cuBLASDx binary found one static
`HMMA.1684.F32.TF32` instruction in the cuBLASDx kernel. Thus, cuBLASDx did
select a tensor-core implementation; its loss here is overhead for staging and
executing an unusually small, skinny contraction, not a fallback to CUDA-core
FMA. The scalar reference contains four static `FFMA` instructions.

Build command (the relaxed-constexpr flag is required by this cuBLASDx release):

```
/usr/local/cuda-12.6/bin/nvcc -O3 -DNDEBUG --expt-relaxed-constexpr \
  -std=c++17 -arch=sm_87 \
  -ccbin aarch64-linux-gnu-g++ --target-directory sbsa-linux \
  -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include \
  mfem_m25n5k4_cublasdx.cu -o mfem_m25n5k4_cublasdx
```

The binary was copied to
`/home/nvidia/polygeist-results/mfem_m25n5k4_cublasdx` and executed on the
Orin. No CUDA compilation was performed on the board.

Raw measurements are retained in:

- `results/mfem_m25n5k4_cublasdx_orin_raw.txt`
- `results/mfem_m25n5k4_manual_orin_rerun_raw.txt`

This is a handwritten library experiment. It is not yet an automatic
Polygeist lowering and must not be reported as one.

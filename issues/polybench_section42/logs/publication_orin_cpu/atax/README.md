# Publication Orin CPU campaign: atax

Native AArch64 CPU correctness and timing are complete at canonical
LARGE/FP64. The GCC 11 `-O3` executable passes all 2,100 values exactly
against the retained canonical reference. Its object disassembly proves the
benchmark entry calls `kernel_atax` rather than an inlined or eliminated copy.

- Native CPU median: 6.709 ms (6.687–6.779 ms, IQR 0.032 ms).
- Raised residual CPU median: 6.081280 ms (6.070496–6.141952 ms,
  IQR 0.055232 ms), 1.1032x versus native.
- Raised OpenBLAS CPU median: 4.856864 ms (4.820800–4.953024 ms,
  IQR 0.031200 ms), 1.3813x versus native.
- Execution: Orin CPU core 0, one process, five warmups plus five samples.
- Threading: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.
- Build placement: AArch64 cross-compilation on the x86 host only.

Both raised configurations pass all 2,100 values exactly. The residual binary
uses the retained compiler-generated LLVM IR and generated ABI wrapper. The
library binary contains compiler-generated calls to real AArch64 OpenBLAS
`cblas_dgemv`; no project-authored computational kernel is present. The main
OpenBLAS archive was incomplete only in its thread-server members, so the link
also uses those real members from the retained companion OpenBLAS archive.

Fixed power/clock/fan state was not reported by the runner, so these timings
remain publication-pending.

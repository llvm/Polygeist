NPB CG Class S five-route application validation, 2026-09-07 PDT

- Implementation branch: `audit/ginsbach-reductions`
- Original application: SNU NPB 3.3.1 serial C, CG Class S
- Composition: original driver with only the file-local `conj_grad` definition
  replaced by the source-faithful Polygeist-raised helper
- Distinct static helper sites: 5
  - 2 cuSPARSE CSR SpMV
  - 2 cuBLAS Ddot
  - 1 CUB squared-L2 transform-reduce
- ABI lowering: 5 runtime calls and zero residual `kernel.launch` operations
- Build location: x86 host only
- Target: AArch64, CUDA 12.6, `sm_87`
- Jetson: Orin #2 through `pva-general`
- Correctness: NASA `VERIFICATION SUCCESSFUL`, 3/3
- Final zeta: `8.5971775078648E+00` in every run
- Relative error: `1.0331046659065E-15` in every run
- Benchmark-reported times: 0.08 s, 0.08 s, 0.08 s
- Mop/s: 839.97, 876.95, 855.66; median 855.66
- Process elapsed: 0.286 s, 0.275 s, 0.275 s; median 0.275 s
- Earlier three-route retained run: median 501.82 Mop/s and 0.397 s process
  elapsed. The historical comparison is same-board and same Class S, but not
  interleaved; treat the apparent improvement as indicative, not a controlled
  performance result.
- Whole-file structural count: 8 matcher launches (4 SpMV, 3 Ddot, 1
  squared-L2) after `conj_grad` is inlined at its call sites. This is not eight
  distinct source occurrences. The linked application contains the five
  distinct static helper sites listed above.
- Executable SHA-256:
  `298ad9b7959a957a520a11054eb0eafe6e75d81d1ff0bbd7216169efc0c62959`
- CUB companion SHA-256:
  `6deee5084f502febc7831a47b92130ef665822225b4850152cea5ed8ecba9d6b`
- Raw retained log:
  `scripts/correctness/logs/npb_cg_five_routes_20260907_20260907_173727.silicon.log`
- No source, MLIR, or CUDA compilation occurred on the Jetson.

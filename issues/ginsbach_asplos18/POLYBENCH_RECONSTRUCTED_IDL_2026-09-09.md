# Reconstructed Ginsbach IDL matcher on PolyBench/C

This is an exploratory recognition-only run of the retained reconstructed
IDL/CAnDL matcher. It is not a reproduction of the unavailable ASPLOS'18
artifact and provides no generated-library runtime results.

## Configuration

- Input: 30 canonical PolyBench/C translation units from this repository.
- Compiler: reconstructed Clang/LLVM 6 build at commit
  `e1fc2eb154550adfd54a5fda943ad92e90411d48` with the unchanged author Clang
  commit `818167bc99a0f5538a2676bf876e5c598d1b449b`.
- Language: C99, compiled from unmodified source with benchmark and PolyBench
  utility include directories.
- Profiles: the public-demo `-O2` profile and the thesis-documented
  `-Os -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -ffast-math`
  profile. Both produced identical result rows.
- Timeout: 120 seconds per translation unit; no timeouts occurred.

## Result

- 30/30 translation units compiled.
- 17 raw solutions across 13 programs.
- 11 scalar reductions, 4 generalized histograms, 2 stencils, 0 GEMMs, and
  0 SpMVs.
- One reduction is in Nussinov's output-print loop and is excluded from
  computational coverage. The audited computational result is therefore 16
  solutions across 12/30 programs.

The four `Histo` results are array-element accumulations in GEMVER, MVT, and
Cholesky. They are consistent with the reconstruction's broad generalized
histogram specification, but are not data-dependent integer histograms such
as `histogram[data[i]] += 1`.

The reconstruction failed to classify PolyBench GEMM, 2MM, 3MM, SYMM, SYRK,
SYR2K, or TRMM as GEMM. This is notable because these are direct dense-linear-
algebra opportunities. It also recognized only Heat-3D among the PolyBench
stencil programs. These misses reinforce the existing finding that the
retained specifications and CC'18 engine are not equivalent to the missing
ASPLOS'18 implementation.

The normalized kernel-level results are in
`polybench_reconstructed_idl_2026-09-09.csv`. Full LLVM modules, JSON
constraint solutions, commands, and diagnostics remain in the local `/tmp`
run directories recorded during the experiment.

# Name-independence audit

Last updated: 2026-09-07 PDT

## Non-negotiable rule

Library matching is semantic. It may inspect operations, dataflow, memory and
indexing structure, loop bounds, control flow, types, constants, and proven
equivalences. It may not inspect benchmark/suite identity, paths, source file
names, source function or symbol names, source text, line numbers, argument
positions, or manifest identity to choose a match or replacement.

The required metamorphic property is rename and relocation invariance: changing
only a function name or source path cannot change whether or how it matches.

## Removed production paths

- NPB FT direct `fftXYZ` cuFFT substitution and application builder.
- NPB EP assertion-checked source-text rewrite for fused Gaussian statistics.
- NPB MG `resid` / `psinv` / `rprj3` renderer, dedicated ABI route, and
  application builder.
- NPB CG and IS application builders that deleted an original function and
  linked a hand-selected extracted replacement.
- ATen name-specific renderers for bilinear upsampling, compressed-block
  conversion, all/any, NaN sum, sparse norm, joint scaling, arg-reduction,
  fixed pooling backward, sparse index conversion, quantized column offsets,
  adjacent difference, and embedding-bag counts.
- Positive matcher tests whose only justification was one of the removed
  name-specific paths.
- PolyBench and GEMM application builders that weakened `kernel_*` symbols so
  a selected wrapper or CUDA adapter won at link time. Their raised CPU,
  PolyBenchGPU, and raised GPU application results are excluded; only the
  untouched native-C CPU measurements remain valid.
- Parboil SGEMM and stencil full-application bridges that omitted the original
  computational helper and linked a selected transformed implementation.

## Allowed boundary behavior

An ABI/test wrapper may invoke the transformed function under test. It must be
algorithm-neutral and cannot choose a library call. Internal runtime symbols
selected *after* semantic recognition are implementation details and may have
descriptive names.

## Result classification

- FT: 0 compliant automatic matches.
- Directly evidenced EP sites: 0/3; one additional inferred final sum is a
  structural CUB match and is kept separate.
- MG: 0 compliant executable stencil matches.
- CG, IS, and UA application composition correctness/performance: excluded.
- Stale generated ATen and Ginsbach artifacts: not claimable until regenerated.

The current viewer and status CSVs explicitly display these exclusions.

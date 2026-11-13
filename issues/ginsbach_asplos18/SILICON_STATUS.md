# Ginsbach cross-compiled silicon status

Last updated: 2026-09-07 PDT

## Authoritative recognition rule

A result counts only when the library/API is selected from operations, dataflow,
memory/indexing structure, loop bounds, control flow, types, and proven
semantics. Benchmark identity, paths, source/function/symbol names, source
text, line numbers, argument positions, and manifests may not influence the
match. Renaming or relocating an unchanged computation must leave the result
unchanged.

Deleting or omitting an original function and linking a hand-selected adapter
is a manual composition, not an automatic end-to-end transformation. Such
results are excluded from match, correctness, silicon, and performance totals.
Neutral wrappers are permitted only to invoke an already transformed entry
point; they may not choose the algorithm or library.

All Jetson artifacts must be cross-compiled on the x86 host. The target is Orin
#2 through `<configured SSH hop>`, `<configured Jetson target>`, AArch64 / `sm_87`.

## Retractions from the strict audit

- **FT — excluded:** `fftXYZ` was directly replaced by a cuFFT adapter. The
  builder, adapter, tests, reports, and new run logs were removed. FT has zero
  compliant automatic matches and no reportable GPU performance result.
- **EP fused statistics — excluded:** an exact source-text rewrite selected the
  Gaussian sums and histogram. That route and its reports/logs were removed.
  The directly evidenced `q[l]`, `sx`, and `sy` occurrences remain 0/3.
  The separately inferred final `gc` reduction has one name-independent CUB
  sum match, but no full-application silicon or performance claim.
- **MG — excluded:** the renderer explicitly dispatched on `resid`, `psinv`,
  and `rprj3`. The renderer, dedicated ABI route, application builder, and
  tests were removed. All former MG cuDNN application timings are historical
  composition results and are not evidence for the current matcher.
- **CG — application result excluded:** the application build deleted
  `conj_grad` and linked a selected replacement. The builder and adapter were
  removed. Static, structurally selected SpMV/dot/squared-L2 launches may be
  audited independently, but the former full-app correctness and timings do
  not count as automatic end-to-end results.
- **IS — application result excluded:** the application build deleted `rank`
  and linked a selected replacement. The builder and adapter were removed.
  Structurally recognized histogram sites remain analysis results only until an
  untouched-source executable is produced.
- **UA — application result excluded:** the three-site binary is a manual
  composition. Its correctness and timing are not an automatic whole-program
  result.
- **ATen name-specific routes — excluded:** bilinear upsampling,
  compressed-block conversion, all/any, NaN sum, sparse norm, joint scaling,
  arg-reduction, fixed pooling backward, sparse index conversion, quantized
  column offsets, adjacent difference, and embedding-bag count renderers were
  removed. Existing generated artifacts for them are stale.

## Results that remain usable

- Parboil SGEMM and stencil retain static, name-independent structural matches,
  but their former full-application binaries omitted an original computational
  helper and linked a selected bridge/replacement. Their application
  correctness and performance claims are therefore excluded.
- Parboil CUTCP: six name-independent strided min/max matches and one
  mixed-precision absolute-sum match have focused correctness evidence. The
  seventh result is standalone ABI validation, not a complete application, and
  correspondence to the paper's seven occurrences remains inferred.
- NPB BT/LU CUDA memset sites are memory initialization and are excluded from
  computational match totals.
- cuSPARSE, cuBLAS, cuDNN, CUB, and cuFFT standalone ABI smokes validate runtime
  plumbing only. They do not establish a source match or application result.

## Current defensible counts

- Paper denominator: 60 static idiom instances, classified by
  suite/program/category.
- Exact author-evidenced source occurrences: 3/60, all in EP.
- Authors' matcher reproduced: 0/60.
- Polygeist on those three exact EP occurrences: 0/3.
- FT compliant automatic matches: 0.
- MG compliant executable stencil matches: 0.
- EP additional inferred structural match: one final scalar CUB sum, outside
  the three confirmed locations.
- No count from an old generated HTML page or retained log supersedes this
  section. Regenerate the viewer after matcher changes before reporting broader
  current-corpus totals.

Historical logs may be retained for debugging, but their claims are retracted
where listed above and they must not be used in paper tables.

# Ginsbach ASPLOS 2018 independent reproduction status

Last updated: 2026-09-07 18:00 PDT

## Active external-library reduction work (2026-09-07)

- CUTCP's seventh reconstructed scalar reduction now has an exact library
  route. The original `src/base/output.c::write_lattice_summary` region widens
  FP32 lattice values to FP64, takes the absolute value, and accumulates in
  FP64. A new generic matcher recognizes that exact typed expression and
  lowers it to a CUB transform-reduce with an FP64 result. The exact retained
  source IR emits one `cubAbsSum_f32_f64_memref` launch and then one
  `polygeist_cub_abs_sum_f32_f64` runtime call, with zero residual launches.
  Focused matcher, ABI-lowering, and independent CPU-reference regressions all
  pass. The completed CUDA 12.6 AArch64 `sm_87` library and reference
  executable were cross-compiled on x86 and passed 3/3 on Orin #2 with the
  independently expected result 16.375. Median host-observed runtime was
  2.569824 ms and median CUDA-event device runtime was 2.117280 ms. No source,
  MLIR, or CUDA compilation occurred on the Jetson. Together with the six
  exact-source cuDNN min/max routes, all seven reconstructed CUTCP scalar
  sites now match and lower. The new absolute-sum test is standalone rather
  than a complete CUTCP application run, and correspondence to the paper's
  seven occurrences remains inferred. Retained details:
  `logs/cutcp_abs_sum_mixed_20260907.md`.
- The original NPB CG Class S driver has now been recomposed with the
  source-faithful raised `conj_grad` helper containing five distinct static
  external-library sites: two cuSPARSE SpMV, two cuBLAS Ddot, and one CUB
  squared-L2 transform-reduce. All five lower to runtime calls with zero
  residual launches. The completed AArch64 CUDA 12.6 `sm_87` executable and
  CUB companion were cross-compiled on the x86 host, then passed NASA
  verification 3/3 on Orin #2. Every run produced zeta
  `8.5971775078648E+00` with relative error `1.0331046659065E-15`.
  Benchmark-reported time was 0.08 s in every run; median rate was 855.66
  Mop/s and median process elapsed was 0.275 s. The earlier three-route run
  reported median 501.82 Mop/s and 0.397 s process elapsed, but this
  same-board historical comparison was not interleaved and is only
  indicative. The whole-file structural count of eight includes duplication
  after `conj_grad` call-site inlining; it is not eight distinct source
  occurrences. Retained details:
  `logs/npb_cg_five_routes_20260907.md`.
- CG direct-reduction work now recognizes zero-seeded FP32/FP64 sum of squares
  as `cublas[D/S]dot(x,x)`, a two-output self-dot/dot reduction as two exact
  dot calls, and `sum((x-y)^2)` as a CUB transform-reduce plus sum. The real
  retained `cg.c` therefore increases from four to eight production launches:
  four cuSPARSE SpMV, three cuBLAS Ddot, and one CUB squared-L2 reduction. ABI
  lowering emits the corresponding eight runtime calls and leaves no launches
  behind. Positive, nonzero-seed negative, focused lowering, and CPU reference
  checks pass. A CUDA 12.6 AArch64 `sm_87` companion and completed smoke-test
  executable were cross-compiled on the x86 host; both FP32 and FP64 passed
  3/3 runs on Orin #2 with the independently expected result 34.0. No source,
  MLIR, or CUDA compilation occurred on the Jetson. The first deployment found
  an older board-side companion without the new symbol; restaging the new
  library under the canonical name resolved it. Retained implementation and
  silicon details: `logs/cg_direct_reductions_20260907.md` in the isolated
  audit worktree, plus
  `scripts/correctness/logs/cg_squared_l2_fixed_20260907_20260907_165833.silicon.log`.
- Expected-match subset rerun after commits `0bcdb646` and `887b131e`:
  BT, CG, IS, UA, CUTCP, SGEMM, SPMV, and STENCIL. All 39 selected
  translation units passed source translation, affine-to-Linalg raising, and
  the production matcher without error diagnostics. The matcher emitted 30
  raw `kernel.launch` operations: BT 8, CG 4, IS 6, UA 3, CUTCP 6, and one
  each for SGEMM, SPMV, and STENCIL. API breakdown: six CUB histograms, three
  cuBLAS DAXPBY, one cuBLAS DGEMM, one cuBLAS DGEMV, one cuBLAS SGEMM, six
  cuDNN min/max reductions, one cuDNN stencil, five cuSPARSE SpMV operations,
  and six zero-fill launches. This is a current-corpus structural count, not
  yet a count on the paper's occurrence denominator. Retained summary:
  `logs/expected_match_subset_20260907.csv` in the isolated audit worktree.
- Polygeist implementation work is isolated in worktree
  `/home/arjaiswal/Polygeist-ginsbach-reductions`, branch
  `audit/ginsbach-reductions`, based on `70340a75936fd5bdc1384bbd6aefce1ff8ce4984`.
- CUTCP source/IR reconstruction identifies the seven Figure 16 scalar
  occurrences with high confidence: six seeded `fminf`/`fmaxf` coordinate
  folds in `get_atom_extent` and one FP32-input/FP64-accumulator absolute sum
  in `write_lattice_summary`. This is source-to-count evidence, not yet
  original-tool confirmation.
- The six coordinate folds already raise to six `linalg.generic` reductions.
  A new generic, function-name-independent matcher recognizes all 6/6 as
  seeded strided FP32 min/max and selects cuDNN reduction calls. ABI lowering
  emits six `polygeist_cudnn_reduce_strided_f32` calls, and both focused lit
  regressions pass. The nested
  `memref<?x!llvm.struct<(memref<?x4xf32>, i32)>>` C-ABI blocker is fixed on
  the isolated audit branch: `polygeist-opt` now registers the transitional
  C-pointer interfaces independently of dialect load order and keeps C-ABI
  preparation in the same registered MLIR context. A second C-ABI defect that
  emitted an empty-index LLVM GEP for rank-zero memrefs was also fixed. The
  exact original `src/base/readatom.c::get_atom_extent` now raises, matches all
  six sites, emits six cuDNN runtime calls, translates to LLVM, builds for x86
  and AArch64, and passes the independent nontrivial harness on both the host
  and Orin (`max 19 31 14`, `min -11 -7 -5`). The AArch64 executable was
  cross-compiled on x86; no source was compiled on the Jetson. These are now
  six exact-source Polygeist executable/correctness/silicon passes, while the
  correspondence to the paper's seven CUTCP occurrences remains inferred.
  Retained logs: `logs/cutcp_strided_minmax_20260907.log` and
  `scripts/correctness/logs/cutcp_atom_extent_exact_fix_20260907_143318.silicon.log`
  in the isolated worktree.
- The seventh absolute sum cannot be mapped exactly to `cublasSasum`, because
  that API returns FP32. The implemented route instead uses CUB transform
  reduction, preserving the source's FP32-to-FP64 conversion before `fabs`
  and its FP64 accumulation/result type.
- DC has been triaged before adding another GPU route. The six paper-reported
  scalar occurrences still have no confirmed source locations. Six plausible
  source loops are bit-mask construction, two population counts, a binomial
  coefficient recurrence, tuple-bitset construction, and a maximum over
  per-task counts, but this list is explicitly inferred. The exact raised IR
  is dominated by loop-carried shift/OR/count state rather than associative
  array reductions. Only the per-task maximum maps directly to CUB
  `DeviceReduce::Max`, and its extent is the small `nTasks` value, making GPU
  offload unjustified. DC is therefore classified as a missing associative
  memory representation plus profitability issue, not as six missing CUB
  matcher rules. Retained analysis: `logs/dc_library_route_triage_20260907.md`.
- FT is the next implementation bucket. Its five units translate and raise,
  but `fft3d.c::fftXYZ` remains 23 nested `affine.for`, `scf.for`, and
  `scf.while` operations with no `linalg.generic`; the existing matcher only
  recognizes a direct 1D DFT tensor form. The required next step is a tested
  3D complex-double cuFFT ABI followed by whole-algorithm Stockham FFT
  recognition. This is a loop/algorithm representation gap, not absence of a
  target library.

## Claim discipline

No apples-to-apples Polygeist comparison is claimed yet. The paper's 60 and
Polygeist's current 18 `kernel.launch` sites have different, unaligned
denominators. Inferred source locations remain explicitly unconfirmed.

## Verified primary-source facts

- The accepted author manuscript was acquired from the University of
  Edinburgh Research Explorer and retained as
  `primary_sources/ginsbach_asplos18.pdf` (SHA-256
  `3541a851670a93f8d224138de7be688028687056c7d8a5ebd47381a1e926e44c`).
- The paper explicitly calls the result "60 idiom instances" detected
  statically over all sequential C/C++ programs in SNU NPB and Parboil.
- Table 1 partitions the 60 as 45 scalar reductions, 5 histogram reductions,
  6 stencils, 1 dense matrix operation, and 3 sparse matrix operations.
- Figure 16 reports per-program/category counts but no source locations or IR
  identifiers. Therefore 60 is confirmed as a count of detected static idiom
  instances, while the precise compiler-IR unit counted per instance still
  requires implementation/evaluation-script evidence.
- The defensible counting interpretation is: one statically detected idiom
  object (for reductions, one accumulator/update; for stencil and matrix
  categories, one matched computation), not one unique loop or dynamic
  execution. Direct evidence is that Figure 16 counts EP as three scalar plus
  one histogram idiom, while the author demo places `q[l]`, `sx`, and `sy`
  together beneath one detected source loop and the thesis identifies those
  accumulator variables individually. Detection operates on optimised LLVM
  IR, so inlining can change the searched representation. Whether the
  evaluation counted every raw constraint solution, one canonical solution
  per accumulator, or applied another equivalence filter remains unavailable;
  the reconstruction's duplicate solutions show that this distinction is
  consequential.
- “60” is not a count of unique idiom types (Table 1 has five categories), nor
  a count of validated replacements: detection is reported separately from
  runtime evaluation, which is restricted to ten programs, and some backend
  generation paths failed (the paper reports Halide GPU generation failures).
- The benchmark population is SNU NPB (the original eight plus UA and DC) and
  all 11 Parboil programs, 21 programs total.
- Reported hardware: AMD A10-7850K with integrated Radeon R7 (driver 1912.5)
  and NVIDIA GTX Titan X (driver 375.66). Reported runtime statistic: median
  of 10 whole-program executions, including GPU transfer overhead.
- The paper says iterative CG, lbm, spmv, and stencil used a manually applied
  lazy-copy optimization. It also says the supplied sgemm and stencil
  baselines were manually loop-interchanged, improving them by about 20x.
- The author's 2020 thesis was retained as
  `primary_sources/ginsbach2020_thesis_redacted.pdf` (SHA-256
  `56b77d1fef14f2e7f545ade685736133db1665ab7a3f92eca5168e9101091845`).
  Its Chapter 6 repeats the 60-instance result but adds no benchmark commit or
  full evaluation command. Chapter 4 documents the related CAnDL normalisation
  configuration as `-Os -fno-unroll-loops -fno-vectorize
  -fno-slp-vectorize -ffast-math`, detection after optimisation, and custom
  reversal/flattening passes. Applying those flags to the ASPLOS experiment is
  currently an inference, not a confirmed evaluation command.

## Author artifact availability

- Paper footnote points to `https://github.com/asplos18ginsbach`.
- `asplos18ginsbach/IDL-Demo` is public at commit
  `1edfb4086a07f1efe2f6b4cf568fe58a2feecfb8`; its README gives the original
  LLVM/Clang build outline and names Python 2.7, GTK+3, Ninja, GHC, and PyPy.
- `asplos18ginsbach/clang` is public at commit
  `818167bc99a0f5538a2676bf876e5c598d1b449b` and is described as unchanged.
- `asplos18ginsbach/llvm` is no longer publicly cloneable. The source fork
  `cc18ginsbach/llvm`, branch `cc18ginsbach`, remains public at commit
  `5d26ac5896055927238d4c3ec01608ae07b7f9d0`; equivalence to the missing
  paper-account repository must be verified, not assumed.
- No supplementary archive, benchmark pin, dataset manifest, or evaluation
  script has yet been found in the paper-account repositories.
- The public demo expects source under `llvm/lib/IDLParser`, invokes
  `clang++ -std=c++17 -S -emit-llvm -O2 -gline-tables-only`, and reads
  `replace-report--.json`. Its README says the default matcher searches for
  reductions, histograms, and matrix multiplications.
- The related `cc18ginsbach/llvm` tree instead uses `lib/CAnDLParser`, is based
  on LLVM `6.0.0svn`, enables only `Experiment` and `SCoP` in
  `ResearchReplacer`, and contains no heterogeneous replacement backends.
  Its four custom commits describe the CC'18 CAnDL work. This directly
  disproves drop-in equivalence to the missing demo repository.
- That related tree nevertheless retains a stale generated `parsed.txt` with
  unused specifications named `Reduction`, `Histo`, `Stencil`, `GEMM`, and
  `SPMV`. This may derive from the IDL work, but its provenance and exact
  version are unconfirmed and it is not enabled by the committed matcher.
- A strictly labelled experimental reconstruction now exists in the related
  repository's separate branch/worktree:
  `audit/ginsbach-parsed-reconstruction` at
  `e1fc2eb154550adfd54a5fda943ad92e90411d48`. It changes the generator
  whitelist to expose the retained parsed `Reduction`, `Histo`, `Stencil`,
  `GEMM`, and `SPMV` specifications and changes no Polygeist code. Generation
  produces all five detector functions (retained patch:
  `primary_sources/parsed_idiom_reconstruction.patch`). Any output from this
  branch is reconstructed evidence, never an original-tool reproduction.
- A clone attempt for `asplos18ginsbach/llvm` failed and is retained in
  `logs/asplos18ginsbach_llvm_ls_remote.log`.
- The unmodified public CAnDL snapshot configures with the demo's documented
  CMake command and reports Clang 6.0.0. Its first build stopped at absent GHC.
  GHC 8.8.4 and PyPy/Python 2.7.18 were then installed as documented demo
  dependencies; the build resumed. Exact configuration/build/package logs are
  retained under `logs/`. Apt also retried configuration of an unrelated,
  already-broken `shim-signed` package and failed because `/dev/sda1` is not
  available; no package was removed.
- The unmodified related snapshot subsequently built successfully. The binary
  identifies itself as Clang 6.0.0 with LLVM commit
  `5d26ac5896055927238d4c3ec01608ae07b7f9d0` and Clang commit
  `818167bc99a0f5538a2676bf876e5c598d1b449b`. On the retained EP source it
  compiled successfully and reported three generic `SCoP` solutions at lines
  127, 156, and 215, but zero `Reduction`, `Histo`, `Stencil`, `GEMM`, or
  `SPMV` solutions because those detectors are not enabled in this snapshot.
  This is a successful build diagnostic and direct non-equivalence result, not
  reproduction of any of the paper's 60.
- The experimental parsed-IDL build initially exposed an upstream Ninja
  dependency-order race: `IdiomSpecifications.cpp` compiled before generated
  `llvm/IR/Attributes.gen` existed. Building the declared Attributes target
  first resolved the prerequisite without a source change; the resumed build
  is retained in the same labelled log.
- Its next compile exposed that `CustomPassPreprocess` still needs the public
  `HoistSelect` and `Distributive` detectors. A second focused commit retained
  those dependencies while leaving the five reconstructed report categories
  unchanged; the build resumed and completed successfully. Both failures and
  corrections are preserved. The resulting X86-only diagnostic compiler is
  Clang 6.0.0 built from reconstruction commit
  `e1fc2eb154550adfd54a5fda943ad92e90411d48` and the unchanged author Clang
  commit `818167bc99a0f5538a2676bf876e5c598d1b449b`. Restricting LLVM targets to
  X86 is a documented build-cost deviation; this compiler is used only for
  host-side matcher diagnostics, not target execution.
- On the retained EP source, the experimental reconstruction reports only one
  match: a scalar `Reduction` in `main` at source line 215 (`gc += q[i]`). It
  misses all three idioms shown by the author's ASPLOS demo in the conditional
  loop beginning at line 197/198: histogram `q[l]` and scalar reductions `sx`
  and `sy`. C compilation at `-O2`, `-O3`, and `-Os` produces the same single
  line-215 match; at `-O0`/`-O1` the pass does not emit a report. This is direct
  evidence that the retained generated specifications plus related CC'18
  engine are not a faithful reconstruction of the missing ASPLOS tool.
- The demo GUI invokes `clang++ -xc++`, but unmodified SNU NPB C cannot be
  compiled as C++ because identifiers such as `false`, `true`, and `class`
  conflict with C++ keywords. The paper describes sequential C/C++ benchmark
  programs; treating EP as C is therefore the source-faithful diagnostic.
- A broad unmodified-public-snapshot corpus diagnostic was deliberately
  stopped after 14 BT translation units because the snapshot was already
  proven non-equivalent and generic SCoP solving was slow. Its partial raw
  results are retained and are not counted as a paper reproduction.
- The author demo screenshot and thesis directly recover three EP occurrence
  locations in `main`: histogram `q[l]` at source line 206 and scalar
  reductions `sx`/`sy` at lines 207/208, all inside the loop the demo reports
  at line 198. These locations are now recorded in both occurrence manifests.
  The exact evaluation checkout and individual solution IR remain unknown.
  The likely third EP scalar, `gc += q[i]` at line 216, is recorded only as a
  candidate because its evidence is the paper's category count plus the
  experimental reconstruction; it is not presented as confirmed.
- The full retained-spec reconstruction diagnostic processed 104 translation
  units: 90 compiled, 13 failed, and one (`UA/mason.c`) timed out at 300
  seconds. It emitted 240 raw solutions: 131 scalar, 24 histogram, 83 stencil,
  2 GEMM, and 0 SPMV. These disagree sharply with the paper's
  45/5/6/1/3 totals. Examples include 28 scalar solutions for SAD (paper: 1),
  17 stencil solutions for BT (paper: 0), and only 1 scalar for EP (paper: 3
  scalar plus 1 histogram). Multiple alternatives frequently share a source
  line. The raw output therefore cannot recover the paper denominator without
  the missing implementation and its solution filtering/counting rules.
  Exact per-program comparison is retained in
  `runs/parsed-idl-reconstruction-corpus/reconstruction_vs_paper.csv`.
- Strict `-std=c99` prevented several Parboil main units from exposing POSIX
  declarations in this old source snapshot. A separate, labelled GNU C99
  diagnostic compiles the paper's unique `histo` program occurrence and finds
  one `Histo` solution at `main` line 96, but finds no SPMV or TPACF match.
  Because the paper's language mode is unavailable, this remains a candidate
  diagnostic, not an original-tool reproduction or confirmed source mapping.
- The timeout implementation originally killed the clang driver but left its
  solver `clang -cc1` child orphaned. The exact orphan was stopped after the
  run, and the runner now starts each compiler in its own process group and
  kills that group on future timeouts. No benchmark source was changed.

## Existing local evidence under audit

- Polygeist: branch `raisetolinalg`, commit
  `2716bebcb56c608d7fb5fb6a51bc076214cc7bdd` at audit start.
- Existing 60-row `published_idiom_manifest.csv` is a Figure 16 transcription,
  not a recovered source-occurrence mapping.
- Existing benchmark snapshots are third-party mirrors, not yet proven to be
  the authors' exact revisions:
  - SNU NPB mirror commit `4f2aa1b4d3127dbb3612c8aef24b24c69e83013c`.
  - Parboil mirror commit `ccf3d3126f1754ca85528722f4ced5894ccb852b`.
- Existing Polygeist audit reports 104 translation units and 18 launch sites.
  These numbers are retained only as an independent current-source baseline.
- The demo screenshot exposes an author path containing
  `snu-npb-1.0.3/NPB3.3-SER-C` and shows an EP match in `main` at source line
  198. The displayed EP source is line-for-line identical in that region to
  the local mirror's `NPB3.3-SER-C/EP/ep.c` (blob SHA-256
  `b7f178978ed0f8dc83cbdf6a24bd5759a5f4d0e821b53116cd5396711ddc1af9`).
  This confirms the demo source/version and an EP matched loop, but does not by
  itself prove that the paper evaluation used the same checkout.
- `ginsbach_60_manifest.csv` and `polygeist_same_inputs.csv` now contain 60
  valid occurrence rows each. At this stage they preserve the Figure 16
  denominator and mark source/IR locations and attributable Polygeist results
  unresolved. They are regenerated by
  `scripts/initialize_occurrence_manifests.sh`.

## Current quantitative status

- Reported instances exactly classified by suite/program/category: 60/60.
- Reported instances supported by direct author evidence to source occurrence:
  3/60 (EP `q[l]`, `sx`, and `sy`; individual IR solutions unresolved).
- Additional explicitly inferred source candidates: 2/60 (EP `gc`, Parboil
  `histo`; neither counted as confirmed).
- Reproduced with authors' matcher: 0/60.
- Authors' replacements generated: 0/60.
- Authors' replacement executables built: 0/60.
- Independently correctness-validated author replacements: 0/60.
- Verified comparable author runtime results reproduced: 0/60.
- Polygeist occurrence audit on the three directly recovered EP sites: 3/3
  translate, 0/3 become Linalg idioms, 0/3 launch. The conditional `q[l]`,
  `sx`, and `sy` updates remain in a residual affine/SCF loop. This is an
  occurrence-aligned current-source result, but still not an apples-to-apples
  compiler/revision comparison.

## Active next steps

1. Search archives/fork networks for the missing ASPLOS LLVM/IDL repository.
2. Recover compiler revision, passes, IDL specifications, output/counting code,
   benchmark paths, build flags, and any embedded evaluation scripts.
3. Locate archived versions of the missing paper-account LLVM repository and
   contemporaneous repository documentation.
4. Use the failed EP reconstruction as a guardrail: broader diagnostics may
   recover candidate locations, but must not be labelled authors' matches or
   used as the paper's occurrence-level denominator.

## Blockers

- The paper does not publish source locations for Figure 16's instances.
- Exact benchmark commits and datasets are not specified in the paper.
- The paper-account LLVM repository referenced by the demo is unavailable.
- The paper does not state the LLVM revision or full Clang optimization,
  preprocessing, or inlining command line.

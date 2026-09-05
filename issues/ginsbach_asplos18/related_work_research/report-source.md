# Related Work for Legacy-Code Semantic Raising and Heterogeneous API Matching

Research date: 2026-09-05

## Executive conclusion

Polygeist's matcher sits at the intersection of three research lines:

1. recognizing computations in ordinary legacy C/C++/Fortran and replacing them with optimized library or accelerator APIs;
2. raising low-level loop or compiler IR into structured tensor/linear-algebra IR;
3. proving or testing that the raised program preserves the source computation.

The closest practical comparisons are not all the same kind of system. **SpEQ and LiLAC are the strongest same-benchmark sparse-library comparisons; KernelFaRer is the strongest simple LLVM/BLAS artifact; SMR/PGL and Multi-Level Tactics are the closest hand-specified MLIR raising systems; mlirSynth and Tensorize are the closest synthesis-based MLIR systems.** SLEB is the newest broad sparse lifting system and already compares itself with LiLAC and SpEQ on NPB-CG and Parboil SpMV.

No paper found is a drop-in substitute for the proposed Polygeist evaluation. Most systems are restricted to one domain (dense tensors, sparse algebra, stencils, or FFT), accept a pre-extracted function or low-level MLIR rather than a whole application, or report a different denominator such as programs lifted rather than static source occurrences. Polygeist can make a defensible contribution by evaluating one occurrence-level pipeline across all of those boundaries: source recovery, raising, semantic matching, GPU lowering, correctness, and silicon runtime.

## Scope and method

I searched for work on idiom recognition, library-call discovery, abstraction raising, verified lifting, tensor/sparse program synthesis, MLIR raising, and automatic heterogeneous offload. I prioritized author papers, publisher pages, official artifact records, and author repositories. Search was expanded through the related-work and artifact references of Ginsbach 2018, LiLAC, SpEQ, SLEB, Tensorize, and Tenspiler.

The inclusion test was whether a system does at least one of the following:

- recovers a higher-level computation from ordinary low-level source or IR;
- substitutes a library/API/accelerator implementation automatically;
- raises loops into a structured IR or DSL that can target heterogeneous hardware;
- supplies a directly relevant recognition, correctness, or offload baseline.

This is a literature and artifact-availability audit, not an independent reproduction of the systems' performance claims. Performance numbers below are authors' reported results.

## Tier 1: closest comparisons

### 1. SpEQ — Translation of Sparse Codes using Equivalences (PLDI 2024)

SpEQ is probably the single best next external baseline. It recovers sparse formats and computations from LLVM IR, uses equivalence reasoning and stuttering simulation, guards assumptions at runtime, and emits OpenMP, MKL, or cuSPARSE replacements. Its evaluation includes NPB-CG, NPB-IS, Parboil, Netlib, PolyBench, SciMark, CSparse, TACO, and TSVC2. That creates direct benchmark overlap with both Ginsbach and our corpus. It reports geometric-mean speedups of 3.25x with OpenMP, 5.09x with MKL, and 8.04x with cuSPARSE. A public artifact is archived on Zenodo. ([paper](https://doi.org/10.1145/3656445), [author PDF](https://www.paramathic.com/wp-content/uploads/2024/04/REV_PLDI_rev2.pdf), [artifact](https://zenodo.org/records/10906216))

Why it matters: it tests the same core proposition as Polygeist for sparse programs, but adds explicit equivalence validation and runtime guards. For NPB-CG and Parboil SpMV, it should be treated as a first-class apples-to-apples competitor.

### 2. LiLAC — Automatically Harnessing Sparse Acceleration (CC 2020)

LiLAC is the direct follow-on from the Ginsbach group. A library implementer supplies a computation description and a data-marshalling harness; the LLVM pass detects equivalent source-language-independent IR, inserts the optimized call, and manages stateful format conversion or device transfers. It targets MKL, cuSPARSE, clSPARSE, and SparseX and evaluates C and Fortran programs including NPB-CG and Parboil SpMV. The authors report improvements from 1.1x to over 10x. The author LLVM and Clang branches are publicly accessible and contain a `lilac` directory and artifact script. ([paper](https://arxiv.org/abs/2001.07938), [LLVM artifact branch](https://github.com/ginsbach/llvm/tree/linearalgebra), [Clang artifact branch](https://github.com/ginsbach/clang/tree/research))

Why it matters: it is the cleanest way to extend the ASPLOS 2018 audit into an available implementation from the same lineage, especially for sparse matching and data movement.

### 3. SLEB — Accelerating Sparse Algebra with Program Synthesis (CC 2026)

SLEB synthesizes both the sparse data binding and the target operation rather than requiring one fixed pattern and format. It evaluates 31 C, C++, and Fortran programs implementing 15 sparse operations over CSR, CSC, COO, and JDS. The corpus includes NPB-CG and Parboil SpMV. The paper reports 94% lifting coverage versus 13% for LiLAC and 19% for SpEQ on its broader suite, with 2.6x CPU and 7.8x GPU geometric-mean speedups; non-lifted programs are assigned 1x in those geometric means. It uses Python 3.10, LLVM 18, cuSPARSE 12, MKL 2025, and TACO 0.1. The paper names a GitHub repository, but that URL returned HTTP 404 during this audit, so the implementation is not currently confirmed obtainable. ([paper](https://josewesley.com/archive/sleb.pdf), [paper-referenced repository, currently unavailable](https://github.com/JWesleySM/sleb))

Why it matters: this is the newest sparse state of the art and already supplies a shared corpus involving the precise prior systems we care about. Its unit of evaluation is a program, however, not the Ginsbach 60-occurrence denominator.

### 4. KernelFaRer — Replacing Native-Code Idioms with High-Performance Library Calls (TACO 2021)

KernelFaRer is an LLVM pass that matches computation and memory-access trees, checks dependence legality, and replaces dense linear-algebra idioms such as GEMM and SYR2K with calls to CBLAS/Eigen/vendor BLAS. Its repository provides the pass, C++ and Fortran tests, and build instructions against an installed LLVM. ([paper](https://webdocs.cs.ualberta.ca/~amaral/papers/CarvalhoTACO21.pdf), [repository](https://github.com/jaopaulolc/KernelFaRer), [DOI](https://doi.org/10.1145/3459010))

Why it matters: it is small enough to reproduce early and is a direct dense-BLAS comparator for Parboil SGEMM and isolated GEMM/SYR2K variants. It is less general than Polygeist and has known sensitivity to IR shape and pointer/layout proofs.

### 5. Source Matching and Rewriting (SMR) and PGL (2022–2026)

SMR lets users express an idiom in source-level C or Fortran and matches it after lowering to MLIR. It first matches control-dependence graphs and then data-dependence graphs, and demonstrates raising to BLAS calls. The implementation is public. ([SMR paper](https://arxiv.org/abs/2202.04153), [repository](https://gitlab.com/parlab/smr), [MLIR presentation](https://mlir.llvm.org/OpenMeetings/2023-07-06-SMR.pdf))

The newer Pattern Generation Language (PGL) automates the generation of pattern variants for SMR. The paper reports 47 matches versus 22 with manually specified SMR patterns, directly addressing the fragility and rule-coverage problem that our Polygeist matcher also faces. ([PGL paper](https://doi.org/10.1145/3777905))

Why they matter: these are the nearest MLIR-native systems for comparing pattern specification effort, tolerance to source/IR variation, and BLAS replacement coverage.

### 6. mlirSynth — Automatic, Retargetable Program Raising in MLIR (PACT 2023)

mlirSynth uses bottom-up enumerative synthesis, MLIR type constraints, and observational equivalences to raise low-level MLIR into Linalg or HLO without a manually coded rule per source shape. It evaluates PolyBench and reports geometric-mean gains of 2.5x on Intel, 3.4x on AMD, and 21.6x on TPU. The code and benchmark driver are public. ([paper](https://arxiv.org/abs/2310.04196), [repository](https://github.com/alexanderb14/mlirSynth))

Why it matters: it is a direct comparator for the `RaiseToLinalg` stage. Its existing corpus is mostly dense kernels and its observational equivalence is not the same guarantee as end-to-end correctness on arbitrary application inputs.

### 7. Tensorize — Fast Synthesis of Tensor Programs from Legacy Code (CGO 2025)

Tensorize advances the mlirSynth line using symbolic traces, sketches, and algebraic solving. It lifts loop-level C/Python-derived programs into NumPy, PyTorch/JAX, and StableHLO-like tensor programs, claims symbolic equivalence for all inputs within its supported language, and is designed for CPU, GPU, and TPU backends. Its source, benchmarks, Dockerfile, and automated evaluation script are public. ([paper](https://doi.org/10.1145/3696443.3708956), [author PDF](https://www.pure.ed.ac.uk/ws/portalfiles/portal/493345271/BrauckmannEtalCGO2025TENSORIZEFastSynthesisofTensorPrograms.pdf), [repository](https://github.com/alexanderb14/tensorize))

Why it matters: this is likely the strongest modern comparison for semantic loop-to-tensor raising, though it does not focus on whole NPB/Parboil applications or sparse/histogram idioms.

### 8. Multi-Level Tactics / Progressive Raising in Multi-Level IR (CGO 2021)

Multi-Level Tactics uses a declarative tactics language to progressively raise affine loop IR to Linalg or BLAS, then exploit high-level transformations such as matrix-chain reordering. Its artifact includes Docker scripts and modified PolyBench/C 4.2.1 inputs. The artifact explicitly notes a missed Darknet case caused by linearized accesses and the absence of delinearization—highly relevant to Polygeist's submap/debufferization work. ([paper and artifact appendix](https://grosser.science/static/7d02fb58ecc49e4d2097d11bc9e8152a/chelini-2021-abstraction-raising.pdf), [artifact branch](https://github.com/LoopTactics/mlir/tree/cgo))

Why it matters: this is the most direct architectural predecessor to declarative Affine-to-Linalg/BLAS raising and provides an immediately reproducible dense-kernel baseline.

### 9. ATC — Matching Linear Algebra and Tensor Code to Specialized Hardware Accelerators (CC 2023)

ATC uses program synthesis, program classification, dynamic analysis, variable constraints, and lexical-distance guidance to map source regions to accelerator APIs despite code and interface mismatch. The paper reports accelerating 2.6x–7x more programs than four prior approaches. A versioned artifact is archived on Zenodo. ([paper](https://arxiv.org/abs/2301.11659), [artifact](https://doi.org/10.5281/zenodo.7506749))

Why it matters: it tackles more variation than rigid graph matching and should inform a robustness benchmark. Because it uses dynamic evidence, its correctness and applicability assumptions differ from a static MLIR matcher.

## Tier 2: very relevant verified/synthesis lifting systems

### 10. Tenspiler (ECOOP 2024)

Tenspiler uses verified lifting to synthesize a TensIR program from sequential C++ or Python and then maps TensIR to six tensor software/hardware targets. The paper reports 105x average kernel and 9.65x end-to-end improvement across ten real-world benchmark suites. Its artifact passed ECOOP artifact evaluation and is archived by Dagstuhl. ([paper](https://doi.org/10.4230/LIPIcs.ECOOP.2024.32), [artifact](https://doi.org/10.4230/DARTS.10.2.17))

Use it to compare semantic guarantees and backend retargetability, but not as a direct substitute for whole-program compiler-IR matching.

### 11. C2TACO (GPCE 2023)

C2TACO performs guided enumerative synthesis from C functions and generated input/output examples to TACO index notation. TACO can then generate OpenMP C or CUDA. The public Docker artifact includes real and artificial benchmarks and detailed synthesis logs. ([paper](https://doi.org/10.1145/3624007.3624053), [repository](https://github.com/JWesleySM/c2taco))

Use it as a dense/sparse tensor-DSL lifting baseline and as an example of separating translation coverage from generated-code correctness.

### 12. Guided Tensor Lifting / STAGG (PLDI 2025)

STAGG asks an LLM to induce domain heuristics as a probabilistic grammar, performs guided synthesis into TACO, and validates candidates with examples and bounded model checking. A Docker-based public repository is available. ([paper](https://doi.org/10.1145/3729330), [author PDF](https://www.pure.ed.ac.uk/ws/portalfiles/portal/522474520/LiEtalPLDI2025GuidedTensorLifting.pdf), [repository](https://github.com/BugBugSurvival/Guided-Tensor-Lifting))

Use it to test whether learned search guidance expands coverage beyond fixed Polygeist matcher rules while retaining an independent correctness check.

### 13. Guess, Measure & Edit / KONRUL (PACT 2025)

KONRUL starts with an LLM-generated tensor expression, lowers both the guess and source into a common low-level form, measures their distance, and iteratively edits the high-level guess. The authors report 98% lifting coverage and geometric-mean speedups of 4.07x on CPU and 38.30x on GPU. ([paper](https://doi.org/10.1109/PACT65351.2025.00029), [author PDF](https://josewesley.com/archive/konrul.pdf))

Use it as a newer search/repair comparison. A public artifact was not confirmed in the primary sources inspected, so its reproducibility must be checked before putting it in the execution plan.

### 14. LIAR — Latent Idiom Recognition using Equality Saturation (CGO 2024)

LIAR represents both programs and BLAS/PyTorch idioms in a very small functional array language and uses eight core semantic rewrite rules plus equality saturation to expose library calls that are not syntactically present. It covers operations such as dot, AXPY, GEMV, GEMM, transpose, and memset and reports a 1.46x geometric-mean BLAS speedup. The paper carries artifact-available, reusable, and results-reproduced badges. ([paper](https://arxiv.org/abs/2312.17682), [author project page](https://jonathanvdc.github.io/compiler-work/latent-idiom-recognition/))

Use it to compare equality-saturation rule economy and discovery of latent composite idioms. Its reported C programs are generated from the high-level minimalist language, so it is not an apples-to-apples arbitrary-C frontend comparison.

## Tier 3: domain-specific precedents

### 15. Verified Lifting of Stencil Computations / STNG (PLDI 2016)

STNG uses synthesis and verification to recover stencil computations from Fortran loops and emit Halide. It is an important correctness precedent for the stencil subset, with reported median speedup of 4.1x and gains up to 24x. ([paper record](https://doi.org/10.1145/2908080.2908117), [author PDF](https://people.csail.mit.edu/shachari/dl/pldi2016.pdf))

Use it for stencil-specific semantic validation methodology, not as a general C/MLIR comparison.

### 16. FACC / Bind the Gap (PLDI 2022)

FACC identifies FFT procedures in unmodified C and synthesizes drop-in adapters across code, data-representation, domain, and behavioral mismatches. It targets FFTW, NXP PowerQuad, and Analog Devices FFTA; the authors report mean speedups of 9x, 17x, and 27x. Source and separate evaluation environments are public. ([paper](https://doi.org/10.1145/3519939.3523439), [repository](https://github.com/FourierACceleratorCompiler/FACC))

Use it if Polygeist adds cuFFT matching. Its generate-and-test behavioral approach is particularly useful for evaluating ABI and layout adaptation.

### 17. Helium (PLDI 2015)

Helium dynamically traces stripped x86 binaries and lifts stencil kernels into Halide. The project distributes source and an installer. It demonstrates that useful high-level structure can be recovered even without source, but its trace-based recovery is neither a static occurrence matcher nor a general semantic guarantee. ([project and artifact](https://projects.csail.mit.edu/helium/), [DOI](https://doi.org/10.1145/2737924.2737974))

## Adjacent baselines, not direct competitors

- **Polly-ACC** and **PPCG** compile affine/polyhedral regions to accelerators and are useful GPU-offload baselines, but they generate parallel code from SCoPs rather than recover semantic library operations. ([Polly-ACC](https://pollylabs.org/publications/grosser-2016-polly-acc-transparent-compilation-to-heterogeneous-hardware.pdf), [PPCG repository](https://github.com/Meinersbur/ppcg))
- **Bones** classifies loops into algorithmic skeletons/species and emits CUDA. It is useful for comparing structural pattern coverage but relies on a restricted/annotated view of code rather than broad semantic API matching. ([paper](https://cnugteren.github.io/downloads/Nugteren2012b.pdf))
- **KernelGen** and more recent “GPU First” work compile legacy CPU programs directly for GPU execution. They answer automatic offload, not library-call discovery or abstraction raising. ([KernelGen](https://github.com/dmikushin/kernelgen), [GPU First](https://arxiv.org/abs/2306.11686))
- LLVM's production **LoopIdiomRecognize** pass is a necessary sanity baseline for narrow idioms such as memset/memcpy, but its own source still lists dot product and matrix multiplication to BLAS as future examples. ([LLVM source](https://llvm.org/doxygen/LoopIdiomRecognize_8cpp_source.html))

## Recommended comparison program

### Priority A: shared sparse corpus

Run Polygeist, LiLAC, and SpEQ on byte-identical NPB-CG and Parboil SpMV sources and inputs; add SLEB if its paper-referenced artifact becomes obtainable. Use one manifest row per static semantic occurrence. Record detection, replacement generation, executable generation, correctness, GPU execution, and runtime separately. This is the highest-value experiment because the sources and cuSPARSE target overlap and the first three implementations have public artifacts.

### Priority B: dense BLAS and structured raising

Run Polygeist, KernelFaRer, SMR/PGL, Multi-Level Tactics, mlirSynth, and Tensorize on a controlled set containing Parboil SGEMM plus the exact PolyBench variants shipped by the artifacts. Split the results into two denominators:

1. identical source occurrences for tools that accept C/LLVM input;
2. identical normalized MLIR regions for tools that require pre-lowered MLIR.

Do not combine those denominators in one headline percentage.

### Priority C: robustness rather than raw count

For GEMM, GEMV, AXPY, SpMV, histogram, and stencil, generate semantics-preserving variants: loop interchange, scalar temporaries, linearized versus multidimensional indexing, pointer offsetting, inlining, strength reduction, split initialization, and benign control-flow changes. Compare coverage and false positives against SMR/PGL, KernelFaRer, LIAR, ATC, and synthesis systems. This directly measures whether Polygeist's e-graph/equality rules buy robustness.

### Priority D: end-to-end Orin comparison

Historical reproduction and modern hardware comparison must be reported separately:

- **Historical reproduction:** original toolchain, source revision, data, and reported hardware when available.
- **Modern cross-tool rerun:** common source/input, AArch64 cross-compilation, CUDA 12.6, `sm_87`, the same Orin power mode/clocks, identical correctness tolerance, and explicit transfer/marshalling costs.

For library replacements, compare against the same cuBLAS/cuSPARSE/cuFFT/cuDNN/cuTENSOR version. For generated kernels, report kernel-only and end-to-end timings. A tool that merely detects an idiom is not counted as compiled, correct, or executed.

## What Polygeist can uniquely demonstrate

The literature is fragmented by domain and stops at different pipeline stages. A strong Polygeist result would not be “more launches than another paper's matches.” It would be:

- one explicit, occurrence-level denominator from real legacy programs;
- loop-to-structured-IR raising and semantic API discovery in the same MLIR pipeline;
- dense, sparse, stencil, histogram/reduction, and possibly FFT coverage;
- independent numerical validation with nontrivial inputs;
- cross-compiled AArch64 executables and measured Orin results;
- failure attribution at each compiler stage rather than a single success percentage.

That combination is more defensible and more informative than any raw match-count comparison found in the surveyed papers.

## Evidence gaps and cautions

- Artifact availability was confirmed by public landing pages/repositories, but none of the artifacts was built during this literature pass.
- SLEB is very recent; its reported 94% is over 31 programs, not 94% of the Ginsbach 60 occurrences.
- The SLEB paper references `https://github.com/JWesleySM/sleb`, but that URL returned HTTP 404 on 2026-09-05.
- LIAR's generated-C evaluation and MLIR-only inputs used by several raising systems are not equivalent to compiling arbitrary legacy C applications.
- Dynamic/example-based synthesis systems need independent tests beyond their synthesis examples to rule out overfitting.
- Compiler-version differences are integral to these artifacts. Forcing every tool onto one current LLVM would test porting effort rather than reproduce the original technique.
- The PGL paper is public, but a standalone PGL artifact was not confirmed during this search.
- KONRUL's public implementation was not confirmed during this search.

## Search gap matrix

- Searched: legacy idiom recognition to heterogeneous APIs; LLVM/MLIR abstraction raising; library-call discovery; sparse lifting; verified tensor lifting; stencil lifting; FFT accelerator mapping; equality-saturation idiom recognition; synthesis-based tensor lifting.
- Strong coverage found: dense BLAS, sparse SpMV/tensor algebra, stencil/Halide, FFT, loop-to-tensor DSL, and polyhedral GPU offload.
- Weak coverage found: automatic recovery of histograms, irregular reductions, graph kernels, cuDNN operations, cuTENSOR/cuTensorNet idioms, and whole-program ABI-safe replacement on current embedded GPUs.
- Unresolved artifact questions: SLEB, standalone PGL source, and KONRUL source; these should be checked with authors or artifact committees before scheduling reproductions.

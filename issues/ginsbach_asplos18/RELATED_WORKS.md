# Related Work: Legacy-Code Semantic Matching, Raising, and Optimized Kernel Generation

Research consolidated: 2026-09-05

## Executive summary

Polygeist occupies a useful gap between two research communities:

1. **Recognition and raising:** recover a known computation from ordinary C, C++, Fortran, LLVM IR, or low-level MLIR and express it as a library call or structured tensor operation.
2. **Optimized implementation generation:** select or generate a GPU implementation after the computation is known, using schedules, templates, autotuning, synthesis, or superoptimization.

The closest recognition baselines are **Ginsbach et al., LiLAC, SpEQ, KernelFaRer, SMR/PGL, Multi-Level Tactics, mlirSynth, Tensorize, and SLEB**. The closest kernel-generation baselines are **MLIR Transform/IREE, Bolt/CUTLASS, TVM/Ansor/TensorIR, ROLLER, Welder, TLM, SparseTIR/TACO, and Mirage**.

The literature does not support comparing Polygeist's current `kernel.launch` count directly with Ginsbach's reported 60 idioms. The tools use different inputs, IRs, preprocessing pipelines, and denominators. A defensible comparison requires an occurrence-level manifest that separately records recognition, replacement generation, compilation, correctness, silicon execution, and runtime.

The recommended architecture is a **hybrid backend**. Polygeist should preserve a semantic kernel contract, call a vendor library when appropriate, use hardware-native templates for supported dense operations, generate transparent AOT CUDA through MLIR for general cases, and invoke offline schedule search only where specialization or fusion justifies its cost.

## End-to-end research landscape

```text
Legacy C/C++/Fortran
        |
        v
Frontend and canonicalization
        |
        v
Recognition / verified lifting / synthesis
        |
        v
Semantic kernel contract or structured MLIR
        |
        +--------------------+---------------------+------------------+
        |                    |                     |                  |
        v                    v                     v                  v
Vendor API            Template generator    Scheduled codegen   Superoptimizer
cuBLAS/cuSPARSE       CUTLASS/Bolt           MLIR/TVM/Triton     Mirage/e-graphs
        |                    |                     |                  |
        +--------------------+---------------------+------------------+
                                     |
                                     v
                         AOT AArch64 + sm_87 executable
                                     |
                                     v
                         Correctness and Orin measurement
```

This decomposition matters because a tool can succeed at recognition but fail to generate a replacement, or generate a kernel that compiles but is incorrect or slower. Those outcomes must never be collapsed into one “match” count.

## Part I: semantic recognition and raising

### Closest sparse and heterogeneous API systems

#### Ginsbach et al. — Automatic Matching of Legacy Code to Heterogeneous APIs (ASPLOS 2018)

This is the historical anchor. It matches legacy LLVM IR to heterogeneous API specifications and reports 60 idioms, but that number must be reconstructed from the original artifact before it can be compared with Polygeist. The open question is whether 60 denotes static occurrences, IR regions, replacements, or another post-processing count. The independent reconstruction belongs in `ginsbach_60_manifest.csv`. ([paper record and accepted manuscript](https://eprints.gla.ac.uk/156406/), [author PDF](https://steuwer.info/files/publications/2018/ASPLOS-Matching-Legacy-Code.pdf))

#### SpEQ — Translation of Sparse Codes using Equivalences (PLDI 2024)

SpEQ recovers sparse formats and computations from LLVM IR, uses equivalence reasoning and stuttering simulation, inserts runtime guards, and emits OpenMP, MKL, or cuSPARSE replacements. It includes NPB-CG and Parboil, creating direct corpus overlap. The authors report geometric-mean speedups of 3.25x with OpenMP, 5.09x with MKL, and 8.04x with cuSPARSE. ([paper](https://doi.org/10.1145/3656445), [artifact](https://zenodo.org/records/10906216))

Relevance: strongest available sparse same-benchmark competitor and an important correctness/guarding baseline.

#### LiLAC — Automatically Harnessing Sparse Acceleration (CC 2020)

LiLAC follows the Ginsbach line. A library implementer describes a computation and data-marshalling harness; an LLVM pass recognizes equivalent code and manages stateful format conversion or device transfer. It targets MKL, cuSPARSE, clSPARSE, and SparseX and evaluates C and Fortran programs including NPB-CG and Parboil SpMV. ([paper](https://arxiv.org/abs/2001.07938), [LLVM artifact branch](https://github.com/ginsbach/llvm/tree/linearalgebra), [Clang artifact branch](https://github.com/ginsbach/clang/tree/research))

Relevance: the most direct available implementation from the same research lineage as the ASPLOS 2018 paper.

#### SLEB — Accelerating Sparse Algebra with Program Synthesis (CC 2026)

SLEB synthesizes both sparse data binding and the target operation. Its evaluation covers 31 C/C++/Fortran programs, 15 operations, and CSR, CSC, COO, and JDS formats. The authors report 94% program-level lifting coverage versus 13% for LiLAC and 19% for SpEQ on their broader corpus, plus 2.6x CPU and 7.8x GPU geometric-mean speedups. Its denominator is programs, not Ginsbach occurrences. The paper-referenced repository was unavailable during the research pass. ([paper](https://josewesley.com/archive/sleb.pdf), [paper-referenced repository](https://github.com/JWesleySM/sleb))

Relevance: newest broad sparse lifting result, but artifact availability must be resolved before reproduction.

### Dense library matching and MLIR-native raising

#### KernelFaRer (TACO 2021)

KernelFaRer matches computation and memory-access trees in LLVM, checks dependence legality, and replaces dense idioms such as GEMM and SYR2K with CBLAS, Eigen, or vendor BLAS. Its compact public implementation makes it a good early artifact target. ([paper](https://webdocs.cs.ualberta.ca/~amaral/papers/CarvalhoTACO21.pdf), [repository](https://github.com/jaopaulolc/KernelFaRer))

#### Source Matching and Rewriting and PGL (2022–2026)

SMR expresses patterns in C or Fortran, lowers them to MLIR, and matches control- and data-dependence graphs before raising to operations such as BLAS. PGL generates pattern variants automatically; its paper reports 47 matches compared with 22 using manually specified SMR patterns. ([SMR paper](https://arxiv.org/abs/2202.04153), [SMR repository](https://gitlab.com/parlab/smr), [PGL paper](https://doi.org/10.1145/3777905))

Relevance: closest MLIR-native comparison for pattern specification effort and robustness to IR variation.

#### Multi-Level Tactics / Progressive Raising (CGO 2021)

Multi-Level Tactics progressively raises affine loops to Linalg or BLAS using a declarative tactics language. Its artifact includes modified PolyBench/C inputs and explicitly documents a missed Darknet case caused by linearized accesses and missing delinearization, directly relevant to Polygeist's submap/debufferization work. ([paper and artifact appendix](https://grosser.science/static/7d02fb58ecc49e4d2097d11bc9e8152a/chelini-2021-abstraction-raising.pdf), [artifact](https://github.com/LoopTactics/mlir/tree/cgo))

#### mlirSynth (PACT 2023)

mlirSynth uses bottom-up enumerative synthesis, MLIR type constraints, and observational equivalences to raise low-level MLIR into Linalg or HLO. It evaluates PolyBench and reports geometric-mean gains of 2.5x on Intel, 3.4x on AMD, and 21.6x on TPU. ([paper](https://arxiv.org/abs/2310.04196), [repository](https://github.com/alexanderb14/mlirSynth))

#### Tensorize (CGO 2025)

Tensorize extends this line with symbolic traces, sketches, and algebraic solving. It raises loop-level programs into NumPy, PyTorch/JAX, and StableHLO-like tensor programs and claims symbolic equivalence within its supported language. ([paper](https://doi.org/10.1145/3696443.3708956), [repository](https://github.com/alexanderb14/tensorize))

Relevance: strongest modern synthesis-based comparator for Polygeist's loop-to-tensor raising stage.

#### ATC (CC 2023)

ATC combines synthesis, program classification, dynamic analysis, constraints, and lexical guidance to map source regions to accelerator APIs despite code and interface mismatches. The paper reports accelerating 2.6x–7x more programs than four earlier approaches. ([paper](https://arxiv.org/abs/2301.11659), [artifact](https://doi.org/10.5281/zenodo.7506749))

### Verified and synthesis-based lifting

- **Tenspiler (ECOOP 2024):** verified lifting from sequential C++ or Python into TensIR, followed by mappings to six tensor targets. It reports 105x average kernel and 9.65x end-to-end improvement over its evaluated suite. ([paper](https://doi.org/10.4230/LIPIcs.ECOOP.2024.32), [artifact](https://doi.org/10.4230/DARTS.10.2.17))
- **C2TACO (GPCE 2023):** guided enumerative synthesis from C and generated examples into TACO index notation. ([paper](https://doi.org/10.1145/3624007.3624053), [repository](https://github.com/JWesleySM/c2taco))
- **STAGG / Guided Tensor Lifting (PLDI 2025):** uses an LLM to induce search heuristics, synthesizes TACO, and validates candidates with examples and bounded model checking. ([paper](https://doi.org/10.1145/3729330), [repository](https://github.com/BugBugSurvival/Guided-Tensor-Lifting))
- **KONRUL / Guess, Measure & Edit (PACT 2025):** iteratively repairs an LLM-generated tensor expression by measuring its distance from the source. The authors report 98% lifting coverage and 4.07x CPU and 38.30x GPU geometric-mean speedups. A public artifact was not confirmed. ([paper](https://doi.org/10.1109/PACT65351.2025.00029))
- **LIAR (CGO 2024):** uses equality saturation over a compact functional array language to expose latent BLAS/PyTorch idioms with eight core semantic rules. It reports a 1.46x BLAS geometric-mean speedup and has artifact badges. ([paper](https://arxiv.org/abs/2312.17682), [project](https://jonathanvdc.github.io/compiler-work/latent-idiom-recognition/))

### Domain-specific recognition precedents

- **STNG / Verified Lifting of Stencils (PLDI 2016):** synthesizes and verifies Halide descriptions from Fortran stencil loops; reported median speedup 4.1x. ([paper](https://doi.org/10.1145/2908080.2908117))
- **FACC / Bind the Gap (PLDI 2022):** detects FFT procedures and synthesizes adapters across code, layout, domain, and behavioral mismatches for FFTW and hardware accelerators. ([paper](https://doi.org/10.1145/3519939.3523439), [repository](https://github.com/FourierACceleratorCompiler/FACC))
- **Helium (PLDI 2015):** dynamically traces stripped x86 binaries and lifts stencil kernels into Halide. ([project](https://projects.csail.mit.edu/helium/), [DOI](https://doi.org/10.1145/2737924.2737974))

### Adjacent offload systems

**Polly-ACC**, **PPCG**, **Bones**, **KernelGen**, and **GPU First** generate accelerator code from affine regions, loop skeletons, or whole legacy programs. They are useful offload baselines but do not primarily recover semantic library operations. LLVM's production `LoopIdiomRecognize` is a sanity baseline for narrow idioms such as memset and memcpy. ([Polly-ACC](https://pollylabs.org/publications/grosser-2016-polly-acc-transparent-compilation-to-heterogeneous-hardware.pdf), [PPCG](https://github.com/Meinersbur/ppcg), [Bones](https://cnugteren.github.io/downloads/Nugteren2012b.pdf), [KernelGen](https://github.com/dmikushin/kernelgen), [GPU First](https://arxiv.org/abs/2306.11686), [LLVM pass source](https://llvm.org/doxygen/LoopIdiomRecognize_8cpp_source.html))

## Part II: optimized kernel generation

### Highest-priority practical systems

#### MLIR Linalg, Transform dialect, and IREE

Linalg preserves structured computation while transformations control tiling, fusion, interchange, vectorization, promotion, and GPU mapping. The Transform dialect makes schedules explicit and composable. IREE provides a production-oriented AOT MLIR path, including CUDA deployment. This is the closest implementation fit for Polygeist. ([Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/), [Transform dialect paper](https://arxiv.org/abs/2409.03864), [Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/), [IREE](https://github.com/iree-org/iree))

Recommended role: deterministic and inspectable baseline, followed by a bounded Polygeist-native schedule search.

#### Bolt and CUTLASS templates (MLSys 2022)

Bolt searches vendor-native CUTLASS templates instead of rediscovering low-level CUDA schedules. It also composes templates for persistent fusion. The authors report 2.5x average inference improvement over the compared state of the art and tuning within 20 minutes. ([paper](https://proceedings.mlsys.org/paper_files/paper/2022/hash/1f8053a67ec8e0b57455713cefdd8218-Abstract.html))

Recommended role: dense GEMM/convolution and epilogue specialization; architectural model for “vendor call first, template second.”

#### TVM, Ansor, and TensorIR

TVM combines graph optimization and target code generation. Ansor samples hierarchical schedules and refines them through evolutionary search and a learned cost model. TensorIR makes tensor primitives and tensorization first-class. TVM explicitly documents host cross-compilation followed by remote upload, execution, and timing, which fits the Orin constraints. ([TVM](https://www.usenix.org/conference/osdi18/presentation/chen), [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng), [TensorIR](https://arxiv.org/abs/2207.04296), [cross-compilation/RPC](https://tvm.apache.org/docs/how_to/tutorials/cross_compilation_and_rpc.html))

Recommended role: strongest full autotuning baseline for operations that can be translated into TensorIR.

#### ROLLER, Welder, and TLM

- **ROLLER (OSDI 2022):** constructs hardware-aligned schedules with `rTile` abstractions and a micro-performance model, reducing generation from hours toward seconds. ([paper](https://www.usenix.org/conference/osdi22/presentation/zhu))
- **Welder (OSDI 2023):** jointly optimizes fusion and data movement through a tile graph; reports up to 2.8x over Ansor and 3x over TensorRT in its workloads. ([paper](https://www.usenix.org/system/files/osdi23-shi.pdf))
- **TLM (OSDI 2024):** represents schedule decisions in a model-friendly tensor language. It reports matching fully tuned Ansor/MetaSchedule with 61x faster compilation and 2.25x higher performance than ROLLER at equal compilation time. ([paper](https://www.usenix.org/conference/osdi24/presentation/zhai), [repository](https://github.com/zhaiyi000/tlm))

Recommended role: compare analytical construction, measured search, and learned schedule proposal under equal tuning budgets.

### Sparse kernel generators

- **TACO (OOPSLA 2017):** generates loops for compound dense/sparse tensor algebra using iteration graphs and merge lattices. ([paper](https://commit.csail.mit.edu/papers/2017/kjolstad-oopsla17-tensor-compiler.pdf))
- **SparseTIR (ASPLOS 2023):** exposes composable sparse formats and sparse transformations in a tensor IR. ([paper](https://arxiv.org/abs/2207.04606), [artifact](https://github.com/uwsampl/sparsetir-artifact))
- **MLIR SparseTensor:** provides structured sparse representations and an evolving GPU-lowering path. ([compiler paper](https://arxiv.org/abs/2202.04305), [documentation](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/))
- **SparseLNR (ICS 2022):** adds distribution and loop fusion to TACO while preserving sparse asymptotics. ([paper](https://arxiv.org/abs/2205.11622))
- **WACO (ASPLOS 2023):** learns joint storage-format and schedule choices from each sparsity pattern. Its published evaluation is CPU-based, so it is a methodology reference rather than an Orin performance baseline. ([paper](https://commit.csail.mit.edu/papers/2023/WACO_ASPLOS23.pdf), [artifact](https://github.com/nullplay/Workload-Aware-Co-Optimization))

Recommended role: compare generated SpMV with cuSPARSE using identical matrices, formats, precision, and conversion accounting.

### Scheduling languages and compiler frameworks

- **Halide (PLDI 2013) and the 2019 autoscheduler:** canonical separation of algorithm and schedule; particularly relevant to stencils and image pipelines. ([original paper](https://research.adobe.com/publication/halide-a-language-and-compiler-for-optimizing-parallelism-locality-and-recomputation-in-image-processing-pipelines/), [autoscheduler](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf))
- **Tiramisu (CGO 2019):** polyhedral dense/sparse code generation with explicit algorithms and schedules across CPUs, NVIDIA GPUs, FPGAs, and distributed systems. ([paper](https://arxiv.org/abs/1804.10694), [project](https://tiramisu-compiler.org/))
- **Tensor Comprehensions (2018):** mathematical tensor language with polyhedral JIT-to-CUDA lowering, fusion, specialization, autotuning, and caching. ([paper](https://arxiv.org/abs/1802.04730), [repository](https://github.com/facebookresearch/tensorcomprehensions))
- **DaCe (SC 2019):** data-centric Stateful DataFlow multiGraph for CPUs, GPUs, and FPGAs. ([repository](https://github.com/spcl/dace))
- **Fireiron (PACT 2020):** composable data-movement-aware scheduling for NVIDIA GPU GEMM and Tensor Cores. ([paper](https://arxiv.org/abs/2003.06324))
- **Exo (PLDI 2022; Exo 2 ASPLOS 2025):** user-schedulable exocompilation with externally defined hardware instructions and memories. Strong design reference, but not currently a ready Polygeist-to-CUDA backend. ([paper](https://people.csail.mit.edu/yuka/pdf/exo_pldi2022_full.pdf), [repository](https://github.com/exo-lang/exo))
- **RISE/Shine:** typed functional patterns and rewrite strategies targeting C, OpenMP, OpenCL, and CUDA. ([paper](https://arxiv.org/abs/2201.03611), [repository](https://github.com/rise-lang/shine))
- **UNIT (CGO 2021):** automatically maps loops to tensorized instructions across NVIDIA Tensor Cores, Intel VNNI, and Arm DOT. ([paper](https://arxiv.org/abs/2101.08458))

### Equality saturation and superoptimization

#### Mirage (OSDI 2025)

Mirage's µGraphs span kernel, thread-block, and thread levels, allowing algebraic transformation, scheduling, and custom-kernel generation in one search. It prunes using abstraction and validates equivalence probabilistically. ([paper](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi), [repository](https://github.com/mirage-project/mirage))

Recommended role: most ambitious comparator after simpler single-operation baselines work. Pin the OSDI 2025 artifact rather than assuming the evolving default branch is equivalent.

#### Other equality-saturation systems

- **Glenside (ASPLOS 2021):** tensor access-pattern IR for layout-aware rewriting and accelerator mapping. ([paper](https://arxiv.org/abs/2105.09377))
- **TENSAT (MLSys 2021):** tensor-graph equality saturation; graph optimizer rather than complete CUDA scheduler. ([paper](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf))
- **SPORES (VLDB 2020):** relational equality saturation for sum-product linear algebra. ([paper](https://www.vldb.org/pvldb/vol13/p1919-wang.pdf))
- **Diospyros (ASPLOS 2021):** equality-saturation DSP vectorization and translation validation. Its artifact documents a 38 GB failed saturation case, demonstrating the need for hard resource limits and failure logging. ([paper](https://cs.wellesley.edu/~avh/diospyros-asplos-2021-preprint.pdf), [artifact notes](https://github.com/cucapra/diospyros/blob/master/evaluation/README.md))

### Tile languages and expert frameworks

- **Triton (MAPL 2019):** productive block/tile GPU language whose compiler manages many within-block choices. Its canonical Python workflow uses JIT kernels and target autotuning, so an AOT cross-deployment path must be established for this project. ([paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf), [documentation](https://triton-lang.org/))
- **TileLang (ICLR 2026):** explicit tile-level placement, movement, and scheduling with inference/recommendation. Its H100/AMD results cannot be assumed to transfer to Orin. ([paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/76fb92288bf90360c527efb0d1c2aba6-Abstract-Conference.html), [repository](https://github.com/tile-ai/tilelang))
- **ThunderKittens (ICLR 2025):** warp/block/grid tile abstractions with vendor-level reported performance for several kernels. It is an expert framework, not automatic Linalg lowering. ([paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/05dc08730e32441edff52b0fa6caab5f-Abstract-Conference.html))

### LLM and learned kernel generation

**KernelBench (ICML 2025)** provides 250 PyTorch workloads and a metric that requires both correctness and a speedup threshold. Initial frontier models matched the PyTorch baseline on fewer than 20% of tasks, underscoring that plausible generated code is not an end-to-end result. ([paper](https://proceedings.mlr.press/v267/ouyang25a.html), [repository](https://github.com/ScalingIntelligence/KernelBench))

**GEAK**, **AutoKernel**, fine-tuned GPT-5 kernel generation, and **RealisticTritonBench** are recent preprints or emerging benchmarks. They should remain exploratory until their claims and target assumptions are independently reproduced. ([GEAK](https://arxiv.org/abs/2507.23194), [AutoKernel](https://arxiv.org/abs/2603.21331), [GPT-5 kernel-generation preprint](https://arxiv.org/abs/2602.11000), [RealisticTritonBench](https://arxiv.org/abs/2608.12004))

## Unified comparison strategy

### Recognition denominator

Create one row per confirmed source/IR occurrence. For every tool, record:

- source and exact revision;
- function and region;
- preprocessing, optimization, and inlining pipeline;
- semantic operation;
- detected only;
- replacement generated;
- executable produced;
- correctness passed;
- silicon execution completed;
- runtime compared.

Do not mix program-level, source-occurrence, IR-region, or dynamic-execution counts.

### Kernel contract

The raising stage should produce a contract containing:

- operation semantics and reduction identity;
- shapes, strides, layouts, aliasing, and alignment assumptions;
- datatype, accumulation mode, and numerical tolerance;
- fusion boundaries and side effects;
- dynamic-shape constraints;
- applicable backend predicates.

This contract decouples recognition from implementation selection and permits the same occurrence to be evaluated with a vendor API, CUTLASS template, MLIR schedule, TVM schedule, sparse generator, or superoptimizer.

### Backend decision order

1. Exact vendor call when the contract and economics fit.
2. Hardware-native template for supported dense operations.
3. Fixed MLIR Transform-dialect AOT schedule.
4. Bounded offline schedule search.
5. Sparse format/schedule co-optimization when conversion is justified.
6. Cross-operation algebraic or superoptimization search.

## Jetson Orin execution model

All source, MLIR, CUDA, and host compilation must remain on x86. Candidate generation should follow:

1. Generate schedules and source on the x86 host.
2. Cross-compile the AArch64 launcher and device image for `sm_87`.
3. Transfer only finished executables and required shared libraries.
4. Run correctness and timing on Orin.
5. Return measurements to the host-side search controller.
6. Cache results by operation, shape, layout, datatype, board, CUDA/library revisions, and power mode.

TVM's remote execution design is a concrete model, though SSH batching may be simpler initially. The target must not invoke nvcc, MLIR, Triton JIT compilation, or any source compiler. NVIDIA identifies Jetson Orin as Ampere `sm_87`; binaries should include a compatible native device image and avoid accidental target-side PTX JIT. ([TVM cross-compilation/RPC](https://tvm.apache.org/docs/how_to/tutorials/cross_compilation_and_rpc.html), [NVIDIA architecture table](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html), [CUDA 12.6 Ampere guide](https://docs.nvidia.com/cuda/archive/12.6.0/ampere-compatibility-guide/index.html))

## Prioritized reproduction roadmap

### Priority 1: reconstruct Ginsbach's denominator

Finish the 60-occurrence manifest before any headline comparison. Preserve the original compiler, source revisions, flags, specifications, and logs.

### Priority 2: shared sparse corpus

Run Polygeist, Ginsbach/LiLAC, and SpEQ on byte-identical NPB-CG and Parboil SpMV inputs. Add SLEB if its artifact becomes available. Compare cuSPARSE with MLIR Sparse/SparseTIR-generated code only after recognition and correctness are established.

### Priority 3: dense raising and implementation selection

Use Parboil SGEMM and exact artifact-shipped PolyBench inputs to compare Polygeist, KernelFaRer, SMR/PGL, Multi-Level Tactics, mlirSynth, and Tensorize. For each recovered GEMM, compare cuBLAS, CUTLASS, fixed MLIR schedules, and TVM/Ansor.

### Priority 4: robustness suite

Generate semantics-preserving variants involving loop interchange, scalar temporaries, linearized indexing, pointer offsets, inlining, split initialization, and benign control-flow changes. Measure coverage and false positives separately from performance.

### Priority 5: representative kernel-generation study

Select GEMM, 3D stencil, reduction/histogram, SpMV, and one fused elementwise/reduction computation. Compare:

- existing Polygeist CUDA lowering;
- upstream MLIR/IREE;
- applicable NVIDIA library;
- CUTLASS template;
- TVM/Ansor schedule;
- specialized sparse or stencil generator where appropriate.

Only after those baselines are stable should Mirage, equality saturation, TLM-style proposal, or LLM-generated kernels be introduced.

## Measurement and fairness requirements

- Validate with independent numerical references and nontrivial inputs before timing.
- Report kernel-only and end-to-end time, including transfers, packing, sparse conversion, initialization, and synchronization.
- Separate compile/search time from steady-state runtime and state the amortization assumption.
- Fix clocks and power mode; record temperature, warm-ups, repetitions, and confidence intervals.
- Record register count, shared memory, occupancy, code size, launch configuration, and compiler diagnostics.
- Include awkward shapes and nonmultiples of tile sizes.
- State floating-point rules explicitly; mixed precision, reassociation, approximate math, and nondeterministic reductions are distinct result classes.
- Tune from Orin execution data. Datacenter-GPU schedules are starting points, not Orin results.
- Retain source revisions, toolchains, commands, generated code, binaries, inputs, and raw logs.

## Defensible research position

The strongest project question is:

> Can semantic information recovered from unmodified legacy programs drive a hybrid library/template/generated-kernel backend that approaches vendor performance on an embedded GPU while preserving an auditable correctness chain?

This differs from tensor compilers that begin with tensor graphs, tile systems that begin with expert-authored kernels, and legacy-code matchers that stop at API replacement. A strong Polygeist result would explain when each implementation path wins:

- vendor library for standardized, sufficiently large operations;
- template for known dense structure and fixed shapes;
- generated/fused kernel for unusual, small, or composition-heavy computations;
- sparse generator when format specialization repays conversion;
- CPU when launch, transfer, or conversion overhead dominates.

The contribution should be the semantics-preserving bridge, transparent backend selection, occurrence-level accounting, and embedded-GPU evidence—not a claim that generated CUDA universally beats mature libraries.

## Evidence gaps

- Artifact availability was inspected, but the systems were not built during these literature passes.
- Headline speedups use incompatible hardware, inputs, baselines, numerical modes, timing boundaries, and tuning budgets.
- SLEB's paper-referenced repository was unavailable during the audit.
- Standalone PGL and KONRUL artifacts were not confirmed.
- Recent H100/Blackwell-oriented systems may rely on features absent from Orin.
- JIT systems require a demonstrated AOT/cross path before they comply with the Jetson policy.
- Equality-saturation systems need time and memory limits plus retained failed-search statistics.
- Example-driven and learned generators require held-out correctness tests.
- Sparse conclusions require multiple real matrices because the sparsity distribution can dominate performance.

## Source research reports

This document consolidates and cleans the two detailed research passes:

- [`related_work_research/report-source.md`](related_work_research/report-source.md)
- [`kernel_generation_research/report-source.md`](kernel_generation_research/report-source.md)

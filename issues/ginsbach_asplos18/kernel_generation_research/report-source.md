# Optimized Kernel Generation: A Research Map for Polygeist and Jetson Orin

Research date: 2026-09-05

## Executive conclusion

The literature suggests that Polygeist should not commit to one universal kernel generator. The strongest architecture is a **hybrid backend**: preserve the semantic operation recovered by Polygeist, select a vendor library when it is a good fit, and otherwise generate and tune an ahead-of-time CUDA kernel. This mirrors the central lesson of Bolt—combine tunable vendor-quality templates with compiler automation—while retaining Polygeist's advantage: it begins with semantics recovered from legacy source rather than a pre-existing ML graph. ([Bolt paper](https://proceedings.mlsys.org/paper_files/paper/2022/hash/1f8053a67ec8e0b57455713cefdd8218-Abstract.html))

For the current project, the best near-term stack is:

1. **MLIR Linalg + Transform dialect + CUDA/IREE** for a deterministic, inspectable AOT baseline;
2. **cuBLAS/cuDNN/cuSPARSE/cuFFT first, and CUTLASS-style generated templates second**, for operations those libraries cover;
3. **TVM TensorIR/Ansor or a smaller Polygeist-native schedule search** for shapes and fused kernels where templates or libraries are insufficient;
4. **MLIR Sparse, TACO, and SparseTIR concepts** for sparse kernels;
5. **Mirage-style multi-level search** as a research comparison after correctness and simpler baselines are established.

This ordering is particularly important for Jetson Orin. TVM documents a host-side cross-compilation and remote-execution workflow in which the target only loads and runs generated modules. That complies with the project's “do not compile on the Jetson” rule. Triton and several newer tile-language systems are valuable intellectually, but their canonical interfaces use JIT-compiled kernels and target-side autotuning; they require an explicit AOT deployment study before they can be treated as Orin-ready baselines. ([TVM cross-compilation/RPC guide](https://tvm.apache.org/docs/how_to/tutorials/cross_compilation_and_rpc.html), [Triton documentation](https://triton-lang.org/), [Triton autotune API](https://triton-lang.org/main/python-api/generated/triton.autotune.html))

No published result found establishes that one generator dominates on embedded Ampere. Most recent headline results are from datacenter GPUs, often H100 or newer, and many evaluations begin from tensor graphs rather than recovered legacy code. The defensible Polygeist contribution is therefore not merely a faster generated kernel. It is a measured end-to-end path from legacy source, through semantic raising, to a correct library call or AOT-generated `sm_87` kernel, with target-specific offline tuning and explicit accounting for compile/search cost.

## Scope and definitions

This pass covers systems that generate or optimize the implementation *after* a computation has been recognized or expressed in a structured form. It includes schedule languages, tensor compilers, autotuners, template generators, sparse compilers, equality-saturation/superoptimization systems, and recent learned generators.

Three adjacent categories are kept distinct:

- **Kernel generation:** emits a GPU kernel or parameterized kernel implementation.
- **Graph optimization:** fuses or algebraically rewrites multiple operators, but may ultimately call existing kernels.
- **Kernel programming frameworks:** make expert-written kernels more productive, but do not automatically derive a kernel from Polygeist/Linalg input.

Performance numbers below are the respective authors' results, not independently reproduced measurements. “Artifact available” means a public paper, repository, or artifact page was found; it does not mean it was built during this literature pass.

## Ranked systems for this project

### Tier 1: practical integration and reproduction targets

#### 1. MLIR structured code generation and the Transform dialect

MLIR's Linalg design represents structured computation while keeping tiling, fusion, interchange, vectorization, and promotion as explicit transformations. The Transform dialect makes these choices programmable and composable rather than burying them in a monolithic pass pipeline. The Transform-dialect paper demonstrates fine-grained transformation control and integration with search across five case studies. This is the closest architectural fit because Polygeist already raises computations into MLIR. ([Linalg documentation](https://mlir.llvm.org/docs/Dialects/Linalg/), [Transform dialect paper](https://arxiv.org/abs/2409.03864), [Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/), [composable MLIR code-generation paper](https://arxiv.org/abs/2202.03293))

Use: establish a deterministic baseline and a compact, inspectable schedule space: tile dimensions, block/warp mapping, shared-memory promotion, vector width, unrolling, reduction strategy, and fusion. IREE is a useful production-oriented AOT comparison because it lowers MLIR programs to deployable backends including CUDA. ([IREE repository](https://github.com/iree-org/iree), [IREE deployment configurations](https://iree.dev/guides/deployment-configurations/))

Risk: upstream transformations do not by themselves choose the best schedule. Polygeist will need fixed expert schedules, search, or both. MLIR APIs also evolve, so exact revisions must be pinned.

#### 2. Bolt and CUTLASS-style template search (MLSys 2022)

Bolt searches hardware-native CUTLASS templates instead of rediscovering every CUDA implementation decision from a generic schedule space. It also composes templates for persistent fusion. The authors report 2.5x average inference improvement over the compared state of the art and tuning within 20 minutes. ([paper](https://proceedings.mlsys.org/paper_files/paper/2022/file/1f8053a67ec8e0b57455713cefdd8218-Paper.pdf))

Use: this is the best model for dense GEMM/convolution lowering. A Polygeist operation can carry shape, layout, type, and epilogue semantics into a CUTLASS manifest search, then emit a fixed AArch64 host executable plus `sm_87` device code. It also naturally supports a policy of “library call first; custom template when specialization or fusion wins.”

Risk: Bolt's reported evaluation is DNN-centric and not an Orin study. CUTLASS support and template availability vary by operation, datatype, and GPU generation. Every selected instantiation still needs independent correctness and Orin measurement.

#### 3. TVM, Ansor, and TensorIR

TVM combines graph optimization, tensor-expression lowering, and target-specific code generation. Ansor constructs a large schedule space through hierarchical program sampling, then uses evolutionary search and a learned cost model; the paper reports up to 1.7x over prior state of the art on NVIDIA GPUs. TensorIR makes tensor computation primitives first-class and supports automatic tensorization. ([TVM OSDI 2018](https://www.usenix.org/conference/osdi18/presentation/chen), [Ansor OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/zheng), [TensorIR](https://arxiv.org/abs/2207.04296))

Use: TVM is the strongest full autotuning baseline for Polygeist-generated tensor operations. Its documented cross-compile/RPC loop can compile on the x86 host, upload modules, and use the target only for execution and timing. That is a direct match for the Jetson policy. ([cross-compilation/RPC guide](https://tvm.apache.org/docs/how_to/tutorials/cross_compilation_and_rpc.html))

Risk: integration requires a stable bridge from Linalg or a semantic operation descriptor into TensorIR. Search time must be reported separately from runtime, and the runtime/RPC component must itself be cross-built rather than compiled on the Orin.

#### 4. ROLLER (OSDI 2022) and Welder (OSDI 2023)

ROLLER constructs schedules from hardware-aligned `rTile` abstractions and a micro-performance model, aiming for seconds of generation rather than hours of measurement-heavy search. Welder adds a tile graph that jointly reasons about operator fusion and data movement. Welder reports up to 2.8x over Ansor and up to 3x over TensorRT in its evaluated workloads. ([ROLLER](https://www.usenix.org/conference/osdi22/presentation/zhu), [Welder paper](https://www.usenix.org/system/files/osdi23-shi.pdf))

Use: ROLLER's construction-based approach is attractive for an embedded board where exhaustive profiling is expensive. Welder is relevant once Polygeist preserves graphs of adjacent raised operations rather than lowering each match independently.

Risk: both primarily target tensor/DNN workloads. Their original hardware and software stacks must be reproduced before using reported performance as a comparator, and neither result should be presumed to transfer to `sm_87`.

#### 5. TLM (OSDI 2024)

The Tensor Language Model represents scheduling decisions in a model-friendly tensor language and samples informed programs using offline learning and previous decisions. The paper reports performance matching fully tuned Ansor/MetaSchedule with 61x faster compilation, and 2.25x higher performance than ROLLER at equal compilation time. The authors publish a TVM-based repository. ([paper and code link](https://www.usenix.org/conference/osdi24/presentation/zhai), [repository](https://github.com/zhaiyi000/tlm))

Use: a strong learned-search comparator if compile latency becomes a first-class concern. It is much closer to compiler scheduling than free-form LLM CUDA generation.

Risk: the released stack is based on an older TVM revision, and its training distribution and evaluated GPUs may not represent Orin. Reproduction should first use the paper's original environment; porting and Orin retuning are separate experiments.

#### 6. SparseTIR, TACO, and MLIR Sparse

TACO introduced automatic code generation for compound dense/sparse tensor algebra using iteration graphs and merge lattices. SparseTIR exposes composable sparse formats and transformations in a tensor-IR framework. MLIR has an actively developed SparseTensor dialect and GPU code-generation path. ([TACO OOPSLA 2017 paper](https://commit.csail.mit.edu/papers/2017/kjolstad-oopsla17-tensor-compiler.pdf), [TACO project record](https://compilers.stanford.edu/publications/ase17/), [SparseTIR paper](https://arxiv.org/abs/2207.04606), [SparseTIR artifact](https://github.com/uwsampl/sparsetir-artifact), [MLIR sparse compiler paper](https://arxiv.org/abs/2202.04305), [MLIR SparseTensor documentation](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/))

Use: these are the natural generator comparisons for SpMV and other sparse idioms recovered in the Ginsbach/Polygeist corpus. Compare generated code with direct cuSPARSE replacement on exactly the same matrix, format, precision, and transfer policy.

Risk: sparse performance is inseparable from storage format and the actual sparsity pattern. A single synthetic matrix is not a valid general conclusion.

### Tier 2: high-value research comparators

#### 7. Mirage (OSDI 2025)

Mirage is a multi-level tensor-program superoptimizer. Its µGraph representation spans kernel, thread-block, and thread levels, permitting algebraic rewrites, schedule transformations, and generation of new custom kernels in one search. It prunes with abstraction and checks candidate equivalence probabilistically. The implementation is public. ([paper](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi), [repository](https://github.com/mirage-project/mirage))

Use: this is the strongest ambitious comparison for Polygeist's equality-saturation direction. It tests whether keeping algebra and scheduling in one representation can discover fused kernels that a sequential “match, then schedule” pipeline misses.

Risk: probabilistic equivalence is not a proof of floating-point identity, and supported operator/type/shape domains need auditing. The repository's evolving default branch must not be mistaken for the OSDI 2025 artifact; pin the paper artifact or corresponding revision.

#### 8. Halide and its autoschedulers

Halide's durable contribution is the separation of algorithm from schedule. Its 2019 autoscheduler combines tree search, a learned cost model, and random-program training; the authors report a 75% average improvement over the previous production autoscheduler without autotuning and 135% with hours of autotuning. ([Halide PLDI 2013](https://research.adobe.com/publication/halide-a-language-and-compiler-for-optimizing-parallelism-locality-and-recomputation-in-image-processing-pipelines/), [2019 autoscheduler paper](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf))

Use: canonical comparison for stencil/image pipelines and for the design of a clean semantic/schedule boundary.

Risk: translating arbitrary raised Linalg into Halide is not free, and Halide's strongest coverage is pipeline-oriented rather than general sparse or irregular GPU code.

#### 9. Tiramisu, Tensor Comprehensions, and DaCe

Tiramisu uses a polyhedral representation and separates algorithms from schedules across dense and sparse computations. Tensor Comprehensions provides mathematical tensor notation, a polyhedral JIT-to-CUDA compiler, specialization, fusion, autotuning, and a compilation cache. DaCe's Stateful DataFlow multiGraph exposes data movement and maps programs to CPUs, GPUs, and FPGAs. ([Tiramisu paper](https://arxiv.org/abs/1804.10694), [Tiramisu project](https://tiramisu-compiler.org/), [Tensor Comprehensions paper](https://arxiv.org/abs/1802.04730), [Tensor Comprehensions repository](https://github.com/facebookresearch/tensorcomprehensions), [DaCe repository](https://github.com/spcl/dace))

Use: architectural and historical baselines for explicit schedules, polyhedral transformation, fusion, and data-centric optimization.

Risk: older artifacts may require obsolete CUDA/LLVM stacks. Reproduction should preserve those versions rather than silently modernize them; Orin porting is a separate result.

#### 10. Fireiron, Exo, and RISE/Shine

Fireiron is a composable, data-movement-aware scheduling language for NVIDIA GPU kernels and tensor-core GEMM. Exo externalizes scheduling and hardware instructions so experts can develop optimized subprograms without modifying the compiler. RISE/Shine uses typed functional patterns and rewrite strategies to derive C, OpenMP, OpenCL, and CUDA. ([Fireiron](https://arxiv.org/abs/2003.06324), [Exo PLDI 2022](https://people.csail.mit.edu/yuka/pdf/exo_pldi2022_full.pdf), [Exo repository](https://github.com/exo-lang/exo), [RISE/Shine paper](https://arxiv.org/abs/2201.03611), [Shine repository](https://github.com/rise-lang/shine))

Use: these systems supply design patterns for a Polygeist scheduling language: composability, instruction mapping, explicit memory placement, and separation of trusted semantics from user-controlled optimization.

Risk: Fireiron is specialized, Exo is currently strongest for CPU/DSP/custom-instruction kernels rather than a ready CUDA backend, and RISE/Shine requires a functional-pattern translation. They are better design references than immediate end-to-end competitors.

#### 11. UNIT and automatic tensorization

UNIT describes tensorized instruction semantics and automatically checks applicability and rewrites loops for NVIDIA Tensor Cores, Intel VNNI, and Arm DOT instructions. The authors report 1.75x over cuDNN on their GPU experiments. ([paper](https://arxiv.org/abs/2101.08458), [author PDF](https://polyarch.cs.ucla.edu/papers/cgo2021-unit.pdf))

Use: relevant when raised GEMM/convolution/reduction operations can exploit Ampere tensor instructions. It also provides a useful distinction between selecting an instruction and scheduling the whole kernel.

Risk: numerical mode, accumulation type, layout constraints, and tolerance must be made explicit. A tensor-core result is not apples-to-apples with strict FP32 semantics unless both implement the same contract.

### Tier 3: sparse specialization, equality saturation, and emerging systems

#### Sparse specialization

- **SparseLNR (ICS 2022)** adds kernel distribution and loop fusion to TACO through a branched iteration graph, improving locality while preserving sparse asymptotics. ([paper](https://arxiv.org/abs/2205.11622))
- **WACO (ASPLOS 2023)** jointly learns sparse format and schedule choices from the actual sparsity pattern. It is especially relevant to a Polygeist experiment that can preprocess a fixed dataset offline, although the paper's evaluation is CPU-based rather than an Orin GPU result. ([paper](https://commit.csail.mit.edu/papers/2023/WACO_ASPLOS23.pdf), [artifact](https://github.com/nullplay/Workload-Aware-Co-Optimization))

#### Equality saturation around kernel generation

- **Glenside (ASPLOS 2021)** represents tensor access patterns explicitly, allowing layout-aware equality saturation and accelerator mapping without low-level loop bookkeeping. ([paper](https://arxiv.org/abs/2105.09377), [project page](https://ztatlock.net/pubs/2021-maps-glenside/))
- **TENSAT (MLSys 2021)** applies equality saturation to tensor computation graphs and reports up to 16% improvement over its compared state of the art with much lower optimization time. It is a graph optimizer, not a complete CUDA kernel scheduler. ([paper](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf))
- **SPORES (VLDB 2020)** uses relational equality saturation for sum-product linear algebra and reports 1.2–5x improvements. It is most relevant to algebraic normalization before kernel selection. ([paper](https://www.vldb.org/pvldb/vol13/p1919-wang.pdf))
- **Diospyros (ASPLOS 2021)** uses equality saturation for DSP vectorization and translation validation. Its artifact documents a case consuming 38 GB and failing to saturate, an important warning that rule selection, extraction, and resource limits must be treated as experimental variables. ([paper](https://cs.wellesley.edu/~avh/diospyros-asplos-2021-preprint.pdf), [artifact evaluation notes](https://github.com/cucapra/diospyros/blob/master/evaluation/README.md))

#### Tile languages and expert kernel frameworks

- **Triton (MAPL 2019)** exposes block/tile programs and lets the compiler handle much of within-block layout, memory, and instruction scheduling. It is a compelling generated target, but the common Python interface uses `@triton.jit`, and autotuning runs candidate configurations; an AOT/cross deployment must be demonstrated before Orin use. ([paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf), [documentation](https://triton-lang.org/))
- **TileLang (ICLR 2026)** exposes tile-level placement, data movement, and schedules, with tile inference and recommendation. Its authors report large gains over Triton on H100 and AMD GPUs. This is promising but not evidence for Orin, and its deployment model needs auditing. ([paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/76fb92288bf90360c527efb0d1c2aba6-Abstract-Conference.html), [repository](https://github.com/tile-ai/tilelang))
- **ThunderKittens (ICLR 2025)** provides warp/block/grid tile abstractions and reports vendor-level results for several kernels. It is an expert framework, not an automatic lowering from Linalg, and is primarily a reference for good templates and abstractions. ([paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/05dc08730e32441edff52b0fa6caab5f-Abstract-Conference.html), [preprint](https://arxiv.org/abs/2410.20399))

#### LLM-generated kernels

**KernelBench (ICML 2025)** supplies 250 PyTorch workloads and evaluates both correctness and speedup thresholds through `fast_p`. It found that initial frontier models matched the PyTorch baseline on fewer than 20% of tasks, showing that plausible kernel text is not the same as a correct optimized implementation. The benchmark and code are public. ([paper](https://proceedings.mlr.press/v267/ouyang25a.html), [repository](https://github.com/ScalingIntelligence/KernelBench))

Recent agentic or fine-tuned systems report stronger results, but many are preprints and often optimize Triton/CUDA through repeated compile-run feedback on datacenter GPUs. They should be an exploratory track, not the primary scientific baseline, until independently reproduced. ([GEAK](https://arxiv.org/abs/2507.23194), [AutoKernel](https://arxiv.org/abs/2603.21331), [fine-tuned GPT-5 kernel generation](https://arxiv.org/abs/2602.11000), [RealisticTritonBench](https://arxiv.org/abs/2608.12004))

## Recommended Polygeist architecture

The matcher should produce a **kernel contract**, not immediately commit to `gpu.launch`. The contract should contain:

- operation semantics and reduction identity;
- shapes, strides, layouts, alias assumptions, and alignment;
- datatype, accumulation mode, and numerical tolerance;
- fusion boundaries and observable side effects;
- dynamic-shape constraints;
- candidate backends and applicability predicates.

Then use the following decision path:

1. Select an exact vendor call when it covers the contract and transfer/marshalling cost is acceptable.
2. Try a pinned CUTLASS or other hardware-native template for dense tensor operations.
3. Generate an MLIR/CUDA kernel from a fixed Transform-dialect schedule.
4. Search a bounded schedule space offline when specialization is likely to repay tuning cost.
5. For sparse operations, jointly consider storage format and schedule, but include conversion cost.
6. Only then apply cross-operation algebraic/superoptimization search.

Every path must return the same external ABI and be checked against an independent reference. This keeps recognition coverage separate from generator quality and lets one Polygeist occurrence be compared across cuBLAS/cuSPARSE, MLIR, CUTLASS, TVM, and research generators.

## Orin-compatible tuning design

The x86 host should own all generation and compilation. For each semantic occurrence and representative shape:

1. Generate candidate MLIR/CUDA/template configurations on the host.
2. Cross-compile the AArch64 launcher and compile device code for `sm_87` on the host;
3. batch complete executables and required shared libraries;
4. transfer them to `/home/nvidia` on the Orin;
5. execute correctness and timing only on the Orin;
6. return results to the host-side search controller;
7. cache the winning configuration by operation, shape/layout/type, device identity, software versions, and power mode.

TVM RPC is a concrete model for this workflow, though the project may initially use simpler SSH batch execution. The target must never invoke nvcc, MLIR, Triton JIT compilation, or source compilation. NVIDIA's current architecture documentation identifies Jetson Orin as Ampere `sm_87`; generated binaries should contain an explicit compatible image rather than rely accidentally on a target-side recompilation path. ([NVIDIA architecture table](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html), [CUDA 12.6 Ampere compatibility guide](https://docs.nvidia.com/cuda/archive/12.6.0/ampere-compatibility-guide/index.html))

## Experimental roadmap

### Phase 1: deterministic baselines

- Choose five representative recovered operations: GEMM, stencil, reduction/histogram, SpMV, and one fused elementwise/reduction case.
- Freeze source, input, Polygeist/LLVM/CUDA revisions, power mode, and clocks.
- Compare the native CPU implementation, existing Polygeist GPU lowering, upstream MLIR/IREE lowering, and the applicable NVIDIA library.
- Validate nontrivial and awkward shapes before timing.

### Phase 2: bounded schedule search

- Express tiling, thread mapping, vectorization, shared-memory promotion, unrolling, and reduction strategy through the Transform dialect or an equivalent manifest.
- Generate and compile candidates on x86; run candidates remotely in batches.
- Compare random search, a small analytical model, and measured search. Record tuning wall time and number of target trials.

### Phase 3: specialized generators

- Dense: CUTLASS/Bolt-style templates versus TVM TensorIR/Ansor.
- Sparse: cuSPARSE versus MLIR Sparse/SparseTIR/TACO-generated code, including format-conversion cost.
- Stencil: MLIR schedules versus Halide autoscheduling if a faithful translation can be constructed.
- Fused graphs: Welder/Mirage concepts only after single-operation baselines are stable.

### Phase 4: research extensions

- Use equality saturation to select algebraically equivalent kernel contracts or expose fusion, with hard resource budgets and extraction diagnostics.
- Evaluate learned schedule proposal using TLM-style decision sequences.
- Optionally test LLM code generation under KernelBench-like rules, never counting a candidate until independent correctness passes.

## Measurement and fairness rules

- Report recognition, kernel generation, compilation, correctness, hardware execution, and performance as separate stages.
- Use the identical semantic occurrence and input for every backend.
- Report kernel-only and end-to-end time; include packing, sparse-format conversion, transfers, initialization, and synchronization where applicable.
- Separate one-time compile/search cost from steady-state execution and state the amortization assumption.
- Use warm-ups, repeated trials, confidence intervals, fixed power mode/clocks, and thermal monitoring.
- Record register count, static/dynamic shared memory, occupancy, code size, launch configuration, and compiler diagnostics.
- Test multiple realistic and adversarial shapes, including nonmultiples of tile sizes and alias/layout edge cases.
- Define floating-point semantics before optimization. Mixed precision, reassociation, approximate math, and nondeterministic reductions are separate result classes.
- Tune on Orin execution data. A schedule selected on A100/H100 is a starting point, not an Orin result.
- Compare against the exact installed vendor-library versions and retain emitted cubins/PTX, commands, logs, and configuration manifests.

## Most defensible paper-level positioning

The strongest research question is:

> Can semantic information recovered from unmodified legacy programs drive a hybrid library/template/generated-kernel backend that approaches vendor performance on an embedded GPU, while preserving an auditable correctness chain?

That differs from most tensor compilers, which begin with a tensor graph; from Triton-like systems, which begin with a manually written tile program; and from Ginsbach-style matching, which largely ends at API replacement. A strong result would quantify where each backend wins and why:

- vendor library for standardized, sufficiently large operations;
- specialized template for known dense structure and fixed shapes;
- generated/fused kernel for small, unusual, or composition-heavy computations;
- sparse generator when format specialization repays conversion cost;
- CPU when launch and transfer overhead dominate.

The novelty is the semantics-preserving bridge and selection policy, not a claim that generated CUDA universally beats mature libraries.

## Evidence gaps and cautions

- None of the cited artifacts was built during this research pass.
- Results across papers use different GPUs, workloads, numerical modes, baselines, tuning budgets, and timing boundaries; their headline speedups are not mutually comparable.
- Recent H100/Blackwell-oriented tile frameworks may depend on hardware features absent from Orin.
- Public repositories can move beyond the paper artifact. Exact tags or commit hashes must be recorded before reproduction.
- JIT-based systems are not disqualified, but compliance with the no-compilation-on-Jetson rule must be demonstrated rather than assumed.
- Equality-saturation and synthesis systems need explicit time/memory limits and must retain failed-search statistics.
- Learned and LLM generators require held-out correctness tests; successful compilation or sample-input agreement is insufficient.
- Sparse results require dataset-level reporting because format and sparsity distribution can dominate the schedule.

## Bottom line

Start with MLIR Transform/IREE, NVIDIA libraries, and CUTLASS templates; add a host-compiled TVM/Ansor tuning loop; then evaluate sparse generators and Mirage. This sequence yields useful Orin results early, respects the deployment constraints, and creates a stable platform on which more speculative equality-saturation or learned kernel generation can be evaluated fairly.

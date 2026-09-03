#ifndef POLYGEIST_PASSES
#define POLYGEIST_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def SelectFunc : Pass<"select-func"> {
  let summary = "Run a pass pipeline on selected functions by name";
  let constructor = "mlir::polygeist::createSelectFuncPass()";
  let options = [
    Option<"pipeline", "pipeline", "std::string", /*default=*/"\"\"",
           "The pass pipeline to run on filtered functions">,
    ListOption<"funcNames", "func-name", "std::string",
               "Function names to process (if empty, process all)">
  ];
}

def AffineCFG : Pass<"affine-cfg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::polygeist::replaceAffineCFGPass()";
}

def PolygeistMem2Reg : Pass<"polygeist-mem2reg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::polygeist::createPolygeistMem2RegPass()";
}

def SCFParallelLoopUnroll : Pass<"scf-parallel-loop-unroll"> {
  let summary = "Unroll and interleave scf parallel loops";
  let dependentDialects = [
    "scf::SCFDialect",
    "arith::ArithDialect",
  ];
  let constructor = "mlir::polygeist::createSCFParallelLoopUnrollPass()";
  let options = [
  Option<"unrollFactor", "unrollFactor", "int", /*default=*/"2", "Unroll factor">
  ];
}

def CollectKernelStatistics : Pass<"collect-kernel-statistics", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to cpu versions";
  let dependentDialects = [];
  let constructor = "mlir::polygeist::createCollectKernelStatisticsPass()";
}

def LowerAlternatives : Pass<"lower-alternatives", "mlir::ModuleOp"> {
  let summary = "Lower alternatives if in opt mode";
  let dependentDialects = [];
  let constructor = "mlir::polygeist::createLowerAlternativesPass()";
}

def ConvertCudaRTtoCPU : Pass<"convert-cudart-to-cpu", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to cpu versions";
  let dependentDialects = [
    "memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect",
    "cf::ControlFlowDialect",
  ];
  let constructor = "mlir::polygeist::createConvertCudaRTtoCPUPass()";
}

def FixGPUFunc : Pass<"fix-gpu-func", "mlir::gpu::GPUModuleOp"> {
  let summary = "Fix nested calls to gpu functions we generate in the frontend";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
  let constructor = "mlir::polygeist::createFixGPUFuncPass()";
}

def ConvertCudaRTtoGPU : Pass<"convert-cudart-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to generic gpu versions";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
  let constructor = "mlir::polygeist::createConvertCudaRTtoGPUPass()";
}

def ConvertCudaRTtoHipRT : Pass<"convert-cudart-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to generic gpu versions";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
  let constructor = "mlir::polygeist::createConvertCudaRTtoGPUPass()";
}

def ParallelLower : Pass<"parallel-lower", "mlir::ModuleOp"> {
  let summary = "Lower gpu launch op to parallel ops";
  let dependentDialects = [
    "scf::SCFDialect",
    "polygeist::PolygeistDialect",
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
    "func::FuncDialect",
    "LLVM::LLVMDialect",
  ];
  let constructor = "mlir::polygeist::createParallelLowerPass()";
}

def AffineReduction : Pass<"detect-reduction"> {
  let summary = "Detect reductions in affine.for";
  let constructor = "mlir::polygeist::detectReductionPass()";
}

def SCFCPUify : Pass<"cpuify"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createCPUifyPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
  let options = [
  Option<"method", "method", "std::string", /*default=*/"\"distribute\"", "Method of doing distribution">
  ];
}

def ConvertParallelToGPU1 : Pass<"convert-parallel-to-gpu1"> {
  let summary = "Convert parallel loops to gpu";
  let constructor = "mlir::polygeist::createConvertParallelToGPUPass1()";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "memref::MemRefDialect", "gpu::GPUDialect"];
  let options = [
  Option<"arch", "arch", "std::string", /*default=*/"\"sm_60\"", "Target GPU architecture">
  ];
}

def ConvertParallelToGPU2 : Pass<"convert-parallel-to-gpu2"> {
  let summary = "Convert parallel loops to gpu";
  let constructor = "mlir::polygeist::createConvertParallelToGPUPass2()";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "memref::MemRefDialect", "gpu::GPUDialect"];
}

def ConvertToOpaquePtrPass : Pass<"convert-to-opaque-ptr"> {
  let summary = "Convert typed llvm pointers to opaque";
  let constructor = "mlir::polygeist::createConvertToOpaquePtrPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def MergeGPUModulesPass : Pass<"merge-gpu-modules", "mlir::ModuleOp"> {
  let summary = "Merge all gpu modules into one";
  let constructor = "mlir::polygeist::createMergeGPUModulesPass()";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
}

def InnerSerialization : Pass<"inner-serialize"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createInnerSerializationPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
}

def Serialization : Pass<"serialize"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createSerializationPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
}

def SCFBarrierRemovalContinuation : InterfacePass<"barrier-removal-continuation", "FunctionOpInterface"> {
  let summary = "Remove scf.barrier using continuations";
  let constructor = "mlir::polygeist::createBarrierRemovalContinuation()";
  let dependentDialects = ["memref::MemRefDialect", "func::FuncDialect"];
}

def SCFRaiseToAffine : Pass<"raise-scf-to-affine"> {
  let summary = "Raise SCF to affine";
  let constructor = "mlir::polygeist::createRaiseSCFToAffinePass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "scf::SCFDialect",
  ];
}

def RemoveIterArgs : Pass<"remove-iter-args"> {
  let summary = "Remove scf iter args";
  let constructor = "mlir::polygeist::createRemoveIterArgsPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect",
  ];
}

def FoldSCFIf : Pass<"fold-scf-if"> {
  let summary = "Fold simple scf.if regions into arith.select";
  let constructor = "mlir::polygeist::createFoldSCFIfPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
  ];
}

def LowerPolygeistSubmap : Pass<"lower-polygeist-submap"> {
  let summary = "Lower polygeist.submap and polygeist.submapInverse to standard MLIR";
  let constructor = "mlir::polygeist::createLowerPolygeistSubmapPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
    "polygeist::PolygeistDialect",
  ];
}

def LowerKernelLaunch : Pass<"lower-kernel-launch", "::mlir::ModuleOp"> {
  let summary = "Inline kernel.defn bodies in place of kernel.launch ops";
  let description = [{
    For each `kernel.launch @<name>(operands)` op, finds the `kernel.defn
    @<name>` symbol (either in the same module or in a separately-loaded
    library file, controlled by the `kernel-library-path` option), clones the
    defn's body into the launch's parent block with block-arg-to-operand
    substitution, and erases the launch. The defn body's terminating
    `kernel.yield` is replaced by remapping the launch's result SSA to the
    yielded value.

    Phase-2 of the kernel-match pipeline. Replaces the Phase-1 comment-marker
    roundtrip lowering with a real canonical-implementation substitution, so
    a wrongly-labeled kernel.launch produces different numerics from the
    user's original code and fails e2e correctness diffs.
  }];
  let constructor = "mlir::polygeist::createLowerKernelLaunchPass()";
  let options = [
    Option<"kernelLibraryPath", "kernel-library-path", "std::string",
           /*default=*/"\"\"",
           "Optional path to an MLIR file with `kernel.defn` entries. When "
           "set, defns are loaded from the file and looked up by symbol "
           "name. When unset, defns are expected in the input module.">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "math::MathDialect",
    "polygeist::kernel::KernelDialect",
  ];
}

def ComposeCutensornetNetworks
    : Pass<"compose-cutensornet-networks", "::mlir::ModuleOp"> {
  let summary = "Compose connected contraction stages into tensor networks";
  let description = [{
    Starting from an additive linalg contraction, follows tensor SSA
    provenance through rank-independent cuTensorNet contraction labels and
    multiplicative pointwise generics. If the complete region is a legal
    sum-of-products network with no escaping intermediates, replaces it with
    one variable-arity cutensornetNetwork launch. This pass is semantic and
    runs before bufferization; the launch remains bufferizable and is lowered
    to the pointer ABI only by --lower-kernel-launch-to-cublas.
  }];
  let constructor = "mlir::polygeist::createComposeCutensornetNetworksPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "polygeist::kernel::KernelDialect",
  ];
}

def WrapKernelLaunchPipeline
    : Pass<"wrap-kernel-launch-pipeline", "::mlir::ModuleOp"> {
  let summary =
      "Insert runtime pipeline begin/end calls around matched kernel dispatches";
  let description = [{
    Inserts calls to a runtime pipeline-scope ABI around functions that contain
    matched kernel dispatches. This gives the runtime one explicit scope for a
    sequence of lowered library calls, so backend implementations can keep
    host mappings, temporary device allocations, streams, descriptors, and
    future device-resident buffers alive across adjacent calls.

    By default the pass targets the cuBLAS/cuDNN CUDA runtime ABI:

      * `polygeist_cublas_pipeline_begin`
      * `polygeist_cublas_pipeline_end`

    The pass recognizes both pre-lowering `kernel.launch` ops and post-lowering
    runtime `func.call`s with Polygeist CUDA/cuBLAS/cuDNN shim prefixes. Running
    it after `--lower-kernel-launch-to-cublas` is the safest execution path
    because the pass then only scopes launches that actually lowered to CUDA
    runtime calls.

    With `cuda-graphs=true`, consecutive result-free calls explicitly marked
    `polygeist.cuda_graph_safe` are additionally guarded by a stable graph id.
    The runtime can warm the sequence, capture it once, and skip the call body
    on later cached replays. Unannotated calls retain the ordinary pipeline
    scope and are never speculatively captured.
  }];
  let constructor =
      "mlir::polygeist::createWrapKernelLaunchPipelinePass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "scf::SCFDialect",
    "polygeist::kernel::KernelDialect",
  ];
  let options = [
    Option<"beginSymbol", "begin-symbol", "std::string",
           /*default=*/"\"polygeist_cublas_pipeline_begin\"",
           "Runtime function called at pipeline-scope entry.">,
    Option<"endSymbol", "end-symbol", "std::string",
           /*default=*/"\"polygeist_cublas_pipeline_end\"",
           "Runtime function called at pipeline-scope exit.">,
    Option<"useCudaGraphs", "cuda-graphs", "bool",
           /*default=*/"false",
           "Wrap capture-safe, result-free CUDA shim sequences in a cached "
           "CUDA Graph replay guard.">,
    Option<"graphBeginSymbol", "graph-begin-symbol", "std::string",
           /*default=*/"\"polygeist_cuda_graph_begin\"",
           "Runtime i32(i64) function that warms, captures, or replays a graph.">,
    Option<"graphEndSymbol", "graph-end-symbol", "std::string",
           /*default=*/"\"polygeist_cuda_graph_end\"",
           "Runtime void(i64) function that completes warmup or capture.">,
    Option<"captureHostMappedCutensornet", "capture-host-mapped-cutensornet",
           "bool", /*default=*/"false",
           "Treat the host-mapped generic cuTensorNet contraction shim as "
           "graph-safe under the stable-pointer CUDA Graph contract.">,
    Option<"maximalDeviceSequence", "maximal-device-sequence", "bool",
           /*default=*/"false",
           "Capture one maximal device sequence, including capture-invariant "
           "host metadata construction between device operations.">
  ];
}

def PlanPersistentGpuWorkspace
    : Pass<"plan-persistent-gpu-workspace", "::mlir::ModuleOp"> {
  let summary =
      "Materialize static tensor/memref scratch as persistent workspace";
  let description = [{
    Replaces static tensor.empty/bufferization.alloc_tensor and memref.alloc
    scratch roots in the selected function with distinct private memref.global
    buffers. This is
    an explicitly selected, single-instance workspace policy for repeated GPU
    execution: addresses remain stable across CUDA Graph warmup, capture, and
    replay. Dynamic scratch and copied alloc_tensor operations are left
    unchanged; recursive functions and missing explicit selections are
    rejected.

    The global policy is intentionally opt-in because simultaneous invocations
    of the same function would alias the workspace. A future caller-owned
    workspace ABI can reuse the same scratch analysis while providing
    re-entrancy.
  }];
  let constructor =
      "mlir::polygeist::createPlanPersistentGpuWorkspacePass()";
  let dependentDialects = [
    "bufferization::BufferizationDialect",
    "func::FuncDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
  ];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Exact function name whose static scratch may be made persistent.">,
  ];
}

def PrepareGpuResidualPipeline
    : Pass<"prepare-gpu-residual-pipeline", "::mlir::ModuleOp"> {
  let summary =
      "Prepare bufferized residual tensor work for GPU outlining and capture";
  let description = [{
    This opt-in bridge is intended to run twice around the standard MLIR
    linalg-to-GPU outlining pipeline. Before outlining it rewrites memref.copy
    operations in the selected function to linalg.copy, allowing copies to be
    represented by generated device kernels. After outlining it marks
    gpu.launch_func operations as CUDA-Graph-safe and registers every host
    memref argument and persistent memref.global workspace before the first
    launch. The runtime keeps those mapped registrations alive across graph
    warmup, capture, and replay.
  }];
  let constructor =
      "mlir::polygeist::createPrepareGpuResidualPipelinePass()";
  let dependentDialects = [
    "func::FuncDialect",
    "gpu::GPUDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
  ];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Exact function whose residual work is prepared.">,
  ];
}

def LowerKernelLaunchToCuBLAS
    : Pass<"lower-kernel-launch-to-cublas", "::mlir::ModuleOp"> {
  let summary = "Lower kernel.launch ops to runtime-shim func.calls (cuBLAS ABI)";
  let description = [{
    Phase-2 *ABI* lowering for the kernel-matcher pipeline. For each
    recognised `kernel.launch @<libsym>(operands)` op, replaces the launch
    with a `func.call` to a runtime-shim ABI function declared in
    `runtime/polygeist_cublas_rt.h`. Linking the shim object file (CPU
    stub for validation, cuBLAS-backed for hardware) produces an executable.

    Distinct from `--lower-kernel-launch`, which inlines a canonical
    `linalg.generic` body for the library symbol and stays in MLIR-land.
    Use this pass instead when you want the matched op to dispatch to
    an actual library implementation at runtime.

    Currently supports:
      * `@cublasDgemm` → `polygeist_cublas_dgemm`

    Expected input: `kernel.launch` ops in TENSOR form (the matcher's
    default output). The pass synthesises `bufferization.to_memref` /
    `bufferization.to_tensor` ops around the call.
  }];
  let constructor = "mlir::polygeist::createLowerKernelLaunchToCuBLASPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
    "polygeist::kernel::KernelDialect",
  ];
  let options = [
    Option<"deviceResidentCutensornet", "device-resident-cutensornet",
           "bool", /*default=*/"false",
           "Lower generic cuTensorNet contractions through the CUDA-device "
           "pointer ABI. Fails if residual host tensor computation remains.">
  ];
}

def LowerKernelLaunchToPVA
    : Pass<"lower-kernel-launch-to-pva", "::mlir::ModuleOp"> {
  let summary = "Lower kernel.launch ops to PVA Solutions runtime-shim func.calls";
  let description = [{
    Phase-2 ABI lowering for kernels routed to NVIDIA PVA Solutions
    (libpva_operator on Jetson Orin's Programmable Vision Accelerator).
    Currently handles `@cudnnConvolution2D_9tap_i{8,16}` → `func.call
    @polygeist_pva_conv2d_3x3_i{8,16}`, the runtime-shim entry point for
    PVA's single-channel integer Conv2d operator.

    Distinct from `--lower-kernel-launch-to-cublas` because PVA is a
    separate backend with its own vendor library, host-side staging
    contract (cuPVA-mapped memory, not cudaMemcpy), and hardware
    semantics (Q-format quantized filter with REPLICATE border, not
    raw integer multiply-accumulate). The two passes handle disjoint
    launch symbol sets and can run in either order.
  }];
  let constructor = "mlir::polygeist::createLowerKernelLaunchToPVAPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "polygeist::kernel::KernelDialect",
  ];
}

def LinalgDebufferize : Pass<"linalg-debufferize"> {
  let summary = "Raise affine to linalg";
  let constructor = "mlir::polygeist::createLinalgDebufferizePass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "bufferization::BufferizationDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
    "polygeist::PolygeistDialect",
  ];
  let options = [
    Option<"useRecursive", "use-recursive", "bool", /*default=*/"true",
           "Use the region-recursive (v2) debufferization implementation. "
           "Set to false to fall back to the legacy v1 pattern.">,
    Option<"useMultiRoot", "use-multi-root", "bool", /*default=*/"false",
           "Force the multi-root walker that processes ALL memref "
           "args of a function jointly. Handles double-buffer stencils, trmm, "
           "symm, etc. where a single linalg.generic touches operands from "
           "multiple memref roots. Cross-root generics select it automatically "
           "when use-recursive is enabled; this option forces it. Overrides "
           "useRecursive when set.">
  ];
}

def AffineRaiseToLinalg : Pass<"raise-affine-to-linalg"> {
  let summary = "Raise affine to linalg";
  let constructor = "mlir::polygeist::createRaiseAffineToLinalgPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "polygeist::PolygeistDialect",
  ];
}

def AffineRaiseToLinalgPipeline : Pass<"raise-affine-to-linalg-pipeline"> {
  let summary = "Pipeline: fold-scf-if, affine-parallelize, raise-affine-to-linalg";
  let constructor = "mlir::polygeist::createRaiseAffineToLinalgPipelinePass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "polygeist::PolygeistDialect",
  ];
}

def SCFCanonicalizeFor : Pass<"canonicalize-scf-for"> {
  let summary = "Run some additional canonicalization for scf::for";
  let constructor = "mlir::polygeist::createCanonicalizeForPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "math::MathDialect",
  ];
}

def ForBreakToWhile : Pass<"for-break-to-while"> {
  let summary = "Rewrite scf.for(scf.if) to scf.while";
  let constructor = "mlir::polygeist::createForBreakToWhilePass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
  ];
}

def ParallelLICM : Pass<"parallel-licm"> {
  let summary = "Perform LICM on known parallel (and serial) loops";
  let constructor = "mlir::polygeist::createParallelLICMPass()";
}

def OpenMPOptPass : Pass<"openmp-opt"> {
  let summary = "Optimize OpenMP";
  let constructor = "mlir::polygeist::createOpenMPOptPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "omp::OpenMPDialect",
    "LLVM::LLVMDialect",
  ];
}

def PolygeistCanonicalize : Pass<"canonicalize-polygeist"> {
  let constructor = "mlir::polygeist::createPolygeistCanonicalizePass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "gpu::GPUDialect",
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "scf::SCFDialect",
    "polygeist::PolygeistDialect",
  ];
  let options = [
    Option<"topDownProcessingEnabled", "top-down", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"enableRegionSimplification", "region-simplify", "bool",
           /*default=*/"true",
           "Perform control flow optimizations to the region tree">,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Max. iterations between applying patterns / simplifying regions">,
    Option<"maxNumRewrites", "max-num-rewrites", "int64_t", /*default=*/"-1",
           "Max. number of pattern rewrites within an iteration">,
    Option<"testConvergence", "test-convergence", "bool", /*default=*/"false",
           "Test only: Fail pass on non-convergence to detect cyclic pattern">
  ] # RewritePassUtils.options;
}

def LoopRestructure : Pass<"loop-restructure"> {
  let constructor = "mlir::polygeist::createLoopRestructurePass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "polygeist::PolygeistDialect",
  ];
}

def RemoveTrivialUse : Pass<"trivialuse"> {
  let constructor = "mlir::polygeist::createRemoveTrivialUsePass()";
}

def LinalgToKernel : Pass<"linalg-to-kernel", "mlir::ModuleOp"> {
  let summary = "Convert linalg.generic operations to kernel operations by matching with kernel.defn patterns";
  let description = [{
    This pass matches linalg.generic operations against patterns defined in 
    kernel.defn_collection operations and converts them to the corresponding 
    specialized kernel operations (e.g., kernel.gemm, kernel.batched_gemm).
    
    The pass performs semantic matching of linalg.generic operations by:
    - Comparing indexing maps and iterator types
    - Matching the operation structure within regions
    - Checking input/output operand counts
    
    Example transformation:
    ```mlir
    // Input: linalg.generic performing matrix multiplication
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %mul, %c : f32
      linalg.yield %add : f32
    } -> tensor<?x?xf32>
    
    // Output: Specialized kernel operation
    %result = kernel.gemm %C, %A, %B, %alpha, %beta : tensor<?x?xf32>
    ```
  }];
  let constructor = "mlir::polygeist::createLinalgToKernelPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "polygeist::kernel::KernelDialect",
    "tensor::TensorDialect",
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
  ];
  let options = [
    Option<"kernelLibraryPath", "kernel-library-path", "std::string", 
           /*default=*/"\"\"", 
           "Path to external MLIR file containing kernel.defn_collection definitions. "
           "If empty, looks for kernel.defn_collection in the input module.">
  ];
}

def ConvertPolygeistToLLVM : Pass<"convert-polygeist-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert scalar and vector operations from the Standard to the "
                "LLVM dialect";
  let description = [{
    Convert standard operations into the LLVM IR dialect operations.

    #### Input invariant

    -   operations including: arithmetic on integers and floats, constants,
        direct calls, returns and branches;
    -   no `tensor` types;
    -   all `vector` are one-dimensional;
    -   all blocks are reachable by following the successors of the first basic
        block;

    If other operations are present and their results are required by the LLVM
    IR dialect operations, the pass will fail.  Any LLVM IR operations or types
    already present in the IR will be kept as is.

    #### Output IR

    Functions converted to LLVM IR. Function arguments types are converted
    one-to-one. Function results are converted one-to-one and, in case more than
    1 value is returned, packed into an LLVM IR struct type. Function calls and
    returns are updated accordingly. Block argument types are updated to use
    LLVM IR types.
  }];
  let constructor = "mlir::polygeist::createConvertPolygeistToLLVMPass()";
  let dependentDialects = [
    "polygeist::PolygeistDialect",
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "gpu::GPUDialect",
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "scf::SCFDialect",
  ];
  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">,
    Option<"useCStyleMemRef", "use-c-style-memref", "bool",
           /*default=*/"true",
           "Use C-style nested-array lowering of memref instead of "
           "the default MLIR descriptor structure">
  ];
}

#endif // POLYGEIST_PASSES

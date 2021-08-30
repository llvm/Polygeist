#ifndef POlYGEIST_PASSES
#define POLYGEIST_PASSES

include "mlir/Pass/PassBase.td"

def AffineCFG : FunctionPass<"affine-cfg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::replaceAffineCFGPass()";
}

def Mem2Reg : FunctionPass<"mem2reg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::replaceAffineCFGPass()";
}

def ParallelLower : FunctionPass<"parallel-lower"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::createParallelLowerPass()";
}

def AffineReduction : FunctionPass<"detect-reduction"> {
  let summary = "Detect reductions in affine.for";
  let constructor = "mlir::detectReductionPass()";
}

def SCFCPUify : FunctionPass<"cpuify"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::createCPUifyPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "StandardOpsDialect", "LLVM::LLVMDialect"];
}

def SCFBarrierRemovalContinuation : Pass<"barrier-removal-continuation"> {
  let summary = "Remove scf.barrier using continuations";
  let constructor = "mlir::createBarrierRemovalContinuation()";
  let dependentDialects = ["memref::MemRefDialect", "StandardOpsDialect"];
}

def SCFRaiseToAffine : FunctionPass<"raise-scf-to-affine"> {
  let summary = "Raise SCF to affine";
  let constructor = "mlir::createRaiseSCFToAffinePass()";
  let dependentDialects = ["AffineDialect"];
}

def SCFCanonicalizeFor : FunctionPass<"canonicalize-scf-for"> {
  let summary = "Run some additional canonicalization for scf::for";
  let constructor = "mlir::createCanonicalizeForPass()";
}

def LoopRestructure : FunctionPass<"loop-restructure"> {
  let constructor = "mlir::createLoopRestructurePass()";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
}

def RemoveTrivialUse : FunctionPass<"trivialuse"> {
  let constructor = "mlir::createRemoveTrivialUse()";
}

def ConvertPolygeistToLLVM : Pass<"convert-polygeist-to-llvm", "ModuleOp"> {
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
  let constructor = "mlir::createConvertPolygeistToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"emitCWrappers", "emit-c-wrappers", "bool", /*default=*/"false",
           "Emit wrappers for C-compatible pointer-to-struct memref "
           "descriptors">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">
  ];
}

#endif // POLYGEIST_PASSES

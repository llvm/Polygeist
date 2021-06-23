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
  let dependentDialects = ["memref::MemRefDialect", "StandardOpsDialect",
                           "LLVM::LLVMDialect"];
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

#endif // POLYGEIST_PASSES
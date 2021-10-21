//===- polymer-opt.cc - The polymer optimisation tool -----------*- C++ -*-===//
//
// This file implements the polymer optimisation tool, which is the polymer
// analog of mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/ExtractScopStmt.h"
#include "polymer/Transforms/FoldSCFIf.h"
#include "polymer/Transforms/LoopAnnotate.h"
#include "polymer/Transforms/LoopExtract.h"
#include "polymer/Transforms/Passes.h"
#include "polymer/Transforms/PlutoTransform.h"
#include "polymer/Transforms/Reg2Mem.h"
#include "polymer/Transforms/ScopStmtOpt.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace polymer;

int main(int argc, char *argv[]) {
  DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<StandardOpsDialect>();
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::arith::ArithmeticDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

// Register the standard passes we want.
#include "mlir/Transforms/Passes.h.inc"
  registerCanonicalizerPass();
  registerCSEPass();
  registerInlinerPass();
  // Register polymer specific passes.
  registerPlutoTransformPass();
  registerRegToMemPass();
  registerExtractScopStmtPass();
  registerScopStmtOptPasses();
  registerLoopAnnotatePasses();
  registerLoopExtractPasses();
  registerFoldSCFIfPass();
  registerAnnotateScopPass();

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  // Register printer command line options.
  registerAsmPrinterCLOptions();

  return failed(MlirOptMain(argc, argv, "Polymer optimizer driver", registry,
                            /*preloadDialectsInContext=*/true));
}

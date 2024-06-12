
#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/ISL.h"
#include "polymer/Target/OpenScop.h"
#include "polymer/Transforms/ExtractScopStmt.h"
#include "polymer/Transforms/PlutoTransform.h"

#include "pluto/internal/pluto.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "isl/id.h"
#include "isl/id_to_id.h"
#include "isl/schedule.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

#define DEBUG_TYPE "pluto-opt"

namespace polymer {
mlir::func::FuncOp tadashiTransform(mlir::func::FuncOp f, OpBuilder &rewriter) {
  LLVM_DEBUG(dbgs() << "Pluto transforming: \n");
  LLVM_DEBUG(f.dump());

  ModuleOp m = f->getParentOfType<ModuleOp>();

  PolymerSymbolTable srcTable, dstTable;
  std::unique_ptr<IslScop> scop = createIslFromFuncOp(f, srcTable);
  scop->dumpTadashi(llvm::outs());

  mlir::func::FuncOp g = cast<mlir::func::FuncOp>(
      createFuncOpFromIsl(std::move(scop), m, dstTable, rewriter.getContext()));

  return nullptr;
}
} // namespace polymer


#include "polymer/Support/IslScop.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/ISL.h"
#include "polymer/Transforms/ExtractScopStmt.h"
#include "polymer/Transforms/PlutoTransform.h"

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
using namespace polymer;

using llvm::dbgs;

#define DEBUG_TYPE "tadashi-opt"

namespace polymer {
mlir::func::FuncOp tadashiTransform(mlir::func::FuncOp f, OpBuilder &rewriter) {
  LLVM_DEBUG(dbgs() << "Tadashi transforming: \n");
  LLVM_DEBUG(f.dump());

  ModuleOp m = f->getParentOfType<ModuleOp>();

  std::unique_ptr<IslScop> scop = createIslFromFuncOp(f);
  scop->dumpSchedule(llvm::outs());
  scop->dumpAccesses(llvm::outs());

  isl_schedule *newSchedule = scop->getSchedule();
  mlir::func::FuncOp g = cast<mlir::func::FuncOp>(
      createFuncOpFromIsl(std::move(scop), f, newSchedule));

  return g;
}
mlir::func::FuncOp plutoTransform(mlir::func::FuncOp f,
                                  mlir::OpBuilder &rewriter,
                                  std::string dumpClastAfterPluto,
                                  bool parallelize = false, bool debug = false,
                                  int cloogf = -1, int cloogl = -1,
                                  bool diamondTiling = false) {
  llvm_unreachable("not compiled with pluto support");
}
void registerPlutoTransformPass() {}
} // namespace polymer

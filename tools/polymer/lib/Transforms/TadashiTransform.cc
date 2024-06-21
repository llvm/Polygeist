
#include "mlir/IR/Verifier.h"
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
#include "isl/schedule_node.h"
#include "isl/val.h"

using namespace mlir;
using namespace polymer;

using llvm::dbgs;

#define DEBUG_TYPE "tadashi-opt"

static isl_schedule_node *tadashi_tile(isl_schedule_node *node, int tile_size) {
  isl_ctx *ctx = isl_schedule_node_get_ctx(node);
  return isl_schedule_node_band_tile(
      node, isl_multi_val_from_val_list(
                isl_schedule_node_band_get_space(node),
                isl_val_list_from_val(isl_val_int_from_si(ctx, tile_size))));
}

namespace polymer {

mlir::func::FuncOp tadashiTransform(mlir::func::FuncOp f, OpBuilder &rewriter) {
  LLVM_DEBUG(dbgs() << "Tadashi transforming: \n");
  LLVM_DEBUG(f.dump());

  std::unique_ptr<IslScop> scop = createIslFromFuncOp(f);
  scop->dumpSchedule(llvm::outs());
  scop->dumpAccesses(llvm::outs());

  isl_schedule *newSchedule = isl_schedule_copy(scop->getSchedule());
  isl_schedule_node *node = isl_schedule_get_root(newSchedule);
  node = isl_schedule_node_first_child(node);
  tadashi_tile(node, 10);

  newSchedule = isl_schedule_node_get_schedule(node);

  isl_schedule_dump(newSchedule);

  mlir::func::FuncOp g = cast<mlir::func::FuncOp>(
      createFuncOpFromIsl(std::move(scop), f, newSchedule));

  assert(mlir::verify(g).succeeded());

  if (g) {
    SmallVector<DictionaryAttr> argAttrs;
    f.getAllArgAttrs(argAttrs);
    g.setAllArgAttrs(argAttrs);
  }

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

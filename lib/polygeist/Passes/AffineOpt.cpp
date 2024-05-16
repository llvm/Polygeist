#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/Passes.h"
#include <utility>
#ifdef POLYGEIST_ENABLE_POLYMER

#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include <deque>

#include "polymer/Transforms/ExtractScopStmt.h"
#include "polymer/Transforms/PlutoTransform.h"
#include "polymer/Transforms/Reg2Mem.h"

#include "AlwaysInliner.h"

#define DEBUG_TYPE "affine-opt"
#define DEBUG_LABEL DEBUG_TYPE ": "

using namespace mlir;
using namespace polygeist;

namespace polymer {
mlir::func::FuncOp plutoTransform(mlir::func::FuncOp f, OpBuilder &rewriter,
                                  std::string dumpClastAfterPluto,
                                  bool parallelize = false, bool debug = false,
                                  int cloogf = -1, int cloogl = -1,
                                  bool diamondTiling = false);
std::unique_ptr<mlir::Pass> createDedupIndexCastPass();
std::unique_ptr<mlir::Pass> createPlutoParallelizePass();
} // namespace polymer

namespace {

SmallVector<Operation *> findAffineRegions(Operation *root) {
  SmallVector<Operation *> affineRegions;
  root->walk<mlir::WalkOrder::PreOrder>([&](Operation *loop) {
    if (!(isa<affine::AffineForOp>(loop) ||
          isa<affine::AffineParallelOp>(loop)))
      return;
    if (!affineRegions.empty() && affineRegions.back()->isAncestor(loop))
      return;

    auto result = loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op)) {
        Operation *cur = op->getParentOp();
        while (cur != loop) {
          if (!isa<mlir::affine::AffineDialect>(cur->getDialect()))
            return WalkResult::interrupt();
          cur = cur->getParentOp();
        }
        return WalkResult::advance();
      }
      if (isa<mlir::affine::AffineDialect>(op->getDialect()))
        return WalkResult::advance();
      if (isReadNone(op))
        return WalkResult::advance();

      return WalkResult::interrupt();
    });
    if (!result.wasInterrupted()) {
      LLVM_DEBUG(llvm::dbgs() << DEBUG_LABEL << "Found affine region\n"
                              << *loop << "\n");
      affineRegions.push_back(loop);
    }
  });
  return affineRegions;
}

FailureOr<func::FuncOp> outlineOp(RewriterBase &rewriter, Location loc,
                                  Operation *op, StringRef funcName,
                                  func::CallOp *callOp) {
  assert(!funcName.empty() && "funcName cannot be empty");

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  auto executeOp = rewriter.create<scf::ExecuteRegionOp>(loc, TypeRange());
  auto executeOpBlock = rewriter.createBlock(&executeOp.getRegion());
  rewriter.clone(*op);
  rewriter.create<scf::YieldOp>(loc);
  auto ret = outlineSingleBlockRegion(rewriter, loc, executeOp.getRegion(),
                                      funcName, callOp);
  if (failed(ret)) {
    rewriter.eraseOp(executeOp);
    return ret;
  }
  rewriter.eraseOp(executeOp.getRegion().front().getTerminator());
  rewriter.inlineBlockBefore(executeOpBlock, op);
  rewriter.eraseOp(executeOp);
  rewriter.eraseOp(op);
  return ret;
}

SmallVector<std::pair<func::FuncOp, func::CallOp>>
outlineAffineRegions(Operation *root) {
  auto affineRegions = findAffineRegions(root);
  auto m = isa<ModuleOp>(root) ? cast<ModuleOp>(root)
                               : root->getParentOfType<ModuleOp>();
  auto loc = root->getLoc();

  unsigned Idx = 0;
  auto getName = [&]() {
    std::string name;
    do {
      name = "__polygeist_outlined_affine_" + std::to_string(Idx++);
    } while (m.lookupSymbol(name));
    return name;
  };
  SmallVector<std::pair<func::FuncOp, func::CallOp>> funcs;
  IRRewriter rewriter(root->getContext());
  for (Operation *op : affineRegions) {
    func::CallOp callOp;
    auto ret = outlineOp(rewriter, loc, op, getName(), &callOp);
    if (failed(ret)) {
      llvm::errs() << "Outlining affine region failed\n" << *root;
      abort();
    } else {
      funcs.push_back(std::make_pair(*ret, callOp));
    }
  }
  return funcs;
}

struct AffineOptPass : public AffineOptBase<AffineOptPass> {
  polymer::PlutoOptPipelineOptions &pipelineOptions;
  AffineOptPass(polymer::PlutoOptPipelineOptions &plutoOpts)
      : pipelineOptions(plutoOpts) {}
  void runOnOperation() override;
};
} // namespace

void AffineOptPass::runOnOperation() {
  Operation *m = getOperation();
  auto &context = *m->getContext();
  GreedyRewriteConfig canonicalizerConfig;

  auto funcOps = outlineAffineRegions(m);
  mlir::PassManager preTransformPm(&context);

  preTransformPm.addPass(polymer::createRegToMemPass());
  preTransformPm.addPass(polymer::createExtractScopStmtPass());
  preTransformPm.addPass(mlir::polygeist::createPolygeistCanonicalizePass(
      canonicalizerConfig, {}, {}));
  preTransformPm.addNestedPass<func::FuncOp>(
      polymer::createDedupIndexCastPass());
  preTransformPm.addPass(createCanonicalizerPass());

  mlir::PassManager postTransformPm(&context);
  postTransformPm.addPass(createCanonicalizerPass());
  if (pipelineOptions.generateParallel) {
    postTransformPm.addNestedPass<func::FuncOp>(
        polymer::createPlutoParallelizePass());
    postTransformPm.addPass(createCanonicalizerPass());
  }

  AlwaysInlinerInterface interface(&getContext());
  IRRewriter b(&context);
  for (auto &pair : funcOps) {
    mlir::func::FuncOp f = pair.first;
    mlir::func::CallOp call = pair.second;
    if (failed(preTransformPm.run(f))) {
      signalPassFailure();
      return;
    }
    func::FuncOp postOp = f;
    if (mlir::func::FuncOp g = polymer::plutoTransform(
            f, b, pipelineOptions.dumpClastAfterPluto,
            pipelineOptions.parallelize, pipelineOptions.debug,
            pipelineOptions.cloogf, pipelineOptions.cloogl,
            pipelineOptions.diamondTiling)) {
      g.setPublic();
      g->setAttrs(f->getAttrs());

      g.setName(f.getName());
      f.erase();

      postOp = g;
    }
    if (failed(postTransformPm.run(postOp))) {
      signalPassFailure();
      return;
    }
    alwaysInlineCall(call);
  }
}

std::unique_ptr<Pass> mlir::polygeist::createAffineOptPass(
    polymer::PlutoOptPipelineOptions &plutoOpts) {
  return std::make_unique<AffineOptPass>(plutoOpts);
}
std::unique_ptr<Pass> mlir::polygeist::createAffineOptPass() {
  return std::make_unique<AffineOptPass>(
      *polymer::PlutoOptPipelineOptions::createFromString(""));
}

#else

std::unique_ptr<Pass> mlir::polygeist::createAffineOptPass() {
  abort();
  return nullptr;
}

#endif

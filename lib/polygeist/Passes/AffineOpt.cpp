#ifdef POLYGEIST_ENABLE_POLYMER

#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/Passes.h"
#include <utility>

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

#include "polymer/Support/PolymerUtils.h"

#include "AlwaysInliner.h"

#define DEBUG_TYPE "polyhedral-opt"
#define DEBUG_LABEL DEBUG_TYPE ": "

using namespace mlir;
using namespace polygeist;

namespace {

#define POLYGEIST_OUTLINED_AFFINE_ATTR "polygeist.outlined_affine"

static SmallVector<Operation *> findAffineRegions(Operation *root) {
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

static FailureOr<func::FuncOp> outlineOp(RewriterBase &rewriter, Location loc,
                                         Operation *op, StringRef funcName,
                                         func::CallOp *callOp) {
  assert(!funcName.empty() && "funcName cannot be empty");

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  auto executeOp = rewriter.create<scf::ExecuteRegionOp>(loc, TypeRange());
  rewriter.createBlock(&executeOp.getRegion());
  rewriter.clone(*op);
  rewriter.create<scf::YieldOp>(loc);
  auto ret = outlineSingleBlockRegion(rewriter, loc, executeOp.getRegion(),
                                      funcName, callOp);
  if (failed(ret)) {
    rewriter.eraseOp(executeOp);
    return ret;
  }
  (*ret)->setAttr(POLYGEIST_OUTLINED_AFFINE_ATTR, rewriter.getUnitAttr());
  rewriter.eraseOp(executeOp.getRegion().front().getTerminator());
  rewriter.inlineBlockBefore(&executeOp.getRegion().front(), op);
  rewriter.eraseOp(executeOp);
  rewriter.eraseOp(op);
  return ret;
}

static SmallVector<std::pair<func::FuncOp, func::CallOp>>
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

static void inlineAll(func::CallOp callOp, ModuleOp m = nullptr) {
  if (!m)
    m = callOp->getParentOfType<ModuleOp>();
  auto name = callOp.getCallee();
  if (auto funcOp = dyn_cast<func::FuncOp>(m.lookupSymbol(name))) {
    if (funcOp->hasAttr(SCOP_STMT_ATTR_NAME) ||
        funcOp->hasAttr(POLYGEIST_OUTLINED_AFFINE_ATTR))
      funcOp->walk(
          [&](func::CallOp nestedCallOp) { inlineAll(nestedCallOp, m); });
    alwaysInlineCall(callOp);
  } else {
    llvm::errs() << "Unexpected call to non-FuncOp\n";
    abort();
  }
}

static void cleanupTempFuncs(ModuleOp m) {
  auto eraseWithAttr = [&](StringRef attr) {
    SmallVector<func::FuncOp> toErase;
    for (auto &op : m.getBodyRegion().front())
      if (op.getAttr(attr))
        toErase.push_back(cast<func::FuncOp>(&op));

    SymbolTable symbolTable(m);
    for (auto func : toErase) {
      assert(func.getSymbolUses(m)->empty());
      symbolTable.erase(func);
    }
  };
  eraseWithAttr(POLYGEIST_OUTLINED_AFFINE_ATTR);
  eraseWithAttr(SCOP_STMT_ATTR_NAME);
}

struct PolyhedralOptPass : public PolyhedralOptBase<PolyhedralOptPass> {
  PolyhedralOptPass(polymer::PlutoOptPipelineOptions *plutoOpts) {}
  void runOnOperation() override;
};
} // namespace

void PolyhedralOptPass::runOnOperation() {
  // TODO use a throw-away module for each function to optimize in so as not to
  // litter scop functions around, then run the inliner in that module at the
  // end
  Operation *op = getOperation();
  ModuleOp m = cast<ModuleOp>(SymbolTable::getNearestSymbolTable(op));
  auto &context = *op->getContext();

  auto funcOps = outlineAffineRegions(op);
  mlir::PassManager preTransformPm(&context);
  preTransformPm.addPass(createCanonicalizerPass());

  AlwaysInlinerInterface interface(&getContext());
  IRRewriter b(&context);
  for (auto &pair : funcOps) {
    mlir::func::FuncOp f = pair.first;
    mlir::func::CallOp call = pair.second;
    // Reg2Mem
    polymer::separateAffineIfBlocks(f, b);
    polymer::demoteLoopReduction(f, b);
    polymer::demoteRegisterToMemory(f, b);
    // Extract scop stmt
    polymer::replaceUsesByStored(f, b);
    polymer::extractScopStmt(f, b);
    if (failed(preTransformPm.run(f))) {
      signalPassFailure();
      return;
    }
    polymer::dedupIndexCast(f);
    if (failed(preTransformPm.run(f))) {
      signalPassFailure();
      return;
    }
    mlir::func::FuncOp g = nullptr;
    if ((g = polymer::plutoTransform(f, b, ""))) {
      g.setPublic();
      g->setAttrs(f->getAttrs());

      g.setName(f.getName());
      f.erase();
    }
    if (g && /*options.parallelize=*/true) {
      polymer::plutoParallelize(g, b);
    }
    inlineAll(call);
  }
  cleanupTempFuncs(m);
}

std::unique_ptr<Pass> mlir::polygeist::createPolyhedralOptPass() {
  return std::make_unique<PolyhedralOptPass>(nullptr);
}

#else

#include "PassDetails.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
class Pass;
}
using namespace mlir;
using namespace polygeist;
namespace {
struct PolyhedralOptPass : public PolyhedralOptBase<PolyhedralOptPass> {
  void runOnOperation() override {}
};
} // namespace
std::unique_ptr<mlir::Pass> mlir::polygeist::createPolyhedralOptPass() {
  return std::make_unique<PolyhedralOptPass>();
}

#endif

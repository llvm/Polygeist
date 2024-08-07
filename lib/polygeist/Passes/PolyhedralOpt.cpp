#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
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

static llvm::cl::opt<std::string>
    UsePolyhedralOptimizerCl("use-polyhedral-optimizer",
                             llvm::cl::init("pluto"),
                             llvm::cl::desc("pluto or islexternal"));

namespace {

#define POLYGEIST_OUTLINED_AFFINE_ATTR "polygeist.outlined_affine"

struct Scop {
  Operation *begin;
  Operation *end;
};

static SmallVector<Scop> findScops(Operation *root) {
  SmallVector<Operation *> affineLoops;
  SmallVector<Scop> scops;
  root->walk<mlir::WalkOrder::PreOrder>([&](Operation *loop) {
    if (!(isa<affine::AffineForOp>(loop) ||
          isa<affine::AffineParallelOp>(loop)))
      return;
    if (!affineLoops.empty() && affineLoops.back()->isAncestor(loop))
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
      affineLoops.push_back(loop);
    }
  });
  size_t size = affineLoops.size();
  for (size_t i = 0; i < size; i++) {
    Operation *loop = affineLoops[i];
    Scop scop = {.begin = loop, .end = loop->getNextNode()};
    while (i + 1 < size && scop.end == affineLoops[i + 1])
      scop.end = affineLoops[++i]->getNextNode();
    scops.push_back(scop);
  }
  return scops;
}

static FailureOr<func::FuncOp> outlineOp(RewriterBase &rewriter, Location loc,
                                         Scop scop, StringRef funcName,
                                         func::CallOp *callOp) {
  assert(!funcName.empty() && "funcName cannot be empty");

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(scop.begin);
  auto executeOp = rewriter.create<scf::ExecuteRegionOp>(loc, TypeRange());
  rewriter.createBlock(&executeOp.getRegion());
  auto cur = scop.begin;
  while (cur != scop.end) {
    rewriter.clone(*cur);
    cur = cur->getNextNode();
  }
  rewriter.create<scf::YieldOp>(loc);
  auto ret = outlineSingleBlockRegion(rewriter, loc, executeOp.getRegion(),
                                      funcName, callOp);
  if (failed(ret)) {
    rewriter.eraseOp(executeOp);
    return ret;
  }
  (*ret)->setAttr(POLYGEIST_OUTLINED_AFFINE_ATTR, rewriter.getUnitAttr());
  rewriter.eraseOp(executeOp.getRegion().front().getTerminator());
  rewriter.inlineBlockBefore(&executeOp.getRegion().front(), scop.begin);
  rewriter.eraseOp(executeOp);
  cur = scop.begin;
  while (cur != scop.end) {
    auto tmp = cur;
    cur = cur->getNextNode();
    rewriter.eraseOp(tmp);
  }
  return ret;
}

static SmallVector<std::pair<func::FuncOp, func::CallOp>>
outlineScops(Operation *root) {
  auto scops = findScops(root);
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
  for (Scop scop : scops) {
    func::CallOp callOp;
    auto ret = outlineOp(rewriter, loc, scop, getName(), &callOp);
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

  auto funcOps = outlineScops(op);
  mlir::PassManager preTransformPm(&context);
  preTransformPm.addPass(createCanonicalizerPass());

  SmallVector<func::CallOp> toInline;

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
    if (UsePolyhedralOptimizerCl == "islexternal") {
      g = polymer::islexternalTransform(f, b);
    } else if (UsePolyhedralOptimizerCl == "pluto") {
      g = polymer::plutoTransform(f, b, "");
    }
    if (g) {
      g.setPublic();
      g->setAttrs(f->getAttrs());

      g.setName(f.getName());
      f.erase();
    }
    // if (g && /*options.parallelize=*/true) {
    //   polymer::plutoParallelize(g, b);
    // }
    toInline.push_back(call);
  }

  mlir::PassManager lowerAffine(&context);
  lowerAffine.addPass(createLowerAffinePass());
  lowerAffine.addPass(createCanonicalizerPass());

  // Conversion from ISL emits scf so we need to lower the statements before
  // inlining them
  if (UsePolyhedralOptimizerCl == "islexternal" && failed(lowerAffine.run(m))) {
    signalPassFailure();
    return;
  }

  for (auto call : toInline)
    inlineAll(call);
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

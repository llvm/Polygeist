#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace mlir::scf;

namespace {
struct CanonicalizeFor : public SCFCanonicalizeForBase<CanonicalizeFor> {
  void runOnFunction() override;
};
} // namespace

static void propagateInLoopBody(scf::ForOp forOp) {

  if (!forOp.hasIterOperands())
    return;

  Block &block = forOp.getRegion().front();
  auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

  for (auto it : llvm::zip(forOp.getIterOperands(), forOp.getRegionIterArgs(),
                           yieldOp.getOperands())) {
    Value iterOperand = std::get<0>(it);
    Value regionArg = std::get<1>(it);
    Value yieldOperand = std::get<2>(it);

    Operation *op = iterOperand.getDefiningOp();
    if (op && (op->getNumResults() == 1) && (iterOperand == yieldOperand))
      regionArg.replaceAllUsesWith(op->getResult(0));
  }
}

static bool hasSameInitValue(Value iter, scf::ForOp forOp) {
  Operation *cst = iter.getDefiningOp();
  if (!cst)
    return false;
  if (auto cstOp = dyn_cast<ConstantOp>(cst)) {
    Attribute attr = cstOp.getValue();
    if (auto intAttr = attr.cast<IntegerAttr>()) {
      Operation *lbDefOp = forOp.lowerBound().getDefiningOp();
      if (!lbDefOp)
        return false;
      ConstantIndexOp lb = dyn_cast_or_null<ConstantIndexOp>(lbDefOp);
      if (lb && lb.getValue() == intAttr.getInt())
        return true;
    }
  }
  return false;
}

static bool hasSameStepValue(Value regIter, Value yieldOp, scf::ForOp forOp) {
  auto addOp = cast<AddIOp>(yieldOp.getDefiningOp());
  Value addStep = addOp.getOperand(1);
  Operation *defOpStep = addStep.getDefiningOp();
  if (!defOpStep)
    return false;
  if (auto cstStep = dyn_cast<ConstantOp>(defOpStep)) {
    Attribute attr = cstStep.getValue();
    if (auto intAttr = attr.cast<IntegerAttr>()) {
      Operation *stepForDefOp = forOp.step().getDefiningOp();
      if (!stepForDefOp)
        return false;
      ConstantIndexOp stepFor = dyn_cast_or_null<ConstantIndexOp>(stepForDefOp);
      if (stepFor && stepFor.getValue() == intAttr.getInt())
        return true;
    }
  }
  return false;
}

static bool preconditionIndVar(Value regIter, Value yieldOp, scf::ForOp forOp) {
  Operation *mustBeAdd = yieldOp.getDefiningOp();
  if (!mustBeAdd || !isa<AddIOp>(mustBeAdd))
    return false;
  auto addOp = cast<AddIOp>(mustBeAdd);
  if (addOp.getOperand(0) != regIter)
    return false;
  // check users. We allow only index cast and 'addOp`.
  for (auto u : regIter.getUsers()) {
    if (isa<IndexCastOp>(u) || (u == addOp.getOperation()))
      continue;
    return false;
  }
  // the user of the add should be a yieldop.
  Value res = addOp.getResult();
  for (auto u : res.getUsers())
    if (!isa<scf::YieldOp>(u))
      return false;

  return true;
}

static bool isIndVar(Value iter, Value regIter, Value yieldOp,
                     scf::ForOp forOp) {
  if (!preconditionIndVar(regIter, yieldOp, forOp))
    return false;
  if (!hasSameInitValue(iter, forOp))
    return false;
  if (!hasSameStepValue(regIter, yieldOp, forOp))
    return false;
  return true;
}

static void detectTrivialIndVarInArgs(scf::ForOp forOp) {

  if (!forOp.getNumIterOperands())
    return;

  Block &block = forOp.region().front();
  auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

  for (auto it : llvm::zip(forOp.getIterOperands(), forOp.getRegionIterArgs(),
                           yieldOp.getOperands())) {
    if (isIndVar(std::get<0>(it), std::get<1>(it), std::get<2>(it), forOp)) {
      OpBuilder builder(forOp);
      builder.setInsertionPointToStart(forOp.getBody());
      auto indexCast = builder.create<IndexCastOp>(
          forOp.getLoc(), forOp.getInductionVar(), builder.getI32Type());
      std::get<1>(it).replaceAllUsesWith(indexCast);
    }
  }
}

struct ForOpInductionReplacement : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    bool canonicalize = false;
    Block &block = forOp.region().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

    // An internal flat vector of block transfer
    // arguments `newBlockTransferArgs` keeps the 1-1 mapping of original to
    // transformed block argument mappings. This plays the role of a
    // BlockAndValueMapping for the particular use case of calling into
    // `mergeBlockBefore`.
    SmallVector<bool, 4> keepMask;
    keepMask.reserve(yieldOp.getNumOperands());
    SmallVector<Value, 4> newBlockTransferArgs, newIterArgs, newYieldValues,
        newResultValues;
    newBlockTransferArgs.reserve(1 + forOp.getNumIterOperands());
    newBlockTransferArgs.push_back(Value()); // iv placeholder with null value
    newIterArgs.reserve(forOp.getNumIterOperands());
    newYieldValues.reserve(yieldOp.getNumOperands());
    newResultValues.reserve(forOp.getNumResults());
    for (auto it : llvm::zip(forOp.getIterOperands(),   // iter from outside
                             forOp.getRegionIterArgs(), // iter inside region
                             forOp.getResults(),        // op results
                             yieldOp.getOperands()      // iter yield
                             )) {
      bool forwarded = true;
      Value init = std::get<0>(it);
      if (auto iter_init = forOp.lowerBound().getDefiningOp<ConstantIndexOp>()) {
        if (auto op = init.getDefiningOp<ConstantOp>()) {
          if (op.getValue().cast<IntegerAttr>().getValue() != iter_init.getValue()) {
            forwarded = false;
          }
        } else forwarded = false;
      } else if (init != forOp.lowerBound()) {
        forwarded = false;
      }

      AddIOp addOp = std::get<3>(it).getDefiningOp<AddIOp>();
      if (!addOp)
        forwarded = false;
      else {
        if (addOp.getOperand(0) != std::get<1>(it)) {
          forwarded = false;
        } else {

          if (auto iter_step = forOp.step().getDefiningOp<ConstantIndexOp>()) {
            if (auto op = addOp.getOperand(1).getDefiningOp<ConstantOp>()) {
              if (op.getValue().cast<IntegerAttr>().getValue() != iter_step.getValue()) {
                forwarded = false;
              }
            } else forwarded = false;
          } else if (addOp.getOperand(1) != forOp.step()) {
            forwarded = false;
          }
        }
      }
      if (forwarded) {
        Value replacement = forOp.getInductionVar();
        if (!std::get<1>(it).getType().isa<IndexType>()) {
          rewriter.setInsertionPointToStart(&forOp.region().front());
          replacement = rewriter.create<IndexCastOp>(forOp.getLoc(), replacement, std::get<1>(it).getType());
        }
        rewriter.updateRootInPlace(forOp, [&]{
          std::get<1>(it).replaceAllUsesWith(replacement);
        });
        canonicalize = true;
      }
    }

    if (!canonicalize)
      return failure();

    return success();
  }
};

/// Remove unused iterator operands.
// TODO: BlockAndValueMapping for indvar.
struct RemoveUnusedArgs : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 2> usedBlockArgs;
    SmallVector<OpResult, 2> usedResults;
    SmallVector<Value, 2> usedOperands;

    unsigned i = 0;
    // if the block argument or the result at the
    // same index position have uses do not eliminate.
    for (auto blockArg : op.getRegionIterArgs()) {
      if ((!blockArg.use_empty()) || (!op.getResult(i).use_empty())) {
        usedOperands.push_back(op.getOperand(op.getNumControlOperands() + i));
        usedResults.push_back(op->getOpResult(i));
        usedBlockArgs.push_back(blockArg);
      }
      i++;
    }

    // no work to do.
    if (usedOperands.size() == op.getIterOperands().size())
      return failure();

    auto newForOp = rewriter.create<ForOp>(
        op.getLoc(), op.lowerBound(), op.upperBound(), op.step(),
        usedOperands);
    
    if (!newForOp.getBody()->empty())
      rewriter.eraseOp(newForOp.getBody()->getTerminator());
  
    newForOp.getBody()->getOperations().splice(newForOp.getBody()->getOperations().begin(), op.getBody()->getOperations());
    op.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());
    for (auto pair : llvm::zip(usedBlockArgs, newForOp.getRegionIterArgs())) {
      std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
    }

    // adjust return.
    auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    SmallVector<Value, 2> usedYieldOperands{};
    llvm::transform(usedResults, std::back_inserter(usedYieldOperands),
                    [&](OpResult result) {
                      return yieldOp.getOperand(result.getResultNumber());
                    });
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, usedYieldOperands);
    rewriter.eraseOp(yieldOp);

    // Replace the operation's results with the new ones.
    SmallVector<Value, 4> repResults(op.getNumResults());
    for (auto en : llvm::enumerate(usedResults))
      repResults[en.value().cast<OpResult>().getResultNumber()] =
          newForOp.getResult(en.index());

    rewriter.replaceOp(op, repResults);
    return success();
  }
};

/*
+struct RemoveNotIf : public OpRewritePattern<IfOp> {
+  using OpRewritePattern<IfOp>::OpRewritePattern;
+
+  LogicalResult matchAndRewrite(IfOp op,
+                                PatternRewriter &rewriter) const override {
+    // Replace the operation if only a subset of its results have uses.
+    if (op.getNumResults() == 0)
+      return failure();
+
+    auto trueYield = cast<scf::YieldOp>(op.thenRegion().back().getTerminator());
+    auto falseYield =
+        cast<scf::YieldOp>(op.thenRegion().back().getTerminator());
+
+    rewriter.setInsertionPoint(op->getBlock(),
+                               op.getOperation()->getIterator());
+    bool changed = false;
+    for (auto tup :
+         llvm::zip(trueYield.results(), falseYield.results(), op.results())) {
+      if (!std::get<0>(tup).getType().isInteger(1))
+        continue;
+      if (auto top = std::get<0>(tup).getDefiningOp<ConstantOp>()) {
+        if (auto fop = std::get<1>(tup).getDefiningOp<ConstantOp>()) {
+          if (top.getValue().cast<IntegerAttr>().getValue() == 0 &&
+              fop.getValue().cast<IntegerAttr>().getValue() == 1) {
+
+            for (OpOperand &use :
+                 llvm::make_early_inc_range(std::get<2>(tup).getUses())) {
+              changed = true;
+              rewriter.updateRootInPlace(use.getOwner(), [&]() {
+                use.set(rewriter.create<XOrOp>(op.getLoc(), op.condition()));
+              });
+            }
+          }
+          if (top.getValue().cast<IntegerAttr>().getValue() == 1 &&
+              fop.getValue().cast<IntegerAttr>().getValue() == 0) {
+            for (OpOperand &use :
+                 llvm::make_early_inc_range(std::get<2>(tup).getUses())) {
+              changed = true;
+              rewriter.updateRootInPlace(use.getOwner(),
+                                         [&]() { use.set(op.condition()); });
+            }
+          }
+        }
+      }
+    }
+    return changed ? success() : failure();
+  }
+};
+struct CombineIfs : public OpRewritePattern<IfOp> {
+  using OpRewritePattern<IfOp>::OpRewritePattern;
+
+  LogicalResult matchAndRewrite(IfOp op,
+                                PatternRewriter &rewriter) const override {
+    if (op.elseRegion().getBlocks().size() >= 2)
+      return failure();
+    assert(op.thenRegion().getBlocks().size());
+    assert(op.elseRegion().getBlocks().size() <= 1);
+    Block *parent = op->getBlock();
+    if (op == &parent->back())
+      return failure();
+    auto nextIf = dyn_cast<IfOp>(op->getNextNode());
+    if (!nextIf)
+      return failure();
+    if (op.results().size() != 0)
+      return failure();
+    if (nextIf.condition() != op.condition())
+      return failure();
+
+    rewriter.updateRootInPlace(nextIf, [&]() {
+      Block &then = *op.thenRegion().begin();
+      rewriter.eraseOp(&then.back());
+      rewriter.mergeBlocks(&*nextIf.thenRegion().begin(), &then);
+      nextIf.thenRegion().getBlocks().splice(
+          nextIf.thenRegion().getBlocks().begin(), op.thenRegion().getBlocks());
+      // rewriter.mergeBlockBefore(&then,
+      // &*nextIf.thenRegion().begin()->begin());
+
+      assert(nextIf.thenRegion().getBlocks().size());
+
+      if (!op.elseRegion().empty()) {
+        Block &elser = *op.elseRegion().begin();
+        if (nextIf.elseRegion().empty()) {
+          auto &eb = *(new Block());
+          nextIf.elseRegion().getBlocks().push_back(&eb);
+          // nextIf.elseRegion().begin()->getOperations().splice(nextIf.elseRegion().begin()->begin(),
+          // elser.getOperations());
+          rewriter.mergeBlocks(&elser, &eb);
+        } else {
+          rewriter.eraseOp(&elser.back());
+          // rewriter.mergeBlockBefore(&elser,
+          // &*nextIf.elseRegion().begin()->begin());
+          rewriter.mergeBlocks(&*nextIf.elseRegion().begin(), &elser);
+          nextIf.elseRegion().getBlocks().splice(
+              nextIf.elseRegion().getBlocks().begin(),
+              op.elseRegion().getBlocks());
+        }
+        assert(nextIf.elseRegion().getBlocks().size());
+      }
+    });
+    rewriter.eraseOp(op);
+    return success();
+  }
+};
+struct RemoveBoolean : public OpRewritePattern<IfOp> {
+  using OpRewritePattern<IfOp>::OpRewritePattern;
+
+  LogicalResult matchAndRewrite(IfOp op,
+                                PatternRewriter &rewriter) const override {
+    bool changed = false;
+
+    if (llvm::all_of(op.results(), [](Value v) {
+          return v.getType().isa<IntegerType>() &&
+                 v.getType().cast<IntegerType>().getWidth() == 1;
+        })) {
+      if (op.thenRegion().getBlocks().size() == 1 &&
+          op.elseRegion().getBlocks().size() == 1) {
+        while (isa<CmpIOp>(op.thenRegion().front().front())) {
+          op.thenRegion().front().front().moveBefore(op);
+          changed = true;
+        }
+        while (isa<CmpIOp>(op.elseRegion().front().front())) {
+          op.elseRegion().front().front().moveBefore(op);
+          changed = true;
+        }
+        if (op.thenRegion().front().getOperations().size() == 1 &&
+            op.elseRegion().front().getOperations().size() == 1) {
+          auto yop1 =
+              cast<scf::YieldOp>(op.thenRegion().front().getTerminator());
+          auto yop2 =
+              cast<scf::YieldOp>(op.elseRegion().front().getTerminator());
+          size_t idx = 0;
+
+          auto c1 = (mlir::Value)rewriter.create<mlir::ConstantOp>(
+              op.getLoc(), op.condition().getType(),
+              rewriter.getIntegerAttr(op.condition().getType(), 1));
+          auto notcond = (mlir::Value)rewriter.create<mlir::XOrOp>(
+              op.getLoc(), op.condition(), c1);
+
+          std::vector<Value> replacements;
+          for (auto res : op.results()) {
+            auto rep = rewriter.create<OrOp>(
+                op.getLoc(),
+                rewriter.create<AndOp>(op.getLoc(), op.condition(),
+                                       yop1.results()[idx]),
+                rewriter.create<AndOp>(op.getLoc(), notcond,
+                                       yop2.results()[idx]));
+            replacements.push_back(rep);
+            idx++;
+          }
+          rewriter.replaceOp(op, replacements);
+          // op.erase();
+          return success();
+        }
+      }
+    }
+
+    if (op.thenRegion().getBlocks().size() == 1 &&
+        op.elseRegion().getBlocks().size() == 1 &&
+        op.thenRegion().front().getOperations().size() == 1 &&
+        op.elseRegion().front().getOperations().size() == 1) {
+      auto yop1 = cast<scf::YieldOp>(op.thenRegion().front().getTerminator());
+      auto yop2 = cast<scf::YieldOp>(op.elseRegion().front().getTerminator());
+      size_t idx = 0;
+
+      std::vector<Value> replacements;
+      for (auto res : op.results()) {
+        auto rep =
+            rewriter.create<SelectOp>(op.getLoc(), op.condition(),
+                                      yop1.results()[idx], yop2.results()[idx]);
+        replacements.push_back(rep);
+        idx++;
+      }
+      rewriter.replaceOp(op, replacements);
+      return success();
+    }
+    return changed ? success() : failure();
+  }
+};
*/

bool isWhile(WhileOp wop) {
  bool hasCondOp = false;
  wop.before().walk([&](Operation *op) {
    if (isa<scf::ConditionOp>(op))
      hasCondOp = true;
  });
  return hasCondOp;
}

struct MoveWhileToFor : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  bool isTopLevelArgValue(Value value, Region *region) const {
    if (auto arg = value.dyn_cast<BlockArgument>())
      return arg.getParentRegion() == region;
    return false;
  }

  bool isBlockArg(Value value) const {
    if (auto arg = value.dyn_cast<BlockArgument>())
      return true;
    return false;
  }

  bool dominateWhile(Value value, WhileOp loop) const {
    Operation *op = value.getDefiningOp();
    assert(op && "expect non-null");
    DominanceInfo dom(loop);
    return dom.properlyDominates(op, loop);
  }

  bool canMoveOpOutsideWhile(Operation *op, WhileOp loop) const {
    DominanceInfo dom(loop);
    for (auto operand : op->getOperands()) {
      if (!dom.properlyDominates(operand, loop))
        return false;
    }
    return true;
  }

  unsigned countOperations(Region &reg) const {
    unsigned count = 0;
    for (auto &block : reg)
      for (auto &nested : block)
        count++;
    return count;
  }

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    if (!isWhile(loop))
      return failure();

    struct LoopInfo {
      Value indVar = nullptr;
      Type indVarType = nullptr;
      Value ub = nullptr;
      Value lb = nullptr;
      Value step = nullptr;
    } loopInfo;

    auto condOp = loop.getConditionOp();
    SmallVector<Value, 2> results = {condOp.args()};
    auto cmpIOp = condOp.condition().getDefiningOp<CmpIOp>();
    if (!cmpIOp) {
      llvm::errs() << condOp << "\n";
      llvm::errs() << condOp.condition() << "\n";
      return failure();
    }
    size_t beforeArgNum;

    Value maybeIndVar = cmpIOp.lhs();
    if (isTopLevelArgValue(maybeIndVar, &loop.before())) {
      beforeArgNum = maybeIndVar.cast<BlockArgument>().getArgNumber();
      loopInfo.lb = loop.getOperand(beforeArgNum);
    } else {
      llvm::errs() << " non top level arg: " << maybeIndVar << "\n";
      return failure();
    }

    SmallVector<size_t, 2> afterArgs;
    for (auto pair : llvm::enumerate(condOp.args())) {
      if (pair.value() == maybeIndVar)
        afterArgs.push_back(pair.index());
    }

    auto endYield = cast<YieldOp>(loop.after().back().getTerminator());

    auto addIOp = endYield.results()[beforeArgNum].getDefiningOp<AddIOp>();
    if (!addIOp) return failure();

    for (auto afterArg : afterArgs) {
      auto arg = loop.after().getArgument(afterArg);
      if (addIOp.getOperand(0) == arg) {
        loopInfo.step = addIOp.getOperand(1);
        break;
      }
      if (addIOp.getOperand(1) == arg) {
        loopInfo.step = addIOp.getOperand(0);
        break;
      }
    }

    if (!loopInfo.step)
      return failure();

    Value indVar = maybeIndVar;

    if (isBlockArg(cmpIOp.rhs()) || dominateWhile(cmpIOp.rhs(), loop)) {
      switch (cmpIOp.getPredicate()) {
      case CmpIPredicate::slt:
      case CmpIPredicate::ult: {
        loopInfo.ub = cmpIOp.rhs();
        break;
      }
      case CmpIPredicate::sle: {
        // TODO: f32 likely not always true.
        auto one =
            rewriter.create<ConstantOp>(loop.getLoc(), rewriter.getI32Type(),
                                        rewriter.getI32IntegerAttr(1));
        auto addIOp =
            rewriter.create<AddIOp>(loop.getLoc(), cmpIOp.rhs(), one);
        loopInfo.ub = addIOp.getResult();
        break;
      }
      case CmpIPredicate::eq:
      case CmpIPredicate::sge:
      case CmpIPredicate::sgt:
      case CmpIPredicate::ne:
      case CmpIPredicate::ule:
      case CmpIPredicate::ugt:
      case CmpIPredicate::uge: {
        return failure();
      }
      }
    } else {
      auto *op = cmpIOp.rhs().getDefiningOp();
      if (!op || !canMoveOpOutsideWhile(op, loop) ||
          (op->getNumResults() != 1))
        return failure();
      auto newOp = rewriter.clone(*op);
      loopInfo.ub = newOp->getResult(0);
      cmpIOp.rhs().replaceAllUsesWith(newOp->getResult(0));
    }

    loopInfo.indVar = indVar;
    loopInfo.indVarType = indVar.getType();

    if ((!loopInfo.ub) || (!loopInfo.lb) || (!loopInfo.step))
      return failure();

    Value ub = rewriter.create<IndexCastOp>(loop.getLoc(), loopInfo.ub,
                                            IndexType::get(loop.getContext()));
    Value lb = rewriter.create<IndexCastOp>(loop.getLoc(), loopInfo.lb,
                                            IndexType::get(loop.getContext()));
    Value step = rewriter.create<IndexCastOp>(
        loop.getLoc(), loopInfo.step, IndexType::get(loop.getContext()));

    // input of the for goes the input of the scf::while plus the output taken
    // from the conditionOp.
    SmallVector<Value, 8> forArgs;
    forArgs.append(loop.inits().begin(), loop.inits().end());

    // auto m = loop.getParentOfType<ModuleOp>();
    // m.dump();
    // llvm::errs() << "******************\n";
    // loop.dump();

    for (Value arg : condOp.args()) {
      if (isTopLevelArgValue(arg, &loop.before())) {
        auto blockArg = arg.dyn_cast<BlockArgument>();
        auto pos = blockArg.getArgNumber();
        forArgs.push_back(loop.inits()[pos]);
      } else
        forArgs.push_back(arg);
    }

    auto forloop = rewriter.create<scf::ForOp>(
        loop.getLoc(), lb, ub, step, forArgs,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
          // map for the conditionOp value.
          size_t pos = loop.inits().size();
          SmallVector<Value, 2> mappedValues;
          mappedValues.append(args.begin() + pos, args.end());

          BlockAndValueMapping mapping;
          mapping.map(loop.after().getArguments(), mappedValues);
          for (auto &block : loop.after().getBlocks())
            for (auto &nested : block.without_terminator())
              b.clone(nested, mapping);

          auto oldYield =
              cast<scf::YieldOp>(loop.after().front().getTerminator());
          SmallVector<Value, 2> yieldOperands;
          for (auto oldYieldArg : oldYield.results())
            yieldOperands.push_back(mapping.lookupOrDefault(oldYieldArg));

          BlockAndValueMapping outmap;
          outmap.map(loop.before().getArguments(), yieldOperands);
          for (auto arg : condOp.args())
            yieldOperands.push_back(outmap.lookupOrDefault(arg));

          b.create<scf::YieldOp>(loop.getLoc(), yieldOperands);
        });

    SmallVector<Value, 2> replacements;
    size_t pos = loop.inits().size();
    replacements.append(forloop.getResults().begin() + pos,
                        forloop.getResults().end());
    rewriter.replaceOp(loop, replacements);
    auto m = forloop->getParentOfType<ModuleOp>();
    return success();
  }
};

struct MoveWhileDown : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.before().front().getTerminator());
    if (auto ifOp = term.condition().getDefiningOp<scf::IfOp>()) {
      if (ifOp.getNumResults() != term.args().size() + 1)
        return failure();
      if (ifOp.getResult(0) != term.condition())
        return failure();
      for (size_t i = 1; i < ifOp.getNumResults(); ++i) {
        if (ifOp.getResult(i) != term.args()[i - 1])
          return failure();
      }
      auto yield1 =
          cast<scf::YieldOp>(ifOp.thenRegion().front().getTerminator());
      auto yield2 =
          cast<scf::YieldOp>(ifOp.elseRegion().front().getTerminator());
      if (auto cop = yield1.getOperand(0).getDefiningOp<ConstantOp>()) {
        if (cop.getValue().cast<IntegerAttr>().getValue() == 0)
          return failure();
      } else
        return failure();
      if (auto cop = yield2.getOperand(0).getDefiningOp<ConstantOp>()) {
        if (cop.getValue().cast<IntegerAttr>().getValue() != 0)
          return failure();
      } else
        return failure();
      if (ifOp.elseRegion().front().getOperations().size() != 1)
        return failure();
      op.after().front().getOperations().splice(
          op.after().front().begin(),
          ifOp.thenRegion().front().getOperations());
      term.conditionMutable().assign(ifOp.condition());
      SmallVector<Value, 2> args;
      for (size_t i = 1; i < yield2.getNumOperands(); ++i) {
        args.push_back(yield2.getOperand(i));
      }
      term.argsMutable().assign(args);
      rewriter.eraseOp(yield2);
      rewriter.eraseOp(ifOp);

      for (size_t i = 0; i < op.after().front().getNumArguments(); ++i) {
        op.after().front().getArgument(i).replaceAllUsesWith(
            yield1.getOperand(i + 1));
      }
      rewriter.eraseOp(yield1);
      // TODO move operands from begin to after
      SmallVector<Value> todo(op.before().front().getArguments().begin(),
                              op.before().front().getArguments().end());
      for (auto &op : op.before().front()) {
        for (auto res : op.getResults()) {
          todo.push_back(res);
        }
      }
      for (auto val : todo) {
        auto na = op.after().front().addArgument(val.getType());
        val.replaceUsesWithIf(na, [&](OpOperand &u) -> bool {
          return op.after().isAncestor(u.getOwner()->getParentRegion());
        });
        args.push_back(val);
      }
      term.argsMutable().assign(args);

      SmallVector<Type, 4> tys;
      for (auto a : args)
        tys.push_back(a.getType());

      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.inits());
      op2.before().takeBody(op.before());
      op2.after().takeBody(op.after());
      SmallVector<Value, 4> replacements;
      for (auto a : op2.getResults()) {
        if (replacements.size() == op.getResults().size())
          break;
        replacements.push_back(a);
      }
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }
};

#if 1
struct MoveWhileDown2 : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;


  /// Populates `crossing` with values (op results) that are defined in the same
  /// block as `op` and above it, and used by at least one op in the same block
  /// below `op`. Uses may be in nested regions.
  static void findValuesUsedBelow(IfOp op,
                                  llvm::SetVector<Value> &crossing) {
    for (Operation *it = op->getPrevNode(); it != nullptr;
        it = it->getPrevNode()) {
      for (Value value : it->getResults()) {
        for (Operation *user : value.getUsers()) {
          // ignore use of condition
          if (user == op) continue;

          if (op->isAncestor(user)) {
            crossing.insert(value);
            break;
          }
        }
      }
    }

      for (Value value : op->getBlock()->getArguments()) {
        for (Operation *user : value.getUsers()) {
          // ignore use of condition
          if (user == op) continue;

          if (op->isAncestor(user)) {
            crossing.insert(value);
            break;
          }
        }
      }
    // No need to process block arguments, they are assumed to be induction
    // variables and will be replicated.
  }

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.before().front().getTerminator());
    if (auto ifOp = dyn_cast_or_null<scf::IfOp>(term->getPrevNode())) {
      if (ifOp.condition() != term.condition())
        return failure();
      
      llvm::errs() << " moving while to for2: " << op <<"\n";
      SmallVector<std::pair<BlockArgument, Value>, 2> m;
      SmallVector<Value, 2> condArgs;
      SmallVector<Value, 2> prevArgs;
      for (auto pair : llvm::zip(op.getResults(), term.args(), op.getAfterArguments()) ) {
        if (std::get<1>(pair).getDefiningOp() == ifOp) {
          if (!std::get<0>(pair).use_empty())
            return failure();
          Value yielded;
          for(auto p : llvm::zip(ifOp.thenYield().results(), ifOp.results())) {
            if (std::get<1>(pair) == std::get<1>(p)) {
              yielded = std::get<0>(p);
              break;
            }
          }
          assert(yielded);
          m.emplace_back(std::get<2>(pair), yielded);
        } else {
          assert(prevArgs.size() == condArgs.size());
          prevArgs.push_back(std::get<0>(pair));
          condArgs.push_back(std::get<1>(pair));
        }
      }

      llvm::SetVector<Value> sv;
      findValuesUsedBelow(ifOp, sv);

      Block* afterB = &op.after().front();

      for(auto v : sv) {
        condArgs.push_back(v);
        auto arg = afterB->addArgument(v.getType());
        for (OpOperand &use :
            llvm::make_early_inc_range(v.getUses())) {
          if (ifOp->isAncestor(use.getOwner()))
            rewriter.updateRootInPlace(use.getOwner(),
                                     [&]() { use.set(arg); });
        }
      }
      

      rewriter.setInsertionPoint(term);
      auto ncond = rewriter.replaceOpWithNewOp<ConditionOp>(term, term.condition(), condArgs);

      for(int i=m.size()-1; i>=0; i--) {
        m[i].first.replaceAllUsesWith(m[i].second);
        afterB->eraseArgument(m[i].first.getArgNumber());
      }

      rewriter.eraseOp(ifOp.thenYield());
      Block* thenB = ifOp.thenBlock();
      afterB->getOperations().splice(afterB->getOperations().begin(), thenB->getOperations());

      rewriter.eraseOp(ifOp);

      SmallVector<Type, 4> resultTypes;
      for(auto v : condArgs) {
        resultTypes.push_back(v.getType());
      }

      
      rewriter.setInsertionPoint(op);
      auto nop = rewriter.create<WhileOp>(op.getLoc(), resultTypes, op.inits());
      nop.before().takeBody(op.before());
      nop.after().takeBody(op.after());
      
      for(auto pair : llvm::enumerate(prevArgs)) {
        pair.value().replaceAllUsesWith(nop.getResult(pair.index()));
        /*
        for (OpOperand &use :
            llvm::make_early_inc_range(pair.value().getUses())) {
          if (nop.after().isAncestor(use.getOwner()->getParentRegion()))
            rewriter.updateRootInPlace(use.getOwner(),
                                     [&]() { use.set(nop.getResult(pair.index())); });
        }
        */
      }

      rewriter.eraseOp(op);


      llvm::errs() << " nop: " << nop << "\n";

      return success();
    }
    return failure();
  }
};
#endif

struct RemoveUnusedCondVar : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.before().front().getTerminator());
    SmallVector<Value, 4> conds;
    SmallVector<unsigned, 4> eraseArgs;
    SmallVector<unsigned, 4> keepArgs;
    SmallVector<Type, 4> tys;
    unsigned i = 0;
    std::map<void *, unsigned> valueOffsets;
    std::map<unsigned, unsigned> resultOffsets;
    SmallVector<Value, 4> resultArgs;
    for (auto arg : term.args()) {
      if (op.after().front().getArgument(i).use_empty() &&
          op.getResult(i).use_empty()) {
        eraseArgs.push_back((unsigned)i);
      } else if (valueOffsets.find(arg.getAsOpaquePointer()) !=
                 valueOffsets.end()) {
        resultOffsets[i] = valueOffsets[arg.getAsOpaquePointer()];
        op.after().front().getArgument(i).replaceAllUsesWith(
            resultArgs[valueOffsets[arg.getAsOpaquePointer()]]);
        eraseArgs.push_back((unsigned)i);
      } else {
        valueOffsets[arg.getAsOpaquePointer()] = keepArgs.size();
        resultOffsets[i] = keepArgs.size();
        resultArgs.push_back(op.after().front().getArgument(i));
        conds.push_back(arg);
        keepArgs.push_back((unsigned)i);
        tys.push_back(arg.getType());
      }
      i++;
    }

    if (eraseArgs.size() != 0) {
      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.inits());
      op2.before().takeBody(op.before());
      op2.after().takeBody(op.after());
      for (auto pair : resultOffsets) {
        op.getResult(pair.first).replaceAllUsesWith(op2.getResult(pair.second));
      }
      rewriter.eraseOp(op);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.condition(),
                                                    conds);
      op2.after().front().eraseArguments(eraseArgs);
      return success();
    }
    return failure();
  }
};

struct MoveSideEffectFreeWhile : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.before().front().getTerminator());
    SmallVector<Value, 4> conds(term.args().begin(), term.args().end());
    bool changed = false;
    unsigned i = 0;
    for (auto arg : term.args()) {
      if (auto IC = arg.getDefiningOp<IndexCastOp>()) {
        if (arg.hasOneUse() && op.getResult(i).use_empty()) {
          auto rep =
              op.after().front().addArgument(IC->getOperand(0).getType());
          IC->moveBefore(&op.after().front(), op.after().front().begin());
          conds.push_back(IC.in());
          IC.inMutable().assign(rep);
          op.after().front().getArgument(i).replaceAllUsesWith(
              IC->getResult(0));
          changed = true;
        }
      }
      i++;
    }
    if (changed) {
      SmallVector<Type, 4> tys;
      for (auto arg : conds) {
        tys.push_back(arg.getType());
      }
      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.inits());
      op2.before().takeBody(op.before());
      op2.after().takeBody(op.after());
      unsigned j = 0;
      for (auto a : op.getResults()) {
        a.replaceAllUsesWith(op2.getResult(j));
        j++;
      }
      rewriter.eraseOp(op);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.condition(),
                                                    conds);
      return success();
    }
    return failure();
  }
};

void CanonicalizeFor::runOnFunction() {
  getFunction().walk([&](scf::ForOp forOp) { propagateInLoopBody(forOp); });
  getFunction().walk(
      [&](scf::ForOp forOp) { detectTrivialIndVarInArgs(forOp); });
  {
    
    mlir::RewritePatternSet rpl(getFunction().getContext());
    rpl.add<ForOpInductionReplacement, RemoveUnusedArgs,
    MoveWhileToFor, MoveWhileDown, MoveWhileDown2,
    RemoveUnusedCondVar, MoveSideEffectFreeWhile>(getFunction().getContext());
    GreedyRewriteConfig config;
    applyPatternsAndFoldGreedily(getFunction().getOperation(), std::move(rpl),
                                 config);
  }
}

std::unique_ptr<Pass> mlir::polygeist::createCanonicalizeForPass() {
  return std::make_unique<CanonicalizeFor>();
}
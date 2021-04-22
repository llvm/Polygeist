#include "PassDetail.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

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
      Operation *lbDefOp = forOp.getLowerBound().getDefiningOp();
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
      Operation *stepForDefOp = forOp.getStep().getDefiningOp();
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

void CanonicalizeFor::runOnFunction() {
  getFunction().walk([&](scf::ForOp forOp) { propagateInLoopBody(forOp); });
  getFunction().walk(
      [&](scf::ForOp forOp) { detectTrivialIndVarInArgs(forOp); });
}

std::unique_ptr<Pass> mlir::createCanonicalizeForPass() {
  return std::make_unique<CanonicalizeFor>();
}

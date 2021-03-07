//===- ScopStmtOpt.cc - Optimise SCoP statement extraction ------------C++-===//

#include "polymer/Transforms/ScopStmtOpt.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/ADT/SetVector.h"

#include <queue>
#include <utility>

#define DEBUG_TYPE "scop-stmt-opt"

using namespace mlir;
using namespace llvm;
using namespace polymer;

/// Return the corresponding AllocaOp.
// static Operation *insertScratchpad(mlir::AffineForOp forOp, OpBuilder &b) {

//   OpBuilder::InsertionGuard guard(b);

//   mlir::AffineBound lowerBound = forOp.getLowerBound();
//   mlir::AffineBound upperBound = forOp.getUpperBound();

//   assert(lowerBound.getMap().getNumResults() == 1);
//   assert(lowerBound.getMap().getNumDims() == 0);
//   assert(upperBound.getMap().getNumResults() == 1);
//   assert(upperBound.getMap().getNumDims() == 0);

//   // TODO: for now we use the upper bound to create a scratchpad. Its size
//   can
//   // be refined later.
//   b.setInsertionPoint(innermostForOp);
//   // b.create<mlir::AllocaOp>(op->getLoc(), upperBound);

//   return nullptr;
// }

static void replace(ValueRange srcValues,
                    SmallVectorImpl<mlir::Value> &dstValues,
                    BlockAndValueMapping &mapping) {
  for (Value src : srcValues) {
    // src could come from an index_cast.
    if (IndexCastOp op = src.getDefiningOp<IndexCastOp>())
      src = op.getOperand();

    dstValues.push_back(mapping.lookup(src));
  }
}

static Operation *apply(mlir::AffineMap affMap, ValueRange operands,
                        BlockAndValueMapping &mapping, mlir::CallOp call,
                        OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);

  SmallVector<mlir::Value, 8> newOperands;
  replace(operands, newOperands, mapping);

  if (newOperands.size() > 0)
    b.setInsertionPointAfterValue(newOperands[0]);
  else
    b.setInsertionPointToStart(
        &(*(call.getParentOfType<FuncOp>().body().begin())));

  // TODO: properly handle these index casting cases.
  for (size_t i = 0; i < newOperands.size(); i++)
    if (newOperands[i].getType() != b.getIndexType())
      newOperands[i] = b.create<IndexCastOp>(call.getLoc(), newOperands[i],
                                             b.getIndexType());

  return b.create<mlir::AffineApplyOp>(call.getLoc(), affMap, newOperands);
}

/// Calculate the lower bound and upper bound through affine apply, before the
/// function is being called.
static mlir::Value getMemRefSize(mlir::AffineForOp forOp, FuncOp f, CallOp call,
                                 OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);

  BlockAndValueMapping mapping;
  mapping.map(f.getArguments(), call.getOperands());
  // for (Value arg : f.getArguments()) {
  // arg.dump();
  // mapping.lookup(arg).dump();
  // }

  mlir::AffineMap lbMap = forOp.getLowerBoundMap();
  mlir::AffineMap ubMap = forOp.getUpperBoundMap();

  // assert(lbMap.getNumResults() == 1 && "Should be a single lower bound.");

  assert(lbMap.isSingleConstant());
  assert(lbMap.getSingleConstantResult() == 0);
  assert(ubMap.getNumResults() == 1 && "Should be a single upper bound.");

  // lbMap.dump();
  // ubMap.dump();

  // TODO: find better way to do this.
  // Operation *lbOp =
  //     apply(lbMap, forOp.getLowerBoundOperands(), mapping, call, b);
  Operation *ubOp =
      apply(ubMap, forOp.getUpperBoundOperands(), mapping, call, b);
  // lbOp->dump();
  // ubOp->dump();

  // AffineMap affMap =
  //     AffineMap::get(0, 2, b.getAffineSymbolExpr(0) -
  //     b.getAffineSymbolExpr(1),
  //                    f.getContext());

  // b.setInsertionPoint(call);
  // Operation *memSize = b.create<mlir::AffineApplyOp>(
  //     forOp.getLoc(), ubMap,
  //     ;
  // Operation *memSize = b.create<mlir::AffineApplyOp>(
  //     forOp.getLoc(), affMap,
  //     ValueRange({ubOp->getResult(0), lbOp->getResult(0)}));
  // memSize->dump();

  return ubOp->getResult(0);
}

/// Append the given argument to the end of the argument list for both the
/// function and the caller.
static Value appendArgument(Value arg, FuncOp func, CallOp call, OpBuilder &b) {
  SmallVector<Type, 4> argTypes;
  SmallVector<Value, 4> operands;
  for (Type type : func.getArgumentTypes())
    argTypes.push_back(type);
  for (Value operand : call.getOperands())
    operands.push_back(operand);

  argTypes.push_back(arg.getType());
  operands.push_back(arg);

  call->setOperands(operands);
  func.setType(b.getFunctionType(argTypes, TypeRange(call.getResults())));
  Block &entryBlock = *(func.body().begin());
  entryBlock.addArgument(arg.getType());

  return entryBlock.getArguments().back();
}

static void scopStmtSplit(ModuleOp m, OpBuilder &b, FuncOp f, mlir::CallOp call,
                          Operation *op) {
  // op->dump();

  SmallVector<mlir::AffineForOp, 4> forOps;
  getLoopIVs(*op, &forOps);

  assert(forOps.size() >= 1 &&
         "The given op to split should be enclosed in at least one affine.for");

  // For now we focus on the innermost for loop.
  mlir::AffineForOp forOp = forOps.back();
  // forOp.dump();

  mlir::Value memSize = getMemRefSize(forOp, f, call, b);
  // Since there is only one loop depth.
  MemRefType memType = MemRefType::get({-1}, op->getResult(0).getType());

  b.setInsertionPointAfterValue(memSize);
  // Allocation of the scratchpad memory.
  Operation *scrAlloc =
      b.create<mlir::AllocOp>(forOp.getLoc(), memType, memSize);

  // Pass it into the target function.
  Value scrInFunc = appendArgument(scrAlloc->getResult(0), f, call, b);
  // scrInFunc.dump();

  // Insert scratchpad read and write.
  b.setInsertionPointAfter(op);
  Operation *loadOp = b.create<mlir::AffineLoadOp>(op->getLoc(), scrInFunc,
                                                   forOp.getInductionVar());
  op->replaceAllUsesWith(loadOp);

  b.setInsertionPointAfter(op);
  b.create<mlir::AffineStoreOp>(op->getLoc(), op->getResult(0), scrInFunc,
                                forOp.getInductionVar());
}

static void scopStmtSplit(ModuleOp m, OpBuilder &b, int toSplit) {
  FuncOp func;
  CallOp call;
  Operation *opToSplit;

  // Find the target function and the op to split within it.
  bool found = false;
  m.walk([&](FuncOp f) {
    if (found)
      return;

    f.walk([&](Operation *op) {
      if (found)
        return;
      if (op->hasAttr("scop.splittable") &&
          op->getAttrOfType<mlir::IntegerAttr>("scop.splittable").getInt() ==
              toSplit) {
        func = f;
        opToSplit = op;
        found = true;
      }
    });
  });

  assert(found && "Given split ID cannot be found");

  // Find the corresponding call op.
  m.walk([&](CallOp callOp) {
    if (callOp.callee() == func.getName())
      call = callOp;
  });

  // call.dump();
  scopStmtSplit(m, b, func, call, opToSplit);
}

namespace {
struct ScopStmtSplitPass
    : public mlir::PassWrapper<ScopStmtSplitPass, OperationPass<ModuleOp>> {
  ScopStmtSplitPass() = default;
  ScopStmtSplitPass(const ScopStmtSplitPass &pass) {}

  ListOption<int> toSplit{
      *this, "to-split",
      llvm::cl::desc(
          "A list of integer IDs describing the selected split points.")};

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    for (auto id : toSplit)
      scopStmtSplit(m, b, id);
  }
};
} // namespace

static bool isSplittable(Operation *op) {

  SmallVector<mlir::AffineForOp, 4> forOps;
  getLoopIVs(*op, &forOps);

  if (forOps.size() < 1)
    return false;

  // For now we focus on the innermost for loop.
  mlir::AffineForOp forOp = forOps.back();
  mlir::AffineMap lbMap = forOp.getLowerBoundMap();
  mlir::AffineMap ubMap = forOp.getUpperBoundMap();
  if (!lbMap.isSingleConstant() || lbMap.getSingleConstantResult() != 0)
    return false;
  if (ubMap.getNumDims() != 0)
    return false;

  return true;
}

static int annotateSplittable(FuncOp f, OpBuilder &b, int startId) {
  int id = startId;

  f.walk([&](mlir::AffineStoreOp storeOp) {
    // Breadth first search to find the splittable operations.

    // Stores operation and depth pairs.
    std::queue<std::pair<Operation *, int>> worklist;
    worklist.push(std::make_pair(storeOp, 0));

    while (!worklist.empty()) {
      std::pair<Operation *, int> front = worklist.front();
      worklist.pop();

      Operation *op = front.first;
      int depth = front.second;

      // Annotation.
      if (depth > 1 && !op->hasAttr("scop.splittable") && isSplittable(op)) {
        op->setAttr("scop.splittable", b.getIndexAttr(id));
        id++;
      }

      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        // Filter out block arguments.
        if (defOp == nullptr || operand.isa<BlockArgument>())
          continue;
        // Filter out operations out of the current region.
        if (defOp->getParentRegion() != storeOp.getParentRegion())
          continue;
        // Filter out defining operations of specific types.
        if (isa<mlir::AffineReadOpInterface, mlir::ConstantOp>(defOp))
          continue;

        worklist.push(std::make_pair(defOp, depth + 1));
      }
    }
  });

  return id - startId;
}

namespace {

struct AnnotateSplittablePass
    : public mlir::PassWrapper<AnnotateSplittablePass,
                               OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    int numSplittable = 0;
    m.walk([&](FuncOp f) {
      numSplittable += annotateSplittable(f, b, numSplittable);
    });
  }
};

} // namespace

void polymer::registerScopStmtOptPasses() {
  PassRegistration<AnnotateSplittablePass>(
      "annotate-splittable",
      "Give operations that are splittable in its expression tree.");
  PassRegistration<ScopStmtSplitPass>(
      "scop-stmt-split", "Split a given set of splittable operations.");
}

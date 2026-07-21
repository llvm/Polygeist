#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "remove-scf-iter-args"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace scf;
using namespace affine;

// ============================================================================
// Shared Helper Functions for Iter Args Removal
// ============================================================================

namespace RemoveIterArgsHelpers {

/// Check if a value is loop-invariant w.r.t. the given loop operation
bool isLoopInvariant(Value val, Operation *loopOp) {
  // Check if the value is defined outside the loop
  if (auto defOp = val.getDefiningOp()) {
    return !loopOp->isAncestor(defOp);
  }
  // Block arguments from parent regions are invariant
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    return blockArg.getOwner()->getParentOp() != loopOp;
  }
  return true;
}

bool isLoopCarriedRegionArg(Value val) {
  auto blockArg = dyn_cast<BlockArgument>(val);
  if (!blockArg)
    return false;

  Operation *parentOp = blockArg.getOwner()->getParentOp();
  if (!isa<scf::ForOp, affine::AffineForOp>(parentOp))
    return false;

  // In both scf.for and affine.for regions, argument 0 is the induction
  // variable and subsequent block arguments are loop-carried values.
  return blockArg.getArgNumber() > 0;
}

/// Result of use chain analysis
struct UseChainAnalysis {
  SmallVector<std::pair<Operation *, Value>, 4>
      opsChain; // (op, invariant_operand)
  Operation *storeOp = nullptr;
  Operation *initLoad = nullptr;
  bool succeeded = false;

  /// Analyze the use chain of a loop result to find transformation
  /// opportunities Returns true if the chain ends in a store and can be
  /// transformed
  template <typename LoadOpType, typename StoreOpType>
  bool analyze(Value loopResult, Value yieldedValue, Operation *loopOp) {
    LLVM_DEBUG(llvm::dbgs() << "  Traversing use chain to find store...\n");

    // Check if yield is an addition (required for distributivity
    // transformations)
    Operation *yieldedAddOp = yieldedValue.getDefiningOp();
    bool yieldIsAddition = yieldedAddOp && (isa<arith::AddFOp>(yieldedAddOp) ||
                                            isa<arith::AddIOp>(yieldedAddOp));
    LLVM_DEBUG(llvm::dbgs() << "  Yielded operation is addition: "
                            << (yieldIsAddition ? "YES" : "NO") << "\n");

    Value currentValue = loopResult;
    int traverseLimit = 10; // Prevent infinite loops

    while (currentValue.hasOneUse() && traverseLimit-- > 0) {
      Operation *user = *currentValue.getUsers().begin();
      LLVM_DEBUG(llvm::dbgs() << "    Checking user: " << *user << "\n");

      // Check if we reached a store
      if (isa<StoreOpType>(user)) {
        storeOp = user;
        LLVM_DEBUG(llvm::dbgs() << "    ✓ Found store!\n");
        succeeded = true;
        return true;
      }

      // Check if this is a multiply that can distribute over addition
      if (isa<arith::MulFOp>(user) || isa<arith::MulIOp>(user)) {
        if (!yieldIsAddition) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    ✗ Cannot pull multiply: yield is not addition\n");
          return false;
        }

        // Check that one operand is the loop result and the other is
        // loop-invariant
        Value lhs = user->getOperand(0);
        Value rhs = user->getOperand(1);
        Value invariantOp;

        if (lhs == currentValue && isLoopInvariant(rhs, loopOp)) {
          invariantOp = rhs;
        } else if (rhs == currentValue && isLoopInvariant(lhs, loopOp)) {
          invariantOp = lhs;
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "    ✗ Multiply operands don't match pattern\n");
          return false;
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "    ✓ Can pull multiply into loop (distributivity)\n");
        opsChain.push_back({user, invariantOp});
        currentValue = user->getResult(0);
        continue;
      }

      // Check if this is an addition with a loop-invariant load
      if (isa<arith::AddFOp>(user) || isa<arith::AddIOp>(user)) {
        if (!yieldIsAddition) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    ✗ Cannot merge addition: yield is not addition\n");
          return false;
        }

        // Get the other operand (not the loop result)
        Value lhs = user->getOperand(0);
        Value rhs = user->getOperand(1);
        Value otherOperand = (lhs == currentValue) ? rhs : lhs;

        // Check if it's a loop-invariant load
        if (auto loadOp = dyn_cast<LoadOpType>(otherOperand.getDefiningOp())) {
          // Check all load operands are loop-invariant
          bool allInvariant = true;
          for (Value operand : loadOp->getOperands()) {
            // Skip memref itself, check indices
            if (operand == loadOp->getOperand(0))
              continue;
            if (!isLoopInvariant(operand, loopOp)) {
              allInvariant = false;
              break;
            }
          }

          if (allInvariant) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "    ✓ Found loop-invariant load, will merge into init\n");
            initLoad = loadOp;
            opsChain.push_back({user, otherOperand});
            currentValue = user->getResult(0);
            continue;
          }
        }

        LLVM_DEBUG(llvm::dbgs() << "    ✗ Addition doesn't match pattern\n");
        return false;
      }

      // Unknown operation
      LLVM_DEBUG(llvm::dbgs() << "    ✗ Unknown operation type: "
                              << user->getName() << "\n");
      return false;
    }

    LLVM_DEBUG(llvm::dbgs() << "  ✗ Could not find store in use chain\n");
    return false;
  }
};

/// Pull operations from outside the loop into the loop body
/// Returns the final accumulator value to be stored
LogicalResult pullOperationsIntoLoop(
    IRMapping &mapper, SmallVectorImpl<std::pair<Operation *, Value>> &opsChain,
    Value yieldedValue, Value accumulatorValue, Operation *loopOp,
    PatternRewriter &rewriter, Location loc, Value &outFinalAccum) {

  LLVM_DEBUG(llvm::dbgs() << "  Pulling operations from outside into loop\n");

  // Get the yielded value (mapped to new loop)
  Value currentAccum = mapper.lookupOrDefault(yieldedValue);
  if (!currentAccum)
    currentAccum = yieldedValue;

  // Get the new loop body
  Block *newBody = nullptr;
  if (auto affineFor = dyn_cast<affine::AffineForOp>(loopOp)) {
    newBody = affineFor.getBody();
  } else if (auto scfFor = dyn_cast<scf::ForOp>(loopOp)) {
    newBody = scfFor.getBody();
  } else {
    return failure();
  }

  // Pull multiply operations into the loop
  for (auto &[op, invariantOp] : opsChain) {
    if (isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Pulling multiply into loop: " << *op << "\n");

      // Find the addition operation that produces currentAccum
      Operation *addOpDef = currentAccum.getDefiningOp();
      if (addOpDef &&
          (isa<arith::AddFOp>(addOpDef) || isa<arith::AddIOp>(addOpDef))) {
        auto addOp = addOpDef;

        // Find which operand is the accumulator vs the value being added
        Value lhs = addOp->getOperand(0);
        Value rhs = addOp->getOperand(1);

        // Use the explicitly mapped loop-carried value.  A load-based
        // heuristic is ambiguous when the reduction term is itself a load.
        Value valueToScale;
        Value accumValue;
        if (lhs == accumulatorValue) {
          accumValue = lhs;
          valueToScale = rhs;
        } else if (rhs == accumulatorValue) {
          accumValue = rhs;
          valueToScale = lhs;
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "      Could not identify accumulator operand\n");
          return failure();
        }

        // Create new multiply (use same type as original)
        rewriter.setInsertionPoint(addOp);
        Value newMulResult;
        if (isa<arith::MulFOp>(op)) {
          auto newMul =
              rewriter.create<arith::MulFOp>(loc, invariantOp, valueToScale);
          newMulResult = newMul.getResult();
          LLVM_DEBUG(llvm::dbgs() << "      Created: " << newMul << "\n");
        } else {
          auto newMul =
              rewriter.create<arith::MulIOp>(loc, invariantOp, valueToScale);
          newMulResult = newMul.getResult();
          LLVM_DEBUG(llvm::dbgs() << "      Created: " << newMul << "\n");
        }

        // Create new addition (use same type as original)
        Value newAddResult;
        if (isa<arith::AddFOp>(addOp)) {
          auto newAdd =
              rewriter.create<arith::AddFOp>(loc, accumValue, newMulResult);
          newAddResult = newAdd.getResult();
          LLVM_DEBUG(llvm::dbgs() << "      Created: " << newAdd << "\n");
        } else {
          auto newAdd =
              rewriter.create<arith::AddIOp>(loc, accumValue, newMulResult);
          newAddResult = newAdd.getResult();
          LLVM_DEBUG(llvm::dbgs() << "      Created: " << newAdd << "\n");
        }

        // Replace the old add
        rewriter.replaceOp(addOp, newAddResult);
        currentAccum = newAddResult;
      }
    }
  }

  outFinalAccum = currentAccum;
  return success();
}

} // namespace RemoveIterArgsHelpers

// ============================================================================
// Pattern Implementations
// ============================================================================

struct RemoveSCFIterArgs : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    using namespace RemoveIterArgsHelpers;

    LLVM_DEBUG(llvm::dbgs()
               << "\n=== RemoveSCFIterArgs::matchAndRewrite ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Processing scf.for loop:\n" << forOp << "\n");

    if (!forOp.getRegion().hasOneBlock()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "REJECTED: Loop doesn't have exactly one block\n");
      return failure();
    }

    unsigned numIterArgs = forOp.getNumRegionIterArgs();
    LLVM_DEBUG(llvm::dbgs() << "Number of iter_args: " << numIterArgs << "\n");

    if (numIterArgs == 0) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: No iter_args to remove\n");
      return failure();
    }

    // This pattern's single-iter_arg incremental rewrite produces an
    // ill-formed terminator when the new loop still has iter_args left.
    // Defer multi-iter_arg loops to the alloca fallback.
    if (numIterArgs > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "REJECTED: numIterArgs > 1 — defer to alloca fallback\n");
      return failure();
    }

    // For now, process only the last iter_arg (like Affine version)
    LLVM_DEBUG(llvm::dbgs() << "Processing last iter_arg (index "
                            << (numIterArgs - 1) << ")\n");

    auto loc = forOp->getLoc();
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

    auto ba = forOp.getRegionIterArgs()[numIterArgs - 1];
    auto init = forOp.getInits()[numIterArgs - 1];
    auto lastOp = yieldOp->getOperand(numIterArgs - 1);

    LLVM_DEBUG(llvm::dbgs() << "  iter_arg type: " << ba.getType() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  yielded value: " << lastOp << "\n");

    auto result = forOp.getResult(numIterArgs - 1);
    LLVM_DEBUG(llvm::dbgs()
               << "  Loop result has "
               << std::distance(result.user_begin(), result.user_end())
               << " use(s)\n");

    if (!result.hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs() << "  ✗ Result has multiple uses or no uses\n");
      for (auto user : result.getUsers()) {
        LLVM_DEBUG(llvm::dbgs() << "    User: " << *user << "\n");
      }
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  Result has exactly one use\n");

    // Use shared helper to analyze use chain
    UseChainAnalysis analysis;
    if (!analysis.analyze<memref::LoadOp, memref::StoreOp>(
            result, lastOp, forOp.getOperation())) {
      LLVM_DEBUG(llvm::dbgs() << "  ✗ Use chain analysis failed\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  ✓ Successfully traced to store!\n");
    LLVM_DEBUG(llvm::dbgs() << "  Operations in chain: "
                            << analysis.opsChain.size() << "\n");

    auto storeOp = cast<memref::StoreOp>(analysis.storeOp);
    auto initLoad =
        analysis.initLoad ? cast<memref::LoadOp>(analysis.initLoad) : nullptr;

    // Adjust initialization if we have a loop-invariant load
    Value newInit = init;
    if (initLoad) {
      LLVM_DEBUG(llvm::dbgs() << "  Using loop-invariant load as init\n");
      newInit = initLoad.getResult();
    }

    LLVM_DEBUG(llvm::dbgs() << "  Creating new scf.for with "
                            << (numIterArgs - 1) << " iter_args...\n");

    // Prepare new iter_args (drop the last one we're removing)
    SmallVector<Value> newIterArgs(forOp.getInits());
    if (!newIterArgs.empty()) {
      newIterArgs[numIterArgs - 1] = newInit; // Use the adjusted init
      newIterArgs.pop_back();                 // Remove last iter_arg
    }

    // For direct overwrite reductions (`sum = init; ...; out = sum`), seed the
    // destination with the original iter_arg init before the rewritten loop.
    // If the analysis found a loop-invariant load, the source was already an
    // update form (`out = old_out + ...`), so keep the existing output value.
    if (!initLoad && !isLoopCarriedRegionArg(init)) {
      rewriter.setInsertionPoint(forOp);
      auto initStore = rewriter.create<memref::StoreOp>(
          loc, init, storeOp.getMemref(), storeOp.getIndices());
      LLVM_DEBUG(llvm::dbgs() << "    Created memref.store for reduction seed: "
                              << initStore << "\n");
    }

    rewriter.setInsertionPoint(forOp);

    // Create new loop with correct signature (fewer iter_args)
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep(), newIterArgs);

    LLVM_DEBUG(llvm::dbgs() << "  Cloning loop body using IRMapping\n");

    // Create IRMapping for value remapping
    IRMapping mapper;

    // Map the induction variable
    mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());

    // Map the iter_args (except the last one we're removing)
    for (unsigned i = 0; i < numIterArgs - 1; i++) {
      mapper.map(forOp.getRegionIterArgs()[i], newForOp.getRegionIterArgs()[i]);
    }

    // Create load at the beginning that will replace the iter_arg
    Block *oldBody = forOp.getBody();
    Block *newBody = newForOp.getBody();
    rewriter.setInsertionPointToStart(newBody);

    auto memrefLoad = rewriter.create<memref::LoadOp>(loc, storeOp.getMemref(),
                                                      storeOp.getIndices());
    LLVM_DEBUG(llvm::dbgs() << "    Created memref.load at loop start: "
                            << memrefLoad << "\n");

    // Map the old iter_arg to the loaded value
    mapper.map(ba, memrefLoad.getResult());

    // Clone all operations - they'll automatically use the mapped load value
    for (Operation &op : oldBody->without_terminator()) {
      rewriter.clone(op, mapper);
    }

    // Use shared helper to pull operations into loop
    Value finalAccum;
    if (failed(pullOperationsIntoLoop(
            mapper, analysis.opsChain, lastOp, memrefLoad.getResult(),
            newForOp.getOperation(), rewriter, loc, finalAccum))) {
      LLVM_DEBUG(llvm::dbgs() << "  ✗ Failed to pull operations into loop\n");
      rewriter.eraseOp(newForOp);
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  Creating store at end of loop\n");

    // Create store before the yield
    rewriter.setInsertionPoint(newBody->getTerminator());
    auto newStore = rewriter.create<memref::StoreOp>(
        loc, finalAccum, storeOp.getMemref(), storeOp.getIndices());
    LLVM_DEBUG(llvm::dbgs() << "    Created memref.store before yield: "
                            << newStore << "\n");

    LLVM_DEBUG(llvm::dbgs() << "  Fixing yield operation\n");

    // Create new yield with mapped operands (excluding the iter_arg we removed)
    SmallVector<Value> newYieldOperands;
    for (unsigned i = 0; i < numIterArgs - 1; i++) {
      Value oldOperand = yieldOp.getOperand(i);
      Value newOperand = mapper.lookupOrDefault(oldOperand);
      if (!newOperand)
        newOperand = oldOperand;
      newYieldOperands.push_back(newOperand);
    }

    rewriter.setInsertionPoint(newBody->getTerminator());
    rewriter.replaceOpWithNewOp<scf::YieldOp>(newBody->getTerminator(),
                                              newYieldOperands);

    LLVM_DEBUG(llvm::dbgs() << "  Erasing old operations outside loop\n");

    // Erase the external store
    LLVM_DEBUG(llvm::dbgs() << "    Erasing store: " << *storeOp << "\n");
    rewriter.eraseOp(storeOp);

    // Erase operations in reverse order
    for (auto it = analysis.opsChain.rbegin(); it != analysis.opsChain.rend();
         ++it) {
      auto &[op, _] = *it;
      LLVM_DEBUG(llvm::dbgs() << "    Erasing: " << *op << "\n");
      rewriter.eraseOp(op);
    }

    // Erase the init load if it exists
    if (initLoad) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Erasing init load: " << *initLoad << "\n");
      rewriter.eraseOp(initLoad);
    }

    LLVM_DEBUG(llvm::dbgs()
               << "  Replacing uses of old loop results with new loop\n");
    for (unsigned i = 0; i < numIterArgs - 1; i++) {
      rewriter.replaceAllUsesWith(forOp.getResult(i), newForOp.getResult(i));
    }

    LLVM_DEBUG(llvm::dbgs() << "  Erasing old loop\n");
    rewriter.eraseOp(forOp);
    LLVM_DEBUG(llvm::dbgs() << "=== RemoveSCFIterArgs SUCCESS ===\n\n");
    return success();
  }
};

// General Case(TODO):
// ALGo:
//  1. Create an alloca(stack) variable
//     How to know it's dims? It should be based on number of reduction
//     loops
//  2. Initialize it with init value just outside the for loop if init
//  value is non-zero
//  3. memref.load that value in the for loop
//  4. Replace all the uses of the iter_arg with the loaded value
//  5. Add a memref.store for the value to be yielded
//  6. Replace all uses of for-loops yielded value with a single inserted
//  memref.load
// Special case:
// ALGo:
// Optimize away memref.store and memref.load, if the only users of
// memref.load are memref.store (can use affine-scalrep pass for that ? No
// it does store to load forwarding) What we need is forwarding of local
// store to final store and deleting the intermediate alloca created. This
// is only possible if the user of alloca is a storeOp.
//  1. Identify the single store of the for loop result
//  2. Initialize it with iter arg init, outside the for loop. (TODO)
//  3. Do a load from the memref
//  4. move the store to memref inside the loop.

struct RemoveAffineIterArgs : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    using namespace RemoveIterArgsHelpers;

    LLVM_DEBUG(llvm::dbgs()
               << "\n=== RemoveAffineIterArgs::matchAndRewrite ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Processing affine.for loop:\n"
                            << forOp << "\n");

    rewriter.setInsertionPoint(forOp);

    unsigned numIterArgs = forOp.getNumRegionIterArgs();
    LLVM_DEBUG(llvm::dbgs() << "Number of iter_args: " << numIterArgs << "\n");

    if (numIterArgs == 0) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: No iter_args to remove\n");
      return failure();
    }

    // This pattern's single-iter_arg incremental rewrite produces an
    // ill-formed terminator when the new loop still has iter_args left.
    // Defer multi-iter_arg loops to the alloca fallback.
    if (numIterArgs > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "REJECTED: numIterArgs > 1 — defer to alloca fallback\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Processing last iter_arg (index "
                            << (numIterArgs - 1) << ")\n");

    auto loc = forOp->getLoc();
    auto yieldOp =
        cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());

    auto ba = forOp.getRegionIterArgs()[numIterArgs - 1];
    auto init = forOp.getInits()[numIterArgs - 1];
    auto lastOp = yieldOp->getOperand(numIterArgs - 1);

    LLVM_DEBUG(llvm::dbgs() << "  iter_arg type: " << ba.getType() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  yielded value: " << lastOp << "\n");

    auto result = forOp.getResult(numIterArgs - 1);
    LLVM_DEBUG(llvm::dbgs()
               << "  Loop result has "
               << std::distance(result.user_begin(), result.user_end())
               << " use(s)\n");

    if (!result.hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs() << "  ✗ Result has multiple uses or no uses\n");
      for (auto user : result.getUsers()) {
        LLVM_DEBUG(llvm::dbgs() << "    User: " << *user << "\n");
      }
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  Result has exactly one use\n");

    // Use shared helper to analyze use chain
    UseChainAnalysis analysis;
    if (!analysis.analyze<affine::AffineLoadOp, affine::AffineStoreOp>(
            result, lastOp, forOp.getOperation())) {
      LLVM_DEBUG(llvm::dbgs() << "  ✗ Use chain analysis failed\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  ✓ Successfully traced to store!\n");
    LLVM_DEBUG(llvm::dbgs() << "  Operations in chain: "
                            << analysis.opsChain.size() << "\n");

    auto storeOp = cast<affine::AffineStoreOp>(analysis.storeOp);
    auto initLoad = analysis.initLoad
                        ? cast<affine::AffineLoadOp>(analysis.initLoad)
                        : nullptr;

    // Adjust initialization if we have a loop-invariant load
    Value newInit = init;
    if (initLoad) {
      LLVM_DEBUG(llvm::dbgs() << "  Using loop-invariant load as init\n");
      newInit = initLoad.getResult();
    }

    LLVM_DEBUG(llvm::dbgs() << "  Creating new affine.for with "
                            << (numIterArgs - 1) << " iter_args...\n");

    // Prepare new iter_args (drop the last one we're removing)
    SmallVector<Value> newIterArgs(forOp.getInits());
    if (!newIterArgs.empty()) {
      newIterArgs[numIterArgs - 1] = newInit; // Use the adjusted init
      newIterArgs.pop_back();                 // Remove last iter_arg
    }

    // For direct overwrite reductions (`sum = init; ...; out = sum`), seed the
    // destination with the original iter_arg init before the rewritten loop.
    // If the analysis found a loop-invariant load, the source was already an
    // update form (`out = old_out + ...`), so keep the existing output value.
    if (!initLoad && !isLoopCarriedRegionArg(init)) {
      rewriter.setInsertionPoint(forOp);
      auto initStore = rewriter.create<affine::AffineStoreOp>(
          loc, init, storeOp.getMemref(), storeOp.getMap(),
          storeOp.getMapOperands());
      LLVM_DEBUG(llvm::dbgs() << "    Created affine.store for reduction seed: "
                              << initStore << "\n");
    }

    rewriter.setInsertionPoint(forOp);

    // Create new loop with correct signature (fewer iter_args)
    auto newForOp = rewriter.create<affine::AffineForOp>(
        loc, forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
        forOp.getStep(), newIterArgs);

    LLVM_DEBUG(llvm::dbgs() << "  Cloning loop body using IRMapping\n");

    // Create IRMapping for value remapping
    IRMapping mapper;

    // Map the induction variable
    mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());

    // Map the iter_args (except the last one we're removing)
    for (unsigned i = 0; i < numIterArgs - 1; i++) {
      mapper.map(forOp.getRegionIterArgs()[i], newForOp.getRegionIterArgs()[i]);
    }

    // Create load at the beginning that will replace the iter_arg
    Block *oldBody = forOp.getBody();
    Block *newBody = newForOp.getBody();
    rewriter.setInsertionPointToStart(newBody);

    auto memrefLoad = rewriter.create<affine::AffineLoadOp>(
        loc, storeOp.getMemref(), storeOp.getMap(), storeOp.getMapOperands());
    LLVM_DEBUG(llvm::dbgs() << "    Created affine.load at loop start: "
                            << memrefLoad << "\n");

    // Map the old iter_arg to the loaded value
    mapper.map(ba, memrefLoad.getResult());

    // Clone all operations - they'll automatically use the mapped load value
    for (Operation &op : oldBody->without_terminator()) {
      rewriter.clone(op, mapper);
    }

    // Use shared helper to pull operations into loop
    Value finalAccum;
    Value oldYieldedValue = yieldOp.getOperand(numIterArgs - 1);
    if (failed(pullOperationsIntoLoop(
            mapper, analysis.opsChain, oldYieldedValue, memrefLoad.getResult(),
            newForOp.getOperation(), rewriter, loc, finalAccum))) {
      LLVM_DEBUG(llvm::dbgs() << "  ✗ Failed to pull operations into loop\n");
      rewriter.eraseOp(newForOp);
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  Creating store at end of loop\n");

    // Create store before the yield (load was already created and mapped
    // earlier)
    rewriter.setInsertionPoint(newBody->getTerminator());
    auto newStore = rewriter.create<affine::AffineStoreOp>(
        loc, finalAccum, storeOp.getMemref(), storeOp.getMap(),
        storeOp.getMapOperands());
    LLVM_DEBUG(llvm::dbgs() << "    Created affine.store before yield: "
                            << newStore << "\n");

    LLVM_DEBUG(llvm::dbgs() << "  Fixing yield operation\n");

    // Create new yield with mapped operands (excluding the iter_arg we removed)
    SmallVector<Value> newYieldOperands;
    for (unsigned i = 0; i < numIterArgs - 1; i++) {
      Value oldOperand = yieldOp.getOperand(i);
      Value newOperand = mapper.lookupOrDefault(oldOperand);
      if (!newOperand)
        newOperand = oldOperand;
      newYieldOperands.push_back(newOperand);
    }

    rewriter.setInsertionPoint(newBody->getTerminator());
    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(newBody->getTerminator(),
                                                       newYieldOperands);

    LLVM_DEBUG(llvm::dbgs() << "  Erasing old operations outside loop\n");

    // Erase the external store
    LLVM_DEBUG(llvm::dbgs() << "    Erasing store: " << *storeOp << "\n");
    rewriter.eraseOp(storeOp);

    // Erase operations in reverse order
    for (auto it = analysis.opsChain.rbegin(); it != analysis.opsChain.rend();
         ++it) {
      auto &[op, _] = *it;
      LLVM_DEBUG(llvm::dbgs() << "    Erasing: " << *op << "\n");
      rewriter.eraseOp(op);
    }

    // Erase the init load if it exists
    if (initLoad) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Erasing init load: " << *initLoad << "\n");
      rewriter.eraseOp(initLoad);
    }

    LLVM_DEBUG(llvm::dbgs()
               << "  Replacing uses of old loop results with new loop\n");
    for (unsigned i = 0; i < numIterArgs - 1; i++) {
      rewriter.replaceAllUsesWith(forOp.getResult(i), newForOp.getResult(i));
    }

    LLVM_DEBUG(llvm::dbgs() << "  Erasing old loop\n");
    rewriter.eraseOp(forOp);
    LLVM_DEBUG(llvm::dbgs() << "=== RemoveAffineIterArgs SUCCESS ===\n\n");
    return success();
  }
};

// ============================================================================
// Universal alloca-based materialization (consumer-blind fallback)
// ============================================================================
//
// This pattern unconditionally converts every iter_arg of an affine.for into a
// 0-D memref slot:
//
//   %slot_i = memref.alloca() : memref<T_i>
//   affine.store %init_i, %slot_i[]
//   affine.for %iv = lb to ub {                 // no iter_args
//     %acc_i = affine.load %slot_i[]            // replaces the iter_arg
//     ... body, with iter_arg_i -> %acc_i ...
//     affine.store %yielded_i, %slot_i[]        // replaces yield operand i
//   }
//   %final_i = affine.load %slot_i[]
//   // RAUW old loop result #i -> %final_i  (handles return / call / store /
//   //                                       cmp / loop bound / multi-use ...)
//
// Registered at lower benefit than RemoveAffineIterArgs, so the existing
// store-fusion fast path is tried first; this pattern catches everything else.

struct MaterializeAffineIterArgsViaAlloca
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "\n=== MaterializeAffineIterArgsViaAlloca ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Processing affine.for:\n" << forOp << "\n");

    unsigned numIterArgs = forOp.getNumRegionIterArgs();
    if (numIterArgs == 0) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: No iter_args\n");
      return failure();
    }
    if (!forOp.getRegion().hasOneBlock()) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: Loop body has != 1 block\n");
      return failure();
    }

    auto loc = forOp.getLoc();
    auto yieldOp =
        cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());

    // Step 1 & 2: alloca + init store for each iter_arg, before the loop.
    rewriter.setInsertionPoint(forOp);
    SmallVector<Value> slots;
    slots.reserve(numIterArgs);
    for (unsigned i = 0; i < numIterArgs; ++i) {
      Type t = forOp.getRegionIterArgs()[i].getType();
      auto slot =
          rewriter.create<memref::AllocaOp>(loc, MemRefType::get({}, t));
      slots.push_back(slot.getResult());
      rewriter.create<affine::AffineStoreOp>(loc, forOp.getInits()[i],
                                             slot.getResult(), ValueRange{});
    }

    // Step 3: new affine.for with the same bounds but no iter_args.
    auto newForOp = rewriter.create<affine::AffineForOp>(
        loc, forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
        forOp.getStep(), /*iterArgs=*/ValueRange{});

    Block *newBody = newForOp.getBody();
    Block *oldBody = forOp.getBody();

    IRMapping mapper;
    mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());

    // Step 4a: at the top of the new body, load each slot and map the
    // corresponding old iter_arg block-arg onto the loaded SSA value.
    rewriter.setInsertionPointToStart(newBody);
    for (unsigned i = 0; i < numIterArgs; ++i) {
      auto load =
          rewriter.create<affine::AffineLoadOp>(loc, slots[i], ValueRange{});
      mapper.map(forOp.getRegionIterArgs()[i], load.getResult());
    }

    // Step 4b: clone every body op (the IRMapping rewires iter_arg uses
    // to the loaded values). The auto-inserted affine.yield in newBody
    // stays at the end; we insert before it.
    for (Operation &op : oldBody->without_terminator()) {
      rewriter.clone(op, mapper);
    }

    // Step 4c: store the (mapped) yielded values back to their slots,
    // just before the new loop's terminator.
    rewriter.setInsertionPoint(newBody->getTerminator());
    for (unsigned i = 0; i < numIterArgs; ++i) {
      Value mappedYielded = mapper.lookupOrDefault(yieldOp.getOperand(i));
      rewriter.create<affine::AffineStoreOp>(loc, mappedYielded, slots[i],
                                             ValueRange{});
    }

    // Step 5: after the loop, load each slot and RAUW the corresponding
    // old loop result.
    rewriter.setInsertionPointAfter(newForOp);
    for (unsigned i = 0; i < numIterArgs; ++i) {
      auto finalLoad =
          rewriter.create<affine::AffineLoadOp>(loc, slots[i], ValueRange{});
      rewriter.replaceAllUsesWith(forOp.getResult(i), finalLoad.getResult());
    }

    rewriter.eraseOp(forOp);
    LLVM_DEBUG(llvm::dbgs()
               << "=== MaterializeAffineIterArgsViaAlloca SUCCESS ===\n\n");
    return success();
  }
};

// Remove a store that only writes an unchanged value back to the exact
// location it was loaded from:
//
//   %old = affine.load %A[...]
//   <read-only / side-effect-free operations>
//   affine.store %old, %A[...]
//
// Top-down iter_arg removal intentionally exposes this shape when an outer
// reduction result establishes the final destination and an inner reduction
// carries that destination through unchanged.  Erasing the round trip leaves
// one canonical load/update/store at the loop level that actually updates the
// accumulator, allowing AffineForOpRaising to compose all reduction loops.
//
// Keep this conservative: both accesses must be in the same block, use the
// identical affine address, and no operation between them may write memory or
// have unknown/recursive effects.  Reads are harmless because the store writes
// back the value already present at that location.
struct EraseUnchangedAffineLoadStore
    : public OpRewritePattern<affine::AffineStoreOp> {
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineStoreOp store,
                                PatternRewriter &rewriter) const override {
    auto load = store.getValueToStore().getDefiningOp<affine::AffineLoadOp>();
    if (!load || load->getBlock() != store->getBlock() ||
        !load->isBeforeInBlock(store))
      return failure();

    if (load.getMemref() != store.getMemref() ||
        load.getAffineMap() != store.getAffineMap() ||
        load.getMapOperands() != store.getMapOperands())
      return failure();

    for (Operation *op = load->getNextNode(); op && op != store;
         op = op->getNextNode()) {
      if (isMemoryEffectFree(op))
        continue;

      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      if (!effects || op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        return failure();

      SmallVector<MemoryEffects::EffectInstance> effectInstances;
      effects.getEffects(effectInstances);
      if (llvm::any_of(effectInstances, [](const auto &effect) {
            return !isa<MemoryEffects::Read>(effect.getEffect());
          }))
        return failure();
    }

    rewriter.eraseOp(store);
    if (load->use_empty())
      rewriter.eraseOp(load);
    return success();
  }
};

namespace {
struct RemoveIterArgs : public RemoveIterArgsBase<RemoveIterArgs> {

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "\n\n");
    LLVM_DEBUG(llvm::dbgs()
               << "===================================================\n");
    LLVM_DEBUG(llvm::dbgs() << "=== STARTING RemoveIterArgs PASS ===\n");
    LLVM_DEBUG(llvm::dbgs()
               << "===================================================\n");

    GreedyRewriteConfig config;
    // Establish the outermost accumulator destination before visiting nested
    // iter_args.  Inner loops can then be fused into that same location rather
    // than materializing one 0-D alloca per nesting level.
    config.useTopDownTraversal = true;
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    // Fast-path patterns (store-fusion): higher benefit, tried first.
    patterns.add<RemoveSCFIterArgs>(context, /*benefit=*/2);
    patterns.add<RemoveAffineIterArgs>(context, /*benefit=*/2);
    // Universal fallback (alloca materialization): lower benefit.
    patterns.add<MaterializeAffineIterArgsViaAlloca>(context, /*benefit=*/1);
    // Cleanup for the unchanged load/store round trips exposed by top-down
    // destination propagation.
    patterns.add<EraseUnchangedAffineLoadStore>(context, /*benefit=*/1);

    LLVM_DEBUG(llvm::dbgs()
               << "Registered patterns: RemoveSCFIterArgs, "
                  "RemoveAffineIterArgs, MaterializeAffineIterArgsViaAlloca\n");
    LLVM_DEBUG(llvm::dbgs() << "Applying patterns greedily...\n\n");

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      LLVM_DEBUG(llvm::dbgs() << "\n!!! RemoveIterArgs PASS FAILED !!!\n");
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "===================================================\n");
    LLVM_DEBUG(llvm::dbgs()
               << "=== RemoveIterArgs PASS COMPLETED SUCCESSFULLY ===\n");
    LLVM_DEBUG(llvm::dbgs()
               << "===================================================\n\n");
  }
};
} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRemoveIterArgsPass() {
  return std::make_unique<RemoveIterArgs>();
}
} // namespace polygeist
} // namespace mlir

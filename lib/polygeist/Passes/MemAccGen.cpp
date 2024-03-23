#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "MemAcc/Ops.h"
#include "MemAcc/Dialect.h"

#define PRINT(x) llvm::errs() << x << "\n"

// Use LLVM's data structures for convenience and performance
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "memory-access-generation"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace mlir::MemAcc;
// Define the data structures at the beginning of your pass

namespace {
struct MemAccGenPass : public MemAccGenBase<MemAccGenPass> {
  void runOnOperation() override;
};
} // end namespace.

namespace {

llvm::SmallPtrSet<Operation *, 16> deepestLoads;
llvm::DenseMap<Operation*, llvm::SmallPtrSet<Operation*, 16>> loadOpToIndirectUses;
llvm::DenseMap<Operation*, llvm::SmallVector<Operation*, 16>> loadOpToIndirectChain;

static void postProcessDeepestLoads(){
  for (auto o : deepestLoads){
    for (auto i : loadOpToIndirectUses[o]){
      if (deepestLoads.count(i) > 0){
        deepestLoads.erase(i);
      }
    }
  }

}

static void updateIndirectUseChain(Operation *curr, SmallVector<Operation *, 16>& indirectUseChain) {
  if (loadOpToIndirectUses.count(curr) == 0){
    return;
  }
  for (auto o : loadOpToIndirectUses[curr]){
        indirectUseChain.push_back(o);
        updateIndirectUseChain(o, indirectUseChain);
    }
}

static SmallVector<Operation *, 16> getIndirectLoadUsers(Operation *op) {
  SmallVector<Operation *, 16> indirectUseChain;
  Operation* curr = op;
  updateIndirectUseChain(curr, indirectUseChain);
  
  return indirectUseChain;
}

// Utility function to create a MemAcc::YieldOp
static void createMemAccYieldOp(PatternRewriter &rewriter, mlir::Location loc) {

  // Specify empty result types and operands for the yield operation
  mlir::TypeRange resultTypes; // No return types
  mlir::ValueRange operands;   // No operands
  llvm::ArrayRef<mlir::NamedAttribute> attributes; // No attributes

  // Finally, build and insert the operation into the IR
  rewriter.create<MemAcc::YieldOp>(loc, resultTypes, operands, attributes);
}

// Define the rewrite pattern
struct StoreOpToGenericStoreOpPattern : public OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp storeOp, PatternRewriter &rewriter) const override {
    // Check if the storeOp is contained within an affine.for operation
    if (!storeOp->getParentOfType<AffineForOp>()) {
      return failure();
    }
    if (storeOp->getParentOfType<GenericStoreOp>()) {
      return failure();
    }

    // Create the new MemAcc::GenericStoreOp, wrapping the original store operation
    // Assuming you have a constructor for GenericStoreOp that takes the original store operation as an argument
    // You might need to adapt this part based on how your GenericStoreOp is defined
    Location loc = storeOp.getLoc();
    auto genericStoreOp = rewriter.create<MemAcc::GenericStoreOp>(loc /* other necessary parameters */);
    
    // Here, you might want to set attributes or otherwise configure the GenericStoreOp
    // For example, setting the indirect_level attribute if needed
    // genericStoreOp.setAttr("indirect_level", rewriter.getI32IntegerAttr(/* value */));

    // Insert the original store operation into the body of the new GenericStoreOp
    // This assumes your GenericStoreOp has a region that can contain the storeOp
    auto &region = genericStoreOp.getBody();
    auto *block = rewriter.createBlock(&region);

    // Remove the original store operation
    storeOp.getOperation()->moveBefore(block, block->end());

    createMemAccYieldOp(rewriter, loc);

    return success();
  }
};

struct LoadOpToGenericLoadOpPattern : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp loadOp, PatternRewriter &rewriter) const override {
    // Check if the loadOp is contained within an affine.for operation
    if (!loadOp->getParentOfType<AffineForOp>()) {
      return failure();
    }
    if (loadOp->getParentOfType<GenericLoadOp>()) {
      return failure();
    }

    // only consider the deepest loads
    if (deepestLoads.count(loadOp) == 0){
      return failure();
    }
    SmallVector<Type, 4> resultTypes;
    SmallVector<Value, 4> resultVals;
    auto indirectLoadUseChain = loadOpToIndirectChain[loadOp];
    loadOpToIndirectUses[loadOp].insert(loadOp);
    PRINT("Indirect Chain:");
    for (int i = indirectLoadUseChain.size() - 1; i >= 0; i--){
      auto I = indirectLoadUseChain[i];
      PRINT(*I);
      for (auto U : I->getUsers()) {
        PRINT("User: " << *U);
        if (loadOpToIndirectUses[loadOp].count(U) == 0){
          PRINT("External User: " << *U);
          resultTypes.push_back(I->getResult(0).getType());
          break;
        }
      }
    }
    PRINT("\n");
    // Count the number of loads in the indirectLoadUseChain for indirection level
    uint64_t indirectionLevel = 0;
     // Prepare the result types based on the loads that have external users
    for (Operation* op : indirectLoadUseChain) {
      if (isa<memref::LoadOp>(op) || isa<affine::AffineLoadOp>(op)) {
        indirectionLevel++;
      }
    }
    auto indirectionAttr = IntegerAttr::get(IntegerType::get(rewriter.getContext(), 64), indirectionLevel-1);

    Location loc = loadOp.getLoc();
    // Prepare to create a new MemAcc::GenericLoadOp
    // SmallVector<Type, 4> resultTypes;
    // Keep track of original loads and their replacements to update external users

    

    // Start creating the GenericLoadOp
    auto genericLoadOp = rewriter.create<MemAcc::GenericLoadOp>(loc, resultTypes, indirectionAttr);
    auto &region = genericLoadOp.getBody();

    // Create a block inside the GenericLoadOp's region
    auto *block = rewriter.createBlock(&region);

    // Move the operations from the indirectLoadUseChain into the block
    for (int i = indirectLoadUseChain.size() - 1; i >= 0; i--) {
      auto clonedOp = rewriter.clone(*indirectLoadUseChain[i]);
      indirectLoadUseChain[i]->getResult(0).replaceAllUsesWith(clonedOp->getResult(0));
      rewriter.eraseOp(indirectLoadUseChain[i]);
    }
    
    for (auto& I : *block) {
      bool hasExternalUses = false;
      for (auto U : I.getUsers()) {
        if (block != U->getBlock()) {
          hasExternalUses = true;
          break;
        }
      }
      if (hasExternalUses) {
        resultVals.push_back(I.getResult(0));
      }
    }

    auto yieldOp = rewriter.create<MemAcc::YieldOp>(loc, resultTypes, resultVals);

    // Update uses of inner-block values
    int idx = 0;
    for (auto& I : *block) {
      bool hasExternalUses = false;
        for (auto U : I.getUsers()) {
          if (block != U->getBlock()){
            for (auto operandIndex = 0; operandIndex < U->getNumOperands(); ++operandIndex) {
              if (U->getOperand(operandIndex) == I.getResult(0)) {
                // Update the operand with a new value
                U->setOperand(operandIndex, genericLoadOp->getResult(idx));
                hasExternalUses = true;
                break; // Break after updating the operand
              }
            } // for
          } // if
        }
        if (hasExternalUses) {
          idx++;
        }
      }
    

    return success();
  }
};

// Modified to populate the mapping
void markIndirectLoadUsers(Operation *op, llvm::SmallPtrSetImpl<Operation *> &visited,
                           Operation *originalLoadOp) {

  if (!op || !visited.insert(op).second)
    return;


  if (isa<memref::LoadOp>(op) || isa<affine::AffineLoadOp>(op) || isa<arith::ArithDialect>(op->getDialect())) {
      loadOpToIndirectUses[originalLoadOp].insert(op);
      loadOpToIndirectChain[originalLoadOp].push_back(op);
    } else {
      return;
    }

    for (auto operand : op->getOperands()) {
      markIndirectLoadUsers(operand.getDefiningOp(), visited, originalLoadOp);
    }

  }

void analyzeLoadOps(Operation *op, llvm::SmallPtrSet<Operation *, 16> &deepestLoads) {
  llvm::SmallPtrSet<Operation *, 16> visited;
  op->walk([&](Operation *currentOp) {
    if (isa<memref::LoadOp>(currentOp) || isa<affine::AffineLoadOp>(currentOp)) {
      visited.clear();
      loadOpToIndirectChain[currentOp].push_back(currentOp);
      // Check all users of the load operation to see if it indirectly contributes to another load
      for (auto operand : currentOp->getOperands()) {
        markIndirectLoadUsers(operand.getDefiningOp(), visited, currentOp);
      }
        deepestLoads.insert(currentOp);
    }
  });
  postProcessDeepestLoads();
}

void MemAccGenPass::runOnOperation() {
  deepestLoads.clear();
  loadOpToIndirectUses.clear();
  loadOpToIndirectChain.clear();
  mlir::MLIRContext* context = getOperation()->getContext();

  analyzeLoadOps(getOperation(), deepestLoads);

  // for (auto& o : loadOpToIndirectUses){
  //   llvm::errs() << "Load: " << *o.first << "\n";
  //   for (auto i : o.second){
  //     llvm::errs() << "Indirect Use: " << *i << "\n";
  //   }
  // }

  for (auto o : deepestLoads){
    llvm::errs() << "Deepest Load: " << *o << "\n";
  }

  for (auto& o : loadOpToIndirectChain){
    llvm::errs() << "Load: " << *o.first << "\n";
    for (auto i : o.second){
      llvm::errs() << "Indirect Chain: " << *i << "\n";
    }
  }
  // context->loadDialect<mlir::MemAcc::MemAccDialect>();
  mlir::RewritePatternSet patterns(context);
  patterns.add<StoreOpToGenericStoreOpPattern>(context);
  patterns.add<LoadOpToGenericLoadOpPattern>(context);
  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config);
}
} // end anonymous namespace

namespace mlir {
namespace polygeist {
    std::unique_ptr<Pass> mlir::polygeist::createMemAccGenPass() {
        return std::make_unique<MemAccGenPass>();
    }
}
} // end namespace mlir::polygeist
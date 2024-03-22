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

#define DEBUG_TYPE "memory-access-generation"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace mlir::MemAcc;

namespace {
struct MemAccGenPass : public MemAccGenBase<MemAccGenPass> {
  void runOnOperation() override;
};
} // end namespace.

namespace {

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
}

void MemAccGenPass::runOnOperation() {
  mlir::MLIRContext* context = getOperation()->getContext();
  // context->loadDialect<mlir::MemAcc::MemAccDialect>();
  mlir::RewritePatternSet patterns(context);
  patterns.add<StoreOpToGenericStoreOpPattern>(context);
  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config);
}

namespace mlir {
namespace polygeist {
    std::unique_ptr<Pass> mlir::polygeist::createMemAccGenPass() {
        return std::make_unique<MemAccGenPass>();
    }
}
} // end namespace mlir::polygeist
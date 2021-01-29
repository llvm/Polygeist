#include "PassDetail.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
//#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

//#include "mlir/Analysis/NumberOfExecutions.h"

#define DEBUG_TYPE "convert-while-to-for"

using namespace mlir;

namespace {
struct ConvertWhileToFor : public SCFWhileToSCFForBase<ConvertWhileToFor> {
  void runOnFunction() override;
};
} // namespace

static bool isLoopInvariant(Value val) {
  if (!val.getDefiningOp<ConstantOp>())
    return false;
  return true;
} 

static void getExitLimitFromCmpIOp(CmpIOp op, scf::WhileOp loop, Value &ub, Value &indVar) {
  auto rhs = op.rhs();
  if (!isLoopInvariant(rhs))
    return;
  CmpIPredicate pred = op.getPredicate();
  if ((pred != CmpIPredicate::slt) && (pred != CmpIPredicate::ult)) 
    return; 
  ub = rhs;
  indVar = op.lhs();
}

static bool isTopLevel(Value val) {
  if (val.cast<BlockArgument>())
    return true;
  return false;
}

static void getLowerBound(Value indVar, scf::WhileOp loop, Value &lb) {
  if (isTopLevel(indVar)) 
    lb = loop.getOperandAtPos(indVar.cast<BlockArgument>().getArgNumber());
  else {
    // FIXME: how to handle here?
    auto op =  indVar.getDefiningOp();
  }
}

static void getStep(Value indVar, scf::WhileOp loop, Value &step) {
  if (isTopLevel(indVar)) {
    loop.after().walk([&](Operation *op) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {  
        Value tailIndVar = yieldOp.getOperand(indVar.cast<BlockArgument>().getArgNumber());
        auto u = tailIndVar.getUses();
        LLVM_DEBUG(llvm::dbgs() << std::distance(u.begin(), u.end()) << "\n");

        Operation *maybeAddI = tailIndVar.getDefiningOp();
        if (auto addIOp = dyn_cast<AddIOp>(maybeAddI))
          step = addIOp.getOperand(1);
      }
    });
  }
  else {  
    // FIXME: how to handle here?
  }  
}

struct WhileOpConversion : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;
 
  LogicalResult matchAndRewrite(scf::WhileOp loop,
                                PatternRewriter &rewriter) const override {
    if (!loop.isWhile())
      return failure();

    struct LoopInfo {
      Value indVar = nullptr;
      Value ub = nullptr;
      Value lb = nullptr;
      Value step = nullptr;
    }loopInfo;

    // locate cmp op and extract indVar and ub.
    loop.before().walk([&](Operation *op) {
      if (auto condOp = dyn_cast<scf::ConditionOp>(op)) {
        Operation *maybeCmpIOp = condOp.condition().getDefiningOp();
        if (auto cmpIOp = dyn_cast<CmpIOp>(maybeCmpIOp)) {
          getExitLimitFromCmpIOp(cmpIOp, loop, loopInfo.ub, loopInfo.indVar);
        } 
      }
    });

    if ((!loopInfo.indVar) || (!loopInfo.ub))
      return failure();

    getLowerBound(loopInfo.indVar, loop, loopInfo.lb);
    if (!loopInfo.lb)
      return failure();
  
    getStep(loopInfo.indVar, loop, loopInfo.step);
    if (!loopInfo.step)
      return failure();

    loopInfo.ub.dump();
    loopInfo.lb.dump();
    loopInfo.step.dump();    

/*
    auto ne = NumberOfExecutions(loop.getParentOp());
    ne.printOperationExecutions(llvm::dbgs(), &loop.before());
    loop.before().walk([&](Operation *op) {
      if (auto condOp = dyn_cast<scf::ConditionOp>(op)) {
        auto ne = NumberOfExecutions(condOp);
        auto res = ne.getNumberOfExecutions(condOp, &loop.before());
        LLVM_DEBUG(llvm::dbgs() << "res:" << res << "\n");
      }
    });
*/    
        
    // copy body from after region (no terminator nor increment) 
    return failure();
  }
};

void ConvertWhileToFor::runOnFunction() {

  // see: test/lib/Transforms/TestNumberOfExecutions.cp
  ConversionTarget target(getContext());  
  //target.addLegalDialect<LLVM::LLVMDialect>();
  
  OwningRewritePatternList patterns;
  patterns.insert<WhileOpConversion>(&getContext());

  if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createWhileOpToForOpPass() {
  return std::make_unique<ConvertWhileToFor>();
}

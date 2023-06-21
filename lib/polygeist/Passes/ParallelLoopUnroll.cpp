//===- ParallelLoopUnroll.cpp - Code to perform loop unrolling ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop unrolling.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/RegionUtils.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <set>

#include "ParallelLoopUnroll.h"

using namespace mlir;
using namespace polygeist;

// Adapted from "mlir/lib/Dialect/SCF/Utils/Utils.cpp"

/// Generates unrolled copies of scf::ParallelOp 'loopBodyBlock', with
/// associated 'popIV' by 'unrollFactor', calling 'ivRemapFn' to remap
/// 'popIV' for each unrolled body. If specified, annotates the Ops in each
/// unrolled iteration using annotateFn.
static LogicalResult generateUnrolledInterleavedLoop(
    Block *srcBlock, Block *dstBlock, int dim, uint64_t unrollFactor,
    function_ref<Value(unsigned, Value, OpBuilder)> ivRemapFn) {
  // Builder to insert unrolled bodies just before the terminator of the body of
  // 'pop'.
  auto builder = OpBuilder::atBlockBegin(dstBlock);

  BlockArgument srcIV = srcBlock->getArgument(dim);
  BlockArgument dstIV = dstBlock->getArgument(dim);

  BlockAndValueMapping barrierBlockArgMap;
  for (unsigned j = 0; j < srcBlock->getNumArguments(); j++)
    barrierBlockArgMap.map(srcBlock->getArgument(j), dstBlock->getArgument(j));
  SmallVector<BlockAndValueMapping, 4> operandMap;
  for (unsigned i = 0; i < unrollFactor; i++) {
    operandMap.emplace_back(BlockAndValueMapping());
    for (unsigned j = 0; j < srcBlock->getNumArguments(); j++)
      operandMap[i].map(srcBlock->getArgument(j), dstBlock->getArgument(j));
    // If the induction variable is used, create a remapping to the value for
    // this unrolled instance.
    if (!srcIV.use_empty()) {
      Value ivUnroll = ivRemapFn(i, dstIV, builder);
      operandMap[i].map(srcIV, ivUnroll);
    }
  }
  auto isBarrier = [&](Operation *op) {
    // TODO this can be improved to answer false for barrier that do not act on
    // the induction variale we are unrolling wrt
    return dyn_cast<polygeist::BarrierOp>(op);
  };
  auto collectNestedBarrierOperands = [&](Operation *op) {
    std::vector<Value> operands;
    op->walk([&](polygeist::BarrierOp barrier) {
      for (auto opr : barrier->getOperands()) {
        operands.push_back(opr);
      }
    });
    return operands;
  };
  auto nestedBarrierSyncsOverArg = [&](Operation *op, Value arg) {
    auto oprs = collectNestedBarrierOperands(op);
    return std::find(oprs.begin(), oprs.end(), arg) != oprs.end();
  };
  auto hasNestedBarrier = [&](Operation *op) {
    return op
        ->walk([&](polygeist::BarrierOp barrier) {
          return WalkResult::interrupt();
        })
        .wasInterrupted();
  };
  auto threadIndependent = [&](Value v) -> bool {
    if (auto BA = v.dyn_cast<BlockArgument>()) {
      if (BA == srcIV)
        return false;
      return BA.getOwner()->getParentOp()->isAncestor(srcBlock->getParentOp());
    } else {
      Operation *op = v.getDefiningOp();
      return op->getBlock()->getParentOp()->isProperAncestor(
          srcBlock->getParentOp());
    }
  };
  std::function<LogicalResult(Block *, Block *)> interleaveBlock =
      [&](Block *srcBlock, Block *dstBlock) {
        auto interleaveOp = [&](Operation *op) {
          // An operation can be recursively interleaved if its control flow is
          // the same across the threads
          if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            if (!(llvm::all_of(SmallVector<Value, 3>({forOp.getUpperBound(),
                                                      forOp.getLowerBound(),
                                                      forOp.getStep()}),
                               threadIndependent) ||
                  nestedBarrierSyncsOverArg(op, srcIV)))
              return failure();
            if (forOp.getNumIterOperands() != 0)
              // TODO I think we should be able to do this?
              return failure();
            auto dstForOp = cast<scf::ForOp>(builder.cloneWithoutRegions(
                *forOp.getOperation(), operandMap[0]));
            dstForOp.getRegion().push_back(new Block());
            for (auto a : forOp.getBody()->getArguments()) {
              auto b =
                  dstForOp.getBody()->addArgument(a.getType(), op->getLoc());
              for (unsigned i = 0; i < unrollFactor; i++)
                operandMap[i].map(a, b);
            }
            OpBuilder::InsertionGuard _(builder);
            builder.setInsertionPointToStart(dstForOp.getBody());
            builder.clone(*forOp.getBody()->getTerminator());
            builder.setInsertionPointToStart(dstForOp.getBody());
            if (interleaveBlock(forOp.getBody(), dstForOp.getBody())
                    .succeeded()) {
              return success();
            } else {
              dstForOp->erase();
              return failure();
            }
          } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            if (!(threadIndependent(ifOp.getCondition()) ||
                  nestedBarrierSyncsOverArg(op, srcIV)))
              return failure();
            auto hasElse = !ifOp.getElseRegion().empty();
            auto dstIfOp = cast<scf::IfOp>(builder.cloneWithoutRegions(
                *ifOp.getOperation(), operandMap[0]));
            dstIfOp.getThenRegion().push_back(new Block());
            OpBuilder::atBlockBegin(dstIfOp.getBody(0))
                .clone(*ifOp.getBody(0)->getTerminator());
            if (hasElse) {
              dstIfOp.getElseRegion().push_back(new Block());
              OpBuilder::atBlockBegin(dstIfOp.getBody(1))
                  .clone(*ifOp.getBody(1)->getTerminator());
            }
            OpBuilder::InsertionGuard _(builder);
            builder.setInsertionPointToStart(dstIfOp.getBody(0));
            auto resThen = interleaveBlock(ifOp.getBody(0), dstIfOp.getBody(0))
                               .succeeded();
            if (hasElse)
              builder.setInsertionPointToStart(dstIfOp.getBody(1));
            auto resElse =
                !hasElse || interleaveBlock(ifOp.getBody(1), dstIfOp.getBody(1))
                                .succeeded();
            if (resThen && resElse) {
              return success();
            } else {
              dstIfOp->erase();
              return failure();
            }
          } else if (auto pop = dyn_cast<scf::ParallelOp>(op)) {
            if (/*interleaveNestedParallelOps*/ true) {
              SmallVector<Value, 9> operands;
              operands.append(pop.getUpperBound().begin(),
                              pop.getUpperBound().end());
              operands.append(pop.getLowerBound().begin(),
                              pop.getLowerBound().end());
              operands.append(pop.getStep().begin(), pop.getStep().end());
              if (!(llvm::all_of(operands, threadIndependent) ||
                    nestedBarrierSyncsOverArg(op, srcIV)))
                return failure();
              auto dstPop = cast<scf::ParallelOp>(builder.cloneWithoutRegions(
                  *pop.getOperation(), operandMap[0]));
              dstPop.getRegion().push_back(new Block());
              for (auto a : pop.getBody()->getArguments()) {
                auto b =
                    dstPop.getBody()->addArgument(a.getType(), op->getLoc());
                for (unsigned i = 0; i < unrollFactor; i++) {
                  operandMap[i].map(a, b);
                }
                barrierBlockArgMap.map(a, b);
              }
              OpBuilder::InsertionGuard _(builder);
              builder.setInsertionPointToStart(dstPop.getBody());
              builder.clone(*pop.getBody()->getTerminator());
              builder.setInsertionPointToStart(dstPop.getBody());
              if (interleaveBlock(pop.getBody(), dstPop.getBody())
                      .succeeded()) {
                return success();
              } else {
                dstPop->erase();
                return failure();
              }
            } else {
              // We can instead increase the trip count by unrollFactor
              return failure();
            }
          } else {
            return failure();
          }
        };

        Block::iterator srcBlockEnd = std::prev(srcBlock->end(), 2);
        for (auto it = srcBlock->begin(); it != std::next(srcBlockEnd); it++) {
          if (isBarrier(&*it)) {
            builder.clone(*it, barrierBlockArgMap);
          } else if (interleaveOp(&*it).failed()) {
            if (hasNestedBarrier(&*it)) {
              if (getenv("POLYGEIST_EMIT_REMARKS_SCF_PARALLEL_LOOP_UNROLL")) {
                it->emitRemark("failed to interleave op with nested barrier");
              }
              return failure();
            }
            for (unsigned i = 0; i < unrollFactor; i++) {
              builder.clone(*it, operandMap[i]);
            }
          }
        }
        return success();
      };
  return interleaveBlock(srcBlock, dstBlock);
}

static bool isNormalized(scf::ParallelOp op) {
  auto isZero = [](Value v) {
    APInt value;
    return matchPattern(v, m_ConstantInt(&value)) && value.isNullValue();
  };
  auto isOne = [](Value v) {
    APInt value;
    return matchPattern(v, m_ConstantInt(&value)) && value.isOneValue();
  };
  return llvm::all_of(op.getLowerBound(), isZero) &&
         llvm::all_of(op.getStep(), isOne);
}

template <int S = 3> SmallVector<Value, S> getUpperBounds(scf::ParallelOp pop) {
  SmallVector<Value, S> bounds;
  for (auto bound : pop.getUpperBound()) {
    bounds.push_back(bound);
  }
  return bounds;
}

// Build the IR that performs ceil division of a positive value by another
// positive value:
//    ceildiv(a, b) = divis(a + (b - 1), b)
// where divis is rounding-to-zero division.
static Value ceilDivPositive(OpBuilder &builder, Location loc, Value dividend,
                             Value divisor) {
  assert(dividend.getType().isIndex() && "expected index-typed value");

  Value cstOne = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value divisorMinusOne = builder.create<arith::SubIOp>(loc, divisor, cstOne);
  Value sum = builder.create<arith::AddIOp>(loc, dividend, divisorMinusOne);
  return builder.create<arith::DivUIOp>(loc, sum, divisor);
}

/// Unrolls 'pop' by 'unrollFactor', returns success if the loop is unrolled.
LogicalResult mlir::polygeist::scfParallelUnrollByFactor(
    scf::ParallelOp &pop, uint64_t unrollFactor, unsigned dim,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn) {
  assert(unrollFactor > 0 && "expected positive unroll factor");
  assert(dim >= 0 && dim < pop.getUpperBound().size());
  assert(isNormalized(pop));

  // Return if the loop body is empty.
  if (llvm::hasSingleElement(pop.getBody()->getOperations()))
    return success();

  // Compute tripCount = ceilDiv((upperBound - lowerBound), step) and populate
  // 'upperBoundUnrolled' and 'stepUnrolled' for static and dynamic cases.
  auto loc = pop.getLoc();
  OpBuilder boundsBuilder(pop);
  Value upperBoundUnrolled = nullptr;

  auto lbCstOp =
      pop.getLowerBound()[dim].getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp =
      pop.getUpperBound()[dim].getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = pop.getStep()[dim].getDefiningOp<arith::ConstantIndexOp>();
  if (lbCstOp && ubCstOp && stepCstOp) {
    // Constant loop bounds computation.
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    if (!(lbCst == 0 && ubCst >= 0 && stepCst == 1)) {
      assert(0 && "expected positive loop bounds and step");
      return failure();
    }
    int64_t upperBoundRem = mlir::mod(ubCst, unrollFactor);

    if (upperBoundRem) {
      return failure();
    }
    upperBoundUnrolled = boundsBuilder.create<arith::ConstantIndexOp>(
        loc, mlir::ceilDiv(ubCst, unrollFactor));
  } else if (lbCstOp && !ubCstOp && stepCstOp) {
    int64_t lbCst = lbCstOp.value();
    int64_t stepCst = stepCstOp.value();
    if (!(lbCst == 0 && stepCst == 1)) {
      assert(0 && "expected positive loop bounds and step");
      return failure();
    }
    auto lowerBound = pop.getLowerBound()[dim];
    auto upperBound = pop.getUpperBound()[dim];
    auto step = pop.getStep()[dim];
    Value diff =
        boundsBuilder.create<arith::SubIOp>(loc, upperBound, lowerBound);
    Value tripCount = ceilDivPositive(boundsBuilder, loc, diff, step);
    Value unrollFactorCst =
        boundsBuilder.create<arith::ConstantIndexOp>(loc, unrollFactor);
    Value tripCountRem =
        boundsBuilder.create<arith::RemSIOp>(loc, tripCount, unrollFactorCst);
    // Compute tripCountEvenMultiple = tripCount - (tripCount % unrollFactor)
    Value tripCountEvenMultiple =
        boundsBuilder.create<arith::SubIOp>(loc, tripCount, tripCountRem);

    // TODO emit a runtime error if Rem != 0
    upperBoundUnrolled = tripCount;
  } else {
    assert(0);
  }

  auto ub = getUpperBounds(pop);
  auto unrollFactorVal =
      boundsBuilder.create<arith::ConstantIndexOp>(loc, unrollFactor);
  ub[dim] = upperBoundUnrolled;
  auto dstPop = boundsBuilder.create<scf::ParallelOp>(
      pop->getLoc(), pop.getLowerBound(), ub, pop.getStep());

  auto res = generateUnrolledInterleavedLoop(
      pop.getBody(), dstPop.getBody(), dim, unrollFactor,
      [&](unsigned i, Value iv, OpBuilder b) {
        // iv' = iv * unrollFactor + i
        auto base = b.create<arith::MulIOp>(loc, iv, unrollFactorVal);
        return b.create<arith::AddIOp>(
            loc, base, b.create<arith::ConstantIndexOp>(loc, i));
      });
  if (res.succeeded()) {
    pop->erase();
    pop = dstPop;
  } else {
    dstPop->erase();
  }
  return res;
}

struct SCFParallelLoopUnroll
    : public SCFParallelLoopUnrollBase<SCFParallelLoopUnroll> {
  SCFParallelLoopUnroll() = default;
  SCFParallelLoopUnroll(int unrollFactor) {
    this->unrollFactor.setValue(unrollFactor);
  }
  void runOnOperation() override {
    // Unroll the innermost parallel loops
    std::vector<scf::ParallelOp> pops;
    getOperation()->walk([&](scf::ParallelOp pop) {
      if (!pop->getParentOfType<scf::ParallelOp>())
        pops.push_back(pop);
    });
    for (auto pop : pops) {
      (void)scfParallelUnrollByFactor(pop, unrollFactor, 0, nullptr)
          .succeeded();
    }
  }
};
namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createSCFParallelLoopUnrollPass(int unrollFactor) {
  return std::make_unique<SCFParallelLoopUnroll>(unrollFactor);
}
} // namespace polygeist
} // namespace mlir

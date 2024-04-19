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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
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

  IRMapping barrierBlockArgMap;
  for (unsigned j = 0; j < srcBlock->getNumArguments(); j++)
    barrierBlockArgMap.map(srcBlock->getArgument(j), dstBlock->getArgument(j));
  SmallVector<IRMapping, 32> operandMap;
  for (unsigned i = 0; i < unrollFactor; i++) {
    operandMap.emplace_back(IRMapping());
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
    if (auto BA = dyn_cast<BlockArgument>(v)) {
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
        auto insertInterleavedYield = [&](Block *srcBlock, Block *dstBlock) {
          auto srcYieldOp = cast<scf::YieldOp>(srcBlock->getTerminator());
          SmallVector<Value> dstYieldArgs;
          for (auto yieldOperand : srcYieldOp.getOperands())
            for (unsigned i = 0; i < unrollFactor; i++)
              dstYieldArgs.push_back(
                  operandMap[i].lookupOrDefault(yieldOperand));
          OpBuilder::atBlockEnd(dstBlock).create<scf::YieldOp>(
              srcYieldOp.getLoc(), dstYieldArgs);
        };
        auto interleaveOp = [&](Operation *op) {
          // An operation can be recursively interleaved if its control flow is
          // the same across the threads
          if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            // Operands include bounds, step and iter arg initial vals
            if (!(llvm::all_of(forOp.getOperands(), threadIndependent) ||
                  nestedBarrierSyncsOverArg(op, srcIV)))
              return failure();
            SmallVector<Value> dstIterOperands;
            for (auto iterOperand : forOp.getInits())
              for (unsigned i = 0; i < unrollFactor; i++)
                dstIterOperands.push_back(
                    operandMap[i].lookupOrDefault(iterOperand));
            auto dstForOp = builder.create<scf::ForOp>(
                forOp.getLoc(),
                operandMap[0].lookupOrDefault(forOp.getLowerBound()),
                operandMap[0].lookupOrDefault(forOp.getUpperBound()),
                operandMap[0].lookupOrDefault(forOp.getStep()),
                dstIterOperands);
            if (forOp.getNumResults() == 0)
              dstForOp.getBody()->getTerminator()->erase();
            Value srcIndVar = forOp.getInductionVar();
            auto dstIndVar = dstForOp.getInductionVar();
            for (unsigned i = 0; i < unrollFactor; i++)
              operandMap[i].map(srcIndVar, dstIndVar);
            for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); j++) {
              auto a = forOp.getRegionIterArg(j);
              for (unsigned i = 0; i < unrollFactor; i++) {
                auto dstI = i + j * unrollFactor;
                auto b = dstForOp.getRegionIterArg(dstI);
                operandMap[i].map(a, b);
                operandMap[i].map(forOp.getResult(j), dstForOp.getResult(dstI));
              }
            }
            OpBuilder::InsertionGuard _(builder);
            builder.setInsertionPointToStart(dstForOp.getBody());
            if (interleaveBlock(forOp.getBody(), dstForOp.getBody())
                    .succeeded()) {
              insertInterleavedYield(forOp.getBody(), dstForOp.getBody());
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
            SmallVector<Type> dstResultTypes;
            for (auto result : ifOp.getResults())
              for (unsigned i = 0; i < unrollFactor; i++)
                dstResultTypes.push_back(result.getType());
            auto dstIfOp = builder.create<scf::IfOp>(
                ifOp.getLoc(), dstResultTypes,
                operandMap[0].lookupOrDefault(ifOp.getCondition()), hasElse);
            for (unsigned j = 0; j < ifOp.getNumResults(); j++) {
              for (unsigned i = 0; i < unrollFactor; i++) {
                auto dstI = i + j * unrollFactor;
                operandMap[i].map(ifOp.getResult(j), dstIfOp.getResult(dstI));
              }
            }
            if (ifOp.getNumResults() == 0) {
              dstIfOp.getBody(0)->getTerminator()->erase();
              if (hasElse)
                dstIfOp.getBody(1)->getTerminator()->erase();
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
              insertInterleavedYield(ifOp.getBody(0), dstIfOp.getBody(0));
              if (hasElse)
                insertInterleavedYield(ifOp.getBody(1), dstIfOp.getBody(1));
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
              if (interleaveBlock(pop.getBody(), dstPop.getBody())
                      .succeeded()) {
                OpBuilder::atBlockEnd(dstPop.getBody())
                    .clone(*pop.getBody()->getTerminator());
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
    return matchPattern(v, m_ConstantInt(&value)) && value.isZero();
  };
  auto isOne = [](Value v) {
    APInt value;
    return matchPattern(v, m_ConstantInt(&value)) && value.isOne();
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
    bool generateEpilogueLoop, bool coalescingFriendlyIndexing,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn) {
  assert(unrollFactor > 0 && "expected positive unroll factor");
  assert(dim >= 0 && dim < pop.getUpperBound().size());
  assert(isNormalized(pop));

  if (unrollFactor == 1) {
    return success();
  }

  // Return if the loop body is empty.
  if (llvm::hasSingleElement(pop.getBody()->getOperations()))
    return success();

  // Compute tripCount = ceilDiv((upperBound - lowerBound), step) and populate
  // 'upperBoundUnrolled' and 'stepUnrolled' for static and dynamic cases.
  auto loc = pop.getLoc();
  OpBuilder builder(pop);
  Value unrollFactorCst =
      builder.create<arith::ConstantIndexOp>(loc, unrollFactor);
  Value upperBoundUnrolled = nullptr;
  Value remUnrolled = nullptr;
  std::optional<int64_t> remUnrolledCst = {};

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
      llvm_unreachable("expected positive loop bounds and step");
      return failure();
    }
    int64_t upperBoundRem = mlir::mod(ubCst, unrollFactor);

    if (upperBoundRem && !generateEpilogueLoop) {
      return failure();
    }
    auto upperBoundUnrolledCst = ubCst / unrollFactor;
    if (upperBoundUnrolledCst == 0)
      return failure();
    upperBoundUnrolled =
        builder.create<arith::ConstantIndexOp>(loc, upperBoundUnrolledCst);
    remUnrolled = builder.create<arith::ConstantIndexOp>(loc, upperBoundRem);
    remUnrolledCst = upperBoundRem;
  } else if (lbCstOp && !ubCstOp && stepCstOp) {
    int64_t lbCst = lbCstOp.value();
    int64_t stepCst = stepCstOp.value();
    if (!(lbCst == 0 && stepCst == 1)) {
      llvm_unreachable("expected positive loop bounds and step");
      return failure();
    }
    // auto lowerBound = pop.getLowerBound()[dim];
    auto upperBound = pop.getUpperBound()[dim];
    // auto step = pop.getStep()[dim];
    upperBoundUnrolled =
        builder.create<arith::DivSIOp>(loc, upperBound, unrollFactorCst);
    // TODO what do we do if we dont generateEpilogueLoop but remUnrolled != 0 ?
    remUnrolled =
        builder.create<arith::RemSIOp>(loc, upperBound, unrollFactorCst);
  } else {
    assert(0);
    return failure();
  }

  auto ub = getUpperBounds(pop);
  ub[dim] = upperBoundUnrolled;
  auto dstPop = builder.create<scf::ParallelOp>(
      pop->getLoc(), pop.getLowerBound(), ub, pop.getStep());
  scf::ParallelOp epiloguePop = nullptr;

  if (generateEpilogueLoop && (!remUnrolledCst || *remUnrolledCst != 0)) {
    auto mainLoopTrips =
        builder.create<arith::MulIOp>(loc, upperBoundUnrolled, unrollFactorCst);
    epiloguePop = cast<scf::ParallelOp>(builder.clone(*pop));
    // TODO more robust way to set the upper bound
    epiloguePop->setOperand(pop.getUpperBound().size() + dim, remUnrolled);
    OpBuilder::InsertionGuard _(builder);
    builder.setInsertionPointToStart(epiloguePop.getBody());
    auto oldIV = epiloguePop.getBody()->getArgument(dim);
    auto newIV = builder.create<arith::AddIOp>(loc, mainLoopTrips, oldIV);
    oldIV.replaceAllUsesExcept(newIV, newIV);
  } else {
    // TODO throw runtime error if rem != 0 or should we expect the caller of
    // this fucntion to handle that case?
  }

  auto res = generateUnrolledInterleavedLoop(
      pop.getBody(), dstPop.getBody(), dim, unrollFactor,
      [&](unsigned i, Value iv, OpBuilder b) {
        if (coalescingFriendlyIndexing) {
          // upperBoundUnrolled = upperBound / unrollFactor;
          // iv(i) = iv + upperBoundUnrolled * i
          auto base =
              b.create<arith::MulIOp>(loc, upperBoundUnrolled,
                                      b.create<arith::ConstantIndexOp>(loc, i));
          return b.create<arith::AddIOp>(loc, base, iv);
        } else {
          // iv(i) = iv * unrollFactor + i
          auto base = b.create<arith::MulIOp>(loc, iv, unrollFactorCst);
          return b.create<arith::AddIOp>(
              loc, base, b.create<arith::ConstantIndexOp>(loc, i));
        }
      });
  if (res.succeeded()) {
    pop->erase();
    pop = dstPop;
  } else {
    if (epiloguePop)
      epiloguePop->erase();
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
      (void)scfParallelUnrollByFactor(pop, unrollFactor, 0, true, false,
                                      nullptr)
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

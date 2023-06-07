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
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/RegionUtils.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace polygeist;

namespace mlir::polygeist {
LogicalResult loopUnrollByFactor(
    scf::ParallelOp pop, uint64_t unrollFactor, int dim,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn);
}

// Adapted from "mlir/lib/Dialect/SCF/Utils/Utils.cpp"

/// Generates unrolled copies of scf::ParallelOp 'loopBodyBlock', with
/// associated 'popIV' by 'unrollFactor', calling 'ivRemapFn' to remap
/// 'popIV' for each unrolled body. If specified, annotates the Ops in each
/// unrolled iteration using annotateFn.
static void generateUnrolledInterleavedLoop(
    Block *srcBlock, Block *dstBlock, int dim, uint64_t unrollFactor,
    function_ref<Value(unsigned, Value, OpBuilder)> ivRemapFn) {
  // Builder to insert unrolled bodies just before the terminator of the body of
  // 'pop'.
  auto builder = OpBuilder::atBlockTerminator(dstBlock);

  Value srcIV = srcBlock->getArgument(dim);
  Value dstIV = dstBlock->getArgument(dim);

  // Keep a pointer to the last non-terminator operation in the original block
  // so that we know what to clone (since we are doing this in-place).
  Block::iterator srcBlockEnd = std::prev(srcBlock->end(), 2);

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
  for (auto it = srcBlock->begin(); it != std::next(srcBlockEnd); it++) {
    for (unsigned i = 0; i < unrollFactor; i++) {
      Operation *clonedOp = builder.clone(*it, operandMap[i]);
    }
  }
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

/// Unrolls 'pop' by 'unrollFactor', returns success if the loop is unrolled.
LogicalResult mlir::polygeist::loopUnrollByFactor(
    scf::ParallelOp pop, uint64_t unrollFactor, int dim,
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
  Value step = pop.getStep()[dim];
  int64_t upperBoundUnrolled;
  Value stepUnrolled;

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
    assert(lbCst == 0 && ubCst >= 0 && stepCst == 1 &&
           "expected positive loop bounds and step");
    int64_t upperBoundRem = mlir::mod(ubCst, unrollFactor);

    if (upperBoundRem) {
      return failure();
    }

    upperBoundUnrolled = mlir::ceilDiv(ubCst, unrollFactor);
  }

  OpBuilder builder(pop);
  auto ub = getUpperBounds(pop);
  auto unrollFactorVal =
      builder.create<arith::ConstantIndexOp>(loc, unrollFactor);
  ub[dim] = builder.create<arith::ConstantIndexOp>(loc, upperBoundUnrolled);
  auto dstPop = builder.create<scf::ParallelOp>(
      pop->getLoc(), pop.getLowerBound(), ub, pop.getStep());

  generateUnrolledInterleavedLoop(
      pop.getBody(), dstPop.getBody(), dim, unrollFactor,
      [&](unsigned i, Value iv, OpBuilder b) {
        // iv' = iv * unrollFactor + i
        auto base = b.create<arith::MulIOp>(loc, iv, unrollFactorVal);
        return b.create<arith::AddIOp>(
            loc, base, b.create<arith::ConstantIndexOp>(loc, i));
      });

  return success();
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
      if (!pop.getBody()
               ->walk([&](scf::ParallelOp nestedForOp) {
                 return WalkResult::interrupt();
               })
               .wasInterrupted())
        pops.push_back(pop);
    });
    for (auto pop : pops) {
      (void)loopUnrollByFactor(pop, unrollFactor, 0, nullptr);
      pop->erase();
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

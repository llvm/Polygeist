//===- BarrierUtil.cpp - Utilities for barrier removal --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/BarrierUtils.h"
#include "polygeist/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

static void findUsersBelowInBlock(Value value, Operation *reference,
                                  llvm::SetVector<Value> &crossing) {
  for (Operation *user : value.getUsers()) {
    // If the user is nested in another op, find its ancestor op that lives
    // in the same block as the barrier.
    while (user->getBlock() != reference->getBlock())
      user = user->getBlock()->getParentOp();

    if (reference->isBeforeInBlock(user)) {
      crossing.insert(value);
      break;
    }
  }
}

/// Populates `crossing` with values (op results) that are defined in the same
/// block as `op` and above it, and used by at least one op in the same block
/// below `op`. Uses may be in nested regions.
static void findOpResultUsesBelowInBlock(Operation *op,
                                         llvm::SetVector<Value> &crossing) {
  for (Operation *it = op->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    for (Value value : it->getResults()) {
      findUsersBelowInBlock(value, op, crossing);
    }
  }
}

void findValuesUsedBelow(polygeist::BarrierOp barrier,
                         llvm::SetVector<Value> &crossing) {
  Operation *op = barrier;
  do {
    findOpResultUsesBelowInBlock(op, crossing);
    assert(op->getBlock()->isEntryBlock() &&
           "only single-block regions are supported");
    if (isa<scf::ParallelOp>(op->getParentOp()))
      break;
    for (Value argument : op->getBlock()->getArguments())
      findUsersBelowInBlock(argument, op, crossing);
  } while ((op = op->getParentOp()));
}

void findValuesUsedBelow2(polygeist::BarrierOp barrier, DominanceInfo &dominanceInfo,
                          llvm::SetVector<Value> &values) {
  findOpResultUsesBelowInBlock(barrier, values);

  //   llvm::SetVector<Block *> successors;
  //   successors.insert(barrier->getBlock());
  //   for (unsigned i = 0; i < successors.size(); ++i) {
  //     Block *current = successors[i];
  //     for (Block *b : current->getSuccessors())
  //       successors.insert(b);
  //   }

  //   for (Block *b : llvm::drop_begin(successors)) {
  //     for (Operation &op : *b) {
  //       for (OpOperand &operand : op.getOperands()) {
  //       }
  //     }
  //   }
}

/// Returns the insertion point (as block pointer and itertor in it) immediately
/// after the definition of `v`.
static std::pair<Block *, Block::iterator> getInsertionPointAfterDef(Value v) {
  if (Operation *op = v.getDefiningOp())
    return {op->getBlock(), std::next(Block::iterator(op))};

  BlockArgument blockArg = v.cast<BlockArgument>();
  return {blockArg.getParentBlock(), blockArg.getParentBlock()->begin()};
}

/// Returns the insertion point that post-dominates `first` and `second`.
static std::pair<Block *, Block::iterator>
findNearestPostDominatingInsertionPoint(
    const std::pair<Block *, Block::iterator> &first,
    const std::pair<Block *, Block::iterator> &second,
    const PostDominanceInfo &postDominanceInfo) {
  // Same block, take the last op.
  if (first.first == second.first)
    return first.second->isBeforeInBlock(&*second.second) ? second : first;

  // Same region, use "normal" dominance analysis.
  if (first.first->getParent() == second.first->getParent()) {
    Block *block =
        postDominanceInfo.findNearestCommonDominator(first.first, second.first);
    assert(block);
    if (block == first.first)
      return first;
    if (block == second.first)
      return second;
    return {block, block->begin()};
  }

  if (first.first->getParent()->isAncestor(second.first->getParent()))
    return second;

  assert(second.first->getParent()->isAncestor(first.first->getParent()) &&
         "expected values to be defined in nested regions");
  return first;
}

/// Returns the insertion point that post-dominates all `values`.
static std::pair<Block *, Block::iterator>
findNesrestPostDominatingInsertionPoint(
    ArrayRef<Value> values, const PostDominanceInfo &postDominanceInfo) {
  assert(!values.empty());
  std::pair<Block *, Block::iterator> insertPoint =
      getInsertionPointAfterDef(values[0]);
  for (unsigned i = 1, e = values.size(); i < e; ++i)
    insertPoint = findNearestPostDominatingInsertionPoint(
        insertPoint, getInsertionPointAfterDef(values[i]), postDominanceInfo);
  return insertPoint;
}

std::pair<Block *, Block::iterator>
findInsertionPointAfterLoopOperands(scf::ParallelOp op) {
  // Find the earliest insertion point where loop bounds are fully defined.
  PostDominanceInfo postDominanceInfo(op->getParentOfType<FuncOp>());
  SmallVector<Value> operands;
  llvm::append_range(operands, op.lowerBound());
  llvm::append_range(operands, op.upperBound());
  llvm::append_range(operands, op.step());
  return findNesrestPostDominatingInsertionPoint(operands, postDominanceInfo);
}

/// Emits the IR  computing the total number of iterations in the loop. We don't
/// need to linearize them since we can allocate an nD array instead.
llvm::SmallVector<Value> emitIterationCounts(OpBuilder &builder,
                                             scf::ParallelOp op) {
  SmallVector<Value> iterationCounts;
  for (auto bounds : llvm::zip(op.lowerBound(), op.upperBound(), op.step())) {
    Value lowerBound = std::get<0>(bounds);
    Value upperBound = std::get<1>(bounds);
    Value step = std::get<2>(bounds);
    Value diff = builder.create<SubIOp>(op.getLoc(), upperBound, lowerBound);
    Value count = builder.create<SignedCeilDivIOp>(op.getLoc(), diff, step);
    iterationCounts.push_back(count);
  }
  return iterationCounts;
}


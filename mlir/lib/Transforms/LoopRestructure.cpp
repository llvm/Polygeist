//===- LoopRestructure.cpp - Find natural Loops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

#define DEBUG_TYPE "NaturalLoops"

template <> struct llvm::GraphTraits<const mlir::Block *> {
  using ChildIteratorType = mlir::Block::succ_iterator;
  using Node = const mlir::Block;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef bb) { return bb; }

  static ChildIteratorType child_begin(NodeRef node) {
    return const_cast<mlir::Block*>(node)->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return const_cast<mlir::Block*>(node)->succ_end(); }
};



namespace {

  struct LoopRestructure : public mlir::LoopRestructureBase<LoopRestructure> {
    void runOnRegion(DominanceInfo &domInfo, Region& region);
    void runOnFunction() override;
  };

} // end anonymous namespace

// Instantiate a variant of LLVM LoopInfo that works on mlir::Block
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopInfoImpl.h"

namespace mlir {
	class Loop : public llvm::LoopBase<mlir::Block, mlir::Loop> {
	private:
		Loop() = default;
		friend class llvm::LoopBase<Block, Loop>;
		friend class llvm::LoopInfoBase<Block, Loop>;
		explicit Loop(Block* B) : llvm::LoopBase<Block, Loop>(B) {}
		~Loop() = default;
	};
	class LoopInfo : public llvm::LoopInfoBase<mlir::Block, mlir::Loop> {
	public:
		LoopInfo(const llvm::DominatorTreeBase<mlir::Block, false> &DomTree) { analyze(DomTree); }
	};
}

template class llvm::LoopBase<::mlir::Block, ::mlir::Loop>;
template class llvm::LoopInfoBase<::mlir::Block, ::mlir::Loop>;

void LoopRestructure::runOnFunction() {
  FuncOp f = getFunction();
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  if (auto region = getOperation().getCallableRegion()) {
	  runOnRegion(domInfo, *region);
  }
}

void LoopRestructure::runOnRegion(DominanceInfo &domInfo, Region& region) {
  assert(domInfo.dominanceInfos.count(&region) != 0);
  auto DT = domInfo.dominanceInfos[&region].get();

  mlir::LoopInfo LI(*DT);
  llvm::errs() << "calling for region: " << &region << "\n";
  for(auto L : LI) {
    llvm::errs() << " found mlir loop " << *L << "\n";

    Block *header = L->getHeader();
    Block *target = L->getUniqueExitBlock();
    if (!target) {
      // Only support one exit block
      llvm::errs() << " found mlir loop with more than one exit, skipping. \n";
    }
    // TODO: Support multiple exit blocks
    //  - Easy case all exit blocks have the same argument set

	  // Create a caller block that will contain the loop op
    Block *wrapper = new Block();
    wrapper->insertBefore(header);

    // Copy the arguments across
    SmallVector<Type, 4> headerArgumentTypes;
    for (auto arg: header->getArguments()) {
      headerArgumentTypes.push_back(arg.getType());
    }
    wrapper->addArguments(headerArgumentTypes);

    SmallVector<Type, 4> returns;
    for (auto arg: target->getArguments()) {
      returns.push_back(arg.getType());
    }
    // now cut the loop from the environment
    SmallVector<Block*, 8> newBlocks;

    // Replace branch to exit block with a new block that calls loop.natural.return
    // In caller block, branch to correct exit block
    SmallVector<Block*, 4> exitingBlocks;
    L->getExitingBlocks(exitingBlocks);
    /* // TODO
    for (auto block: exitingBlocks) {
      Operation *terminator = block->getTerminator();
      for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
        Block *successor = terminator->getSuccessor(i);
        if (successor == target) {
          Block *pseudoExit = new Block();
          pseudoExit->insertBefore(target);
          pseudoExit->addArguments(returns);

          OpBuilder builder(pseudoExit);
          builder.create<scf::NaturalReturnOp>(terminator->getLoc(), returns, pseudoExit->getArguments());
          terminator->setSuccessor(pseudoExit, i);
          newBlocks.push_back(pseudoExit);
        }
      }
    }
    */

    // For each back edge create a new block and replace
    // the destination of that edge with said new block
    // in that new block call loop.natural.next
    SmallVector<Block*, 4> loopLatches;
    L->getLoopLatches(loopLatches);
    for (auto block: loopLatches) {
      Operation *terminator = block->getTerminator();
      for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
        Block *successor = terminator->getSuccessor(i);
        if (successor == header) {
          Block *pseudoLatch = new Block();
          pseudoLatch->insertBefore(target);
          pseudoLatch->addArguments(headerArgumentTypes);

          OpBuilder builder(pseudoLatch, pseudoLatch->begin());
          builder.create<scf::YieldOp>(terminator->getLoc(), returns, pseudoLatch->getArguments());
          terminator->setSuccessor(pseudoLatch, i);
          newBlocks.push_back(pseudoLatch);
        }
      }
    }

	  // Set branches into loop (header) to branch into caller block
    // Note: This breaks the back-edges, which is why we rewrote them earlier
    header->replaceAllUsesWith(wrapper);

    // Create loop operation in caller block

	  OpBuilder builder(wrapper, wrapper->begin());
    
    /*
    auto loop = builder.create<scf::ForOp>(header->front().getLoc(), returns, wrapper->getArguments());
    builder.create<BranchOp>(loop.getLoc(), target, loop.getResults());

    // Move loop header and loop blocks into loop operation
    Region *loopBody = &loop.getLoopBody();
    for (auto block: L->getBlocks()) {
      block->moveBefore(&loopBody->back());
    }

    for (auto block: newBlocks) {
      block->moveBefore(&loopBody->back());
    }
    // delete placeholder
    loopBody->back().erase();
    */
  }
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createLoopRestructurePass() {
  return std::make_unique<LoopRestructure>();
}

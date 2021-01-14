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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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
    void removeIfFromRegion(DominanceInfo &domInfo, Region& region, Block* pseudoExit);
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

bool attemptToFoldIntoPredecessor(Block* target) {
  SmallVector<Block*, 2> P(target->pred_begin(), target->pred_end());
  if (P.size() == 1) {
    if (auto op = dyn_cast<BranchOp>(P[0]->getTerminator())) {
      assert(target->getNumArguments() == op.getNumOperands());
      for(size_t i=0; i<target->getNumArguments(); ++i) {
        target->getArgument(i).replaceAllUsesWith(op.getOperand(i));
      }
      P[0]->getOperations().splice(P[0]->getOperations().end(),
                                   target->getOperations());
      op->erase();
      target->erase();
      return true;
    }
  }
  return false;
}

void LoopRestructure::removeIfFromRegion(DominanceInfo &domInfo, Region& region, Block* pseudoExit) {
  SmallVector<Block*, 4> Preds;
  for (auto block: pseudoExit->getPredecessors()) {
    Preds.push_back(block);
  }
  SmallVector<Type, 4> emptyTys;
  SmallVector<Type, 4> condTys;
  for(auto a : pseudoExit->getArguments()) {
    condTys.push_back(a.getType());
  }
  if (Preds.size() == 2) {
    for (size_t i=0; i<Preds.size(); ++i) {
      SmallVector<Block*, 4> Succs;
      for (auto block: Preds[i]->getSuccessors()) {
        Succs.push_back(block);
      }
      if (Succs.size() == 2) {
        for(size_t j=0; j<Succs.size(); ++j) {
          if (Succs[j] == pseudoExit && Succs[1-j] == Preds[1-i]) {
            OpBuilder builder(Preds[i]->getTerminator());
            auto condBr = cast<CondBranchOp>(Preds[i]->getTerminator());
            auto ifOp = builder.create<scf::IfOp>(builder.getUnknownLoc(), condTys, condBr.getCondition(), /*hasElse*/true);
            Succs[j] = new Block();
            if (j == 0) {
              ifOp.elseRegion().getBlocks().splice(ifOp.elseRegion().getBlocks().end(), region.getBlocks(), Succs[1-j]);
              SmallVector<unsigned, 4> idx;
              for(size_t i=0; i<Succs[1-j]->getNumArguments(); ++i) {
                Succs[1-j]->getArgument(i).replaceAllUsesWith(condBr.getFalseOperand(i));
                idx.push_back(i);
              }
              Succs[1-j]->eraseArguments(idx);
              assert(!ifOp.elseRegion().getBlocks().empty());
              assert(condTys.size() == condBr.getTrueOperands().size());
              OpBuilder tbuilder(&ifOp.thenRegion().front(),ifOp.thenRegion().front().begin());
              tbuilder.create<scf::YieldOp>(tbuilder.getUnknownLoc(), emptyTys, condBr.getTrueOperands());
            } else {
              if (!ifOp.thenRegion().getBlocks().empty()) {
                ifOp.thenRegion().front().erase();
              }
              ifOp.thenRegion().getBlocks().splice(ifOp.thenRegion().getBlocks().end(), region.getBlocks(), Succs[1-j]);
              SmallVector<unsigned, 4> idx;
              for(size_t i=0; i<Succs[1-j]->getNumArguments(); ++i) {
                Succs[1-j]->getArgument(i).replaceAllUsesWith(condBr.getTrueOperand(i));
                idx.push_back(i);
              }
              Succs[1-j]->eraseArguments(idx);
              assert(!ifOp.elseRegion().getBlocks().empty());
              OpBuilder tbuilder(&ifOp.elseRegion().front(),ifOp.elseRegion().front().begin());
              assert(condTys.size() == condBr.getFalseOperands().size());
              tbuilder.create<scf::YieldOp>(tbuilder.getUnknownLoc(), emptyTys, condBr.getFalseOperands());
            }
            auto oldTerm = Succs[1-j]->getTerminator();
            OpBuilder tbuilder(Succs[1-j], Succs[1-j]->end());
            tbuilder.create<scf::YieldOp>(tbuilder.getUnknownLoc(), emptyTys, oldTerm->getOperands());
            oldTerm->erase();

            SmallVector<Value, 4> res;
            for(size_t i=1; i<ifOp->getNumResults(); ++i) {
              res.push_back(ifOp->getResult(i));
            }
            builder.create<scf::ConditionOp>(builder.getUnknownLoc(), ifOp->getResult(0), res);
            condBr->erase();

            pseudoExit->erase();
            return;
          }
        }
      }
    }
  }
}

void LoopRestructure::runOnRegion(DominanceInfo &domInfo, Region& region) {
  assert(domInfo.dominanceInfos.count(&region) != 0);
  auto DT = domInfo.dominanceInfos[&region].get();

  mlir::LoopInfo LI(*DT);
  for(auto L : LI.getTopLevelLoops()) {

    Block *header = L->getHeader();
    Block *target = L->getUniqueExitBlock();
    if (!target) {
      // Only support one exit block
      llvm::errs() << " found mlir loop with more than one exit, skipping. \n";
    }
    // TODO: Support multiple exit blocks
    //  - Easy case all exit blocks have the same argument set

	  // Create a caller block that will contain the loop op

    Block* wrapper = new Block();
    region.push_back(wrapper);
    mlir::OpBuilder builder(wrapper, wrapper->begin());

    // Copy the arguments across
    SmallVector<Type, 4> headerArgumentTypes;
    for (auto arg: header->getArguments()) {
      headerArgumentTypes.push_back(arg.getType());
    }
    wrapper->addArguments(headerArgumentTypes);

    SmallVector<Type, 4> combinedTypes(headerArgumentTypes.begin(), headerArgumentTypes.end());
    SmallVector<Type, 4> returns;
    for (auto arg: target->getArguments()) {
      returns.push_back(arg.getType());
      combinedTypes.push_back(arg.getType());
    }

    auto loop = builder.create<mlir::scf::WhileOp>(builder.getUnknownLoc(), combinedTypes, wrapper->getArguments());
    {
    SmallVector<Value, 4> RetVals;
    for(size_t i=0; i<returns.size(); ++i) {
      RetVals.push_back(loop.getResult(i+headerArgumentTypes.size()));
    }
    builder.create<BranchOp>(builder.getUnknownLoc(), target, RetVals);
    }

    SmallVector<Block*, 4> Preds;

    for (auto block: header->getPredecessors()) {
      if (!L->contains(block))
        Preds.push_back(block);
    }

    loop.before().getBlocks().splice(loop.before().getBlocks().begin(), region.getBlocks(), header);
    for (Block* b : L->getBlocks()) {
      if (b != header) {
        loop.before().getBlocks().splice(loop.before().getBlocks().end(), region.getBlocks(), b);
      }
    }

    // Replace branch to exit block with a new block that calls loop.natural.return
    // In caller block, branch to correct exit block
    SmallVector<Block*, 4> exitingBlocks;
    L->getExitingBlocks(exitingBlocks);

    Block *pseudoExit = new Block();
    auto i1Ty = builder.getI1Type();
    {
    loop.before().push_back(pseudoExit);
    SmallVector<Type, 4> tys = {i1Ty};
    for(auto t : combinedTypes) tys.push_back(t);
    pseudoExit->addArguments(tys);
    OpBuilder builder(pseudoExit, pseudoExit->begin());
    tys.clear();
    builder.create<scf::ConditionOp>(builder.getUnknownLoc(), tys, pseudoExit->getArguments());
    }


    for (auto block: exitingBlocks) {
      Operation *terminator = block->getTerminator();
      for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
        Block *successor = terminator->getSuccessor(i);
        if (successor == target) {

          OpBuilder builder(terminator);
          auto vfalse = builder.create<mlir::ConstantOp>(
              builder.getUnknownLoc(), i1Ty, builder.getIntegerAttr(i1Ty, 0));
              
          std::vector<Value> args = { vfalse };
          for (auto arg : header->getArguments())
            args.push_back(arg);

          if (auto op = dyn_cast<BranchOp>(terminator)) {
            args.insert(args.end(), op.getOperands().begin(),
                                    op.getOperands().end());
            builder.create<BranchOp>(op.getLoc(), pseudoExit, args);
            op.erase();
          }
          if (auto op = dyn_cast<CondBranchOp>(terminator)) {
            std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                        op.getTrueOperands().end());
            std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                        op.getFalseOperands().end());
            if (op.getTrueDest() == target) {
              trueargs.insert(trueargs.begin(), args.begin(), args.end());
            }
            if (op.getFalseDest() == target) {
              falseargs.insert(falseargs.begin(), args.begin(), args.end());
            }
            builder.create<CondBranchOp>(op.getLoc(), op.getCondition(),
                                            op.getTrueDest() == target ? pseudoExit : op.getTrueDest(), trueargs,
                                            op.getFalseDest() == target ? pseudoExit : op.getFalseDest(), falseargs);
            op.erase();
          }
        }
      }
    }

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

          OpBuilder builder(terminator);
          auto vtrue = builder.create<mlir::ConstantOp>(
              builder.getUnknownLoc(), i1Ty, builder.getIntegerAttr(i1Ty, 1));
              
          if (auto op = dyn_cast<BranchOp>(terminator)) {
            std::vector<Value> args(op.getOperands().begin(),
                                    op.getOperands().end());
            args.insert(args.begin(), vtrue);
            for(auto ty : returns) {
              //args.push_back(builder.create<mlir::LLVM::UndefOp>(builder.getUnknownLoc(), ty));
              args.push_back(builder.create<mlir::ConstantOp>(builder.getUnknownLoc(), ty, builder.getIntegerAttr(ty, 0)));
            }
            builder.create<BranchOp>(op.getLoc(), pseudoExit, args);
            op.erase();
          }
          if (auto op = dyn_cast<CondBranchOp>(terminator)) {
            std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                        op.getTrueOperands().end());
            std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                        op.getFalseOperands().end());
            if (op.getTrueDest() == header) {
              trueargs.insert(trueargs.begin(), vtrue);
              for(auto ty : returns) {
                trueargs.push_back(builder.create<mlir::LLVM::UndefOp>(builder.getUnknownLoc(), ty));
              }
            }
            if (op.getFalseDest() == header) {
              falseargs.insert(falseargs.begin(), vtrue);
              for(auto ty : returns) {
                trueargs.push_back(builder.create<mlir::LLVM::UndefOp>(builder.getUnknownLoc(), ty));
              }
            }
            builder.create<CondBranchOp>(op.getLoc(), op.getCondition(),
                                            op.getTrueDest() == header ? pseudoExit : op.getTrueDest(), trueargs,
                                            op.getFalseDest() == header ? pseudoExit : op.getFalseDest(), falseargs);
            op.erase();
          }
        }
      }
    }

    Block* after = new Block();
    after->addArguments(combinedTypes);
    loop.after().push_back(after);
    OpBuilder builder2(after, after->begin());
    SmallVector<Value, 4> yieldargs;
    for(auto a : after->getArguments()) {
      if (yieldargs.size() == headerArgumentTypes.size()) break;
      yieldargs.push_back(a);
    }

    for (auto block: Preds) {
      Operation *terminator = block->getTerminator();
      for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
        Block *successor = terminator->getSuccessor(i);
        if (successor == header) {
          terminator->setSuccessor(wrapper, i);
        }
      }
    }

    for (size_t i=0; i<header->getNumArguments(); i++) {
      header->getArgument(i).replaceUsesWithIf(loop->getResult(i), [&](OpOperand& u) -> bool {
        return !loop.getOperation()->isProperAncestor(u.getOwner());
      });
    }


    builder2.create<scf::YieldOp>(builder.getUnknownLoc(), yieldargs);
    domInfo.recalculate(loop.getOperation());
    runOnRegion(domInfo, loop.before());
    removeIfFromRegion(domInfo, loop.before(), pseudoExit);

    attemptToFoldIntoPredecessor(wrapper);
    attemptToFoldIntoPredecessor(target);
    assert(loop.before().getBlocks().size() == 1);
  }
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createLoopRestructurePass() {
  return std::make_unique<LoopRestructure>();
}

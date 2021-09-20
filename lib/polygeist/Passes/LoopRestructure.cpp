//===- LoopRestructure.cpp - Find natural Loops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// TODO fix uses of induction or inner variables outside of loop
/*
see %2 in
func @kernel_gemm(%arg0: i32, %arg1: memref<?xf64>) {
  %c0 = constant 0 : index
  %c0_i32 = constant 0 : i32
  %c0_i64 = constant 0 : i64
  %c1_i32 = constant 1 : i32
  %c1_i64 = constant 1 : i64
  %c32_i32 = constant 32 : i32
  %cst = constant 1.000000e+00 : f64
  br ^bb1(%c0_i64 : i64)
^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
  %1 = subi %arg0, %c1_i32 : i32
  %2 = cmpi "slt", %1, %c0_i32 : i32
  %3 = scf.if %2 -> (i32) {
    %14 = subi %c0_i32, %1 : i32
    %15 = addi %14, %c32_i32 : i32
    %16 = subi %15, %c1_i32 : i32
    %17 = divi_signed %16, %c32_i32 : i32
    %18 = subi %c0_i32, %17 : i32
    scf.yield %18 : i32
  } else {
    %14 = divi_signed %1, %c32_i32 : i32
    scf.yield %14 : i32
  }
  %4 = sexti %3 : i32 to i64
  %5 = cmpi "sle", %0, %4 : i64
  cond_br %5, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %6 = load %arg1[%c0] : memref<?xf64>
  %7 = mulf %6, %cst : f64
  store %7, %arg1[%c0] : memref<?xf64>
  %8 = addi %0, %c1_i64 : i64
  br ^bb1(%8 : i64)
^bb3:  // pred: ^bb1
  %9 = scf.if %2 -> (i32) {
    %14 = subi %c0_i32, %1 : i32
    %15 = addi %14, %c32_i32 : i32
    %16 = subi %15, %c1_i32 : i32
    %17 = divi_signed %16, %c32_i32 : i32
    %18 = subi %c0_i32, %17 : i32
    scf.yield %18 : i32
  } else {
    %14 = divi_signed %1, %c32_i32 : i32
    scf.yield %14 : i32
  }
  %10 = sexti %9 : i32 to i64
  %11 = index_cast %10 : i64 to index
  %12 = load %arg1[%11] : memref<?xf64>
  %13 = addf %12, %cst : f64
  store %13, %arg1[%11] : memref<?xf64>
  return
}
*/
#include "polygeist/Passes/Passes.h"

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

#define DEBUG_TYPE "LoopRestructure"

struct Wrapper;

struct RWrapper {
  RWrapper(int x){};
  Wrapper &front();
};

struct Wrapper {
  mlir::Block blk;
  Wrapper() = delete;
  Wrapper(Wrapper &w) = delete;
  bool isLegalToHoistInto() const { return true; }
  void print(llvm::raw_ostream &OS) const {
    // B->print(OS);
  }
  void printAsOperand(llvm::raw_ostream &OS, bool b) const {
    // B->print(OS, b);
  }
  RWrapper *getParent() const {
    Region *R = ((Block *)(this))->getParent();
    return (RWrapper *)R;
  }
  mlir::Block &operator*() const { return *(Block *)(this); }
  mlir::Block *operator->() const { return (Block *)(this); }
};

Wrapper &RWrapper::front() { return *(Wrapper *)&((Region *)this)->front(); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Wrapper &w) {
  return os << "<cannot print wrapper>";
}

template <typename T>
struct Iter
    : public std::iterator<std::input_iterator_tag, // iterator_category
                           Wrapper *, std::ptrdiff_t, Wrapper **, Wrapper *> {
  T it;
  Iter(T it) : it(it) {}
  Wrapper *operator*() const;
  bool operator!=(Iter I) const { return it != I.it; }
  bool operator==(Iter I) const { return it == I.it; }
  void operator++() { ++it; }
  Iter<T> operator--() { return --it; }
  Iter<T> operator++(int) {
    auto prev = *this;
    it++;
    return prev;
  }
};

template <> Wrapper *Iter<Region::iterator>::operator*() const {
  Block &B = *it;
  return (Wrapper *)&B;
}
template <> Wrapper *Iter<Region::reverse_iterator>::operator*() const {
  Block &B = *it;
  return (Wrapper *)&B;
}

template <typename T> Wrapper *Iter<T>::operator*() const {
  Block *B = *it;
  return (Wrapper *)B;
}

namespace llvm {
template <> struct GraphTraits<RWrapper *> {
  using nodes_iterator = Iter<Region::iterator>;
  static Wrapper *getEntryNode(RWrapper *bb) {
    return (Wrapper *)&((Region *)bb)->front();
  }
  static nodes_iterator nodes_begin(RWrapper *bb) {
    return ((Region *)bb)->begin();
  }
  static nodes_iterator nodes_end(RWrapper *bb) {
    return ((Region *)bb)->end();
  }
};
template <> struct GraphTraits<Inverse<RWrapper *>> {
  using nodes_iterator = Iter<Region::reverse_iterator>;
  static Wrapper *getEntryNode(RWrapper *bb) {
    return (Wrapper *)&((Region *)bb)->front();
  }
  static nodes_iterator nodes_begin(RWrapper *bb) {
    return ((Region *)bb)->rbegin();
  }
  static nodes_iterator nodes_end(RWrapper *bb) {
    return ((Region *)bb)->rend();
  }
};
template <> struct GraphTraits<const Wrapper *> {
  using ChildIteratorType = Iter<Block::succ_iterator>;
  using Node = const Wrapper;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef bb) { return bb; }

  static ChildIteratorType child_begin(NodeRef node) {
    return (*node)->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return (*node)->succ_end();
  }
};
template <> struct GraphTraits<Wrapper *> {
  using ChildIteratorType = Iter<Block::succ_iterator>;
  using Node = Wrapper;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef bb) { return bb; }

  static ChildIteratorType child_begin(NodeRef node) {
    return (*node)->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return (*node)->succ_end();
  }
};

template <> struct GraphTraits<Inverse<Wrapper *>> {
  using ChildIteratorType = Iter<Block::pred_iterator>;
  using Node = Wrapper;
  using NodeRef = Node *;

  static ChildIteratorType child_begin(NodeRef node) {
    return (*node)->pred_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return (*node)->pred_end();
  }
};
template <> struct GraphTraits<Inverse<const Wrapper *>> {
  using ChildIteratorType = Iter<Block::pred_iterator>;
  using Node = const Wrapper;
  using NodeRef = Node *;

  static ChildIteratorType child_begin(NodeRef node) {
    return (*node)->pred_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return (*node)->pred_end();
  }
};

template <>
struct GraphTraits<const DomTreeNodeBase<Wrapper> *>
    : public DomTreeGraphTraitsBase<const DomTreeNodeBase<Wrapper>,
                                    DomTreeNodeBase<Wrapper>::const_iterator> {
};

} // namespace llvm

namespace {

struct LoopRestructure : public mlir::LoopRestructureBase<LoopRestructure> {
  void runOnRegion(DominanceInfo &domInfo, Region &region);
  bool removeIfFromRegion(DominanceInfo &domInfo, Region &region,
                          Block *pseudoExit);
  void runOnFunction() override;
};

} // end anonymous namespace

// Instantiate a variant of LLVM LoopInfo that works on mlir::Block
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopInfoImpl.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

template class llvm::DominatorTreeBase<Wrapper, false>;
template class llvm::DomTreeNodeBase<Wrapper>;
// template void
// llvm::DomTreeBuilder::ApplyUpdates<llvm::DominatorTreeBase<Wrapper, false>>;

namespace mlir {
class Loop : public llvm::LoopBase<Wrapper, mlir::Loop> {
private:
  Loop() = default;
  friend class llvm::LoopBase<Wrapper, Loop>;
  friend class llvm::LoopInfoBase<Wrapper, Loop>;
  explicit Loop(Wrapper *B) : llvm::LoopBase<Wrapper, Loop>(B) {}
  ~Loop() = default;
};
class LoopInfo : public llvm::LoopInfoBase<Wrapper, mlir::Loop> {
public:
  LoopInfo(const llvm::DominatorTreeBase<Wrapper, false> &DomTree) {
    analyze(DomTree);
  }
};
} // namespace mlir

template class llvm::LoopBase<Wrapper, ::mlir::Loop>;
template class llvm::LoopInfoBase<Wrapper, ::mlir::Loop>;

void LoopRestructure::runOnFunction() {
  // FuncOp f = getFunction();
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  if (auto region = getOperation().getCallableRegion()) {
    runOnRegion(domInfo, *region);
  }
}

bool attemptToFoldIntoPredecessor(Block *target) {
  SmallVector<Block *, 2> P(target->pred_begin(), target->pred_end());
  if (P.size() == 1) {
    if (auto op = dyn_cast<BranchOp>(P[0]->getTerminator())) {
      assert(target->getNumArguments() == op.getNumOperands());
      for (size_t i = 0; i < target->getNumArguments(); ++i) {
        target->getArgument(i).replaceAllUsesWith(op.getOperand(i));
      }
      P[0]->getOperations().splice(P[0]->getOperations().end(),
                                   target->getOperations());
      op->erase();
      target->erase();
      return true;
    }
  } else if (P.size() == 2) {
    if (auto op = dyn_cast<CondBranchOp>(P[0]->getTerminator())) {
      assert(target->getNumArguments() == op.getNumTrueOperands());
      assert(target->getNumArguments() == op.getNumFalseOperands());

      mlir::OpBuilder builder(op);
      SmallVector<mlir::Type> types;
      for (auto T : op.getTrueOperands()) {
        types.push_back(T.getType());
      }

      for (size_t i = 0; i < target->getNumArguments(); ++i) {
        auto sel = builder.create<mlir::SelectOp>(
            op.getLoc(), op.getCondition(), op.getTrueOperand(i),
            op.getFalseOperand(i));
        target->getArgument(i).replaceAllUsesWith(sel);
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

bool LoopRestructure::removeIfFromRegion(DominanceInfo &domInfo, Region &region,
                                         Block *pseudoExit) {
  SmallVector<Block *, 4> Preds;
  for (auto block : pseudoExit->getPredecessors()) {
    Preds.push_back(block);
  }
  SmallVector<Type, 4> emptyTys;
  SmallVector<Type, 4> condTys;
  for (auto a : pseudoExit->getArguments()) {
    condTys.push_back(a.getType());
  }
  if (Preds.size() == 2) {
    for (size_t i = 0; i < Preds.size(); ++i) {
      SmallVector<Block *, 4> Succs;
      for (auto block : Preds[i]->getSuccessors()) {
        Succs.push_back(block);
      }
      if (Succs.size() == 2) {
        for (size_t j = 0; j < Succs.size(); ++j) {
          if (Succs[j] == pseudoExit && Succs[1 - j] == Preds[1 - i]) {
            OpBuilder builder(Preds[i]->getTerminator());
            auto condBr = cast<CondBranchOp>(Preds[i]->getTerminator());
            auto ifOp = builder.create<scf::IfOp>(
                builder.getUnknownLoc(), condTys, condBr.getCondition(),
                /*hasElse*/ true);
            Succs[j] = new Block();
            if (j == 0) {
              ifOp.elseRegion().getBlocks().splice(
                  ifOp.elseRegion().getBlocks().end(), region.getBlocks(),
                  Succs[1 - j]);
              SmallVector<unsigned, 4> idx;
              for (size_t i = 0; i < Succs[1 - j]->getNumArguments(); ++i) {
                Succs[1 - j]->getArgument(i).replaceAllUsesWith(
                    condBr.getFalseOperand(i));
                idx.push_back(i);
              }
              Succs[1 - j]->eraseArguments(idx);
              assert(!ifOp.elseRegion().getBlocks().empty());
              assert(condTys.size() == condBr.getTrueOperands().size());
              OpBuilder tbuilder(&ifOp.thenRegion().front(),
                                 ifOp.thenRegion().front().begin());
              tbuilder.create<scf::YieldOp>(tbuilder.getUnknownLoc(), emptyTys,
                                            condBr.getTrueOperands());
            } else {
              if (!ifOp.thenRegion().getBlocks().empty()) {
                ifOp.thenRegion().front().erase();
              }
              ifOp.thenRegion().getBlocks().splice(
                  ifOp.thenRegion().getBlocks().end(), region.getBlocks(),
                  Succs[1 - j]);
              SmallVector<unsigned, 4> idx;
              for (size_t i = 0; i < Succs[1 - j]->getNumArguments(); ++i) {
                Succs[1 - j]->getArgument(i).replaceAllUsesWith(
                    condBr.getTrueOperand(i));
                idx.push_back(i);
              }
              Succs[1 - j]->eraseArguments(idx);
              assert(!ifOp.elseRegion().getBlocks().empty());
              OpBuilder tbuilder(&ifOp.elseRegion().front(),
                                 ifOp.elseRegion().front().begin());
              assert(condTys.size() == condBr.getFalseOperands().size());
              tbuilder.create<scf::YieldOp>(tbuilder.getUnknownLoc(), emptyTys,
                                            condBr.getFalseOperands());
            }
            auto oldTerm = Succs[1 - j]->getTerminator();
            OpBuilder tbuilder(Succs[1 - j], Succs[1 - j]->end());
            tbuilder.create<scf::YieldOp>(tbuilder.getUnknownLoc(), emptyTys,
                                          oldTerm->getOperands());
            oldTerm->erase();

            SmallVector<Value, 4> res;
            for (size_t i = 1; i < ifOp->getNumResults(); ++i) {
              res.push_back(ifOp->getResult(i));
            }
            builder.create<scf::ConditionOp>(builder.getUnknownLoc(),
                                             ifOp->getResult(0), res);
            condBr->erase();

            pseudoExit->erase();
            return true;
          }
        }
      }
    }
  }
  return false;
}

void LoopRestructure::runOnRegion(DominanceInfo &domInfo, Region &region) {
  if (region.getBlocks().size() > 1) {
    const llvm::DominatorTreeBase<Block, false> *DT =
        &domInfo.getDomTree(&region);
    mlir::LoopInfo LI(*(const llvm::DominatorTreeBase<Wrapper, false> *)DT);
    for (auto L : LI.getTopLevelLoops()) {
      Block *header = (Block *)L->getHeader();
      Block *target = (Block *)L->getUniqueExitBlock();
      if (!target) {
        // Only support one exit block
        llvm::errs()
            << " found mlir loop with more than one exit, skipping. \n";
        continue;
      }

      // Replace branch to exit block with a new block that calls
      // loop.natural.return In caller block, branch to correct exit block
      SmallVector<Wrapper *, 4> exitingBlocks;
      L->getExitingBlocks(exitingBlocks);

      // TODO: Support multiple exit blocks
      //  - Easy case all exit blocks have the same argument set

      // Create a caller block that will contain the loop op

      Block *wrapper = new Block();
      region.push_back(wrapper);
      mlir::OpBuilder builder(wrapper, wrapper->begin());

      // Copy the arguments across
      SmallVector<Type, 4> headerArgumentTypes;
      for (auto arg : header->getArguments()) {
        headerArgumentTypes.push_back(arg.getType());
      }
      // TODO values used outside loop should be wrapped.
      wrapper->addArguments(headerArgumentTypes);

      SmallVector<Type, 4> combinedTypes(headerArgumentTypes.begin(),
                                         headerArgumentTypes.end());
      SmallVector<Type, 4> returns;
      for (auto arg : target->getArguments()) {
        returns.push_back(arg.getType());
        combinedTypes.push_back(arg.getType());
      }

      auto loop = builder.create<mlir::scf::WhileOp>(
          builder.getUnknownLoc(), combinedTypes, wrapper->getArguments());
      {
        SmallVector<Value, 4> RetVals;
        for (size_t i = 0; i < returns.size(); ++i) {
          RetVals.push_back(loop.getResult(i + headerArgumentTypes.size()));
        }
        builder.create<BranchOp>(builder.getUnknownLoc(), target, RetVals);
      }

      SmallVector<Block *, 4> Preds;

      for (auto block : header->getPredecessors()) {
        if (!L->contains((Wrapper *)block))
          Preds.push_back(block);
      }

      loop.before().getBlocks().splice(loop.before().getBlocks().begin(),
                                       region.getBlocks(), header);
      for (auto *w : L->getBlocks()) {
        Block *b = &**w;
        if (b != header) {
          loop.before().getBlocks().splice(loop.before().getBlocks().end(),
                                           region.getBlocks(), b);
        }
      }

      Block *pseudoExit = new Block();
      auto i1Ty = builder.getI1Type();
      {
        loop.before().push_back(pseudoExit);
        SmallVector<Type, 4> tys = {i1Ty};
        for (auto t : combinedTypes)
          tys.push_back(t);
        pseudoExit->addArguments(tys);
        OpBuilder builder(pseudoExit, pseudoExit->begin());
        tys.clear();
        builder.create<scf::ConditionOp>(builder.getUnknownLoc(), tys,
                                         pseudoExit->getArguments());
      }

      for (auto *w : exitingBlocks) {
        Block *block = &**w;
        Operation *terminator = block->getTerminator();
        for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
          Block *successor = terminator->getSuccessor(i);
          if (successor == target) {

            OpBuilder builder(terminator);
            auto vfalse = builder.create<mlir::ConstantOp>(
                builder.getUnknownLoc(), i1Ty, builder.getIntegerAttr(i1Ty, 0));

            std::vector<Value> args = {vfalse};
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
              builder.create<CondBranchOp>(
                  op.getLoc(), op.getCondition(),
                  op.getTrueDest() == target ? pseudoExit : op.getTrueDest(),
                  trueargs,
                  op.getFalseDest() == target ? pseudoExit : op.getFalseDest(),
                  falseargs);
              op.erase();
            }
            break;
          }
        }
      }

      // For each back edge create a new block and replace
      // the destination of that edge with said new block
      // in that new block call loop.natural.next
      SmallVector<Wrapper *, 4> loopLatches;
      L->getLoopLatches(loopLatches);
      for (auto *w : loopLatches) {
        Block *block = &**w;
        Operation *terminator = block->getTerminator();
        // Note: the terminator may be reassigned in the loop body so not
        // caching numSuccessors here.
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
              for (auto ty : returns) {
                // args.push_back(builder.create<mlir::LLVM::UndefOp>(builder.getUnknownLoc(),
                // ty));
                args.push_back(builder.create<mlir::ConstantOp>(
                    builder.getUnknownLoc(), ty,
                    builder.getIntegerAttr(ty, 0)));
              }
              terminator =
                  builder.create<BranchOp>(op.getLoc(), pseudoExit, args);
              op.erase();
            } else if (auto op = dyn_cast<CondBranchOp>(terminator)) {
              std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                          op.getTrueOperands().end());
              std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                           op.getFalseOperands().end());
              if (op.getTrueDest() == header) {
                trueargs.insert(trueargs.begin(), vtrue);
                for (auto ty : returns) {
                  trueargs.push_back(builder.create<mlir::LLVM::UndefOp>(
                      builder.getUnknownLoc(), ty));
                }
              }
              if (op.getFalseDest() == header) {
                falseargs.insert(falseargs.begin(), vtrue);
                for (auto ty : returns) {
                  falseargs.push_back(builder.create<mlir::LLVM::UndefOp>(
                      builder.getUnknownLoc(), ty));
                }
              }
              // Recreate the terminator and store it so that its other
              // successor is visited on the next iteration of the loop.
              terminator = builder.create<CondBranchOp>(
                  op.getLoc(), op.getCondition(),
                  op.getTrueDest() == header ? pseudoExit : op.getTrueDest(),
                  trueargs,
                  op.getFalseDest() == header ? pseudoExit : op.getFalseDest(),
                  falseargs);
              op.erase();
            }
          }
        }
      }

      Block *after = new Block();
      after->addArguments(combinedTypes);
      loop.after().push_back(after);
      OpBuilder builder2(after, after->begin());
      SmallVector<Value, 4> yieldargs;
      for (auto a : after->getArguments()) {
        if (yieldargs.size() == headerArgumentTypes.size())
          break;
        yieldargs.push_back(a);
      }

      for (auto block : Preds) {
        Operation *terminator = block->getTerminator();
        for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
          Block *successor = terminator->getSuccessor(i);
          if (successor == header) {
            terminator->setSuccessor(wrapper, i);
          }
        }
      }

      for (size_t i = 0; i < header->getNumArguments(); i++) {
        header->getArgument(i).replaceUsesWithIf(
            loop->getResult(i), [&](OpOperand &u) -> bool {
              return !loop.getOperation()->isProperAncestor(u.getOwner());
            });
      }

      builder2.create<scf::YieldOp>(builder.getUnknownLoc(), yieldargs);
      domInfo.invalidate(&loop.before());
      runOnRegion(domInfo, loop.before());
      if (!removeIfFromRegion(domInfo, loop.before(), pseudoExit)) {
        attemptToFoldIntoPredecessor(pseudoExit);
      }

      attemptToFoldIntoPredecessor(wrapper);
      attemptToFoldIntoPredecessor(target);
      assert(loop.before().getBlocks().size() == 1);
      runOnRegion(domInfo, loop.after());
      assert(loop.after().getBlocks().size() == 1);
    }
  }

  for (auto &blk : region) {
    for (auto &op : blk) {
      for (auto &reg : op.getRegions()) {
        domInfo.invalidate(&reg);
        runOnRegion(domInfo, reg);
      }
    }
  }
}

namespace mlir {
namespace polygeist {
std::unique_ptr<OperationPass<FuncOp>> createLoopRestructurePass() {
  return std::make_unique<LoopRestructure>();
}
} // namespace polygeist
} // namespace mlir

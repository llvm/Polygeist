//===- Mem2Reg.cpp - MemRef DataFlow Optimization pass ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to forward memref stores to loads, thereby
// potentially getting rid of intermediate memref's entirely.
// TODO: In the future, similar techniques could be used to eliminate
// dead memref store's and perform more complex forwarding when support for
// SSA scalars live out of 'affine.for'/'affine.if' statements is available.
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>

#define DEBUG_TYPE "memref-dataflow-opt"

using namespace mlir;
#include <set>

typedef std::set<std::vector<ssize_t>> StoreMap;

namespace {
// The store to load forwarding relies on three conditions:
//
// 1) they need to have mathematically equivalent affine access functions
// (checked after full composition of load/store operands); this implies that
// they access the same single memref element for all iterations of the common
// surrounding loop,
//
// 2) the store op should dominate the load op,
//
// 3) among all op's that satisfy both (1) and (2), the one that postdominates
// all store op's that have a dependence into the load, is provably the last
// writer to the particular memref location being loaded at the load op, and its
// store value can be forwarded to the load. Note that the only dependences
// that are to be considered are those that are satisfied at the block* of the
// innermost common surrounding loop of the <store, load> being considered.
//
// (* A dependence being satisfied at a block: a dependence that is satisfied by
// virtue of the destination operation appearing textually / lexically after
// the source operation within the body of a 'affine.for' operation; thus, a
// dependence is always either satisfied by a loop or by a block).
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - they are sufficient, but not necessary --- since they
// don't reason about loops that are guaranteed to execute at least once or
// multiple sources to forward from.
//
// TODO: more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO: do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
struct Mem2Reg : public Mem2RegBase<Mem2Reg> {
  void runOnFunction() override;

  // return if changed
  bool forwardStoreToLoad(mlir::Value AI, std::vector<ssize_t> idx,
                          SmallVectorImpl<Operation *> &loadOpsToErase);
};

} // end anonymous namespace

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
std::unique_ptr<OperationPass<FuncOp>> mlir::createMem2RegPass() {
  return std::make_unique<Mem2Reg>();
}

bool matchesIndices(mlir::OperandRange ops, const std::vector<ssize_t> &idx) {
  if (ops.size() != idx.size())
    return false;
  for (size_t i = 0; i < idx.size(); i++) {
    if (auto op = ops[i].getDefiningOp<ConstantOp>()) {
      if (op.getValue().cast<IntegerAttr>().getInt() != idx[i]) {
        return false;
      }
    } else if (auto op = ops[i].getDefiningOp<ConstantIndexOp>()) {
      if (op.getValue() != idx[i]) {
        return false;
      }
    } else {
      assert(0 && "unhandled op");
    }
  }
  return true;
}

// This is a straightforward implementation not optimized for speed. Optimize
// if needed.
bool Mem2Reg::forwardStoreToLoad(
    mlir::Value AI, std::vector<ssize_t> idx,
    SmallVectorImpl<Operation *> &loadOpsToErase) {
  bool changed = false;
  std::set<mlir::LoadOp> loadOps;
  mlir::Type subType = nullptr;
  std::map<mlir::Block *, std::set<mlir::StoreOp>> storeOps;
  std::set<mlir::StoreOp> allStoreOps;

  std::deque<mlir::Value> list = {AI};

  while (list.size()) {
    auto val = list.front();
    list.pop_front();
    for (auto *user : val.getUsers()) {
      if (auto co = dyn_cast<mlir::MemRefCastOp>(user)) {
        list.push_back(co);
        continue;
      }
      if (auto loadOp = dyn_cast<mlir::LoadOp>(user)) {
        if (matchesIndices(loadOp.getIndices(), idx)) {
          subType = loadOp.getType();
          loadOps.insert(loadOp);
        }
      }
      if (auto storeOp = dyn_cast<mlir::StoreOp>(user)) {
        if (matchesIndices(storeOp.getIndices(), idx)) {
          storeOps[storeOp.getOperation()->getBlock()].insert(storeOp);
          allStoreOps.insert(storeOp);
        }
      }
    }
  }

  if (loadOps.size() == 0)
    return changed;
  /*
  // this is a valid optimization, however it should occur naturally
  // from the logic to follow anyways
  if (allStoreOps.size() == 1) {
    auto store = *allStoreOps.begin();
    for(auto loadOp : loadOps) {
      if (domInfo->dominates(store, loadOp)) {
        loadOp.replaceAllUsesWith(store.getValueToStore());
        loadOpsToErase.push_back(loadOp);
      }
    }
    return changed;
  }
  */

  // List of operations which may store that are not storeops
  SmallPtrSet<Operation *, 4> StoringOperations;
  {
    std::deque<Block *> todo;
    for (auto &pair : storeOps)
      todo.push_back(pair.first);
    while (todo.size()) {
      auto block = todo.front();
      assert(block);
      todo.pop_front();
      if (auto op = block->getParentOp()) {
        StoringOperations.insert(op);
        if (auto next = op->getBlock())
          todo.push_back(next);
      }
    }
  }

  // Last value stored in an individual block and the operation which stored it
  std::map<mlir::Block *, mlir::Value> lastStoreInBlock;

  // Last value stored in an individual block and the operation which stored it
  std::map<mlir::Block *, mlir::Value> valueAtStartOfBlock;

  // Start by setting lastStoreInBlock to the last store directly in that block
  // Note that this may miss a store within a region of an operation in that
  // block
  for (auto &pair : storeOps) {
    auto &block = *pair.first;
    Operation *last = nullptr;
    mlir::Value lastVal = nullptr;
    bool seenSubStore = false;
    for (auto &a : block) {
      if (StoringOperations.count(&a)) {
        last = &a;
        lastVal = nullptr;
        seenSubStore = true;
        // llvm::errs() << "erased store due to: " << a << "\n";
      } else if (auto loadOp = dyn_cast<LoadOp>(&a)) {
        if (loadOps.count(loadOp)) {
          if (lastVal) {
            changed = true;
            // llvm::errs() << "replacing " << loadOp << " with " << lastVal <<
            // "\n";
            loadOp.replaceAllUsesWith(lastVal);
            // Record this to erase later.
            loadOpsToErase.push_back(loadOp);
            loadOps.erase(loadOp);
          } else if (seenSubStore) {
            // llvm::errs() << "no lastval found for: " << loadOp << "\n";
            loadOps.erase(loadOp);
          }
        }
      } else if (auto storeOp = dyn_cast<StoreOp>(&a)) {
        if (pair.second.count(storeOp)) {
          last = &a;
          lastVal = storeOp.getValueToStore();
        }
      } else {
        // since not storing operation the value at the start and end of block
        // is lastVal
        a.walk([&](LoadOp loadOp) {
          if (loadOps.count(loadOp)) {
            if (lastVal) {
              changed = true;
              loadOp.replaceAllUsesWith(lastVal);
              // Record this to erase later.
              loadOpsToErase.push_back(loadOp);
              loadOps.erase(loadOp);
            } else if (seenSubStore) {
              // llvm::errs() << "ano lastval found for: " << loadOp << "\n";
              loadOps.erase(loadOp);
            }
          }
        });
      }
    }
    lastStoreInBlock[&block] = lastVal;
  }

  if (loadOps.size() == 0)
    return changed;

  std::set<Block *> unreachableBlocks;
  {
    std::deque<Block *> todo;
    for (auto &pair : lastStoreInBlock) {
      if (pair.second == nullptr) {
        for (auto succ : pair.first->getSuccessors()) {
          todo.push_back(succ);
        }
      }
    }
    while (todo.size()) {
      auto block = todo.front();
      todo.pop_front();
      if (valueAtStartOfBlock.find(block) != valueAtStartOfBlock.end())
        continue;
      if (unreachableBlocks.count(block))
        continue;
      unreachableBlocks.insert(block);
      for (auto succ : block->getSuccessors()) {
        todo.push_back(succ);
      }
    }
  }

  std::set<Block *> reachableBlocks;
  {
    std::deque<Block *> todo;
    for (auto &pair : lastStoreInBlock) {
      if (pair.second != nullptr) {
        if (isa<BranchOp, CondBranchOp>(pair.first->getTerminator())) {
          for (auto succ : pair.first->getSuccessors()) {
            assert(succ);
            assert(!succ->empty());
            todo.push_back(succ);
          }
        }
      }
    }
    while (todo.size()) {
      auto block = todo.front();
      assert(block);
      assert(!block->empty());
      todo.pop_front();
      if (reachableBlocks.count(block))
        continue;
      reachableBlocks.insert(block);
      for (auto succ : block->getSuccessors()) {
        todo.push_back(succ);
        assert(succ);
        assert(!succ->empty());
      }
    }
  }

  for (Block *blk : unreachableBlocks) {
    reachableBlocks.erase(blk);
  }

  {
    std::deque<Block *> todo(reachableBlocks.begin(), reachableBlocks.end());
    while (todo.size()) {
      auto block = todo.front();
      todo.pop_front();
      if (!reachableBlocks.count(block))
        continue;

      bool legal = true;
      for (auto pred : block->getPredecessors()) {

        if (lastStoreInBlock.find(pred) != lastStoreInBlock.end()) {
          if (lastStoreInBlock[pred] != nullptr) {
            continue;
          }
        }
        if (reachableBlocks.count(pred))
          continue;

        legal = false;
        break;
      }
      if (!legal) {
        reachableBlocks.erase(block);
        for (auto succ : block->getSuccessors()) {
          todo.push_back(succ);
        }
      }
    }
  }

  for (auto block : reachableBlocks) {
    if (valueAtStartOfBlock.find(block) != valueAtStartOfBlock.end())
      continue;
    auto arg = block->addArgument(subType);
    valueAtStartOfBlock[block] = arg;
    for (Operation &op : *block) {
      if (!StoringOperations.count(&op)) {
        op.walk([&](Block *blk) { valueAtStartOfBlock[blk] = arg; });
      }
    }
    if (lastStoreInBlock.find(block) == lastStoreInBlock.end()) {
      lastStoreInBlock[block] = arg;
    }
  }

  for (auto loadOp : loadOps) {
    auto blk = loadOp.getOperation()->getBlock();
    if (valueAtStartOfBlock.find(blk) != valueAtStartOfBlock.end()) {
      changed = true;
      loadOp.replaceAllUsesWith(valueAtStartOfBlock[blk]);
      loadOpsToErase.push_back(loadOp);
    } else {
      // TODO inter-op
      // llvm::errs() << "no value at start of block:\n";
      // loadOp.dump();
    }
  }

  for (auto block : reachableBlocks) {
    SmallVector<Block*, 4> preds;
    for (auto pred : block->getPredecessors()) {
      preds.push_back(pred);
    }
    for (auto pred : preds) {
      mlir::Value pval = lastStoreInBlock[pred];
      assert(pred->getTerminator());
      if (auto op = dyn_cast<BranchOp>(pred->getTerminator())) {
        mlir::OpBuilder subbuilder(op.getOperation());
        std::vector<Value> args(op.getOperands().begin(),
                                op.getOperands().end());
        args.push_back(pval);
        auto op2 = subbuilder.create<BranchOp>(op.getLoc(), op.getDest(), args);
        //op.replaceAllUsesWith(op2);
        op.erase();
      }
      if (auto op = dyn_cast<CondBranchOp>(pred->getTerminator())) {

        mlir::OpBuilder subbuilder(op.getOperation());
        std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                    op.getTrueOperands().end());
        std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                     op.getFalseOperands().end());
        if (op.getTrueDest() == block) {
          trueargs.push_back(pval);
        }
        if (op.getFalseDest() == block) {
          falseargs.push_back(pval);
        }
        auto op2 = subbuilder.create<CondBranchOp>(op.getLoc(), op.getCondition(),
                                        op.getTrueDest(), trueargs,
                                        op.getFalseDest(), falseargs);
        //op.replaceAllUsesWith(op2);
        op.erase();
      }
    }
  }

  // Remove block arguments if possible
  {
    std::deque<Block *> todo(reachableBlocks.begin(), reachableBlocks.end());
    while (todo.size()) {
      auto block = todo.front();
      todo.pop_front();
      if (!reachableBlocks.count(block))
        continue;
      if (valueAtStartOfBlock.find(block) == valueAtStartOfBlock.end())
        continue;

      auto maybeblockArg = valueAtStartOfBlock[block];
      auto blockArg = maybeblockArg.dyn_cast<BlockArgument>();
      if (!blockArg)
        continue;
      assert(blockArg.getOwner() == block);

      mlir::Value val = nullptr;
      bool legal = true;
      for (auto pred : block->getPredecessors()) {
        mlir::Value pval = nullptr;

        if (auto op = dyn_cast<BranchOp>(pred->getTerminator())) {
          pval = op.getOperands()[blockArg.getArgNumber()];
          if (pval == blockArg)
            pval = nullptr;
        } else if (auto op = dyn_cast<CondBranchOp>(pred->getTerminator())) {
          if (op.getTrueDest() == block) {
            pval = op.getTrueOperands()[blockArg.getArgNumber()];
            if (pval == blockArg)
              pval = nullptr;
          }
          if (op.getFalseDest() == block) {
            auto pval2 = op.getFalseOperands()[blockArg.getArgNumber()];
            if (pval2 != blockArg) {
              if (pval == nullptr) {
                pval = pval2;
              } else if (pval != pval2) {
                legal = false;
                break;
              }
            }
            if (pval == blockArg)
              pval = nullptr;
          }
        } else {
          pred->dump();
          block->dump();
          assert(0 && "unknown branch");
        }

        assert(pval != blockArg);
        if (val == nullptr) {
          val = pval;
        } else {
          if (pval != nullptr && val != pval) {
            legal = false;
            break;
          }
        }
      }

      bool used = false;
      for (auto U : blockArg.getUsers()) {

        if (auto op = dyn_cast<BranchOp>(U)) {
          size_t i = 0;
          for (auto V : op.getOperands()) {
            if (V == blockArg &&
                !(i == blockArg.getArgNumber() && op.getDest() == block)) {
              used = true;
              break;
            }
          }
          if (used)
            break;
        } else if (auto op = dyn_cast<CondBranchOp>(U)) {
          size_t i = 0;
          for (auto V : op.getTrueOperands()) {
            if (V == blockArg &&
                !(i == blockArg.getArgNumber() && op.getTrueDest() == block)) {
              used = true;
              break;
            }
          }
          if (used)
            break;
          i = 0;
          for (auto V : op.getFalseOperands()) {
            if (V == blockArg &&
                !(i == blockArg.getArgNumber() && op.getFalseDest() == block)) {
              used = true;
              break;
            }
          }
        } else
          used = true;
      }
      if (!used)
        legal = true;

      if (legal) {
        for (auto U : blockArg.getUsers()) {
          if (auto block = U->getBlock()) {
            todo.push_back(block);
            for (auto succ : block->getSuccessors())
              todo.push_back(succ);
          }
        }
        if (val != nullptr) {
          blockArg.replaceAllUsesWith(val);
        } else {
        }
        valueAtStartOfBlock.erase(block);

        SmallVector<Block*, 4> preds;
        for (auto pred : block->getPredecessors()) {
          preds.push_back(pred);
        }
        for (auto pred : preds) {
          if (auto op = dyn_cast<BranchOp>(pred->getTerminator())) {
            mlir::OpBuilder subbuilder(op.getOperation());
            std::vector<Value> args(op.getOperands().begin(),
                                    op.getOperands().end());
            args.erase(args.begin() + blockArg.getArgNumber());
            assert(args.size() == op.getOperands().size() - 1);
            subbuilder.create<BranchOp>(op.getLoc(), op.getDest(), args);
            op.erase();
          }
          if (auto op = dyn_cast<CondBranchOp>(pred->getTerminator())) {

            mlir::OpBuilder subbuilder(op.getOperation());
            std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                        op.getTrueOperands().end());
            std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                         op.getFalseOperands().end());
            if (op.getTrueDest() == block) {
              trueargs.erase(trueargs.begin() + blockArg.getArgNumber());
            }
            if (op.getFalseDest() == block) {
              falseargs.erase(falseargs.begin() + blockArg.getArgNumber());
            }
            assert(trueargs.size() < op.getTrueOperands().size() ||
                   falseargs.size() < op.getFalseOperands().size());
            subbuilder.create<CondBranchOp>(op.getLoc(), op.getCondition(),
                                            op.getTrueDest(), trueargs,
                                            op.getFalseDest(), falseargs);
            op.erase();
          }
        }
        block->eraseArgument(blockArg.getArgNumber());
      }
    }
  }
  return changed;
}

bool isPromotable(mlir::Value AI) {
  std::deque<mlir::Value> list = {AI};

  while (list.size()) {
    auto val = list.front();
    list.pop_front();

    for (auto U : val.getUsers()) {
      if (auto LO = dyn_cast<LoadOp>(U)) {
        for (auto idx : LO.getIndices()) {
          if (!idx.getDefiningOp<ConstantOp>() &&
              !idx.getDefiningOp<ConstantIndexOp>()) {
            // llvm::errs() << "non promotable "; AI.dump(); llvm::errs() << "
            // ldue to " << idx << "\n";
            return false;
          }
        }
        continue;
      } else if (auto SO = dyn_cast<StoreOp>(U)) {
        if (SO.value() == val)
          return false;
        for (auto idx : SO.getIndices()) {
          if (!idx.getDefiningOp<ConstantOp>() &&
              !idx.getDefiningOp<ConstantIndexOp>()) {
            // llvm::errs() << "non promotable "; AI.dump(); llvm::errs() << "
            // sdue to " << idx << "\n";
            return false;
          }
        }
        continue;
      } else if (isa<DeallocOp>(U)) {
        continue;
      } else if (isa<CallOp>(U) && cast<CallOp>(U).callee() == "free") {
        continue;
      } else if (auto CO = dyn_cast<MemRefCastOp>(U)) {
        list.push_back(CO);
      } else {
        // llvm::errs() << "non promotable "; AI.dump(); llvm::errs() << "  udue
        // to " << *U << "\n";
        return false;
      }
    }
  }
  return true;
}

StoreMap getLastStored(mlir::Value AI) {
  StoreMap lastStored;

  std::deque<mlir::Value> list = {AI};

  while (list.size()) {
    auto val = list.front();
    list.pop_front();
    for (auto U : val.getUsers()) {
      if (auto SO = dyn_cast<StoreOp>(U)) {
        std::vector<ssize_t> vec;
        for (auto idx : SO.getIndices()) {
          if (auto op = idx.getDefiningOp<ConstantOp>()) {
            vec.push_back(op.getValue().cast<IntegerAttr>().getInt());
          } else if (auto op = idx.getDefiningOp<ConstantIndexOp>()) {
            vec.push_back(op.getValue());
          } else {
            assert(0 && "unhandled op");
          }
        }
        lastStored.insert(vec);
      } else if (auto CO = dyn_cast<MemRefCastOp>(U)) {
        list.push_back(CO);
      }
    }
  }
  return lastStored;
}

void Mem2Reg::runOnFunction() {
  // Only supports single block functions at the moment.
  FuncOp f = getFunction();

  // Variable indicating that a memref has had a load removed
  // and or been deleted. Because there can be memrefs of
  // memrefs etc, we may need to do multiple passes (first
  // to eliminate the outermost one, then inner ones)
  bool changed;
  do {
    changed = false;

    // A list of memref's that are potentially dead / could be eliminated.
    SmallPtrSet<Value, 4> memrefsToErase;

    // Load op's whose results were replaced by those forwarded from stores.
    SmallVector<Operation *, 8> loadOpsToErase;

    // Walk all load's and perform store to load forwarding.
    SmallVector<mlir::Value, 4> toPromote;
    f.walk([&](mlir::AllocaOp AI) {
      if (isPromotable(AI)) {
        toPromote.push_back(AI);
      }
    });
    f.walk([&](mlir::AllocOp AI) {
      if (isPromotable(AI)) {
        toPromote.push_back(AI);
      }
    });

    for(auto AI : toPromote) {
      auto lastStored = getLastStored(AI);
      for (auto &vec : lastStored) {
        changed |= forwardStoreToLoad(AI, vec, loadOpsToErase);
      }
      memrefsToErase.insert(AI);
    }

    // Erase all load op's whose results were replaced with store fwd'ed ones.
    for (auto *loadOp : loadOpsToErase) {
      changed = true;
      loadOp->erase();
    }

    // Check if the store fwd'ed memrefs are now left with only stores and can
    // thus be completely deleted. Note: the canonicalize pass should be able
    // to do this as well, but we'll do it here since we collected these anyway.
    for (auto memref : memrefsToErase) {

      // If the memref hasn't been alloc'ed in this function, skip.
      Operation *defOp = memref.getDefiningOp();
      if (!defOp || !(isa<AllocOp>(defOp) || isa<AllocaOp>(defOp)))
        // TODO: if the memref was returned by a 'call' operation, we
        // could still erase it if the call had no side-effects.
        continue;

      std::deque<mlir::Value> list = {memref};
      std::vector<mlir::Operation *> toErase;
      bool error = false;
      while (list.size()) {
        auto val = list.front();
        list.pop_front();

        for (auto U : val.getUsers()) {
          if (isa<StoreOp, DeallocOp>(U)) {
            toErase.push_back(U);
          } else if (isa<CallOp>(U) && cast<CallOp>(U).callee() == "free") {
            toErase.push_back(U);
          } else if (auto CO = dyn_cast<MemRefCastOp>(U)) {
            toErase.push_back(U);
            list.push_back(CO);
          } else {
            error = true;
            break;
          }
        }
        if (error)
          break;
      }

      if (!error) {
        std::reverse(toErase.begin(), toErase.end());
        for (auto *user : toErase) {
          user->erase();
        }
        defOp->erase();
        changed = true;
      } else {
        // llvm::errs() << " failed to remove: " << memref << "\n";
      }
    }
  } while (changed);
}

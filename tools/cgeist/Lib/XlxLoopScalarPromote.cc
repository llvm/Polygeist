//===- XlxLoopScalarPromote.cc - Scalars carried by structured loops ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A local scalar that a loop reads and updates (a loop counter) is promoted
// to an SSA value carried by the loops around its accesses. Polygeist's
// mem2reg forwards stores to loads within a block and through scf.if, but a
// value carried around an scf.while stays in its `memref<T>` alloca: a loop
// counted by an `ap_uint<16>` (whose ++ / < are the Vitis header's inlined
// member functions) keeps its counter in memory, so --raise-scf-to-affine
// sees no induction variable and hls-est no counted loop (OPEN_ISSUES O101
// in the hls-est tree: rosetta 3d-rendering OUTPUT_FB_ROW / COL, Dehazing
// debayer, which Vitis pipelines; with `int` counters both are counted
// loops).
//
// For every rank-0 alloca of an integer, index or float element whose uses
// are only loads and stores of it, each nested in scf.if / scf.while /
// scf.for / affine.for / affine.if regions of its function, the value is
// threaded through those regions in program order (structured SSA
// construction): a load takes the current value, a store sets it, an if gets
// a result carrying the value out of both arms, a loop gets an iteration
// argument (scf.while: one in the before region forwarded to the after
// region and yielded back). A load before any store reads an undefined value
// (llvm.mlir.undef), as the alloca did. A loop body (a while's condition)
// starts from an undefined value instead of its iteration argument only where
// the carried value is dead: no path from there (through the body, a
// zero-trip body, the next iteration or the code after the loop) reads it
// before writing it. Any other use or region op leaves the alloca as it is.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

/// The memref a load / store accesses, or null.
Value accessedMemref(Operation *op) {
  if (auto l = dyn_cast<affine::AffineLoadOp>(op))
    return l.getMemRef();
  if (auto s = dyn_cast<affine::AffineStoreOp>(op))
    return s.getMemRef();
  if (auto l = dyn_cast<memref::LoadOp>(op))
    return l.getMemRef();
  if (auto s = dyn_cast<memref::StoreOp>(op))
    return s.getMemRef();
  return Value();
}

bool isLoad(Operation *op) {
  return isa<affine::AffineLoadOp, memref::LoadOp>(op);
}

/// A region op the promotion threads the value through.
bool isSupportedRegionOp(Operation *op) {
  return isa<scf::IfOp, scf::WhileOp, scf::ForOp, affine::AffineForOp,
             affine::AffineIfOp>(op);
}

class Promoter {
public:
  Promoter(memref::AllocaOp slot, IRRewriter &rewriter)
      : slot(slot), rewriter(rewriter) {}

  /// Whether every use is a plain load / store of the scalar inside
  /// supported region ops of the slot's function, and at least one use
  /// lies in a loop (only those need this pass).
  bool promotable() {
    auto fn = slot->getParentOfType<func::FuncOp>();
    if (!fn)
      return false;
    bool inLoop = false;
    for (OpOperand &use : slot.getResult().getUses()) {
      Operation *user = use.getOwner();
      if (accessedMemref(user) != slot.getResult())
        return false; // the address is stored, passed, viewed
      if (auto s = dyn_cast<affine::AffineStoreOp>(user);
          s && s.getValueToStore() == slot.getResult())
        return false;
      if (auto s = dyn_cast<memref::StoreOp>(user);
          s && s.getValueToStore() == slot.getResult())
        return false;
      for (Operation *p = user->getParentOp(); p != fn.getOperation();
           p = p->getParentOp()) {
        if (!p || !isSupportedRegionOp(p))
          return false;
        for (Region &r : p->getRegions())
          if (!r.empty() && !r.hasOneBlock())
            return false;
        if (isa<scf::WhileOp, scf::ForOp, affine::AffineForOp>(p))
          inLoop = true;
      }
    }
    return inLoop;
  }

  void run() {
    computeLiveness();
    Type t = slot.getType().getElementType();
    rewriter.setInsertionPointAfter(slot);
    Value undef = rewriter.create<LLVM::UndefOp>(slot.getLoc(), t);
    promoteBlock(*slot->getBlock(), undef);
    for (Operation *op : dead)
      rewriter.eraseOp(op);
    rewriter.eraseOp(slot);
  }

private:
  bool usesInside(Operation *op) {
    bool found = false;
    op->walk([&](Operation *inner) {
      if (accessedMemref(inner) == slot.getResult()) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  }

  enum class First { None, Exposed, Defined };

  /// Whether the operations [begin, end) may read the value before writing
  /// it (a load first on some path: Exposed), surely write it first
  /// (Defined), or neither (None: no read first, and some path leaves the
  /// range without writing).
  First firstAccess(Block::iterator begin, Block::iterator end) {
    for (Operation &op : llvm::make_range(begin, end)) {
      if (accessedMemref(&op) == slot.getResult())
        return isLoad(&op) ? First::Exposed : First::Defined;
      if (op.getNumRegions() == 0 || !usesInside(&op))
        continue;
      if (isa<scf::IfOp, affine::AffineIfOp>(op)) {
        First t = op.getRegion(0).empty() ? First::None
                                          : firstAccess(op.getRegion(0).front());
        First e = op.getRegion(1).empty() ? First::None
                                          : firstAccess(op.getRegion(1).front());
        if (t == First::Exposed || e == First::Exposed)
          return First::Exposed;
        if (t == First::Defined && e == First::Defined)
          return First::Defined;
        continue;
      }
      if (auto w = dyn_cast<scf::WhileOp>(op)) {
        First b = firstAccess(w.getBefore().front());
        if (b != First::None)
          return b; // the condition runs at least once
        if (firstAccess(w.getAfter().front()) == First::Exposed)
          return First::Exposed;
        continue;
      }
      // A loop body may run zero times: only a read in it counts.
      for (Region &r : op.getRegions())
        if (!r.empty() && firstAccess(r.front()) == First::Exposed)
          return First::Exposed;
    }
    return First::None;
  }

  First firstAccess(Block &block) {
    return firstAccess(block.begin(), block.end());
  }

  /// Whether the value held right after `op` may still be read: a path from
  /// there reaches a load before a store. A path that leaves `op`'s block
  /// continues where control goes next: after an if, at the next iteration
  /// or the exit of a loop, into a while's body or out of it. The function's
  /// end ends the scalar's life.
  bool liveAfter(Operation *op) {
    Block *block = op->getBlock();
    First rest = firstAccess(std::next(Block::iterator(op)), block->end());
    if (rest != First::None)
      return rest == First::Exposed;
    return liveAtEnd(block);
  }

  /// Whether the value at the end of `block` may still be read.
  bool liveAtEnd(Block *block) {
    Operation *parent = block->getParentOp();
    if (isa<func::FuncOp>(parent))
      return false;
    if (isa<scf::IfOp, affine::AffineIfOp>(parent))
      return liveAfter(parent);
    if (auto w = dyn_cast<scf::WhileOp>(parent)) {
      if (block == &w.getBefore().front()) // the condition: body or exit
        return liveAtStart(w.getAfter().front(), w) || liveAfter(w);
      return liveAtWhileStart(w); // the body: back to the condition
    }
    // A loop body's end: the next iteration or the exit.
    return liveAtStart(*block, parent) || liveAfter(parent);
  }

  /// Whether the value at the start of `body` (of `loop`, a for loop or a
  /// while's body) may be read. Going round the loop again adds no path
  /// that the body itself does not already have.
  bool liveAtStart(Block &body, Operation *loop) {
    First f = firstAccess(body);
    if (f != First::None)
      return f == First::Exposed;
    if (auto w = dyn_cast<scf::WhileOp>(loop)) // the body ends at the condition
      return liveAtWhileStart(w);
    return liveAfter(loop); // the exit
  }

  /// Whether the value at the start of a while's condition may be read: the
  /// condition reads it, or a path through the condition reaches the body
  /// (which reads it first) or the exit (after which it is read).
  bool liveAtWhileStart(scf::WhileOp w) {
    First fb = firstAccess(w.getBefore().front());
    if (fb != First::None)
      return fb == First::Exposed;
    return firstAccess(w.getAfter().front()) == First::Exposed ||
           liveAfter(w);
  }

  /// The value a loop body starts from: the iteration argument when the
  /// value carried in may still be read (by the body before it writes, or,
  /// on a path through the body that does not write it, after the loop),
  /// else a fresh undefined value. A scalar each iteration writes before
  /// reading (an accumulator reset in an outer loop's body) so carries no
  /// recurrence around that loop, as LLVM's mem2reg builds no live phi for
  /// it (hyungseok lab2: the sum of an inner product reset in the two loops
  /// around it). Decided on the IR before any rewrite (`liveStart`).
  Value bodyStart(Operation *loop, Block &body, Value arg) {
    if (liveStart.lookup(loop))
      return arg;
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&body);
    return rewriter.create<LLVM::UndefOp>(slot.getLoc(), arg.getType());
  }

  /// Records, for every loop that holds an access, whether the value at the
  /// start of its body (a while: of its condition) may be read.
  void computeLiveness() {
    slot->getParentOfType<func::FuncOp>().walk([&](Operation *op) {
      if (!usesInside(op))
        return;
      if (auto w = dyn_cast<scf::WhileOp>(op))
        liveStart[op] = liveAtWhileStart(w);
      else if (isa<scf::ForOp, affine::AffineForOp>(op))
        liveStart[op] = liveAtStart(op->getRegion(0).front(), op);
    });
  }

  /// Threads the value through `block` starting from `cur`; returns the value
  /// at the block's end.
  Value promoteBlock(Block &block, Value cur) {
    SmallVector<Operation *> ops;
    for (Operation &op : block)
      ops.push_back(&op);
    for (Operation *op : ops) {
      if (accessedMemref(op) == slot.getResult()) {
        if (isLoad(op))
          op->getResult(0).replaceAllUsesWith(cur);
        else if (auto s = dyn_cast<affine::AffineStoreOp>(op))
          cur = s.getValueToStore();
        else
          cur = cast<memref::StoreOp>(op).getValueToStore();
        dead.push_back(op);
        continue;
      }
      if (op->getNumRegions() == 0 || !usesInside(op))
        continue;
      if (auto ifOp = dyn_cast<scf::IfOp>(op))
        cur = promoteIf(ifOp, cur);
      else if (auto ifOp = dyn_cast<affine::AffineIfOp>(op))
        cur = promoteAffineIf(ifOp, cur);
      else if (auto w = dyn_cast<scf::WhileOp>(op))
        cur = promoteWhile(w, cur);
      else if (auto loop = dyn_cast<LoopLikeOpInterface>(op))
        cur = promoteLoop(loop, cur);
    }
    return cur;
  }

  Value promoteIf(scf::IfOp ifOp, Value cur) {
    Value thenV = promoteBlock(*ifOp.thenBlock(), cur);
    Value elseV = ifOp.elseBlock() ? promoteBlock(*ifOp.elseBlock(), cur) : cur;
    if (thenV == cur && elseV == cur)
      return cur;
    SmallVector<Type> types(ifOp.getResultTypes());
    types.push_back(cur.getType());
    rewriter.setInsertionPoint(ifOp);
    auto newIf = rewriter.create<scf::IfOp>(ifOp.getLoc(), types,
                                            ifOp.getCondition(),
                                            /*addThenBlock=*/false,
                                            /*addElseBlock=*/false);
    Operation *thenYield = ifOp.thenBlock()->getTerminator();
    thenYield->insertOperands(thenYield->getNumOperands(), thenV);
    if (ifOp.elseBlock()) {
      Operation *elseYield = ifOp.elseBlock()->getTerminator();
      elseYield->insertOperands(elseYield->getNumOperands(), elseV);
    }
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty()) {
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    } else {
      Block *b = rewriter.createBlock(&newIf.getElseRegion());
      rewriter.setInsertionPointToEnd(b);
      rewriter.create<scf::YieldOp>(ifOp.getLoc(), ValueRange{elseV});
    }
    copyAttrs(ifOp, newIf);
    for (auto [o, n] : llvm::zip(ifOp.getResults(), newIf.getResults()))
      o.replaceAllUsesWith(n);
    rewriter.eraseOp(ifOp);
    return newIf.getResults().back();
  }

  Value promoteAffineIf(affine::AffineIfOp ifOp, Value cur) {
    Value thenV = promoteBlock(*ifOp.getThenBlock(), cur);
    Value elseV =
        ifOp.hasElse() ? promoteBlock(*ifOp.getElseBlock(), cur) : cur;
    if (thenV == cur && elseV == cur)
      return cur;
    SmallVector<Type> types(ifOp.getResultTypes());
    types.push_back(cur.getType());
    rewriter.setInsertionPoint(ifOp);
    // An affine.if with results needs an else region; the builder's blocks
    // are replaced by the old ones (or a block yielding the value).
    auto newIf = rewriter.create<affine::AffineIfOp>(
        ifOp.getLoc(), types, ifOp.getIntegerSet(), ifOp.getOperands(),
        /*withElseRegion=*/true);
    newIf.getThenRegion().getBlocks().clear();
    newIf.getElseRegion().getBlocks().clear();
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    Operation *thenYield = newIf.getThenBlock()->getTerminator();
    thenYield->insertOperands(thenYield->getNumOperands(), thenV);
    if (ifOp.hasElse()) {
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());
      Operation *elseYield = newIf.getElseBlock()->getTerminator();
      elseYield->insertOperands(elseYield->getNumOperands(), elseV);
    } else {
      Block *b = rewriter.createBlock(&newIf.getElseRegion());
      rewriter.setInsertionPointToEnd(b);
      rewriter.create<affine::AffineYieldOp>(ifOp.getLoc(),
                                             ValueRange{elseV});
    }
    copyAttrs(ifOp, newIf);
    for (auto [o, n] : llvm::zip(ifOp.getResults(), newIf.getResults()))
      o.replaceAllUsesWith(n);
    rewriter.eraseOp(ifOp);
    return newIf.getResults().back();
  }

  Value promoteWhile(scf::WhileOp w, Value cur) {
    Type t = cur.getType();
    Location loc = w.getLoc();
    Block &before = w.getBefore().front();
    Block &after = w.getAfter().front();
    // The condition starts from the carried value when it may be read: by
    // the condition or the body before they write it, or after the loop (the
    // loop's result is the value the condition passes on when it exits).
    BlockArgument bArg = before.addArgument(t, loc);
    Value bStart = bArg;
    if (!liveStart.lookup(w)) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&before);
      bStart = rewriter.create<LLVM::UndefOp>(loc, t);
    }
    Value bEnd = promoteBlock(before, bStart);
    auto cond = cast<scf::ConditionOp>(before.getTerminator());
    cond.getArgsMutable().append(bEnd);
    BlockArgument aArg = after.addArgument(t, loc);
    Value aEnd = promoteBlock(after, aArg);
    auto yield = cast<scf::YieldOp>(after.getTerminator());
    yield.getResultsMutable().append(aEnd);
    SmallVector<Value> inits(w.getInits());
    inits.push_back(cur);
    SmallVector<Type> types(w.getResultTypes());
    types.push_back(t);
    rewriter.setInsertionPoint(w);
    auto nw = rewriter.create<scf::WhileOp>(loc, types, inits);
    nw.getBefore().takeBody(w.getBefore());
    nw.getAfter().takeBody(w.getAfter());
    copyAttrs(w, nw);
    for (auto [o, n] : llvm::zip(w.getResults(), nw.getResults()))
      o.replaceAllUsesWith(n);
    rewriter.eraseOp(w);
    return nw.getResults().back();
  }

  Value promoteLoop(LoopLikeOpInterface loop, Value cur) {
    Operation *original = loop.getOperation();
    FailureOr<LoopLikeOpInterface> nl = loop.replaceWithAdditionalYields(
        rewriter, ValueRange{cur}, /*replaceInitOperandUsesInLoop=*/false,
        [&](OpBuilder &b, Location, ArrayRef<BlockArgument> args) {
          // The builder stands at the body's yield; the body's operations
          // may still be in the old loop's block (scf.for moves them into
          // the new loop after this call), so thread the value through the
          // block that holds the yield.
          Block *body = b.getInsertionBlock();
          return SmallVector<Value>{
              promoteBlock(*body, bodyStart(original, *body, args.front()))};
        });
    if (failed(nl))
      return cur; // not reached: promotable() admits scf.for / affine.for
    return (*nl)->getResults().back();
  }

  static void copyAttrs(Operation *from, Operation *to) {
    for (NamedAttribute a : from->getDiscardableAttrs())
      to->setDiscardableAttr(a.getName(), a.getValue());
  }

  memref::AllocaOp slot;
  IRRewriter &rewriter;
  SmallVector<Operation *> dead;
  llvm::DenseMap<Operation *, bool> liveStart;
};

struct XlxLoopScalarPromotePass
    : public PassWrapper<XlxLoopScalarPromotePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XlxLoopScalarPromotePass)
  StringRef getArgument() const override {
    return "xlx-loop-scalar-promote";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect,
                    affine::AffineDialect>();
  }
  void runOnOperation() override {
    SmallVector<memref::AllocaOp> candidates;
    getOperation().walk([&](memref::AllocaOp a) {
      MemRefType t = a.getType();
      if (t.getRank() == 0 && (t.getElementType().isIntOrIndexOrFloat()))
        candidates.push_back(a);
    });
    IRRewriter rewriter(&getContext());
    for (memref::AllocaOp a : candidates) {
      Promoter p(a, rewriter);
      if (!p.promotable())
        continue;
      p.run();
    }
  }
};

} // namespace

namespace cgeist {
std::unique_ptr<mlir::Pass> createXlxLoopScalarPromotePass() {
  return std::make_unique<XlxLoopScalarPromotePass>();
}
} // namespace cgeist

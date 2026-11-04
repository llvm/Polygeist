//===- XlxDeadStore.cc - Overwritten stores of the Vitis HLS headers ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// --xlx-headers: delete a store that the next memory access in the same block
// overwrites, location for location, with nothing in between that touches
// memory.
//
// The Vitis HLS headers' conversions start with a guard store that the
// following code always overwrites, e.g. the ap_fixed_base conversion
// (etc/ap_fixed_base.h:691, equal_constexpr):
//
//   if (!op) Base::V = 0;
//   ...
//   Base::V = op.V >> sh_amt;            // unconditional
//
// After inlining, cgeist writes such a `Base::V` straight into the user's
// array element, so `out[0] += a * b` becomes
//
//   scf.if %iszero { affine.store %c0, %out[0] }
//   affine.store %v, %out[0]
//
// Vitis' LLVM pipeline removes the guarded store: in the Vitis run of
// forgebench/mult_op_module_dot the dot_product loop body of db/a.o.1.ll (the
// first reflow stage) has only the unconditional `store i16 %trunc.ln138.N,
// i16* %output` per unrolled iteration, no conditional store. Left in place
// it is an extra memory operation in hls-est's schedule (the same loop
// scheduled with and without it: 15 RAM_1WNR copies without, 1 with).
//
// What is deleted: an affine.store / memref.store, or an scf.if / affine.if
// without results and without an else branch whose then branch holds only
// memory-effect-free ops and stores to one location, when the next op after
// it in the block that has any memory effect is an unconditional store to the
// same location (the same memref value and the same indices: identical index
// operands, and for affine.store the identical map). Only memory-effect-free
// ops may lie in between, so no load can observe the deleted store.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

using namespace mlir;

namespace {

/// The location a store writes: the memref, the index operands and, for an
/// affine.store, its map.
struct StoreLoc {
  Value memref;
  AffineMap map; // null for memref.store
  SmallVector<Value, 4> operands;

  bool operator==(const StoreLoc &o) const {
    return memref == o.memref && map == o.map && operands == o.operands;
  }
};

std::optional<StoreLoc> storeLoc(Operation *op) {
  if (auto s = dyn_cast<affine::AffineStoreOp>(op)) {
    StoreLoc l{s.getMemRef(), s.getAffineMap(), {}};
    l.operands.append(s.getMapOperands().begin(), s.getMapOperands().end());
    return l;
  }
  if (auto s = dyn_cast<memref::StoreOp>(op)) {
    StoreLoc l{s.getMemRef(), AffineMap(), {}};
    l.operands.append(s.getIndices().begin(), s.getIndices().end());
    return l;
  }
  return std::nullopt;
}

/// The single location `op` may write when it is a store, or a guarded
/// (then-only, result-less) if whose then branch writes only that location
/// and has no other memory effect. std::nullopt otherwise.
std::optional<StoreLoc> deletableWrite(Operation *op) {
  if (auto l = storeLoc(op))
    return l;
  Block *then = nullptr;
  if (auto i = dyn_cast<scf::IfOp>(op)) {
    if (i.getNumResults() || !i.getElseRegion().empty())
      return std::nullopt;
    then = i.thenBlock();
  } else if (auto i = dyn_cast<affine::AffineIfOp>(op)) {
    if (i.getNumResults() || i.hasElse())
      return std::nullopt;
    then = i.getThenBlock();
  } else {
    return std::nullopt;
  }
  std::optional<StoreLoc> loc;
  for (Operation &inner : *then) {
    if (auto l = storeLoc(&inner)) {
      if (loc && !(*loc == *l))
        return std::nullopt;
      // The memref and the index operands must be defined outside the if, so
      // that the location means the same thing at the later store.
      if (op->isAncestor(l->memref.getParentBlock()->getParentOp()))
        return std::nullopt;
      for (Value v : l->operands)
        if (op->isAncestor(v.getParentBlock()->getParentOp()))
          return std::nullopt;
      loc = l;
      continue;
    }
    if (!isMemoryEffectFree(&inner))
      return std::nullopt;
  }
  return loc;
}

/// True when the next op after `op` in its block with a memory effect is a
/// store to `loc`.
bool overwrittenLater(Operation *op, const StoreLoc &loc) {
  for (Operation *next = op->getNextNode(); next; next = next->getNextNode()) {
    if (auto l = storeLoc(next))
      return *l == loc;
    if (!isMemoryEffectFree(next))
      return false;
  }
  return false;
}

} // namespace

namespace cgeist {
void eraseOverwrittenStores(mlir::ModuleOp module) {
  SmallVector<Operation *> dead;
  module.walk([&](Operation *op) {
    if (auto loc = deletableWrite(op))
      if (overwrittenLater(op, *loc))
        dead.push_back(op);
  });
  // The walk is post-order: a store inside a guarded if comes before the if.
  // Erase only the outermost dead ops (an op nested in another dead op goes
  // with it).
  llvm::SmallPtrSet<Operation *, 16> deadSet(dead.begin(), dead.end());
  for (Operation *op : dead) {
    bool nested = false;
    for (Operation *p = op->getParentOp(); p && !nested; p = p->getParentOp())
      nested = deadSet.contains(p);
    deadSet.erase(op);
    if (nested)
      continue;
    // The guard's condition and the stored value may be dead now too.
    llvm::SetVector<Operation *> defs;
    op->walk([&](Operation *o) {
      for (Value v : o->getOperands())
        if (Operation *d = v.getDefiningOp())
          if (!op->isAncestor(d))
            defs.insert(d);
    });
    op->erase();
    while (!defs.empty()) {
      Operation *d = defs.pop_back_val();
      if (!isOpTriviallyDead(d))
        continue;
      for (Value v : d->getOperands())
        if (Operation *dd = v.getDefiningOp())
          defs.insert(dd);
      d->erase();
    }
  }
}
} // namespace cgeist

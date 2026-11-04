//===- XlxProxyScalarize.cc - Scalarise ap_range_ref-like proxies ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Vitis HLS headers return proxy objects from x.range(h, l), x[i], (a, b)
// and friends: ap_range_ref / ap_bit_ref / ap_concat_ref / af_range_ref /
// af_bit_ref, structs of a reference to the ap value plus bit indices
// (etc/ap_int_ref.h:295, etc/ap_fixed_ref.h). cgeist gives such a struct a
// `memref<1x!llvm.struct<(memref<?xiN>, i32, i32)>>` alloca whose fields are
// written and read through polygeist.memref2pointer / llvm.getelementptr /
// llvm.store / llvm.load, plus whole-struct copies (affine.load/store of the
// element). A memref of a struct holding a memref is not valid MLIR once
// printed, and hls-est cannot read it.
//
// After the proxies' always_inline member functions are inlined
// (--hls-inline-polygeist-ops), a proxy is filled, copied and read in its
// caller, possibly inside the scf / affine regions the inlined bodies become.
// This pass forwards the stored field values to the loads in program order
// over the structured regions (through whole-struct copies and
// llvm.extractvalue) and deletes the proxy allocas:
//  - a region of an op that runs it at most once (scf.if, affine.if,
//    scf.execute_region, scf.index_switch) starts from the state before the
//    op; any other region (loops) starts with the fields it stores unknown;
//  - after an op, the fields stored inside its regions are unknown;
//  - a region with more than one block is not analysed.
// The same applies to any other single-element struct alloca, in particular
// the headers' type-punning unions (doubleToRawBits: written as double, read
// as uint64): a field read as another integer / float type of the same width
// becomes an arith.bitcast of the stored value.
// A proxy that cannot be resolved this way (passed to a call, its address
// stored, a dynamic field index, a field read while unknown, a whole-struct
// value that escapes) is left as it is, together with every proxy whose
// value flows into it; the others are still scalarised.
// A proxy whose address is stored into a field of another proxy (the
// ap_range_ref / ap_bit_ref operands of an ap_concat_ref, which holds
// references to them, etc/ap_int_ref.h) is left in the first round; once the
// outer proxy is scalarised, its field loads are the inner proxy's address,
// so the pass repeats until a round scalarises nothing.
// Under --xlx-headers the driver then rejects any memref of a struct that
// still holds a memref (reportUnscalarisedProxies): printed, it is not valid
// MLIR, and hls-est cannot read it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {

/// How one op touches a proxy alloca.
struct Access {
  enum Kind { FieldStore, FieldLoad, WholeStore, WholeLoad } kind;
  memref::AllocaOp alloca;
  unsigned field = 0;
};

struct ProxyInfo {
  memref::AllocaOp alloca;
  LLVM::LLVMStructType type;
  llvm::SmallVector<Operation *> erase; ///< accesses and address ops
};

/// The field's own type, or an integer / float of the same width: a union
/// read through another member (the headers' doubleToRawBits, a
/// `union { double d; uint64_t i; }` written as double and read as
/// integer) is that value's bit pattern.
bool sameBits(Type a, Type b) {
  if (a == b)
    return true;
  return a.isIntOrFloat() && b.isIntOrFloat() &&
         a.getIntOrFloatBitWidth() == b.getIntOrFloatBitWidth();
}

/// The struct element type of a single-element memref alloca, or null.
LLVM::LLVMStructType proxyType(memref::AllocaOp alloca) {
  MemRefType mt = alloca.getType();
  if (mt.getRank() > 1 || (mt.getRank() == 1 && mt.getDimSize(0) != 1))
    return nullptr;
  auto st = mt.getElementType().dyn_cast<LLVM::LLVMStructType>();
  if (!st || st.isOpaque())
    return nullptr;
  return st;
}

class ProxyAnalysis {
public:
  llvm::DenseMap<Operation *, Access> accesses;
  llvm::SmallVector<ProxyInfo> proxies;

  /// Classify every use of the alloca; false if one is not understood.
  bool add(memref::AllocaOp alloca) {
    ProxyInfo P;
    P.alloca = alloca;
    P.type = proxyType(alloca);
    if (!P.type)
      return false;
    llvm::DenseMap<Operation *, Access> mine;
    llvm::SmallVector<Value> worklist{alloca.getResult()};
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      for (Operation *user : v.getUsers()) {
        if (auto cast = dyn_cast<memref::CastOp>(user)) {
          worklist.push_back(cast.getResult());
          P.erase.push_back(cast);
          continue;
        }
        if (auto m2p = dyn_cast<polygeist::Memref2PointerOp>(user)) {
          P.erase.push_back(m2p);
          if (!addPointer(P, m2p.getResult(), 0, mine))
            return false;
          continue;
        }
        Access A{Access::WholeLoad, alloca, 0};
        if (isa<affine::AffineLoadOp, memref::LoadOp>(user)) {
          A.kind = Access::WholeLoad;
        } else if (auto st = dyn_cast<affine::AffineStoreOp>(user)) {
          if (st.getValueToStore() == v)
            return false; // the proxy's address escapes into memory
          A.kind = Access::WholeStore;
        } else if (auto st = dyn_cast<memref::StoreOp>(user)) {
          if (st.getValueToStore() == v)
            return false;
          A.kind = Access::WholeStore;
        } else {
          return false; // a call or anything else: not a local proxy
        }
        mine[user] = A;
        P.erase.push_back(user);
      }
    }
    if (mine.empty())
      return false;
    for (auto &kv : mine)
      accesses[kv.first] = kv.second;
    proxies.push_back(P);
    return true;
  }

private:
  bool addPointer(ProxyInfo &P, Value ptr, unsigned field,
                  llvm::DenseMap<Operation *, Access> &mine) {
    for (Operation *user : ptr.getUsers()) {
      if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
        if (field != 0 || !gep.getDynamicIndices().empty())
          return false;
        auto raw = gep.getRawConstantIndices();
        std::optional<Type> elem = gep.getElemType();
        if (raw.size() != 2 || raw[0] != 0 || !elem || *elem != P.type ||
            raw[1] < 0 || (size_t)raw[1] >= P.type.getBody().size())
          return false;
        P.erase.push_back(gep);
        if (!addPointer(P, gep.getResult(), raw[1], mine))
          return false;
        continue;
      }
      Access A{Access::FieldLoad, P.alloca, field};
      Type fieldTy = P.type.getBody()[field];
      if (auto ld = dyn_cast<LLVM::LoadOp>(user)) {
        if (!sameBits(ld.getType(), fieldTy))
          return false;
        A.kind = Access::FieldLoad;
      } else if (auto st = dyn_cast<LLVM::StoreOp>(user)) {
        if (st.getValue() == ptr || !sameBits(st.getValue().getType(), fieldTy))
          return false;
        A.kind = Access::FieldStore;
      } else {
        return false;
      }
      mine[user] = A;
      P.erase.push_back(user);
    }
    return true;
  }
};

/// Program-order forwarding of the proxies' field values over the function's
/// structured regions. Records the proxies it cannot resolve in `failed`.
class Forwarder {
public:
  using Fields = llvm::SmallVector<Value>;
  using State = llvm::DenseMap<Operation *, Fields>; // alloca -> fields

  Forwarder(ProxyAnalysis &PA) : PA(PA) {}

  ProxyAnalysis &PA;
  llvm::DenseMap<Value, Value> replace;
  llvm::SmallVector<Operation *> extracts;
  llvm::DenseSet<Operation *> failed; // allocas
  /// Whole-struct loads: their field values and the proxy they came from.
  llvm::DenseMap<Value, std::pair<Fields, Operation *>> structValues;

  void run(func::FuncOp func) {
    State state;
    walkRegion(func.getBody(), state);
    // Every use of a whole-struct load must have been resolved.
    for (auto &kv : structValues)
      for (Operation *user : kv.first.getUsers())
        if (!PA.accesses.count(user) && !llvm::is_contained(extracts, user))
          failed.insert(kv.second.second);
  }

private:
  unsigned numFields(memref::AllocaOp a) {
    return a.getType()
        .getElementType()
        .cast<LLVM::LLVMStructType>()
        .getBody()
        .size();
  }

  Fields &fieldsOf(State &state, memref::AllocaOp a) {
    Fields &cur = state[a];
    if (cur.empty())
      cur.assign(numFields(a), Value());
    return cur;
  }

  /// Every proxy accessed inside `region` fails.
  void failAccessesIn(Region &region) {
    region.walk([&](Operation *op) {
      auto it = PA.accesses.find(op);
      if (it != PA.accesses.end())
        failed.insert(it->second.alloca);
    });
  }

  /// Mark the fields stored anywhere inside `op`'s regions unknown.
  void forgetStoresIn(Operation *op, State &state) {
    for (Region &region : op->getRegions())
      region.walk([&](Operation *inner) {
        auto it = PA.accesses.find(inner);
        if (it == PA.accesses.end())
          return;
        const Access &A = it->second;
        if (A.kind == Access::FieldStore)
          fieldsOf(state, A.alloca)[A.field] = Value();
        else if (A.kind == Access::WholeStore)
          fieldsOf(state, A.alloca).assign(numFields(A.alloca), Value());
      });
  }

  static bool runsAtMostOnce(Operation *op) {
    return isa<scf::IfOp, affine::AffineIfOp, scf::ExecuteRegionOp,
               scf::IndexSwitchOp>(op);
  }

  void walkRegion(Region &region, State &state) {
    if (region.empty())
      return;
    if (!region.hasOneBlock()) {
      walkBlocks(region, state);
      return;
    }
    walkBlock(region.front(), state);
  }

  /// The fields two paths agree on (a field the paths disagree on, or that
  /// one of them does not know, is unknown).
  static State merge(const State &a, const State &b) {
    State out;
    for (auto &kv : a) {
      auto it = b.find(kv.first);
      if (it == b.end())
        continue;
      Fields f(kv.second.size(), Value());
      for (size_t i = 0; i < f.size() && i < it->second.size(); ++i)
        if (kv.second[i] == it->second[i])
          f[i] = kv.second[i];
      out[kv.first] = f;
    }
    return out;
  }

  /// A region of several blocks (a `switch` is an scf.execute_region whose
  /// blocks are its cases, CGStmt.cc VisitSwitchStmt): when every branch
  /// goes to a later block (no cycle), the blocks are walked in order, each
  /// starting from what its predecessors agree on; a proxy filled and read
  /// in one case is resolved there. With a branch back to an earlier block
  /// every block but the entry starts with the fields unknown.
  void walkBlocks(Region &region, State &state) {
    llvm::DenseMap<Block *, unsigned> order;
    for (Block &b : region)
      order[&b] = order.size();
    bool forward = true;
    for (Block &b : region)
      for (Block *succ : b.getSuccessors())
        forward &= order.lookup(succ) > order.lookup(&b);
    llvm::DenseMap<Block *, State> exitState;
    for (Block &b : region) {
      State in;
      if (&b == &region.front()) {
        in = state;
      } else if (forward) {
        bool first = true;
        for (Block *pred : b.getPredecessors()) {
          auto it = exitState.find(pred);
          if (it == exitState.end()) {
            in.clear();
            first = false;
            break;
          }
          in = first ? it->second : merge(in, it->second);
          first = false;
        }
      }
      walkBlock(b, in);
      exitState[&b] = std::move(in);
    }
  }

  void walkBlock(Block &block, State &state) {
    for (Operation &op : block) {
      if (op.getNumRegions() != 0) {
        bool once = runsAtMostOnce(&op);
        for (Region &region : op.getRegions()) {
          State inner = state;
          if (!once)
            forgetStoresIn(&op, inner);
          walkRegion(region, inner);
        }
        forgetStoresIn(&op, state);
        continue;
      }
      visit(op, state);
    }
  }

  void visit(Operation &op, State &state) {
    // llvm.extractvalue of a whole-struct load.
    if (auto ex = dyn_cast<LLVM::ExtractValueOp>(&op)) {
      auto it = structValues.find(ex.getContainer());
      if (it == structValues.end())
        return;
      const Fields &fs = it->second.first;
      int64_t k = ex.getPosition().size() == 1 ? ex.getPosition()[0] : -1;
      if (k < 0 || (size_t)k >= fs.size() || !fs[k] ||
          fs[k].getType() != ex.getType()) {
        failed.insert(it->second.second);
        return;
      }
      replace[ex.getResult()] = fs[k];
      extracts.push_back(ex);
      return;
    }
    auto ai = PA.accesses.find(&op);
    if (ai == PA.accesses.end())
      return;
    const Access &A = ai->second;
    Fields &cur = fieldsOf(state, A.alloca);
    switch (A.kind) {
    case Access::FieldStore:
      cur[A.field] = cast<LLVM::StoreOp>(op).getValue();
      break;
    case Access::FieldLoad: {
      Value v = cur[A.field];
      if (!v) {
        failed.insert(A.alloca); // read while unknown
        break;
      }
      auto r = replace.find(v);
      // Read the forwarded value before inserting: operator[] may grow the
      // map and free the bucket `r` points into.
      Value forwarded = r == replace.end() ? v : r->second;
      replace[op.getResult(0)] = forwarded;
      break;
    }
    case Access::WholeLoad:
      structValues[op.getResult(0)] = {cur, A.alloca};
      break;
    case Access::WholeStore: {
      Value v = isa<affine::AffineStoreOp>(op)
                    ? cast<affine::AffineStoreOp>(op).getValueToStore()
                    : cast<memref::StoreOp>(op).getValueToStore();
      auto it = structValues.find(v);
      if (it != structValues.end()) {
        cur = it->second.first;
      } else if (v.getDefiningOp<LLVM::UndefOp>()) {
        cur.assign(cur.size(), Value());
      } else {
        failed.insert(A.alloca); // a struct value from elsewhere
        cur.assign(cur.size(), Value());
      }
      break;
    }
    }
  }
};

struct XlxProxyScalarizePass
    : public PassWrapper<XlxProxyScalarizePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XlxProxyScalarizePass)
  StringRef getArgument() const final { return "xlx-proxy-scalarize"; }
  StringRef getDescription() const final {
    return "Scalarise the Vitis HLS headers' range / bit / concat proxies";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    splitProxyBlockArguments(func);
    // Each round erases at least one alloca, so this terminates.
    while (scalarizeOnce(func)) {
    }
  }

  /// A block that receives proxies as block arguments: the canonicalizer
  /// merges the identical cases of a `switch` (`case 0: d.range(7, 0) = a;
  /// break; default: d.range(7, 0) = b;`, vision xf_threshold.hpp) into one
  /// block whose arguments are each case's proxy. The forwarding cannot see
  /// a proxy through a block argument, so the block is copied once per
  /// incoming edge with the edge's values in place of the arguments (the
  /// blocks before the merge). Only blocks whose every predecessor is a
  /// branch op of the same region are split; at most kMaxCopies copies.
  static bool isProxyArgument(BlockArgument a);

  void splitProxyBlockArguments(func::FuncOp func) {
    constexpr unsigned kMaxCopies = 64;
    SmallVector<Block *> work;
    func.walk([&](Block *b) {
      if (!b->isEntryBlock() &&
          llvm::any_of(b->getArguments(), isProxyArgument))
        work.push_back(b);
    });
    for (Block *b : work) {
      // The incoming edges: (terminator, successor index).
      SmallVector<std::pair<Operation *, unsigned>> edges;
      bool ok = true;
      for (auto it = b->pred_begin(), e = b->pred_end(); it != e; ++it) {
        Operation *term = (*it)->getTerminator();
        auto br = dyn_cast<BranchOpInterface>(term);
        if (!br) {
          ok = false;
          break;
        }
        edges.push_back({term, it.getSuccessorIndex()});
      }
      if (!ok || edges.empty() || edges.size() > kMaxCopies)
        continue;
      for (auto [term, idx] : edges) {
        auto br = cast<BranchOpInterface>(term);
        SuccessorOperands ops = br.getSuccessorOperands(idx);
        if (ops.getProducedOperandCount() != 0 ||
            ops.size() != b->getNumArguments()) {
          ok = false;
          break;
        }
      }
      // A value of the block used in another block would need a merge.
      for (Operation &op : *b)
        for (Value r : op.getResults())
          for (Operation *u : r.getUsers())
            if (b->findAncestorOpInBlock(*u) == nullptr)
              ok = false;
      if (!ok)
        continue;
      Region *region = b->getParent();
      for (auto [term, idx] : edges) {
        auto br = cast<BranchOpInterface>(term);
        SuccessorOperands ops = br.getSuccessorOperands(idx);
        IRMapping map;
        for (auto [arg, v] : llvm::zip(b->getArguments(),
                                       ops.getForwardedOperands()))
          map.map(arg, v);
        Block *copy = new Block();
        region->getBlocks().insertAfter(Region::iterator(b), copy);
        for (Operation &op : *b)
          copy->push_back(op.clone(map));
        ops.erase(0, ops.size());
        term->setSuccessor(copy, idx);
      }
      // Every edge goes to a copy now.
      b->dropAllDefinedValueUses();
      b->erase();
    }
  }

  /// One round over the function's proxy allocas. True if at least one was
  /// scalarised and erased.
  bool scalarizeOnce(func::FuncOp func) {
    llvm::SetVector<Operation *> candidates;
    func.walk([&](memref::AllocaOp a) {
      if (proxyType(a))
        candidates.insert(a);
    });
    // Drop the proxies that cannot be resolved, and with them every proxy
    // whose whole-struct value flows into one of them, until the rest
    // resolves.
    while (!candidates.empty()) {
      ProxyAnalysis PA;
      llvm::SmallVector<Operation *> rejected;
      for (Operation *a : candidates)
        if (!PA.add(cast<memref::AllocaOp>(a)))
          rejected.push_back(a);
      if (!rejected.empty()) {
        for (Operation *a : rejected)
          candidates.remove(a);
        continue;
      }
      Forwarder F(PA);
      F.run(func);
      if (!F.failed.empty()) {
        for (Operation *a : F.failed)
          candidates.remove(a);
        continue;
      }
      return apply(PA, F);
    }
    return false;
  }

  /// Rewrite the forwarded loads and erase the proxies. True if at least one
  /// proxy alloca was erased.
  bool apply(ProxyAnalysis &PA, Forwarder &F) {
    // Resolve chains (a forwarded value that is itself replaced).
    for (auto &kv : F.replace) {
      Value v = kv.second;
      for (auto it = F.replace.find(v); it != F.replace.end();
           it = F.replace.find(v))
        v = it->second;
      if (v.getType() != kv.first.getType()) {
        // A union member read as another type of the same width (sameBits).
        Operation *at = kv.first.getDefiningOp();
        OpBuilder b(at);
        v = b.create<arith::BitcastOp>(at->getLoc(), kv.first.getType(), v);
      }
      kv.first.replaceAllUsesWith(v);
    }
    for (Operation *ex : F.extracts)
      ex->erase();
    // Erase the accesses, then the address ops, then the allocas (users
    // first: the erase lists are in use-before-def discovery order, so walk
    // them until nothing is left).
    llvm::SetVector<Operation *> dead;
    for (ProxyInfo &P : PA.proxies)
      for (Operation *op : P.erase)
        dead.insert(op);
    while (!dead.empty()) {
      llvm::SmallVector<Operation *> ready;
      for (Operation *op : dead)
        if (op->use_empty())
          ready.push_back(op);
      if (ready.empty())
        break;
      for (Operation *op : ready) {
        dead.remove(op);
        op->erase();
      }
    }
    bool erased = false;
    for (ProxyInfo &P : PA.proxies)
      if (P.alloca->use_empty()) {
        P.alloca->erase();
        erased = true;
      }
    return erased;
  }
};

/// A memref whose element type is (or contains) an LLVM struct or array that
/// holds a memref: an unscalarised proxy.
bool holdsMemref(Type t) {
  if (t.isa<MemRefType>())
    return true;
  if (auto st = t.dyn_cast<LLVM::LLVMStructType>())
    return !st.isOpaque() && llvm::any_of(st.getBody(), holdsMemref);
  if (auto at = t.dyn_cast<LLVM::LLVMArrayType>())
    return holdsMemref(at.getElementType());
  return false;
}

bool isProxyMemref(Type t) {
  auto mt = t.dyn_cast<MemRefType>();
  if (!mt)
    return false;
  Type e = mt.getElementType();
  return e.isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>() && holdsMemref(e);
}

/// A block argument holding a proxy (a memref of a struct that holds a
/// memref, which only the headers' proxies are).
bool XlxProxyScalarizePass::isProxyArgument(BlockArgument a) {
  return isProxyMemref(a.getType());
}

} // namespace

namespace cgeist {
std::unique_ptr<mlir::Pass> createXlxProxyScalarizePass() {
  return std::make_unique<XlxProxyScalarizePass>();
}
} // namespace cgeist

namespace cgeist {
/// --xlx-headers: an error at every op that still defines or takes a memref of
/// a struct holding a memref (a proxy the pass could not scalarise, e.g. one
/// passed to a call). Such MLIR does not parse back ("invalid memref element
/// type"), so cgeist fails instead of printing it.
mlir::LogicalResult reportUnscalarisedProxies(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    bool bad = llvm::any_of(op->getResultTypes(), isProxyMemref);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        bad |= llvm::any_of(block.getArgumentTypes(), isProxyMemref);
    if (auto f = dyn_cast<func::FuncOp>(op))
      bad |= llvm::any_of(f.getFunctionType().getInputs(), isProxyMemref) ||
             llvm::any_of(f.getFunctionType().getResults(), isProxyMemref);
    if (auto g = dyn_cast<memref::GlobalOp>(op))
      bad |= isProxyMemref(g.getType());
    if (!bad)
      return;
    found = true;
    op->emitError("a Vitis HLS proxy object (ap_range_ref, ap_bit_ref, "
                  "ap_concat_ref, ...) could not be scalarised: a memref of a "
                  "struct holding a memref is not valid MLIR");
  });
  return mlir::failure(found);
}

/// --xlx-headers: drop every linkonce_odr function (an inline function of the
/// headers) that nothing in the module references any more, as a C++
/// compiler drops an unused inline function. Such a function can still hold
/// a proxy struct, which is only scalarised once inlined into a caller, and
/// whose type does not survive printing.
void dropUnusedLinkonceODR(mlir::ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    // Every symbol referenced anywhere in the module, collected in one walk
    // (asking symbolKnownUseEmpty per function walks the module once per
    // function: minutes on the fixed-point math headers).
    llvm::DenseSet<StringAttr> referenced;
    auto uses = SymbolTable::getSymbolUses(&module.getBodyRegion());
    if (!uses)
      return; // unknown symbol uses: keep everything
    for (const SymbolTable::SymbolUse &use : *uses)
      referenced.insert(use.getSymbolRef().getRootReference());
    llvm::SmallVector<func::FuncOp> dead;
    for (func::FuncOp f : module.getOps<func::FuncOp>()) {
      auto lnk = f->getAttrOfType<LLVM::LinkageAttr>("llvm.linkage");
      if (!lnk || lnk.getLinkage() != LLVM::Linkage::LinkonceODR)
        continue;
      if (!referenced.contains(f.getSymNameAttr()))
        dead.push_back(f);
    }
    for (func::FuncOp f : dead) {
      f.erase();
      changed = true;
    }
  }
}
} // namespace cgeist

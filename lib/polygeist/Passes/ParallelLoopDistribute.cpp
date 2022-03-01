//===- ParallelLoopDistrbute.cpp - Distribute loops around barriers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "polygeist/BarrierUtils.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Passes/Utils.h"
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>

#include <deque>

#define DEBUG_TYPE "cpuify"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

static bool couldWrite(Operation *op) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects<MemoryEffects::Write>(localEffects);
    return localEffects.size() > 0;
  }
  return true;
}

struct Node {
  Operation *O;
  Value V;
  enum Type {
    NONE,
    VAL,
    OP,
  } type;
  Node(Operation *O) : O(O), type(OP){};
  Node(Value V) : V(V), type(VAL){};
  Node() : type(NONE){};
  bool operator<(const Node N) const {
    // TODO are the VAL and OP definitions fine?
    if (type != N.type)
      return type < N.type;
    else if (type == OP)
      return O < N.O;
    else if (type == VAL)
      return V.getAsOpaquePointer() < N.V.getAsOpaquePointer();
    else
      return true;
  }
  void dump() const {
    if (type == VAL)
      llvm::errs() << "[" << V << ", "
                   << "Value"
                   << "]\n";
    else if (type == OP)
      llvm::errs() << "[" << *O << ", "
                   << "Operation"
                   << "]\n";
    else
      llvm::errs() << "["
                   << "NULL"
                   << ", "
                   << "None"
                   << "]\n";
  }
};

typedef std::map<Node, std::set<Node>> Graph;

void dump(Graph &G) {
  for (auto &pair : G) {
    pair.first.dump();
    for (auto &N : pair.second) {
      llvm::errs() << "\t";
      N.dump();
    }
  }
}

namespace mlir {
bool operator<(const Value &a, const Value &b) {
  return a.getAsOpaquePointer() < b.getAsOpaquePointer();
}
} // namespace mlir

/* Returns true if there is a path from source 's' to sink 't' in
   residual graph. Also fills parent[] to store the path */
static inline void bfs(const Graph &G,
                       const llvm::SetVector<Operation *> &Sources,
                       std::map<Node, Node> &parent) {
  std::deque<Node> q;
  for (auto O : Sources) {
    Node N(O);
    parent.emplace(N, Node(nullptr));
    q.push_back(N);
  }

  // Standard BFS Loop
  while (!q.empty()) {
    auto u = q.front();
    q.pop_front();
    auto found = G.find(u);
    if (found == G.end())
      continue;
    for (auto v : found->second) {
      if (parent.find(v) == parent.end()) {
        q.push_back(v);
        parent.emplace(v, u);
      }
    }
  }
}

static bool is_recomputable(Operation &op) {
  // TODO does this mess with minCutCache's logic? can we somehow come across
  // another parallel's cache load there and would that be a problem?
  if (isa<polygeist::CacheLoad>(op))
    return true;
  // TODO is this correct?
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(&op)) {
    return memInterface.hasNoEffect();
  } else {
    return false;
    // return !op.hasTrait<OpTrait::HasRecursiveSideEffects>();
  }
}

static void minCutCache(polygeist::BarrierOp barrier,
                        llvm::SetVector<Value> &Required,
                        llvm::SetVector<Value> &Cache) {
  Graph G;
  llvm::SetVector<Operation *> NonRecomputable;
  for (Operation *it = barrier->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    auto &op = *it;
    if (!is_recomputable(op))
      NonRecomputable.insert(&op);
    for (Value value : op.getResults()) {
      G[Node(&op)].insert(Node(value));
      for (Operation *user : value.getUsers()) {
        // If the user is nested in another op, find its ancestor op that lives
        // in the same block as the barrier.
        while (user->getBlock() != barrier->getBlock())
          user = user->getBlock()->getParentOp();

        G[Node(value)].insert(Node(user));
      }
    }
  }

  Graph Orig = G;

  // Augment the flow while there is a path from source to sink
  while (1) {
    std::map<Node, Node> parent;
    bfs(G, NonRecomputable, parent);
    Node end;
    for (auto req : Required) {
      if (parent.find(Node(req)) != parent.end()) {
        end = Node(req);
        break;
      }
    }
    if (end.type == Node::NONE)
      break;
    // update residual capacities of the edges and reverse edges
    // along the path
    Node v = end;
    while (1) {
      assert(parent.find(v) != parent.end());
      Node u = parent.find(v)->second;
      assert(u.type != Node::NONE);
      assert(G[u].count(v) == 1);
      assert(G[v].count(u) == 0);
      G[u].erase(v);
      G[v].insert(u);
      if (u.type == Node::OP && NonRecomputable.count(u.O))
        break;
      v = u;
    }
  }
  // Flow is maximum now, find vertices reachable from s

  std::map<Node, Node> parent;
  bfs(G, NonRecomputable, parent);

  // All edges that are from a reachable vertex to non-reachable vertex in the
  // original graph
  for (auto &pair : Orig) {
    if (parent.find(pair.first) != parent.end()) {
      for (auto N : pair.second) {
        if (parent.find(N) == parent.end()) {
          assert(pair.first.type == Node::OP && N.type == Node::VAL);
          assert(pair.first.O == N.V.dyn_cast<OpResult>().getOwner());
          Cache.insert(N.V);
        }
      }
    }
  }

  // When ambiguous, push to cache the last value in a computation chain
  // This should be considered in a cost for the max flow
  std::deque<Node> todo;
  for (auto V : Cache)
    todo.push_back(Node(V));

  while (todo.size()) {
    auto N = todo.front();
    todo.pop_front();
    auto found = Orig.find(N);
    // TODO
    break;
  }
}

/// Populates `crossing` with values (op results) that are defined in the same
/// block as `op` and above it, and used by at least one op in the same block
/// below `op`. Uses may be in nested regions.
static void findValuesUsedBelow(polygeist::BarrierOp op,
                                llvm::SetVector<Value> &crossing,
                                llvm::SetVector<Operation *> &preserveAllocas) {
  llvm::SetVector<Operation *> descendantsUsed;

  // A set of pre-barrier operations which are potentially captured by a
  // subsequent pre-barrier operation.
  SmallVector<Operation *> Allocas;

  for (Operation *it = op->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    if (isa<memref::AllocaOp, LLVM::AllocaOp>(it))
      Allocas.push_back(it);
    for (Value value : it->getResults()) {
      for (Operation *user : value.getUsers()) {

        // If the user is nested in another op, find its ancestor op that lives
        // in the same block as the barrier.
        while (user->getBlock() != op->getBlock())
          user = user->getBlock()->getParentOp();

        if (op->isBeforeInBlock(user)) {
          crossing.insert(value);
        }
      }
    }
  }

  llvm::SmallVector<std::pair<Operation *, Operation *>> todo;
  for (auto A : Allocas)
    todo.emplace_back(A, A);

  std::map<Operation *, SmallPtrSet<Operation *, 2>> descendants;
  while (todo.size()) {
    auto current = todo.back();
    todo.pop_back();
    if (descendants[current.first].count(current.second))
      continue;
    descendants[current.first].insert(current.second);
    for (Value value : current.first->getResults()) {
      for (Operation *user : value.getUsers()) {
        Operation *origUser = user;
        while (user->getBlock() != op->getBlock())
          user = user->getBlock()->getParentOp();

        if (!op->isBeforeInBlock(user)) {
          if (couldWrite(origUser) ||
              origUser->hasTrait<OpTrait::IsTerminator>()) {
            preserveAllocas.insert(current.second);
          }
          if (!isa<LLVM::LoadOp, memref::LoadOp, AffineLoadOp>(origUser)) {
            for (auto res : origUser->getResults()) {
              if (crossing.contains(res)) {
                preserveAllocas.insert(current.second);
              }
            }
            todo.emplace_back(user, current.second);
          }
        }
      }
    }
  }

  for (auto v : crossing) {
    if (isa<memref::AllocaOp, LLVM::AllocaOp>(v.getDefiningOp())) {
      preserveAllocas.insert(v.getDefiningOp());
    }
  }
}

/// Returns `true` if the given operation has a BarrierOp transitively nested in
/// one of its regions, but not within any nested ParallelOp.
static bool hasNestedBarrier(Operation *op, SmallVector<BlockArgument> &vals) {
  op->walk([&](polygeist::BarrierOp barrier) {
    // If there is a `parallel` op nested inside the given op (alternatively,
    // the `parallel` op is not an ancestor of `op` or `op` itself), the
    // barrier is considered nested in that `parallel` op and _not_ in `op`.
    for (auto arg : barrier->getOperands()) {
      if (auto ba = arg.dyn_cast<BlockArgument>()) {
        if (auto parallel =
                dyn_cast<scf::ParallelOp>(ba.getOwner()->getParentOp())) {
          if (parallel->isAncestor(op))
            vals.push_back(ba);
        } else if (auto parallel = dyn_cast<AffineParallelOp>(
                       ba.getOwner()->getParentOp())) {
          if (parallel->isAncestor(op))
            vals.push_back(ba);
        } else {
          assert(0 && "unknown barrier arg\n");
        }
      } else if (arg.getDefiningOp<ConstantIndexOp>())
        continue;
      else {
        assert(0 && "unknown barrier arg\n");
      }
    }
  });
  return vals.size();
}

namespace {

#if 0
/// Returns `true` if the loop has a form expected by interchange patterns.
static bool isNormalized(scf::ForOp op) {
  return isDefinedAbove(op.getLowerBound(), op) &&
         isDefinedAbove(op.getStep(), op);
}

/// Transforms a loop to the normal form expected by interchange patterns, i.e.
/// with zero lower bound and unit step.
struct NormalizeLoop : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (isNormalized(op) || !isa<scf::ParallelOp, AffineParallelOp>(op->getParentOp())) {
      LLVM_DEBUG(DBGS() << "[normalize-loop] loop already normalized\n");
      return failure();
    }
    if (op.getNumResults()) {
      LLVM_DEBUG(DBGS() << "[normalize-loop] not handling reduction loops\n");
      return failure();
    }

    OpBuilder::InsertPoint point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op->getParentOp());
    Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    Value one = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);
    rewriter.restoreInsertionPoint(point);

    Value difference = rewriter.create<SubIOp>(op.getLoc(), op.getUpperBound(),
                                               op.getLowerBound());
    Value tripCount = rewriter.create<AddIOp>(
        op.getLoc(),
        rewriter.create<DivUIOp>(
            op.getLoc(), rewriter.create<SubIOp>(op.getLoc(), difference, one),
            op.getStep()),
        one);
    // rewriter.create<CeilDivSIOp>(op.getLoc(), difference, op.getStep());
    auto newForOp =
        rewriter.create<scf::ForOp>(op.getLoc(), zero, tripCount, one);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    Value scaled = rewriter.create<MulIOp>(
        op.getLoc(), newForOp.getInductionVar(), op.getStep());
    Value iv = rewriter.create<AddIOp>(op.getLoc(), op.getLowerBound(), scaled);
    rewriter.mergeBlockBefore(op.getBody(), &newForOp.getBody()->back(), {iv});
    rewriter.eraseOp(&newForOp.getBody()->back());
    rewriter.eraseOp(op);
    return success();
  }
};
#endif

/// Returns `true` if the loop has a form expected by interchange patterns.
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
static bool isNormalized(AffineParallelOp op) {
  auto isZero = [](AffineExpr v) {
    if (auto ce = v.dyn_cast<AffineConstantExpr>())
      return ce.getValue() == 0;
    return false;
  };
  return llvm::all_of(op.lowerBoundsMap().getResults(), isZero) &&
         llvm::all_of(op.getSteps(), [](int64_t s) { return s == 1; });
}

/// Transforms a loop to the normal form expected by interchange patterns, i.e.
/// with zero lower bounds and unit steps.
struct NormalizeParallel : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[normalize-parallel] loop already normalized\n");
      return failure();
    }
    if (op->getNumResults() != 0) {
      LLVM_DEBUG(
          DBGS() << "[normalize-parallel] not processing reduction loops\n");
      return failure();
    }
    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(op, args)) {
      LLVM_DEBUG(DBGS() << "[normalize-parallel] no nested barrier\n");
      return failure();
    }

    Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    Value one = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);
    SmallVector<Value> iterationCounts = emitIterationCounts(rewriter, op);
    auto newOp = rewriter.create<scf::ParallelOp>(
        op.getLoc(), SmallVector<Value>(iterationCounts.size(), zero),
        iterationCounts, SmallVector<Value>(iterationCounts.size(), one));

    SmallVector<Value> inductionVars;
    inductionVars.reserve(iterationCounts.size());
    rewriter.setInsertionPointToStart(newOp.getBody());
    for (unsigned i = 0, e = iterationCounts.size(); i < e; ++i) {
      Value scaled = rewriter.create<MulIOp>(
          op.getLoc(), newOp.getInductionVars()[i], op.getStep()[i]);
      Value shifted =
          rewriter.create<AddIOp>(op.getLoc(), op.getLowerBound()[i], scaled);
      inductionVars.push_back(shifted);
    }

    rewriter.mergeBlockBefore(op.getBody(), &newOp.getBody()->back(),
                              inductionVars);
    rewriter.eraseOp(&newOp.getBody()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Checks if `op` may need to be wrapped in a pair of barriers. This is a
/// necessary but insufficient condition.
static LogicalResult canWrapWithBarriers(Operation *op,
                                         SmallVector<BlockArgument> &vals) {
  if (!isa<scf::ParallelOp, AffineParallelOp>(op->getParentOp())) {
    LLVM_DEBUG(DBGS() << "[wrap] not nested in a pfor\n");
    return failure();
  }

  if (op->getNumResults() != 0) {
    LLVM_DEBUG(DBGS() << "[wrap] ignoring loop with reductions\n");
    return failure();
  }

  if (!hasNestedBarrier(op, vals)) {
    LLVM_DEBUG(DBGS() << "[wrap] no nested barrier\n");
    return failure();
  }

  return success();
}

bool isBarrierContainingAll(Operation *op, SmallVector<BlockArgument> &args) {
  auto bar = dyn_cast<polygeist::BarrierOp>(op);
  if (!bar)
    return false;
  SmallPtrSet<Value, 3> bargs(op->getOperands().begin(),
                              op->getOperands().end());
  for (auto a : args)
    if (!bargs.contains(a))
      return false;
  return true;
}

/// Puts a barrier before and/or after `op` if there isn't already one.
/// `extraPrevCheck` is called on the operation immediately preceding `op` and
/// can be used to look further upward if the immediately preceding operation is
/// not a barrier.
static LogicalResult wrapWithBarriers(
    Operation *op, PatternRewriter &rewriter, SmallVector<BlockArgument> &args,
    llvm::function_ref<bool(Operation *)> extraPrevCheck = nullptr) {
  Operation *prevOp = op->getPrevNode();
  Operation *nextOp = op->getNextNode();
  bool hasPrevBarrierLike =
      prevOp == nullptr || isBarrierContainingAll(prevOp, args);
  if (extraPrevCheck && !hasPrevBarrierLike)
    hasPrevBarrierLike = extraPrevCheck(prevOp);
  bool hasNextBarrierLike =
      nextOp == &op->getBlock()->back() || isBarrierContainingAll(nextOp, args);

  if (hasPrevBarrierLike && hasNextBarrierLike) {
    LLVM_DEBUG(DBGS() << "[wrap] already has sufficient barriers\n");
    return failure();
  }

  SmallVector<Value> vargs(args.begin(), args.end());

  if (!hasPrevBarrierLike)
    rewriter.create<polygeist::BarrierOp>(op->getLoc(), vargs);

  if (!hasNextBarrierLike) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    rewriter.create<polygeist::BarrierOp>(op->getLoc(), vargs);
  }

  // We don't actually change the op, but the pattern infra wants us to. Just
  // pretend we changed it in-place.
  rewriter.updateRootInPlace(op, [] {});
  LLVM_DEBUG(DBGS() << "[wrap] wrapped '" << op->getName().getStringRef()
                    << "' with barriers\n");
  return success();
}

/// Puts a barrier before and/or after an "if" operation if there isn't already
/// one, potentially with a single load that supplies the upper bound of a
/// (normalized) loop.
template <typename IfType>
struct WrapIfWithBarrier : public OpRewritePattern<IfType> {
  WrapIfWithBarrier(MLIRContext *ctx) : OpRewritePattern<IfType>(ctx) {}
  LogicalResult matchAndRewrite(IfType op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument> vals;
    if (failed(canWrapWithBarriers(op, vals)))
      return failure();

    if (op.getNumResults() != 0)
      return failure();

    SmallPtrSet<Value, 2> indVars;
    if (auto pop = dyn_cast<scf::ParallelOp>(op->getParentOp()))
      for (auto var : pop.getInductionVars())
        indVars.insert(var);
    else
      for (auto var :
           cast<AffineParallelOp>(op->getParentOp()).getBody()->getArguments())
        indVars.insert(var);

    return wrapWithBarriers(op, rewriter, vals, [&](Operation *prevOp) {
      if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(prevOp)) {
        if (inBound(op, loadOp.result()) &&
            llvm::all_of(loadOp.indices(),
                         [&](Value v) { return indVars.contains(v); })) {
          prevOp = prevOp->getPrevNode();
          return prevOp == nullptr || isBarrierContainingAll(prevOp, vals);
        }
      }
      return false;
    });
  }
};

/// Puts a barrier before and/or after a "for" operation if there isn't already
/// one, potentially with a single load that supplies the upper bound of a
/// (normalized) loop.

struct WrapForWithBarrier : public OpRewritePattern<scf::ForOp> {
  WrapForWithBarrier(MLIRContext *ctx) : OpRewritePattern<scf::ForOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument> vals;
    if (failed(canWrapWithBarriers(op, vals)))
      return failure();

    SmallPtrSet<Value, 2> indVars;
    if (auto pop = dyn_cast<scf::ParallelOp>(op->getParentOp()))
      for (auto var : pop.getInductionVars())
        indVars.insert(var);
    else
      for (auto var :
           cast<AffineParallelOp>(op->getParentOp()).getBody()->getArguments())
        indVars.insert(var);

    return wrapWithBarriers(op, rewriter, vals, [&](Operation *prevOp) {
      // If it is a load of the for upper bound
      if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(prevOp)) {
        if (loadOp.result() == op.getUpperBound() &&
            llvm::all_of(loadOp.indices(),
                         [&](Value v) { return indVars.contains(v); })) {
          prevOp = prevOp->getPrevNode();
          return prevOp == nullptr || isBarrierContainingAll(prevOp, vals);
        }
      }
      // If all preceeding ops are recomputable
      while (prevOp) {
        // TODO we could check if our cacheload ops have the parallel op indVars
        // as arguments, is that needed?
        if (!is_recomputable(*prevOp))
          return false;
        prevOp = prevOp->getPrevNode();
      }
      return true;
    });
  }
};

struct WrapAffineForWithBarrier : public OpRewritePattern<AffineForOp> {
  WrapAffineForWithBarrier(MLIRContext *ctx)
      : OpRewritePattern<AffineForOp>(ctx) {}

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument> vals;
    if (failed(canWrapWithBarriers(op, vals)))
      return failure();

    SmallPtrSet<Value, 2> indVars;
    if (auto pop = dyn_cast<scf::ParallelOp>(op->getParentOp()))
      for (auto var : pop.getInductionVars())
        indVars.insert(var);
    else
      for (auto var :
           cast<AffineParallelOp>(op->getParentOp()).getBody()->getArguments())
        indVars.insert(var);

    return wrapWithBarriers(op, rewriter, vals, [&](Operation *prevOp) {
      if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(prevOp)) {
        if (llvm::any_of(op.getUpperBoundOperands(),
                         [&](Value v) { return v == loadOp.result(); }) &&
            llvm::all_of(loadOp.indices(),
                         [&](Value v) { return indVars.contains(v); })) {
          prevOp = prevOp->getPrevNode();
          return prevOp == nullptr || isBarrierContainingAll(prevOp, vals);
        }
      }
      return false;
    });
  }
};

/// Puts a barrier before and/or after a "while" operation if there isn't
/// already one.
struct WrapWhileWithBarrier : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 0 || op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[wrap-while] ignoring non-mem2reg'd loop ops: "
                        << op.getNumOperands() << " res: " << op.getNumResults()
                        << "\n";);
      return failure();
    }

    SmallVector<BlockArgument> vals;
    if (failed(canWrapWithBarriers(op, vals)))
      return failure();

    return wrapWithBarriers(op, rewriter, vals);
  }
};

/// Moves the body from `ifOp` contained in `op` to a parallel op newly
/// created at the start of `newIf`.
template <typename T, typename IfType>
static void moveBodiesIf(PatternRewriter &rewriter, T op, IfType ifOp,
                         IfType newIf) {
  rewriter.startRootUpdate(op);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(getThenBlock(newIf));
    auto newParallel = rewriter.cloneWithoutRegions<T>(op);
    newParallel.getRegion().push_back(new Block());
    for (auto a : op.getBody()->getArguments())
      newParallel.getBody()->addArgument(a.getType(), op->getLoc());
    rewriter.setInsertionPointToEnd(newParallel.getBody());
    rewriter.clone(*op.getBody()->getTerminator());

    for (auto tup : llvm::zip(newParallel.getBody()->getArguments(),
                              op.getBody()->getArguments())) {
      std::get<1>(tup).replaceUsesWithIf(
          std::get<0>(tup), [&](OpOperand &op) -> bool {
            return getThenBlock(ifOp)->getParent()->isAncestor(
                op.getOwner()->getParentRegion());
          });
    }

    rewriter.eraseOp(&getThenBlock(ifOp)->back());
    rewriter.mergeBlockBefore(getThenBlock(ifOp),
                              &newParallel.getBody()->back());
  }

  if (hasElse(ifOp)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(getElseBlock(newIf));
    auto newParallel = rewriter.cloneWithoutRegions<T>(op);
    newParallel.getRegion().push_back(new Block());
    for (auto a : op.getBody()->getArguments())
      newParallel.getBody()->addArgument(a.getType(), op->getLoc());
    rewriter.setInsertionPointToEnd(newParallel.getBody());
    rewriter.clone(*op.getBody()->getTerminator());

    for (auto tup : llvm::zip(newParallel.getBody()->getArguments(),
                              op.getBody()->getArguments())) {
      std::get<1>(tup).replaceUsesWithIf(
          std::get<0>(tup), [&](OpOperand &op) -> bool {
            return getElseBlock(ifOp)->getParent()->isAncestor(
                op.getOwner()->getParentRegion());
          });
    }
    rewriter.eraseOp(&getElseBlock(ifOp)->back());
    rewriter.mergeBlockBefore(getElseBlock(ifOp),
                              &newParallel.getBody()->back());
  }

  rewriter.eraseOp(ifOp);
  rewriter.eraseOp(op);
  rewriter.finalizeRootUpdate(op);
}

/// Interchanges a parallel for loop with a for loop perfectly nested within it.
template <typename T, typename IfType>
struct InterchangeIfPFor : public OpRewritePattern<T> {
  InterchangeIfPFor(MLIRContext *ctx) : OpRewritePattern<T>(ctx) {}

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // A perfect nest must have two operations in the outermost body: an "if"
    // and a terminator.
    if (std::next(op.getBody()->begin(), 2) != op.getBody()->end() ||
        !isa<IfType>(op.getBody()->front())) {
      LLVM_DEBUG(DBGS() << "[interchange] not a perfect pfor(if) nest\n");
      return failure();
    }

    // We shouldn't have parallel reduction loops coming from GPU anyway, and
    // sequential reduction loops can be transformed by reg2mem.
    auto ifOp = cast<IfType>(op.getBody()->front());
    if (op.getNumResults() != 0 || ifOp.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[interchange] not matching reduction loops\n");
      return failure();
    }

    SmallVector<BlockArgument> blockArgs;
    if (!hasNestedBarrier(ifOp, blockArgs)) {
      LLVM_DEBUG(DBGS() << "[interchange] no nested barrier\n";);
      return failure();
    }

    auto newIf = cloneWithoutResults(ifOp, rewriter);
    moveBodiesIf(rewriter, op, ifOp, newIf);
    return success();
  }
};

mlir::OperandRange getLowerBounds(scf::ParallelOp op,
                                  PatternRewriter &rewriter) {
  return op.getLowerBound();
}
SmallVector<Value> getLowerBounds(AffineParallelOp op,
                                  PatternRewriter &rewriter) {
  SmallVector<Value> vals;
  for (AffineExpr expr : op.lowerBoundsMap().getResults()) {
    vals.push_back(rewriter
                       .create<AffineApplyOp>(op.getLoc(), expr,
                                              op.getLowerBoundsOperands())
                       .getResult());
  }
  return vals;
}

/// Interchanges a parallel for loop with a normalized (zero lower bound and
/// unit step) for loop nested within it. The for loop must have a barrier
/// inside and is preceeded by a load operation that supplies its upper bound.
/// The barrier semantics implies that all threads must executed the same number
/// of times, which means that the inner loop must have the same trip count in
/// all iterations of the outer loop. Therefore, the load of the upper bound can
/// be hoisted and read any value, because all values are identical in a
/// semantically valid program.
template <typename T, typename IfType>
struct InterchangeIfPForLoad : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (std::next(op.getBody()->begin(), 2) == op.getBody()->end() ||
        std::next(op.getBody()->begin(), 3) != op.getBody()->end()) {
      LLVM_DEBUG(DBGS() << "[interchange-load-if] expected two nested ops\n");
      return failure();
    }
    auto loadOp = dyn_cast<memref::LoadOp>(op.getBody()->front());
    auto ifOp = dyn_cast<IfType>(op.getBody()->front().getNextNode());
    if (!loadOp || !ifOp || !inBound(ifOp, loadOp.result()) ||
        loadOp.indices() != op.getBody()->getArguments()) {
      LLVM_DEBUG(DBGS() << "[interchange-load-if] expected pfor(load, for/if)");
      return failure();
    }

    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(ifOp, args)) {
      LLVM_DEBUG(DBGS() << "[interchange-load-if] no nested barrier\n");
      return failure();
    }

    // In the GPU model, the trip count of the inner sequential containing a
    // barrier must be the same for all threads. So read the value written by
    // the first thread outside of the loop to enable interchange.
    Value condition = rewriter.create<memref::LoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(), getLowerBounds(op, rewriter));

    BlockAndValueMapping mapping;
    mapping.map(loadOp.result(), condition);
    auto newIf = cloneWithoutResults(ifOp, rewriter, mapping);
    moveBodiesIf(rewriter, op, ifOp, newIf);
    return success();
  }
};

/// Moves the body from `forLoop` contained in `op` to a parallel op newly
/// created at the start of `newForLoop`.
template <typename T, typename ForType>
static void moveBodiesFor(PatternRewriter &rewriter, T op, ForType forLoop,
                          ForType newForLoop) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newForLoop.getBody());
  auto newParallel = rewriter.cloneWithoutRegions<T>(op);
  newParallel.getRegion().push_back(new Block());
  for (auto a : op.getBody()->getArguments())
    newParallel.getBody()->addArgument(a.getType(), op->getLoc());

  // Keep recomputable values in the parallel op
  BlockAndValueMapping mapping;
  mapping.map(op.getBody()->getArguments(),
              newParallel.getBody()->getArguments());
  rewriter.setInsertionPointToEnd(newParallel.getBody());
  for (auto it = op.getBody()->begin(); dyn_cast<ForType>(*it) != forLoop;
       ++it) {
    if (is_recomputable(*it))
      rewriter.clone(*it, mapping);
  }
  rewriter.setInsertionPointToEnd(newParallel.getBody());
  rewriter.clone(*op.getBody()->getTerminator());

  // Merge in two stages so we can properly replace uses of two induction
  // varibales defined in different blocks.
  rewriter.mergeBlockBefore(op.getBody(), &newParallel.getBody()->back(),
                            newParallel.getBody()->getArguments());
  rewriter.eraseOp(&newParallel.getBody()->back());
  rewriter.eraseOp(&forLoop.getBody()->back());
  rewriter.mergeBlockBefore(forLoop.getBody(), &newParallel.getBody()->back(),
                            newForLoop.getBody()->getArguments());
  rewriter.eraseOp(op);
  rewriter.eraseOp(forLoop);
}

/// Interchanges a parallel for loop with a for loop perfectly nested within it.
template <typename T, typename ForType>
struct InterchangeForPFor : public OpRewritePattern<T> {
  InterchangeForPFor(MLIRContext *ctx) : OpRewritePattern<T>(ctx) {}

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // A perfect nest must have two operations in the outermost body: a "for"
    // loop, and a terminator.
    if (std::next(op.getBody()->begin(), 2) != op.getBody()->end() ||
        !isa<ForType>(op.getBody()->front())) {
      LLVM_DEBUG(DBGS() << "[interchange] not a perfect pfor(for) nest\n");
      return failure();
    }

    // We shouldn't have parallel reduction loops coming from GPU anyway, and
    // sequential reduction loops can be transformed by reg2mem.
    auto forLoop = cast<ForType>(op.getBody()->front());
    if (op.getNumResults() != 0 || forLoop.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[interchange] not matching reduction loops\n");
      return failure();
    }

    // if (!isNormalized(op)) {
    //  LLVM_DEBUG(DBGS() << "[interchange] non-normalized loop\n");
    //}

    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(forLoop, args)) {
      LLVM_DEBUG(DBGS() << "[interchange] no nested barrier\n";);
      return failure();
    }

    auto newForLoop = cloneWithoutResults(forLoop, rewriter);
    moveBodiesFor(rewriter, op, forLoop, newForLoop);
    return success();
  }
};

/// Interchanges a parallel for loop with a normalized (zero lower bound and
/// unit step) for loop nested within it. The for loop must have a barrier
/// inside and is preceeded by a load operation that supplies its upper bound.
/// The barrier semantics implies that all threads must executed the same number
/// of times, which means that the inner loop must have the same trip count in
/// all iterations of the outer loop. Therefore, the load of the upper bound can
/// be hoisted and read any value, because all values are identical in a
/// semantically valid program.
template <typename T, typename ForType>
struct InterchangeForPForLoad : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (std::next(op.getBody()->begin(), 2) == op.getBody()->end() ||
        std::next(op.getBody()->begin(), 3) != op.getBody()->end()) {
      LLVM_DEBUG(DBGS() << "[interchange-load] expected two nested ops\n");
      return failure();
    }
    auto loadOp = dyn_cast<memref::LoadOp>(op.getBody()->front());
    auto forOp = dyn_cast<ForType>(op.getBody()->front().getNextNode());
    if (!loadOp || !forOp || !inBound(forOp, loadOp.result()) ||
        loadOp.indices() != op.getBody()->getArguments()) {
      LLVM_DEBUG(DBGS() << "[interchange-load] expected pfor(load, for/if)\n");
      return failure();
    }

    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(forOp, args)) {
      LLVM_DEBUG(DBGS() << "[interchange-load] no nested barrier\n");
      return failure();
    }

    // In the GPU model, the trip count of the inner sequential containing a
    // barrier must be the same for all threads. So read the value written by
    // the first thread outside of the loop to enable interchange.
    /*
    Value val = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Type mt = loadOp.getMemRef().getType();
    SmallVector<Value> zeros(mt.cast<MemRefType>().getShape().size(), val);
    Value tripCount = rewriter.create<memref::LoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(), zeros);
    */
    Value tripCount = rewriter.create<memref::LoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(), getLowerBounds(op, rewriter));
    BlockAndValueMapping mapping;
    mapping.map(loadOp.result(), tripCount);
    auto newForLoop = cloneWithoutResults(forOp, rewriter, mapping);

    moveBodiesFor(rewriter, op, forOp, newForLoop);
    return success();
  }
};

/// Checks if the block consists of recomputable operations (either ops with no
/// side effects or polygeist cache loads) and with the last operation of type
/// LastOpType which has a nested barrier
template <typename ParallelOpType, typename LastOpType>
bool canInterchangeParallelWithLastOp(StringRef Pattern, ParallelOpType op) {
  if (std::next(op.getBody()->begin(), 1) == op.getBody()->end()) {
    LLVM_DEBUG(DBGS() << "[" << Pattern
                      << "] expected one or more nested ops\n");
    return false;
  }

  // The actualy last op is a yield, get the one before that
  auto lastOpIt = std::prev(op.getBody()->end(), 2);
  auto lastOp = dyn_cast<LastOpType>(*lastOpIt);
  if (!lastOp) {
    LLVM_DEBUG(DBGS() << "[" << Pattern << "] unexpected last op type\n");
    return false;
  }

  SmallVector<BlockArgument> args;
  if (!hasNestedBarrier(lastOp, args)) {
    LLVM_DEBUG(DBGS() << "[" << Pattern << "] no nested barrier\n");
    return false;
  }

  for (auto it = op.getBody()->begin(); it != lastOpIt; ++it) {
    if (!is_recomputable(*it)) {
      LLVM_DEBUG(DBGS() << "[" << Pattern << "] found a nonrecomputable op\n");
      return false;
    }
  }

  return true;
}

/// Interchanges a parallel for loop with a for loop with preceeding value
/// recomputations
template <typename T, typename ForType>
struct InterchangeForPForRecomputable : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (!canInterchangeParallelWithLastOp<T, ForType>(
            "interchange-recomputable-for", op))
      return failure();

    auto lastOpIt = op.getBody()->back().getPrevNode();
    auto forLoop = dyn_cast<ForType>(lastOpIt);
    assert(forLoop);

    auto newForLoop = cloneWithoutResults(forLoop, rewriter);
    moveBodiesFor(rewriter, op, forLoop, newForLoop);

    return success();
  }
};

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
findNearestPostDominatingInsertionPoint(
    ArrayRef<Value> values, const PostDominanceInfo &postDominanceInfo) {
  assert(!values.empty());
  std::pair<Block *, Block::iterator> insertPoint =
      getInsertionPointAfterDef(values[0]);
  for (unsigned i = 1, e = values.size(); i < e; ++i)
    insertPoint = findNearestPostDominatingInsertionPoint(
        insertPoint, getInsertionPointAfterDef(values[i]), postDominanceInfo);
  return insertPoint;
}

/// Interchanges a parallel for loop with a while loop it contains. The while
/// loop is expected to have an empty "after" region.
template <typename T> struct InterchangeWhilePFor : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // A perfect nest must have two operations in the outermost body: a "while"
    // loop, and a terminator.
    if (std::next(op.getBody()->begin(), 2) != op.getBody()->end() ||
        !isa<scf::WhileOp>(op.getBody()->front())) {
      LLVM_DEBUG(
          DBGS() << "[interchange-while] not a perfect pfor(while) nest\n");
      return failure();
    }

    auto whileOp = cast<scf::WhileOp>(op.getBody()->front());
    if (whileOp.getNumOperands() != 0 || whileOp.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[interchange-while] loop-carried values\n");
      return failure();
    }
    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(whileOp, args)) {
      LLVM_DEBUG(DBGS() << "[interchange-while] no nested barrier\n");
      return failure();
    }

    auto conditionOp =
        cast<scf::ConditionOp>(whileOp.getBefore().front().back());

    auto beforeParallelOp = rewriter.cloneWithoutRegions<T>(op);
    beforeParallelOp.getRegion().push_back(new Block());
    for (auto a : op.getBody()->getArguments())
      beforeParallelOp.getBody()->addArgument(a.getType(), a.getLoc());
    auto afterParallelOp = rewriter.cloneWithoutRegions<T>(op);
    afterParallelOp.getRegion().push_back(new Block());
    for (auto a : op.getBody()->getArguments())
      afterParallelOp.getBody()->addArgument(a.getType(), a.getLoc());
    rewriter.setInsertionPointToEnd(beforeParallelOp.getBody());
    rewriter.clone(*op.getBody()->getTerminator());
    rewriter.setInsertionPointToEnd(afterParallelOp.getBody());
    rewriter.clone(*op.getBody()->getTerminator());

    rewriter.mergeBlockBefore(&whileOp.getBefore().front(),
                              beforeParallelOp.getBody()->getTerminator());
    whileOp.getBefore().push_back(new Block());
    conditionOp->moveBefore(&whileOp.getBefore().front(),
                            whileOp.getBefore().front().begin());
    beforeParallelOp->moveBefore(&whileOp.getBefore().front(),
                                 whileOp.getBefore().front().begin());

    auto yieldOp = cast<scf::YieldOp>(whileOp.getAfter().front().back());

    rewriter.mergeBlockBefore(&whileOp.getAfter().front(),
                              afterParallelOp.getBody()->getTerminator());
    whileOp.getAfter().push_back(new Block());
    yieldOp->moveBefore(&whileOp.getAfter().front(),
                        whileOp.getAfter().front().begin());
    afterParallelOp->moveBefore(&whileOp.getAfter().front(),
                                whileOp.getAfter().front().begin());

    for (auto tup : llvm::zip(op.getBody()->getArguments(),
                              beforeParallelOp.getBody()->getArguments(),
                              afterParallelOp.getBody()->getArguments())) {
      std::get<0>(tup).replaceUsesWithIf(std::get<1>(tup), [&](OpOperand &op) {
        return beforeParallelOp.getRegion().isAncestor(
            op.getOwner()->getParentRegion());
      });
      std::get<0>(tup).replaceUsesWithIf(std::get<2>(tup), [&](OpOperand &op) {
        return afterParallelOp.getRegion().isAncestor(
            op.getOwner()->getParentRegion());
      });
    }

    whileOp->moveBefore(op);
    rewriter.eraseOp(op);

    Operation *conditionDefiningOp = conditionOp.getCondition().getDefiningOp();
    if (conditionDefiningOp &&
        !conditionOp.getCondition().getParentRegion()->isAncestor(
            whileOp->getParentRegion())) {
      rewriter.setInsertionPoint(beforeParallelOp);
      Value allocated = rewriter.create<memref::AllocaOp>(
          conditionDefiningOp->getLoc(),
          MemRefType::get({}, rewriter.getI1Type()));
      rewriter.setInsertionPointAfter(conditionDefiningOp);
      Value cond = rewriter.create<ConstantIntOp>(conditionDefiningOp->getLoc(),
                                                  true, 1);
      for (auto tup : llvm::zip(getLowerBounds(beforeParallelOp, rewriter),
                                beforeParallelOp.getBody()->getArguments())) {
        cond = rewriter.create<AndIOp>(
            conditionDefiningOp->getLoc(),
            rewriter.create<CmpIOp>(conditionDefiningOp->getLoc(),
                                    CmpIPredicate::eq, std::get<0>(tup),
                                    std::get<1>(tup)),
            cond);
      }
      auto ifOp =
          rewriter.create<scf::IfOp>(conditionDefiningOp->getLoc(), cond);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      rewriter.create<memref::StoreOp>(conditionDefiningOp->getLoc(),
                                       conditionOp.getCondition(), allocated);

      rewriter.setInsertionPoint(conditionOp);

      Value reloaded = rewriter.create<memref::LoadOp>(
          conditionDefiningOp->getLoc(), allocated);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(conditionOp, reloaded,
                                                    ValueRange());
    }
    return success();
  }
};

/// Moves the "after" region of a while loop into its "before" region using a
/// conditional, that is
///
/// scf.while {
///   @before()
///   scf.conditional(%cond)
/// } do {
///   @after()
///   scf.yield
/// }
///
/// is transformed into
///
/// scf.while {
///   @before()
///   scf.if (%cond) {
///     @after()
///   }
///   scf.conditional(%cond)
/// } do {
///   scf.yield
/// }
struct RotateWhile : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    if (llvm::hasSingleElement(op.getAfter().front())) {
      LLVM_DEBUG(DBGS() << "[rotate-while] the after region is empty");
      return failure();
    }
    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(op, args)) {
      LLVM_DEBUG(DBGS() << "[rotate-while] no nested barrier\n");
      return failure();
    }
    if (op.getNumOperands() != 0 || op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[rotate-while] loop-carried values\n");
      return failure();
    }

    auto condition = cast<scf::ConditionOp>(op.getBefore().front().back());
    rewriter.setInsertionPoint(condition);
    auto conditional =
        rewriter.create<scf::IfOp>(op.getLoc(), condition.getCondition());
    rewriter.mergeBlockBefore(&op.getAfter().front(),
                              &conditional.getBody()->back());
    rewriter.eraseOp(&conditional.getBody()->back());

    rewriter.createBlock(&op.getAfter());
    rewriter.clone(conditional.getBody()->back());

    LLVM_DEBUG(DBGS() << "[rotate-while] done\n");
    return success();
  }
};

/// Splits a parallel loop around the first barrier it immediately contains.
/// Values defined before the barrier are stored in newly allocated buffers and
/// loaded back when needed.
template <typename T>
struct DistributeAroundBarrier : public OpRewritePattern<T> {
  DistributeAroundBarrier(MLIRContext *ctx) : OpRewritePattern<T>(ctx) {}

  LogicalResult splitSubLoop(T op, PatternRewriter &rewriter, BarrierOp barrier,
                             SmallVector<Value> &iterCounts, T &preLoop,
                             T &postLoop, Block *&outerBlock, T &outerLoop,
                             memref::AllocaScopeOp &outerEx) const;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[distribute] not matching reduction loops\n");
      return failure();
    }

    if (!isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[distribute] non-normalized loop\n");
      return failure();
    }

    BarrierOp barrier = nullptr;
    {
      auto it =
          llvm::find_if(op.getBody()->getOperations(), [](Operation &nested) {
            return isa<polygeist::BarrierOp>(nested);
          });
      if (it == op.getBody()->end()) {
        LLVM_DEBUG(DBGS() << "[distribute] no barrier in the loop\n");
        return failure();
      }
      barrier = cast<BarrierOp>(&*it);
    }

    llvm::SetVector<Value> usedBelow;
    llvm::SetVector<Operation *> preserveAllocas;
    findValuesUsedBelow(barrier, usedBelow, preserveAllocas);

    llvm::SetVector<Value> minCache;
    // TODO make it an option I guess
    bool MIN_CUT_OPTIMIZATION = true;
    if (MIN_CUT_OPTIMIZATION) {

      minCutCache(barrier, usedBelow, minCache);

      LLVM_DEBUG(DBGS() << "[distribute] min cut cache optimisation: "
                        << "preserveAllocas: " << preserveAllocas.size() << ", "
                        << "usedBelow: " << usedBelow.size() << ", "
                        << "minCache: " << minCache.size() << "\n");

      BlockAndValueMapping mapping;
      for (Value v : minCache)
        mapping.map(v, v);

      // Recalculate values used below the barrier up to available ones
      rewriter.setInsertionPointAfter(barrier);
      std::function<void(Value)> recalculateVal;
      recalculateVal = [&recalculateVal, &barrier, &mapping,
                        &rewriter](Value v) {
        auto op = v.getDefiningOp();
        if (mapping.contains(v)) {
          return;
        } else if (op && op->getBlock() == barrier->getBlock()) {
          for (Value operand : op->getOperands()) {
            recalculateVal(operand);
          }
          Operation *clonedOp = rewriter.clone(*op, mapping);
          for (auto pair : llvm::zip(op->getResults(), clonedOp->getResults()))
            mapping.map(std::get<0>(pair), std::get<1>(pair));
        } else {
          mapping.map(v, v);
        }
      };
      for (auto v : usedBelow) {
        recalculateVal(v);
        // Remap the uses of the recalculated val below the barrier
        for (auto &u : llvm::make_early_inc_range(v.getUses())) {
          auto user = u.getOwner();
          while (user->getBlock() != barrier->getBlock())
            user = user->getBlock()->getParentOp();
          if (barrier->isBeforeInBlock(user)) {
            rewriter.startRootUpdate(user);
            u.set(mapping.lookup(v));
            rewriter.finalizeRootUpdate(user);
          }
        }
      }
    } else {
      minCache = usedBelow;
    }

    // TODO should we integrate this in minCutCache?
    for (auto alloca : preserveAllocas) {
      minCache.remove(alloca->getResult(0));
    }

    SmallVector<Value> iterCounts;
    T preLoop;
    T postLoop;

    Block *outerBlock;
    T outerLoop = nullptr;
    memref::AllocaScopeOp outerEx = nullptr;

    rewriter.setInsertionPoint(op);
    if (splitSubLoop(op, rewriter, barrier, iterCounts, preLoop, postLoop,
                     outerBlock, outerLoop, outerEx)
            .failed())
      return failure();

    assert(iterCounts.size() == preLoop.getBody()->getArguments().size());

    size_t outIdx = 0;
    size_t inIdx = 0;
    for (auto en : op.getBody()->getArguments()) {
      bool found = false;
      for (auto v : barrier.getOperands())
        if (v == en)
          found = true;
      if (found) {
        en.replaceAllUsesWith(preLoop.getBody()->getArguments()[inIdx]);
        inIdx++;
      } else {
        en.replaceAllUsesWith(outerLoop.getBody()->getArguments()[outIdx]);
        outIdx++;
      }
    }
    op.getBody()->eraseArguments([](BlockArgument) { return true; });
    rewriter.mergeBlocks(op.getBody(), preLoop.getBody());

    rewriter.setInsertionPointToStart(outerBlock);
    // Allocate space for values crossing the barrier.
    SmallVector<Value> minCacheAllocations;
    SmallVector<Value> allocaAllocations;
    minCacheAllocations.reserve(minCache.size());
    allocaAllocations.reserve(preserveAllocas.size());
    auto mod = ((Operation *)op)->getParentOfType<ModuleOp>();
    assert(mod);
    DataLayout DLI(mod);
    auto addToAllocations = [&](Value v, SmallVector<Value> &allocations) {
      if (auto ao = v.getDefiningOp<LLVM::AllocaOp>()) {
        allocations.push_back(allocateTemporaryBuffer<LLVM::AllocaOp>(
            rewriter, v, iterCounts, true, &DLI));
      } else {
        allocations.push_back(
            allocateTemporaryBuffer<memref::AllocaOp>(rewriter, v, iterCounts));
      }
    };
    for (Value v : minCache)
      addToAllocations(v, minCacheAllocations);
    for (Operation *o : preserveAllocas)
      addToAllocations(o->getResult(0), allocaAllocations);

    // Allocate alloca's we need to preserve outside the loop
    for (auto pair : llvm::zip(preserveAllocas, allocaAllocations)) {
      Operation *o = std::get<0>(pair);
      Value alloc = std::get<1>(pair);
      if (auto ao = dyn_cast<memref::AllocaOp>(o)) {
        for (auto &u : llvm::make_early_inc_range(ao.getResult().getUses())) {
          rewriter.setInsertionPoint(u.getOwner());
          auto buf = alloc;
          for (auto idx : preLoop.getBody()->getArguments()) {
            auto mt0 = buf.getType().cast<MemRefType>();
            std::vector<int64_t> shape(mt0.getShape());
            assert(shape.size() > 0);
            shape.erase(shape.begin());
            auto mt = MemRefType::get(shape, mt0.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      // mt0.getLayout(),
                                      mt0.getMemorySpace());
            auto subidx = rewriter.create<polygeist::SubIndexOp>(alloc.getLoc(),
                                                                 mt, buf, idx);
            buf = subidx;
          }
          u.set(buf);
        }
        rewriter.eraseOp(ao);
      } else if (auto ao = dyn_cast<LLVM::AllocaOp>(o)) {
        Value sz = ao.getArraySize();
        rewriter.setInsertionPointAfter(alloc.getDefiningOp());
        alloc =
            rewriter.create<LLVM::BitcastOp>(ao.getLoc(), ao.getType(), alloc);
        for (auto &u : llvm::make_early_inc_range(ao.getResult().getUses())) {
          rewriter.setInsertionPoint(u.getOwner());
          Value idx = nullptr;
          // i0
          // i0 * s1 + i1
          // ( i0 * s1 + i1 ) * s2 + i2
          for (auto pair :
               llvm::zip(iterCounts, preLoop.getBody()->getArguments())) {
            if (idx) {
              idx = rewriter.create<arith::MulIOp>(ao.getLoc(), idx,
                                                   std::get<0>(pair));
              idx = rewriter.create<arith::AddIOp>(ao.getLoc(), idx,
                                                   std::get<1>(pair));
            } else
              idx = std::get<1>(pair);
          }
          idx = rewriter.create<MulIOp>(ao.getLoc(), sz,
                                        rewriter.create<arith::IndexCastOp>(
                                            ao.getLoc(), sz.getType(), idx));
          SmallVector<Value> vec = {idx};
          u.set(rewriter.create<LLVM::GEPOp>(ao.getLoc(), ao.getType(), alloc,
                                             idx));
        }
      } else {
        assert(false && "Wrong operation type in preserveAllocas");
      }
    }

    // Store values in the min cache immediately when ready and reload them
    // after the barrier
    for (auto pair : llvm::zip(minCache, minCacheAllocations)) {
      Value v = std::get<0>(pair);
      Value alloc = std::get<1>(pair);
      // Store
      rewriter.setInsertionPointAfter(v.getDefiningOp());
      rewriter.create<memref::StoreOp>(v.getLoc(), v, alloc,
                                       preLoop.getBody()->getArguments());
      // Reload
      rewriter.setInsertionPointAfter(barrier);
      Value reloaded = rewriter.create<polygeist::CacheLoad>(
          v.getLoc(), alloc, preLoop.getBody()->getArguments());
      for (auto &u : llvm::make_early_inc_range(v.getUses())) {
        auto user = u.getOwner();
        while (user->getBlock() != barrier->getBlock())
          user = user->getBlock()->getParentOp();

        if (barrier->isBeforeInBlock(user)) {
          rewriter.startRootUpdate(user);
          u.set(reloaded);
          rewriter.finalizeRootUpdate(user);
        }
      }
    }

    // Insert the terminator for the new loop immediately before the barrier.
    rewriter.setInsertionPoint(barrier);
    rewriter.clone(preLoop.getBody()->back());
    Operation *postBarrier = barrier->getNextNode();
    rewriter.eraseOp(barrier);

    // Create the second loop.
    rewriter.setInsertionPointToEnd(outerBlock);
    auto freefn = GetOrCreateFreeFunction(mod);
    // TODO do this more efficiently
    SmallVector<Value> allocations;
    allocations.append(minCacheAllocations.begin(), minCacheAllocations.end());
    allocations.append(allocaAllocations.begin(), allocaAllocations.end());
    for (auto alloc : allocations) {
      if (alloc.getType().isa<LLVM::LLVMPointerType>()) {
        Value args[1] = {alloc};
        rewriter.create<LLVM::CallOp>(alloc.getLoc(), freefn, args);
      } else
        rewriter.create<memref::DeallocOp>(alloc.getLoc(), alloc);
    }
    if (outerLoop) {
      if (isa<scf::ParallelOp>(outerLoop))
        rewriter.create<scf::YieldOp>(op.getLoc());
      else {
        assert(isa<AffineParallelOp>(outerLoop));
        rewriter.create<AffineYieldOp>(op.getLoc());
      }
    } else {
      rewriter.create<memref::AllocaScopeReturnOp>(op.getLoc());
    }

    // Recreate the operations in the new loop with new values.
    rewriter.setInsertionPointToStart(postLoop.getBody());
    BlockAndValueMapping mapping;
    mapping.map(preLoop.getBody()->getArguments(),
                postLoop.getBody()->getArguments());
    SmallVector<Operation *> toDelete;
    for (Operation *o = postBarrier; o != nullptr; o = o->getNextNode()) {
      rewriter.clone(*o, mapping);
      toDelete.push_back(o);
    }

    // Erase original operations and the barrier.
    for (Operation *o : llvm::reverse(toDelete))
      rewriter.eraseOp(o);

    for (auto ao : allocations) {
      if (ao.getDefiningOp<LLVM::AllocaOp>() ||
          ao.getDefiningOp<memref::AllocaOp>()) {
        assert(false && "TODO I feel like this never happens, no?");
        rewriter.eraseOp(ao.getDefiningOp());
      }
    }

    if (!outerLoop) {
      rewriter.mergeBlockBefore(outerBlock, op);
      rewriter.eraseOp(outerEx);
    }
    rewriter.eraseOp(op);

    LLVM_DEBUG(DBGS() << "[distribute] distributed around a barrier\n");
    return success();
  }
};
template <>
LogicalResult DistributeAroundBarrier<scf::ParallelOp>::splitSubLoop(
    scf::ParallelOp op, PatternRewriter &rewriter, BarrierOp barrier,
    SmallVector<Value> &iterCounts, scf::ParallelOp &preLoop,
    scf::ParallelOp &postLoop, Block *&outerBlock, scf::ParallelOp &outerLoop,
    memref::AllocaScopeOp &outerEx) const {

  SmallVector<Value> outerLower;
  SmallVector<Value> outerUpper;
  SmallVector<Value> outerStep;
  SmallVector<Value> innerLower;
  SmallVector<Value> innerUpper;
  SmallVector<Value> innerStep;
  for (auto en : llvm::zip(op.getBody()->getArguments(), op.getLowerBound(),
                           op.getUpperBound(), op.getStep())) {
    bool found = false;
    for (auto v : barrier.getOperands())
      if (v == std::get<0>(en))
        found = true;
    if (found) {
      innerLower.push_back(std::get<1>(en));
      innerUpper.push_back(std::get<2>(en));
      innerStep.push_back(std::get<3>(en));
    } else {
      outerLower.push_back(std::get<1>(en));
      outerUpper.push_back(std::get<2>(en));
      outerStep.push_back(std::get<3>(en));
    }
  }
  if (!innerLower.size())
    return failure();
  if (outerLower.size()) {
    outerLoop = rewriter.create<scf::ParallelOp>(op.getLoc(), outerLower,
                                                 outerUpper, outerStep);
    rewriter.eraseOp(&outerLoop.getBody()->back());
    outerBlock = outerLoop.getBody();
  } else {
    outerEx = rewriter.create<memref::AllocaScopeOp>(op.getLoc(), TypeRange());
    outerBlock = new Block();
    outerEx.getRegion().push_back(outerBlock);
  }

  rewriter.setInsertionPointToEnd(outerBlock);
  for (auto tup : llvm::zip(innerLower, innerUpper, innerStep)) {
    iterCounts.push_back(rewriter.create<DivUIOp>(
        op.getLoc(),
        rewriter.create<SubIOp>(op.getLoc(), std::get<1>(tup),
                                std::get<0>(tup)),
        std::get<2>(tup)));
  }
  preLoop = rewriter.create<scf::ParallelOp>(op.getLoc(), innerLower,
                                             innerUpper, innerStep);
  rewriter.eraseOp(&preLoop.getBody()->back());
  postLoop = rewriter.create<scf::ParallelOp>(op.getLoc(), innerLower,
                                              innerUpper, innerStep);
  rewriter.eraseOp(&postLoop.getBody()->back());
  return success();
}

template <>
LogicalResult DistributeAroundBarrier<AffineParallelOp>::splitSubLoop(
    AffineParallelOp op, PatternRewriter &rewriter, BarrierOp barrier,
    SmallVector<Value> &iterCounts, AffineParallelOp &preLoop,
    AffineParallelOp &postLoop, Block *&outerBlock, AffineParallelOp &outerLoop,
    memref::AllocaScopeOp &outerEx) const {

  SmallVector<AffineMap> outerLower;
  SmallVector<AffineMap> outerUpper;
  SmallVector<int64_t> outerStep;
  SmallVector<AffineMap> innerLower;
  SmallVector<AffineMap> innerUpper;
  SmallVector<int64_t> innerStep;
  unsigned idx = 0;
  for (auto en : llvm::enumerate(
           llvm::zip(op.getBody()->getArguments(), op.getSteps()))) {
    bool found = false;
    for (auto v : barrier.getOperands())
      if (v == std::get<0>(en.value()))
        found = true;
    if (found) {
      innerLower.push_back(op.lowerBoundsMap().getSliceMap(en.index(), 1));
      innerUpper.push_back(op.upperBoundsMap().getSliceMap(en.index(), 1));
      innerStep.push_back(std::get<1>(en.value()));
    } else {
      outerLower.push_back(op.lowerBoundsMap().getSliceMap(en.index(), 1));
      outerUpper.push_back(op.upperBoundsMap().getSliceMap(en.index(), 1));
      outerStep.push_back(std::get<1>(en.value()));
    }
    idx++;
  }
  if (!innerLower.size())
    return failure();
  if (outerLower.size()) {
    outerLoop = rewriter.create<AffineParallelOp>(
        op.getLoc(), TypeRange(), ArrayRef<AtomicRMWKind>(), outerLower,
        op.getLowerBoundsOperands(), outerUpper, op.getUpperBoundsOperands(),
        outerStep);
    rewriter.eraseOp(&outerLoop.getBody()->back());
    outerBlock = outerLoop.getBody();
  } else {
    outerEx = rewriter.create<memref::AllocaScopeOp>(op.getLoc(), TypeRange());
    outerBlock = new Block();
    outerEx.getRegion().push_back(outerBlock);
  }

  rewriter.setInsertionPointToEnd(outerBlock);
  for (auto tup : llvm::zip(innerLower, innerUpper, innerStep)) {
    auto expr = (std::get<1>(tup).getResult(0) -
                 std::get<0>(tup)
                     .getResult(0)
                     .shiftDims(op.lowerBoundsMap().getNumDims(),
                                op.upperBoundsMap().getNumDims())
                     .shiftSymbols(op.lowerBoundsMap().getNumSymbols(),
                                   op.upperBoundsMap().getNumSymbols()))
                    .floorDiv(std::get<2>(tup));
    SmallVector<Value> symbols;
    SmallVector<Value> dims;
    size_t idx = 0;
    for (auto v : op.getUpperBoundsOperands()) {
      if (idx < op.upperBoundsMap().getNumDims())
        dims.push_back(v);
      else
        symbols.push_back(v);
      idx++;
    }
    idx = 0;
    for (auto v : op.getLowerBoundsOperands()) {
      if (idx < op.lowerBoundsMap().getNumDims())
        dims.push_back(v);
      else
        symbols.push_back(v);
      idx++;
    }
    SmallVector<Value> ops = dims;
    ops.append(symbols);
    iterCounts.push_back(rewriter.create<AffineApplyOp>(
        op.getLoc(), AffineMap::get(dims.size(), symbols.size(), expr), ops));
  }
  preLoop = rewriter.create<AffineParallelOp>(
      op.getLoc(), TypeRange(), ArrayRef<AtomicRMWKind>(), innerLower,
      op.getLowerBoundsOperands(), innerUpper, op.getUpperBoundsOperands(),
      innerStep);
  rewriter.eraseOp(&preLoop.getBody()->back());
  postLoop = rewriter.create<AffineParallelOp>(
      op.getLoc(), TypeRange(), ArrayRef<AtomicRMWKind>(), innerLower,
      op.getLowerBoundsOperands(), innerUpper, op.getUpperBoundsOperands(),
      innerStep);
  rewriter.eraseOp(&postLoop.getBody()->back());
  return success();
}

static void loadValues(Location loc, ArrayRef<Value> pointers,
                       PatternRewriter &rewriter,
                       SmallVectorImpl<Value> &loaded) {
  loaded.reserve(loaded.size() + pointers.size());
  for (Value alloc : pointers)
    loaded.push_back(rewriter.create<memref::LoadOp>(loc, alloc, ValueRange()));
}

template <typename T> struct Reg2MemFor : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument> args;
    if (op.getNumResults() == 0 || !hasNestedBarrier(op, args))
      return failure();

    SmallVector<Value> allocated;
    allocated.reserve(op.getNumIterOperands());
    for (Value operand : op.getIterOperands()) {
      Value alloc = rewriter.create<memref::AllocaOp>(
          op.getLoc(), MemRefType::get(ArrayRef<int64_t>(), operand.getType()),
          ValueRange());
      allocated.push_back(alloc);
      rewriter.create<memref::StoreOp>(op.getLoc(), operand, alloc,
                                       ValueRange());
    }

    auto newOp = cloneWithoutResults(op, rewriter);
    rewriter.setInsertionPointToStart(newOp.getBody());
    SmallVector<Value> newRegionArguments;
    newRegionArguments.push_back(newOp.getInductionVar());
    loadValues(op.getLoc(), allocated, rewriter, newRegionArguments);

    auto oldTerminator = op.getBody()->getTerminator();
    rewriter.mergeBlockBefore(op.getBody(), newOp.getBody()->getTerminator(),
                              newRegionArguments);
    SmallVector<Value> oldOps;
    llvm::append_range(oldOps, oldTerminator->getOperands());
    rewriter.eraseOp(oldTerminator);

    Operation *IP = newOp.getBody()->getTerminator();
    while (IP != &IP->getBlock()->front()) {
      if (isa<BarrierOp>(IP->getPrevNode())) {
        IP = IP->getPrevNode();
      }
      break;
    }
    rewriter.setInsertionPoint(IP);
    for (auto en : llvm::enumerate(oldOps)) {
      rewriter.create<memref::StoreOp>(op.getLoc(), en.value(),
                                       allocated[en.index()], ValueRange());
    }

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> loaded;
    for (Value alloc : allocated) {
      loaded.push_back(
          rewriter.create<memref::LoadOp>(op.getLoc(), alloc, ValueRange()));
    }
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

template <typename T> struct Reg2MemIf : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument> args;
    if (!op.getResults().size() || !hasNestedBarrier(op, args))
      return failure();

    SmallVector<Value> allocated;
    allocated.reserve(op.getNumResults());
    for (Type opType : op.getResultTypes()) {
      Value alloc = rewriter.create<memref::AllocaOp>(
          op.getLoc(), MemRefType::get(ArrayRef<int64_t>(), opType),
          ValueRange());
      allocated.push_back(alloc);
    }

    auto newOp = cloneWithoutResults(op, rewriter);

    Operation *thenYield = &getThenBlock(op)->back();
    rewriter.setInsertionPoint(thenYield);
    for (auto en : llvm::enumerate(thenYield->getOperands())) {
      rewriter.create<memref::StoreOp>(op.getLoc(), en.value(),
                                       allocated[en.index()], ValueRange());
    }
    thenYield->setOperands(ValueRange());

    Operation *elseYield = &getElseBlock(op)->back();
    rewriter.setInsertionPoint(elseYield);
    for (auto en : llvm::enumerate(elseYield->getOperands())) {
      rewriter.create<memref::StoreOp>(op.getLoc(), en.value(),
                                       allocated[en.index()], ValueRange());
    }
    elseYield->setOperands(ValueRange());

    rewriter.eraseOp(&getThenBlock(newOp)->back());
    rewriter.mergeBlocks(getThenBlock(op), getThenBlock(newOp));

    rewriter.eraseOp(&getElseBlock(newOp)->back());
    rewriter.mergeBlocks(getElseBlock(op), getElseBlock(newOp));

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> loaded;
    for (Value alloc : allocated) {
      loaded.push_back(
          rewriter.create<memref::LoadOp>(op.getLoc(), alloc, ValueRange()));
    }
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

static void storeValues(Location loc, ValueRange values, ValueRange pointers,
                        PatternRewriter &rewriter) {
  for (auto pair : llvm::zip(values, pointers)) {
    rewriter.create<memref::StoreOp>(loc, std::get<0>(pair), std::get<1>(pair),
                                     ValueRange());
  }
}

static void allocaValues(Location loc, ValueRange values,
                         PatternRewriter &rewriter,
                         SmallVector<Value> &allocated) {
  allocated.reserve(values.size());
  for (Value value : values) {
    Value alloc = rewriter.create<memref::AllocaOp>(
        loc, MemRefType::get(ArrayRef<int64_t>(), value.getType()),
        ValueRange());
    allocated.push_back(alloc);
  }
}

struct Reg2MemWhile : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() == 0 && op.getNumResults() == 0)
      return failure();
    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(op, args)) {
      return failure();
    }

    // Value stackPtr = rewriter.create<LLVM::StackSaveOp>(
    //     op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
    SmallVector<Value> beforeAllocated, afterAllocated;
    allocaValues(op.getLoc(), op.getOperands(), rewriter, beforeAllocated);
    storeValues(op.getLoc(), op.getOperands(), beforeAllocated, rewriter);
    allocaValues(op.getLoc(), op.getResults(), rewriter, afterAllocated);

    auto newOp =
        rewriter.create<scf::WhileOp>(op.getLoc(), TypeRange(), ValueRange());
    Block *newBefore =
        rewriter.createBlock(&newOp.getBefore(), newOp.getBefore().begin());
    SmallVector<Value> newBeforeArguments;
    loadValues(op.getLoc(), beforeAllocated, rewriter, newBeforeArguments);
    rewriter.mergeBlocks(&op.getBefore().front(), newBefore,
                         newBeforeArguments);

    auto beforeTerminator =
        cast<scf::ConditionOp>(newOp.getBefore().front().getTerminator());
    rewriter.setInsertionPoint(beforeTerminator);
    storeValues(op.getLoc(), beforeTerminator.getArgs(), afterAllocated,
                rewriter);

    rewriter.updateRootInPlace(
        beforeTerminator, [&] { beforeTerminator.getArgsMutable().clear(); });

    Block *newAfter =
        rewriter.createBlock(&newOp.getAfter(), newOp.getAfter().begin());
    SmallVector<Value> newAfterArguments;
    loadValues(op.getLoc(), afterAllocated, rewriter, newAfterArguments);
    rewriter.mergeBlocks(&op.getAfter().front(), newAfter, newAfterArguments);

    auto afterTerminator =
        cast<scf::YieldOp>(newOp.getAfter().front().getTerminator());
    rewriter.setInsertionPoint(afterTerminator);
    storeValues(op.getLoc(), afterTerminator.getResults(), beforeAllocated,
                rewriter);

    rewriter.updateRootInPlace(
        afterTerminator, [&] { afterTerminator.getResultsMutable().clear(); });

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> results;
    loadValues(op.getLoc(), afterAllocated, rewriter, results);
    // rewriter.create<LLVM::StackRestoreOp>(op.getLoc(), stackPtr);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct CPUifyPass : public SCFCPUifyBase<CPUifyPass> {
  CPUifyPass() = default;
  CPUifyPass(StringRef method) { this->method.setValue(method.str()); }
  void runOnOperation() override {
    if (method == "distribute") {
      RewritePatternSet patterns(&getContext());
      patterns.insert<Reg2MemFor<scf::ForOp>, Reg2MemFor<AffineForOp>,
                      Reg2MemWhile, Reg2MemIf<scf::IfOp>, Reg2MemIf<AffineIfOp>,
                      WrapForWithBarrier, WrapAffineForWithBarrier,
                      WrapIfWithBarrier<scf::IfOp>,
                      WrapIfWithBarrier<AffineIfOp>, WrapWhileWithBarrier,
                      InterchangeForPFor<scf::ParallelOp, scf::ForOp>,
                      InterchangeForPFor<AffineParallelOp, scf::ForOp>,
                      InterchangeForPForLoad<scf::ParallelOp, scf::ForOp>,
                      InterchangeForPForLoad<AffineParallelOp, scf::ForOp>,
                      InterchangeForPForRecomputable<scf::ParallelOp, scf::ForOp>,
                      InterchangeForPForRecomputable<AffineParallelOp, scf::ForOp>,
                      InterchangeIfPFor<scf::ParallelOp, scf::IfOp>,
                      InterchangeIfPFor<AffineParallelOp, scf::IfOp>,
                      InterchangeIfPForLoad<scf::ParallelOp, scf::IfOp>,
                      InterchangeIfPForLoad<AffineParallelOp, scf::IfOp>,

                      InterchangeForPFor<scf::ParallelOp, AffineForOp>,
                      InterchangeForPFor<AffineParallelOp, AffineForOp>,
                      InterchangeForPForLoad<scf::ParallelOp, AffineForOp>,
                      InterchangeForPForLoad<AffineParallelOp, AffineForOp>,
                      InterchangeIfPFor<scf::ParallelOp, AffineIfOp>,
                      InterchangeIfPFor<AffineParallelOp, AffineIfOp>,
                      InterchangeIfPForLoad<scf::ParallelOp, AffineIfOp>,
                      InterchangeIfPForLoad<AffineParallelOp, AffineIfOp>,

                      InterchangeWhilePFor<scf::ParallelOp>,
                      InterchangeWhilePFor<AffineParallelOp>,
                      // NormalizeLoop,
                      NormalizeParallel,
                      // RotateWhile,
                      DistributeAroundBarrier<scf::ParallelOp>,
                      DistributeAroundBarrier<AffineParallelOp>>(&getContext());
      GreedyRewriteConfig config;
      config.maxIterations = 142;
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns), config)))
        signalPassFailure();
    } else if (method == "omp") {
      SmallVector<polygeist::BarrierOp> toReplace;
      getOperation()->walk(
          [&](polygeist::BarrierOp b) { toReplace.push_back(b); });
      for (auto b : toReplace) {
        OpBuilder Builder(b);
        Builder.create<omp::BarrierOp>(b.getLoc());
        b->erase();
      }
    } else {
      llvm::errs() << "unknown cpuify type: " << method << "\n";
      llvm_unreachable("unknown cpuify type");
    }
  }
};

} // end namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createCPUifyPass(StringRef str) {
  return std::make_unique<CPUifyPass>(str);
}
} // namespace polygeist
} // namespace mlir

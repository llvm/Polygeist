#include "PassDetails.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::scf;
using namespace mlir::func;
using namespace mlir::arith;
using namespace polygeist;

namespace {
struct CanonicalizeFor : public SCFCanonicalizeForBase<CanonicalizeFor> {
  void runOnOperation() override;
};
} // namespace

struct PropagateInLoopBody : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    if (!forOp.getInits().size())
      return failure();

    Block &block = forOp.getRegion().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());
    bool matched = false;
    for (auto it : llvm::zip(forOp.getInits(), forOp.getRegionIterArgs(),
                             yieldOp.getOperands())) {
      Value iterOperand = std::get<0>(it);
      Value regionArg = std::get<1>(it);
      Value yieldOperand = std::get<2>(it);

      Operation *op = iterOperand.getDefiningOp();
      if (op && (op->getNumResults() == 1) && (iterOperand == yieldOperand)) {
        regionArg.replaceAllUsesWith(op->getResult(0));
        matched = true;
      }
    }
    return success(matched);
  }
};

// %f = scf.for %c = true, %a = ...
//    %r = if %c {
//      %d = cond()
//      %q = scf.if not %d {
//        yield %a
//      } else {
//        %a2 = addi %a, ...
//        yield %a2
//      }
//      yield %d, %q
//    } else {
//      yield %false, %a
//    }
//    yield %r#0, %r#1
// }
// no use of %f#1
//
// becomes
//
// %f = scf.for %c = true, %a = ...
//    %r = if %c {
//      %d = cond()
//      %a2 = addi %a, ...
//      yield %d, %a2
//    } else {
//      yield %false, %a
//    }
//    yield %r#0, %r#1
// }
// no use of %f#1
//
// and finally
//
// %f = scf.for %c = true, %a = ...
//    %a2 = addi %a, ...
//    %r = if %c {
//      %d = cond()
//      yield %d, %a2
//    } else {
//      yield %false, %a
//    }
//    yield %r#0, %a2
// }
// no use of %f#1
//
// Picking the "continue" value changes what the for carries in %a after the
// iteration that clears %c. That is unobservable only when %f#1 is unused
// and %a itself is read nowhere but inside the `then` region of the guard
// (which never runs again once %c is false) and in its own slot of the
// `else` yield. Only that one yield slot is rewritten: the same inner value
// may also feed another slot, e.g. the copy of the counter whose result is
// used, that MoveWhileToFor creates next to the unused one. Rewriting the
// inner if's result for every user made `for (i = 0; i < n; i++) if (..)
// { ..; break; }` leave i one too high after the break (W8b review).
static bool carriedOnlyUnderGuard(BlockArgument regionArg, scf::IfOp guard,
                                  unsigned slot) {
  for (OpOperand &use : regionArg.getUses()) {
    Operation *user = use.getOwner();
    if (guard.getThenRegion().isAncestor(user->getParentRegion()))
      continue;
    if (user == guard.elseYield().getOperation() &&
        use.getOperandNumber() == slot)
      continue;
    return false;
  }
  return true;
}

struct ForBreakAddUpgrade : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    if (!forOp.getInits().size())
      return failure();

    Block &block = forOp.getRegion().front();
    // First check there is an outermost if
    auto outerIfOp = dyn_cast<scf::IfOp>(*block.begin());
    if (!outerIfOp)
      return failure();

    auto condition = outerIfOp.getCondition();
    //  and that the outermost if's condition is an iter arg of the for
    auto condArg = dyn_cast<BlockArgument>(condition);
    if (!condArg)
      return failure();
    if (condArg.getOwner()->getParentOp() != forOp)
      return failure();
    // which starts as true
    if (!matchPattern(forOp.getInits()[condArg.getArgNumber() - 1], m_One()))
      return failure();
    // and is false unless coming from inside the if
    auto forYieldOp = cast<scf::YieldOp>(block.getTerminator());
    auto opres =
        dyn_cast<OpResult>(forYieldOp.getOperand(condArg.getArgNumber() - 1));
    if (!opres)
      return failure();
    if (opres.getOwner() != outerIfOp)
      return failure();

    if (!matchPattern(outerIfOp.elseYield().getOperand(opres.getResultNumber()),
                      m_Zero()))
      return failure();

    bool changed = false;
    for (auto it :
         llvm::zip(forOp.getRegionIterArgs(), forYieldOp.getOperands(),
                   forOp.getResults(), forOp.getInits())) {
      auto regionArg = std::get<0>(it);
      Value forYieldOperand = std::get<1>(it);
      Value res = std::get<2>(it);
      Value iterOp = std::get<3>(it);

      if (opres.getResultNumber() == regionArg.getArgNumber() - 1)
        continue;

      auto opres2 = dyn_cast<OpResult>(forYieldOperand);
      if (!opres2)
        continue;
      if (opres2.getOwner() != outerIfOp)
        continue;
      auto topOp = outerIfOp.thenYield().getOperand(opres2.getResultNumber());

      auto trueYield =
          outerIfOp.thenYield().getOperand(opres.getResultNumber());
      bool negated = false;
      while (auto neg = trueYield.getDefiningOp<XOrIOp>())
        if (matchPattern(neg.getOperand(1), m_One())) {
          trueYield = neg.getOperand(0);
          negated = !negated;
        }

      if (auto innerIfOp = topOp.getDefiningOp<scf::IfOp>()) {
        Value ifcond = innerIfOp.getCondition();
        while (auto neg = ifcond.getDefiningOp<XOrIOp>())
          if (matchPattern(neg.getOperand(1), m_One())) {
            ifcond = neg.getOperand(0);
            negated = !negated;
          }

        if (ifcond == trueYield) {
          // If never used, can always pick the "continue" value
          if (res.use_empty() &&
              carriedOnlyUnderGuard(regionArg, outerIfOp,
                                    opres2.getResultNumber())) {
            auto idx = topOp.cast<OpResult>().getResultNumber();
            Value val =
                (negated ? innerIfOp.elseYield() : innerIfOp.thenYield())
                    .getOperand(idx);
            Region *reg = (negated ? &innerIfOp.getElseRegion()
                                   : &innerIfOp.getThenRegion());
            if (reg->isAncestor(val.getParentRegion())) {
              if (auto addi = val.getDefiningOp<arith::AddIOp>()) {
                if (reg->isAncestor(addi.getOperand(0).getParentRegion()) ||
                    reg->isAncestor(addi.getOperand(1).getParentRegion()))
                  continue;
                rewriter.setInsertionPoint(innerIfOp);
                val = rewriter.replaceOpWithNewOp<arith::AddIOp>(
                    addi, addi.getOperand(0), addi.getOperand(1));
              } else
                continue;
            }

            scf::YieldOp thenYield = outerIfOp.thenYield();
            rewriter.updateRootInPlace(thenYield, [&] {
              thenYield->setOperand(opres2.getResultNumber(), val);
            });
            changed = true;
            continue;
          }
        }
      }

      if (auto innerSelOp = topOp.getDefiningOp<arith::SelectOp>()) {
        bool negated = false;
        Value ifcond = innerSelOp.getCondition();
        while (auto neg = ifcond.getDefiningOp<XOrIOp>())
          if (matchPattern(neg.getOperand(1), m_One())) {
            ifcond = neg.getOperand(0);
            negated = !negated;
          }
        if (ifcond == trueYield) {
          // If never used, can always pick the "continue" value
          if (res.use_empty() &&
              carriedOnlyUnderGuard(regionArg, outerIfOp,
                                    opres2.getResultNumber())) {
            Value val = (negated ? innerSelOp.getFalseValue()
                                 : innerSelOp.getTrueValue());
            scf::YieldOp thenYield = outerIfOp.thenYield();
            rewriter.updateRootInPlace(thenYield, [&] {
              thenYield->setOperand(opres2.getResultNumber(), val);
            });
            changed = true;
            continue;
          }
          bool seenSelf = false;
          bool seenIllegal = false;
          for (auto &u : regionArg.getUses()) {
            if (u.getOwner() == innerSelOp &&
                regionArg != innerSelOp.getCondition()) {
              seenSelf = true;
              continue;
            }
            if (u.getOwner() != outerIfOp.elseYield()) {
              seenIllegal = true;
              break;
            }
            if (u.getOperandNumber() != opres2.getResultNumber()) {
              seenIllegal = true;
              break;
            }
          }
          // if this is only used by itself and yielding out of the for, remove
          // the induction of the region arg and just simply set it to be the
          // incoming iteration arg.
          if (seenSelf && !seenIllegal) {
            rewriter.setInsertionPoint(innerSelOp);
            rewriter.replaceOpWithNewOp<arith::SelectOp>(
                innerSelOp, innerSelOp.getCondition(),
                negated ? iterOp : innerSelOp.getTrueValue(),
                negated ? innerSelOp.getFalseValue() : iterOp);
            changed = true;
            continue;
          }
        }
      }

      // Only do final hoisting on a variable which can be loop induction
      // replaced.
      //  Otherwise additional work is added outside the break
      if (res.use_empty() &&
          carriedOnlyUnderGuard(regionArg, outerIfOp,
                                opres2.getResultNumber())) {
        if (auto add = topOp.getDefiningOp<arith::AddIOp>()) {
          if (forOp.getRegion().isAncestor(add.getOperand(1).getParentRegion()))
            continue;

          if (add.getOperand(0) != regionArg)
            continue;
          rewriter.setInsertionPoint(outerIfOp);
          SmallVector<Value> results(forYieldOp->getOperands());
          results[regionArg.getArgNumber() - 1] =
              rewriter.replaceOpWithNewOp<arith::AddIOp>(add, add.getOperand(0),
                                                         add.getOperand(1));
          rewriter.setInsertionPoint(forYieldOp);
          rewriter.replaceOpWithNewOp<scf::YieldOp>(forYieldOp, results);
          return success();
        }
      }
    }
    return success(changed);
  }
};

/// Integer values of up to 64 bits, signed or unsigned, and sums and small
/// products of them.
using WideInt = __int128;

/// Width of an integer or index type (index: 64, the width the index casts
/// and the affine maps compute in); 0 for any other type.
static unsigned intWidth(Type t) {
  if (t.isa<IndexType>())
    return 64;
  if (auto it = t.dyn_cast<IntegerType>())
    return it.getWidth();
  return 0;
}

/// Smallest / largest value of a `width`-bit integer read as unsigned or
/// signed.
static WideInt viewMin(unsigned width, bool asUnsigned) {
  return asUnsigned ? 0 : -((WideInt)1 << (width - 1));
}
static WideInt viewMax(unsigned width, bool asUnsigned) {
  return asUnsigned ? ((WideInt)1 << width) - 1
                    : ((WideInt)1 << (width - 1)) - 1;
}

/// The value of the integer constant `v` read as unsigned or signed.
static std::optional<WideInt> constantValue(Value v, bool asUnsigned) {
  APInt c;
  if (!matchPattern(v, m_ConstantInt(&c)) || c.getBitWidth() > 64)
    return std::nullopt;
  return asUnsigned ? (WideInt)c.getZExtValue() : (WideInt)c.getSExtValue();
}

/// The range of values the integer `v` can hold, read as unsigned or signed,
/// as far as its definition shows it: a constant is itself, extui(x) holds
/// x's unsigned range, extsi(x) x's signed range when read as signed;
/// anything else the whole range of its type.
static std::pair<WideInt, WideInt> valueRange(Value v, bool asUnsigned) {
  if (auto c = constantValue(v, asUnsigned))
    return {*c, *c};
  if (auto ext = v.getDefiningOp<ExtUIOp>())
    return {0, viewMax(intWidth(ext.getIn().getType()), true)};
  if (auto ext = v.getDefiningOp<ExtSIOp>())
    if (!asUnsigned) {
      unsigned w = intWidth(ext.getIn().getType());
      return {viewMin(w, false), viewMax(w, false)};
    }
  unsigned w = intWidth(v.getType());
  return {viewMin(w, asUnsigned), viewMax(w, asUnsigned)};
}

/// The range of the index value `v` when its definition bounds it: a
/// constant, an index_cast / index_castui of an integer (valueRange), or the
/// sum of two such values.
static std::optional<std::pair<WideInt, WideInt>> indexRange(Value v) {
  if (auto c = constantValue(v, /*asUnsigned=*/false))
    return std::make_pair(*c, *c);
  if (auto cast = v.getDefiningOp<IndexCastOp>())
    if (intWidth(cast.getIn().getType()) <= 64)
      return valueRange(cast.getIn(), /*asUnsigned=*/false);
  if (auto cast = v.getDefiningOp<IndexCastUIOp>())
    if (intWidth(cast.getIn().getType()) <= 64)
      return valueRange(cast.getIn(), /*asUnsigned=*/true);
  if (auto add = v.getDefiningOp<AddIOp>()) {
    auto a = indexRange(add.getLhs()), b = indexRange(add.getRhs());
    if (a && b)
      return std::make_pair(a->first + b->first, a->second + b->second);
  }
  return std::nullopt;
}

/// Width of C `int` on the targets cgeist compiles for: a signed counter
/// narrower than it is incremented in int and converted back.
static constexpr unsigned kCIntWidth = 32;

// A while loop over a counter narrower than int whose range check
// (WhileToForHelper::counterStaysInRange) cannot show that the counter stays
// in range, because its bound is a runtime value that may lie beyond it
// (`for (uint8_t i = 0; i < n; i++)` with int n, which never ends for
// n > 255) or because a narrow bound lets a power-of-two step overshoot it
// (`for (uint8_t i = 0; i < m; i += 2)`, uint8_t m, which never ends for
// m = 255), is raised to a for loop by assuming that it terminates. C11
// 6.8.5p6 and C++ [intro.progress] let an implementation assume that a loop
// whose controlling expression is not a constant expression terminates when
// it performs no input / output, no volatile access and no synchronisation or
// atomic operation; here that is a body without calls to functions cgeist
// cannot see into, atomics, fences, barriers or volatile LLVM loads and
// stores, in a function the frontend did not mark polygeist.volatile_access
// and calling no function so marked (cgeist does not represent C volatile on
// memref accesses; the frontend marks each function whose body accesses a
// volatile-qualified object, and inlining carries the mark to the caller,
// Passes.h). Such a loop terminates only
// if its counter never wraps: counting by +-2^k with a test read with the
// counter's own signedness, a wrap returns the counter to values the test
// has already accepted, so a loop that wrapped never ends. The raised loop is
// marked polygeist.counter_nowrap and the casts of its counter
// polygeist.nuw / polygeist.nsw (Passes.h), so that AffineCFG reads them as
// the induction variable; a loop whose bound is of a wider type than its
// counter is marked hls.uncounted as well. Not applied to a constant bound
// (the range check is exact there: such a loop never ends), to other steps
// (a wrap can be followed by a different residue that ends the loop) or to a
// signed counter compared unsigned (a negative value ends the loop). Vitis
// HLS pipelines such loops as it does the `int` ones (hls-est-validate
// VITIS_NOTES.md N15).
static llvm::cl::opt<bool> AssumeTermination(
    "canonicalize-for-assume-termination", llvm::cl::init(true),
    llvm::cl::Hidden,
    llvm::cl::desc(
        "Raise a while loop whose counter (narrower than int) is not "
        "provably kept in range by its runtime bound to a for loop by "
        "assuming that the loop terminates, so that the counter does not "
        "wrap (C11 6.8.5p6); false keeps such loops as while loops"));

/// Whether `root` holds an operation that makes running forever observable
/// (see AssumeTermination): a call to a function without a visible body (the
/// bodies of visible callees are searched in turn), an atomic operation, a
/// fence, a barrier, a volatile LLVM load / store, or a call to a function
/// marked polygeist.volatile_access. The mark on the function enclosing the
/// loop is checked by the caller (terminationImpliesNoWrap).
static bool mayDoObservableWork(Operation *root,
                                SmallPtrSetImpl<Operation *> &visited) {
  WalkResult r = root->walk([&](Operation *op) {
    if (isa<FunctionOpInterface>(op) &&
        op->hasAttr(polygeist::kVolatileAccessAttrName))
      return WalkResult::interrupt();
    if (isa<memref::AtomicRMWOp, memref::GenericAtomicRMWOp,
            LLVM::AtomicRMWOp, LLVM::AtomicCmpXchgOp, LLVM::FenceOp,
            polygeist::BarrierOp>(op))
      return WalkResult::interrupt();
    if (auto ld = dyn_cast<LLVM::LoadOp>(op); ld && ld.getVolatile_())
      return WalkResult::interrupt();
    if (auto st = dyn_cast<LLVM::StoreOp>(op); st && st.getVolatile_())
      return WalkResult::interrupt();
    if (auto call = dyn_cast<CallOpInterface>(op)) {
      auto callee =
          dyn_cast_or_null<FunctionOpInterface>(call.resolveCallable());
      if (!callee || callee.isExternal())
        return WalkResult::interrupt();
      if (visited.insert(callee.getOperation()).second &&
          mayDoObservableWork(callee.getOperation(), visited))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return r.wasInterrupted();
}

/// ceil(a / b) for b > 0.
static WideInt ceilDivWide(WideInt a, WideInt b) {
  WideInt q = a / b;
  return (a % b != 0 && a > 0) ? q + 1 : q;
}

struct ForOpInductionReplacement : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  /// The signedness of the index cast `v` is of `init`, if it is one.
  static std::optional<bool> castOf(Value v, Value init) {
    if (auto c = v.getDefiningOp<IndexCastUIOp>())
      if (c.getIn() == init)
        return true;
    if (auto c = v.getDefiningOp<IndexCastOp>())
      if (c.getIn() == init)
        return false;
    return std::nullopt;
  }

  /// The signedness with which `bound` (a bound of the for) was cast from
  /// `init`, possibly plus a constant (a countdown's init + 1): the signedness
  /// MoveWhileToFor counted the loop's counter with. Only the choice of cast
  /// for a counter as wide as int uses it, for any counter starting at `init`.
  static std::optional<bool> boundCastOf(Value bound, Value init) {
    if (auto add = bound.getDefiningOp<AddIOp>())
      if (matchPattern(add.getRhs(), m_Constant()))
        bound = add.getLhs();
    return castOf(bound, init);
  }

  /// The counter's own signedness, when its values are those of the counter
  /// MoveWhileToFor raised the for from, which counterStaysInRange kept in
  /// that range: it starts at the lower bound, cast from `init`, and steps
  /// with the loop (`sameValue`), or, counting down, the upper bound is that
  /// cast plus one and it steps by minus the loop's step. Another counter
  /// with the same start value (the same SSA value: a CSE'd constant, a
  /// shared runtime start) but another step is not: `for (uint8_t i = 0,
  /// j = 0; i < 10; i++, j += 30)` takes j to 270, which wraps.
  static std::optional<bool> ownCastOf(scf::ForOp forOp, Value init,
                                       bool sameValue, Value addStep) {
    if (sameValue)
      return castOf(forOp.getLowerBound(), init);
    std::optional<WideInt> step = constantValue(forOp.getStep(), false);
    std::optional<WideInt> inc = constantValue(addStep, false);
    if (!step || !inc || *inc != -*step)
      return std::nullopt;
    auto add = forOp.getUpperBound().getDefiningOp<AddIOp>();
    if (!add || !matchPattern(add.getRhs(), m_One()))
      return std::nullopt;
    return castOf(add.getLhs(), init);
  }

  /// The loop's own counter in a loop MoveWhileToFor raised by assuming that
  /// it terminates (polygeist.counter_nowrap, whose value is the signedness
  /// its exit test reads the counter with): the signedness that ownCastOf
  /// gives, or, once the start's cast to the bound has been folded to a
  /// constant, the mark's for a constant start that is, read that way, the
  /// lower bound (counting up with the loop's step) or the upper bound minus
  /// one (counting down by it). Any counter with those values never wraps
  /// in such a loop. std::nullopt for other counters and loops.
  static std::optional<bool> assumedOwnCastOf(scf::ForOp forOp, Value init,
                                              bool sameValue, Value addStep) {
    auto mark = forOp->getAttrOfType<BoolAttr>(kCounterNoWrapAttrName);
    if (!mark)
      return std::nullopt;
    bool view = mark.getValue();
    std::optional<bool> own = ownCastOf(forOp, init, sameValue, addStep);
    if (own)
      return *own == view ? own : std::nullopt;
    std::optional<WideInt> start = constantValue(init, view);
    if (!start)
      return std::nullopt;
    if (sameValue) {
      std::optional<WideInt> lb = constantValue(forOp.getLowerBound(), false);
      return lb && *lb == *start ? std::optional<bool>(view) : std::nullopt;
    }
    std::optional<WideInt> step = constantValue(forOp.getStep(), false);
    std::optional<WideInt> inc = constantValue(addStep, false);
    if (!step || !inc || *inc != -*step)
      return std::nullopt;
    std::optional<WideInt> ub = constantValue(forOp.getUpperBound(), false);
    return ub && *ub == *start + 1 ? std::optional<bool>(view) : std::nullopt;
  }

  /// How to cast the iN start `init` of the counter `like` (an iter_arg of
  /// the for, or with `atExit` its result) to index; true is index_castui.
  /// The counter is replaced by index_cast(X -> iN), X = start + k * inc in
  /// index, and the folds downstream (AffineCFG's cast stripping, the
  /// index_cast(index_cast) / index_castui(index_cast) round trips) read each
  /// extui / extsi / index_cast* of it back as X. That is exact only while X
  /// stays in the range of every view the counter is read with: [0, 2^N)
  /// through extui / index_castui, [-2^(N-1), 2^(N-1)) through extsi /
  /// index_cast. A call it is passed to is both views: after inlining, the
  /// callee's parameter reads it with the callee's signedness, which the
  /// upstream index_cast(index_cast) fold does not check. For a counter
  /// narrower than int (its values cross 2^(N-1) in ordinary loops: a
  /// uint8_t over 120..135) a cast is taken when
  ///  - every view is the counter's own signedness and its values are those
  ///    of the loop's own counter (ownCastOf): X is then the counter's
  ///    value, which never leaves that range (counterStaysInRange);
  ///  - or the range of X, from the bounds, steps and start as far as their
  ///    definitions show them, lies in every view.
  /// std::nullopt when neither holds (e.g. a uint8_t counter read as
  /// `(int8_t)i` while it runs over 120..135, or as both `i` and `(int8_t)i`
  /// over 200..209): the counter then stays an iter_arg, whose own iN
  /// arithmetic is exact. A counter as wide as int is assumed to stay below
  /// 2^31 as before, and takes its own signedness when the bounds show it
  /// (`int jj < unsigned row_end`: the for counts from index_castui(start),
  /// so X is the induction variable), else the users' view.
  static std::optional<bool> chooseInitCast(scf::ForOp forOp, Value init,
                                            Value like, bool sameValue,
                                            Value addStep, bool atExit) {
    bool hasUnsigned = false, hasSigned = false;
    for (Operation *u : like.getUsers()) {
      if (isa<IndexCastUIOp, ExtUIOp>(u))
        hasUnsigned = true;
      else if (isa<IndexCastOp, ExtSIOp>(u))
        hasSigned = true;
      else if (isa<CallOpInterface>(u))
        hasUnsigned = hasSigned = true;
    }
    bool usersView = hasUnsigned && !hasSigned;
    unsigned w = intWidth(init.getType());
    // A counter as wide as int, or one no extension reads: its own
    // signedness when the bounds show it (X is then the induction variable
    // itself), else the users' view (a signed one when they disagree).
    if (w >= kCIntWidth || (!hasUnsigned && !hasSigned)) {
      std::optional<bool> own = boundCastOf(forOp.getLowerBound(), init);
      if (!own)
        own = boundCastOf(forOp.getUpperBound(), init);
      return own ? *own : usersView;
    }
    std::optional<bool> own = ownCastOf(forOp, init, sameValue, addStep);
    if (!own)
      own = assumedOwnCastOf(forOp, init, sameValue, addStep);
    if (own && (hasUnsigned ? *own : !*own) && !(hasUnsigned && hasSigned))
      return *own;

    auto lbR = indexRange(forOp.getLowerBound());
    auto ubR = indexRange(forOp.getUpperBound());
    auto stR = indexRange(forOp.getStep());
    if (!lbR || !ubR || !stR || stR->first <= 0)
      return std::nullopt;
    // X = start + d, d in [dLo, dHi].
    WideInt dLo = 0, dHi = 0;
    WideInt incLo = 1, incHi = 1;
    if (!sameValue) {
      if (auto c = constantValue(addStep, /*asUnsigned=*/false)) {
        incLo = incHi = *c;
      } else {
        incLo = viewMin(intWidth(addStep.getType()), false);
        incHi = viewMax(intWidth(addStep.getType()), false);
      }
    }
    if (!atExit) {
      if (ubR->second <= lbR->first) // the body never runs: any cast
        return usersView;
      // iv - lb in [0, span]; with !sameValue, k = (iv - lb) / step.
      WideInt span = ubR->second - 1 - lbR->first;
      WideInt kHi = sameValue ? span : span / stR->first;
      dLo = std::min<WideInt>(0, kHi * incLo);
      dHi = std::max<WideInt>(0, kHi * incHi);
    } else {
      // trips steps of inc (of the loop's step with sameValue), trips =
      // max(0, ceil((ub - lb) / step)) as the exit value is computed.
      WideInt tLo = std::max<WideInt>(
          0, ceilDivWide(ubR->first - lbR->second, stR->second));
      WideInt tHi = std::max<WideInt>(
          0, ceilDivWide(ubR->second - lbR->first, stR->first));
      if (sameValue) {
        incLo = stR->first;
        incHi = stR->second;
      }
      const WideInt big = (WideInt)1 << 62;
      if (tHi > big || incHi > big || incLo < -big)
        return std::nullopt;
      WideInt c[4] = {tLo * incLo, tLo * incHi, tHi * incLo, tHi * incHi};
      dLo = *std::min_element(c, c + 4);
      dHi = *std::max_element(c, c + 4);
    }
    for (bool asUnsigned : {usersView, !usersView}) {
      WideInt sLo = viewMin(w, asUnsigned), sHi = viewMax(w, asUnsigned);
      if (auto c = constantValue(init, asUnsigned))
        sLo = sHi = *c;
      WideInt lo = sLo + dLo, hi = sHi + dHi;
      if ((!hasUnsigned ||
           (lo >= viewMin(w, true) && hi <= viewMax(w, true))) &&
          (!hasSigned || (lo >= viewMin(w, false) && hi <= viewMax(w, false))))
        return asUnsigned;
    }
    return std::nullopt;
  }

  static Value castInit(PatternRewriter &rewriter, Location loc, Value init,
                        bool asUnsigned) {
    if (asUnsigned)
      return rewriter.create<IndexCastUIOp>(loc, rewriter.getIndexType(), init);
    return rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), init);
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    // Defer until after the step is a constant, if possible
    if (auto *icast = forOp.getStep().getDefiningOp())
      if (isa<IndexCastOp, IndexCastUIOp>(icast) &&
          matchPattern(icast->getOperand(0), m_Constant()))
        return failure();
    bool canonicalize = false;
    Block &block = forOp.getRegion().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

    for (auto it : llvm::zip(forOp.getInits(),          // iter from outside
                             forOp.getRegionIterArgs(), // iter inside region
                             forOp.getResults(),        // op results
                             yieldOp.getOperands()      // iter yield
                             )) {

      AddIOp addOp = std::get<3>(it).getDefiningOp<AddIOp>();
      if (!addOp)
        continue;

      if (addOp.getOperand(0) != std::get<1>(it))
        continue;

      if (!addOp.getOperand(1).getParentRegion()->isAncestor(
              forOp->getParentRegion()))
        continue;

      bool sameValue = addOp.getOperand(1) == forOp.getStep();

      APInt rattr;
      APInt sattr;
      if (matchPattern(addOp.getOperand(1), m_ConstantInt(&rattr)))
        if (matchPattern(forOp.getStep(), m_ConstantInt(&sattr))) {
          size_t maxWidth = (rattr.getBitWidth() > sattr.getBitWidth())
                                ? rattr.getBitWidth()
                                : sattr.getBitWidth();
          sameValue |= rattr.zext(maxWidth) == sattr.zext(maxWidth);
        }

      // A narrow counter is replaced only when one cast of its start makes
      // every read of it exact (chooseInitCast).
      bool narrow = !std::get<0>(it).getType().isa<IndexType>();
      std::optional<bool> bodyCast, exitCast;
      if (narrow && !std::get<1>(it).use_empty())
        bodyCast = chooseInitCast(forOp, std::get<0>(it), std::get<1>(it),
                                  sameValue, addOp.getOperand(1),
                                  /*atExit=*/false);
      if (narrow && !std::get<2>(it).use_empty())
        exitCast = chooseInitCast(forOp, std::get<0>(it), std::get<2>(it),
                                  sameValue, addOp.getOperand(1),
                                  /*atExit=*/true);
      // In a loop raised by assuming that it terminates (MoveWhileToFor,
      // AssumeTermination), the loop's own counter never wraps: its value,
      // read with its own signedness, is the index it is cast from, inside
      // the loop and at its exit. The cast is marked so for AffineCFG.
      std::optional<bool> ownNoWrap;
      if (narrow)
        ownNoWrap = assumedOwnCastOf(forOp, std::get<0>(it), sameValue,
                                     addOp.getOperand(1));
      auto markOwnCast = [&](Value cast, std::optional<bool> chosen) {
        if (ownNoWrap && chosen && *chosen == *ownNoWrap)
          cast.getDefiningOp()->setAttr(*ownNoWrap ? kNoUnsignedWrapAttrName
                                                   : kNoSignedWrapAttrName,
                                        rewriter.getUnitAttr());
      };

      if (!std::get<1>(it).use_empty() && (!narrow || bodyCast)) {
        Value init = std::get<0>(it);
        rewriter.setInsertionPointToStart(&forOp.getRegion().front());
        Value replacement = rewriter.create<SubIOp>(
            forOp.getLoc(), forOp.getInductionVar(), forOp.getLowerBound());

        if (!sameValue)
          replacement = rewriter.create<DivUIOp>(forOp.getLoc(), replacement,
                                                 forOp.getStep());

        if (!sameValue) {
          Value step = addOp.getOperand(1);

          if (!step.getType().isa<IndexType>()) {
            step = rewriter.create<IndexCastOp>(forOp.getLoc(),
                                                replacement.getType(), step);
          }

          replacement =
              rewriter.create<MulIOp>(forOp.getLoc(), replacement, step);
        }

        if (narrow)
          init = castInit(rewriter, forOp.getLoc(), init, *bodyCast);

        replacement =
            rewriter.create<AddIOp>(forOp.getLoc(), init, replacement);

        if (!std::get<1>(it).getType().isa<IndexType>()) {
          replacement = rewriter.create<IndexCastOp>(
              forOp.getLoc(), std::get<1>(it).getType(), replacement);
          markOwnCast(replacement, bodyCast);
        }

        rewriter.updateRootInPlace(
            forOp, [&] { std::get<1>(it).replaceAllUsesWith(replacement); });
        canonicalize = true;
      }

      if (!std::get<2>(it).use_empty() && (!narrow || exitCast)) {
        Value init = std::get<0>(it);
        Location loc = forOp.getLoc();
        rewriter.setInsertionPoint(forOp);
        // The counter leaves the loop at init + trips * inc (inc: the loop's
        // step with sameValue), trips = max(0, ceil((ub - lb) / step)), the
        // number of iterations (step > 0): none when the loop does not run
        // (ub < lb), a last partial step counted whole (`i < 10; i += 3`
        // leaves at 12).
        Value trips = rewriter.create<SubIOp>(loc, forOp.getUpperBound(),
                                              forOp.getLowerBound());
        if (!matchPattern(forOp.getStep(), m_One()))
          trips = rewriter.create<CeilDivSIOp>(loc, trips, forOp.getStep());
        auto lbR = indexRange(forOp.getLowerBound());
        auto ubR = indexRange(forOp.getUpperBound());
        if (!lbR || !ubR || ubR->first < lbR->second)
          trips = rewriter.create<MaxSIOp>(
              loc, trips, rewriter.create<ConstantIndexOp>(loc, 0));
        Value inc = sameValue ? forOp.getStep() : addOp.getOperand(1);
        if (!inc.getType().isa<IndexType>())
          inc = rewriter.create<IndexCastOp>(loc, trips.getType(), inc);
        Value replacement = trips;
        if (!matchPattern(inc, m_One()))
          replacement = rewriter.create<MulIOp>(loc, trips, inc);

        if (narrow)
          init = castInit(rewriter, forOp.getLoc(), init, *exitCast);

        replacement =
            rewriter.create<AddIOp>(forOp.getLoc(), init, replacement);

        if (!std::get<1>(it).getType().isa<IndexType>()) {
          replacement = rewriter.create<IndexCastOp>(
              forOp.getLoc(), std::get<1>(it).getType(), replacement);
          markOwnCast(replacement, exitCast);
        }

        rewriter.updateRootInPlace(
            forOp, [&] { std::get<2>(it).replaceAllUsesWith(replacement); });
        canonicalize = true;
      }
    }

    return success(canonicalize);
  }
};

/// Remove unused iterator operands.
// TODO: IRMapping for indvar.
struct RemoveUnusedArgs : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 2> usedBlockArgs;
    SmallVector<OpResult, 2> usedResults;
    SmallVector<Value, 2> usedOperands;

    unsigned i = 0;
    // if the block argument or the result at the
    // same index position have uses do not eliminate.
    for (auto blockArg : op.getRegionIterArgs()) {
      if ((!blockArg.use_empty()) || (!op.getResult(i).use_empty())) {
        usedOperands.push_back(op.getOperand(op.getNumControlOperands() + i));
        usedResults.push_back(op->getOpResult(i));
        usedBlockArgs.push_back(blockArg);
      }
      i++;
    }

    // no work to do.
    if (usedOperands.size() == op.getInits().size())
      return failure();

    auto newForOp =
        rewriter.create<ForOp>(op.getLoc(), op.getLowerBound(),
                               op.getUpperBound(), op.getStep(), usedOperands);
    for (StringRef mark : {kCounterNoWrapAttrName, kUncountedLoopAttrName})
      if (Attribute a = op->getAttr(mark))
        newForOp->setAttr(mark, a);

    if (!newForOp.getBody()->empty())
      rewriter.eraseOp(newForOp.getBody()->getTerminator());

    newForOp.getBody()->getOperations().splice(
        newForOp.getBody()->getOperations().begin(),
        op.getBody()->getOperations());

    rewriter.updateRootInPlace(op, [&] {
      op.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());
      for (auto pair : llvm::zip(usedBlockArgs, newForOp.getRegionIterArgs())) {
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
      }
    });

    // adjust return.
    auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    SmallVector<Value, 2> usedYieldOperands{};
    llvm::transform(usedResults, std::back_inserter(usedYieldOperands),
                    [&](OpResult result) {
                      return yieldOp.getOperand(result.getResultNumber());
                    });
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, usedYieldOperands);

    // Replace the operation's results with the new ones.
    SmallVector<Value, 4> repResults(op.getNumResults());
    for (auto en : llvm::enumerate(usedResults))
      repResults[en.value().cast<OpResult>().getResultNumber()] =
          newForOp.getResult(en.index());

    rewriter.replaceOp(op, repResults);
    return success();
  }
};

struct ReplaceRedundantArgs : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    auto yieldOp = cast<scf::YieldOp>(op.getBody()->getTerminator());
    bool replaced = false;
    unsigned i = 0;
    for (auto blockArg : op.getRegionIterArgs()) {
      for (unsigned j = 0; j < i; j++) {
        if (op.getOperand(op.getNumControlOperands() + i) ==
                op.getOperand(op.getNumControlOperands() + j) &&
            yieldOp.getOperand(i) == yieldOp.getOperand(j)) {

          rewriter.updateRootInPlace(op, [&] {
            op.getResult(i).replaceAllUsesWith(op.getResult(j));
            blockArg.replaceAllUsesWith(op.getRegionIterArgs()[j]);
          });
          replaced = true;
          goto skip;
        }
      }
    skip:
      i++;
    }

    return success(replaced);
  }
};

/*
+struct RemoveNotIf : public OpRewritePattern<IfOp> {
+  using OpRewritePattern<IfOp>::OpRewritePattern;
+
+  LogicalResult matchAndRewrite(IfOp op,
+                                PatternRewriter &rewriter) const override {
+    // Replace the operation if only a subset of its results have uses.
+    if (op.getNumResults() == 0)
+      return failure();
+
+    auto trueYield =
cast<scf::YieldOp>(op.thenRegion().back().getTerminator());
+    auto falseYield =
+        cast<scf::YieldOp>(op.thenRegion().back().getTerminator());
+
+    rewriter.setInsertionPoint(op->getBlock(),
+                               op.getOperation()->getIterator());
+    bool changed = false;
+    for (auto tup :
+         llvm::zip(trueYield.results(), falseYield.results(), op.results())) {
+      if (!std::get<0>(tup).getType().isInteger(1))
+        continue;
+      if (auto top = std::get<0>(tup).getDefiningOp<ConstantOp>()) {
+        if (auto fop = std::get<1>(tup).getDefiningOp<ConstantOp>()) {
+          if (top.getValue().cast<IntegerAttr>().getValue() == 0 &&
+              fop.getValue().cast<IntegerAttr>().getValue() == 1) {
+
+            for (OpOperand &use :
+                 llvm::make_early_inc_range(std::get<2>(tup).getUses())) {
+              changed = true;
+              rewriter.updateRootInPlace(use.getOwner(), [&]() {
+                use.set(rewriter.create<XOrOp>(op.getLoc(), op.condition()));
+              });
+            }
+          }
+          if (top.getValue().cast<IntegerAttr>().getValue() == 1 &&
+              fop.getValue().cast<IntegerAttr>().getValue() == 0) {
+            for (OpOperand &use :
+                 llvm::make_early_inc_range(std::get<2>(tup).getUses())) {
+              changed = true;
+              rewriter.updateRootInPlace(use.getOwner(),
+                                         [&]() { use.set(op.condition()); });
+            }
+          }
+        }
+      }
+    }
+    return changed ? success() : failure();
+  }
+};
+struct RemoveBoolean : public OpRewritePattern<IfOp> {
+  using OpRewritePattern<IfOp>::OpRewritePattern;
+
+  LogicalResult matchAndRewrite(IfOp op,
+                                PatternRewriter &rewriter) const override {
+    bool changed = false;
+
+    if (llvm::all_of(op.results(), [](Value v) {
+          return v.getType().isa<IntegerType>() &&
+                 v.getType().cast<IntegerType>().getWidth() == 1;
+        })) {
+      if (op.thenRegion().getBlocks().size() == 1 &&
+          op.elseRegion().getBlocks().size() == 1) {
+        while (isa<CmpIOp>(op.thenRegion().front().front())) {
+          op.thenRegion().front().front().moveBefore(op);
+          changed = true;
+        }
+        while (isa<CmpIOp>(op.elseRegion().front().front())) {
+          op.elseRegion().front().front().moveBefore(op);
+          changed = true;
+        }
+        if (op.thenRegion().front().getOperations().size() == 1 &&
+            op.elseRegion().front().getOperations().size() == 1) {
+          auto yop1 =
+              cast<scf::YieldOp>(op.thenRegion().front().getTerminator());
+          auto yop2 =
+              cast<scf::YieldOp>(op.elseRegion().front().getTerminator());
+          size_t idx = 0;
+
+          auto c1 = (mlir::Value)rewriter.create<ConstantOp>(
+              op.getLoc(), op.condition().getType(),
+              rewriter.getIntegerAttr(op.condition().getType(), 1));
+          auto notcond = (mlir::Value)rewriter.create<mlir::XOrOp>(
+              op.getLoc(), op.condition(), c1);
+
+          std::vector<Value> replacements;
+          for (auto res : op.results()) {
+            auto rep = rewriter.create<OrOp>(
+                op.getLoc(),
+                rewriter.create<AndOp>(op.getLoc(), op.condition(),
+                                       yop1.results()[idx]),
+                rewriter.create<AndOp>(op.getLoc(), notcond,
+                                       yop2.results()[idx]));
+            replacements.push_back(rep);
+            idx++;
+          }
+          rewriter.replaceOp(op, replacements);
+          // op.erase();
+          return success();
+        }
+      }
+    }
+
+    if (op.thenRegion().getBlocks().size() == 1 &&
+        op.elseRegion().getBlocks().size() == 1 &&
+        op.thenRegion().front().getOperations().size() == 1 &&
+        op.elseRegion().front().getOperations().size() == 1) {
+      auto yop1 = cast<scf::YieldOp>(op.thenRegion().front().getTerminator());
+      auto yop2 = cast<scf::YieldOp>(op.elseRegion().front().getTerminator());
+      size_t idx = 0;
+
+      std::vector<Value> replacements;
+      for (auto res : op.results()) {
+        auto rep =
+            rewriter.create<SelectOp>(op.getLoc(), op.condition(),
+                                      yop1.results()[idx],
yop2.results()[idx]);
+        replacements.push_back(rep);
+        idx++;
+      }
+      rewriter.replaceOp(op, replacements);
+      return success();
+    }
+    return changed ? success() : failure();
+  }
+};
*/

bool isTopLevelArgValue(Value value, Region *region) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    return arg.getParentRegion() == region;
  return false;
}

bool isBlockArg(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    return true;
  return false;
}

bool dominateWhile(Value value, WhileOp loop) {
  if (Operation *op = value.getDefiningOp()) {
    DominanceInfo dom(loop);
    return dom.properlyDominates(op, loop);
  } else if (auto arg = dyn_cast<BlockArgument>(value)) {
    return arg.getOwner()->getParentOp()->isProperAncestor(loop);
  } else {
    assert("????");
    return false;
  }
}

bool canMoveOpOutsideWhile(Operation *op, WhileOp loop) {
  DominanceInfo dom(loop);
  for (auto operand : op->getOperands()) {
    if (!dom.properlyDominates(operand, loop))
      return false;
  }
  return true;
}

static bool isUnsignedPredicate(CmpIPredicate p) {
  return p == CmpIPredicate::ult || p == CmpIPredicate::ule ||
         p == CmpIPredicate::ugt || p == CmpIPredicate::uge;
}

struct WhileToForHelper {
  WhileOp loop;
  CmpIOp cmpIOp;
  Value step;
  Value lb;
  bool lb_addOne;
  Value ub;
  bool ub_addOne;
  bool ub_cloneMove;
  bool negativeStep;
  // The counter's own values (its start and step) are read as unsigned: it is
  // compared through an extui (an unsigned C counter narrower than its
  // bound) or, unextended, with an unsigned predicate.
  bool indVarUnsigned;
  // The comparison is unsigned (ult/ule/ugt/uge): its bound is read as
  // unsigned.
  bool boundUnsigned;
  // Which of lb / ub is the comparison's bound (the other is the counter's
  // start).
  bool lbIsBound;
  bool ubIsBound;
  AddIOp addIOp;
  BlockArgument indVar;
  size_t afterArgIdx;
  // Set by counterStaysInRange: the counter is not provably kept in range
  // and the loop is raised by assuming that it terminates (AssumeTermination);
  // with uncounted, its bound may lie beyond the counter's range.
  bool assumedNoWrap = false;
  bool uncounted = false;

  /// The for loop counts in index arithmetic, the while in the counter's own
  /// iW arithmetic, which wraps modulo 2^W (arith.addi); they agree only if
  /// the counter never leaves its range before the test fails, so a counter
  /// narrower than int is raised only when that is provable. Such counters
  /// wrap in C: unsigned ones modulo 2^W (defined), signed ones because their
  /// `++` / `+=` is computed in int and converted back (implementation-
  /// defined; clang and Vitis wrap). A counter as wide as int is assumed not
  /// to wrap, as before: for a signed one that is C's undefined overflow, and
  /// the IR cannot tell an unsigned one (`unsigned j < n`, which could wrap
  /// at 2^32) from a signed one compared unsigned (`int jj < unsigned
  /// row_end; jj += 4`, spector_hls/spmv: cmpi ult on i32 either way).
  /// The range is that of the counter's values as the test reads them:
  /// [0, 2^W) through an extui or an unextended unsigned compare,
  /// [-2^(W-1), 2^(W-1)) through an extsi or an unextended signed compare,
  /// [0, 2^(W-1)) through an extsi read by an unsigned compare (a negative
  /// value is huge there). Counting up by s with `i < B` (`i <= B`), the
  /// largest value that passes is at most B - 1 (B), and it plus s must stay
  /// in range; counting down by d with `i > B` (`i >= B`), the smallest is at
  /// least B + 1 (B), and it minus d must stay in range. A constant start
  /// makes this exact (the last value is start + k * s); a runtime bound
  /// counts with the largest (smallest) value its definition allows, so a
  /// bound zero-extended from at most W bits is in range while an `int n`
  /// against a uint8_t counter is not (`for (uint8_t i = 0; i < n; i++)`
  /// never ends for n > 255). A runtime step is not provable.
  /// When the proof fails, the loop may still be raised by assuming that it
  /// terminates (terminationImpliesNoWrap; AssumeTermination).
  bool counterStaysInRange(Operation *ext) {
    assumedNoWrap = false;
    uncounted = false;
    if (counterProvablyInRange(ext))
      return true;
    if (!terminationImpliesNoWrap(ext))
      return false;
    assumedNoWrap = true;
    // The bound's values as the test reads them, against the counter's.
    unsigned w = intWidth(indVar.getType());
    auto [bMin, bMax] = valueRange(cmpIOp.getRhs(), boundUnsigned);
    WideInt lo = viewMin(w, indVarUnsigned), hi = viewMax(w, indVarUnsigned);
    uncounted = bMin < lo || bMax > hi;
    return true;
  }

  /// Whether the loop may be assumed to terminate and its termination
  /// implies that its counter never wraps (see AssumeTermination): a counter
  /// narrower than int, a runtime bound, a constant step of +-2^k with
  /// 2^k <= 2^(W-1), a test that reads the counter with its own signedness
  /// (not a sign-extended counter compared unsigned) and nothing in the loop
  /// that makes running forever observable (mayDoObservableWork).
  bool terminationImpliesNoWrap(Operation *ext) {
    if (!AssumeTermination)
      return false;
    unsigned w = intWidth(indVar.getType());
    if (w == 0 || w >= kCIntWidth || intWidth(cmpIOp.getRhs().getType()) > 64)
      return false;
    if (ext && isa<ExtSIOp>(ext) && boundUnsigned)
      return false;
    if (constantValue(cmpIOp.getRhs(), boundUnsigned))
      return false;
    std::optional<WideInt> s = constantValue(step, /*asUnsigned=*/false);
    if (!s || *s == 0)
      return false;
    WideInt mag = *s < 0 ? -*s : *s;
    if ((mag & (mag - 1)) != 0 || mag > ((WideInt)1 << (w - 1)))
      return false;
    if (auto fn = loop->getParentOfType<FunctionOpInterface>();
        fn && fn->hasAttr(polygeist::kVolatileAccessAttrName))
      return false;
    SmallPtrSet<Operation *, 4> visited;
    return !mayDoObservableWork(loop.getOperation(), visited);
  }

  bool counterProvablyInRange(Operation *ext) {
    CmpIPredicate pred = cmpIOp.getPredicate();
    unsigned w = intWidth(indVar.getType());
    if (w >= kCIntWidth)
      return true;
    if (w == 0 || intWidth(cmpIOp.getRhs().getType()) > 64)
      return false;

    WideInt lo = viewMin(w, indVarUnsigned);
    WideInt hi = viewMax(w, indVarUnsigned);
    Value init = loop.getInits()[indVar.getArgNumber()];
    std::optional<WideInt> start = constantValue(init, indVarUnsigned);
    if (ext && isa<ExtSIOp>(ext) && boundUnsigned) {
      lo = 0;
      if (start ? *start < 0 : valueRange(init, false).first < 0)
        return false;
    }

    std::optional<WideInt> s = constantValue(step, /*asUnsigned=*/false);
    if (!s || *s == 0)
      return false;
    auto [bMin, bMax] = valueRange(cmpIOp.getRhs(), boundUnsigned);
    if (*s > 0) {
      // The largest value the test accepts.
      WideInt last = (pred == CmpIPredicate::slt || pred == CmpIPredicate::ult)
                         ? bMax - 1
                         : bMax;
      if (start) {
        if (*start > last) // the body never runs
          return true;
        last = *start + (last - *start) / *s * *s;
      }
      return last + *s <= hi;
    }
    WideInt d = -*s;
    // The smallest value the test accepts.
    WideInt first = (pred == CmpIPredicate::sgt || pred == CmpIPredicate::ugt)
                        ? bMin + 1
                        : bMin;
    if (start) {
      if (*start < first) // the body never runs
        return true;
      first = *start - (*start - first) / d * d;
    }
    return first - d >= lo;
  }

  bool computeLegality(bool sizeCheck, Value lookThrough = nullptr) {
    step = nullptr;
    lb = nullptr;
    lb_addOne = false;
    ub = nullptr;
    ub_addOne = false;
    ub_cloneMove = false;
    negativeStep = false;
    indVarUnsigned = false;
    boundUnsigned = isUnsignedPredicate(cmpIOp.getPredicate());
    lbIsBound = false;
    ubIsBound = false;
    assumedNoWrap = false;
    uncounted = false;

    auto condOp = loop.getConditionOp();
    indVar = dyn_cast<BlockArgument>(cmpIOp.getLhs());
    Type extType = nullptr;
    Operation *ext = nullptr;
    // A counter narrower than its bound is compared after a widening:
    // `for (int8_t i ...; i < 64;)` compares extsi(i), `for (uint8_t i ...)`
    // compares extui(i). Both extensions are monotone, so the bound test on
    // the extended value is the same test on the counter while the counter
    // does not wrap (counterStaysInRange).
    if (auto extsi = cmpIOp.getLhs().getDefiningOp<ExtSIOp>()) {
      indVar = dyn_cast<BlockArgument>(extsi.getIn());
      extType = extsi.getType();
      ext = extsi;
    } else if (auto extui = cmpIOp.getLhs().getDefiningOp<ExtUIOp>()) {
      indVar = dyn_cast<BlockArgument>(extui.getIn());
      extType = extui.getType();
      ext = extui;
      indVarUnsigned = true;
    } else if (boundUnsigned) {
      indVarUnsigned = true;
    }
    // Condition is not the same as an induction variable
    {
      if (!indVar) {
        return false;
      }

      if (indVar.getOwner() != &loop.getBefore().front())
        return false;
    }

    // Before region contains more than just the comparison
    {
      size_t size = loop.getBefore().front().getOperations().size();
      if (extType)
        size--;
      if (!sizeCheck)
        size--;
      if (size != 2) {
        return false;
      }
    }

    SmallVector<size_t, 2> afterArgs;
    for (auto pair : llvm::enumerate(condOp.getArgs())) {
      if (pair.value() == indVar)
        afterArgs.push_back(pair.index());
    }

    auto endYield = cast<YieldOp>(loop.getAfter().back().getTerminator());

    // Check that the block argument is actually an induction var:
    //   Namely, its next value adds to the previous with an invariant step.
    addIOp =
        endYield.getResults()[indVar.getArgNumber()].getDefiningOp<AddIOp>();
    if (!addIOp && lookThrough) {
      bool negateLookThrough = false;
      while (auto neg = lookThrough.getDefiningOp<XOrIOp>())
        if (matchPattern(neg.getOperand(1), m_One())) {
          lookThrough = neg.getOperand(0);
          negateLookThrough = !negateLookThrough;
        }

      if (auto ifOp = endYield.getResults()[indVar.getArgNumber()]
                          .getDefiningOp<IfOp>()) {
        Value condition = ifOp.getCondition();
        while (auto neg = condition.getDefiningOp<XOrIOp>())
          if (matchPattern(neg.getOperand(1), m_One())) {
            condition = neg.getOperand(0);
            negateLookThrough = !negateLookThrough;
          }
        if (ifOp.getCondition() == lookThrough) {
          for (auto r : llvm::enumerate(ifOp.getResults())) {
            if (r.value() == endYield.getResults()[indVar.getArgNumber()]) {
              addIOp = (negateLookThrough ? ifOp.elseYield() : ifOp.thenYield())
                           .getOperand(r.index())
                           .getDefiningOp<AddIOp>();
              break;
            }
          }
        }
      } else if (auto selOp = endYield.getResults()[indVar.getArgNumber()]
                                  .getDefiningOp<SelectOp>()) {
        Value condition = selOp.getCondition();
        while (auto neg = condition.getDefiningOp<XOrIOp>())
          if (matchPattern(neg.getOperand(1), m_One())) {
            condition = neg.getOperand(0);
            negateLookThrough = !negateLookThrough;
          }
        if (selOp.getCondition() == lookThrough)
          addIOp =
              (negateLookThrough ? selOp.getFalseValue() : selOp.getTrueValue())
                  .getDefiningOp<AddIOp>();
      }
    }
    if (!addIOp) {
      return false;
    }

    for (auto afterArg : afterArgs) {
      auto arg = loop.getAfter().getArgument(afterArg);
      if (addIOp.getOperand(0) == arg) {
        step = addIOp.getOperand(1);
        afterArgIdx = afterArg;
        break;
      }
      if (addIOp.getOperand(1) == arg) {
        step = addIOp.getOperand(0);
        afterArgIdx = afterArg;
        break;
      }
    }

    if (!step) {
      return false;
    }

    // Cannot transform for if step is not loop-invariant
    if (auto *op = step.getDefiningOp()) {
      if (loop->isAncestor(op)) {
        return false;
      }
    }
    // Nor when the step is a block argument of the loop itself (or of a
    // region nested in it): `m += m` doubles the counter, it does not step it
    // by a constant amount. The scf.for built from it would read its own
    // iteration argument as the step (invalid IR: "operand does not dominate
    // this use", W8b repro r_step_iv.c, machsuite sort_merge).
    if (auto arg = dyn_cast<BlockArgument>(step))
      if (loop->isAncestor(arg.getOwner()->getParentOp()))
        return false;

    negativeStep = false;
    if (auto cop = step.getDefiningOp<ConstantIntOp>()) {
      if (cop.value() < 0) {
        negativeStep = true;
      }
    } else if (auto cop = step.getDefiningOp<ConstantIndexOp>()) {
      if (cop.value() < 0)
        negativeStep = true;
    }

    if (!negativeStep)
      lb = loop.getOperand(indVar.getArgNumber());
    else {
      ub = loop.getOperand(indVar.getArgNumber());
      ub_addOne = true;
    }

    auto cmpRhs = cmpIOp.getRhs();
    if (dominateWhile(cmpRhs, loop)) {
      switch (cmpIOp.getPredicate()) {
      case CmpIPredicate::slt:
      case CmpIPredicate::ult: {
        ub = cmpRhs;
        ubIsBound = true;
        break;
      }
      case CmpIPredicate::ule:
      case CmpIPredicate::sle: {
        ub = cmpRhs;
        ubIsBound = true;
        ub_addOne = true;
        break;
      }
      case CmpIPredicate::uge:
      case CmpIPredicate::sge: {
        lb = cmpRhs;
        lbIsBound = true;
        break;
      }

      case CmpIPredicate::ugt:
      case CmpIPredicate::sgt: {
        lb = cmpRhs;
        lbIsBound = true;
        lb_addOne = true;
        break;
      }
      case CmpIPredicate::eq:
      case CmpIPredicate::ne: {
        return false;
      }
      }
    } else {
      if (negativeStep)
        return false;
      auto *op = cmpIOp.getRhs().getDefiningOp();
      if (!op || !canMoveOpOutsideWhile(op, loop) || (op->getNumResults() != 1))
        return false;
      ub = cmpIOp.getRhs();
      ubIsBound = true;
      ub_cloneMove = true;
    }

    if (!lb || !ub)
      return false;
    return counterStaysInRange(ext);
  }

  void prepareFor(PatternRewriter &rewriter) {
    if (ub_cloneMove) {
      auto *op = ub.getDefiningOp();
      assert(op);
      auto *newOp = rewriter.clone(*op);
      rewriter.replaceOp(op, newOp->getResults());
      ub = newOp->getResult(0);
    }

    if (negativeStep) {
      if (auto cop = step.getDefiningOp<ConstantIntOp>()) {
        step = rewriter.create<ConstantIndexOp>(cop.getLoc(), -cop.value());
      } else {
        auto cop2 = step.getDefiningOp<ConstantIndexOp>();
        step = rewriter.create<ConstantIndexOp>(cop2.getLoc(), -cop2.value());
      }
    }

    // The counter's own values (its start and step) are read with the
    // counter's signedness, the comparison's bound with the predicate's
    // (`ap_uint<8> i < ap_uint<8> len` compares ult in i8: len >= 128 is a
    // large bound, not a negative one). The `+ 1` of an inclusive bound or a
    // countdown's start is added in index, where it cannot wrap (a uint8_t
    // countdown from 255 has ub 256, not 0).
    auto toIndex = [&](Value v, bool asUnsigned) -> Value {
      auto idxTy = IndexType::get(loop.getContext());
      if (v.getType().isa<IndexType>())
        return v;
      if (asUnsigned)
        return rewriter.create<IndexCastUIOp>(loop.getLoc(), idxTy, v);
      return rewriter.create<IndexCastOp>(loop.getLoc(), idxTy, v);
    };
    auto plusOne = [&](Value v) -> Value {
      return rewriter.create<AddIOp>(
          loop.getLoc(), v,
          rewriter.create<ConstantIndexOp>(loop.getLoc(), 1));
    };
    ub = toIndex(ub, ubIsBound ? boundUnsigned : indVarUnsigned);
    if (ub_addOne)
      ub = plusOne(ub);
    lb = toIndex(lb, lbIsBound ? boundUnsigned : indVarUnsigned);
    if (lb_addOne)
      lb = plusOne(lb);
    step = toIndex(step, indVarUnsigned);
  }
};

struct MoveWhileToFor : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    auto condOp = loop.getConditionOp();
    SmallVector<Value, 2> results = {condOp.getArgs()};
    WhileToForHelper helper;
    helper.loop = loop;
    helper.cmpIOp = condOp.getCondition().getDefiningOp<CmpIOp>();
    if (!helper.cmpIOp) {
      return failure();
    }
    if (!helper.computeLegality(/*sizeCheck*/ true))
      return failure();
    // Every value the condition passes to the body seeds an iter_arg of the
    // for and is yielded from its body, so it has to exist outside the loop:
    // a before-block argument (its init is used) or a value defined before
    // the loop. A before-region op result (the extension a narrow counter is
    // compared through, when the body shares it) would be erased with the
    // while while still in use; MoveSideEffectFreeWhile recomputes such
    // casts in the body first.
    for (Value arg : condOp.getArgs())
      if (!isTopLevelArgValue(arg, &loop.getBefore()) &&
          !dominateWhile(arg, loop))
        return failure();
    helper.prepareFor(rewriter);

    // input of the for goes the input of the scf::while plus the output taken
    // from the conditionOp.
    SmallVector<Value, 8> forArgs;
    forArgs.append(loop.getInits().begin(), loop.getInits().end());

    for (Value arg : condOp.getArgs()) {
      Type cst = nullptr;
      bool castUnsigned = false;
      if (auto idx = arg.getDefiningOp<IndexCastOp>()) {
        cst = idx.getType();
        arg = idx.getIn();
      } else if (auto idx = arg.getDefiningOp<IndexCastUIOp>()) {
        cst = idx.getType();
        arg = idx.getIn();
        castUnsigned = true;
      }
      Value res;
      if (isTopLevelArgValue(arg, &loop.getBefore())) {
        auto blockArg = arg.cast<BlockArgument>();
        auto pos = blockArg.getArgNumber();
        res = loop.getInits()[pos];
      } else
        res = arg;
      if (cst) {
        if (castUnsigned)
          res = rewriter.create<IndexCastUIOp>(res.getLoc(), cst, res);
        else
          res = rewriter.create<IndexCastOp>(res.getLoc(), cst, res);
      }
      forArgs.push_back(res);
    }

    auto forloop = rewriter.create<scf::ForOp>(loop.getLoc(), helper.lb,
                                               helper.ub, helper.step, forArgs);
    // Raised by assuming that the loop terminates (AssumeTermination): its
    // counter casts are marked (ForOpInductionReplacement), and a bound that
    // may lie beyond the counter's range makes it an uncounted loop.
    if (helper.assumedNoWrap)
      forloop->setAttr(kCounterNoWrapAttrName,
                       rewriter.getBoolAttr(helper.indVarUnsigned));
    if (helper.uncounted)
      forloop->setAttr(kUncountedLoopAttrName, rewriter.getUnitAttr());

    if (!forloop.getBody()->empty())
      rewriter.eraseOp(forloop.getBody()->getTerminator());

    auto oldYield = cast<scf::YieldOp>(loop.getAfter().front().getTerminator());

    rewriter.updateRootInPlace(loop, [&] {
      for (auto pair :
           llvm::zip(loop.getAfter().getArguments(), condOp.getArgs())) {
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
      }
    });
    loop.getAfter().front().eraseArguments([](BlockArgument) { return true; });

    SmallVector<Value, 2> yieldOperands;
    for (auto oldYieldArg : oldYield.getResults())
      yieldOperands.push_back(oldYieldArg);

    IRMapping outmap;
    outmap.map(loop.getBefore().getArguments(), yieldOperands);
    for (auto arg : condOp.getArgs())
      yieldOperands.push_back(outmap.lookupOrDefault(arg));

    rewriter.setInsertionPoint(oldYield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(oldYield, yieldOperands);

    size_t pos = loop.getInits().size();

    rewriter.updateRootInPlace(loop, [&] {
      for (auto pair : llvm::zip(loop.getBefore().getArguments(),
                                 forloop.getRegionIterArgs().drop_back(pos))) {
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
      }
    });

    forloop.getBody()->getOperations().splice(
        forloop.getBody()->getOperations().begin(),
        loop.getAfter().front().getOperations());

    SmallVector<Value, 2> replacements;
    replacements.append(forloop.getResults().begin() + pos,
                        forloop.getResults().end());

    rewriter.replaceOp(loop, replacements);
    return success();
  }
};

// If and and with something is preventing creating a for
// move the and into the after body guarded by an if
struct MoveWhileAndDown : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    auto condOp = loop.getConditionOp();
    auto andIOp = condOp.getCondition().getDefiningOp<AndIOp>();
    if (!andIOp)
      return failure();
    for (int i = 0; i < 2; i++) {
      WhileToForHelper helper;
      helper.loop = loop;
      helper.cmpIOp = andIOp->getOperand(i).getDefiningOp<CmpIOp>();
      if (!helper.cmpIOp)
        continue;

      YieldOp oldYield = cast<YieldOp>(loop.getAfter().front().getTerminator());

      Value extraCmp = andIOp->getOperand(1 - i);
      Value lookThrough = nullptr;
      if (auto BA = dyn_cast<BlockArgument>(extraCmp)) {
        lookThrough = oldYield.getOperand(BA.getArgNumber());
      }
      if (!helper.computeLegality(/*sizeCheck*/ false, lookThrough)) {
        continue;
      }

      SmallVector<BlockArgument, 2> origBeforeArgs(
          loop.getBeforeArguments().begin(), loop.getBeforeArguments().end());

      SmallVector<BlockArgument, 2> origAfterArgs(
          loop.getAfterArguments().begin(), loop.getAfterArguments().end());

      IRMapping preMap;
      for (auto tup : llvm::zip(origBeforeArgs, loop.getInits()))
        preMap.map(std::get<0>(tup), std::get<1>(tup));
      for (auto &op : loop.getBefore().front()) {
        if (&op == condOp)
          break;
        preMap.map(op.getResults(), rewriter.clone(op, preMap)->getResults());
      }
      IfOp unroll = rewriter.create<IfOp>(loop.getLoc(), loop.getResultTypes(),
                                          preMap.lookup(condOp.getCondition()));

      if (unroll.getThenRegion().getBlocks().size())
        rewriter.eraseBlock(unroll.thenBlock());
      rewriter.createBlock(&unroll.getThenRegion());
      rewriter.createBlock(&unroll.getElseRegion());

      rewriter.setInsertionPointToEnd(unroll.elseBlock());
      SmallVector<Value> unrollYield;
      for (auto v : condOp.getArgs())
        unrollYield.push_back(preMap.lookup(v));
      rewriter.create<YieldOp>(loop.getLoc(), unrollYield);
      rewriter.setInsertionPointToEnd(unroll.thenBlock());

      SmallVector<Value, 2> nextInits(unrollYield.begin(), unrollYield.end());
      Value falsev =
          rewriter.create<ConstantIntOp>(loop.getLoc(), 0, extraCmp.getType());
      Value truev =
          rewriter.create<ConstantIntOp>(loop.getLoc(), 1, extraCmp.getType());
      nextInits.push_back(truev);
      nextInits.push_back(loop.getInits()[helper.indVar.getArgNumber()]);

      SmallVector<Type> resTys;
      for (auto a : nextInits)
        resTys.push_back(a.getType());

      auto nop = rewriter.create<WhileOp>(loop.getLoc(), resTys, nextInits);
      rewriter.createBlock(&nop.getBefore());
      SmallVector<Value> newBeforeYieldArgs;
      for (auto a : origAfterArgs) {
        auto arg = nop.getBefore().addArgument(a.getType(), a.getLoc());
        newBeforeYieldArgs.push_back(arg);
      }
      Value notExited = nop.getBefore().front().addArgument(extraCmp.getType(),
                                                            loop.getLoc());
      newBeforeYieldArgs.push_back(notExited);

      Value trueInd = nop.getBefore().front().addArgument(
          helper.indVar.getType(), loop.getLoc());
      newBeforeYieldArgs.push_back(trueInd);

      {
        IRMapping postMap;
        postMap.map(helper.indVar, trueInd);
        // The counter may be compared through an extension (computeLegality
        // looked through it): rebuild it on the new counter first.
        if (helper.cmpIOp.getLhs() != helper.indVar)
          if (Operation *ext = helper.cmpIOp.getLhs().getDefiningOp())
            postMap.map(ext->getResults(),
                        rewriter.clone(*ext, postMap)->getResults());
        auto newCmp = cast<CmpIOp>(rewriter.clone(*helper.cmpIOp, postMap));
        rewriter.create<ConditionOp>(condOp.getLoc(), newCmp,
                                     newBeforeYieldArgs);
      }

      rewriter.createBlock(&nop.getAfter());
      SmallVector<Value> postElseYields;
      for (auto a : origAfterArgs) {
        auto arg = nop.getAfter().front().addArgument(a.getType(), a.getLoc());
        postElseYields.push_back(arg);
        a.replaceAllUsesWith(arg);
      }
      SmallVector<Type, 4> resultTypes(loop.getResultTypes());
      resultTypes.push_back(notExited.getType());
      notExited = nop.getAfter().front().addArgument(notExited.getType(),
                                                     loop.getLoc());

      trueInd =
          nop.getAfter().front().addArgument(trueInd.getType(), loop.getLoc());

      IfOp guard = rewriter.create<IfOp>(loop.getLoc(), resultTypes, notExited);
      if (guard.getThenRegion().getBlocks().size())
        rewriter.eraseBlock(guard.thenBlock());
      Block *post = rewriter.splitBlock(&loop.getAfter().front(),
                                        loop.getAfter().front().begin());
      rewriter.createBlock(&guard.getThenRegion());
      rewriter.createBlock(&guard.getElseRegion());
      rewriter.mergeBlocks(post, guard.thenBlock());

      {
        IRMapping postMap;
        for (auto tup : llvm::zip(origBeforeArgs, oldYield.getOperands())) {
          postMap.map(std::get<0>(tup), std::get<1>(tup));
        }
        rewriter.setInsertionPoint(oldYield);
        for (auto &op : loop.getBefore().front()) {
          if (&op == condOp)
            break;
          postMap.map(op.getResults(),
                      rewriter.clone(op, postMap)->getResults());
        }
        SmallVector<Value> postIfYields;
        for (auto a : condOp.getArgs()) {
          postIfYields.push_back(postMap.lookup(a));
        }
        postIfYields.push_back(postMap.lookup(extraCmp));
        oldYield->setOperands(postIfYields);
      }

      rewriter.setInsertionPointToEnd(guard.elseBlock());
      postElseYields.push_back(falsev);
      rewriter.create<YieldOp>(loop.getLoc(), postElseYields);

      rewriter.setInsertionPointToEnd(&nop.getAfter().front());
      SmallVector<Value> postAfter(guard.getResults());
      IRMapping postMap;
      postMap.map(helper.indVar, trueInd);
      postMap.map(postElseYields[helper.afterArgIdx], trueInd);
      assert(helper.addIOp.getLhs() == postElseYields[helper.afterArgIdx] ||
             helper.addIOp.getRhs() == postElseYields[helper.afterArgIdx]);
      postAfter.push_back(
          cast<AddIOp>(rewriter.clone(*helper.addIOp, postMap)));
      rewriter.create<YieldOp>(loop.getLoc(), postAfter);

      rewriter.setInsertionPointToEnd(unroll.thenBlock());
      rewriter.create<YieldOp>(
          loop.getLoc(), nop.getResults().take_front(loop.getResults().size()));

      rewriter.replaceOp(loop, unroll.getResults());

      return success();
    }

    return failure();
  }
};

struct MoveWhileDown : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    if (auto ifOp = term.getCondition().getDefiningOp<scf::IfOp>()) {
      if (ifOp.getNumResults() != term.getArgs().size() + 1)
        return failure();
      if (ifOp.getResult(0) != term.getCondition())
        return failure();
      for (size_t i = 1; i < ifOp.getNumResults(); ++i) {
        if (ifOp.getResult(i) != term.getArgs()[i - 1])
          return failure();
      }
      auto yield1 =
          cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
      auto yield2 =
          cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
      if (auto cop = yield1.getOperand(0).getDefiningOp<ConstantIntOp>()) {
        if (cop.value() == 0)
          return failure();
      } else
        return failure();
      if (auto cop = yield2.getOperand(0).getDefiningOp<ConstantIntOp>()) {
        if (cop.value() != 0)
          return failure();
      } else
        return failure();
      if (ifOp.getElseRegion().front().getOperations().size() != 1)
        return failure();
      op.getAfter().front().getOperations().splice(
          op.getAfter().front().begin(),
          ifOp.getThenRegion().front().getOperations());
      rewriter.updateRootInPlace(term, [&] {
        term.getConditionMutable().assign(ifOp.getCondition());
      });
      SmallVector<Value, 2> args;
      for (size_t i = 1; i < yield2.getNumOperands(); ++i) {
        args.push_back(yield2.getOperand(i));
      }
      rewriter.updateRootInPlace(term,
                                 [&] { term.getArgsMutable().assign(args); });
      rewriter.eraseOp(yield2);
      rewriter.eraseOp(ifOp);

      for (size_t i = 0; i < op.getAfter().front().getNumArguments(); ++i) {
        op.getAfter().front().getArgument(i).replaceAllUsesWith(
            yield1.getOperand(i + 1));
      }
      rewriter.eraseOp(yield1);
      // TODO move operands from begin to after
      SmallVector<Value> todo(op.getBefore().front().getArguments().begin(),
                              op.getBefore().front().getArguments().end());
      for (auto &op : op.getBefore().front()) {
        for (auto res : op.getResults()) {
          todo.push_back(res);
        }
      }

      rewriter.updateRootInPlace(op, [&] {
        for (auto val : todo) {
          auto na =
              op.getAfter().front().addArgument(val.getType(), op->getLoc());
          val.replaceUsesWithIf(na, [&](OpOperand &u) -> bool {
            return op.getAfter().isAncestor(u.getOwner()->getParentRegion());
          });
          args.push_back(val);
        }
      });

      rewriter.updateRootInPlace(term,
                                 [&] { term.getArgsMutable().assign(args); });

      SmallVector<Type, 4> tys;
      for (auto a : args)
        tys.push_back(a.getType());

      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.getInits());
      op2.getBefore().takeBody(op.getBefore());
      op2.getAfter().takeBody(op.getAfter());
      SmallVector<Value, 4> replacements;
      for (auto a : op2.getResults()) {
        if (replacements.size() == op.getResults().size())
          break;
        replacements.push_back(a);
      }
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }
};

// Given code of the structure
// scf.while ()
//    ...
//    %z = if (%c) {
//       %i1 = ..
//       ..
//    } else {
//    }
//    condition (%c) %z#0 ..
//  } loop {
//    ...
//  }
// Move the body of the if into the lower loo

struct MoveWhileDown2 : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  /// Populates `crossing` with values (op results) that are defined in the same
  /// block as `op` and above it, and used by at least one op in the same block
  /// below `op`. Uses may be in nested regions.
  static void findValuesUsedBelow(IfOp op, llvm::SetVector<Value> &crossing) {
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      for (Value value : it->getResults()) {
        for (Operation *user : value.getUsers()) {
          // ignore use of condition
          if (user == op)
            continue;

          if (op->isAncestor(user)) {
            crossing.insert(value);
            break;
          }
        }
      }
    }

    for (Value value : op->getBlock()->getArguments()) {
      for (Operation *user : value.getUsers()) {
        // ignore use of condition
        if (user == op)
          continue;

        if (op->isAncestor(user)) {
          crossing.insert(value);
          break;
        }
      }
    }
    // No need to process block arguments, they are assumed to be induction
    // variables and will be replicated.
  }

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    if (auto ifOp = dyn_cast_or_null<scf::IfOp>(term->getPrevNode())) {
      if (ifOp.getCondition() != term.getCondition())
        return failure();

      SmallVector<std::pair<BlockArgument, Value>, 2> m;

      // The return results of the while which are used
      SmallVector<Value, 2> prevResults;
      // The corresponding value in the before which
      // is to be returned
      SmallVector<Value, 2> condArgs;

      SmallVector<std::pair<size_t, Value>, 2> afterYieldRewrites;
      auto afterYield = cast<YieldOp>(op.getAfter().front().back());
      for (auto pair :
           llvm::zip(op.getResults(), term.getArgs(), op.getAfterArguments())) {
        if (std::get<1>(pair).getDefiningOp() == ifOp) {

          Value thenYielded, elseYielded;
          for (auto p :
               llvm::zip(ifOp.thenYield().getResults(), ifOp.getResults(),
                         ifOp.elseYield().getResults())) {
            if (std::get<1>(pair) == std::get<1>(p)) {
              thenYielded = std::get<0>(p);
              elseYielded = std::get<2>(p);
              break;
            }
          }
          assert(thenYielded);
          assert(elseYielded);

          // If one of the if results is returned, only handle the case
          // where the value yielded is a block argument
          // %out-i:pair<0> = scf.while (... i:%blockArg=... ) {
          //   %z:j = scf.if (%c) {
          //      ...
          //   } else {
          //      yield ... j:%blockArg
          //   }
          //   condition %c ... i:pair<1>=%z:j
          // } loop ( ... i:) {
          //    yield   i:pair<2>
          // }
          //
          // The result need not sit at the block argument's position: the
          // condition may forward other values in front of it, as for a
          // rotated variable (`B = A; A = t;`), where the condition passes
          // the old B, the old A and the if results. What the rewrite needs
          // is that the after argument feeds nothing but the yield into
          // %blockArg (it is replaced there by the then value and receives
          // the old value afterwards), and that the then value is visible in
          // the after region (defined in the then region, which moves there,
          // or outside the loop).
          if (!std::get<0>(pair).use_empty()) {
            if (auto blockArg = dyn_cast<BlockArgument>(elseYielded))
              if (blockArg.getOwner() == &op.getBefore().front()) {
                Value afterArg = std::get<2>(pair);
                bool thenValueVisible =
                    ifOp.getThenRegion().isAncestor(
                        thenYielded.getParentRegion()) ||
                    !op->isAncestor(thenYielded.getParentRegion()
                                        ->getParentOp());
                if (afterYield.getResults()[blockArg.getArgNumber()] ==
                        afterArg &&
                    afterArg.hasOneUse() && thenValueVisible) {
                  prevResults.push_back(std::get<0>(pair));
                  condArgs.push_back(blockArg);
                  afterYieldRewrites.emplace_back(blockArg.getArgNumber(),
                                                  thenYielded);
                  continue;
                }
                // An exit-only value: the loop body never reads the after
                // argument (the variable is carried through other
                // positions, its then value is typically undef), so the
                // result is the block argument at the failing check and the
                // body needs no rewrite.
                if (afterArg.use_empty()) {
                  prevResults.push_back(std::get<0>(pair));
                  condArgs.push_back(blockArg);
                  continue;
                }
              }
            return failure();
          }
          // If the value yielded from then then is defined in the while before
          // but not being moved down with the if, don't change anything.
          if (!ifOp.getThenRegion().isAncestor(thenYielded.getParentRegion()) &&
              op.getBefore().isAncestor(thenYielded.getParentRegion())) {
            prevResults.push_back(std::get<0>(pair));
            condArgs.push_back(thenYielded);
          } else {
            // Otherwise, mark the corresponding after argument to be replaced
            // with the value yielded in the if statement.
            m.emplace_back(std::get<2>(pair), thenYielded);
          }
        } else {
          assert(prevResults.size() == condArgs.size());
          prevResults.push_back(std::get<0>(pair));
          condArgs.push_back(std::get<1>(pair));
        }
      }

      SmallVector<Value> yieldArgs = afterYield.getResults();
      for (auto pair : afterYieldRewrites) {
        yieldArgs[pair.first] = pair.second;
      }

      rewriter.updateRootInPlace(afterYield, [&] {
        afterYield.getResultsMutable().assign(yieldArgs);
      });
      Block *afterB = &op.getAfter().front();

      {
        llvm::SetVector<Value> sv;
        findValuesUsedBelow(ifOp, sv);

        for (auto v : sv) {
          condArgs.push_back(v);
          auto arg = afterB->addArgument(v.getType(), ifOp->getLoc());
          for (OpOperand &use : llvm::make_early_inc_range(v.getUses())) {
            if (ifOp->isAncestor(use.getOwner()) ||
                use.getOwner() == afterYield)
              rewriter.updateRootInPlace(use.getOwner(),
                                         [&]() { use.set(arg); });
          }
        }
      }

      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
                                               condArgs);

      BitVector toErase(afterB->getNumArguments());
      for (int i = m.size() - 1; i >= 0; i--) {
        assert(m[i].first.getType() == m[i].second.getType());
        m[i].first.replaceAllUsesWith(m[i].second);
        toErase[m[i].first.getArgNumber()] = true;
      }
      afterB->eraseArguments(toErase);

      rewriter.eraseOp(ifOp.thenYield());
      Block *thenB = ifOp.thenBlock();
      afterB->getOperations().splice(afterB->getOperations().begin(),
                                     thenB->getOperations());

      rewriter.eraseOp(ifOp);

      SmallVector<Type, 4> resultTypes;
      for (auto v : condArgs) {
        resultTypes.push_back(v.getType());
      }

      rewriter.setInsertionPoint(op);
      auto nop =
          rewriter.create<WhileOp>(op.getLoc(), resultTypes, op.getInits());
      nop.getBefore().takeBody(op.getBefore());
      nop.getAfter().takeBody(op.getAfter());

      rewriter.updateRootInPlace(op, [&] {
        for (auto pair : llvm::enumerate(prevResults)) {
          pair.value().replaceAllUsesWith(nop.getResult(pair.index()));
        }
      });

      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct MoveWhileInvariantIfResult : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument, 2> origAfterArgs(op.getAfterArguments().begin(),
                                                op.getAfterArguments().end());
    bool changed = false;
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    assert(origAfterArgs.size() == op.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());

    for (auto pair :
         llvm::zip(op.getResults(), term.getArgs(), origAfterArgs)) {
      if (!std::get<0>(pair).use_empty()) {
        if (auto ifOp = std::get<1>(pair).getDefiningOp<scf::IfOp>()) {
          if (ifOp.getCondition() == term.getCondition()) {
            auto idx = std::get<1>(pair).cast<OpResult>().getResultNumber();
            Value returnWith = ifOp.elseYield().getResults()[idx];
            if (!op.getBefore().isAncestor(returnWith.getParentRegion())) {
              rewriter.updateRootInPlace(op, [&] {
                std::get<0>(pair).replaceAllUsesWith(returnWith);
              });
              changed = true;
            }
          }
        } else if (auto selOp =
                       std::get<1>(pair).getDefiningOp<arith::SelectOp>()) {
          if (selOp.getCondition() == term.getCondition()) {
            Value returnWith = selOp.getFalseValue();
            if (!op.getBefore().isAncestor(returnWith.getParentRegion())) {
              rewriter.updateRootInPlace(op, [&] {
                std::get<0>(pair).replaceAllUsesWith(returnWith);
              });
              changed = true;
            }
          }
        }
      }
    }

    return success(changed);
  }
};

struct WhileLogicalNegation : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());

    SmallPtrSet<Value, 1> condOps;
    SmallVector<Value> todo = {term.getCondition()};
    while (todo.size()) {
      Value val = todo.back();
      todo.pop_back();
      condOps.insert(val);
      if (auto ao = val.getDefiningOp<AndIOp>()) {
        todo.push_back(ao.getLhs());
        todo.push_back(ao.getRhs());
      }
    }

    for (auto pair :
         llvm::zip(op.getResults(), term.getArgs(), op.getAfterArguments())) {
      auto termArg = std::get<1>(pair);
      bool afterValue;
      if (condOps.count(termArg)) {
        afterValue = true;
      } else {
        bool found = false;
        if (auto termCmp = termArg.getDefiningOp<arith::CmpIOp>()) {
          for (auto cond : condOps) {
            if (auto condCmp = cond.getDefiningOp<CmpIOp>()) {
              if (termCmp.getLhs() == condCmp.getLhs() &&
                  termCmp.getRhs() == condCmp.getRhs()) {
                // TODO generalize to logical negation of
                if (condCmp.getPredicate() == CmpIPredicate::slt &&
                    termCmp.getPredicate() == CmpIPredicate::sge) {
                  found = true;
                  afterValue = false;
                  break;
                }
              }
            }
          }
        }
        if (!found)
          continue;
      }

      if (!std::get<0>(pair).use_empty()) {
        rewriter.updateRootInPlace(op, [&] {
          rewriter.setInsertionPoint(op);
          auto truev =
              rewriter.create<ConstantIntOp>(op.getLoc(), !afterValue, 1);
          std::get<0>(pair).replaceAllUsesWith(truev);
        });
        changed = true;
      }
      if (!std::get<2>(pair).use_empty()) {
        rewriter.updateRootInPlace(op, [&] {
          rewriter.setInsertionPointToStart(&op.getAfter().front());
          auto truev =
              rewriter.create<ConstantIntOp>(op.getLoc(), afterValue, 1);
          std::get<2>(pair).replaceAllUsesWith(truev);
        });
        changed = true;
      }
    }

    return success(changed);
  }
};

/// TODO move the addi down and repalce below with a subi
/// A while loop whose condition is a conjunction of two signed upper bounds
/// on its counter, `i < B1 && i + a < B2` (either conjunct may be `i`,
/// `i + a` or `a + i` against a value defined before the loop), tests the one
/// bound `i < min(B1, B2 - a)`: the loop is then counted and MoveWhileToFor
/// raises it. Vitis treats it so (VITIS_NOTES N153, kernels w9r21p_d1 / d3:
/// `for (u = 0; u < 32 && k + u < B; ++u)` is unrolled and its array
/// partitioned like the plain bound; class_ece8893 pranay_kamineny lab2).
/// The two tests agree while `B2 - a` does not overflow; the program never
/// computes that difference, so the pattern requires B2 to be a constant and
/// `a` a constant or the counter of an enclosing scf.for with constant
/// bounds, all within 2^30 in magnitude (then `i + a < B2` with `i` from 0
/// cannot differ from `i < B2 - a` either). Only `slt` with the counter on
/// the left; the counter must be a before-block argument.
static llvm::cl::opt<bool> WhileConjunctiveBounds(
    "while-conjunctive-bounds", llvm::cl::init(true),
    llvm::cl::desc("Raise `while (i < B1 && i + a < B2)` loops as counted "
                   "loops of min(B1, B2 - a) trips (VITIS_NOTES N153)"));

/// [lo, hi] of a constant or of the counter of an scf.for with constant
/// bounds (through integer casts), each within 2^30 in magnitude.
static bool boundedSmallValue(Value v, int64_t &lo, int64_t &hi,
                              unsigned depth = 0) {
  constexpr int64_t kLimit = int64_t(1) << 30;
  if (depth > 8)
    return false;
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c))) {
    if (c.getSignificantBits() > 32)
      return false;
    lo = hi = c.getSExtValue();
    return lo > -kLimit && lo < kLimit;
  }
  if (Operation *def = v.getDefiningOp()) {
    if (isa<IndexCastOp, ExtSIOp, TruncIOp>(def))
      return boundedSmallValue(def->getOperand(0), lo, hi, depth + 1);
    return false;
  }
  auto arg = cast<BlockArgument>(v);
  auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!loop || arg != loop.getInductionVar())
    return false;
  int64_t l, u, l2, u2;
  if (!boundedSmallValue(loop.getLowerBound(), l, l2, depth + 1) ||
      !boundedSmallValue(loop.getUpperBound(), u, u2, depth + 1))
    return false;
  lo = std::min(l, u);
  hi = std::max(l, u);
  return true;
}

struct WhileConjunctiveBound : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    if (!WhileConjunctiveBounds)
      return failure();
    Block &before = op.getBefore().front();
    auto term = cast<scf::ConditionOp>(before.getTerminator());
    auto conj = term.getCondition().getDefiningOp<AndIOp>();
    if (!conj || conj->getBlock() != &before)
      return failure();
    auto outside = [&](Value v) {
      if (auto arg = dyn_cast<BlockArgument>(v))
        return !op->isAncestor(arg.getOwner()->getParentOp());
      return !op->isAncestor(v.getDefiningOp());
    };
    // `cmp` as `iv < bound`, the bound a value defined before the loop or
    // `B - a`, built before the loop.
    BlockArgument iv;
    auto boundOf = [&](Value c, SmallVectorImpl<std::pair<Value, Value>> &out) {
      auto cmp = c.getDefiningOp<CmpIOp>();
      if (!cmp || cmp.getPredicate() != CmpIPredicate::slt ||
          !outside(cmp.getRhs()))
        return false;
      Value lhs = cmp.getLhs(), offset;
      if (auto add = lhs.getDefiningOp<AddIOp>()) {
        if (add->getBlock() != &before)
          return false;
        if (outside(add.getRhs())) {
          lhs = add.getLhs();
          offset = add.getRhs();
        } else if (outside(add.getLhs())) {
          lhs = add.getRhs();
          offset = add.getLhs();
        } else {
          return false;
        }
      }
      auto arg = dyn_cast<BlockArgument>(lhs);
      if (!arg || arg.getOwner() != &before || (iv && arg != iv))
        return false;
      iv = arg;
      out.push_back({cmp.getRhs(), offset});
      return true;
    };
    SmallVector<std::pair<Value, Value>, 2> bounds;
    if (!boundOf(conj.getLhs(), bounds) || !boundOf(conj.getRhs(), bounds))
      return failure();
    if (iv.getType() != bounds[0].first.getType() ||
        iv.getType() != bounds[1].first.getType())
      return failure();
    // No overflow in `B - a` (see the comment above).
    for (auto [b, a] : bounds) {
      if (!a)
        continue;
      int64_t blo, bhi, alo, ahi;
      if (!boundedSmallValue(b, blo, bhi) || blo != bhi ||
          !boundedSmallValue(a, alo, ahi))
        return failure();
    }
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 2> limits;
    for (auto [b, a] : bounds)
      limits.push_back(a ? rewriter.create<SubIOp>(op.getLoc(), b, a)
                               .getResult()
                         : b);
    Value limit = rewriter.create<MinSIOp>(op.getLoc(), limits[0], limits[1]);
    rewriter.setInsertionPoint(conj);
    Value test =
        rewriter.create<CmpIOp>(conj.getLoc(), CmpIPredicate::slt, iv, limit);
    rewriter.replaceOp(conj, test);
    return success();
  }
};

struct WhileCmpOffset : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument, 2> origAfterArgs(op.getAfterArguments().begin(),
                                                op.getAfterArguments().end());
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    assert(origAfterArgs.size() == op.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());

    if (auto condCmp = term.getCondition().getDefiningOp<CmpIOp>()) {
      if (auto addI = condCmp.getLhs().getDefiningOp<AddIOp>()) {
        if (addI.getOperand(1).getDefiningOp() &&
            !op.getBefore().isAncestor(
                addI.getOperand(1).getDefiningOp()->getParentRegion()))
          if (auto blockArg = dyn_cast<BlockArgument>(addI.getOperand(0))) {
            if (blockArg.getOwner() == &op.getBefore().front()) {
              auto rng = llvm::make_early_inc_range(blockArg.getUses());

              {
                rewriter.setInsertionPoint(op);
                SmallVector<Value> oldInits = op.getInits();
                oldInits[blockArg.getArgNumber()] = rewriter.create<AddIOp>(
                    addI.getLoc(), oldInits[blockArg.getArgNumber()],
                    addI.getOperand(1));
                op.getInitsMutable().assign(oldInits);
                rewriter.updateRootInPlace(
                    addI, [&] { addI.replaceAllUsesWith(blockArg); });
              }

              YieldOp afterYield = cast<YieldOp>(op.getAfter().front().back());
              rewriter.setInsertionPoint(afterYield);
              SmallVector<Value> oldYields = afterYield.getResults();
              oldYields[blockArg.getArgNumber()] = rewriter.create<AddIOp>(
                  addI.getLoc(), oldYields[blockArg.getArgNumber()],
                  addI.getOperand(1));
              rewriter.updateRootInPlace(afterYield, [&] {
                afterYield.getResultsMutable().assign(oldYields);
              });

              rewriter.setInsertionPointToStart(&op.getBefore().front());
              auto sub = rewriter.create<SubIOp>(addI.getLoc(), blockArg,
                                                 addI.getOperand(1));
              for (OpOperand &use : rng) {
                rewriter.updateRootInPlace(use.getOwner(),
                                           [&]() { use.set(sub); });
              }
              rewriter.eraseOp(addI);
              return success();
            }
          }
      }
    }

    return failure();
  }
};

/// Given a while loop which yields a select whose condition is
/// the same as the condition, remove the select.
struct RemoveWhileSelect : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp term =
        cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());

    SmallVector<BlockArgument, 2> origAfterArgs(
        loop.getAfterArguments().begin(), loop.getAfterArguments().end());
    SmallVector<unsigned> newResults;
    SmallVector<unsigned> newAfter;
    SmallVector<Value> newYields;
    bool changed = false;
    for (auto pair :
         llvm::zip(loop.getResults(), term.getArgs(), origAfterArgs)) {
      auto selOp = std::get<1>(pair).getDefiningOp<arith::SelectOp>();
      if (!selOp || selOp.getCondition() != term.getCondition()) {
        newResults.push_back(newYields.size());
        newAfter.push_back(newYields.size());
        newYields.push_back(std::get<1>(pair));
        continue;
      }
      newResults.push_back(newYields.size());
      newYields.push_back(selOp.getFalseValue());
      newAfter.push_back(newYields.size());
      newYields.push_back(selOp.getTrueValue());
      changed = true;
    }
    if (!changed)
      return failure();

    SmallVector<Type, 4> resultTypes;
    for (auto v : newYields) {
      resultTypes.push_back(v.getType());
    }
    auto nop =
        rewriter.create<WhileOp>(loop.getLoc(), resultTypes, loop.getInits());

    nop.getBefore().takeBody(loop.getBefore());

    auto *after = rewriter.createBlock(&nop.getAfter());
    for (auto y : newYields)
      after->addArgument(y.getType(), loop.getLoc());

    SmallVector<Value> replacedArgs;
    for (auto idx : newAfter)
      replacedArgs.push_back(after->getArgument(idx));
    rewriter.mergeBlocks(&loop.getAfter().front(), after, replacedArgs);

    SmallVector<Value> replacedReturns;
    for (auto idx : newResults)
      replacedReturns.push_back(nop.getResult(idx));
    rewriter.replaceOp(loop, replacedReturns);
    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
                                             newYields);
    return success();
  }
};

struct MoveWhileDown3 : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    SmallVector<unsigned int, 2> toErase;
    SmallVector<Value, 2> newOps;
    SmallVector<Value, 2> condOps;
    SmallVector<BlockArgument, 2> origAfterArgs(op.getAfterArguments().begin(),
                                                op.getAfterArguments().end());
    SmallVector<Value, 2> returns;
    assert(origAfterArgs.size() == op.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());
    for (auto pair :
         llvm::zip(op.getResults(), term.getArgs(), origAfterArgs)) {
      if (std::get<0>(pair).use_empty()) {
        if (std::get<2>(pair).use_empty()) {
          toErase.push_back(std::get<2>(pair).getArgNumber());
          continue;
        }
        // TODO generalize to any non memory effecting op
        if (auto idx =
                std::get<1>(pair).getDefiningOp<MemoryEffectOpInterface>()) {
          if (idx.hasNoEffect() &&
              !llvm::is_contained(newOps, std::get<1>(pair))) {
            Operation *cloned = std::get<1>(pair).getDefiningOp();
            if (!std::get<1>(pair).hasOneUse()) {
              cloned = std::get<1>(pair).getDefiningOp()->clone();
              op.getAfter().front().push_front(cloned);
            } else {
              cloned->moveBefore(&op.getAfter().front().front());
            }
            rewriter.updateRootInPlace(std::get<1>(pair).getDefiningOp(), [&] {
              std::get<2>(pair).replaceAllUsesWith(cloned->getResult(0));
            });
            toErase.push_back(std::get<2>(pair).getArgNumber());
            for (auto &o :
                 llvm::make_early_inc_range(cloned->getOpOperands())) {
              {
                newOps.push_back(o.get());
                o.set(op.getAfter().front().addArgument(o.get().getType(),
                                                        o.get().getLoc()));
              }
            }
            continue;
          }
        }
      }
      condOps.push_back(std::get<1>(pair));
      returns.push_back(std::get<0>(pair));
    }
    if (toErase.size() == 0)
      return failure();

    condOps.append(newOps.begin(), newOps.end());

    BitVector toEraseVec(op.getAfter().front().getNumArguments());
    for (auto argNum : toErase)
      toEraseVec[argNum] = true;
    rewriter.updateRootInPlace(
        term, [&] { op.getAfter().front().eraseArguments(toEraseVec); });
    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
                                             condOps);

    rewriter.setInsertionPoint(op);
    SmallVector<Type, 4> resultTypes;
    for (auto v : condOps) {
      resultTypes.push_back(v.getType());
    }
    auto nop =
        rewriter.create<WhileOp>(op.getLoc(), resultTypes, op.getInits());

    nop.getBefore().takeBody(op.getBefore());
    nop.getAfter().takeBody(op.getAfter());

    rewriter.updateRootInPlace(op, [&] {
      for (auto pair : llvm::enumerate(returns)) {
        pair.value().replaceAllUsesWith(nop.getResult(pair.index()));
      }
    });

    assert(resultTypes.size() == nop.getAfter().front().getNumArguments());
    assert(resultTypes.size() == condOps.size());

    rewriter.eraseOp(op);
    return success();
  }
};

// Rewritten from LoopInvariantCodeMotion.cpp
struct WhileLICM : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;
  static bool canBeHoisted(Operation *op,
                           function_ref<bool(Value)> definedOutside,
                           bool isSpeculatable, WhileOp whileOp) {
    // TODO consider requirement of isSpeculatable

    // Check that dependencies are defined outside of loop.
    if (!llvm::all_of(op->getOperands(), definedOutside))
      return false;
    // Check whether this op is side-effect free. If we already know that there
    // can be no side-effects because the surrounding op has claimed so, we can
    // (and have to) skip this step.
    if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!memInterface.hasNoEffect()) {
        if (isReadOnly(op) && !isSpeculatable) {

          SmallVector<MemoryEffects::EffectInstance> whileEffects;
          collectEffects(whileOp, whileEffects, /*ignoreBarriers*/ false);

          SmallVector<MemoryEffects::EffectInstance> opEffects;
          collectEffects(op, opEffects, /*ignoreBarriers*/ false);

          bool conflict = false;
          for (auto before : opEffects)
            for (auto after : whileEffects) {
              if (mayAlias(before, after)) {
                // Read, read is okay
                if (isa<MemoryEffects::Read>(before.getEffect()) &&
                    isa<MemoryEffects::Read>(after.getEffect())) {
                  continue;
                }

                // Write, write is not okay because may be different offsets and
                // the later must subsume other conflicts are invalid.
                conflict = true;
                break;
              }
            }
          if (conflict)
            return false;
        } else
          return false;
      }
      // If the operation doesn't have side effects and it doesn't recursively
      // have side effects, it can always be hoisted.
      if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        return true;

      // Otherwise, if the operation doesn't provide the memory effect interface
      // and it doesn't have recursive side effects we treat it conservatively
      // as side-effecting.
    } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      return false;
    }

    // Recurse into the regions for this op and check whether the contained ops
    // can be hoisted.
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block.without_terminator())
          if (!canBeHoisted(&innerOp, definedOutside, isSpeculatable, whileOp))
            return false;
      }
    }
    return true;
  }

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    // We use two collections here as we need to preserve the order for
    // insertion and this is easiest.
    SmallPtrSet<Operation *, 8> willBeMovedSet;
    SmallVector<Operation *, 8> opsToMove;

    // Helper to check whether an operation is loop invariant wrt. SSA
    // properties.
    auto isDefinedOutsideOfBody = [&](Value value) {
      auto *definingOp = value.getDefiningOp();
      if (!definingOp) {
        if (auto ba = dyn_cast<BlockArgument>(value))
          definingOp = ba.getOwner()->getParentOp();
        assert(definingOp);
      }
      if (willBeMovedSet.count(definingOp))
        return true;
      return op != definingOp && !op->isAncestor(definingOp);
    };

    // Do not use walk here, as we do not want to go into nested regions and
    // hoist operations from there. These regions might have semantics unknown
    // to this rewriting. If the nested regions are loops, they will have been
    // processed.
    for (auto &block : op.getBefore()) {
      for (auto &iop : block.without_terminator()) {
        bool legal = canBeHoisted(&iop, isDefinedOutsideOfBody, false, op);
        if (legal) {
          opsToMove.push_back(&iop);
          willBeMovedSet.insert(&iop);
        }
      }
    }

    for (auto &block : op.getAfter()) {
      for (auto &iop : block.without_terminator()) {
        bool legal = canBeHoisted(&iop, isDefinedOutsideOfBody, true, op);
        if (legal) {
          opsToMove.push_back(&iop);
          willBeMovedSet.insert(&iop);
        }
      }
    }

    for (auto *moveOp : opsToMove)
      rewriter.updateRootInPlace(moveOp, [&] { moveOp->moveBefore(op); });

    return success(opsToMove.size() > 0);
  }
};

struct RemoveUnusedCondVar : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    SmallVector<Value, 4> conds;
    BitVector eraseArgs;
    SmallVector<unsigned, 4> keepArgs;
    SmallVector<Type, 4> tys;
    unsigned i = 0;
    std::map<void *, unsigned> valueOffsets;
    std::map<unsigned, unsigned> resultOffsets;
    SmallVector<Value, 4> resultArgs;
    for (auto pair :
         llvm::zip(term.getArgs(), op.getAfter().front().getArguments(),
                   op.getResults())) {
      auto arg = std::get<0>(pair);
      auto afarg = std::get<1>(pair);
      auto res = std::get<2>(pair);
      if (!op.getBefore().isAncestor(arg.getParentRegion())) {
        res.replaceAllUsesWith(arg);
        afarg.replaceAllUsesWith(arg);
      }
      if (afarg.use_empty() && res.use_empty()) {
        eraseArgs.push_back(true);
      } else if (valueOffsets.find(arg.getAsOpaquePointer()) !=
                 valueOffsets.end()) {
        resultOffsets[i] = valueOffsets[arg.getAsOpaquePointer()];
        afarg.replaceAllUsesWith(
            resultArgs[valueOffsets[arg.getAsOpaquePointer()]]);
        eraseArgs.push_back(true);
      } else {
        valueOffsets[arg.getAsOpaquePointer()] = keepArgs.size();
        resultOffsets[i] = keepArgs.size();
        resultArgs.push_back(afarg);
        conds.push_back(arg);
        keepArgs.push_back((unsigned)i);
        eraseArgs.push_back(false);
        tys.push_back(arg.getType());
      }
      i++;
    }
    assert(i == op.getAfter().front().getArguments().size());

    if (eraseArgs.any()) {

      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                    conds);

      rewriter.setInsertionPoint(op);
      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.getInits());

      op2.getBefore().takeBody(op.getBefore());
      op2.getAfter().takeBody(op.getAfter());
      for (auto pair : resultOffsets) {
        op.getResult(pair.first).replaceAllUsesWith(op2.getResult(pair.second));
      }
      rewriter.eraseOp(op);
      op2.getAfter().front().eraseArguments(eraseArgs);
      return success();
    }
    return failure();
  }
};

struct MoveSideEffectFreeWhile : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    SmallVector<Value, 4> conds(term.getArgs().begin(), term.getArgs().end());
    bool changed = false;
    unsigned i = 0;
    for (auto arg : term.getArgs()) {
      // A cast the condition passes to the body (an index cast of a counter,
      // or the extension a narrow counter is compared through, when the body
      // shares it) is recomputed in the body from the cast's input, which is
      // passed instead. The before-region op moves when the condition was its
      // only use and is cloned when the comparison still needs it there.
      Operation *IC = arg.getDefiningOp();
      if (IC && isa<IndexCastOp, IndexCastUIOp, ExtSIOp, ExtUIOp>(IC) &&
          op.getResult(i).use_empty() &&
          (arg.hasOneUse() || IC->getParentRegion() == &op.getBefore())) {
        Block &after = op.getAfter().front();
        auto rep = after.addArgument(IC->getOperand(0).getType(),
                                     IC->getOperand(0).getLoc());
        conds.push_back(IC->getOperand(0));
        Operation *bodyCast = IC;
        if (arg.hasOneUse()) {
          IC->moveBefore(&after, after.begin());
        } else {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&after);
          bodyCast = rewriter.clone(*IC);
        }
        bodyCast->setOperand(0, rep);
        after.getArgument(i).replaceAllUsesWith(bodyCast->getResult(0));
        changed = true;
      }
      i++;
    }
    if (changed) {
      SmallVector<Type, 4> tys;
      for (auto arg : conds) {
        tys.push_back(arg.getType());
      }
      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.getInits());
      op2.getBefore().takeBody(op.getBefore());
      op2.getAfter().takeBody(op.getAfter());
      unsigned j = 0;
      for (auto a : op.getResults()) {
        a.replaceAllUsesWith(op2.getResult(j));
        j++;
      }
      rewriter.eraseOp(op);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                    conds);
      return success();
    }
    return failure();
  }
};

struct SubToAdd : public OpRewritePattern<SubIOp> {
  using OpRewritePattern<SubIOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SubIOp op,
                                PatternRewriter &rewriter) const override {
    if (auto cop = op.getOperand(1).getDefiningOp<ConstantIntOp>()) {
      // Negate in the constant's own width (APInt): going through the
      // int64_t value() zero-extended -C into integers wider than 64 bits
      // (x - 1 : i128 became x + (2^64 - 1)). Same constants as before for
      // widths <= 64.
      APInt c = cop.getValue().cast<IntegerAttr>().getValue();
      // x - C and x + (-C) have the same exact result when -C does not
      // overflow: a non-wrapping subtraction (polygeist.nsw) stays one.
      bool nsw = op->hasAttr(kNoSignedWrapAttrName) && !c.isMinSignedValue();
      auto add = rewriter.replaceOpWithNewOp<AddIOp>(
          op, op.getOperand(0),
          rewriter.create<arith::ConstantOp>(
              cop.getLoc(), cop.getType(),
              rewriter.getIntegerAttr(cop.getType(), -c)));
      if (nsw)
        add->setAttr(kNoSignedWrapAttrName, rewriter.getUnitAttr());
      return success();
    }
    return failure();
  }
};

// The arith canonicalizer folds a chain of two add / sub with constant
// operands into one (`(x + 1) + 2` into `x + 3`, `3 - (x + 1)` into `2 - x`)
// and two mul by constants into one, through new ops without the
// polygeist.nsw mark of the originals (Passes.h). When both ops of a chain
// are marked, their exact results are in range, so the folded op, which
// computes the same value, does not wrap either. These patterns fold such a
// chain first (benefit 10: a DRR pattern's benefit is the number of ops it
// matches) into the form the arith patterns produce and keep
// the mark. The folded constant must fit its width (`x + 2^30 + 2^30`
// does not), or the folded op would compute a different value.

/// A marked add / sub `op` with one constant operand, read as
/// `sign * x + k`: `x + c` and `c + x` are x + c, `x - c` is x + (-c), `c - x`
/// is -x + c.
struct AddSubConst {
  Value x;
  int sign;
  APInt k;
};

static std::optional<AddSubConst> matchNoSignedWrapAddSubConst(Operation *op) {
  if (!isa<AddIOp, SubIOp>(op) ||
      !op->hasAttr(polygeist::kNoSignedWrapAttrName))
    return std::nullopt;
  APInt c;
  if (matchPattern(op->getOperand(1), m_ConstantInt(&c))) {
    if (isa<AddIOp>(op))
      return AddSubConst{op->getOperand(0), 1, c};
    if (c.isMinSignedValue())
      return std::nullopt;
    return AddSubConst{op->getOperand(0), 1, -c};
  }
  if (matchPattern(op->getOperand(0), m_ConstantInt(&c)))
    return AddSubConst{op->getOperand(1), isa<AddIOp>(op) ? 1 : -1, c};
  return std::nullopt;
}

/// Folds the chain `op` = outer(inner(x)) of two marked add / sub with
/// constant operands into `x + k` or `k - x`, marked.
static LogicalResult foldNoSignedWrapAddSubChain(Operation *op,
                                                 PatternRewriter &rewriter) {
  auto outer = matchNoSignedWrapAddSubConst(op);
  if (!outer)
    return failure();
  Operation *innerOp = outer->x.getDefiningOp();
  if (!innerOp)
    return failure();
  auto inner = matchNoSignedWrapAddSubConst(innerOp);
  if (!inner)
    return failure();
  // outer->sign * (inner->sign * x + inner->k) + outer->k
  if (outer->sign < 0 && inner->k.isMinSignedValue())
    return failure();
  bool overflow = false;
  APInt k =
      (outer->sign < 0 ? -inner->k : inner->k).sadd_ov(outer->k, overflow);
  if (overflow)
    return failure();
  Type t = op->getResult(0).getType();
  Value cst = rewriter.create<arith::ConstantOp>(op->getLoc(), t,
                                                 rewriter.getIntegerAttr(t, k));
  Operation *folded =
      outer->sign * inner->sign > 0
          ? rewriter.create<AddIOp>(op->getLoc(), inner->x, cst).getOperation()
          : rewriter.create<SubIOp>(op->getLoc(), cst, inner->x).getOperation();
  folded->setAttr(polygeist::kNoSignedWrapAttrName, rewriter.getUnitAttr());
  rewriter.replaceOp(op, folded->getResults());
  return success();
}

template <typename OpTy>
struct NoSignedWrapAddSubChain : public OpRewritePattern<OpTy> {
  NoSignedWrapAddSubChain(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx, /*benefit=*/10) {}
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    return foldNoSignedWrapAddSubChain(op, rewriter);
  }
};

/// `(x * c0) * c1`, both marked, into `x * (c0 * c1)`, marked, when the
/// product fits.
struct NoSignedWrapMulChain : public OpRewritePattern<MulIOp> {
  NoSignedWrapMulChain(MLIRContext *ctx)
      : OpRewritePattern<MulIOp>(ctx, /*benefit=*/10) {}
  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    APInt c0, c1;
    if (!op->hasAttr(polygeist::kNoSignedWrapAttrName) ||
        !matchPattern(op.getRhs(), m_ConstantInt(&c1)))
      return failure();
    auto inner = op.getLhs().getDefiningOp<MulIOp>();
    if (!inner || !inner->hasAttr(polygeist::kNoSignedWrapAttrName) ||
        !matchPattern(inner.getRhs(), m_ConstantInt(&c0)))
      return failure();
    bool overflow = false;
    APInt k = c0.smul_ov(c1, overflow);
    if (overflow)
      return failure();
    Type t = op.getType();
    Value cst = rewriter.create<arith::ConstantOp>(
        op.getLoc(), t, rewriter.getIntegerAttr(t, k));
    auto folded = rewriter.replaceOpWithNewOp<MulIOp>(op, inner.getLhs(), cst);
    folded->setAttr(polygeist::kNoSignedWrapAttrName, rewriter.getUnitAttr());
    return success();
  }
};

struct ReturnSq : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern<ReturnOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    SmallVector<Operation *> toErase;
    for (auto iter = op->getBlock()->rbegin();
         iter != op->getBlock()->rend() && &*iter != op; iter++) {
      changed = true;
      toErase.push_back(&*iter);
    }
    for (auto *op : toErase) {
      rewriter.eraseOp(op);
    }
    return success(changed);
  }
};

// From SCF.cpp
// Pattern to remove unused IfOp results.
struct RemoveUnusedResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  void transferBody(Block *source, Block *dest, ArrayRef<OpResult> usedResults,
                    PatternRewriter &rewriter) const {
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest);
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    SmallVector<Value, 4> usedOperands;
    llvm::transform(usedResults, std::back_inserter(usedOperands),
                    [&](OpResult result) {
                      return yieldOp.getOperand(result.getResultNumber());
                    });
    rewriter.updateRootInPlace(yieldOp,
                               [&]() { yieldOp->setOperands(usedOperands); });
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Compute the list of used results.
    SmallVector<OpResult, 4> usedResults;
    llvm::copy_if(op.getResults(), std::back_inserter(usedResults),
                  [](OpResult result) { return !result.use_empty(); });

    // Replace the operation if only a subset of its results have uses.
    if (usedResults.size() == op.getNumResults())
      return failure();

    // Compute the result types of the replacement operation.
    SmallVector<Type, 4> newTypes;
    llvm::transform(usedResults, std::back_inserter(newTypes),
                    [](OpResult result) { return result.getType(); });

    // Create a replacement operation with empty then and else regions.
    auto emptyBuilder = [](OpBuilder &, Location) {};
    auto newOp = rewriter.create<IfOp>(op.getLoc(), newTypes, op.getCondition(),
                                       emptyBuilder, emptyBuilder);

    // Move the bodies and replace the terminators (note there is a then and
    // an else region since the operation returns results).
    transferBody(op.getBody(0), newOp.getBody(0), usedResults, rewriter);
    transferBody(op.getBody(1), newOp.getBody(1), usedResults, rewriter);

    // Replace the operation by the new one.
    SmallVector<Value, 4> repResults(op.getNumResults());
    for (const auto &en : llvm::enumerate(usedResults))
      repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
    rewriter.replaceOp(op, repResults);
    return success();
  }
};

// If and and with something is preventing creating a for
// move the and into the after body guarded by an if
struct WhileShiftToInduction : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    auto condOp = loop.getConditionOp();

    if (!llvm::hasNItems(loop.getBefore().back(), 2))
      return failure();

    auto cmpIOp = condOp.getCondition().getDefiningOp<CmpIOp>();
    if (!cmpIOp) {
      return failure();
    }

    if (cmpIOp.getPredicate() != CmpIPredicate::ugt)
      return failure();

    if (!matchPattern(cmpIOp.getRhs(), m_Zero()))
      return failure();

    auto indVar = dyn_cast<BlockArgument>(cmpIOp.getLhs());
    if (!indVar)
      return failure();

    if (indVar.getOwner() != &loop.getBefore().front())
      return failure();

    auto endYield = cast<YieldOp>(loop.getAfter().back().getTerminator());

    // Check that the block argument is actually an induction var:
    //   Namely, its next value adds to the previous with an invariant step.
    auto shiftOp =
        endYield.getResults()[indVar.getArgNumber()].getDefiningOp<ShRUIOp>();
    if (!shiftOp)
      return failure();

    if (!matchPattern(shiftOp.getRhs(), m_One()))
      return failure();

    auto prevIndVar = dyn_cast<BlockArgument>(shiftOp.getLhs());
    if (!prevIndVar)
      return failure();

    if (prevIndVar.getOwner() != &loop.getAfter().front())
      return failure();

    if (condOp.getOperand(1 + prevIndVar.getArgNumber()) != indVar)
      return failure();

    auto startingV = loop.getInits()[indVar.getArgNumber()];

    Value lz =
        rewriter.create<math::CountLeadingZerosOp>(loop.getLoc(), startingV);
    if (!lz.getType().isIndex())
      lz = rewriter.create<IndexCastOp>(loop.getLoc(), rewriter.getIndexType(),
                                        lz);

    auto len = rewriter.create<SubIOp>(
        loop.getLoc(),
        rewriter.create<ConstantIndexOp>(
            loop.getLoc(), indVar.getType().getIntOrFloatBitWidth()),
        lz);

    SmallVector<Value> newInits(loop.getInits());
    newInits[indVar.getArgNumber()] =
        rewriter.create<ConstantIndexOp>(loop.getLoc(), 0);
    SmallVector<Type> postTys(loop.getResultTypes());
    postTys.push_back(rewriter.getIndexType());

    auto newWhile = rewriter.create<WhileOp>(loop.getLoc(), postTys, newInits);
    rewriter.createBlock(&newWhile.getBefore());

    IRMapping map;
    Value newIndVar;
    for (auto a : loop.getBefore().front().getArguments()) {
      auto arg = newWhile.getBefore().addArgument(
          a == indVar ? rewriter.getIndexType() : a.getType(), a.getLoc());
      if (a != indVar)
        map.map(a, arg);
      else
        newIndVar = arg;
    }

    rewriter.setInsertionPointToEnd(&newWhile.getBefore().front());
    Value newCmp = rewriter.create<CmpIOp>(cmpIOp.getLoc(), CmpIPredicate::ult,
                                           newIndVar, len);
    map.map(cmpIOp, newCmp);

    Value newIndVarTyped = newIndVar;
    if (newIndVarTyped.getType() != indVar.getType())
      newIndVarTyped = rewriter.create<arith::IndexCastOp>(
          shiftOp.getLoc(), indVar.getType(), newIndVar);
    map.map(indVar, rewriter.create<ShRUIOp>(shiftOp.getLoc(), startingV,
                                             newIndVarTyped));
    SmallVector<Value> remapped;
    for (auto o : condOp.getArgs())
      remapped.push_back(map.lookup(o));
    remapped.push_back(newIndVar);
    rewriter.create<ConditionOp>(condOp.getLoc(), newCmp, remapped);

    newWhile.getAfter().takeBody(loop.getAfter());

    auto newPostInd = newWhile.getAfter().front().addArgument(
        rewriter.getIndexType(), loop.getLoc());
    auto yieldOp =
        cast<scf::YieldOp>(newWhile.getAfter().front().getTerminator());
    SmallVector<Value> yields(yieldOp.getOperands());
    rewriter.setInsertionPointToEnd(&newWhile.getAfter().front());
    yields[indVar.getArgNumber()] = rewriter.create<AddIOp>(
        loop.getLoc(), newPostInd,
        rewriter.create<arith::ConstantIndexOp>(loop.getLoc(), 1));
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, yields);

    SmallVector<Value> res(newWhile.getResults());
    res.pop_back();
    rewriter.replaceOp(loop, res);
    return success();
  }
};

void CanonicalizeFor::runOnOperation() {
  mlir::RewritePatternSet rpl(getOperation()->getContext());
  rpl.add<PropagateInLoopBody, ForOpInductionReplacement, RemoveUnusedArgs,
          MoveWhileToFor, RemoveWhileSelect, WhileConjunctiveBound,

          MoveWhileDown, MoveWhileDown2,

          ReplaceRedundantArgs,

          WhileShiftToInduction,

          ForBreakAddUpgrade, RemoveUnusedResults,

          MoveWhileAndDown, MoveWhileDown3, MoveWhileInvariantIfResult,
          WhileLogicalNegation, SubToAdd, WhileCmpOffset, WhileLICM,
          RemoveUnusedCondVar, ReturnSq, MoveSideEffectFreeWhile>(
      getOperation()->getContext());
  GreedyRewriteConfig config;
  config.maxIterations = 247;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
}

std::unique_ptr<Pass> mlir::polygeist::createCanonicalizeForPass() {
  return std::make_unique<CanonicalizeFor>();
}

void mlir::polygeist::populateNoSignedWrapPatterns(
    RewritePatternSet &patterns) {
  patterns.add<NoSignedWrapAddSubChain<AddIOp>, NoSignedWrapAddSubChain<SubIOp>,
               NoSignedWrapMulChain>(patterns.getContext());
}

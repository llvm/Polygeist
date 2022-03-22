#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include <deque>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>

#define DEBUG_TYPE "affine-cfg"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

bool isReadOnly(Operation *op);

// isValidSymbol, even if not index
bool isValidSymbolInt(Value value, bool recur = true) {
  // Check that the value is a top level value.
  if (isTopLevelValue(value))
    return true;

  if (auto *defOp = value.getDefiningOp()) {
    Attribute operandCst;
    if (matchPattern(defOp, m_Constant(&operandCst)))
      return true;

    if (recur) {
      if (isa<SelectOp, IndexCastOp, AddIOp, MulIOp, DivSIOp, DivUIOp, RemSIOp,
              RemUIOp, SubIOp, CmpIOp>(defOp))
        if (llvm::all_of(defOp->getOperands(), [&](Value v) {
              bool b = isValidSymbolInt(v, true);
              // if (!b)
              //	LLVM_DEBUG(llvm::dbgs() << "illegal isValidSymbolInt: "
              //<< value << " due to " << v << "\n");
              return b;
            }))
          return true;
    }
    return isValidSymbol(value, getAffineScope(defOp));
  }
  return false;
}

struct AffineApplyNormalizer {
  AffineApplyNormalizer(AffineMap map, ArrayRef<Value> operands);

  /// Returns the AffineMap resulting from normalization.
  AffineMap getAffineMap() { return affineMap; }

  SmallVector<Value, 8> getOperands() {
    SmallVector<Value, 8> res(reorderedDims);
    res.append(concatenatedSymbols.begin(), concatenatedSymbols.end());
    return res;
  }

  unsigned getNumSymbols() { return concatenatedSymbols.size(); }
  unsigned getNumDims() { return reorderedDims.size(); }

private:
  /// Helper function to insert `v` into the coordinate system of the current
  /// AffineApplyNormalizer. Returns the AffineDimExpr with the corresponding
  /// renumbered position.
  AffineDimExpr renumberOneDim(Value v);

  /// Given an `other` normalizer, this rewrites `other.affineMap` in the
  /// coordinate system of the current AffineApplyNormalizer.
  /// Returns the rewritten AffineMap and updates the dims and symbols of
  /// `this`.
  AffineMap renumber(const AffineApplyNormalizer &other);

  /// Maps of Value to position in `affineMap`.
  DenseMap<Value, unsigned> dimValueToPosition;

  /// Ordered dims and symbols matching positional dims and symbols in
  /// `affineMap`.
  SmallVector<Value, 8> reorderedDims;
  SmallVector<Value, 8> concatenatedSymbols;

  /// The number of symbols in concatenated symbols that belong to the original
  /// map as opposed to those concatendated during map composition.
  unsigned numProperSymbols;

  AffineMap affineMap;

  /// Used with RAII to control the depth at which AffineApply are composed
  /// recursively. Only accepts depth 1 for now to allow a behavior where a
  /// newly composed AffineApplyOp does not increase the length of the chain of
  /// AffineApplyOps. Full composition is implemented iteratively on top of
  /// this behavior.
  static unsigned &affineApplyDepth() {
    static thread_local unsigned depth = 0;
    return depth;
  }
  static constexpr unsigned kMaxAffineApplyDepth = 1;

  AffineApplyNormalizer() : numProperSymbols(0) { affineApplyDepth()++; }

public:
  ~AffineApplyNormalizer() { affineApplyDepth()--; }
};

static bool isAffineForArg(Value val) {
  if (!val.isa<BlockArgument>())
    return false;
  Operation *parentOp = val.cast<BlockArgument>().getOwner()->getParentOp();
  return (parentOp && isa<AffineForOp, AffineParallelOp>(parentOp));
}

static bool legalCondition(Value en, bool dim = false) {
  if (en.getDefiningOp<AffineApplyOp>())
    return true;

  if (!dim && !isValidSymbolInt(en, /*recur*/ false)) {
    if (isValidIndex(en) || isValidSymbolInt(en, /*recur*/ true)) {
      return true;
    }
  }
  // if (auto IC = dyn_cast_or_null<IndexCastOp>(en.getDefiningOp())) {
  //	if (!outer || legalCondition(IC.getOperand(), false)) return true;
  //}
  if (!dim)
    if (auto BA = en.dyn_cast<BlockArgument>()) {
      if (isa<AffineForOp, AffineParallelOp>(BA.getOwner()->getParentOp()))
        return true;
    }
  return false;
}

/// The AffineNormalizer composes AffineApplyOp recursively. Its purpose is to
/// keep a correspondence between the mathematical `map` and the `operands` of
/// a given AffineApplyOp. This correspondence is maintained by iterating over
/// the operands and forming an `auxiliaryMap` that can be composed
/// mathematically with `map`. To keep this correspondence in cases where
/// symbols are produced by affine.apply operations, we perform a local rewrite
/// of symbols as dims.
///
/// Rationale for locally rewriting symbols as dims:
/// ================================================
/// The mathematical composition of AffineMap must always concatenate symbols
/// because it does not have enough information to do otherwise. For example,
/// composing `(d0)[s0] -> (d0 + s0)` with itself must produce
/// `(d0)[s0, s1] -> (d0 + s0 + s1)`.
///
/// The result is only equivalent to `(d0)[s0] -> (d0 + 2 * s0)` when
/// applied to the same mlir::Value for both s0 and s1.
/// As a consequence mathematical composition of AffineMap always concatenates
/// symbols.
///
/// When AffineMaps are used in AffineApplyOp however, they may specify
/// composition via symbols, which is ambiguous mathematically. This corner case
/// is handled by locally rewriting such symbols that come from AffineApplyOp
/// into dims and composing through dims.
/// TODO: Composition via symbols comes at a significant code
/// complexity. Alternatively we should investigate whether we want to
/// explicitly disallow symbols coming from affine.apply and instead force the
/// user to compose symbols beforehand. The annoyances may be small (i.e. 1 or 2
/// extra API calls for such uses, which haven't popped up until now) and the
/// benefit potentially big: simpler and more maintainable code for a
/// non-trivial, recursive, procedure.
AffineApplyNormalizer::AffineApplyNormalizer(AffineMap map,
                                             ArrayRef<Value> operands)
    : AffineApplyNormalizer() {
  static_assert(kMaxAffineApplyDepth > 0, "kMaxAffineApplyDepth must be > 0");
  assert(map.getNumInputs() == operands.size() &&
         "number of operands does not match the number of map inputs");

  LLVM_DEBUG(map.print(llvm::dbgs() << "\nInput map: "));

  SmallVector<AffineExpr, 8> auxiliaryExprs;
  SmallVector<Value, 8> addedValues;

  unsigned numDimsBeforeRewrite = map.getNumDims();
  llvm::SmallSet<unsigned, 1> symbolsToPromote;

  // 2. Compose AffineApplyOps and dispatch dims or symbols.
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    auto t = operands[i];
    if (!isValidSymbolInt(t, /*recur*/ false)) {
      while (auto idx = t.getDefiningOp<IndexCastOp>()) {
        t = idx.getIn();
      }
    }

    if (!isValidSymbolInt(t, /*recur*/ false) &&
        (t.getDefiningOp<AddIOp>() || t.getDefiningOp<SubIOp>() ||
         t.getDefiningOp<MulIOp>() || t.getDefiningOp<DivSIOp>() ||
         t.getDefiningOp<DivUIOp>() || t.getDefiningOp<RemUIOp>() ||
         t.getDefiningOp<RemSIOp>() || t.getDefiningOp<ConstantIntOp>() ||
         t.getDefiningOp<ConstantIndexOp>())) {

      AffineMap affineApplyMap;
      SmallVector<Value, 8> affineApplyOperands;

      // llvm::dbgs() << "\nop to start: " << t << "\n";

      if (auto op = t.getDefiningOp<AddIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2,
                           getAffineSymbolExpr(0, op.getContext()) +
                               getAffineSymbolExpr(1, op.getContext()));
      } else if (auto op = t.getDefiningOp<SubIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2,
                           getAffineSymbolExpr(0, op.getContext()) -
                               getAffineSymbolExpr(1, op.getContext()));
      } else if (auto op = t.getDefiningOp<MulIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2,
                           getAffineSymbolExpr(0, op.getContext()) *
                               getAffineSymbolExpr(1, op.getContext()));
      } else if (auto op = t.getDefiningOp<DivSIOp>()) {
        affineApplyMap = AffineMap::get(
            0, 2,
            getAffineSymbolExpr(0, op.getContext())
                .floorDiv(getAffineSymbolExpr(1, op.getContext())));
      } else if (auto op = t.getDefiningOp<DivUIOp>()) {
        affineApplyMap = AffineMap::get(
            0, 2,
            getAffineSymbolExpr(0, op.getContext())
                .floorDiv(getAffineSymbolExpr(1, op.getContext())));
      } else if (auto op = t.getDefiningOp<RemSIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2,
                           getAffineSymbolExpr(0, op.getContext()) %
                               getAffineSymbolExpr(1, op.getContext()));
      } else if (auto op = t.getDefiningOp<RemUIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2,
                           getAffineSymbolExpr(0, op.getContext()) %
                               getAffineSymbolExpr(1, op.getContext()));
      } else if (auto op = t.getDefiningOp<ConstantIntOp>()) {
        affineApplyMap = AffineMap::get(
            0, 0, getAffineConstantExpr(op.value(), op.getContext()));
      } else if (auto op = t.getDefiningOp<ConstantIndexOp>()) {
        affineApplyMap = AffineMap::get(
            0, 0, getAffineConstantExpr(op.value(), op.getContext()));
      } else {
        llvm_unreachable("");
      }

      for (auto op : t.getDefiningOp()->getOperands()) {
        affineApplyOperands.push_back(op);
      }

      SmallVector<AffineExpr, 0> dimRemapping;
      unsigned numOtherSymbols = affineApplyOperands.size();
      SmallVector<AffineExpr, 2> symRemapping(numOtherSymbols);
      for (unsigned idx = 0; idx < numOtherSymbols; ++idx) {
        symRemapping[idx] = getAffineSymbolExpr(addedValues.size(),
                                                affineApplyMap.getContext());
        addedValues.push_back(affineApplyOperands[idx]);
      }
      affineApplyMap = affineApplyMap.replaceDimsAndSymbols(
          dimRemapping, symRemapping, reorderedDims.size(), addedValues.size());

      if (i >= numDimsBeforeRewrite)
        symbolsToPromote.insert(i - numDimsBeforeRewrite);

      LLVM_DEBUG(affineApplyMap.print(
          llvm::dbgs() << "\nRenumber into current normalizer: "));
      auxiliaryExprs.push_back(affineApplyMap.getResult(0));
      /*
      llvm::dbgs() << "\n";
      for(auto op : affineApplyOperands) {
        llvm::dbgs() << " + prevop: " << op << "\n";
      }
      */
    } else if (isAffineForArg(t)) {
      auxiliaryExprs.push_back(renumberOneDim(t));
      if (i >= numDimsBeforeRewrite)
        symbolsToPromote.insert(i - numDimsBeforeRewrite);
    } else if (auto affineApply = t.getDefiningOp<AffineApplyOp>()) {
      // a. Compose affine.apply operations.
      LLVM_DEBUG(affineApply->print(
          llvm::dbgs() << "\nCompose AffineApplyOp recursively: "));
      AffineMap affineApplyMap = affineApply.getAffineMap();
      SmallVector<Value, 8> affineApplyOperands(
          affineApply.getOperands().begin(), affineApply.getOperands().end());

      SmallVector<AffineExpr, 0> dimRemapping(affineApplyMap.getNumDims());

      for (size_t i = 0; i < affineApplyMap.getNumDims(); ++i) {
        assert(i < affineApplyOperands.size());
        dimRemapping[i] = renumberOneDim(affineApplyOperands[i]);
      }
      unsigned numOtherSymbols = affineApplyOperands.size();
      SmallVector<AffineExpr, 2> symRemapping(numOtherSymbols -
                                              affineApplyMap.getNumDims());
      for (unsigned idx = 0; idx < symRemapping.size(); ++idx) {
        symRemapping[idx] = getAffineSymbolExpr(addedValues.size(),
                                                affineApplyMap.getContext());
        addedValues.push_back(
            affineApplyOperands[idx + affineApplyMap.getNumDims()]);
      }
      affineApplyMap = affineApplyMap.replaceDimsAndSymbols(
          dimRemapping, symRemapping, reorderedDims.size(), addedValues.size());

      if (i >= numDimsBeforeRewrite)
        symbolsToPromote.insert(i - numDimsBeforeRewrite);

      LLVM_DEBUG(
          affineApplyMap.print(llvm::dbgs() << "\nAffine apply fixup map: "));
      auxiliaryExprs.push_back(affineApplyMap.getResult(0));
    } else {
      if (!isValidSymbolInt(t, /*recur*/ false)) {
        if (auto idx = t.getDefiningOp()) {
          auto scope = getAffineScope(idx)->getParentOp();
          DominanceInfo DI(scope);

          std::function<bool(Value)> fix = [&](Value v) -> bool /*legal*/ {
            if (isValidSymbolInt(v, /*recur*/ false))
              return true;
            auto op = v.getDefiningOp();
            if (!op)
              llvm::errs() << v << "\n";
            assert(op);
            if (isa<ConstantOp>(op) || isa<ConstantIndexOp>(op))
              return true;
            if (!isReadOnly(op)) {
              return false;
            }
            Operation *front = nullptr;
            for (auto o : op->getOperands()) {
              Operation *next;
              if (auto op = o.getDefiningOp()) {
                if (!fix(o)) {
                  return false;
                }
                next = op;
              } else {
                auto BA = o.cast<BlockArgument>();
                if (!isValidSymbolInt(o, /*recur*/ false)) {
                  return false;
                }
                next = &BA.getOwner()->front();
              }
              if (front == nullptr)
                front = next;
              else if (DI.dominates(front, next))
                front = next;
            }
            if (!front)
              op->dump();
            assert(front);
            op->moveAfter(front);
            return true;
          };
          if (fix(t))
            assert(isValidSymbolInt(t, /*recur*/ false));
          else
            assert(0 && "cannot move");
        } else
          assert(0 && "cannot move");
      }
      if (i < numDimsBeforeRewrite) {
        // b. The mathematical composition of AffineMap composes dims.
        auxiliaryExprs.push_back(renumberOneDim(t));
      } else {
        // c. The mathematical composition of AffineMap concatenates symbols.
        //    Note that the map composition will put symbols already present
        //    in the map before any symbols coming from the auxiliary map, so
        //    we insert them before any symbols that are due to renumbering,
        //    and after the proper symbols we have seen already.
        concatenatedSymbols.insert(
            std::next(concatenatedSymbols.begin(), numProperSymbols++), t);
      }
    }
  }

  for (auto val : addedValues) {
    concatenatedSymbols.push_back(val);
  }

  {
    // Create the new map by replacing each symbol at pos by the next new dim.
    unsigned numDims = map.getNumDims();
    unsigned numSymbols = map.getNumSymbols();
    unsigned numNewDims = 0;
    unsigned numNewSymbols = 0;
    SmallVector<AffineExpr, 8> symReplacements(numSymbols);
    for (unsigned i = 0; i < numSymbols; ++i) {
      symReplacements[i] =
          symbolsToPromote.count(i) > 0
              ? getAffineDimExpr(numDims + numNewDims++, map.getContext())
              : getAffineSymbolExpr(numNewSymbols++, map.getContext());
    }
    assert(numSymbols >= numNewDims);
    map = map.replaceDimsAndSymbols({}, symReplacements, numDims + numNewDims,
                                    numNewSymbols);
  }

  LLVM_DEBUG(map.print(llvm::dbgs() << "\nRewritten map: "));

  assert(concatenatedSymbols.size() >= map.getNumSymbols() &&
         "Unexpected number of concatenated symbols");
  auto numDims = dimValueToPosition.size();
  assert(dimValueToPosition.size() == reorderedDims.size());
  auto numSymbols = concatenatedSymbols.size() - map.getNumSymbols();
  auto auxiliaryMap =
      AffineMap::get(numDims, numSymbols, auxiliaryExprs, map.getContext());

  /*
  llvm::dbgs() << "prev operands\n";
  for(auto a : concatenatedSymbols) {
    llvm::dbgs() << " &&& concatsym: " << a << "\n";
  }
  llvm::dbgs() << "\nprev operands\n";
  for(auto a : operands) {
    llvm::dbgs() << " *** pop: " << a << "\n";
  }
  */

  LLVM_DEBUG(map.print(llvm::dbgs() << "\nCompose map: "));
  LLVM_DEBUG(auxiliaryMap.print(llvm::dbgs() << "\nWith map: "));
  LLVM_DEBUG(map.compose(auxiliaryMap).print(llvm::dbgs() << "\nResult: "));

  // TODO: Disabling simplification results in major speed gains.
  // Another option is to cache the results as it is expected a lot of redundant
  // work is performed in practice.
  affineMap = simplifyAffineMap(map.compose(auxiliaryMap));

  LLVM_DEBUG(affineMap.print(llvm::dbgs() << "\nSimplified result: "));
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

AffineDimExpr AffineApplyNormalizer::renumberOneDim(Value v) {
  DenseMap<Value, unsigned>::iterator iterPos;
  bool inserted = false;
  std::tie(iterPos, inserted) =
      dimValueToPosition.insert(std::make_pair(v, dimValueToPosition.size()));
  if (inserted) {
    reorderedDims.push_back(v);
  }
  return getAffineDimExpr(iterPos->second, v.getContext())
      .cast<AffineDimExpr>();
}

static void composeAffineMapAndOperands(AffineMap *map,
                                        SmallVectorImpl<Value> *operands) {
  AffineApplyNormalizer normalizer(*map, *operands);
  auto normalizedMap = normalizer.getAffineMap();
  auto normalizedOperands = normalizer.getOperands();
  canonicalizeMapAndOperands(&normalizedMap, &normalizedOperands);
  *map = normalizedMap;
  *operands = normalizedOperands;
  assert(*map);
}

bool need(AffineMap *map, SmallVectorImpl<Value> *operands) {
  assert(map->getNumInputs() == operands->size());
  for (size_t i = 0; i < map->getNumInputs(); ++i) {
    auto v = (*operands)[i];
    if (legalCondition(v, i < map->getNumDims()))
      return true;
  }
  return false;
}
bool need(IntegerSet *map, SmallVectorImpl<Value> *operands) {
  for (size_t i = 0; i < map->getNumInputs(); ++i) {
    auto v = (*operands)[i];
    if (legalCondition(v, i < map->getNumDims()))
      return true;
  }
  return false;
}

void fully2ComposeAffineMapAndOperands(OpBuilder &builder, AffineMap *map,
                                       SmallVectorImpl<Value> *operands) {
  BlockAndValueMapping indexMap;
  for (auto op : *operands) {
    SmallVector<IndexCastOp> attempt;
    auto idx0 = op.getDefiningOp<IndexCastOp>();
    attempt.push_back(idx0);
    if (!idx0)
      continue;

    for (auto &u : idx0.getIn().getUses()) {
      if (auto idx = dyn_cast<IndexCastOp>(u.getOwner()))
        attempt.push_back(idx);
    }

    for (auto idx : attempt) {
      Operation *start = idx;
      bool immediate = false;

      while (1) {
        if (start == idx.getIn().getDefiningOp()) {
          immediate = true;
          break;
        }
        if (isa<IndexCastOp>(start)) {
          if (start == &start->getBlock()->front()) {
            if (auto BA = idx.getIn().dyn_cast<BlockArgument>())
              if (start->getBlock() == BA.getOwner()) {
                immediate = true;
                break;
              }
            break;
          }
          start = start->getPrevNode();
        }
        break;
      }
      if (immediate) {
        indexMap.map(idx.getIn(), idx);
        break;
      }
    }
  }
  assert(map->getNumInputs() == operands->size());
  while (need(map, operands)) {
    composeAffineMapAndOperands(map, operands);
    assert(map->getNumInputs() == operands->size());
  }
  for (auto &op : *operands) {
    if (!op.getType().isIndex()) {
      Operation *toInsert;
      if (auto o = op.getDefiningOp())
        toInsert = o->getNextNode();
      else {
        auto BA = op.cast<BlockArgument>();
        toInsert = &BA.getOwner()->front();
      }

      if (auto v = indexMap.lookupOrNull(op))
        op = v;
      else {
        OpBuilder::InsertionGuard B(builder);
        builder.setInsertionPoint(toInsert);
        op = builder.create<IndexCastOp>(op.getLoc(), builder.getIndexType(),
                                         op);
      }
    }
  }
}

static void composeIntegerSetAndOperands(IntegerSet *set,
                                         SmallVectorImpl<Value> *operands) {
  auto amap = AffineMap::get(set->getNumDims(), set->getNumSymbols(),
                             set->getConstraints(), set->getContext());
  AffineApplyNormalizer normalizer(amap, *operands);
  auto normalizedMap = normalizer.getAffineMap();
  auto normalizedOperands = normalizer.getOperands();
  canonicalizeMapAndOperands(&normalizedMap, &normalizedOperands);
  *set =
      IntegerSet::get(normalizedMap.getNumDims(), normalizedMap.getNumSymbols(),
                      normalizedMap.getResults(), set->getEqFlags());
  *operands = normalizedOperands;
}

void fully2ComposeIntegerSetAndOperands(OpBuilder &builder, IntegerSet *set,
                                        SmallVectorImpl<Value> *operands) {
  BlockAndValueMapping indexMap;
  for (auto op : *operands) {
    if (auto idx = op.getDefiningOp<IndexCastOp>()) {
      Operation *start = idx;
      bool immediate = false;

      while (1) {
        if (start == idx.getIn().getDefiningOp()) {
          immediate = true;
          break;
        }
        if (isa<IndexCastOp>(start)) {
          if (start == &start->getBlock()->front()) {
            if (auto BA = idx.getIn().dyn_cast<BlockArgument>())
              if (start->getBlock() == BA.getOwner()) {
                immediate = true;
                break;
              }
            break;
          }
          start = start->getPrevNode();
        }
        break;
      }
      if (immediate)
        indexMap.map(idx.getIn(), idx);
    }
  }
  while (need(set, operands)) {
    composeIntegerSetAndOperands(set, operands);
  }
  for (auto &op : *operands) {
    if (!op.getType().isIndex()) {
      Operation *toInsert;
      if (auto o = op.getDefiningOp())
        toInsert = o->getNextNode();
      else {
        auto BA = op.cast<BlockArgument>();
        toInsert = &BA.getOwner()->front();
      }

      if (auto v = indexMap.lookupOrNull(op))
        op = v;
      else {
        OpBuilder::InsertionGuard B(builder);
        builder.setInsertionPoint(toInsert);
        op = builder.create<IndexCastOp>(op.getLoc(), builder.getIndexType(),
                                         op);
      }
    }
  }
}

namespace {
struct AffineCFGPass : public AffineCFGBase<AffineCFGPass> {
  void runOnOperation() override;
};
} // namespace

static void setLocationAfter(OpBuilder &b, mlir::Value val) {
  if (val.getDefiningOp()) {
    auto it = val.getDefiningOp()->getIterator();
    it++;
    b.setInsertionPoint(val.getDefiningOp()->getBlock(), it);
  }
  if (auto bop = val.dyn_cast<mlir::BlockArgument>())
    b.setInsertionPoint(bop.getOwner(), bop.getOwner()->begin());
}

struct IndexCastMovement : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    mlir::Value val = op.getOperand();
    if (auto bop = val.dyn_cast<mlir::BlockArgument>()) {
      if (op.getOperation()->getBlock() != bop.getOwner()) {
        op.getOperation()->moveBefore(bop.getOwner(), bop.getOwner()->begin());
        return success();
      }
      return failure();
    }

    if (val.getDefiningOp()) {
      if (op.getOperation()->getBlock() != val.getDefiningOp()->getBlock()) {
        auto it = val.getDefiningOp()->getIterator();
        op.getOperation()->moveAfter(val.getDefiningOp()->getBlock(), it);
      }
      return failure();
    }
    return failure();
  }
};

struct SimplfyIntegerCastMath : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<AddIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<AddIOp>(
          op,
          b.create<IndexCastOp>(op.getLoc(), op.getType(), iadd.getOperand(0)),
          b2.create<IndexCastOp>(op.getLoc(), op.getType(),
                                 iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<SubIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<SubIOp>(
          op,
          b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                       iadd.getOperand(0)),
          b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                        iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<MulIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<MulIOp>(
          op,
          b.create<IndexCastOp>(op.getLoc(), op.getType(), iadd.getOperand(0)),
          b2.create<IndexCastOp>(op.getLoc(), op.getType(),
                                 iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<DivUIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<DivUIOp>(
          op,
          b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                       iadd.getOperand(0)),
          b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                        iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<DivSIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<DivSIOp>(
          op,
          b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                       iadd.getOperand(0)),
          b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                        iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<RemUIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<RemUIOp>(
          op,
          b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                       iadd.getOperand(0)),
          b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                        iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<RemSIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<RemSIOp>(
          op,
          b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                       iadd.getOperand(0)),
          b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                        iadd.getOperand(1)));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<SelectOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getTrueValue());
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getFalseValue());
      auto cond = iadd.getCondition();
      OpBuilder b3(rewriter);
      setLocationAfter(b3, cond);
      if (auto cmp = iadd.getCondition().getDefiningOp<CmpIOp>()) {
        if (cmp.getLhs() == iadd.getTrueValue() &&
            cmp.getRhs() == iadd.getFalseValue()) {

          auto truev = b.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                                    iadd.getTrueValue());
          auto falsev = b2.create<arith::IndexCastOp>(op.getLoc(), op.getType(),
                                                      iadd.getFalseValue());
          cond = b3.create<CmpIOp>(cmp.getLoc(), cmp.getPredicate(), truev,
                                   falsev);
          rewriter.replaceOpWithNewOp<SelectOp>(op, cond, truev, falsev);
          return success();
        }
      }
    }
    return failure();
  }
};

struct CanonicalizeAffineApply : public OpRewritePattern<AffineApplyOp> {
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp affineOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 4> mapOperands(affineOp.mapOperands());
    auto map = affineOp.map();
    auto prevMap = map;

    fully2ComposeAffineMapAndOperands(rewriter, &map, &mapOperands);
    canonicalizeMapAndOperands(&map, &mapOperands);
    map = removeDuplicateExprs(map);

    if (map == prevMap)
      return failure();

    rewriter.replaceOpWithNewOp<AffineApplyOp>(affineOp, map, mapOperands);
    return success();
  }
};

struct CanonicalizeIndexCast : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp indexcastOp,
                                PatternRewriter &rewriter) const override {

    // Fold IndexCast(IndexCast(x)) -> x
    auto cast = indexcastOp.getOperand().getDefiningOp<IndexCastOp>();
    if (cast && cast.getOperand().getType() == indexcastOp.getType()) {
      mlir::Value vals[] = {cast.getOperand()};
      rewriter.replaceOp(indexcastOp, vals);
      return success();
    }

    // Fold IndexCast(constant) -> constant
    // A little hack because we go through int.  Otherwise, the size
    // of the constant might need to change.
    if (auto cst = indexcastOp.getOperand().getDefiningOp<ConstantIntOp>()) {
      rewriter.replaceOpWithNewOp<ConstantIndexOp>(indexcastOp, cst.value());
      return success();
    }
    return failure();
  }
};

/*
struct CanonicalizeAffineIf : public OpRewritePattern<AffineIfOp> {
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AffineIfOp affineOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> mapOperands(affineOp.mapOperands());
    auto map = affineOp.map();
    auto prevMap = map;
    fully2ComposeAffineMapAndOperands(&map, &mapOperands);
    canonicalizeMapAndOperands(&map, &mapOperands);
    map = removeDuplicateExprs(map);
    if (map == prevMap)
      return failure();
    rewriter.replaceOpWithNewOp<AffineApplyOp>(affineOp, map, mapOperands);
    return success();
  }
};
*/

bool isValidIndex(Value val) {
  if (isValidSymbolInt(val))
    return true;

  if (auto cast = val.getDefiningOp<IndexCastOp>())
    return isValidIndex(cast.getOperand());

  if (auto bop = val.getDefiningOp<AddIOp>())
    return isValidIndex(bop.getOperand(0)) && isValidIndex(bop.getOperand(1));

  if (auto bop = val.getDefiningOp<MulIOp>())
    return (isValidIndex(bop.getOperand(0)) &&
            isValidSymbolInt(bop.getOperand(1))) ||
           (isValidIndex(bop.getOperand(1)) &&
            isValidSymbolInt(bop.getOperand(0)));

  if (auto bop = val.getDefiningOp<DivSIOp>())
    return (isValidIndex(bop.getOperand(0)) &&
            isValidSymbolInt(bop.getOperand(1)));

  if (auto bop = val.getDefiningOp<DivUIOp>())
    return (isValidIndex(bop.getOperand(0)) &&
            isValidSymbolInt(bop.getOperand(1)));

  if (auto bop = val.getDefiningOp<RemSIOp>()) {
    return (isValidIndex(bop.getOperand(0)) &&
            isValidSymbolInt(bop.getOperand(1)));
  }

  if (auto bop = val.getDefiningOp<RemUIOp>())
    return (isValidIndex(bop.getOperand(0)) &&
            isValidSymbolInt(bop.getOperand(1)));

  if (auto bop = val.getDefiningOp<SubIOp>())
    return isValidIndex(bop.getOperand(0)) && isValidIndex(bop.getOperand(1));

  if (val.getDefiningOp<ConstantIndexOp>())
    return true;

  if (val.getDefiningOp<ConstantIntOp>())
    return true;

  if (auto ba = val.dyn_cast<BlockArgument>()) {
    auto owner = ba.getOwner();
    assert(owner);

    auto parentOp = owner->getParentOp();
    if (!parentOp) {
      owner->dump();
      llvm::errs() << " ba: " << ba << "\n";
    }
    assert(parentOp);
    if (isa<FunctionOpInterface>(parentOp))
      return true;
    if (auto af = dyn_cast<AffineForOp>(parentOp))
      return af.getInductionVar() == ba;

    // TODO ensure not a reduced var
    if (isa<AffineParallelOp>(parentOp))
      return true;

    if (isa<FunctionOpInterface>(parentOp))
      return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "illegal isValidIndex: " << val << "\n");
  return false;
}

// returns legality
bool handleMinMax(Value start, SmallVectorImpl<Value> &out, bool &min,
                  bool &max) {

  SmallVector<Value> todo = {start};
  while (todo.size()) {
    auto cur = todo.back();
    todo.pop_back();
    if (isValidIndex(cur)) {
      out.push_back(cur);
      continue;
    } else if (auto selOp = cur.getDefiningOp<SelectOp>()) {
      // UB only has min of operands
      if (auto cmp = selOp.getCondition().getDefiningOp<CmpIOp>()) {
        if (cmp.getLhs() == selOp.getTrueValue() &&
            cmp.getRhs() == selOp.getFalseValue()) {
          todo.push_back(cmp.getLhs());
          todo.push_back(cmp.getRhs());
          if (cmp.getPredicate() == CmpIPredicate::sle ||
              cmp.getPredicate() == CmpIPredicate::slt) {
            min = true;
            continue;
          }
          if (cmp.getPredicate() == CmpIPredicate::sge ||
              cmp.getPredicate() == CmpIPredicate::sgt) {
            max = true;
            continue;
          }
        }
      }
    }
    return false;
  }
  return !(min && max);
}

bool handle(OpBuilder &b, CmpIOp cmpi, SmallVectorImpl<AffineExpr> &exprs,
            SmallVectorImpl<bool> &eqflags, SmallVectorImpl<Value> &applies) {
  SmallVector<Value> lhs;
  bool lhs_min = false;
  bool lhs_max = false;
  if (!handleMinMax(cmpi.getLhs(), lhs, lhs_min, lhs_max)) {
    LLVM_DEBUG(llvm::dbgs()
               << "illegal lhs: " << cmpi.getLhs() << " - " << cmpi << "\n");
    return false;
  }
  assert(lhs.size());
  SmallVector<Value> rhs;
  bool rhs_min = false;
  bool rhs_max = false;
  if (!handleMinMax(cmpi.getRhs(), rhs, rhs_min, rhs_max)) {
    LLVM_DEBUG(llvm::dbgs()
               << "illegal rhs: " << cmpi.getRhs() << " - " << cmpi << "\n");
    return false;
  }
  assert(rhs.size());
  for (auto &lhspack : lhs)
    if (!lhspack.getType().isa<IndexType>()) {
      lhspack = b.create<arith::IndexCastOp>(
          cmpi.getLoc(), IndexType::get(cmpi.getContext()), lhspack);
    }

  for (auto &rhspack : rhs)
    if (!rhspack.getType().isa<IndexType>()) {
      rhspack = b.create<arith::IndexCastOp>(
          cmpi.getLoc(), IndexType::get(cmpi.getContext()), rhspack);
    }

  switch (cmpi.getPredicate()) {
  case CmpIPredicate::eq: {
    if (lhs_min || lhs_max || rhs_min || rhs_max)
      return false;
    eqflags.push_back(true);

    applies.push_back(lhs[0]);
    applies.push_back(rhs[0]);
    AffineExpr dims[2] = {b.getAffineSymbolExpr(2 * exprs.size() + 0),
                          b.getAffineSymbolExpr(2 * exprs.size() + 1)};
    exprs.push_back(dims[0] - dims[1]);
  } break;

  case CmpIPredicate::sge:
  case CmpIPredicate::sgt: {
    // if lhs >=? rhs
    // if lhs is a min(a, b) both must be true and this is fine
    // if lhs is a max(a, b) either may be true, and sets require and
    // similarly if rhs is a max(), both must be true;
    if (lhs_max || rhs_min)
      return false;
    for (auto lhspack : lhs)
      for (auto rhspack : rhs) {
        eqflags.push_back(false);
        applies.push_back(lhspack);
        applies.push_back(rhspack);
        AffineExpr dims[2] = {b.getAffineSymbolExpr(2 * exprs.size() + 0),
                              b.getAffineSymbolExpr(2 * exprs.size() + 1)};
        auto expr = dims[0] - dims[1];
        if (cmpi.getPredicate() == CmpIPredicate::sgt)
          expr = expr + 1;
        exprs.push_back(expr);
      }
  } break;

  case CmpIPredicate::slt:
  case CmpIPredicate::sle: {
    if (lhs_min || rhs_max)
      return false;
    for (auto lhspack : lhs)
      for (auto rhspack : rhs) {
        eqflags.push_back(false);
        applies.push_back(lhspack);
        applies.push_back(rhspack);
        AffineExpr dims[2] = {b.getAffineSymbolExpr(2 * exprs.size() + 0),
                              b.getAffineSymbolExpr(2 * exprs.size() + 1)};
        auto expr = dims[1] - dims[0];
        if (cmpi.getPredicate() == CmpIPredicate::slt)
          expr = expr - 1;
        exprs.push_back(expr);
      }
  } break;

  case CmpIPredicate::ne:
  case CmpIPredicate::ult:
  case CmpIPredicate::ule:
  case CmpIPredicate::ugt:
  case CmpIPredicate::uge:
    LLVM_DEBUG(llvm::dbgs() << "illegal icmp: " << cmpi << "\n");
    return false;
  }
  return true;
}
/*
static void replaceStore(memref::StoreOp store,
                         const SmallVector<Value, 2> &newIndexes) {
  auto memrefType = store.getMemRef().getType().cast<MemRefType>();
  size_t rank = memrefType.getRank();
  if (rank != newIndexes.size()) {
    llvm::errs() << store << "\n";
  }
  assert(rank == newIndexes.size() && "Expect rank to match new indexes");

  OpBuilder builder(store);
  Location loc = store.getLoc();
  builder.create<AffineStoreOp>(loc, store.getValueToStore(), store.getMemRef(),
                                newIndexes);
  store.erase();
}

static void replaceLoad(memref::LoadOp load,
                        const SmallVector<Value, 2> &newIndexes) {
  OpBuilder builder(load);
  Location loc = load.getLoc();

  auto memrefType = load.getMemRef().getType().cast<MemRefType>();
  size_t rank = memrefType.getRank();
  if (rank != newIndexes.size()) {
    llvm::errs() << load << "\n";
  }
  assert(rank == newIndexes.size() && "rank must equal new indexes size");

  AffineLoadOp affineLoad =
      builder.create<AffineLoadOp>(loc, load.getMemRef(), newIndexes);
  load.getResult().replaceAllUsesWith(affineLoad.getResult());
  load.erase();
}
*/
struct MoveLoadToAffine : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp load,
                                PatternRewriter &rewriter) const override {
    if (!llvm::all_of(load.getIndices(), isValidIndex))
      return failure();

    auto memrefType = load.getMemRef().getType().cast<MemRefType>();
    int64_t rank = memrefType.getRank();

    // Create identity map for memrefs with at least one dimension or () -> ()
    // for zero-dimensional memrefs.
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(rewriter.getAffineSymbolExpr(i));
    auto map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs,
                              rewriter.getContext());

    SmallVector<Value, 4> operands = load.getIndices();

    if (map.getNumInputs() != operands.size()) {
      // load->getParentOfType<FuncOp>().dump();
      llvm::errs() << " load: " << load << "\n";
    }
    assert(map.getNumInputs() == operands.size());
    fully2ComposeAffineMapAndOperands(rewriter, &map, &operands);
    assert(map.getNumInputs() == operands.size());
    canonicalizeMapAndOperands(&map, &operands);
    assert(map.getNumInputs() == operands.size());

    AffineLoadOp affineLoad = rewriter.create<AffineLoadOp>(
        load.getLoc(), load.getMemRef(), map, operands);
    load.getResult().replaceAllUsesWith(affineLoad.getResult());
    rewriter.eraseOp(load);
    return success();
  }
};

struct MoveStoreToAffine : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp store,
                                PatternRewriter &rewriter) const override {
    if (!llvm::all_of(store.getIndices(), isValidIndex))
      return failure();

    auto memrefType = store.getMemRef().getType().cast<MemRefType>();
    int64_t rank = memrefType.getRank();

    // Create identity map for memrefs with at least one dimension or () -> ()
    // for zero-dimensional memrefs.
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(rewriter.getAffineSymbolExpr(i));
    auto map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs,
                              rewriter.getContext());
    SmallVector<Value, 4> operands = store.getIndices();

    fully2ComposeAffineMapAndOperands(rewriter, &map, &operands);
    canonicalizeMapAndOperands(&map, &operands);

    rewriter.create<AffineStoreOp>(store.getLoc(), store.getValueToStore(),
                                   store.getMemRef(), map, operands);
    rewriter.eraseOp(store);
    return success();
  }
};

static bool areChanged(SmallVectorImpl<Value> &afterOperands,
                       SmallVectorImpl<Value> &beforeOperands) {
  if (afterOperands.size() != beforeOperands.size())
    return true;
  if (!std::equal(afterOperands.begin(), afterOperands.end(),
                  beforeOperands.begin()))
    return true;
  return false;
}

template <typename T> struct AffineFixup : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  /// Replace the affine op with another instance of it with the supplied
  /// map and mapOperands.
  void replaceAffineOp(PatternRewriter &rewriter, T affineOp, AffineMap map,
                       ArrayRef<Value> mapOperands) const;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    auto map = op.getAffineMap();
    SmallVector<Value, 4> operands = op.getMapOperands();

    auto prevMap = map;
    auto prevOperands = operands;

    assert(map.getNumInputs() == operands.size());
    fully2ComposeAffineMapAndOperands(rewriter, &map, &operands);
    assert(map.getNumInputs() == operands.size());
    canonicalizeMapAndOperands(&map, &operands);
    assert(map.getNumInputs() == operands.size());

    if (map == prevMap && !areChanged(operands, prevOperands))
      return failure();

    replaceAffineOp(rewriter, op, map, operands);
    return success();
  }
};

// Specialize the template to account for the different build signatures for
// affine load, store, and apply ops.
template <>
void AffineFixup<AffineLoadOp>::replaceAffineOp(
    PatternRewriter &rewriter, AffineLoadOp load, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<AffineLoadOp>(load, load.getMemRef(), map,
                                            mapOperands);
}
template <>
void AffineFixup<AffinePrefetchOp>::replaceAffineOp(
    PatternRewriter &rewriter, AffinePrefetchOp prefetch, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<AffinePrefetchOp>(
      prefetch, prefetch.memref(), map, mapOperands, prefetch.localityHint(),
      prefetch.isWrite(), prefetch.isDataCache());
}
template <>
void AffineFixup<AffineStoreOp>::replaceAffineOp(
    PatternRewriter &rewriter, AffineStoreOp store, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<AffineStoreOp>(
      store, store.getValueToStore(), store.getMemRef(), map, mapOperands);
}
template <>
void AffineFixup<AffineVectorLoadOp>::replaceAffineOp(
    PatternRewriter &rewriter, AffineVectorLoadOp vectorload, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<AffineVectorLoadOp>(
      vectorload, vectorload.getVectorType(), vectorload.getMemRef(), map,
      mapOperands);
}
template <>
void AffineFixup<AffineVectorStoreOp>::replaceAffineOp(
    PatternRewriter &rewriter, AffineVectorStoreOp vectorstore, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<AffineVectorStoreOp>(
      vectorstore, vectorstore.getValueToStore(), vectorstore.getMemRef(), map,
      mapOperands);
}

// Generic version for ops that don't have extra operands.
template <typename AffineOpTy>
void AffineFixup<AffineOpTy>::replaceAffineOp(
    PatternRewriter &rewriter, AffineOpTy op, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<AffineOpTy>(op, map, mapOperands);
}

struct CanonicalieForBounds : public OpRewritePattern<AffineForOp> {
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundOperands());
    SmallVector<Value, 4> ubOperands(forOp.getUpperBoundOperands());
    SmallVector<Value, 4> origLbOperands(forOp.getLowerBoundOperands());
    SmallVector<Value, 4> origUbOperands(forOp.getUpperBoundOperands());

    auto lbMap = forOp.getLowerBoundMap();
    auto ubMap = forOp.getUpperBoundMap();
    auto prevLbMap = lbMap;
    auto prevUbMap = ubMap;

    // llvm::errs() << "*********\n";
    // ubMap.dump();

    fully2ComposeAffineMapAndOperands(rewriter, &lbMap, &lbOperands);
    canonicalizeMapAndOperands(&lbMap, &lbOperands);
    lbMap = removeDuplicateExprs(lbMap);

    fully2ComposeAffineMapAndOperands(rewriter, &ubMap, &ubOperands);
    canonicalizeMapAndOperands(&ubMap, &ubOperands);
    ubMap = removeDuplicateExprs(ubMap);

    // ubMap.dump();
    // forOp.dump();

    // Any canonicalization change in map or operands always leads to updated
    // map(s).
    if ((lbMap == prevLbMap && ubMap == prevUbMap) &&
        (!areChanged(lbOperands, origLbOperands)) &&
        (!areChanged(ubOperands, origUbOperands)))
      return failure();

    // llvm::errs() << "oldParent:" << *forOp.getParentOp() << "\n";
    // llvm::errs() << "oldfor:" << forOp << "\n";

    if ((lbMap != prevLbMap) || areChanged(lbOperands, origLbOperands))
      forOp.setLowerBound(lbOperands, lbMap);
    if ((ubMap != prevUbMap) || areChanged(ubOperands, origUbOperands))
      forOp.setUpperBound(ubOperands, ubMap);

    // llvm::errs() << "newfor:" << forOp << "\n";
    return success();
  }
};

struct CanonicalizIfBounds : public OpRewritePattern<AffineIfOp> {
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> operands(op.getOperands());
    SmallVector<Value, 4> origOperands(operands);

    auto map = op.getIntegerSet();
    auto prevMap = map;

    // llvm::errs() << "*********\n";
    // ubMap.dump();

    fully2ComposeIntegerSetAndOperands(rewriter, &map, &operands);
    canonicalizeSetAndOperands(&map, &operands);

    // map(s).
    if (map == prevMap && !areChanged(operands, origOperands))
      return failure();

    op.setConditional(map, operands);

    return success();
  }
};

struct MoveIfToAffine : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!ifOp->getParentOfType<AffineForOp>() &&
        !ifOp->getParentOfType<AffineParallelOp>())
      return failure();

    std::vector<mlir::Type> types;
    for (auto v : ifOp.getResults()) {
      types.push_back(v.getType());
    }

    SmallVector<AffineExpr, 2> exprs;
    SmallVector<bool, 2> eqflags;
    SmallVector<Value, 4> applies;

    std::deque<Value> todo = {ifOp.getCondition()};
    while (todo.size()) {
      auto cur = todo.front();
      todo.pop_front();
      if (auto cmpi = cur.getDefiningOp<CmpIOp>()) {
        if (!handle(rewriter, cmpi, exprs, eqflags, applies)) {
          return failure();
        }
        continue;
      }
      if (auto andi = cur.getDefiningOp<AndIOp>()) {
        todo.push_back(andi.getOperand(0));
        todo.push_back(andi.getOperand(1));
        continue;
      }
      return failure();
    }

    auto iset =
        IntegerSet::get(/*dim*/ 0, /*symbol*/ 2 * exprs.size(), exprs, eqflags);
    fully2ComposeIntegerSetAndOperands(rewriter, &iset, &applies);
    canonicalizeSetAndOperands(&iset, &applies);
    AffineIfOp affineIfOp =
        rewriter.create<AffineIfOp>(ifOp.getLoc(), types, iset, applies,
                                    /*elseBlock=*/true);

    rewriter.setInsertionPoint(ifOp.thenYield());
    rewriter.replaceOpWithNewOp<AffineYieldOp>(ifOp.thenYield(),
                                               ifOp.thenYield().getOperands());

    if (ifOp.getElseRegion().getBlocks().size()) {
      rewriter.setInsertionPoint(ifOp.elseYield());
      rewriter.replaceOpWithNewOp<AffineYieldOp>(
          ifOp.elseYield(), ifOp.elseYield().getOperands());
    }

    affineIfOp.thenRegion().takeBody(ifOp.getThenRegion());
    affineIfOp.elseRegion().takeBody(ifOp.getElseRegion());

    rewriter.replaceOp(ifOp, affineIfOp.getResults());
    return success();
  }
};

void AffineCFGPass::runOnOperation() {
  mlir::RewritePatternSet rpl(getOperation()->getContext());
  rpl.add<SimplfyIntegerCastMath, CanonicalizeAffineApply,
          CanonicalizeIndexCast, IndexCastMovement, AffineFixup<AffineLoadOp>,
          AffineFixup<AffineStoreOp>, CanonicalizIfBounds, MoveStoreToAffine,
          MoveIfToAffine, MoveLoadToAffine, CanonicalieForBounds>(
      getOperation()->getContext());
  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
}

std::unique_ptr<Pass> mlir::polygeist::replaceAffineCFGPass() {
  return std::make_unique<AffineCFGPass>();
}

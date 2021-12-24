#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"
#include <deque>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>

#define DEBUG_TYPE "affine-cfg"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

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
  return (parentOp && isa<AffineForOp>(parentOp));
}

static bool legalCondition(Value en, bool outer = true, bool dim = false) {
  if (en.getDefiningOp<AffineApplyOp>() || en.getDefiningOp<ExtUIOp>() ||
      en.getDefiningOp<AddIOp>() || en.getDefiningOp<SubIOp>() ||
      en.getDefiningOp<MulIOp>()) {
    return true;
  }
  // if (auto IC = dyn_cast_or_null<IndexCastOp>(en.getDefiningOp())) {
  //	if (!outer || legalCondition(IC.getOperand(), false)) return true;
  //}
  if (!dim)
    if (auto BA = en.dyn_cast<BlockArgument>()) {
      if (isa<AffineForOp>(BA.getOwner()->getParentOp()))
        return true;
    }
  return false;
}

// Gather the positions of the operands that are produced by an AffineApplyOp.
static llvm::SetVector<unsigned>
indicesFromAffineApplyOp(ArrayRef<Value> operands) {
  llvm::SetVector<unsigned> res;
  for (auto en : llvm::enumerate(operands)) {
    if (legalCondition(en.value()))
      res.insert(en.index());
  }
  return res;
}

static AffineMap promoteComposedSymbolsAsDims(AffineMap map,
                                              ArrayRef<Value> symbols) {
  if (symbols.empty()) {
    return map;
  }

  // Sanity check on symbols.
  for (auto sym : symbols) {
    // assert(isValidSymbol(sym) && "Expected only valid symbols");
    (void)sym;
  }

  // Extract the symbol positions that come from an AffineApplyOp and
  // needs to be rewritten as dims.
  auto symPositions = indicesFromAffineApplyOp(symbols);
  if (symPositions.empty()) {
    return map;
  }

  // Create the new map by replacing each symbol at pos by the next new dim.
  unsigned numDims = map.getNumDims();
  unsigned numSymbols = map.getNumSymbols();
  unsigned numNewDims = 0;
  unsigned numNewSymbols = 0;
  SmallVector<AffineExpr, 8> symReplacements(numSymbols);
  for (unsigned i = 0; i < numSymbols; ++i) {
    symReplacements[i] =
        symPositions.count(i) > 0
            ? getAffineDimExpr(numDims + numNewDims++, map.getContext())
            : getAffineSymbolExpr(numNewSymbols++, map.getContext());
  }
  assert(numSymbols >= numNewDims);
  AffineMap newMap = map.replaceDimsAndSymbols(
      {}, symReplacements, numDims + numNewDims, numNewSymbols);

  return newMap;
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

  // Promote symbols that come from an AffineApplyOp to dims by rewriting the
  // map to always refer to:
  //   (dims, symbols coming from AffineApplyOp, other symbols).
  // The order of operands can remain unchanged.
  // This is a simplification that relies on 2 ordering properties:
  //   1. rewritten symbols always appear after the original dims in the map;
  //   2. operands are traversed in order and either dispatched to:
  //      a. auxiliaryExprs (dims and symbols rewritten as dims);
  //      b. concatenatedSymbols (all other symbols)
  // This allows operand order to remain unchanged.
  unsigned numDimsBeforeRewrite = map.getNumDims();
  map = promoteComposedSymbolsAsDims(map,
                                     operands.take_back(map.getNumSymbols()));

  LLVM_DEBUG(map.print(llvm::dbgs() << "\nRewritten map: "));

  SmallVector<AffineExpr, 8> auxiliaryExprs;
  bool furtherCompose = (affineApplyDepth() <= kMaxAffineApplyDepth);
  // We fully spell out the 2 cases below. In this particular instance a little
  // code duplication greatly improves readability.
  // Note that the first branch would disappear if we only supported full
  // composition (i.e. infinite kMaxAffineApplyDepth).
  if (!furtherCompose) {
    // 1. Only dispatch dims or symbols.
    for (auto en : llvm::enumerate(operands)) {
      auto t = en.value();
      assert(t.getType().isIndex());
      bool isDim = (en.index() < map.getNumDims());
      if (isDim) {
        // a. The mathematical composition of AffineMap composes dims.
        auxiliaryExprs.push_back(renumberOneDim(t));
      } else {
        // b. The mathematical composition of AffineMap concatenates symbols.
        //    We do the same for symbol operands.
        concatenatedSymbols.push_back(t);
      }
    }
  } else {
    assert(numDimsBeforeRewrite <= operands.size());

    SmallVector<Value, 8> addedValues;

    // 2. Compose AffineApplyOps and dispatch dims or symbols.
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      auto t = operands[i];

      if (t.getDefiningOp<AddIOp>() || t.getDefiningOp<SubIOp>() ||
          t.getDefiningOp<MulIOp>()) {

        AffineMap affineApplyMap;
        SmallVector<Value, 8> affineApplyOperands;

        // llvm::dbgs() << "\nop to start: " << t << "\n";

        if (auto op = t.getDefiningOp<AddIOp>()) {
          affineApplyMap =
              AffineMap::get(0, 2,
                             getAffineSymbolExpr(0, op.getContext()) +
                                 getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.append(op.getOperands().begin(),
                                     op.getOperands().end());
        } else if (auto op = t.getDefiningOp<SubIOp>()) {
          affineApplyMap =
              AffineMap::get(0, 2,
                             getAffineSymbolExpr(0, op.getContext()) -
                                 getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.append(op.getOperands().begin(),
                                     op.getOperands().end());
        } else if (auto op = t.getDefiningOp<MulIOp>()) {
          affineApplyMap =
              AffineMap::get(0, 2,
                             getAffineSymbolExpr(0, op.getContext()) *
                                 getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.append(op.getOperands().begin(),
                                     op.getOperands().end());
        } else {
          llvm_unreachable("");
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
            dimRemapping, symRemapping, reorderedDims.size(),
            addedValues.size());

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
        /*
        } else if (auto op = t.getDefiningOp<IndexCastOp>()) {
          // Todo index cast
          if (legalCondition(op.getOperand())) {
                  if (i < numDimsBeforeRewrite) {
                          auxiliaryExprs.push_back(renumberOneDim(t));
                  } else {
                          auxiliaryExprs.push_back(getAffineSymbolExpr(addedValues.size(),
        op.getContext())); addedValues.push_back(op.getOperand());
                  }
          } else {
                  auxiliaryExprs.push_back(getAffineSymbolExpr(addedValues.size(),
        op.getContext())); addedValues.push_back(op);
          }
        } else if (auto op = t.getDefiningOp<ZeroExtendIOp>()) {
          auxiliaryExprs.push_back(getAffineSymbolExpr(addedValues.size(),
        op.getContext())); addedValues.push_back(op.getOperand());
          */
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
            dimRemapping, symRemapping, reorderedDims.size(),
            addedValues.size());

        LLVM_DEBUG(
            affineApplyMap.print(llvm::dbgs() << "\nAffine apply fixup map: "));
        auxiliaryExprs.push_back(affineApplyMap.getResult(0));
      } else {
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
  }

  // Early exit if `map` is already composed.
  if (auxiliaryExprs.empty()) {
    affineMap = map;
    return;
  }

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
    if (legalCondition(v, true, i < map->getNumDims()))
      return true;
  }
  return false;
}
bool need(IntegerSet *map, SmallVectorImpl<Value> *operands) {
  for (size_t i = 0; i < map->getNumInputs(); ++i) {
    auto v = (*operands)[i];
    if (legalCondition(v, true, i < map->getNumDims()))
      return true;
  }
  return false;
}

void fully2ComposeAffineMapAndOperands(AffineMap *map,
                                       SmallVectorImpl<Value> *operands) {
  assert(map->getNumInputs() == operands->size());
  while (need(map, operands)) {
    // llvm::errs() << "pre: " << *map << "\n";
    // for(auto op : *operands) {
    //  llvm::errs() << " -- operands: " << op << "\n";
    //}
    composeAffineMapAndOperands(map, operands);
    assert(map->getNumInputs() == operands->size());
    // llvm::errs() << "post: " << *map << "\n";
    // for(auto op : *operands) {
    //  llvm::errs() << " -- operands: " << op << "\n";
    //}
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

void fully2ComposeIntegerSetAndOperands(IntegerSet *set,
                                        SmallVectorImpl<Value> *operands) {

  // llvm::errs() << "tpre: ";
  // set->dump(); llvm::errs() << "\n";

  // for(auto op : *operands) {
  //  llvm::errs() << " -- operands: " << op << "\n";
  //}
  while (need(set, operands)) {
    // llvm::errs() << "pre: ";
    // set->dump(); llvm::errs() << "\n";

    // for(auto op : *operands) {
    //  llvm::errs() << " -- operands: " << op << "\n";
    //}
    composeIntegerSetAndOperands(set, operands);
    // llvm::errs() << "post: ";
    // set->dump(); llvm::errs() << "\n";
    // for(auto op : *operands) {
    //  llvm::errs() << " -- operands: " << op << "\n";
    //}
  }
}

namespace {
struct AffineCFGPass : public AffineCFGBase<AffineCFGPass> {
  void runOnFunction() override;
};
} // namespace

static bool inAffine(Operation *op) {
  auto *curOp = op;
  while (auto *parentOp = curOp->getParentOp()) {
    if (isa<AffineForOp, AffineParallelOp>(parentOp))
      return true;
    curOp = parentOp;
  }
  return false;
}

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
          b.create<IndexCastOp>(op.getLoc(), iadd.getOperand(0), op.getType()),
          b2.create<IndexCastOp>(op.getLoc(), iadd.getOperand(1),
                                 op.getType()));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<SubIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<SubIOp>(
          op,
          b.create<IndexCastOp>(op.getLoc(), iadd.getOperand(0), op.getType()),
          b2.create<IndexCastOp>(op.getLoc(), iadd.getOperand(1),
                                 op.getType()));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<MulIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<MulIOp>(
          op,
          b.create<IndexCastOp>(op.getLoc(), iadd.getOperand(0), op.getType()),
          b2.create<IndexCastOp>(op.getLoc(), iadd.getOperand(1),
                                 op.getType()));
      return success();
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

    fully2ComposeAffineMapAndOperands(&map, &mapOperands);
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
  if (mlir::isValidSymbol(val))
    return true;

  if (auto cast = val.getDefiningOp<IndexCastOp>())
    return isValidIndex(cast.getOperand());

  if (auto bop = val.getDefiningOp<AddIOp>())
    return isValidIndex(bop.getOperand(0)) && isValidIndex(bop.getOperand(1));

  if (auto bop = val.getDefiningOp<MulIOp>())
    return isValidIndex(bop.getOperand(0)) && isValidIndex(bop.getOperand(1));

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

    if (auto af = dyn_cast<AffineForOp>(parentOp))
      return af.getInductionVar() == ba;

    // TODO ensure not a reduced var
    if (isa<AffineParallelOp>(parentOp))
      return true;

    if (isa<AffineParallelOp>(parentOp))
      return true;

    if (isa<FuncOp>(parentOp))
      return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "illegal isValidIndex: " << val << "\n");
  return false;
}

bool handle(OpBuilder &b, CmpIOp cmpi, SmallVectorImpl<AffineExpr> &exprs,
            SmallVectorImpl<bool> &eqflags, SmallVectorImpl<Value> &applies) {
  AffineMap lhsmap =
      AffineMap::get(0, 1, getAffineSymbolExpr(0, cmpi.getContext()));
  if (!isValidIndex(cmpi.getLhs())) {
    LLVM_DEBUG(llvm::dbgs()
               << "illegal lhs: " << cmpi.getLhs() << " - " << cmpi << "\n");
    return false;
  }
  if (!isValidIndex(cmpi.getRhs())) {
    LLVM_DEBUG(llvm::dbgs()
               << "illegal rhs: " << cmpi.getRhs() << " - " << cmpi << "\n");
    return false;
  }
  SmallVector<Value, 4> lhspack = {cmpi.getLhs()};
  if (!lhspack[0].getType().isa<IndexType>()) {
    auto op = b.create<IndexCastOp>(cmpi.getLoc(), lhspack[0],
                                    IndexType::get(cmpi.getContext()));
    lhspack[0] = op;
  }

  AffineMap rhsmap =
      AffineMap::get(0, 1, getAffineSymbolExpr(0, cmpi.getContext()));
  SmallVector<Value, 4> rhspack = {cmpi.getRhs()};
  if (!rhspack[0].getType().isa<IndexType>()) {
    auto op = b.create<IndexCastOp>(cmpi.getLoc(), rhspack[0],
                                    IndexType::get(cmpi.getContext()));
    rhspack[0] = op;
  }

  applies.push_back(
      b.create<mlir::AffineApplyOp>(cmpi.getLoc(), lhsmap, lhspack));
  applies.push_back(
      b.create<mlir::AffineApplyOp>(cmpi.getLoc(), rhsmap, rhspack));
  AffineExpr dims[2] = {b.getAffineDimExpr(2 * exprs.size() + 0),
                        b.getAffineDimExpr(2 * exprs.size() + 1)};
  switch (cmpi.getPredicate()) {
  case CmpIPredicate::eq:
    exprs.push_back(dims[0] - dims[1]);
    eqflags.push_back(true);
    break;

  case CmpIPredicate::sge:
    exprs.push_back(dims[0] - dims[1]);
    eqflags.push_back(false);
    break;

  case CmpIPredicate::sle:
    exprs.push_back(dims[1] - dims[0]);
    eqflags.push_back(false);
    break;

  case CmpIPredicate::sgt:
    exprs.push_back(dims[0] - dims[1] + 1);
    eqflags.push_back(false);
    break;

  case CmpIPredicate::slt:
    exprs.push_back(dims[1] - dims[0] - 1);
    eqflags.push_back(false);
    break;

  case CmpIPredicate::ne:
  case CmpIPredicate::ult:
  case CmpIPredicate::ule:
  case CmpIPredicate::ugt:
  case CmpIPredicate::uge:
    llvm_unreachable("unhandled icmp");
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
    auto map = rank ? rewriter.getMultiDimIdentityMap(rank)
                    : rewriter.getEmptyAffineMap();
    SmallVector<Value, 4> operands = load.getIndices();

    if (map.getNumInputs() != operands.size()) {
      load->getParentOfType<FuncOp>().dump();
      llvm::errs() << " load: " << load << "\n";
    }
    assert(map.getNumInputs() == operands.size());
    fully2ComposeAffineMapAndOperands(&map, &operands);
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
    auto map = rank ? rewriter.getMultiDimIdentityMap(rank)
                    : rewriter.getEmptyAffineMap();
    SmallVector<Value, 4> operands = store.getIndices();

    fully2ComposeAffineMapAndOperands(&map, &operands);
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
    fully2ComposeAffineMapAndOperands(&map, &operands);
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

    fully2ComposeAffineMapAndOperands(&lbMap, &lbOperands);
    canonicalizeMapAndOperands(&lbMap, &lbOperands);
    lbMap = removeDuplicateExprs(lbMap);

    fully2ComposeAffineMapAndOperands(&ubMap, &ubOperands);
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

    fully2ComposeIntegerSetAndOperands(&map, &operands);
    canonicalizeSetAndOperands(&map, &operands);

    // map(s).
    if (map == prevMap && !areChanged(operands, origOperands))
      return failure();

    op.setConditional(map, operands);

    return success();
  }
};

void AffineCFGPass::runOnFunction() {
  getFunction().walk([&](scf::IfOp ifOp) {
    if (inAffine(ifOp)) {
      OpBuilder b(ifOp);
      AffineIfOp affineIfOp;
      std::vector<mlir::Type> types;
      for (auto v : ifOp.results()) {
        types.push_back(v.getType());
      }

      SmallVector<AffineExpr, 2> exprs;
      SmallVector<bool, 2> eqflags;
      SmallVector<Value, 4> applies;

      std::deque<Value> todo = {ifOp.condition()};
      while (todo.size()) {
        auto cur = todo.front();
        todo.pop_front();
        if (auto cmpi = cur.getDefiningOp<CmpIOp>()) {
          if (!handle(b, cmpi, exprs, eqflags, applies)) {
            return;
          }
          continue;
        }
        if (auto andi = cur.getDefiningOp<AndIOp>()) {
          todo.push_back(andi.getOperand(0));
          todo.push_back(andi.getOperand(1));
          continue;
        }
        return;
      }

      auto iset = IntegerSet::get(/*dim*/ 2 * exprs.size(), /*symbol*/ 0, exprs,
                                  eqflags);
      affineIfOp = b.create<AffineIfOp>(ifOp.getLoc(), types, iset, applies,
                                        /*elseBlock=*/true);
      affineIfOp.thenRegion().takeBody(ifOp.thenRegion());
      affineIfOp.elseRegion().takeBody(ifOp.elseRegion());

      for (auto &blk : affineIfOp.thenRegion()) {
        if (auto yop = dyn_cast<scf::YieldOp>(blk.getTerminator())) {
          OpBuilder b(yop);
          b.create<AffineYieldOp>(yop.getLoc(), yop.results());
          yop.erase();
        }
      }
      for (auto &blk : affineIfOp.elseRegion()) {
        if (auto yop = dyn_cast<scf::YieldOp>(blk.getTerminator())) {
          OpBuilder b(yop);
          b.create<AffineYieldOp>(yop.getLoc(), yop.results());
          yop.erase();
        }
      }
      ifOp.replaceAllUsesWith(affineIfOp);
      ifOp.erase();
    }
  });

  /*
  getFunction().walk([](memref::StoreOp store) {
    if (!inAffine(store) && !llvm::all_of(store.getIndices(), [](mlir::Value V)
  { return V.getDefiningOp<mlir::ConstantOp>() != nullptr;
    }))
      return;
    if (!llvm::all_of(store.getIndices(),
                      [](Value index) { return isValidIndex(index); }))
      return;
    LLVM_DEBUG(llvm::dbgs() << "  affine store checks -> ok\n");
    SmallVector<Value, 2> newIndices;
    newIndices.reserve(store.getIndices().size());
    OpBuilder b(store);
    for (auto idx : store.getIndices()) {
      AffineMap idxmap =
          AffineMap::get(0, 1, getAffineSymbolExpr(0, idx.getContext()));
      if (!idx.getType().isa<IndexType>())
        idx = b.create<IndexCastOp>(
            idx.getLoc(), idx, IndexType::get(idx.getContext()));
      Value idxpack[1] = {idx};
      newIndices.push_back(
          b.create<mlir::AffineApplyOp>(idx.getLoc(), idxmap, idxpack));
    }
    replaceStore(store, newIndices);
  });
  getFunction().walk([](memref::LoadOp load) {
    if (!inAffine(load) && !llvm::all_of(load.getIndices(), [](mlir::Value V) {
      return V.getDefiningOp<mlir::ConstantOp>() != nullptr;
    }))
      return;
    if (!llvm::all_of(load.getIndices(),
                      [](Value index) { return isValidIndex(index); }))
      return;
    LLVM_DEBUG(llvm::dbgs() << "  affine load checks -> ok\n");
    SmallVector<Value, 2> newIndices;
    newIndices.reserve(load.getIndices().size());
    OpBuilder b(load);
    for (auto idx : load.getIndices()) {
      AffineMap idxmap =
          AffineMap::get(0, 1, getAffineSymbolExpr(0, idx.getContext()));
      if (!idx.getType().isa<IndexType>())
        idx = b.create<IndexCastOp>(
            idx.getLoc(), idx, IndexType::get(idx.getContext()));
      Value idxpack[1] = {idx};
      newIndices.push_back(
          b.create<mlir::AffineApplyOp>(idx.getLoc(), idxmap, idxpack));
    }
    replaceLoad(load, newIndices);
  });
  */

  {

    mlir::RewritePatternSet rpl(getFunction().getContext());
    rpl.add<SimplfyIntegerCastMath, CanonicalizeAffineApply,
            CanonicalizeIndexCast, IndexCastMovement, AffineFixup<AffineLoadOp>,
            AffineFixup<AffineStoreOp>, CanonicalizIfBounds, MoveStoreToAffine,
            MoveLoadToAffine, CanonicalieForBounds>(getFunction().getContext());
    GreedyRewriteConfig config;
    (void)applyPatternsAndFoldGreedily(getFunction().getOperation(),
                                       std::move(rpl), config);
  }
}

std::unique_ptr<OperationPass<FuncOp>> mlir::polygeist::replaceAffineCFGPass() {
  return std::make_unique<AffineCFGPass>();
}

#include "PassDetails.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "lower-polygeist-submap"

using namespace mlir;
using namespace polygeist;

namespace {

// Compose pure-dim-bearing polygeist.submap operands of a linalg.generic into
// the linalg's indexing_maps and switch the operands to the submap bases.
// This is done per-linalg.generic (rather than per-submap) so we can verify
// the resulting indexing_maps collectively cover every iter dim — otherwise
// linalg's shape-to-loops inference becomes ill-defined.
//
// Eligible submaps: numSymbols == 0 AND every result expression contains at
// least one DimExpr (allows `d0`, `d0 + const`, etc.; rejects pure-symbol or
// pure-constant slots). Symbol-bearing or constant-only forms are handled by
// the Subview/ExtractSlice patterns separately.
// Decompose a submap's affine map into a per-base-dim structure. Each base-
// dim is classified as either "live" (the view contributes data along this
// dim; passes through into the subview's result shape) or "dead" (the view
// reduces this base-dim to a single element via a fixed offset; subview
// rank-reduces it).
//
// Each result expression of submap.map must be one of:
//   d_i              → live, offset 0,           view_dim = d_i
//   d_i + const      → live, offset const,       view_dim = d_i
//   const + d_i      → live, offset const,       view_dim = d_i
//   d_i + symbol     → live, offset symbol,      view_dim = d_i
//   symbol + d_i     → live, offset symbol,      view_dim = d_i
//   symbol           → dead, offset symbol value
//   const            → dead, offset constant value
//
// The "live view_dim" tells the caller which iter-dim of the consumer linalg
// maps to this base-dim AFTER the subview rank-reduction. The offsets feed
// memref.subview's offsets. "dead" base-dims rank-reduce out — they don't
// appear in the consumer linalg's new indexing_map for this operand.
struct PerBaseDim {
  bool live;
  OpFoldResult offset;   // for !live, the fixed offset; for live, the base offset (0 or symbol/const)
  unsigned viewDim;      // only valid when live
};
struct DecomposedMap {
  SmallVector<PerBaseDim> base; // one per result of submap.map (= base rank)
};

static std::optional<DecomposedMap>
decomposeMapForLowering(AffineMap m, ValueRange symbols,
                        OpBuilder &builder) {
  DecomposedMap d;
  d.base.reserve(m.getNumResults());
  unsigned numDims = m.getNumDims();
  OpFoldResult zeroAttr = builder.getIndexAttr(0);
  for (unsigned k = 0; k < m.getNumResults(); ++k) {
    AffineExpr e = m.getResult(k);
    // Pure DimExpr.
    if (auto dim = e.dyn_cast<AffineDimExpr>()) {
      if (dim.getPosition() >= numDims) return std::nullopt;
      d.base.push_back(PerBaseDim{true, zeroAttr, dim.getPosition()});
      continue;
    }
    // Pure SymbolExpr.
    if (auto sym = e.dyn_cast<AffineSymbolExpr>()) {
      unsigned si = sym.getPosition();
      if (si >= symbols.size()) return std::nullopt;
      d.base.push_back(PerBaseDim{false, symbols[si], 0});
      continue;
    }
    // Pure ConstantExpr.
    if (auto c = e.dyn_cast<AffineConstantExpr>()) {
      d.base.push_back(PerBaseDim{false, builder.getIndexAttr(c.getValue()), 0});
      continue;
    }
    // AffineBinaryOpExpr: dim + (const|symbol).
    if (auto add = e.dyn_cast<AffineBinaryOpExpr>()) {
      if (add.getKind() != AffineExprKind::Add) return std::nullopt;
      AffineExpr lhs = add.getLHS(), rhs = add.getRHS();
      AffineExpr dimSide, offSide;
      if (lhs.isa<AffineDimExpr>()) {
        dimSide = lhs; offSide = rhs;
      } else if (rhs.isa<AffineDimExpr>()) {
        dimSide = rhs; offSide = lhs;
      } else {
        return std::nullopt;
      }
      auto dimExpr = dimSide.cast<AffineDimExpr>();
      if (dimExpr.getPosition() >= numDims) return std::nullopt;
      OpFoldResult off;
      if (auto c = offSide.dyn_cast<AffineConstantExpr>()) {
        off = builder.getIndexAttr(c.getValue());
      } else if (auto s = offSide.dyn_cast<AffineSymbolExpr>()) {
        unsigned si = s.getPosition();
        if (si >= symbols.size()) return std::nullopt;
        off = symbols[si];
      } else {
        return std::nullopt;
      }
      d.base.push_back(PerBaseDim{true, off, dimExpr.getPosition()});
      continue;
    }
    return std::nullopt;
  }
  return d;
}

// Returns true iff any base-dim has a non-zero static offset (signaling that
// a subview is structurally required because base.dim values can't directly
// serve as the iteration bound — they'd let the loop run past the original
// submap's smaller view).
static bool hasAnyNonZeroOffset(const DecomposedMap &d) {
  for (const auto &b : d.base) {
    if (!b.live) return true; // rank-reduced — needs subview
    if (auto attr = b.offset.dyn_cast<Attribute>())
      if (auto i = attr.dyn_cast<IntegerAttr>())
        if (i.getInt() != 0) return true;
    if (b.offset.is<Value>()) return true; // symbol offset — needs subview
  }
  return false;
}

static std::optional<int64_t> getConstantIndex(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  return std::nullopt;
}

static std::optional<SmallVector<int64_t>>
getStaticSizeOperands(ValueRange sizes) {
  SmallVector<int64_t> staticSizes;
  staticSizes.reserve(sizes.size());
  for (Value size : sizes) {
    auto c = getConstantIndex(size);
    if (!c)
      return std::nullopt;
    staticSizes.push_back(*c);
  }
  return staticSizes;
}

static bool accumulateLinearDimCoefficients(
    AffineExpr e, SmallVectorImpl<int64_t> &coeffs, int64_t &constant,
    int64_t scale = 1);

// Prove injectivity by enumerating a bounded static view domain.  This is an
// exact check for the small fixed-shape tensor-product views produced by the
// current MFEM corpus.  Large or dynamic domains deliberately return
// std::nullopt: callers must not turn "unknown" into an ordinary scatter.
static std::optional<bool>
isInjectiveOnStaticDomain(AffineMap map, ArrayRef<int64_t> sizes) {
  if (map.getNumDims() != sizes.size())
    return std::nullopt;

  constexpr int64_t kEnumerationLimit = 1'000'000;
  int64_t domainSize = 1;
  for (int64_t size : sizes) {
    if (size <= 0 || domainSize > kEnumerationLimit / size)
      return std::nullopt;
    domainSize *= size;
  }

  SmallVector<SmallVector<int64_t>> resultCoefficients;
  SmallVector<int64_t> resultConstants;
  resultCoefficients.reserve(map.getNumResults());
  resultConstants.reserve(map.getNumResults());
  for (AffineExpr result : map.getResults()) {
    SmallVector<int64_t> coefficients(map.getNumDims(), 0);
    int64_t constant = 0;
    if (!accumulateLinearDimCoefficients(result, coefficients, constant))
      return std::nullopt;
    resultCoefficients.push_back(std::move(coefficients));
    resultConstants.push_back(constant);
  }

  std::set<SmallVector<int64_t>> image;
  SmallVector<int64_t> coordinates(map.getNumDims(), 0);
  for (int64_t linear = 0; linear < domainSize; ++linear) {
    int64_t remaining = linear;
    for (int64_t dim = static_cast<int64_t>(sizes.size()) - 1; dim >= 0;
         --dim) {
      coordinates[dim] = remaining % sizes[dim];
      remaining /= sizes[dim];
    }

    SmallVector<int64_t> destination;
    destination.reserve(map.getNumResults());
    for (auto [coefficients, constant] :
         llvm::zip(resultCoefficients, resultConstants)) {
      int64_t value = constant;
      for (auto [coefficient, coordinate] :
           llvm::zip(coefficients, coordinates))
        value += coefficient * coordinate;
      destination.push_back(value);
    }
    if (!image.insert(std::move(destination)).second)
      return false;
  }
  return true;
}

static bool accumulateLinearDimCoefficients(AffineExpr e,
                                            SmallVectorImpl<int64_t> &coeffs,
                                            int64_t &constant,
                                            int64_t scale) {
  if (auto d = e.dyn_cast<AffineDimExpr>()) {
    unsigned pos = d.getPosition();
    if (pos >= coeffs.size())
      return false;
    coeffs[pos] += scale;
    return true;
  }
  if (auto c = e.dyn_cast<AffineConstantExpr>()) {
    constant += scale * c.getValue();
    return true;
  }
  if (auto bin = e.dyn_cast<AffineBinaryOpExpr>()) {
    if (bin.getKind() == AffineExprKind::Add) {
      return accumulateLinearDimCoefficients(bin.getLHS(), coeffs, constant,
                                             scale) &&
             accumulateLinearDimCoefficients(bin.getRHS(), coeffs, constant,
                                             scale);
    }
    if (bin.getKind() == AffineExprKind::Mul) {
      if (auto c = bin.getLHS().dyn_cast<AffineConstantExpr>())
        return accumulateLinearDimCoefficients(bin.getRHS(), coeffs, constant,
                                               scale * c.getValue());
      if (auto c = bin.getRHS().dyn_cast<AffineConstantExpr>())
        return accumulateLinearDimCoefficients(bin.getLHS(), coeffs, constant,
                                               scale * c.getValue());
    }
  }
  return false;
}

static bool parseSingleDimConstantStride(AffineExpr e, unsigned numDims,
                                         unsigned &dim, int64_t &stride,
                                         int64_t &offset) {
  SmallVector<int64_t> coeffs(numDims, 0);
  int64_t constant = 0;
  if (!accumulateLinearDimCoefficients(e, coeffs, constant))
    return false;
  int64_t seenDim = -1;
  for (auto it : llvm::enumerate(coeffs)) {
    if (it.value() == 0)
      continue;
    if (seenDim != -1)
      return false;
    seenDim = it.index();
    stride = it.value();
  }
  if (seenDim == -1 || stride <= 0)
    return false;
  dim = static_cast<unsigned>(seenDim);
  offset = constant;
  return true;
}

static bool isRowMajorLinearizedMap(AffineMap map,
                                    ArrayRef<int64_t> staticSizes) {
  if (map.getNumResults() != 1 || map.getNumDims() != staticSizes.size())
    return false;
  SmallVector<int64_t> coeffs(staticSizes.size(), 0);
  int64_t constant = 0;
  if (!accumulateLinearDimCoefficients(map.getResult(0), coeffs, constant))
    return false;
  if (constant != 0)
    return false;
  int64_t expectedStride = 1;
  for (int64_t i = static_cast<int64_t>(staticSizes.size()) - 1; i >= 0; --i) {
    if (staticSizes[i] <= 0 || coeffs[i] != expectedStride)
      return false;
    expectedStride *= staticSizes[i];
  }
  return true;
}

static bool isLeadingDimProjection(AffineMap map, unsigned projectedRank) {
  if (map.getNumResults() != projectedRank || map.getNumDims() < projectedRank)
    return false;
  for (unsigned i = 0; i < projectedRank; ++i) {
    auto dim = map.getResult(i).dyn_cast<AffineDimExpr>();
    if (!dim || dim.getPosition() != i)
      return false;
  }
  return true;
}

static int64_t product(ArrayRef<int64_t> values) {
  int64_t prod = 1;
  for (int64_t v : values)
    prod *= v;
  return prod;
}

static SmallVector<ReassociationIndices>
getSingleSourceReassociation(unsigned resultRank) {
  SmallVector<ReassociationIndices> reassociation(1);
  reassociation[0].reserve(resultRank);
  for (unsigned i = 0; i < resultRank; ++i)
    reassociation[0].push_back(i);
  return reassociation;
}

static std::optional<SmallVector<OpFoldResult>>
getMixedSizeOperands(ValueRange sizes, unsigned rank, OpBuilder &builder) {
  if (sizes.size() < rank)
    return std::nullopt;
  SmallVector<OpFoldResult> mixedSizes;
  mixedSizes.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) {
    if (auto c = getConstantIndex(sizes[i])) {
      mixedSizes.push_back(builder.getIndexAttr(*c));
      continue;
    }
    mixedSizes.push_back(sizes[i]);
  }
  return mixedSizes;
}

static bool sameTrailingOperands(ValueRange lhs, ValueRange rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [l, r] : llvm::zip(lhs, rhs))
    if (l != r)
      return false;
  return true;
}

static Value stripTensorCasts(Value v) {
  while (auto cast = v.getDefiningOp<tensor::CastOp>())
    v = cast.getSource();
  return v;
}

struct FoldIdentitySubmapInverse : public OpRewritePattern<SubmapInverseOp> {
  FoldIdentitySubmapInverse(MLIRContext *context)
      : OpRewritePattern<SubmapInverseOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(SubmapInverseOp inv,
                                PatternRewriter &rewriter) const final {
    auto submap = inv.getViewModified().getDefiningOp<SubmapOp>();
    if (!submap)
      return failure();
    if (stripTensorCasts(submap.getBase()) !=
        stripTensorCasts(inv.getBaseOriginal()))
      return failure();
    if (submap.getMap() != inv.getMap())
      return failure();
    if (!sameTrailingOperands(submap->getOperands().drop_front(1),
                              inv->getOperands().drop_front(2)))
      return failure();
    rewriter.replaceOp(inv, inv.getBaseOriginal());
    return success();
  }
};

// Rank-reduce a symbol-free, aliasing output submap of a linalg.generic.
// Unlike the slice-oriented pattern below, this also handles flattened
// strided maps such as
//   (d0,d1,d2,d3,d4) -> d3 + 64*d0 + 16*d1 + 4*d2
// by constructing an equivalent rank-4 output submap and changing the output
// indexing map to (d0,d1,d2,d3).  The dropped d4 remains a real Linalg
// reduction iterator, while the new submap is injective and can safely be
// scattered back after debufferization.
struct ComposeAffineSubmapIntoLinalgGeneric
    : public OpRewritePattern<linalg::GenericOp> {
  ComposeAffineSubmapIntoLinalgGeneric(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const final {
    SmallVector<AffineMap> maps(generic.getIndexingMapsArray());
    struct WorkItem {
      unsigned operandNumber;
      SubmapOp submap;
      AffineMap reducedSubmapMap;
      AffineMap outputIndexingMap;
      SmallVector<Value> sizes;
      MemRefType type;
    };
    SmallVector<WorkItem> work;
    unsigned numInputs = generic.getNumDpsInputs();
    unsigned numLoops = generic.getNumLoops();
    auto iteratorTypes = generic.getIteratorTypesArray();

    for (OpOperand &operand : generic->getOpOperands()) {
      // Keep input submaps as shaped views for now: their projected maps are
      // often what Linalg uses to infer reduction-loop bounds.  The semantic
      // bug addressed here is specifically an aliasing output submap.
      if (operand.getOperandNumber() < numInputs)
        continue;
      auto submap = operand.get().getDefiningOp<SubmapOp>();
      if (!submap || submap.getMap().getNumSymbols() != 0)
        continue;
      if (!isa<MemRefType>(submap.getBase().getType()) ||
          !isa<MemRefType>(submap.getType()))
        continue;
      if (submap.getMap().getNumResults() !=
          cast<MemRefType>(submap.getBase().getType()).getRank())
        continue;

      AffineMap operandMap = maps[operand.getOperandNumber()];
      if (operandMap.getNumSymbols() != 0 ||
          operandMap.getNumResults() != submap.getMap().getNumDims())
        continue;
      AffineMap composed = submap.getMap().compose(operandMap);

      // Recover which original view dimension supplies each loop bound.  A
      // projected permutation is sufficient for the canonical reduction
      // outputs produced by RaiseToLinalg and lets us carry the exact dynamic
      // size SSA values into the reduced view.
      SmallVector<int64_t> loopToView(numLoops, -1);
      for (auto [viewDim, expr] : llvm::enumerate(operandMap.getResults())) {
        auto dim = expr.dyn_cast<AffineDimExpr>();
        if (!dim || dim.getPosition() >= numLoops ||
            loopToView[dim.getPosition()] != -1)
          return rewriter.notifyMatchFailure(
              generic, "output indexing map is not a projected permutation");
        loopToView[dim.getPosition()] = viewDim;
      }

      // An output may omit loop dimensions only when those dimensions are
      // reductions.  Conversely, a reduction iterator must not select a
      // distinct output element.  This is the legality condition that turns
      // a non-injective memref view into a canonical Linalg reduction.
      SmallVector<bool> used(numLoops, false);
      for (AffineExpr result : composed.getResults())
        result.walk([&](AffineExpr expr) {
          if (auto dim = expr.dyn_cast<AffineDimExpr>())
            if (dim.getPosition() < numLoops)
              used[dim.getPosition()] = true;
        });

      SmallVector<unsigned> retainedLoops;
      for (unsigned dim = 0; dim < numLoops; ++dim) {
        bool isReduction =
            iteratorTypes[dim] == utils::IteratorType::reduction;
        if (used[dim] == isReduction)
          return rewriter.notifyMatchFailure(
              generic, "submap output collisions do not agree with "
                       "Linalg reduction iterators");
        if (!isReduction)
          retainedLoops.push_back(dim);
      }
      if (submap.getMap().getNumDims() == retainedLoops.size())
        continue;

      MLIRContext *ctx = generic.getContext();
      SmallVector<AffineExpr> loopReplacements(
          numLoops, getAffineConstantExpr(0, ctx));
      SmallVector<AffineExpr> outputResults;
      SmallVector<Value> reducedSizes;
      SmallVector<int64_t> reducedShape;
      ValueRange oldSizes = submap.getSizes();
      for (auto [newDim, loopDim] : llvm::enumerate(retainedLoops)) {
        if (loopToView[loopDim] < 0 ||
            static_cast<unsigned>(loopToView[loopDim]) >= oldSizes.size())
          return rewriter.notifyMatchFailure(
              generic, "cannot recover a size for a retained output loop");
        loopReplacements[loopDim] = getAffineDimExpr(newDim, ctx);
        outputResults.push_back(getAffineDimExpr(loopDim, ctx));
        Value size = oldSizes[loopToView[loopDim]];
        reducedSizes.push_back(size);
        auto constant = getConstantIndex(size);
        reducedShape.push_back(constant ? *constant : ShapedType::kDynamic);
      }

      SmallVector<AffineExpr> reducedBaseResults;
      for (AffineExpr result : composed.getResults())
        reducedBaseResults.push_back(
            result.replaceDimsAndSymbols(loopReplacements, {}));
      AffineMap reducedSubmapMap = AffineMap::get(
          retainedLoops.size(), 0, reducedBaseResults, ctx);
      AffineMap outputIndexingMap =
          AffineMap::get(numLoops, 0, outputResults, ctx);
      auto oldType = cast<MemRefType>(submap.getType());
      auto reducedType = MemRefType::get(reducedShape,
                                         oldType.getElementType());
      work.push_back({operand.getOperandNumber(), submap,
                      reducedSubmapMap, outputIndexingMap,
                      std::move(reducedSizes), reducedType});
    }
    if (work.empty())
      return failure();

    SmallVector<AffineMap> tentativeMaps(maps);
    for (WorkItem &item : work)
      tentativeMaps[item.operandNumber] = item.outputIndexingMap;

    // Keep Linalg loop-bound inference well-defined: every loop dimension
    // must remain represented by at least one operand after composition.
    SmallVector<bool> covered(numLoops, false);
    for (AffineMap map : tentativeMaps)
      for (AffineExpr result : map.getResults())
        result.walk([&](AffineExpr expr) {
          if (auto dim = expr.dyn_cast<AffineDimExpr>())
            if (dim.getPosition() < numLoops)
              covered[dim.getPosition()] = true;
        });
    if (!llvm::all_of(covered, [](bool value) { return value; }))
      return rewriter.notifyMatchFailure(
          generic, "submap composition loses a Linalg loop dimension");

    rewriter.setInsertionPoint(generic);
    for (WorkItem &item : work) {
      auto reducedSubmap = rewriter.create<SubmapOp>(
          item.submap.getLoc(), item.type, item.submap.getBase(), item.sizes,
          item.reducedSubmapMap);
      generic->setOperand(item.operandNumber, reducedSubmap.getResult());
    }
    generic.setIndexingMapsAttr(
        rewriter.getAffineMapArrayAttr(tentativeMaps));
    return success();
  }
};

// Rewrites a linalg.generic's submap-defined operands. For each operand
// defined by a polygeist.submap whose map decomposes via
// decomposeMapForLowering:
//   - Emit a memref.subview when needed (any offset is non-zero, or any
//     base-dim is rank-reduced/broadcast). The subview rank-reduces dead
//     base-dims and uses the offsets/sizes from the decomp.
//   - Compose the surviving live view-dims into the consumer linalg's
//     indexing_map for that operand: the new map's results are
//     (perm[live_0], perm[live_1], ...) in original-base-dim order. For
//     broadcasts (a view-dim doesn't appear in any live base-dim), the
//     consumer linalg simply omits that iter-dim from this operand's map.
struct ComposeSubmapIntoLinalgGeneric
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genOp,
                                PatternRewriter &rewriter) const final {
    SmallVector<AffineMap> newIndexingMaps(genOp.getIndexingMapsArray());
    struct WorkItem {
      unsigned operandIdx;
      SubmapOp submap;
      DecomposedMap decomp;
      bool needsSubview;
    };
    SmallVector<WorkItem> work;

    for (OpOperand &opd : genOp->getOpOperands()) {
      auto submap = opd.get().getDefiningOp<SubmapOp>();
      if (!submap) continue;
      // This rewrite changes a view operand to its base (or memref.subview)
      // and is only type-correct for the buffer form.  Tensor submaps must be
      // materialized by the tensor-specific patterns below; replacing a
      // ranked tensor output with its differently-ranked base would leave
      // the linalg result type inconsistent with its output operand.
      if (!isa<MemRefType>(submap.getType()) ||
          !isa<MemRefType>(submap.getBase().getType()))
        continue;
      auto decomp = decomposeMapForLowering(submap.getMap(),
                                             submap.getSymbols(),
                                             rewriter);
      if (!decomp) continue;
      work.push_back(WorkItem{opd.getOperandNumber(), submap, *decomp,
                              /*needsSubview=*/false});
    }
    if (work.empty()) return failure();

    // Decide which work items need a subview. A subview is needed for any
    // operand that has rank-reducing dead base-dims (broadcasts / fixed
    // offsets) or non-zero offsets. Additionally, if ANY operand in the
    // group needs one, force a subview for all of them so iter-bounds are
    // consistent across the linalg.
    bool anyNeeds = false;
    for (auto &w : work) {
      // Even at offset zero, replacing a smaller logical view with its full
      // base changes the loop bound (for example memref<2xf64> over
      // memref<10xf64>).  Preserve such extents with a real subview.
      if (hasAnyNonZeroOffset(w.decomp) ||
          w.submap.getType() != w.submap.getBase().getType()) {
        anyNeeds = true;
        break;
      }
    }
    for (auto &w : work)
      w.needsSubview = anyNeeds;

    // Build the new indexing_map for each operand upfront so we can
    // validate iter-dim coverage before any IR mutation. The new map's
    // results are, per live base-dim in order, d_(view_dim).
    MLIRContext *ctx = genOp.getContext();
    SmallVector<AffineMap> tentativeMaps(newIndexingMaps);
    for (auto &w : work) {
      SmallVector<AffineExpr> liveResults;
      for (const auto &b : w.decomp.base) {
        if (!b.live) continue;
        liveResults.push_back(getAffineDimExpr(b.viewDim, ctx));
      }
      AffineMap permMap = AffineMap::get(
          w.submap.getMap().getNumDims(), 0, liveResults, ctx);
      tentativeMaps[w.operandIdx] =
          permMap.compose(tentativeMaps[w.operandIdx]);
    }
    unsigned numIterDims = genOp.getNumLoops();
    SmallVector<bool, 8> dimCovered(numIterDims, false);
    for (AffineMap m : tentativeMaps) {
      for (AffineExpr e : m.getResults()) {
        e.walk([&](AffineExpr sub) {
          if (auto d = sub.dyn_cast<AffineDimExpr>())
            if (d.getPosition() < numIterDims)
              dimCovered[d.getPosition()] = true;
        });
      }
    }
    for (bool b : dimCovered)
      if (!b) return failure();

    // Apply the rewrite.
    for (auto &w : work) {
      Value newOperand;
      if (w.needsSubview) {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointAfter(w.submap);
        auto baseTy = cast<MemRefType>(w.submap.getBase().getType());
        ValueRange submapSizes = w.submap.getSizes();
        SmallVector<OpFoldResult> offsets, sizes, strides;
        OpFoldResult oneAttr = rewriter.getIndexAttr(1);
        SmallVector<int64_t> resultShape;
        for (const auto &b : w.decomp.base) {
          offsets.push_back(b.offset);
          if (b.live) {
            if (b.viewDim >= submapSizes.size()) return failure();
            sizes.push_back(submapSizes[b.viewDim]);
            resultShape.push_back(ShapedType::kDynamic);
          } else {
            sizes.push_back(oneAttr);
            // dead base-dim — gets rank-reduced.
          }
          strides.push_back(oneAttr);
        }
        MemRefType subTy = cast<MemRefType>(
            memref::SubViewOp::inferRankReducedResultType(
                resultShape, baseTy, offsets, sizes, strides));
        auto subview = rewriter.create<memref::SubViewOp>(
            w.submap.getLoc(), subTy, w.submap.getBase(), offsets, sizes,
            strides);
        newOperand = subview.getResult();
      } else {
        newOperand = w.submap.getBase();
      }
      genOp->setOperand(w.operandIdx, newOperand);
    }
    genOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(tentativeMaps));
    return success();
  }
};

// Lower polygeist.submap on a memref result, when the affine map has symbols,
// to an equivalent memref.subview. Each map result expression must be of one
// of the supported shapes:
//   - a pure DimExpr `d_k`            (identity slice on that view-dim)
//   - a pure SymbolExpr `s_k`         (fixed offset, rank-reduced dim)
//   - `s_k + d_j` (or `d_j + s_k`)    (offset + identity stride along view-dim j)
//
// More complex expressions (multiplications by constants, multiple symbols in
// one expression, etc.) are unsupported and the pattern fails. The current
// raise pass produces only these shapes for symbol-bearing submaps.
struct LowerSymbolBearingSubmapToSubview : public OpRewritePattern<SubmapOp> {
  using OpRewritePattern<SubmapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubmapOp submap,
                                PatternRewriter &rewriter) const final {
    AffineMap submapMap = submap.getMap();
    auto outTy = dyn_cast<MemRefType>(submap.getResult().getType());
    auto baseTy = dyn_cast<MemRefType>(submap.getBase().getType());
    if (!outTy || !baseTy) return failure();
    if (submapMap.getNumResults() != (unsigned)baseTy.getRank())
      return failure();
    // Skip cases ComposeSubmapIntoLinalgGeneric handles (pure DimExpr results
    // with no symbols). Anything with symbols, constants, or dim+constant
    // shifts falls here.
    bool anyNonPureDim = false;
    for (AffineExpr e : submapMap.getResults()) {
      if (!e.isa<AffineDimExpr>()) { anyNonPureDim = true; break; }
    }
    if (submapMap.getNumSymbols() == 0 && !anyNonPureDim) return failure();

    Location loc = submap.getLoc();
    ValueRange symbols = submap.getSymbols();
    ValueRange sizes = submap.getSizes();
    unsigned numViewDims = submapMap.getNumDims();

    // Parse each result expression of the submap's map. For each base-dim k,
    // determine (offset_k, size_k, stride_k) AND whether this base-dim is
    // contributed by a view-dim (i.e., it must appear in the output of the
    // subview) or is symbol-fixed (rank-reduced).
    SmallVector<OpFoldResult> offsets, subSizes, strides;
    // Track, for each view-dim, which base-dim it maps to (or -1).
    SmallVector<int64_t> viewDimToBaseDim(numViewDims, -1);

    OpFoldResult zeroAttr = rewriter.getIndexAttr(0);
    OpFoldResult oneAttr = rewriter.getIndexAttr(1);

    // Helper: classify each result expr into (offset, has-view-dim?, view-dim-idx).
    auto classify = [&](AffineExpr e, OpFoldResult &offset, bool &hasViewDim,
                        unsigned &viewDim) -> bool {
      // Pure SymbolExpr: fixed offset, no view-dim.
      if (auto s = e.dyn_cast<AffineSymbolExpr>()) {
        unsigned si = s.getPosition();
        if (si >= symbols.size()) return false;
        offset = symbols[si];
        hasViewDim = false;
        return true;
      }
      // Pure ConstantExpr: static offset, no view-dim.
      if (auto c = e.dyn_cast<AffineConstantExpr>()) {
        offset = rewriter.getIndexAttr(c.getValue());
        hasViewDim = false;
        return true;
      }
      // Pure DimExpr: identity slice, view-dim present, offset 0.
      if (auto d = e.dyn_cast<AffineDimExpr>()) {
        unsigned di = d.getPosition();
        if (di >= numViewDims) return false;
        offset = zeroAttr;
        hasViewDim = true;
        viewDim = di;
        return true;
      }
      // AffineBinaryOp Add: combinations of (Symbol|Constant) + Dim.
      if (auto add = e.dyn_cast<AffineBinaryOpExpr>()) {
        if (add.getKind() != AffineExprKind::Add) return false;
        AffineExpr lhs = add.getLHS();
        AffineExpr rhs = add.getRHS();
        AffineExpr dimSide;
        AffineExpr offExpr;
        if (lhs.isa<AffineDimExpr>()) {
          dimSide = lhs; offExpr = rhs;
        } else if (rhs.isa<AffineDimExpr>()) {
          dimSide = rhs; offExpr = lhs;
        } else {
          return false;
        }
        unsigned di = dimSide.cast<AffineDimExpr>().getPosition();
        if (di >= numViewDims) return false;
        // Offset side: must be a SymbolExpr or a ConstantExpr.
        if (auto s = offExpr.dyn_cast<AffineSymbolExpr>()) {
          unsigned si = s.getPosition();
          if (si >= symbols.size()) return false;
          offset = symbols[si];
        } else if (auto c = offExpr.dyn_cast<AffineConstantExpr>()) {
          offset = rewriter.getIndexAttr(c.getValue());
        } else {
          return false;
        }
        hasViewDim = true;
        viewDim = di;
        return true;
      }
      return false;
    };

    for (unsigned k = 0; k < submapMap.getNumResults(); ++k) {
      AffineExpr e = submapMap.getResult(k);
      OpFoldResult offset;
      bool hasViewDim;
      unsigned viewDim = 0;
      if (!classify(e, offset, hasViewDim, viewDim)) return failure();
      offsets.push_back(offset);
      if (hasViewDim) {
        if (viewDim >= sizes.size()) return failure();
        subSizes.push_back(sizes[viewDim]);
        strides.push_back(oneAttr);
        viewDimToBaseDim[viewDim] = k;
      } else {
        subSizes.push_back(oneAttr);
        strides.push_back(oneAttr);
      }
    }

    // Verify every view-dim is represented exactly once. If a view-dim isn't
    // represented in any output expression, this is a broadcast — handle in
    // a separate pass.
    for (unsigned j = 0; j < numViewDims; ++j)
      if (viewDimToBaseDim[j] == -1) return failure();

    // The output rank must equal the count of view-dim-bearing base-dims.
    // Otherwise the shape can't be expressed via a single rank-reducing
    // subview — bail.
    unsigned dimBearingBaseDims = 0;
    for (int64_t bk : viewDimToBaseDim)
      if (bk != -1) ++dimBearingBaseDims;
    if (dimBearingBaseDims != numViewDims) return failure();

    SmallVector<int64_t> resultShape(numViewDims, ShapedType::kDynamic);

    MemRefType inferredTy = cast<MemRefType>(
        memref::SubViewOp::inferRankReducedResultType(
            resultShape, baseTy, offsets, subSizes, strides));

    // A polygeist.submap's logical result type does not always encode the
    // physical stride selected from the base.  For example, fixing the last
    // component of an [..., 3] coordinate tensor and varying the preceding
    // dimension produces stride 3, which cannot be memref.cast to an
    // identity-layout memref.  Leave those maps in semantic submap form for
    // the general affine lowering instead of constructing invalid IR.
    if (inferredTy != outTy &&
        !memref::CastOp::areCastCompatible(inferredTy, outTy))
      return failure();

    Value sub = rewriter.create<memref::SubViewOp>(
        loc, inferredTy, submap.getBase(), offsets, subSizes, strides);

    // If the inferred type matches the submap's result type exactly, we can
    // RAUW. Otherwise we need a cast.
    if (sub.getType() == outTy) {
      rewriter.replaceOp(submap, sub);
      return success();
    }
    Value casted = rewriter.create<memref::CastOp>(loc, outTy, sub);
    rewriter.replaceOp(submap, casted);
    return success();
  }
};

// Tensor variant of polygeist.submap is handled by replacing with
// tensor.extract_slice (analogous to memref.subview).
struct LowerSymbolBearingSubmapToExtractSlice
    : public OpRewritePattern<SubmapOp> {
  using OpRewritePattern<SubmapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubmapOp submap,
                                PatternRewriter &rewriter) const final {
    AffineMap submapMap = submap.getMap();
    auto outTy = dyn_cast<RankedTensorType>(submap.getResult().getType());
    auto baseTy = dyn_cast<RankedTensorType>(submap.getBase().getType());
    if (!outTy || !baseTy) return failure();
    if (submapMap.getNumResults() != (unsigned)baseTy.getRank())
      return failure();

    SmallVector<SubmapInverseOp> identityInverseUsers;
    for (Operation *user : submap->getUsers()) {
      auto inv = dyn_cast<SubmapInverseOp>(user);
      if (!inv)
        continue;
      if (stripTensorCasts(inv.getBaseOriginal()) !=
          stripTensorCasts(submap.getBase()))
        continue;
      if (inv.getMap() != submap.getMap())
        continue;
      if (!sameTrailingOperands(submap->getOperands().drop_front(1),
                                inv->getOperands().drop_front(2)))
        continue;
      identityInverseUsers.push_back(inv);
    }
    if (!identityInverseUsers.empty()) {
      for (SubmapInverseOp inv : identityInverseUsers)
        rewriter.replaceOp(inv, submap.getBase());
      if (submap->use_empty())
        rewriter.eraseOp(submap);
      return success();
    }

    bool anyNonPureDim = false;
    for (AffineExpr e : submapMap.getResults()) {
      if (!e.isa<AffineDimExpr>()) { anyNonPureDim = true; break; }
    }
    if (submapMap.getNumSymbols() == 0 && !anyNonPureDim &&
        outTy.getRank() <= baseTy.getRank()) {
      // The stage pipeline frequently leaves exact identity views around
      // scratch tensors after a neighboring library launch has consumed the
      // inverse.  Eliminate the type-identical case directly so cleanup does
      // not depend on the broad canonicalizer (which may fold rank-expanding
      // DPS output views through linalg.generic incorrectly).
      if (submapMap.isIdentity() && submap.getBase().getType() == outTy) {
        rewriter.replaceOp(submap, submap.getBase());
        return success();
      }
      return failure();
    }

    Location loc = submap.getLoc();
    ValueRange symbols = submap.getSymbols();
    ValueRange sizes = submap.getSizes();
    unsigned numViewDims = submapMap.getNumDims();

    // Do not represent a rank-expanding DPS output view as tensor.expand_shape.
    // Canonicalizing that reshape through linalg.generic can replace the
    // output operand with its rank-1 base while leaving the generic's ranked
    // result unchanged.  The generic materialization path below preserves a
    // type-consistent logical output tensor and is also the right semantics
    // for reading an existing accumulator before an `Y += contraction` stage.
    bool isLinalgDpsOutput = llvm::any_of(
        submap->getUses(), [&](OpOperand &use) {
          auto generic = dyn_cast<linalg::GenericOp>(use.getOwner());
          return generic &&
                 use.getOperandNumber() >= generic.getNumDpsInputs();
        });

    if (!isLinalgDpsOutput && baseTy.getRank() == 1 &&
        outTy.getRank() == (int64_t)numViewDims) {
      auto staticSizes = getStaticSizeOperands(sizes);
      if (staticSizes && isRowMajorLinearizedMap(submapMap, *staticSizes) &&
          !baseTy.isDynamicDim(0) &&
          baseTy.getDimSize(0) == product(*staticSizes)) {
        int64_t flatSize = product(*staticSizes);
        // A row-major address map proves that the view itself is contiguous;
        // it does not prove that it covers the entire flat base.  Casting a
        // dynamic tensor<...> base to tensor<flatSize...> used to truncate
        // partial views (and could create an invalid tensor<100> ->
        // tensor<25> cast).  Use the reshape fast path only when full coverage
        // is statically proven; the generic materialization below handles
        // partial/dynamic bases.
        auto staticBaseTy =
            RankedTensorType::get({flatSize}, baseTy.getElementType());
        Value baseForExpand = submap.getBase();
        if (baseForExpand.getType() != staticBaseTy)
          baseForExpand =
              rewriter.create<tensor::CastOp>(loc, staticBaseTy, baseForExpand);

        auto staticOutTy =
            RankedTensorType::get(*staticSizes, baseTy.getElementType());
        auto reassociation = getSingleSourceReassociation(numViewDims);
        Value expanded = rewriter.create<tensor::ExpandShapeOp>(
            loc, staticOutTy, baseForExpand, reassociation);
        if (expanded.getType() != outTy)
          expanded = rewriter.create<tensor::CastOp>(loc, outTy, expanded);
        rewriter.replaceOp(submap, expanded);
        return success();
      }
    }

    if (submapMap.getNumSymbols() == 0 &&
        outTy.getRank() == (int64_t)numViewDims &&
        outTy.getRank() > baseTy.getRank()) {
      auto mixedSizes = getMixedSizeOperands(sizes, numViewDims, rewriter);
      if (mixedSizes) {
        Value empty = rewriter.create<tensor::EmptyOp>(
            loc, *mixedSizes, outTy.getElementType());
        AffineMap outMap =
            AffineMap::getMultiDimIdentityMap(numViewDims, submap.getContext());
        SmallVector<AffineMap> indexingMaps{submapMap, outMap};
        SmallVector<utils::IteratorType> iteratorTypes(
            numViewDims, utils::IteratorType::parallel);
        SmallVector<Type> resultTypes{empty.getType()};
        auto generic = rewriter.create<linalg::GenericOp>(
            loc, TypeRange(resultTypes), ValueRange{submap.getBase()},
            ValueRange{empty}, indexingMaps, iteratorTypes,
            [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
              nested.create<linalg::YieldOp>(nestedLoc, args[0]);
            });
        Value result = generic->getResult(0);
        if (result.getType() != outTy)
          result = rewriter.create<tensor::CastOp>(loc, outTy, result);
        rewriter.replaceOp(submap, result);
        return success();
      }
    }

    SmallVector<OpFoldResult> offsets, subSizes, strides;
    SmallVector<int64_t> viewDimToBaseDim(numViewDims, -1);
    OpFoldResult zeroAttr = rewriter.getIndexAttr(0);
    OpFoldResult oneAttr = rewriter.getIndexAttr(1);

    auto classify = [&](AffineExpr e, OpFoldResult &offset, bool &hasViewDim,
                        unsigned &viewDim, OpFoldResult &stride) -> bool {
      if (auto s = e.dyn_cast<AffineSymbolExpr>()) {
        unsigned si = s.getPosition();
        if (si >= symbols.size()) return false;
        offset = symbols[si];
        hasViewDim = false;
        stride = oneAttr;
        return true;
      }
      if (auto c = e.dyn_cast<AffineConstantExpr>()) {
        offset = rewriter.getIndexAttr(c.getValue());
        hasViewDim = false;
        stride = oneAttr;
        return true;
      }
      unsigned di;
      int64_t strideInt;
      int64_t offsetInt;
      if (parseSingleDimConstantStride(e, numViewDims, di, strideInt,
                                       offsetInt)) {
        offset = rewriter.getIndexAttr(offsetInt);
        hasViewDim = true;
        viewDim = di;
        stride = rewriter.getIndexAttr(strideInt);
        return true;
      }
      if (auto d = e.dyn_cast<AffineDimExpr>()) {
        unsigned di = d.getPosition();
        if (di >= numViewDims) return false;
        offset = zeroAttr;
        hasViewDim = true;
        viewDim = di;
        stride = oneAttr;
        return true;
      }
      if (auto add = e.dyn_cast<AffineBinaryOpExpr>()) {
        if (add.getKind() != AffineExprKind::Add) return false;
        AffineExpr lhs = add.getLHS(), rhs = add.getRHS();
        AffineExpr dimSide;
        AffineExpr offExpr;
        if (lhs.isa<AffineDimExpr>()) {
          dimSide = lhs; offExpr = rhs;
        } else if (rhs.isa<AffineDimExpr>()) {
          dimSide = rhs; offExpr = lhs;
        } else {
          return false;
        }
        unsigned di = dimSide.cast<AffineDimExpr>().getPosition();
        if (di >= numViewDims) return false;
        if (auto s = offExpr.dyn_cast<AffineSymbolExpr>()) {
          unsigned si = s.getPosition();
          if (si >= symbols.size()) return false;
          offset = symbols[si];
        } else if (auto c = offExpr.dyn_cast<AffineConstantExpr>()) {
          offset = rewriter.getIndexAttr(c.getValue());
        } else {
          return false;
        }
        hasViewDim = true;
        viewDim = di;
        stride = oneAttr;
        return true;
      }
      return false;
    };

    for (unsigned k = 0; k < submapMap.getNumResults(); ++k) {
      AffineExpr e = submapMap.getResult(k);
      OpFoldResult offset;
      bool hasViewDim;
      unsigned viewDim = 0;
      OpFoldResult stride = oneAttr;
      if (!classify(e, offset, hasViewDim, viewDim, stride)) return failure();
      offsets.push_back(offset);
      if (hasViewDim) {
        if (viewDim >= sizes.size()) return failure();
        subSizes.push_back(sizes[viewDim]);
        strides.push_back(stride);
        viewDimToBaseDim[viewDim] = k;
      } else {
        subSizes.push_back(oneAttr);
        strides.push_back(oneAttr);
      }
    }
    for (unsigned j = 0; j < numViewDims; ++j)
      if (viewDimToBaseDim[j] == -1) return failure();
    unsigned dimBearingBaseDims = 0;
    for (int64_t bk : viewDimToBaseDim)
      if (bk != -1) ++dimBearingBaseDims;
    if (dimBearingBaseDims != numViewDims) return failure();

    SmallVector<int64_t> resultShape(numViewDims, ShapedType::kDynamic);
    auto inferredTy = RankedTensorType::get(resultShape, baseTy.getElementType());
    Value sliced = rewriter.create<tensor::ExtractSliceOp>(
        loc, inferredTy, submap.getBase(), offsets, subSizes, strides);
    if (sliced.getType() == outTy) {
      rewriter.replaceOp(submap, sliced);
      return success();
    }
    Value casted = rewriter.create<tensor::CastOp>(loc, outTy, sliced);
    rewriter.replaceOp(submap, casted);
    return success();
  }
};

// Lower polygeist.submapInverse on tensors to tensor.insert_slice.
// For memref form, submapInverse is conceptually a no-op (modifications are
// already in place via the view) — we replace it with its base operand.
struct LowerSubmapInverse : public OpRewritePattern<SubmapInverseOp> {
  using OpRewritePattern<SubmapInverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubmapInverseOp inv,
                                PatternRewriter &rewriter) const final {
    Value base = inv.getBaseOriginal();
    Value view = inv.getViewModified();

    if (isa<MemRefType>(inv.getType())) {
      // For memref, the view's writes have already mutated the base. The
      // submapInverse simply returns the base.
      rewriter.replaceOp(inv, base);
      return success();
    }

    auto outTy = dyn_cast<RankedTensorType>(inv.getType());
    auto baseTy = dyn_cast<RankedTensorType>(base.getType());
    auto viewTy = dyn_cast<RankedTensorType>(view.getType());
    if (!outTy || !baseTy || !viewTy) return failure();

    AffineMap m = inv.getMap();
    if (m.getNumResults() != (unsigned)baseTy.getRank()) return failure();

    Location loc = inv.getLoc();
    ValueRange symbols = inv.getSymbols();
    unsigned numViewDims = m.getNumDims();
    ValueRange sizes =
        inv.getOperands().slice(m.getNumSymbols() + 2, numViewDims);
    auto staticSizes = getStaticSizeOperands(sizes);

    if (baseTy.getRank() == 1 && viewTy.getRank() == (int64_t)numViewDims) {
      if (staticSizes && isRowMajorLinearizedMap(m, *staticSizes) &&
          !baseTy.isDynamicDim(0) &&
          baseTy.getDimSize(0) == product(*staticSizes)) {
        auto staticViewTy =
            RankedTensorType::get(*staticSizes, viewTy.getElementType());
        Value viewForCollapse = view;
        if (viewForCollapse.getType() != staticViewTy)
          viewForCollapse =
              rewriter.create<tensor::CastOp>(loc, staticViewTy,
                                              viewForCollapse);

        int64_t flatSize = product(*staticSizes);
        // Contiguity is not full-base coverage.  In particular, writing the
        // first 32 elements of a dynamic 64-element application output must
        // retain the untouched suffix for a later +32 view.  Fall through to
        // affine write-back unless the base extent is statically identical.
        auto staticOutTy =
            RankedTensorType::get({flatSize}, viewTy.getElementType());
        auto reassociation = getSingleSourceReassociation(numViewDims);
        Value collapsed = rewriter.create<tensor::CollapseShapeOp>(
            loc, staticOutTy, viewForCollapse, reassociation);
        if (collapsed.getType() != outTy)
          collapsed = rewriter.create<tensor::CastOp>(loc, outTy, collapsed);
        rewriter.replaceOp(inv, collapsed);
        return success();
      }
    }

    bool projectedDimsAreUnit = false;
    if (staticSizes && numViewDims > (unsigned)baseTy.getRank()) {
      projectedDimsAreUnit = llvm::all_of(
          ArrayRef<int64_t>(*staticSizes).drop_front(baseTy.getRank()),
          [](int64_t size) { return size == 1; });
    }
    if (numViewDims > (unsigned)baseTy.getRank() && projectedDimsAreUnit &&
        viewTy.getRank() == (int64_t)numViewDims &&
        isLeadingDimProjection(m, baseTy.getRank())) {
      SmallVector<OpFoldResult> offsets, sliceSizes, strides;
      offsets.reserve(numViewDims);
      sliceSizes.reserve(numViewDims);
      strides.reserve(numViewDims);
      OpFoldResult zeroAttr = rewriter.getIndexAttr(0);
      OpFoldResult oneAttr = rewriter.getIndexAttr(1);
      for (unsigned i = 0; i < numViewDims; ++i) {
        offsets.push_back(zeroAttr);
        if (i < (unsigned)baseTy.getRank()) {
          if (i >= sizes.size())
            return failure();
          sliceSizes.push_back(sizes[i]);
        } else {
          sliceSizes.push_back(oneAttr);
        }
        strides.push_back(oneAttr);
      }
      SmallVector<int64_t> resultShape(baseTy.getRank(),
                                       ShapedType::kDynamic);
      auto sliceTy = RankedTensorType::get(resultShape, viewTy.getElementType());
      Value sliced = rewriter.create<tensor::ExtractSliceOp>(
          loc, sliceTy, view, offsets, sliceSizes, strides);
      if (sliced.getType() != outTy)
        sliced = rewriter.create<tensor::CastOp>(loc, outTy, sliced);
      rewriter.replaceOp(inv, sliced);
      return success();
    }

    // A general affine submap is not necessarily a rectangular slice.  For
    // example, a rank-4 view of one component of a flattened vector may have
    // row-major strides within each element but a gap between elements.  Such
    // maps cannot be represented by tensor.insert_slice after collapsing the
    // view.  Preserve their exact semantics with an elementwise affine
    // write-back.  One-shot bufferization subsequently turns the loop-carried
    // tensor into indexed stores, so this is a correctness fallback rather
    // than a new runtime abstraction.
    auto lowerElementwiseAffineWriteback = [&]() -> LogicalResult {
      if (viewTy.getRank() != (int64_t)numViewDims ||
          sizes.size() != numViewDims)
        return failure();
      if (!staticSizes)
        return rewriter.notifyMatchFailure(
            inv, "affine write-back injectivity is unknown for dynamic sizes");
      std::optional<bool> isInjective =
          isInjectiveOnStaticDomain(m, *staticSizes);
      if (!isInjective || !*isInjective)
        return rewriter.notifyMatchFailure(
            inv, "affine write-back map is not proven injective");

      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      SmallVector<Value> inductionVars;

      auto emitLoopNest = [&](auto &&self, unsigned depth,
                              Value destination) -> Value {
        auto loop = rewriter.create<scf::ForOp>(
            loc, zero, sizes[depth], one, ValueRange{destination});
        if (!loop.getBody()->empty())
          rewriter.eraseOp(loop.getBody()->getTerminator());

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());
        inductionVars.push_back(loop.getInductionVar());

        Value updated;
        if (depth + 1 < numViewDims) {
          updated = self(self, depth + 1, loop.getRegionIterArg(0));
        } else {
          Value element = rewriter.create<tensor::ExtractOp>(
              loc, view, inductionVars);
          SmallVector<Value> mapOperands(inductionVars.begin(),
                                         inductionVars.end());
          mapOperands.append(symbols.begin(), symbols.end());
          SmallVector<Value> baseIndices;
          baseIndices.reserve(m.getNumResults());
          for (AffineExpr resultExpr : m.getResults()) {
            AffineMap resultMap = AffineMap::get(
                m.getNumDims(), m.getNumSymbols(), resultExpr,
                rewriter.getContext());
            baseIndices.push_back(rewriter.create<affine::AffineApplyOp>(
                loc, resultMap, mapOperands));
          }
          updated = rewriter.create<tensor::InsertOp>(
              loc, element, loop.getRegionIterArg(0), baseIndices);
        }

        inductionVars.pop_back();
        rewriter.create<scf::YieldOp>(loc, updated);
        return loop.getResult(0);
      };

      if (numViewDims == 0)
        return failure();
      Value result = emitLoopNest(emitLoopNest, 0, base);
      rewriter.replaceOp(inv, result);
      return success();
    };

    SmallVector<OpFoldResult> offsets, subSizes, strides;
    SmallVector<int64_t> viewDimSeen(numViewDims, 0);
    OpFoldResult zeroAttr = rewriter.getIndexAttr(0);
    OpFoldResult oneAttr = rewriter.getIndexAttr(1);

    auto classify = [&](AffineExpr e, OpFoldResult &offset, bool &hasViewDim,
                        unsigned &viewDim, OpFoldResult &stride) -> bool {
      if (auto s = e.dyn_cast<AffineSymbolExpr>()) {
        unsigned si = s.getPosition();
        if (si >= symbols.size()) return false;
        offset = symbols[si];
        hasViewDim = false;
        stride = oneAttr;
        return true;
      }
      if (auto c = e.dyn_cast<AffineConstantExpr>()) {
        offset = rewriter.getIndexAttr(c.getValue());
        hasViewDim = false;
        stride = oneAttr;
        return true;
      }
      unsigned di;
      int64_t strideInt;
      int64_t offsetInt;
      if (parseSingleDimConstantStride(e, numViewDims, di, strideInt,
                                       offsetInt)) {
        offset = rewriter.getIndexAttr(offsetInt);
        hasViewDim = true;
        viewDim = di;
        stride = rewriter.getIndexAttr(strideInt);
        return true;
      }
      if (auto d = e.dyn_cast<AffineDimExpr>()) {
        unsigned di = d.getPosition();
        if (di >= numViewDims) return false;
        offset = zeroAttr;
        hasViewDim = true;
        viewDim = di;
        stride = oneAttr;
        return true;
      }
      if (auto add = e.dyn_cast<AffineBinaryOpExpr>()) {
        if (add.getKind() != AffineExprKind::Add) return false;
        AffineExpr lhs = add.getLHS(), rhs = add.getRHS();
        AffineExpr dimSide;
        AffineExpr offExpr;
        if (lhs.isa<AffineDimExpr>()) {
          dimSide = lhs; offExpr = rhs;
        } else if (rhs.isa<AffineDimExpr>()) {
          dimSide = rhs; offExpr = lhs;
        } else {
          return false;
        }
        unsigned di = dimSide.cast<AffineDimExpr>().getPosition();
        if (di >= numViewDims) return false;
        if (auto s = offExpr.dyn_cast<AffineSymbolExpr>()) {
          unsigned si = s.getPosition();
          if (si >= symbols.size()) return false;
          offset = symbols[si];
        } else if (auto c = offExpr.dyn_cast<AffineConstantExpr>()) {
          offset = rewriter.getIndexAttr(c.getValue());
        } else {
          return false;
        }
        hasViewDim = true;
        viewDim = di;
        stride = oneAttr;
        return true;
      }
      return false;
    };

    for (unsigned k = 0; k < m.getNumResults(); ++k) {
      AffineExpr e = m.getResult(k);
      OpFoldResult offset;
      bool hasViewDim;
      unsigned viewDim = 0;
      OpFoldResult stride = oneAttr;
      if (!classify(e, offset, hasViewDim, viewDim, stride))
        return lowerElementwiseAffineWriteback();
      offsets.push_back(offset);
      if (hasViewDim) {
        if (viewDim >= sizes.size())
          return lowerElementwiseAffineWriteback();
        subSizes.push_back(sizes[viewDim]);
        strides.push_back(stride);
        viewDimSeen[viewDim] = 1;
      } else {
        subSizes.push_back(oneAttr);
        strides.push_back(oneAttr);
      }
    }
    for (unsigned j = 0; j < numViewDims; ++j)
      if (!viewDimSeen[j])
        return lowerElementwiseAffineWriteback();

    // If the view's rank differs from the slice's rank (because of symbol-
    // only base-dims that rank-reduced on the way in), we need to reshape
    // the view to match. For now we only support the case where view's rank
    // equals the count of dim-bearing base-dims.
    unsigned numDimBearingBaseDims = 0;
    for (unsigned k = 0; k < m.getNumResults(); ++k)
      if (!m.getResult(k).isa<AffineSymbolExpr>())
        ++numDimBearingBaseDims;
    if (numDimBearingBaseDims != (unsigned)viewTy.getRank())
      return lowerElementwiseAffineWriteback();

    Value result = rewriter.create<tensor::InsertSliceOp>(
        loc, view, base, offsets, subSizes, strides);
    rewriter.replaceOp(inv, result);
    return success();
  }
};

struct LowerPolygeistSubmapPass
    : public mlir::polygeist::LowerPolygeistSubmapBase<
          LowerPolygeistSubmapPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldIdentitySubmapInverse,
                 ComposeAffineSubmapIntoLinalgGeneric,
                 ComposeSubmapIntoLinalgGeneric,
                 LowerSymbolBearingSubmapToSubview,
                 LowerSymbolBearingSubmapToExtractSlice,
                 LowerSubmapInverse>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns)))) {
      // Some submaps remain — caller may want to know but it's not fatal.
    }
  }
};

} // anonymous namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createLowerPolygeistSubmapPass() {
  return std::make_unique<LowerPolygeistSubmapPass>();
}
} // namespace polygeist
} // namespace mlir

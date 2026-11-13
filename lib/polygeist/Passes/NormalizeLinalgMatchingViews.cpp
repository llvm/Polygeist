//===- NormalizeLinalgMatchingViews.cpp - Expose collapsible iteration spaces ===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.

#include "PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Ops.h"

using namespace mlir;
using namespace mlir::polygeist;

namespace {

// Reading exactly the just-written injective view recovers its SSA value.
// Keep differing bounds and noninjective maps explicit: those are not the
// same tensor, even when their printed element types happen to agree.
struct ForwardUpdatedSubview : OpRewritePattern<SubmapOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SubmapOp op,
                                PatternRewriter &rewriter) const override {
    auto inverse = op.getBase().getDefiningOp<SubmapInverseOp>();
    if (!inverse || !op.getMap().isPermutation() ||
        op.getMap() != inverse.getMap() ||
        op.getType() != inverse.getViewModified().getType() ||
        op->getOperands().drop_front(1) != inverse->getOperands().drop_front(2))
      return failure();
    rewriter.replaceOp(op, inverse.getViewModified());
    return success();
  }
};

// A later write to exactly the same injective view supersedes the earlier
// write. Keep dependencies through the new value itself; only its destination
// base no longer needs the overwritten intermediate contents.
struct DropOverwrittenSubview : OpRewritePattern<SubmapInverseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SubmapInverseOp op,
                                PatternRewriter &rewriter) const override {
    auto prior = op.getBaseOriginal().getDefiningOp<SubmapInverseOp>();
    if (!prior || !op.getMap().isPermutation() ||
        op.getMap() != prior.getMap() ||
        op.getViewModified().getType() != prior.getViewModified().getType() ||
        op->getOperands().drop_front(2) != prior->getOperands().drop_front(2))
      return failure();
    rewriter.updateRootInPlace(op, [&] { op->setOperand(0, prior.getBaseOriginal()); });
    return success();
  }
};

// A scalar initializer has no element computation or reads of its destination.
// This is the generic form of a fill; its scalar value is not inspected.
static bool isScalarInitializer(Value value) {
  auto op = value.getDefiningOp<linalg::GenericOp>();
  if (!op || op.getNumDpsInputs() || op.getNumDpsInits() != 1 ||
      op.getNumResults() != 1 || op.getNumParallelLoops() != op.getNumLoops() ||
      !op.getIndexingMapsArray().front().isPermutation() ||
      op.getBody()->getOperations().size() != 1)
    return false;
  Value yielded = op.getBody()->getTerminator()->getOperand(0);
  auto argument = dyn_cast<BlockArgument>(yielded);
  return !argument || argument.getOwner() != op.getBody();
}

// A pointwise consumer may read an initializer through a DPS input after
// upstream removes an unused output dependence. Fuse that captured scalar
// using upstream elementwise fusion, just as tiled output initializers are
// fused through the upstream SCF interface below.
struct FuseScalarInitializerInput : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    for (OpOperand &operand : op->getOpOperands()) {
      if (!isScalarInitializer(operand.get()) ||
          !linalg::areElementwiseOpsFusable(&operand))
        continue;
      Operation *producer = operand.get().getDefiningOp();
      auto fusion = linalg::fuseElementwiseOps(rewriter, &operand);
      if (failed(fusion))
        return failure();
      for (auto [original, replacement] : fusion->replacements)
        rewriter.replaceUsesWithIf(original, replacement, [&](OpOperand &use) {
          return use.get().getDefiningOp() != producer;
        });
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// A rank-zero initializer writes exactly one scalar, independent of its old
// destination value. Keep the seed as an ordinary SSA operand.
struct NormalizeScalarInitializer : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1 || op.getNumLoops() != 0 ||
        !isScalarInitializer(op.getResult(0)))
      return failure();
    auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!type || type.getRank() != 0)
      return failure();
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(
        op, op.getBody()->getTerminator()->getOperand(0),
        op.getDpsInitOperand(0)->get(), ValueRange{});
    return success();
  }
};

// Scalar views have no iteration coordinates. Compare every destination
// coordinate exactly, retaining unsupported affine expressions unchanged.
static bool sameScalarCoordinates(AffineMap map, ValueRange symbols,
                                  ValueRange indices) {
  if (map.getNumDims() || map.getNumResults() != indices.size() ||
      map.getNumSymbols() != symbols.size())
    return false;
  for (auto [expr, index] : llvm::zip(map.getResults(), indices)) {
    APInt indexConstant;
    bool constantIndex = matchPattern(index, m_ConstantInt(&indexConstant));
    if (auto value = expr.dyn_cast<AffineConstantExpr>()) {
      if (!constantIndex || indexConstant.getSExtValue() != value.getValue())
        return false;
    } else if (auto value = expr.dyn_cast<AffineSymbolExpr>()) {
      Value symbol = symbols[value.getPosition()];
      if (symbol == index)
        continue;
      APInt symbolConstant;
      if (!constantIndex || !matchPattern(symbol, m_ConstantInt(&symbolConstant)) ||
          symbolConstant != indexConstant)
        return false;
    } else {
      return false;
    }
  }
  return true;
}

struct ForwardScalarInsertedView : OpRewritePattern<SubmapOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SubmapOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    auto insert = op.getBase().getDefiningOp<tensor::InsertOp>();
    if (!type || type.getRank() || !insert ||
        !sameScalarCoordinates(op.getMap(), op.getSymbols(), insert.getIndices()))
      return failure();
    Value empty = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), ArrayRef<int64_t>{}, type.getElementType());
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(
        op, insert.getScalar(), empty, ValueRange{});
    return success();
  }
};

struct DropOverwrittenScalarInsert : OpRewritePattern<SubmapInverseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SubmapInverseOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getViewModified().getType());
    auto insert = op.getBaseOriginal().getDefiningOp<tensor::InsertOp>();
    if (!type || type.getRank() || !insert ||
        !sameScalarCoordinates(op.getMap(), op.getSymbols(), insert.getIndices()))
      return failure();
    rewriter.updateRootInPlace(op, [&] { op->setOperand(0, insert.getDest()); });
    return success();
  }
};

// A rank-zero source inserted into unit extents is a write to one coordinate.
// Express that same point using the view representation used by source raising.
struct NormalizeScalarSliceWrite : OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceType = cast<RankedTensorType>(op.getSource().getType());
    if (sourceType.getRank() ||
        !llvm::all_of(op.getStaticSizes(), [](int64_t v) { return v == 1; }) ||
        !llvm::all_of(op.getStaticStrides(), [](int64_t v) { return v == 1; }))
      return failure();
    SmallVector<Value> symbols;
    SmallVector<AffineExpr> coordinates;
    for (OpFoldResult offset : op.getMixedOffsets()) {
      if (auto value = offset.dyn_cast<Attribute>()) {
        coordinates.push_back(rewriter.getAffineConstantExpr(
            cast<IntegerAttr>(value).getInt()));
      } else {
        coordinates.push_back(rewriter.getAffineSymbolExpr(symbols.size()));
        symbols.push_back(offset.get<Value>());
      }
    }
    auto map = AffineMap::get(0, symbols.size(), coordinates, op.getContext());
    rewriter.replaceOpWithNewOp<SubmapInverseOp>(
        op, op.getType(), op.getDest(), op.getSource(), symbols, map);
    return success();
  }
};

// Preserve explicit submap bounds as static tensor dimensions where known.
// Multiple-output operations need all result types updated together; cast
// external users back to their original types without changing any indexing.
struct RefineMultiOutputBounds : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics() || op.getNumResults() < 2 ||
        op.getNumResults() != op.getNumDpsInits())
      return failure();
    SmallVector<Value> inputs(op.getInputs()), outputs(op.getOutputs());
    SmallVector<Type> resultTypes(op.getResultTypes());
    struct Refinement {
      unsigned operand;
      SubmapOp view;
      RankedTensorType type;
    };
    SmallVector<Refinement> refinements;
    for (OpOperand &operand : op->getOpOperands()) {
      auto view = operand.get().getDefiningOp<SubmapOp>();
      if (!view)
        continue;
      auto type = dyn_cast<RankedTensorType>(view.getType());
      if (!type || view.getSizes().size() != unsigned(type.getRank()))
        continue;
      SmallVector<int64_t> shape(type.getShape());
      bool refine = false;
      for (auto [i, size] : llvm::enumerate(view.getSizes())) {
        APInt constant;
        if (!matchPattern(size, m_ConstantInt(&constant)))
          continue;
        int64_t value = constant.getSExtValue();
        if (value < 0 || (!type.isDynamicDim(i) && shape[i] != value))
          return failure();
        refine |= type.isDynamicDim(i);
        shape[i] = value;
      }
      if (!refine)
        continue;
      refinements.push_back({operand.getOperandNumber(), view,
          RankedTensorType::get(shape, type.getElementType(), type.getEncoding())});
    }
    if (refinements.empty())
      return failure();
    // Validate every operand before creating replacement operations.
    for (const Refinement &refine : refinements) {
      auto view = refine.view;
      Value value = rewriter.create<SubmapOp>(
          view.getLoc(), refine.type, view.getBase(),
          view->getOperands().drop_front(1), view.getMap());
      if (refine.operand < inputs.size())
        inputs[refine.operand] = value;
      else {
        unsigned output = refine.operand - inputs.size();
        outputs[output] = value;
        resultTypes[output] = refine.type;
      }
    }
    auto replacement = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultTypes, inputs, outputs, op.getIndexingMapsArray(),
        op.getIteratorTypesArray(), rewriter.getStringAttr(""),
        rewriter.getStringAttr(""));
    rewriter.cloneRegionBefore(op.getRegion(), replacement.getRegion(),
                               replacement.getRegion().end());
    SmallVector<Value> results;
    for (auto [old, value] : llvm::zip(op.getResults(), replacement.getResults())) {
      Value restored = value;
      if (old.getType() != value.getType())
        restored = rewriter.create<tensor::CastOp>(op.getLoc(), old.getType(), value);
      results.push_back(restored);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

// Put broadcast/projection semantics on the Linalg indexing maps. Preserve a
// bounded identity submap rather than replacing a partial view by its base:
// the latter changes the iteration domain. No scalar computation is inspected.
struct ExposeProjectedSubmaps : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDpsInits() != 1 || op.getNumResults() != 1)
      return failure();
    SmallVector<AffineMap> maps(op.getIndexingMapsArray());
    struct View {
      unsigned operand;
      SubmapOp submap;
      SmallVector<Value> sizes;
      RankedTensorType type;
    };
    SmallVector<View> views;
    SubmapInverseOp inverse;
    auto iterators = op.getIteratorTypesArray();
    SmallVector<Value> extentValues(op.getNumLoops());
    SmallVector<int64_t> extentConstants(op.getNumLoops(), ShapedType::kDynamic);
    SmallVector<bool> extentSeen(op.getNumLoops(), false);
    SmallVector<std::pair<unsigned, unsigned>> extentOrigins(op.getNumLoops());
    for (OpOperand &operand : op->getOpOperands()) {
      auto type = dyn_cast<RankedTensorType>(operand.get().getType());
      if (!type)
        return failure();
      auto view = operand.get().getDefiningOp<SubmapOp>();
      for (auto [position, expr] :
           llvm::enumerate(maps[operand.getOperandNumber()].getResults())) {
        auto dim = expr.dyn_cast<AffineDimExpr>();
        if (!dim || extentSeen[dim.getPosition()])
          continue;
        unsigned loop = dim.getPosition();
        extentSeen[loop] = true;
        extentOrigins[loop] = {operand.getOperandNumber(), position};
        extentConstants[loop] = type.getDimSize(position);
        if (view) {
          extentValues[loop] = view.getSizes()[position];
          APInt constant;
          if (matchPattern(extentValues[loop], m_ConstantInt(&constant)))
            extentConstants[loop] = constant.getSExtValue();
        }
      }
    }
    for (OpOperand &operand : op->getOpOperands()) {
      auto submap = operand.get().getDefiningOp<SubmapOp>();
      if (!submap || !isa<RankedTensorType>(submap.getBase().getType()) ||
          !isa<RankedTensorType>(submap.getType()) ||
          !submap.getMap().isProjectedPermutation())
        continue;
      unsigned number = operand.getOperandNumber();
      AffineMap composed = submap.getMap().compose(maps[number]);
      if (!composed.isProjectedPermutation())
        continue;
      SmallVector<Value> sizes;
      SmallVector<int64_t> shape;
      for (AffineExpr e : submap.getMap().getResults()) {
        Value size = submap.getSizes()[e.cast<AffineDimExpr>().getPosition()];
        APInt constant;
        shape.push_back(matchPattern(size, m_ConstantInt(&constant))
                            ? constant.getSExtValue() : ShapedType::kDynamic);
        sizes.push_back(size);
      }
      auto type = RankedTensorType::get(
          shape, cast<RankedTensorType>(submap.getType()).getElementType());
      // Projection must not change which extent supplies a loop bound. A
      // broadcast may have been the first operand mentioning that loop.
      bool sameDomain = true;
      for (auto [position, expr] : llvm::enumerate(composed.getResults())) {
        unsigned loop = expr.cast<AffineDimExpr>().getPosition();
        sameDomain &= (sizes[position] == extentValues[loop]) ||
                      (shape[position] != ShapedType::kDynamic &&
                       shape[position] == extentConstants[loop]);
      }
      if (!sameDomain)
        continue;
      if (submap.getMap().isIdentity() && type == submap.getType())
        continue;
      if (number == op.getNumDpsInputs()) {
        if (!op.getResult(0).hasOneUse())
          continue;
        auto candidate = dyn_cast<SubmapInverseOp>(
            *op.getResult(0).getUsers().begin());
        if (!candidate || candidate.getViewModified() != op.getResult(0) ||
            candidate.getBaseOriginal() != submap.getBase() ||
            candidate.getMap() != submap.getMap() ||
            candidate->getOperands().drop_front(2) !=
                submap->getOperands().drop_front(1))
          continue;
        // Every parallel loop must write a distinct output; precisely the
        // reduction dimensions may be absent from its physical indexing map.
        SmallVector<bool> used(op.getNumLoops(), false);
        for (AffineExpr e : composed.getResults())
          used[e.cast<AffineDimExpr>().getPosition()] = true;
        bool valid = true;
        for (unsigned i = 0; i < used.size(); ++i)
          valid &= used[i] == (iterators[i] == utils::IteratorType::parallel);
        if (!valid)
          continue;
        inverse = candidate;
      }
      maps[number] = composed;
      views.push_back({number, submap, sizes, type});
    }
    if (views.empty() || !inversePermutation(concatAffineMaps(maps)))
      return failure();

    // Check the new first occurrence of every loop dimension, including loops
    // dropped from a broadcast but still present in an unchanged operand.
    SmallVector<bool> checked(op.getNumLoops(), false);
    for (OpOperand &operand : op->getOpOperands()) {
      unsigned number = operand.getOperandNumber();
      auto updated = llvm::find_if(views, [&](const View &v) {
        return v.operand == number;
      });
      for (auto [position, expr] : llvm::enumerate(maps[number].getResults())) {
        auto dim = expr.dyn_cast<AffineDimExpr>();
        if (!dim || checked[dim.getPosition()])
          continue;
        unsigned loop = dim.getPosition();
        checked[loop] = true;
        if (updated == views.end() &&
            extentOrigins[loop] == std::make_pair(number, unsigned(position)))
          continue;
        Value value;
        int64_t constant = ShapedType::kDynamic;
        if (updated != views.end()) {
          value = updated->sizes[position];
          constant = updated->type.getDimSize(position);
        } else {
          constant = cast<RankedTensorType>(operand.get().getType()).getDimSize(position);
          if (auto view = operand.get().getDefiningOp<SubmapOp>()) {
            value = view.getSizes()[position];
            APInt integer;
            if (matchPattern(value, m_ConstantInt(&integer)))
              constant = integer.getSExtValue();
          }
        }
        if ((!value || value != extentValues[loop]) &&
            (constant == ShapedType::kDynamic || constant != extentConstants[loop]))
          return failure();
      }
    }

    SmallVector<Value> inputs(op.getInputs()), outputs(op.getOutputs());
    for (View &view : views) {
      auto identity = rewriter.getMultiDimIdentityMap(view.type.getRank());
      Value replacement = rewriter.create<SubmapOp>(
          op.getLoc(), view.type, view.submap.getBase(), view.sizes, identity);
      if (view.operand < inputs.size())
        inputs[view.operand] = replacement;
      else
        outputs[0] = replacement;
    }
    auto replacement = rewriter.create<linalg::GenericOp>(
        op.getLoc(), TypeRange{outputs[0].getType()}, inputs, outputs, maps,
        iterators, rewriter.getStringAttr(""), rewriter.getStringAttr(""));
    rewriter.cloneRegionBefore(op.getRegion(), replacement.getRegion(),
                               replacement.getRegion().end());
    if (inverse) {
      const View &view = views.back();
      rewriter.setInsertionPoint(inverse);
      auto restored = rewriter.create<SubmapInverseOp>(
          inverse.getLoc(), inverse.getType(), inverse.getBaseOriginal(),
          replacement.getResult(0), view.sizes,
          rewriter.getMultiDimIdentityMap(view.type.getRank()));
      rewriter.replaceOp(inverse, restored.getResult());
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, replacement.getResults());
    }
    return success();
  }
};

// A group must occur consecutively and in order in every operand that uses it.
// In particular, an operand indexed by only one member blocks the collapse.
// Upstream performs the remaining legality checks and rewrites linalg.index.
static SmallVector<ReassociationIndices>
parallelGroups(linalg::GenericOp op) {
  auto iterators = op.getIteratorTypesArray();
  auto maps = op.getIndexingMapsArray();
  auto compatible = [&](int64_t left, int64_t right) {
    if (iterators[left] != utils::IteratorType::parallel ||
        iterators[right] != utils::IteratorType::parallel)
      return false;
    for (AffineMap map : maps) {
      if (!map.isProjectedPermutation())
        return false;
      int64_t a = -1, b = -1;
      for (auto [i, e] : llvm::enumerate(map.getResults())) {
        unsigned dim = e.cast<AffineDimExpr>().getPosition();
        if (dim == left) a = i;
        if (dim == right) b = i;
      }
      if ((a < 0) != (b < 0) || (a >= 0 && b != a + 1))
        return false;
    }
    return true;
  };
  SmallVector<ReassociationIndices> groups;
  auto restorable = [&](ArrayRef<int64_t> group) {
    // This vendored MLIR version's expand_shape cannot recover two dynamic
    // factors from one collapsed extent. Keep those output dimensions apart.
    for (auto [number, output] : llvm::enumerate(op.getOutputs())) {
      auto type = dyn_cast<RankedTensorType>(output.getType());
      if (!type)
        return false;
      unsigned dynamic = 0;
      auto map = maps[op.getNumDpsInputs() + number];
      for (auto [position, expr] : llvm::enumerate(map.getResults())) {
        auto dim = expr.dyn_cast<AffineDimExpr>();
        if (!dim)
          return false;
        if (llvm::is_contained(group, dim.getPosition()) && type.isDynamicDim(position))
          ++dynamic;
      }
      if (dynamic > 1)
        return false;
    }
    return true;
  };
  for (int64_t i = 0, n = op.getNumLoops(); i < n;) {
    ReassociationIndices group{i++};
    while (i < n && compatible(i - 1, i)) {
      group.push_back(i);
      if (!restorable(group)) {
        group.pop_back();
        break;
      }
      ++i;
    }
    if (group.size() > 1)
      groups.push_back(group);
  }
  return groups;
}

// A rectangular prefix remains rectangular after collapse only if each inner
// dimension of a group covers the whole base dimension. Preserve partial outer
// dimensions and reject unknown inner extents instead of assuming contiguity.
static bool contiguousPrefix(Value base, ValueRange sizes,
                             ArrayRef<ReassociationIndices> groups) {
  auto type = dyn_cast<RankedTensorType>(base.getType());
  if (!type || static_cast<size_t>(type.getRank()) != sizes.size())
    return false;
  for (const auto &group : groups)
    for (int64_t dim : ArrayRef<int64_t>(group).drop_front()) {
      APInt size;
      if (type.isDynamicDim(dim) ||
          !matchPattern(sizes[dim], m_ConstantInt(&size)) ||
          size.getSExtValue() != type.getDimSize(dim))
        return false;
    }
  return true;
}

static SmallVector<Value>
collapseSizes(PatternRewriter &rewriter, Location loc, ValueRange sizes,
              ArrayRef<ReassociationIndices> groups) {
  SmallVector<Value> result;
  for (const auto &group : groups) {
    Value size = sizes[group.front()];
    for (int64_t dim : ArrayRef<int64_t>(group).drop_front())
      size = rewriter.createOrFold<arith::MulIOp>(loc, size, sizes[dim]);
    result.push_back(size);
  }
  return result;
}

static Value collapseTensor(PatternRewriter &rewriter, Location loc, Value value,
                            ArrayRef<ReassociationIndices> groups) {
  auto type = tensor::CollapseShapeOp::inferCollapsedType(
      cast<RankedTensorType>(value.getType()), SmallVector<ReassociationIndices>(groups));
  return rewriter.createOrFold<tensor::CollapseShapeOp>(loc, type, value, groups);
}

struct CollapseSubmap : OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::CollapseShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto view = op.getSrc().getDefiningOp<SubmapOp>();
    auto groups = op.getReassociationIndices();
    if (!view || !view.getMap().isIdentity() ||
        !contiguousPrefix(view.getBase(), view.getSizes(), groups))
      return failure();
    Value base = collapseTensor(rewriter, op.getLoc(), view.getBase(), groups);
    auto sizes = collapseSizes(rewriter, op.getLoc(), view.getSizes(), groups);
    rewriter.replaceOpWithNewOp<SubmapOp>(
        op, op.getType(), base, sizes,
        rewriter.getMultiDimIdentityMap(groups.size()));
    return success();
  }
};

struct ExpandInverse : OpRewritePattern<SubmapInverseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SubmapInverseOp op,
                                PatternRewriter &rewriter) const override {
    auto expand = op.getViewModified().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expand || !op.getMap().isIdentity())
      return failure();
    auto groups = expand.getReassociationIndices();
    ValueRange sizes = op->getOperands().drop_front(2);
    if (!contiguousPrefix(op.getBaseOriginal(), sizes, groups))
      return failure();
    auto baseType = cast<RankedTensorType>(op.getType());
    // One unknown factor is inferable from the collapsed extent and the
    // remaining nonzero static dimensions. Two unknown factors (or a zero
    // divisor) require explicit dimension witnesses in this MLIR version.
    for (const auto &group : groups) {
      unsigned dynamic = 0;
      bool zero = false;
      for (int64_t dim : group) {
        dynamic += baseType.isDynamicDim(dim);
        zero |= baseType.getDimSize(dim) == 0;
      }
      if (dynamic > 1 || (dynamic && zero))
        return failure();
    }
    Value base = collapseTensor(rewriter, op.getLoc(), op.getBaseOriginal(), groups);
    auto collapsedSizes = collapseSizes(rewriter, op.getLoc(), sizes, groups);
    Value inverse = rewriter.create<SubmapInverseOp>(
        op.getLoc(), base.getType(), base, expand.getSrc(), collapsedSizes,
        rewriter.getMultiDimIdentityMap(groups.size()));
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(op, baseType, inverse, groups);
    return success();
  }
};

struct CollapseTensorMemref : OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::CollapseShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto tensor = op.getSrc().getDefiningOp<bufferization::ToTensorOp>();
    if (!tensor)
      return failure();
    auto type = dyn_cast<MemRefType>(tensor.getMemref().getType());
    auto groups = op.getReassociationIndices();
    if (!type || !memref::CollapseShapeOp::isGuaranteedCollapsible(type, groups))
      return failure();
    auto collapsedType = memref::CollapseShapeOp::computeCollapsedType(type, groups);
    Value view = rewriter.create<memref::CollapseShapeOp>(
        op.getLoc(), collapsedType, tensor.getMemref(), groups);
    OperationState state(op.getLoc(), bufferization::ToTensorOp::getOperationName());
    state.addOperands(view);
    state.addTypes(op.getType());
    state.addAttributes(tensor->getAttrs());
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return success();
  }
};

// Construct an optional lower-parallel-rank candidate using upstream tiling.
// Every tiled loop is parallel and has unit step; reduction order is unchanged.
// No scalar opcode, library name or workload identity participates in selection.
static LogicalResult sliceParallelDimensions(Operation *root,
                                            unsigned parallelLimit,
                                            unsigned operandLimit,
                                            bool fuseInitializers) {
  SmallVector<linalg::GenericOp> candidates;
  root->walk([&](linalg::GenericOp op) {
    if (op.hasTensorSemantics() && op.getNumDpsInits() &&
        op.getNumDpsInits() == op.getNumResults() &&
        (op.getNumResults() == 1 || fuseInitializers))
      candidates.push_back(op);
  });
  SmallVector<Operation *> tiledRoots;
  IRRewriter rewriter(root->getContext());
  if (fuseInitializers)
    std::reverse(candidates.begin(), candidates.end());
  for (auto op : candidates) {
    if (fuseInitializers && op->use_empty())
      continue;
    SmallVector<int64_t> tileSizes(op.getNumLoops(), 0);
    auto iterators = op.getIteratorTypesArray();
    unsigned remaining = parallelLimit && op.getNumParallelLoops() > parallelLimit
                             ? op.getNumParallelLoops() - parallelLimit : 0;
    for (auto [i, type] : llvm::enumerate(iterators))
      if (remaining && type == utils::IteratorType::parallel) {
        tileSizes[i] = 1;
        --remaining;
      }
    auto maps = op.getIndexingMapsArray();
    // For projected permutations, removing a unit iterator removes exactly
    // the corresponding operand dimension. Prefer a parallel coordinate
    // shared by the most operands; this exposes independent batches even
    // when the batch coordinate is not the leading physical dimension.
    if (operandLimit && llvm::all_of(maps, [](AffineMap m) {
          return m.isProjectedPermutation();
        })) {
      while (true) {
        SmallVector<AffineMap> oversized;
        for (AffineMap map : maps) {
          unsigned rank = 0;
          for (AffineExpr e : map.getResults())
            rank += !tileSizes[e.cast<AffineDimExpr>().getPosition()];
          if (rank > operandLimit)
            oversized.push_back(map);
        }
        if (oversized.empty())
          break;
        int best = -1;
        unsigned bestScore = 0;
        for (unsigned i = 0; i < tileSizes.size(); ++i) {
          if (tileSizes[i] || iterators[i] != utils::IteratorType::parallel)
            continue;
          auto includes = [&](AffineMap map) {
            return llvm::any_of(map.getResults(), [&](AffineExpr e) {
              return e.cast<AffineDimExpr>().getPosition() == i;
            });
          };
          if (!llvm::any_of(oversized, includes))
            continue;
          unsigned score = llvm::count_if(maps, includes);
          if (best < 0 || score > bestScore) {
            best = i;
            bestScore = score;
          }
        }
        if (best < 0)
          break;
        tileSizes[best] = 1;
      }
    }
    if (llvm::none_of(tileSizes, [](int64_t size) { return size != 0; }))
      continue;
    // Multiple destinations must all distinguish the tiled parallel
    // coordinates. A hidden broadcast of such a coordinate is shared state,
    // not an independent batch. Keep unsupported affine maps unchanged.
    bool separateParallelOutputs = true;
    if (op.getNumDpsInits() > 1) {
      for (unsigned output = 0; output < op.getNumDpsInits(); ++output) {
        AffineMap physical = maps[op.getNumDpsInputs() + output];
        Value destination = op.getDpsInitOperand(output)->get();
        while (auto view = destination.getDefiningOp<SubmapOp>()) {
          if (view.getMap().getNumDims() != physical.getNumResults()) {
            separateParallelOutputs = false;
            break;
          }
          physical = view.getMap().compose(physical);
          destination = view.getBase();
        }
        if (!physical.isProjectedPermutation()) {
          separateParallelOutputs = false;
          break;
        }
        for (unsigned loop = 0; loop < tileSizes.size(); ++loop)
          if (tileSizes[loop] &&
              !llvm::any_of(physical.getResults(), [&](AffineExpr expr) {
                return expr.cast<AffineDimExpr>().getPosition() == loop;
              }))
            separateParallelOutputs = false;
      }
    }
    if (!separateParallelOutputs)
      continue;
    rewriter.setInsertionPoint(op);
    linalg::LinalgTilingOptions options;
    // Imported definitions are kernel.defn regions, not func.func. Keep tile
    // constants local instead of using the convenience helper's FuncOp entry.
    options.setTileSizeComputationFunction(
        [tileSizes](OpBuilder &builder, Operation *operation) {
          SmallVector<Value, 4> values;
          for (int64_t size : tileSizes)
            values.push_back(builder.create<arith::ConstantIndexOp>(
                operation->getLoc(), size));
          return values;
        });
    SmallVector<unsigned> initializerOutputs;
    if (fuseInitializers)
      for (unsigned i = 0; i < op.getNumDpsInits(); ++i)
        if (isScalarInitializer(op.getDpsInitOperand(i)->get()))
          initializerOutputs.push_back(i);
    auto tiled = linalg::tileLinalgOp(rewriter, op, options);
    if (failed(tiled))
      return failure();
    if (!tiled->loops.empty()) {
      tiledRoots.push_back(tiled->loops.front());
      if (!initializerOutputs.empty()) {
        // Only parallel dimensions were tiled. Upstream fusion moves exactly
        // the needed initializer slice into that loop, preserving reduction
        // order and threading the original destination through its iter_args.
        SmallVector<scf::ForOp> loops;
        for (Operation *loop : tiled->loops)
          loops.push_back(cast<scf::ForOp>(loop));
        for (unsigned output : initializerOutputs) {
          auto slice = tiled->op.getDpsInitOperand(output)->get()
                           .getDefiningOp<tensor::ExtractSliceOp>();
          if (slice)
            (void)scf::tileAndFuseProducerOfSlice(rewriter, slice, loops);
        }
      }
    }
    // An externally unused destination is private to each independent tile.
    // Its updates cannot affect another tile because every tiled coordinate
    // occurs in the physical output map checked above. Forward its loop state
    // unchanged so the unused result does not retain a cyclic scratch update.
    // The local recurrence itself remains until ordinary use-based cleanup.
    if (op.getNumResults() > 1)
      for (auto [index, result] : llvm::enumerate(op.getResults()))
        if (result.use_empty())
          for (Operation *loopOp : tiled->loops) {
            auto loop = cast<scf::ForOp>(loopOp);
            auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
            rewriter.updateRootInPlace(yield, [&] {
              yield->setOperand(index, loop.getRegionIterArgs()[index]);
            });
          }
    rewriter.replaceOp(op, tiled->tensorResults);
  }
  RewritePatternSet patterns(root->getContext());
  linalg::ControlDropUnitDims options;
  options.rankReductionStrategy =
      linalg::ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice;
  linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
  // Clean only the new tiled regions. Keep semantic submap bounds explicit;
  // unrelated region computations must not be changed by this candidate.
  for (auto *dialect : root->getContext()->getLoadedDialects())
    if (dialect->getNamespace() != "polygeist" &&
        dialect->getNamespace() != "kernel")
      dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName operation :
       root->getContext()->getRegisteredOperations())
    if (operation.getDialectNamespace() != "polygeist" &&
        operation.getDialectNamespace() != "kernel")
      operation.getCanonicalizationPatterns(patterns, root->getContext());
  FrozenRewritePatternSet cleanup(std::move(patterns));
  for (Operation *loop : tiledRoots) {
    SmallVector<Operation *> operations;
    loop->walk([&](Operation *nested) {
      if (nested != loop)
        operations.push_back(nested);
    });
    GreedyRewriteConfig config;
    config.scope = &loop->getRegion(0);
    if (failed(applyOpPatternsAndFold(operations, cleanup, config)))
      return failure();
  }
  return success();
}

// A static rank-reduced tile already has exact extents. Expose those extents as
// identity views for the same matcher used by source-imported references.
static void exposeStaticSliceBounds(Operation *root, bool scalarViews) {
  SmallVector<linalg::GenericOp> candidates;
  root->walk([&](linalg::GenericOp op) {
    if (!op.getNumDpsInits() || op.getNumDpsInits() != op.getNumResults() ||
        (!scalarViews && op.getNumResults() != 1) ||
        !op->getParentOfType<scf::ForOp>())
      return;
    bool needsBounds = false;
    for (Value value : op->getOperands()) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || !type.hasStaticShape() || (!scalarViews && type.getRank() == 0))
        return;
      if (type.getRank() == 0)
        continue;
      needsBounds |= !value.getDefiningOp<SubmapOp>();
    }
    if (needsBounds)
      candidates.push_back(op);
  });
  IRRewriter rewriter(root->getContext());
  for (auto op : candidates) {
    rewriter.setInsertionPoint(op);
    SmallVector<Value> originalOutputs(op.getOutputs());
    SmallVector<bool> wrapOutputs;
    for (Value original : originalOutputs)
      wrapOutputs.push_back(!original.getDefiningOp<SubmapOp>() &&
                            cast<RankedTensorType>(original.getType()).getRank());
    SmallVector<SmallVector<Value>> outputSizes(originalOutputs.size());
    for (OpOperand &operand : op->getOpOperands()) {
      if (operand.get().getDefiningOp<SubmapOp>())
        continue;
      auto type = cast<RankedTensorType>(operand.get().getType());
      if (type.getRank() == 0)
        continue;
      SmallVector<Value> sizes;
      for (int64_t size : type.getShape())
        sizes.push_back(rewriter.create<arith::ConstantIndexOp>(op.getLoc(), size));
      auto view = rewriter.create<SubmapOp>(
          op.getLoc(), type, operand.get(), sizes,
          rewriter.getMultiDimIdentityMap(type.getRank()));
      operand.set(view.getResult());
      if (operand.getOperandNumber() >= op.getNumDpsInputs())
        outputSizes[operand.getOperandNumber() - op.getNumDpsInputs()] = sizes;
    }
    rewriter.setInsertionPointAfter(op);
    for (unsigned output = 0; output < originalOutputs.size(); ++output) {
      if (!wrapOutputs[output])
        continue;
      Value result = op.getResult(output);
      auto type = cast<RankedTensorType>(result.getType());
      auto restored = rewriter.create<SubmapInverseOp>(
          op.getLoc(), type, originalOutputs[output], result, outputSizes[output],
          rewriter.getMultiDimIdentityMap(type.getRank()));
      result.replaceAllUsesExcept(restored.getResult(), restored);
    }
  }
}

// Materialize pure scalar producers that do not depend on the reduction state.
// This changes storage only: the residual iterators, initializer and operation
// order are retained. No arithmetic identity or reduction reassociation is used.
struct SplitReductionInputs : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics() || !op.getNumReductionLoops())
      return failure();
    Block *body = op.getBody();
    Operation &first = body->front();
    if (first.getNumResults() != 1 || !isa<FloatType>(first.getResult(0).getType()) ||
        first.getNumRegions() || isa<arith::ConstantOp>(first) ||
        !isMemoryEffectFree(&first) || !isSpeculatable(&first))
      return failure();
    unsigned inputs = op.getNumDpsInputs();
    for (Value operand : first.getOperands()) {
      if (auto argument = dyn_cast<BlockArgument>(operand)) {
        if (argument.getOwner() == body && argument.getArgNumber() >= inputs)
          return failure();
      }
    }
    auto maps = op.getIndexingMapsArray();
    if (!llvm::all_of(maps, [](AffineMap map) {
          return map.isProjectedPermutation();
        }) || !op.getShapesToLoopsMap())
      return failure();
    // A reduction destination must expose its state coordinates explicitly.
    // Broadcast submaps are normalized before this pattern; if that proof
    // fails, do not treat hidden aliases as independent tensor elements.
    auto iterators = op.getIteratorTypesArray();
    for (AffineMap map : llvm::drop_begin(maps, inputs))
      for (AffineExpr expression : map.getResults())
        if (iterators[expression.cast<AffineDimExpr>().getPosition()] ==
            utils::IteratorType::reduction)
          return failure();
    for (Value operand : op.getOperands())
      if (auto type = dyn_cast<RankedTensorType>(operand.getType()))
        if (type.getEncoding())
          return failure();

    Location loc = op.getLoc();
    auto ranges = cast<linalg::LinalgOp>(op.getOperation())
                      .createLoopRanges(rewriter, loc);
    SmallVector<int64_t> shape;
    SmallVector<Value> dynamic;
    for (auto range : ranges) {
      auto size = getConstantIntValue(range.size);
      shape.push_back(size.value_or(ShapedType::kDynamic));
      if (!size)
        dynamic.push_back(cast<Value>(range.size));
    }
    Type element = first.getResult(0).getType();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, shape, element, dynamic);
    SmallVector<AffineMap> producerMaps(maps.begin(), maps.begin() + inputs);
    producerMaps.push_back(rewriter.getMultiDimIdentityMap(op.getNumLoops()));
    SmallVector<utils::IteratorType> parallel(op.getNumLoops(),
                                            utils::IteratorType::parallel);
    auto producer = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{empty.getType()}, op.getInputs(), ValueRange{empty},
        producerMaps, parallel, [&](OpBuilder &builder, Location nested,
                                   ValueRange arguments) {
          IRMapping mapping;
          for (unsigned i = 0; i < inputs; ++i)
            mapping.map(body->getArgument(i), arguments[i]);
          auto copy = builder.clone(first, mapping);
          builder.create<linalg::YieldOp>(nested, copy->getResults());
        });
    SmallVector<Value> residualInputs(op.getInputs());
    residualInputs.push_back(producer.getResult(0));
    maps.insert(maps.begin() + inputs,
                rewriter.getMultiDimIdentityMap(op.getNumLoops()));
    auto residual = rewriter.create<linalg::GenericOp>(
        loc, op.getResultTypes(), residualInputs, op.getOutputs(), maps,
        op.getIteratorTypesArray(), [&](OpBuilder &builder, Location nested,
                                       ValueRange arguments) {
          IRMapping mapping;
          for (unsigned i = 0; i < inputs; ++i)
            mapping.map(body->getArgument(i), arguments[i]);
          for (unsigned i = inputs; i < body->getNumArguments(); ++i)
            mapping.map(body->getArgument(i), arguments[i + 1]);
          mapping.map(first.getResult(0), arguments[inputs]);
          for (Operation &operation : llvm::drop_begin(*body))
            builder.clone(operation, mapping);
        });
    rewriter.replaceOp(op, residual->getResults());
    return success();
  }
};

struct NormalizeLinalgMatchingViewsPass
    : NormalizeLinalgMatchingViewsBase<NormalizeLinalgMatchingViewsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    NormalizeLinalgMatchingViewsBase::getDependentDialects(registry);
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    if (maxParallelDims || maxOperandRank || splitReductionInputs) {
      RewritePatternSet outputs(&getContext());
      populateNormalizeTensorReductionOutputPatterns(outputs);
      if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(outputs)))) {
        signalPassFailure();
        return;
      }
    }
    if (splitReductionInputs) {
      RewritePatternSet split(&getContext());
      split.add<SplitReductionInputs>(&getContext());
      linalg::populateEraseUnusedOperandsAndResultsPatterns(split);
      if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(split)))) {
        signalPassFailure();
        return;
      }
    }
    RewritePatternSet views(&getContext());
    views.add<ExposeProjectedSubmaps>(&getContext());
    if ((maxParallelDims || maxOperandRank) && fuseInitializers)
      views.add<ForwardUpdatedSubview, FuseScalarInitializerInput,
                RefineMultiOutputBounds>(&getContext());
    linalg::populateMoveInitOperandsToInputPattern(views);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(views)))) {
      signalPassFailure();
      return;
    }
    if ((maxParallelDims || maxOperandRank) && fuseInitializers) {
      // Complete fusion before removing dead initializer results: changing
      // DPS operands earlier can hide the producer's bounded view identity.
      RewritePatternSet unused(&getContext());
      unused.add<DropOverwrittenSubview>(&getContext());
      linalg::populateEraseUnusedOperandsAndResultsPatterns(unused);
      if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(unused)))) {
        signalPassFailure();
        return;
      }
    }
    RewritePatternSet collapse(&getContext());
    linalg::populateCollapseDimensions(collapse, [&](linalg::GenericOp op) {
      // Preserve scalar initializer shape until the consumer is tiled. A
      // preceding flatten/expand pair would hide that producer from fusion.
      if ((maxParallelDims || maxOperandRank) && fuseInitializers &&
          ((op.getNumResults() == 1 &&
            isScalarInitializer(op.getResult(0))) ||
           llvm::any_of(op.getOutputs(), isScalarInitializer)))
        return SmallVector<ReassociationIndices>{};
      return parallelGroups(op);
    });
    collapse.add<CollapseSubmap, ExpandInverse, CollapseTensorMemref>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(collapse)))) {
      signalPassFailure();
      return;
    }
    if (maxParallelDims || maxOperandRank) {
      if (failed(sliceParallelDimensions(getOperation(), maxParallelDims, maxOperandRank, fuseInitializers))) {
        signalPassFailure();
        return;
      }
      exposeStaticSliceBounds(getOperation(), fuseInitializers);
      if (fuseInitializers) {
        RewritePatternSet finalViews(&getContext());
        finalViews.add<ForwardUpdatedSubview, DropOverwrittenSubview,
                       NormalizeScalarInitializer, ForwardScalarInsertedView,
                       DropOverwrittenScalarInsert, NormalizeScalarSliceWrite>(&getContext());
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(finalViews)))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::polygeist::createNormalizeLinalgMatchingViewsPass() {
  return std::make_unique<NormalizeLinalgMatchingViewsPass>();
}

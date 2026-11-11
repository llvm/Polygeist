//===- NormalizeLinalgMatchingViews.cpp - Expose collapsible iteration spaces ===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.

#include "PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Ops.h"

using namespace mlir;
using namespace mlir::polygeist;

namespace {

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
                                            unsigned operandLimit) {
  SmallVector<linalg::GenericOp> candidates;
  root->walk([&](linalg::GenericOp op) {
    if (op.hasTensorSemantics() && op.getNumDpsInits() == 1 &&
        op.getNumResults() == 1)
      candidates.push_back(op);
  });
  SmallVector<Operation *> tiledRoots;
  IRRewriter rewriter(root->getContext());
  for (auto op : candidates) {
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
    auto tiled = linalg::tileLinalgOp(rewriter, op, options);
    if (failed(tiled))
      return failure();
    if (!tiled->loops.empty())
      tiledRoots.push_back(tiled->loops.front());
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
static void exposeStaticSliceBounds(Operation *root) {
  SmallVector<linalg::GenericOp> candidates;
  root->walk([&](linalg::GenericOp op) {
    if (op.getNumDpsInits() != 1 || op.getNumResults() != 1 ||
        !op->getParentOfType<scf::ForOp>())
      return;
    bool needsBounds = false;
    for (Value value : op->getOperands()) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || !type.hasStaticShape() || type.getRank() == 0)
        return;
      needsBounds |= !value.getDefiningOp<SubmapOp>();
    }
    if (needsBounds)
      candidates.push_back(op);
  });
  IRRewriter rewriter(root->getContext());
  for (auto op : candidates) {
    rewriter.setInsertionPoint(op);
    Value originalOutput = op.getDpsInitOperand(0)->get();
    bool wrapOutput = !originalOutput.getDefiningOp<SubmapOp>();
    SmallVector<Value> outputSizes;
    for (OpOperand &operand : op->getOpOperands()) {
      if (operand.get().getDefiningOp<SubmapOp>())
        continue;
      auto type = cast<RankedTensorType>(operand.get().getType());
      SmallVector<Value> sizes;
      for (int64_t size : type.getShape())
        sizes.push_back(rewriter.create<arith::ConstantIndexOp>(op.getLoc(), size));
      auto view = rewriter.create<SubmapOp>(
          op.getLoc(), type, operand.get(), sizes,
          rewriter.getMultiDimIdentityMap(type.getRank()));
      operand.set(view.getResult());
      if (operand.getOperandNumber() == op.getNumDpsInputs())
        outputSizes = sizes;
    }
    if (!wrapOutput)
      continue;
    rewriter.setInsertionPointAfter(op);
    Value result = op.getResult(0);
    auto type = cast<RankedTensorType>(result.getType());
    auto restored = rewriter.create<SubmapInverseOp>(
        op.getLoc(), type, originalOutput, result, outputSizes,
        rewriter.getMultiDimIdentityMap(type.getRank()));
    result.replaceAllUsesExcept(restored.getResult(), restored);
  }
}

struct NormalizeLinalgMatchingViewsPass
    : NormalizeLinalgMatchingViewsBase<NormalizeLinalgMatchingViewsPass> {
  void runOnOperation() override {
    if (maxParallelDims || maxOperandRank) {
      RewritePatternSet outputs(&getContext());
      populateNormalizeTensorReductionOutputPatterns(outputs);
      if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(outputs)))) {
        signalPassFailure();
        return;
      }
    }
    RewritePatternSet views(&getContext());
    views.add<ExposeProjectedSubmaps>(&getContext());
    linalg::populateMoveInitOperandsToInputPattern(views);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(views)))) {
      signalPassFailure();
      return;
    }
    RewritePatternSet collapse(&getContext());
    linalg::populateCollapseDimensions(collapse, parallelGroups);
    collapse.add<CollapseSubmap, ExpandInverse, CollapseTensorMemref>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(collapse)))) {
      signalPassFailure();
      return;
    }
    if (maxParallelDims || maxOperandRank) {
      if (failed(sliceParallelDimensions(getOperation(), maxParallelDims, maxOperandRank))) {
        signalPassFailure();
        return;
      }
      exposeStaticSliceBounds(getOperation());
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::polygeist::createNormalizeLinalgMatchingViewsPass() {
  return std::make_unique<NormalizeLinalgMatchingViewsPass>();
}

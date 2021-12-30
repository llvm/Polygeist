//===- PolygeistOps.cpp - BFV dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "polygeist/Dialect.h"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>

#define GET_OP_CLASSES
#include "polygeist/PolygeistOps.cpp.inc"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"

using namespace mlir;
using namespace polygeist;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// BarrierOp
//===----------------------------------------------------------------------===//
void print(OpAsmPrinter &out, BarrierOp) {
  out << BarrierOp::getOperationName();
}

LogicalResult verify(BarrierOp) { return success(); }

ParseResult parseBarrierOp(OpAsmParser &, OperationState &) {
  return success();
}

/// Collect the memory effects of the given op in 'effects'. Returns 'true' it
/// could extract the effect information from the op, otherwise returns 'false'
/// and conservatively populates the list with all possible effects.
static bool
collectEffects(Operation *op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Skip over barriers to avoid infinite recursion (those barriers would ask
  // this barrier again).
  if (isa<BarrierOp>(op))
    return true;

  // Collect effect instances the operation. Note that the implementation of
  // getEffects erases all effect instances that have the type other than the
  // template parameter so we collect them first in a local buffer and then
  // copy.
  SmallVector<MemoryEffects::EffectInstance> localEffects;
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    iface.getEffects<MemoryEffects::Read>(localEffects);
    llvm::append_range(effects, localEffects);
    iface.getEffects<MemoryEffects::Write>(localEffects);
    llvm::append_range(effects, localEffects);
    iface.getEffects<MemoryEffects::Allocate>(localEffects);
    llvm::append_range(effects, localEffects);
    iface.getEffects<MemoryEffects::Free>(localEffects);
    llvm::append_range(effects, localEffects);
    return true;
  }

  // We need to be conservative here in case the op doesn't have the interface
  // and assume it can have any possible effect.
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
  return false;
}

void BarrierOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  Operation *op = getOperation();
  for (Operation *it = op->getPrevNode(); it != nullptr; it = it->getPrevNode())
    if (!collectEffects(it, effects))
      return;
  for (Operation *it = op->getNextNode(); it != nullptr; it = it->getNextNode())
    if (!collectEffects(it, effects))
      return;

  // TODO: we need to handle regions in case the parent op isn't an SCF parallel
}

/// Replace cast(subindex(x, InterimType), FinalType) with subindex(x,
/// FinalType)
class CastOfSubIndex final : public OpRewritePattern<memref::CastOp> {
public:
  using OpRewritePattern<memref::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto subindexOp = castOp.source().getDefiningOp<SubIndexOp>();
    if (!subindexOp)
      return failure();

    if (castOp.getType().cast<MemRefType>().getShape().size() !=
        subindexOp.getType().cast<MemRefType>().getShape().size())
      return failure();
    if (castOp.getType().cast<MemRefType>().getElementType() != subindexOp.result().getType().cast<MemRefType>().getElementType())
      return failure();

    rewriter.replaceOpWithNewOp<SubIndexOp>(
        castOp, castOp.getType(), subindexOp.source(), subindexOp.index());
    return success();
  }
};

// Replace subindex(subindex(x)) with subindex(x) with appropriate
// indexing.
class SubIndex2 final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp subViewOp,
                                PatternRewriter &rewriter) const override {
    auto prevOp = subViewOp.source().getDefiningOp<SubIndexOp>();
    if (!prevOp)
      return failure();

    auto mt0 = prevOp.source().getType().cast<MemRefType>();
    auto mt1 = prevOp.getType().cast<MemRefType>();
    auto mt2 = subViewOp.getType().cast<MemRefType>();
    if (mt0.getShape().size() == mt2.getShape().size() &&
        mt1.getShape().size() == mt0.getShape().size() + 1) {
      rewriter.replaceOpWithNewOp<SubIndexOp>(subViewOp, mt2, prevOp.source(),
                                              subViewOp.index());
      return success();
    }
    if (mt0.getShape().size() == mt2.getShape().size() &&
        mt1.getShape().size() == mt0.getShape().size()) {
      rewriter.replaceOpWithNewOp<SubIndexOp>(
          subViewOp, mt2, prevOp.source(),
          rewriter.create<AddIOp>(prevOp.getLoc(), subViewOp.index(),
                                  prevOp.index()));
      return success();
    }
    return failure();
  }
};

// When possible, simplify subindex(x) to cast(x)
class SubToCast final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp subViewOp,
                                PatternRewriter &rewriter) const override {
    auto prev = subViewOp.source().getType().cast<MemRefType>();
    auto post = subViewOp.getType().cast<MemRefType>();
    bool legal = prev.getShape().size() == post.getShape().size();
    if (legal) {

      auto cidx = subViewOp.index().getDefiningOp<ConstantIndexOp>();
      if (!cidx)
        return failure();

      if (cidx.value() != 0)
        return failure();

      rewriter.replaceOpWithNewOp<memref::CastOp>(subViewOp, subViewOp.source(),
                                                  post);
      return success();
    }

    return failure();
  }
};

// Simplify polygeist.subindex to memref.subview.
class SubToSubView final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp op,
                                PatternRewriter &rewriter) const override {
    auto srcMemRefType = op.source().getType().cast<MemRefType>();
    auto resMemRefType = op.result().getType().cast<MemRefType>();
    auto dims = srcMemRefType.getShape().size();

    // For now, restrict subview lowering to statically defined memref's
    if (!srcMemRefType.hasStaticShape() | !resMemRefType.hasStaticShape())
      return failure();

    // For now, restrict to simple rank-reducing indexing
    if (srcMemRefType.getShape().size() <= resMemRefType.getShape().size())
      return failure();

    // Build offset, sizes and strides
    SmallVector<OpFoldResult> sizes(dims, rewriter.getIndexAttr(0));
    sizes[0] = op.index();
    SmallVector<OpFoldResult> offsets(dims);
    for (auto dim : llvm::enumerate(srcMemRefType.getShape())) {
      if (dim.index() == 0)
        offsets[0] = rewriter.getIndexAttr(1);
      else
        offsets[dim.index()] = rewriter.getIndexAttr(dim.value());
    }
    SmallVector<OpFoldResult> strides(dims, rewriter.getIndexAttr(1));

    // Generate the appropriate return type:
    auto subMemRefType = MemRefType::get(srcMemRefType.getShape().drop_front(),
                                         srcMemRefType.getElementType());

    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        op, subMemRefType, op.source(), sizes, offsets, strides);

    return success();
  }
};

// Simplify redundant dynamic subindex patterns which tries to represent
// rank-reducing indexing:
//   %3 = "polygeist.subindex"(%1, %arg0) : (memref<2x1000xi32>, index) ->
//   memref<?x1000xi32> %4 = "polygeist.subindex"(%3, %c0) :
//   (memref<?x1000xi32>, index) -> memref<1000xi32>
// simplifies to:
//   %4 = "polygeist.subindex"(%1, %arg0) : (memref<2x1000xi32>, index) ->
//   memref<1000xi32>

class RedundantDynSubIndex final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp op,
                                PatternRewriter &rewriter) const override {
    auto srcOp = op.source().getDefiningOp<SubIndexOp>();
    if (!srcOp)
      return failure();

    auto preMemRefType = srcOp.source().getType().cast<MemRefType>();
    auto srcMemRefType = op.source().getType().cast<MemRefType>();
    auto resMemRefType = op.result().getType().cast<MemRefType>();

    // Check that this is indeed a rank reducing operation
    if (srcMemRefType.getShape().size() !=
        (resMemRefType.getShape().size() + 1))
      return failure();

    // Check that the previous op is the same rank.
    if (srcMemRefType.getShape().size() != preMemRefType.getShape().size())
      return failure();

    // Valid optimization target; perform the substitution.
    rewriter.replaceOpWithNewOp<SubIndexOp>(
        op, op.result().getType(), srcOp.source(),
        rewriter.create<arith::AddIOp>(op.getLoc(), op.index(), srcOp.index()));
    return success();
  }
};

/// Simplify all uses of subindex, specifically
//    store subindex(x) = ...
//    affine.store subindex(x) = ...
//    load subindex(x)
//    affine.load subindex(x)
//    dealloc subindex(x)
struct SimplifySubIndexUsers : public OpRewritePattern<SubIndexOp> {
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp subindex,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    for (OpOperand &use : llvm::make_early_inc_range(subindex->getUses())) {
      rewriter.setInsertionPoint(use.getOwner());
      if (auto dealloc = dyn_cast<memref::DeallocOp>(use.getOwner())) {
        changed = true;
        rewriter.replaceOpWithNewOp<memref::DeallocOp>(dealloc,
                                                       subindex.source());
      } else if (auto loadOp = dyn_cast<memref::LoadOp>(use.getOwner())) {
        if (loadOp.memref() == subindex) {
          SmallVector<Value, 4> indices = loadOp.indices();
          if (subindex.getType().cast<MemRefType>().getShape().size() ==
              subindex.source()
                  .getType()
                  .cast<MemRefType>()
                  .getShape()
                  .size()) {
            assert(indices.size() > 0);
            indices[0] = rewriter.create<AddIOp>(subindex.getLoc(), indices[0],
                                                 subindex.index());
          } else {
            assert (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
                subindex.source()
                    .getType()
                    .cast<MemRefType>()
                    .getShape()
                    .size());
            indices.insert(indices.begin(), subindex.index());
          }

          assert(subindex.source()
                     .getType()
                     .cast<MemRefType>()
                     .getShape()
                     .size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::LoadOp>(loadOp, subindex.source(),
                                                      indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
        if (storeOp.memref() == subindex) {
          SmallVector<Value, 4> indices = storeOp.indices();
          if (subindex.getType().cast<MemRefType>().getShape().size() ==
              subindex.source()
                  .getType()
                  .cast<MemRefType>()
                  .getShape()
                  .size()) {
            assert(indices.size() > 0);
            indices[0] = rewriter.create<AddIOp>(subindex.getLoc(), indices[0],
                                                 subindex.index());
          } else {
            assert (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
                subindex.source()
                    .getType()
                    .cast<MemRefType>()
                    .getShape()
                    .size());
            indices.insert(indices.begin(), subindex.index());
          }
          assert(subindex.source()
                     .getType()
                     .cast<MemRefType>()
                     .getShape()
                     .size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::StoreOp>(
              storeOp, storeOp.value(), subindex.source(), indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<memref::AtomicRMWOp>(use.getOwner())) {
        if (storeOp.memref() == subindex) {
          SmallVector<Value, 4> indices = storeOp.indices();
          if (subindex.getType().cast<MemRefType>().getShape().size() ==
              subindex.source()
                  .getType()
                  .cast<MemRefType>()
                  .getShape()
                  .size()) {
            assert(indices.size() > 0);
            indices[0] = rewriter.create<AddIOp>(subindex.getLoc(), indices[0],
                                                 subindex.index());
          } else {
            assert (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
                subindex.source()
                    .getType()
                    .cast<MemRefType>()
                    .getShape()
                    .size());
            indices.insert(indices.begin(), subindex.index());
          }
          assert(subindex.source()
                     .getType()
                     .cast<MemRefType>()
                     .getShape()
                     .size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::AtomicRMWOp>(
              storeOp, storeOp.getType(), storeOp.kind(), storeOp.value(), subindex.source(), indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<AffineStoreOp>(use.getOwner())) {
        if (storeOp.memref() == subindex) {
          if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
              subindex.source()
                  .getType()
                  .cast<MemRefType>()
                  .getShape()
                  .size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(subindex.index());
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }

            assert(subindex.source()
                       .getType()
                       .cast<MemRefType>()
                       .getShape()
                       .size() == indices.size());
            rewriter.replaceOpWithNewOp<memref::StoreOp>(
                storeOp, storeOp.value(), subindex.source(), indices);
            changed = true;
          }
        }
      } else if (auto storeOp = dyn_cast<AffineLoadOp>(use.getOwner())) {
        if (storeOp.memref() == subindex) {
          if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
              subindex.source()
                  .getType()
                  .cast<MemRefType>()
                  .getShape()
                  .size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(subindex.index());
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }
            assert(subindex.source()
                       .getType()
                       .cast<MemRefType>()
                       .getShape()
                       .size() == indices.size());
            rewriter.replaceOpWithNewOp<memref::LoadOp>(
                storeOp, subindex.source(), indices);
            changed = true;
          }
        }
      }
    }

    return success(changed);
  }
};

/// Simplify all uses of subindex, specifically
//    store subindex(x) = ...
//    affine.store subindex(x) = ...
//    load subindex(x)
//    affine.load subindex(x)
//    dealloc subindex(x)
struct SimplifySubViewUsers : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp subindex,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    int64_t offs = -1;
    for (auto tup :
         llvm::zip(subindex.static_offsets(), subindex.static_sizes(),
                   subindex.static_strides())) {
      auto sz = std::get<1>(tup).dyn_cast<IntegerAttr>().getValue();

      auto stride = std::get<2>(tup).dyn_cast<IntegerAttr>().getValue();
      if (stride != 1)
        return failure();

      if (offs == -1) {
        offs = std::get<0>(tup)
                   .dyn_cast<IntegerAttr>()
                   .getValue()
                   .getLimitedValue();
        if (sz != 1)
          return failure();
      }
    }
    Value off = rewriter.create<ConstantIndexOp>(subindex.getLoc(), offs);
    assert(off);

    for (OpOperand &use : llvm::make_early_inc_range(subindex->getUses())) {
      rewriter.setInsertionPoint(use.getOwner());
      if (auto dealloc = dyn_cast<memref::DeallocOp>(use.getOwner())) {
        changed = true;
        rewriter.replaceOpWithNewOp<memref::DeallocOp>(dealloc,
                                                       subindex.source());
      } else if (auto loadOp = dyn_cast<memref::LoadOp>(use.getOwner())) {
        if (loadOp.memref() == subindex) {
          SmallVector<Value, 4> indices = loadOp.indices();
          if (subindex.getType().cast<MemRefType>().getShape().size() ==
              subindex.source()
                  .getType()
                  .cast<MemRefType>()
                  .getShape()
                  .size()) {
            assert(indices.size() > 0);
            indices[0] =
                rewriter.create<AddIOp>(subindex.getLoc(), indices[0], off);
          } else {
            if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
                subindex.source()
                    .getType()
                    .cast<MemRefType>()
                    .getShape()
                    .size())
              indices.insert(indices.begin(), off);
            else {
              assert(indices.size() > 0);
              indices.erase(indices.begin());
            }
          }

          assert(subindex.source()
                     .getType()
                     .cast<MemRefType>()
                     .getShape()
                     .size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::LoadOp>(loadOp, subindex.source(),
                                                      indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
        if (storeOp.memref() == subindex) {
          SmallVector<Value, 4> indices = storeOp.indices();
          if (subindex.getType().cast<MemRefType>().getShape().size() ==
              subindex.source()
                  .getType()
                  .cast<MemRefType>()
                  .getShape()
                  .size()) {
            assert(indices.size() > 0);
            indices[0] =
                rewriter.create<AddIOp>(subindex.getLoc(), indices[0], off);
          } else {
            if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
                subindex.source()
                    .getType()
                    .cast<MemRefType>()
                    .getShape()
                    .size())
              indices.insert(indices.begin(), off);
            else {
              if (indices.size() == 0) {
                llvm::errs() << " storeOp: " << storeOp
                             << " - subidx: " << subindex << "\n";
              }
              assert(indices.size() > 0);
              indices.erase(indices.begin());
            }
          }

          if (subindex.source()
                  .getType()
                  .cast<MemRefType>()
                  .getShape()
                  .size() != indices.size()) {
            llvm::errs() << " storeOp: " << storeOp << " - subidx: " << subindex
                         << "\n";
          }
          assert(subindex.source()
                     .getType()
                     .cast<MemRefType>()
                     .getShape()
                     .size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::StoreOp>(
              storeOp, storeOp.value(), subindex.source(), indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<AffineStoreOp>(use.getOwner())) {
        if (storeOp.memref() == subindex) {
          if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
              subindex.source()
                  .getType()
                  .cast<MemRefType>()
                  .getShape()
                  .size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(off);
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }

            assert(subindex.source()
                       .getType()
                       .cast<MemRefType>()
                       .getShape()
                       .size() == indices.size());
            rewriter.replaceOpWithNewOp<memref::StoreOp>(
                storeOp, storeOp.value(), subindex.source(), indices);
            changed = true;
          }
        }
      } else if (auto storeOp = dyn_cast<AffineLoadOp>(use.getOwner())) {
        if (storeOp.memref() == subindex) {
          if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
              subindex.source()
                  .getType()
                  .cast<MemRefType>()
                  .getShape()
                  .size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(off);
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }
            assert(subindex.source()
                       .getType()
                       .cast<MemRefType>()
                       .getShape()
                       .size() == indices.size());
            rewriter.replaceOpWithNewOp<memref::LoadOp>(
                storeOp, subindex.source(), indices);
            changed = true;
          }
        }
      }
    }

    return success(changed);
  }
};

/// Simplify select cast(x), cast(y) to cast(select x, y)
struct SelectOfCast : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto cst1 = op.getTrueValue().getDefiningOp<memref::CastOp>();
    if (!cst1)
      return failure();

    auto cst2 = op.getFalseValue().getDefiningOp<memref::CastOp>();
    if (!cst2)
      return failure();

    if (cst1.source().getType() != cst2.source().getType())
      return failure();

    auto newSel = rewriter.create<SelectOp>(op.getLoc(), op.getCondition(),
                                            cst1.source(), cst2.source());

    rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(), newSel);
    return success();
  }
};

/// Simplify select subindex(x), subindex(y) to subindex(select x, y)
struct SelectOfSubIndex : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto cst1 = op.getTrueValue().getDefiningOp<SubIndexOp>();
    if (!cst1)
      return failure();

    auto cst2 = op.getFalseValue().getDefiningOp<SubIndexOp>();
    if (!cst2)
      return failure();

    if (cst1.source().getType() != cst2.source().getType())
      return failure();

    auto newSel = rewriter.create<SelectOp>(op.getLoc(), op.getCondition(),
                                            cst1.source(), cst2.source());
    auto newIdx = rewriter.create<SelectOp>(op.getLoc(), op.getCondition(),
                                            cst1.index(), cst2.index());
    rewriter.replaceOpWithNewOp<SubIndexOp>(op, op.getType(), newSel, newIdx);
    return success();
  }
};

void SubIndexOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<CastOfSubIndex, SubIndex2,
                 SubToCast, SimplifySubViewUsers, SimplifySubIndexUsers,
                 SelectOfCast, SelectOfSubIndex, RedundantDynSubIndex>(context);
  // Disabled: SubToSubView
}

/// Simplify pointer2memref(memref2pointer(x)) to cast(x)
class Memref2Pointer2MemrefCast final
    : public OpRewritePattern<Pointer2MemrefOp> {
public:
  using OpRewritePattern<Pointer2MemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Pointer2MemrefOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.source().getDefiningOp<Memref2PointerOp>();
    if (!src)
      return failure();
    if (src.source().getType().cast<MemRefType>().getShape().size() !=
            op.getType().cast<MemRefType>().getShape().size())
      return failure();
    if (src.source().getType().cast<MemRefType>().getElementType() !=
            op.getType().cast<MemRefType>().getElementType())
      return failure();
    if (src.source().getType().cast<MemRefType>().getMemorySpace() !=
            op.getType().cast<MemRefType>().getMemorySpace())
      return failure();

    rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(), src.source());
    return success();
  }
};
/// Simplify pointer2memref(memref2pointer(x)) to cast(x)
class Memref2PointerIndex final
    : public OpRewritePattern<Memref2PointerOp> {
public:
  using OpRewritePattern<Memref2PointerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Memref2PointerOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.source().getDefiningOp<SubIndexOp>();
    if (!src)
      return failure();

    if (src.source().getType().cast<MemRefType>().getShape().size() != 1) return failure();

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, op.getType(), rewriter.create<Memref2PointerOp>(op.getLoc(), op.getType(), src.source()),
            std::vector<Value>({rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getI64Type(), src.index())}));
    return success();
  }
};

OpFoldResult Memref2PointerOp::fold(ArrayRef<Attribute> operands) {
    if (auto subindex = source().getDefiningOp<SubIndexOp>()) {
        if (auto cop = subindex.index().getDefiningOp<ConstantIndexOp>()) {
            if (cop.value() == 0) {
                sourceMutable().assign(subindex.source());
                return result();
            }
        }
    }
    /// Simplify memref2pointer(cast(x)) to memref2pointer(x)
    if (auto mc = source().getDefiningOp<memref::CastOp>()) {
        sourceMutable().assign(mc.source());
        return result();
    }
    if (auto mc = source().getDefiningOp<polygeist::Pointer2MemrefOp>()) {
        if (mc.source().getType() == getType()) {
            return mc.source();
        }
    }
    return nullptr;
}

void Memref2PointerOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<Memref2Pointer2MemrefCast, Memref2PointerIndex>(context);
}

/// Simplify cast(pointer2memref(x)) to pointer2memref(x)
class Pointer2MemrefCast final : public OpRewritePattern<memref::CastOp> {
public:
  using OpRewritePattern<memref::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.source().getDefiningOp<Pointer2MemrefOp>();
    if (!src)
      return failure();

    rewriter.replaceOpWithNewOp<polygeist::Pointer2MemrefOp>(op, op.getType(),
                                                             src.source());
    return success();
  }
};

/// Simplify memref2pointer(pointer2memref(x)) to cast(x)
class Pointer2Memref2PointerCast final
    : public OpRewritePattern<Memref2PointerOp> {
public:
  using OpRewritePattern<Memref2PointerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Memref2PointerOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.source().getDefiningOp<Pointer2MemrefOp>();
    if (!src)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getType(),
                                                 src.source());
    return success();
  }
};

/// Simplify load (pointer2memref(x)) to llvm.load x
class Pointer2MemrefLoad final : public OpRewritePattern<memref::LoadOp> {
public:
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.memref().getDefiningOp<Pointer2MemrefOp>();
    if (!src)
      return failure();

    Value val = src.source();
    if (val.getType().cast<LLVM::LLVMPointerType>().getElementType() != 
        src.getType().cast<MemRefType>().getElementType())
      return failure();

    Value idx = nullptr;
    for (size_t i = 0; i < op.indices().size(); i++) {
      auto cur = rewriter.create<IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), op.indices()[i]);
      if (idx == nullptr) {
        idx = cur;
      } else {
        idx = rewriter.create<AddIOp>(
            op.getLoc(),
            rewriter.create<MulIOp>(
                op.getLoc(), idx,
                rewriter.create<ConstantIntOp>(
                    op.getLoc(),
                    op.memref().getType().cast<MemRefType>().getShape()[i],
                    32)),
            cur);
      }
    }
    Value idxs[] = {idx};
    if (idx)
      val = rewriter.create<LLVM::GEPOp>(op.getLoc(), val.getType(), val, idxs);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), val);
    return success();
  }
};

/// Simplify store (pointer2memref(x)) to llvm.store x
class Pointer2MemrefStore final : public OpRewritePattern<memref::StoreOp> {
public:
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.memref().getDefiningOp<Pointer2MemrefOp>();
    if (!src)
      return failure();
    
    Value val = src.source();
    if (val.getType().cast<LLVM::LLVMPointerType>().getElementType() != 
        src.getType().cast<MemRefType>().getElementType())
      return failure();
    Value idx = nullptr;
    for (size_t i = 0; i < op.indices().size(); i++) {
      auto cur = rewriter.create<IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), op.indices()[i]);
      if (idx == nullptr) {
        idx = cur;
      } else {
        idx = rewriter.create<AddIOp>(
            op.getLoc(),
            rewriter.create<MulIOp>(
                op.getLoc(), idx,
                rewriter.create<ConstantIntOp>(
                    op.getLoc(),
                    op.memref().getType().cast<MemRefType>().getShape()[i],
                    32)),
            cur);
      }
    }
    Value idxs[] = {idx};
    if (idx)
      val = rewriter.create<LLVM::GEPOp>(op.getLoc(), val.getType(), val, idxs);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, op.value(), val);
    return success();
  }
};

void Pointer2MemrefOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<Pointer2MemrefCast, Pointer2Memref2PointerCast,
                 Pointer2MemrefLoad, Pointer2MemrefStore>(
      context);
}

OpFoldResult Pointer2MemrefOp::fold(ArrayRef<Attribute> operands) {
    /// Simplify pointer2memref(cast(x)) to pointer2memref(x)
    if (auto mc = source().getDefiningOp<LLVM::BitcastOp>()) {
        sourceMutable().assign(mc.getArg());
        return result();
    }
    if (auto mc = source().getDefiningOp<polygeist::Memref2PointerOp>()) {
        if (mc.source().getType() == getType()) {
            return mc.source();
        }
    }
    return nullptr;
}

OpFoldResult SubIndexOp::fold(ArrayRef<Attribute> operands) {
    if (result().getType() == source().getType()) {
        if (matchPattern(index(), m_Zero()))
            return source();
    }
    /// Replace subindex(cast(x)) with subindex(x)
    if (auto castOp = source().getDefiningOp<memref::CastOp>()) {
      if (castOp.getType().cast<MemRefType>().getElementType() == result().getType().cast<MemRefType>().getElementType()) {
        sourceMutable().assign(castOp.source());
        return result();
      }
    }
    return nullptr;
}

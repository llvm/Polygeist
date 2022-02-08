//===- PolygeistOps.cpp - BFV dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "polygeist/Dialect.h"

#define GET_OP_CLASSES
#include "polygeist/PolygeistOps.cpp.inc"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace mlir;
using namespace polygeist;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// BarrierOp
//===----------------------------------------------------------------------===//
LogicalResult verify(BarrierOp) { return success(); }

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
    if (castOp.getType().cast<MemRefType>().getElementType() !=
        subindexOp.result().getType().cast<MemRefType>().getElementType())
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

      rewriter.replaceOpWithNewOp<memref::CastOp>(subViewOp, post,
                                                  subViewOp.source());
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
            assert(subindex.getType().cast<MemRefType>().getShape().size() +
                       1 ==
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
            assert(subindex.getType().cast<MemRefType>().getShape().size() +
                       1 ==
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
            assert(subindex.getType().cast<MemRefType>().getShape().size() +
                       1 ==
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
              storeOp, storeOp.getType(), storeOp.kind(), storeOp.value(),
              subindex.source(), indices);
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

/// Simplify select subindex(x), subindex(y) to subindex(select x, y)
template <typename T> struct LoadSelect : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  static Value ptr(T op);
  static MutableOperandRange ptrMutable(T op);

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    auto mem0 = ptr(op);
    SelectOp mem = dyn_cast_or_null<SelectOp>(mem0.getDefiningOp());
    if (!mem)
      return failure();

    Type tys[] = {op.getType()};
    auto iop = rewriter.create<scf::IfOp>(mem.getLoc(), tys, mem.getCondition(),
                                          /*hasElse*/ true);

    auto vop = cast<T>(op->clone());
    iop.thenBlock()->push_front(vop);
    ptrMutable(vop).assign(mem.getTrueValue());
    rewriter.setInsertionPointToEnd(iop.thenBlock());
    rewriter.create<scf::YieldOp>(op.getLoc(), vop->getResults());

    auto eop = cast<T>(op->clone());
    iop.elseBlock()->push_front(eop);
    ptrMutable(eop).assign(mem.getFalseValue());
    rewriter.setInsertionPointToEnd(iop.elseBlock());
    rewriter.create<scf::YieldOp>(op.getLoc(), eop->getResults());

    rewriter.replaceOp(op, iop.getResults());
    return success();
  }
};

template <> Value LoadSelect<memref::LoadOp>::ptr(memref::LoadOp op) {
  return op.memref();
}
template <>
MutableOperandRange LoadSelect<memref::LoadOp>::ptrMutable(memref::LoadOp op) {
  return op.memrefMutable();
}
template <> Value LoadSelect<AffineLoadOp>::ptr(AffineLoadOp op) {
  return op.memref();
}
template <>
MutableOperandRange LoadSelect<AffineLoadOp>::ptrMutable(AffineLoadOp op) {
  return op.memrefMutable();
}
template <> Value LoadSelect<LLVM::LoadOp>::ptr(LLVM::LoadOp op) {
  return op.getAddr();
}
template <>
MutableOperandRange LoadSelect<LLVM::LoadOp>::ptrMutable(LLVM::LoadOp op) {
  return op.getAddrMutable();
}

void SubIndexOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<CastOfSubIndex, SubIndex2, SubToCast, SimplifySubViewUsers,
                 SimplifySubIndexUsers, SelectOfCast, SelectOfSubIndex,
                 RedundantDynSubIndex, LoadSelect<memref::LoadOp>,
                 LoadSelect<AffineLoadOp>, LoadSelect<LLVM::LoadOp>>(context);
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
class Memref2PointerIndex final : public OpRewritePattern<Memref2PointerOp> {
public:
  using OpRewritePattern<Memref2PointerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Memref2PointerOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.source().getDefiningOp<SubIndexOp>();
    if (!src)
      return failure();

    if (src.source().getType().cast<MemRefType>().getShape().size() != 1)
      return failure();

    Value idx[] = {src.index()};
    auto PET = op.getType().cast<LLVM::LLVMPointerType>().getElementType();
    auto MET = src.source().getType().cast<MemRefType>().getElementType();
    if (PET != MET) {
      auto ps = rewriter.create<polygeist::TypeSizeOp>(
          op.getLoc(), rewriter.getIndexType(), mlir::TypeAttr::get(PET));
      auto ms = rewriter.create<polygeist::TypeSizeOp>(
          op.getLoc(), rewriter.getIndexType(), mlir::TypeAttr::get(MET));
      idx[0] = rewriter.create<MulIOp>(op.getLoc(), idx[0], ms);
      idx[0] = rewriter.create<DivUIOp>(op.getLoc(), idx[0], ps);
    }
    idx[0] = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                                 rewriter.getI64Type(), idx[0]);
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, op.getType(),
        rewriter.create<Memref2PointerOp>(op.getLoc(), op.getType(),
                                          src.source()),
        idx);
    return success();
  }
};

/// Simplify pointer2memref(memref2pointer(x)) to cast(x)
template <typename T>
class CopySimplification final : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {

    Value dstv = op.getDst();
    auto dst = dstv.getDefiningOp<polygeist::Memref2PointerOp>();
    if (!dst)
      return failure();

    auto dstTy = dst.source().getType().cast<MemRefType>();

    Value srcv = op.getSrc();
    auto src = srcv.getDefiningOp<polygeist::Memref2PointerOp>();
    if (!src)
      return failure();
    auto srcTy = src.source().getType().cast<MemRefType>();
    if (srcTy.getShape().size() != dstTy.getShape().size())
      return failure();

    if (dstTy.getElementType() != srcTy.getElementType())
      return failure();
    Type elTy = dstTy.getElementType();

    size_t width = 1;
    if (auto IT = elTy.dyn_cast<IntegerType>())
      width = IT.getWidth() / 8;
    else if (auto FT = elTy.dyn_cast<FloatType>())
      width = FT.getWidth() / 8;
    else {
      // TODO extend to llvm compatible type
      return failure();
    }
    bool first = true;
    SmallVector<size_t> bounds;
    for (auto pair : llvm::zip(dstTy.getShape(), srcTy.getShape())) {
      if (first) {
        first = false;
        continue;
      }
      if (std::get<0>(pair) != std::get<1>(pair))
        return failure();
      bounds.push_back(std::get<0>(pair));
      width *= std::get<0>(pair);
    }

    Value len = op.getLen();
    size_t factor = 1;
    while (factor % width != 0) {
      IntegerAttr constValue;
      if (auto ext = len.getDefiningOp<arith::ExtUIOp>())
        len = ext.getIn();
      else if (auto ext = len.getDefiningOp<arith::ExtSIOp>())
        len = ext.getIn();
      else if (auto mul = len.getDefiningOp<arith::MulIOp>())
        len = mul.getRhs();
      else if (matchPattern(len, m_Constant(&constValue))) {
        factor *= constValue.getValue().getLimitedValue();
      } else
        return failure();
    }

    if (factor % width != 0)
      return failure();

    Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    SmallVector<Value> idxs;
    auto forOp = rewriter.create<scf::ForOp>(
        op.getLoc(), c0,
        rewriter.create<arith::DivUIOp>(
            op.getLoc(),
            rewriter.create<arith::IndexCastOp>(
                op.getLoc(), rewriter.getIndexType(), op.getLen()),
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), width)),
        c1);

    rewriter.setInsertionPointToStart(&forOp.getLoopBody().front());
    idxs.push_back(forOp.getInductionVar());

    for (auto bound : bounds) {
      auto forOp = rewriter.create<scf::ForOp>(
          op.getLoc(), c0, rewriter.create<ConstantIndexOp>(op.getLoc(), bound),
          c1);
      rewriter.setInsertionPointToStart(&forOp.getLoopBody().front());
      idxs.push_back(forOp.getInductionVar());
    }

    rewriter.create<memref::StoreOp>(
        op.getLoc(),
        rewriter.create<memref::LoadOp>(op.getLoc(), src.source(), idxs),
        dst.source(), idxs);

    rewriter.eraseOp(op);
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

void Memref2PointerOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<Memref2Pointer2MemrefCast, Memref2PointerIndex,
                 CopySimplification<LLVM::MemcpyOp>,
                 CopySimplification<LLVM::MemmoveOp>>(context);
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
template <typename Op>
class MetaPointer2Memref final : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  Value computeIndex(Op op, size_t idx, PatternRewriter &rewriter) const;

  void rewrite(Op op, Value ptr, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Value opPtr = op.memref();
    Pointer2MemrefOp src = opPtr.getDefiningOp<polygeist::Pointer2MemrefOp>();
    if (!src)
      return failure();

    auto mt = src.getType().cast<MemRefType>();

    // Fantastic optimization, disabled for now to make a hard debug case easier
    // to find.
    if (auto before =
            src.source().getDefiningOp<polygeist::Memref2PointerOp>()) {
      auto mt0 = before.source().getType().cast<MemRefType>();
      if (mt0.getElementType() == mt.getElementType()) {
        auto sh0 = mt0.getShape();
        auto sh = mt.getShape();
        if (sh.size() == sh0.size()) {
          bool eq = true;
          for (size_t i = 1; i < sh.size(); i++) {
            if (sh[i] != sh0[i]) {
              eq = false;
              break;
            }
          }
          if (eq) {
            op.memrefMutable().assign(before.source());
            return success();
          }
        }
      }
    }

    for (size_t i = 1; i < mt.getShape().size(); i++)
      if (mt.getShape()[i] == ShapedType::kDynamicSize)
        return failure();

    Value val = src.source();
    if (val.getType().cast<LLVM::LLVMPointerType>().getElementType() !=
        mt.getElementType())
      val = rewriter.create<LLVM::BitcastOp>(
          op.getLoc(),
          LLVM::LLVMPointerType::get(
              mt.getElementType(),
              val.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          val);

    Value idx = nullptr;
    auto shape = mt.getShape();
    for (size_t i = 0; i < shape.size(); i++) {
      auto off = computeIndex(op, i, rewriter);
      auto cur = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), off);
      if (idx == nullptr) {
        idx = cur;
      } else {
        idx = rewriter.create<AddIOp>(
            op.getLoc(),
            rewriter.create<MulIOp>(op.getLoc(), idx,
                                    rewriter.create<arith::ConstantIntOp>(
                                        op.getLoc(), shape[i], 32)),
            cur);
      }
    }

    if (idx) {
      Value idxs[] = {idx};
      val = rewriter.create<LLVM::GEPOp>(op.getLoc(), val.getType(), val, idxs);
    }
    rewrite(op, val, rewriter);
    return success();
  }
};

template <>
Value MetaPointer2Memref<memref::LoadOp>::computeIndex(
    memref::LoadOp op, size_t i, PatternRewriter &rewriter) const {
  return op.indices()[i];
}

template <>
void MetaPointer2Memref<memref::LoadOp>::rewrite(
    memref::LoadOp op, Value ptr, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), ptr);
}

template <>
Value MetaPointer2Memref<memref::StoreOp>::computeIndex(
    memref::StoreOp op, size_t i, PatternRewriter &rewriter) const {
  return op.indices()[i];
}

template <>
void MetaPointer2Memref<memref::StoreOp>::rewrite(
    memref::StoreOp op, Value ptr, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, op.value(), ptr);
}

template <>
Value MetaPointer2Memref<AffineLoadOp>::computeIndex(
    AffineLoadOp op, size_t i, PatternRewriter &rewriter) const {
  auto map = op.getAffineMap();
  auto apply = rewriter.create<AffineApplyOp>(
      op.getLoc(), map.getSliceMap(i, 1), op.getMapOperands());
  return apply->getResult(0);
}

template <>
void MetaPointer2Memref<AffineLoadOp>::rewrite(
    AffineLoadOp op, Value ptr, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), ptr);
}

template <>
Value MetaPointer2Memref<AffineStoreOp>::computeIndex(
    AffineStoreOp op, size_t i, PatternRewriter &rewriter) const {
  auto map = op.getAffineMap();
  auto apply = rewriter.create<AffineApplyOp>(
      op.getLoc(), map.getSliceMap(i, 1), op.getMapOperands());
  return apply->getResult(0);
}

template <>
void MetaPointer2Memref<AffineStoreOp>::rewrite(
    AffineStoreOp op, Value ptr, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, op.value(), ptr);
}

// Below is actually wrong as and(40, 1) != 0   !=== and(40 != 0, 1 != 0) =
// and(true, true) = true and(x, y) != 0  -> and(x != 0, y != 0)
/*
class CmpAnd final : public OpRewritePattern<arith::CmpIOp> {
public:
  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getLhs().getDefiningOp<AndIOp>();
    if (!src)
      return failure();

    if (!matchPattern(op.getRhs(), m_Zero()))
      return failure();
    if (op.getPredicate() != arith::CmpIPredicate::ne)
      return failure();

    rewriter.replaceOpWithNewOp<arith::AndIOp>(
        op,
        rewriter.create<arith::CmpIOp>(op.getLoc(), CmpIPredicate::ne,
                                       src.getLhs(), op.getRhs()),
        rewriter.create<arith::CmpIOp>(op.getLoc(), CmpIPredicate::ne,
                                       src.getRhs(), op.getRhs()));
    return success();
  }
};
*/

#include "mlir/Dialect/SCF/SCF.h"
struct IfAndLazy : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp nextIf,
                                PatternRewriter &rewriter) const override {
    using namespace scf;
    Block *parent = nextIf->getBlock();
    if (nextIf == &parent->front())
      return failure();

    auto prevIf = dyn_cast<scf::IfOp>(nextIf->getPrevNode());
    if (!prevIf)
      return failure();

    if (nextIf.getCondition().getDefiningOp() != prevIf)
      return failure();

    // %c = if %x {
    //         yield %y
    //      } else {
    //         yield false
    //      }
    //  if %c {
    //
    //  } {
    //    yield s
    //  }

    Value nextIfCondition = nullptr;
    bool thenRegion = true;
    for (auto it :
         llvm::zip(prevIf.getResults(), prevIf.elseYield().getOperands(),
                   prevIf.thenYield().getOperands())) {
      if (std::get<0>(it) == nextIf.getCondition()) {
        if (matchPattern(std::get<1>(it), m_Zero())) {
          nextIfCondition = std::get<2>(it);
          thenRegion = true;
        } else if (matchPattern(std::get<2>(it), m_Zero())) {
          nextIfCondition = std::get<1>(it);
          thenRegion = false;
        } else
          return failure();
      }
    }

    YieldOp yield = thenRegion ? prevIf.thenYield() : prevIf.elseYield();
    YieldOp otherYield = thenRegion ? prevIf.elseYield() : prevIf.thenYield();

    // If the nextIf has an else region that computes, fail as this won't be
    // duplicated in the previous else.
    if (!nextIf.getElseRegion().empty()) {
      if (nextIf.elseBlock()->getOperations().size() != 1)
        return failure();

      // Moreover, if any of the other yielded values are computed in the if
      // statement, they cannot be used in the moved nextIf.
      for (auto v : otherYield.getOperands())
        if (otherYield->getParentRegion()->isAncestor(v.getParentRegion()))
          return failure();
    }

    rewriter.startRootUpdate(nextIf);
    nextIf->moveBefore(yield);
    nextIf.getConditionMutable().assign(nextIfCondition);
    for (auto it : llvm::zip(prevIf.getResults(), yield.getOperands())) {
      for (OpOperand &use :
           llvm::make_early_inc_range(std::get<0>(it).getUses()))
        if (nextIf.getThenRegion().isAncestor(
                use.getOwner()->getParentRegion())) {
          rewriter.startRootUpdate(use.getOwner());
          use.set(std::get<1>(it));
          rewriter.finalizeRootUpdate(use.getOwner());
        }
    }
    rewriter.finalizeRootUpdate(nextIf);

    // Handle else region
    if (!nextIf.getElseRegion().empty()) {
      SmallVector<Type> resTys;
      for (auto T : prevIf.getResultTypes())
        resTys.push_back(T);
      for (auto T : nextIf.getResultTypes())
        resTys.push_back(T);

      {
        SmallVector<Value> elseVals = otherYield.getOperands();
        BlockAndValueMapping elseMapping;
        elseMapping.map(prevIf.getResults(), otherYield.getOperands());
        SmallVector<Value> nextElseVals;
        for (auto v : nextIf.elseYield().getOperands())
          nextElseVals.push_back(elseMapping.lookupOrDefault(v));
        elseVals.append(nextElseVals);
        otherYield->setOperands(elseVals);
        nextIf.elseYield()->setOperands(nextElseVals);
      }

      SmallVector<Type> postTys;
      for (auto T : yield.getOperands())
        postTys.push_back(T.getType());
      for (auto T : nextIf.thenYield().getOperands())
        postTys.push_back(T.getType());

      rewriter.setInsertionPoint(prevIf);
      auto postIf = rewriter.create<scf::IfOp>(prevIf.getLoc(), postTys,
                                               prevIf.getCondition(), false);
      postIf.getThenRegion().takeBody(prevIf.getThenRegion());
      postIf.getElseRegion().takeBody(prevIf.getElseRegion());

      SmallVector<Value> res;
      SmallVector<Value> postRes;
      for (auto R : postIf.getResults())
        if (res.size() < prevIf.getNumResults())
          res.push_back(R);
        else
          postRes.push_back(R);

      rewriter.replaceOp(prevIf, res);
      nextIf->replaceAllUsesWith(postRes);

      SmallVector<Value> thenVals = yield.getOperands();
      thenVals.append(nextIf.getResults().begin(), nextIf.getResults().end());
      yield->setOperands(thenVals);
    }
    return success();
  }
};

struct CombineIfs : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp nextIf,
                                PatternRewriter &rewriter) const override {
    using namespace scf;
    Block *parent = nextIf->getBlock();
    if (nextIf == &parent->front())
      return failure();

    auto prevIf = dyn_cast<scf::IfOp>(nextIf->getPrevNode());
    if (!prevIf)
      return failure();

    if (nextIf.getCondition() != prevIf.getCondition())
      return failure();

    //* Changed*//
    SmallVector<Value> prevElseYielded;
    if (!prevIf.getElseRegion().empty())
      prevElseYielded = prevIf.elseYield().getOperands();
    // Replace all uses of return values of op within nextIf with the
    // corresponding yields
    for (auto it : llvm::zip(prevIf.getResults(),
                             prevIf.thenYield().getOperands(), prevElseYielded))
      for (OpOperand &use :
           llvm::make_early_inc_range(std::get<0>(it).getUses())) {
        if (nextIf.getThenRegion().isAncestor(
                use.getOwner()->getParentRegion())) {
          rewriter.startRootUpdate(use.getOwner());
          use.set(std::get<1>(it));
          rewriter.finalizeRootUpdate(use.getOwner());
        } else if (nextIf.getElseRegion().isAncestor(
                       use.getOwner()->getParentRegion())) {
          rewriter.startRootUpdate(use.getOwner());
          use.set(std::get<2>(it));
          rewriter.finalizeRootUpdate(use.getOwner());
        }
      }
    //* End Changed*//

    SmallVector<Type> mergedTypes(prevIf.getResultTypes());
    llvm::append_range(mergedTypes, nextIf.getResultTypes());

    //* Changed nextIf cond to nextIf cond*//
    scf::IfOp combinedIf = rewriter.create<scf::IfOp>(
        nextIf.getLoc(), mergedTypes, prevIf.getCondition(), /*hasElse=*/false);
    rewriter.eraseBlock(&combinedIf.getThenRegion().back());

    scf::YieldOp thenYield = prevIf.thenYield();
    scf::YieldOp thenYield2 = nextIf.thenYield();

    combinedIf.getThenRegion().getBlocks().splice(
        combinedIf.getThenRegion().getBlocks().begin(),
        prevIf.getThenRegion().getBlocks());

    rewriter.mergeBlocks(nextIf.thenBlock(), combinedIf.thenBlock());
    rewriter.setInsertionPointToEnd(combinedIf.thenBlock());

    SmallVector<Value> mergedYields(thenYield.getOperands());
    llvm::append_range(mergedYields, thenYield2.getOperands());
    rewriter.create<scf::YieldOp>(thenYield2.getLoc(), mergedYields);
    rewriter.eraseOp(thenYield);
    rewriter.eraseOp(thenYield2);

    combinedIf.getElseRegion().getBlocks().splice(
        combinedIf.getElseRegion().getBlocks().begin(),
        prevIf.getElseRegion().getBlocks());

    if (!nextIf.getElseRegion().empty()) {
      if (combinedIf.getElseRegion().empty()) {
        combinedIf.getElseRegion().getBlocks().splice(
            combinedIf.getElseRegion().getBlocks().begin(),
            nextIf.getElseRegion().getBlocks());
      } else {
        scf::YieldOp elseYield = combinedIf.elseYield();
        scf::YieldOp elseYield2 = nextIf.elseYield();
        rewriter.mergeBlocks(nextIf.elseBlock(), combinedIf.elseBlock());

        rewriter.setInsertionPointToEnd(combinedIf.elseBlock());

        SmallVector<Value> mergedElseYields(elseYield.getOperands());
        llvm::append_range(mergedElseYields, elseYield2.getOperands());

        rewriter.create<scf::YieldOp>(elseYield2.getLoc(), mergedElseYields);
        rewriter.eraseOp(elseYield);
        rewriter.eraseOp(elseYield2);
      }
    }

    SmallVector<Value> prevValues;
    SmallVector<Value> nextValues;
    for (const auto &pair : llvm::enumerate(combinedIf.getResults())) {
      if (pair.index() < prevIf.getNumResults())
        prevValues.push_back(pair.value());
      else
        nextValues.push_back(pair.value());
    }
    rewriter.replaceOp(prevIf, prevValues);
    rewriter.replaceOp(nextIf, nextValues);
    return success();
  }
};
struct MoveIntoIfs : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp nextIf,
                                PatternRewriter &rewriter) const override {
    using namespace scf;
    Block *parent = nextIf->getBlock();
    if (nextIf == &parent->front())
      return failure();

    auto prevOp = nextIf->getPrevNode();

    // Only move if op doesn't write or free memory (only read)
    if (!wouldOpBeTriviallyDead(prevOp))
      return failure();
    if (isa<arith::ConstantOp>(prevOp))
      return failure();

    bool thenUse = false;
    bool elseUse = false;
    bool outsideUse = false;
    for (auto &use : prevOp->getUses()) {
      if (nextIf.getThenRegion().isAncestor(use.getOwner()->getParentRegion()))
        thenUse = true;
      else if (nextIf.getElseRegion().isAncestor(
                   use.getOwner()->getParentRegion()))
        elseUse = true;
      else
        outsideUse = true;
    }
    // Do not move if the op is used outside the if, or used in both branches
    if (outsideUse)
      return failure();
    if (thenUse && elseUse)
      return failure();
    // If no use, this should've been folded / eliminated
    if (!thenUse && !elseUse)
      return failure();

    rewriter.startRootUpdate(nextIf);
    rewriter.startRootUpdate(prevOp);
    prevOp->moveBefore(thenUse ? &nextIf.thenBlock()->front()
                               : &nextIf.elseBlock()->front());
    for (OpOperand &use : llvm::make_early_inc_range(prevOp->getUses())) {
      rewriter.setInsertionPoint(use.getOwner());
      if (auto storeOp = dyn_cast<AffineLoadOp>(use.getOwner())) {
        std::vector<Value> indices;
        auto map = storeOp.getAffineMap();
        for (size_t i = 0; i < map.getNumResults(); i++) {
          auto apply = rewriter.create<AffineApplyOp>(storeOp.getLoc(),
                                                      map.getSliceMap(i, 1),
                                                      storeOp.getMapOperands());
          indices.push_back(apply->getResult(0));
        }
        rewriter.replaceOpWithNewOp<memref::LoadOp>(storeOp, storeOp.memref(),
                                                    indices);
      } else if (auto storeOp = dyn_cast<AffineStoreOp>(use.getOwner())) {
        std::vector<Value> indices;
        auto map = storeOp.getAffineMap();
        for (size_t i = 0; i < map.getNumResults(); i++) {
          auto apply = rewriter.create<AffineApplyOp>(storeOp.getLoc(),
                                                      map.getSliceMap(i, 1),
                                                      storeOp.getMapOperands());
          indices.push_back(apply->getResult(0));
        }
        rewriter.replaceOpWithNewOp<memref::StoreOp>(storeOp, storeOp.value(),
                                                     storeOp.memref(), indices);
      }
    }
    rewriter.finalizeRootUpdate(prevOp);
    rewriter.finalizeRootUpdate(nextIf);
    return success();
  }
};

void Pointer2MemrefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<
      Pointer2MemrefCast, Pointer2Memref2PointerCast,
      MetaPointer2Memref<memref::LoadOp>, MetaPointer2Memref<memref::StoreOp>,
      MetaPointer2Memref<AffineLoadOp>, MetaPointer2Memref<AffineStoreOp>,
      CombineIfs, MoveIntoIfs, IfAndLazy>(context);
}

OpFoldResult Pointer2MemrefOp::fold(ArrayRef<Attribute> operands) {
  /// Simplify pointer2memref(cast(x)) to pointer2memref(x)
  if (auto mc = source().getDefiningOp<LLVM::BitcastOp>()) {
    sourceMutable().assign(mc.getArg());
    return result();
  }
  if (auto mc = source().getDefiningOp<LLVM::AddrSpaceCastOp>()) {
    sourceMutable().assign(mc.getArg());
    return result();
  }
  if (auto mc = source().getDefiningOp<LLVM::GEPOp>()) {
    for (auto idx : mc.getIndices()) {
      assert(idx);
      if (!matchPattern(idx, m_Zero()))
        return nullptr;
    }
    auto staticIndices = mc.getStructIndices().getValues<int32_t>();
    for (auto pair : llvm::enumerate(staticIndices)) {
      if (pair.value() != LLVM::GEPOp::kDynamicIndex)
        if (pair.value() != 0)
          return nullptr;
    }

    sourceMutable().assign(mc.getBase());
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
    if (castOp.getType().cast<MemRefType>().getElementType() ==
        result().getType().cast<MemRefType>().getElementType()) {
      sourceMutable().assign(castOp.source());
      return result();
    }
  }
  return nullptr;
}

OpFoldResult TypeSizeOp::fold(ArrayRef<Attribute> operands) {
  Type T = sourceAttr().getValue();
  if (T.isa<IntegerType, FloatType>() || LLVM::isCompatibleType(T)) {
    DataLayout DLI(((Operation *)*this)->getParentOfType<ModuleOp>());
    return IntegerAttr::get(getResult().getType(),
                            APInt(64, DLI.getTypeSize(T)));
  }
  return nullptr;
}
struct TypeSizeCanonicalize : public OpRewritePattern<TypeSizeOp> {
  using OpRewritePattern<TypeSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TypeSizeOp op,
                                PatternRewriter &rewriter) const override {
    Type T = op.sourceAttr().getValue();
    if (T.isa<IntegerType, FloatType>() || LLVM::isCompatibleType(T)) {
      DataLayout DLI(op->getParentOfType<ModuleOp>());
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op,
                                                          DLI.getTypeSize(T));
      return success();
    }
    return failure();
  }
};

void TypeSizeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<TypeSizeCanonicalize>(context);
}

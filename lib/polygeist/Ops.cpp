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

#define GET_OP_CLASSES
#include "polygeist/PolygeistOps.cpp.inc"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"

using namespace mlir;
using namespace mlir::polygeist;

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

Value SubIndexOp::getViewSource() { return source(); }

class SubIndexOpMemRefCastFolder final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp subViewOp,
                                PatternRewriter &rewriter) const override {
    auto castOp = subViewOp.source().getDefiningOp<memref::CastOp>();
    if (!castOp)
      return failure();

    if (!memref::CastOp::canFoldIntoConsumerOp(castOp))
      return failure();

    rewriter.replaceOpWithNewOp<SubIndexOp>(
        subViewOp, subViewOp.result().getType().cast<MemRefType>(),
        castOp.source(), subViewOp.index());
    return success();
  }
};

class CastOfSubIndex final : public OpRewritePattern<memref::CastOp> {
public:
  using OpRewritePattern<memref::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto subindexOp = castOp.source().getDefiningOp<SubIndexOp>();
    if (!subindexOp)
      return failure();

    if (castOp.getType().cast<MemRefType>().getShape().size() != subindexOp.getType().cast<MemRefType>().getShape().size())
      return failure();

    rewriter.replaceOpWithNewOp<SubIndexOp>(
        castOp, castOp.getType(),
        subindexOp.source(), subindexOp.index());
    return success();
  }
};

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
      rewriter.replaceOpWithNewOp<SubIndexOp>(subViewOp, mt2, prevOp.source(),
                                              rewriter.create<AddIOp>(prevOp.getLoc(), subViewOp.index(), prevOp.index()));
      return success();
    }
    return failure();
  }
};

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

      if (cidx.getValue() != 0 && cidx.getValue() != -1)
        return failure();

      rewriter.replaceOpWithNewOp<memref::CastOp>(subViewOp, subViewOp.source(),
                                                  post);
      return success();
    }

    return failure();
  }
};

struct DeallocSubView : public OpRewritePattern<SubIndexOp> {
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
            if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
                subindex.source()
                    .getType()
                    .cast<MemRefType>()
                    .getShape()
                    .size())
              indices.insert(indices.begin(), subindex.index());
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
            indices[0] = rewriter.create<AddIOp>(subindex.getLoc(), indices[0],
                                                 subindex.index());
          } else {
            if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
                subindex.source()
                    .getType()
                    .cast<MemRefType>()
                    .getShape()
                    .size())
              indices.insert(indices.begin(), subindex.index());
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
              indices.push_back(subindex.index());
              for (size_t i=0; i<map.getNumResults(); i++) {
                auto apply = rewriter.create<AffineApplyOp>(storeOp.getLoc(), map.getSliceMap(i, 1), storeOp.getMapOperands());
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
              for (size_t i=0; i<map.getNumResults(); i++) {
                auto apply = rewriter.create<AffineApplyOp>(storeOp.getLoc(), map.getSliceMap(i, 1), storeOp.getMapOperands());
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

void SubIndexOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results
      .insert<CastOfSubIndex, SubIndexOpMemRefCastFolder, SubIndex2, SubToCast, DeallocSubView>(
          context);
}
  
  LogicalResult selectOfSubIndexHack(SelectOp op, PatternRewriter &rewriter) {
    auto cst1 = op.getTrueValue().getDefiningOp<SubIndexOp>();
    if (!cst1)
      return failure();

    auto cst2 = op.getFalseValue().getDefiningOp<SubIndexOp>();
    if (!cst2)
      return failure();
    
    if (cst1.source().getType() != cst2.source().getType())
      return failure();

    auto newSel = rewriter.create<SelectOp>(op.getLoc(), op.condition(), cst1.source(), cst2.source());
    auto newIdx = rewriter.create<SelectOp>(op.getLoc(), op.condition(), cst1.index(), cst2.index());
    rewriter.replaceOpWithNewOp<SubIndexOp>(op, op.getType(), newSel, newIdx);
    return success();
  }

Value Memref2PointerOp::getViewSource() { return source(); }

class MemRef2PointerCast final : public OpRewritePattern<Memref2PointerOp> {
public:
  using OpRewritePattern<Memref2PointerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Memref2PointerOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.source().getDefiningOp<memref::CastOp>();
    if (!src)
      return failure();

    rewriter.replaceOpWithNewOp<polygeist::Memref2PointerOp>(op, op.getType(),
                                                             src.source());
    return success();
  }
};
void Memref2PointerOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MemRef2PointerCast>(context);
}

//===- TrivialUse.cpp - Remove trivial use instruction ---------------- -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//

#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"

#define DEBUG_TYPE "convert-polygeist-to-llvm"

using namespace mlir;
using namespace polygeist;

/// Conversion pattern that transforms a subview op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The subview op is replaced by the descriptor.
struct SubIndexOpLowering : public ConvertOpToLLVMPattern<SubIndexOp> {
  using ConvertOpToLLVMPattern<SubIndexOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SubIndexOp subViewOp, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = subViewOp.getLoc();

    auto sourceMemRefType = subViewOp.source().getType().cast<MemRefType>();

    auto viewMemRefType = subViewOp.getType().cast<MemRefType>();

    if (sourceMemRefType.getShape().size() != viewMemRefType.getShape().size())
      return failure();

    // MemRefDescriptor sourceMemRef(operands.front());
    MemRefDescriptor targetMemRef(
        transformed
            .source()); // MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    if (false) {
      Value baseOffset = targetMemRef.offset(rewriter, loc);
      Value stride = targetMemRef.stride(rewriter, loc, 0);
      Value offset = transformed.index();
      Value mul = rewriter.create<LLVM::MulOp>(loc, offset, stride);
      baseOffset = rewriter.create<LLVM::AddOp>(loc, baseOffset, mul);
      targetMemRef.setOffset(rewriter, loc, baseOffset);
    } else {
      Value prev = targetMemRef.alignedPtr(rewriter, loc);
      Value idxs[] = {transformed.index()};
      targetMemRef.setAlignedPtr(
          rewriter, loc,
          rewriter.create<LLVM::GEPOp>(loc, prev.getType(), prev, idxs));
    }

    rewriter.replaceOp(subViewOp, {targetMemRef});
    return success();
  }
};

struct Memref2PointerOpLowering
    : public ConvertOpToLLVMPattern<Memref2PointerOp> {
  using ConvertOpToLLVMPattern<Memref2PointerOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Memref2PointerOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // MemRefDescriptor sourceMemRef(operands.front());
    MemRefDescriptor targetMemRef(
        transformed
            .source()); // MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Offset.
    Value baseOffset = targetMemRef.offset(rewriter, loc);
    Value ptr = targetMemRef.alignedPtr(rewriter, loc);
    Value idxs[] = {baseOffset};
    ptr = rewriter.create<LLVM::GEPOp>(loc, ptr.getType(), ptr, idxs);
    ptr = rewriter.create<LLVM::BitcastOp>(loc, op.getType(), ptr);

    rewriter.replaceOp(op, {ptr});
    return success();
  }
};

struct Pointer2MemrefOpLowering
    : public ConvertOpToLLVMPattern<Pointer2MemrefOp> {
  using ConvertOpToLLVMPattern<Pointer2MemrefOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Pointer2MemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // MemRefDescriptor sourceMemRef(operands.front());
    auto convertedType = getTypeConverter()->convertType(op.getType());
    assert(convertedType && "unexpected failure in memref type conversion");
    auto descr = MemRefDescriptor::undef(rewriter, loc, convertedType);
    auto ptr = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), descr.getElementPtrType(), adaptor.source());

    // Extract all strides and offsets and verify they are static.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto result = getStridesAndOffset(op.getType(), strides, offset);
    (void)result;
    assert(succeeded(result) && "unexpected failure in stride computation");
    assert(!MemRefType::isDynamicStrideOrOffset(offset) &&
           "expected static offset");

    bool first = true;
    assert(!llvm::any_of(strides, [&](int64_t stride) {
      if (first) {
        first = false;
        return false;
      }
      return MemRefType::isDynamicStrideOrOffset(stride);
    }) && "expected static strides except first element");

    descr.setAllocatedPtr(rewriter, loc, ptr);
    descr.setAlignedPtr(rewriter, loc, ptr);
    descr.setConstantOffset(rewriter, loc, offset);

    // Fill in sizes and strides
    for (unsigned i = 0, e = op.getType().getRank(); i != e; ++i) {
      descr.setConstantSize(rewriter, loc, i, op.getType().getDimSize(i));
      descr.setConstantStride(rewriter, loc, i, strides[i]);
    }

    rewriter.replaceOp(op, {descr});
    return success();
  }
};

void populatePolygeistToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<SubIndexOpLowering>(converter);
  patterns.add<Memref2PointerOpLowering>(converter);
  patterns.add<Pointer2MemrefOpLowering>(converter);
  // clang-format on
}

namespace {
struct ConvertPolygeistToLLVMPass
    : public ConvertPolygeistToLLVMBase<ConvertPolygeistToLLVMPass> {
  ConvertPolygeistToLLVMPass() = default;
  ConvertPolygeistToLLVMPass(bool useBarePtrCallConv, bool emitCWrappers,
                             unsigned indexBitwidth, bool useAlignedAlloc,
                             const llvm::DataLayout &dataLayout) {
    this->useBarePtrCallConv = useBarePtrCallConv;
    this->emitCWrappers = emitCWrappers;
    this->indexBitwidth = indexBitwidth;
    this->dataLayout = dataLayout.getStringRepresentation();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    options.useBarePtrCallConv = useBarePtrCallConv;
    options.emitCWrappers = emitCWrappers;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    options.dataLayout = llvm::DataLayout(this->dataLayout);

    LLVMTypeConverter converter(&getContext(), options, &dataLayoutAnalysis);
    RewritePatternSet patterns(&getContext());
    populatePolygeistToLLVMConversionPatterns(converter, patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    populateStdToLLVMConversionPatterns(converter, patterns);
    populateOpenMPToLLVMConversionPatterns(converter, patterns);
    arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
    populateStdExpandOpsPatterns(patterns);

    LLVMConversionTarget target(getContext());
    target.addDynamicallyLegalOp<omp::ParallelOp, omp::WsLoopOp>(
        [&](Operation *op) { return converter.isLegal(&op->getRegion(0)); });
    target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                      omp::BarrierOp, omp::TaskwaitOp>();
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::polygeist::createConvertPolygeistToLLVMPass(
    const LowerToLLVMOptions &options) {
  auto allocLowering = options.allocLowering;
  // There is no way to provide additional patterns for pass, so
  // AllocLowering::None will always fail.
  assert(allocLowering != LowerToLLVMOptions::AllocLowering::None &&
         "LLVMLoweringPass doesn't support AllocLowering::None");
  bool useAlignedAlloc =
      (allocLowering == LowerToLLVMOptions::AllocLowering::AlignedAlloc);
  return std::make_unique<ConvertPolygeistToLLVMPass>(
      options.useBarePtrCallConv, options.emitCWrappers,
      options.getIndexBitwidth(), useAlignedAlloc, options.dataLayout);
}

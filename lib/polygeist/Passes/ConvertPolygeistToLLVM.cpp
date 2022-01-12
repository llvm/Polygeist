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
#include "PassDetails.h"

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
#include "polygeist/Ops.h"

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

    if (!subViewOp.source().getType().isa<MemRefType>()) {
      llvm::errs() << " func: " << subViewOp->getParentOfType<FuncOp>() << "\n";
      llvm::errs() << " sub: " << subViewOp << " - " << subViewOp.source()
                   << "\n";
    }
    auto sourceMemRefType = subViewOp.source().getType().cast<MemRefType>();

    auto viewMemRefType = subViewOp.getType().cast<MemRefType>();

    MemRefDescriptor targetMemRef(transformed.source());
    Value prev = targetMemRef.alignedPtr(rewriter, loc);
    Value idxs[] = {transformed.index()};

    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;

    if (sourceMemRefType.getShape().size() !=
        viewMemRefType.getShape().size()) {
      if (sourceMemRefType.getShape().size() !=
          viewMemRefType.getShape().size() + 1) {
        return failure();
      }
      size_t sz = 1;
      for (size_t i = 1; i < sourceMemRefType.getShape().size(); i++) {
        if (sourceMemRefType.getShape()[i] == ShapedType::kDynamicSize)
          return failure();
        sz *= sourceMemRefType.getShape()[i];
      }
      Value cop = rewriter.create<LLVM::ConstantOp>(
          loc, idxs[0].getType(),
          rewriter.getIntegerAttr(idxs[0].getType(), sz));
      idxs[0] = rewriter.create<LLVM::MulOp>(loc, idxs[0], cop);
      for (size_t i = 1; i < sourceMemRefType.getShape().size(); i++) {
        sizes.push_back(targetMemRef.size(rewriter, loc, i));
        strides.push_back(targetMemRef.stride(rewriter, loc, i));
      }
    } else {
      for (size_t i = 0; i < sourceMemRefType.getShape().size(); i++) {
        sizes.push_back(targetMemRef.size(rewriter, loc, i));
        strides.push_back(targetMemRef.stride(rewriter, loc, i));
      }
    }

    // nexRef.setOffset(targetMemRef.offset());
    // nexRef.setSize(targetMemRef.size());
    // nexRef.setStride(targetMemRef.stride());

    if (false) {
      Value baseOffset = targetMemRef.offset(rewriter, loc);
      Value stride = targetMemRef.stride(rewriter, loc, 0);
      Value offset = transformed.index();
      Value mul = rewriter.create<LLVM::MulOp>(loc, offset, stride);
      baseOffset = rewriter.create<LLVM::AddOp>(loc, baseOffset, mul);
      targetMemRef.setOffset(rewriter, loc, baseOffset);
    }

    MemRefDescriptor nexRef = createMemRefDescriptor(
        loc, subViewOp.getType(), targetMemRef.allocatedPtr(rewriter, loc),
        rewriter.create<LLVM::GEPOp>(loc, prev.getType(), prev, idxs), sizes,
        strides, rewriter);

    rewriter.replaceOp(subViewOp, {nexRef});
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

struct TypeSizeOpLowering : public ConvertOpToLLVMPattern<TypeSizeOp> {
  using ConvertOpToLLVMPattern<TypeSizeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TypeSizeOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {

    Type NT = op.sourceAttr().getValue();
    if (auto T = getTypeConverter()->convertType(NT)) {
      NT = T;
    }
    assert(NT);

    auto type = getTypeConverter()->convertType(op.getType());

    if (NT.isa<IntegerType, FloatType>() || LLVM::isCompatibleType(NT)) {
      DataLayout DLI(op->getParentOfType<ModuleOp>());
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, type, rewriter.getIntegerAttr(type, DLI.getTypeSize(NT)));
      return success();
    }

    if (NT != op.sourceAttr().getValue() || type != op.getType()) {
      rewriter.replaceOpWithNewOp<TypeSizeOp>(op, type, NT);
      return success();
    }
    return failure();
  }
};

void populatePolygeistToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<TypeSizeOpLowering>(converter);
  patterns.add<SubIndexOpLowering>(converter);
  patterns.add<Memref2PointerOpLowering>(converter);
  patterns.add<Pointer2MemrefOpLowering>(converter);
  // clang-format on
}

namespace {
struct LLVMOpLowering : public ConversionPattern {
  explicit LLVMOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(), 1,
                          &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = getTypeConverter();
    SmallVector<Type> convertedResultTypes;
    if (failed(converter->convertTypes(op->getResultTypes(),
                                       convertedResultTypes))) {
      return failure();
    }
    SmallVector<Type> convertedOperandTypes;
    if (failed(converter->convertTypes(op->getOperandTypes(),
                                       convertedOperandTypes))) {
      return failure();
    }
    if (convertedResultTypes == op->getResultTypes() &&
        convertedOperandTypes == op->getOperandTypes()) {
      return failure();
    }
    if (isa<UnrealizedConversionCastOp>(op))
      return failure();

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(convertedResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();

    Operation *rewritten = rewriter.createOperation(state);
    rewriter.replaceOp(op, rewritten->getResults());

    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      rewriter.inlineRegionBefore(op->getRegion(i), rewritten->getRegion(i),
                                  rewritten->getRegion(i).begin());

    return success();
  }
};

struct URLLVMOpLowering
    : public ConvertOpToLLVMPattern<UnrealizedConversionCastOp> {
  using ConvertOpToLLVMPattern<
      UnrealizedConversionCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getResult(0).getType() != op->getOperand(0).getType())
      return failure();

    rewriter.replaceOp(op, op->getOperands());
    return success();
  }
};

struct GlobalOpTypeConversion : public OpConversionPattern<LLVM::GlobalOp> {
  explicit GlobalOpTypeConversion(LLVMTypeConverter &converter)
      : OpConversionPattern<LLVM::GlobalOp>(converter,
                                            &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(LLVM::GlobalOp op, LLVM::GlobalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = getTypeConverter();
    Type globalType = adaptor.getGlobalType();
    Type convertedType = converter->convertType(globalType);
    if (!convertedType)
      return failure();
    if (convertedType == globalType)
      return failure();

    rewriter.updateRootInPlace(
        op, [&]() { op.setGlobalTypeAttr(TypeAttr::get(convertedType)); });
    return success();
  }
};

struct ReturnOpTypeConversion : public ConvertOpToLLVMPattern<LLVM::ReturnOp> {
  using ConvertOpToLLVMPattern<LLVM::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LLVM::ReturnOp op, LLVM::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement =
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getArgs());
    replacement->setAttrs(adaptor.getAttributes());
    return success();
  }
};

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
    patterns
        .add<LLVMOpLowering, GlobalOpTypeConversion, ReturnOpTypeConversion>(
            converter);
    patterns.add<URLLVMOpLowering>(converter);

    LLVMConversionTarget target(getContext());
    target.addDynamicallyLegalOp<omp::ParallelOp, omp::WsLoopOp>(
        [&](Operation *op) { return converter.isLegal(&op->getRegion(0)); });
    target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                      omp::BarrierOp, omp::TaskwaitOp>();
    target.addDynamicallyLegalDialect<LLVM::LLVMDialect>(
        [&](Operation *op) -> Optional<bool> {
          SmallVector<Type> convertedResultTypes;
          if (failed(converter.convertTypes(op->getResultTypes(),
                                            convertedResultTypes)))
            return llvm::None;
          SmallVector<Type> convertedOperandTypes;
          if (failed(converter.convertTypes(op->getOperandTypes(),
                                            convertedOperandTypes)))
            return llvm::None;
          return convertedResultTypes == op->getResultTypes() &&
                 convertedOperandTypes == op->getOperandTypes();
        });
    target.addDynamicallyLegalOp<LLVM::GlobalOp>(
        [&](LLVM::GlobalOp op) -> Optional<bool> {
          if (converter.convertType(op.getGlobalType()) == op.getGlobalType())
            return true;
          return llvm::None;
        });
    target.addDynamicallyLegalOp<LLVM::ReturnOp>(
        [&](LLVM::ReturnOp op) -> Optional<bool> {
          if (!isa<LLVM::GlobalOp>(op->getParentOp()))
            return llvm::None;
          SmallVector<Type> convertedOperandTypes;
          if (failed(converter.convertTypes(op->getOperandTypes(),
                                            convertedOperandTypes)))
            return llvm::None;
          return convertedOperandTypes == op->getOperandTypes();
        });
    target.addIllegalOp<UnrealizedConversionCastOp>();
    /*
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](Operation *op) { return op->getOperand(0).getType() !=
    op->getResult(0).getType(); });
        */
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

std::unique_ptr<Pass> mlir::polygeist::createConvertPolygeistToLLVMPass() {
  // TODO: meaningful arguments to this pass should be specified as
  // Option<...>'s to the pass in Passes.td. For now, we'll provide some dummy
  // default values to allow for pass creation.
  auto dl = llvm::DataLayout("");
  return std::make_unique<ConvertPolygeistToLLVMPass>(true, true, 64u, true,
                                                      dl);
}

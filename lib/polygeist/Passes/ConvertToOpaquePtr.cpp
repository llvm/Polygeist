#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "polygeist/Passes/Passes.h"
#include "polygeist/Passes/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

using namespace mlir;
using namespace polygeist;

namespace {

static constexpr const char todoAttr[] = "polygeist.to.opaque.todo";
static constexpr const char kElemTypeAttrName[] = "elem_type";

static LogicalResult convertPtrsToOpaque(Operation *op, Operation *&rewritten,
                                         TypeAttr attr, ValueRange operands,
                                         ConversionPatternRewriter &rewriter,
                                         const TypeConverter *converter) {
  SmallVector<Type> convertedOperandTypes;
  if (converter->convertTypes(op->getOperandTypes(), convertedOperandTypes)
          .failed()) {
    return failure();
  }
  SmallVector<Type> convertedResultTypes;
  if (failed(converter->convertTypes(op->getResultTypes(),
                                     convertedResultTypes))) {
    return failure();
  }

  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  state.addTypes(convertedResultTypes);
  state.addAttributes(op->getAttrs());
  if (attr)
    state.addAttribute(kElemTypeAttrName, attr);
  state.addSuccessors(op->getSuccessors());
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
    state.addRegion();

  rewriter.setInsertionPoint(op);
  rewritten = rewriter.create(state);
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
    rewriter.inlineRegionBefore(op->getRegion(i), rewritten->getRegion(i),
                                rewritten->getRegion(i).begin());
  return success();
}

struct OpConversion : public ConversionPattern {
  const TypeConverter *typeConverter;
  OpConversion(const TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(), 1, ctx),
        typeConverter(&converter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Operation *rewritten;
    TypeAttr elty = nullptr;
    if (convertPtrsToOpaque(op, rewritten, elty, operands, rewriter,
                            typeConverter)
            .failed())
      return failure();
    rewriter.replaceOp(op, rewritten->getResults());
    rewritten->removeAttr(todoAttr);
    return success();
  }
};

struct FuncOpConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    TypeConverter::SignatureConversion signatureConversion(
        funcOp.getNumArguments());
    for (const auto &[index, type] :
         llvm::enumerate(funcOp.getArgumentTypes())) {
      Type converted = getTypeConverter()->convertType(type);
      if (!converted)
        return failure();
      signatureConversion.addInputs(index, converted);
    }
    SmallVector<Type> convertedResultTypes;
    if (getTypeConverter()
            ->convertTypes(funcOp.getFunctionType().getResults(),
                           convertedResultTypes)
            .failed())
      return failure();
    auto convertedType = FunctionType::get(
        rewriter.getContext(), signatureConversion.getConvertedTypes(),
        convertedResultTypes);

    auto newFuncOp = rewriter.create<func::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), convertedType,
        funcOp.getSymVisibilityAttr(), funcOp.getArgAttrsAttr(),
        funcOp.getResAttrsAttr());
    newFuncOp->setDiscardableAttrs(funcOp->getDiscardableAttrs());

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &signatureConversion))) {
      return rewriter.notifyMatchFailure(
          funcOp->getLoc(), "failed to apply signature conversion");
    }

    rewriter.eraseOp(funcOp);
    newFuncOp->removeAttr(todoAttr);
    return success();
  }
};

struct LLVMFuncOpConversion : public OpConversionPattern<LLVM::LLVMFuncOp> {
  using OpConversionPattern<LLVM::LLVMFuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    TypeConverter::SignatureConversion signatureConversion(
        funcOp.getNumArguments());
    for (const auto &[index, type] :
         llvm::enumerate(funcOp.getArgumentTypes())) {
      Type converted = getTypeConverter()->convertType(type);
      if (!converted)
        return failure();
      signatureConversion.addInputs(index, converted);
    }
    Type resultType = getTypeConverter()->convertType(
        funcOp.getFunctionType().getReturnType());
    auto convertedType = LLVM::LLVMFunctionType::get(
        resultType, signatureConversion.getConvertedTypes(),
        funcOp.getFunctionType().isVarArg());

    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getNameAttr(), convertedType,
        funcOp.getLinkage(), funcOp.getDsoLocal(), funcOp.getCConv(),
        funcOp.getComdatAttr(), funcOp->getDiscardableAttrs());
    newFuncOp->setDiscardableAttrs(funcOp->getDiscardableAttrs());

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &signatureConversion))) {
      return rewriter.notifyMatchFailure(
          funcOp->getLoc(), "failed to apply signature conversion");
    }

    rewriter.eraseOp(funcOp);
    newFuncOp->removeAttr(todoAttr);
    return success();
  }
};

struct AllocaConversion : public OpConversionPattern<LLVM::AllocaOp> {
  using OpConversionPattern<LLVM::AllocaOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Operation *rewritten;
    auto resTy = op.getRes().getType();
    assert(!resTy.isOpaque());
    TypeAttr elty =
        TypeAttr::get(getTypeConverter()->convertType(resTy.getElementType()));
    if (convertPtrsToOpaque(op, rewritten, elty, adaptor.getOperands(),
                            rewriter, getTypeConverter())
            .failed())
      return failure();
    rewriter.replaceOp(op, rewritten->getResults());
    return success();
  }
};

struct GEPConversion : public OpConversionPattern<LLVM::GEPOp> {
  using OpConversionPattern<LLVM::GEPOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::GEPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Operation *rewritten;
    TypeAttr elty = nullptr;
    if (!op->getAttr(kElemTypeAttrName))
      elty = TypeAttr::get(getTypeConverter()->convertType(
          dyn_cast<LLVM::LLVMPointerType>(op.getOperand(0).getType())
              .getElementType()));
    if (convertPtrsToOpaque(op, rewritten, elty, adaptor.getOperands(),
                            rewriter, getTypeConverter())
            .failed())
      return failure();
    rewriter.replaceOp(op, rewritten->getResults());
    assert(op.getResult().getType() != rewriter.getI32Type());
    rewritten->removeAttr(todoAttr);
    return success();
  }
};

struct ConvertToOpaquePtrPass
    : public ConvertToOpaquePtrPassBase<ConvertToOpaquePtrPass> {
  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (!isa<ModuleOp>(op))
        op->setAttr(todoAttr, UnitAttr::get(getOperation()->getContext()));
    });
    auto isOpOpaque = [&](Operation *op) { return !op->hasAttr(todoAttr); };
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (isa<ModuleOp>(op))
        return true;
      else
        return isOpOpaque(op);
    });

    std::map<StringRef, LLVM::LLVMStructType> typeCache;
    TypeConverter converter;
    converter.addConversion([&](Type ty) -> Type {
      if (auto pt = dyn_cast<LLVM::LLVMPointerType>(ty)) {
        return LLVM::LLVMPointerType::get(pt.getContext(),
                                          pt.getAddressSpace());
      } else if (auto mt = dyn_cast<MemRefType>(ty)) {
        return MemRefType::get(mt.getShape(),
                               converter.convertType(mt.getElementType()),
                               mt.getLayout(), mt.getMemorySpace());
      } else if (auto st = dyn_cast<LLVM::LLVMStructType>(ty)) {
        StringRef key = "";
        if (st.isIdentified()) {
          key = st.getName();
          if (typeCache.find(key) != typeCache.end()) {
            return typeCache[key];
          }
        }
        SmallVector<Type> bodyTypes;
        if (st.isIdentified()) {
          typeCache[key] = LLVM::LLVMStructType::getIdentified(
              &getContext(), "opaque@" + st.getName().str());
        }
        for (auto ty : st.getBody()) {
          StringRef fieldKey = "";
          if (auto fieldST = dyn_cast<LLVM::LLVMStructType>(ty)) {
            if (fieldST.isIdentified())
              fieldKey = fieldST.getName();
          }
          if (typeCache.find(fieldKey) != typeCache.end()) {
            bodyTypes.push_back(typeCache[fieldKey]);
          } else {
            bodyTypes.push_back(converter.convertType(ty));
          }
        }
        if (st.isIdentified()) {
          auto res = typeCache[key].setBody(bodyTypes, st.isPacked());
          assert(res.succeeded());
          return typeCache[key];
        } else {
          return LLVM::LLVMStructType::getLiteral(&getContext(), bodyTypes,
                                                  st.isPacked());
        }
      } else if (auto at = dyn_cast<LLVM::LLVMArrayType>(ty)) {
        return LLVM::LLVMArrayType::get(
            converter.convertType(at.getElementType()), at.getNumElements());
      } else {
        return ty;
      }
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<GEPConversion, AllocaConversion, OpConversion,
                 FuncOpConversion, LLVMFuncOpConversion>(converter,
                                                         &getContext());
    (void)(applyPartialConversion(getOperation(), target, std::move(patterns)));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::polygeist::createConvertToOpaquePtrPass() {
  return std::make_unique<ConvertToOpaquePtrPass>();
}

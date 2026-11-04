//===--- CGCall.cpp - Encapsulate calling convention details --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HLSPragma.h"
#include "TypeUtils.h"
#include "clang-mlir.h"
#include "XlxMathCore.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "utils.h"
#include "clang/Basic/Builtins.h"

#define DEBUG_TYPE "CGCall"

using namespace mlir;
using namespace std;
using namespace mlir::arith;
using namespace mlir::func;
using namespace mlirclang;

extern llvm::cl::opt<bool> XlxPrelude; // driver.cc

extern llvm::cl::opt<bool> CStyleMemRef;

/// Try to typecast the caller arg of type MemRef to fit the corresponding
/// callee arg type. We only deal with the cast where src and dst have the same
/// shape size and elem type, and just the first shape differs: src has -1 and
/// dst has a constant integer.
static mlir::Value castCallerMemRefArg(mlir::Value callerArg,
                                       mlir::Type calleeArgType,
                                       mlir::OpBuilder &b) {
  mlir::OpBuilder::InsertionGuard guard(b);
  mlir::Type callerArgType = callerArg.getType();

  if (MemRefType dstTy = dyn_cast_or_null<MemRefType>(calleeArgType)) {
    MemRefType srcTy = dyn_cast<MemRefType>(callerArgType);
    if (srcTy && dstTy.getElementType() == srcTy.getElementType() &&
        dstTy.getMemorySpace() == srcTy.getMemorySpace()) {
      auto srcShape = srcTy.getShape();
      auto dstShape = dstTy.getShape();

      if (srcShape.size() == dstShape.size() && !srcShape.empty() &&
          srcShape[0] == ShapedType::kDynamic &&
          std::equal(std::next(srcShape.begin()), srcShape.end(),
                     std::next(dstShape.begin()))) {
        // A static array seen through a cast to a dynamic first dimension
        // (memref.get_global or memref.alloca, then memref.cast to ?xN) whose own
        // first dimension differs from the parameter's: the chain of two
        // casts folds into an invalid one, so with --hls-keep-misfit-inline-off
        // (the call may stay) the static array is reinterpreted with the
        // parameter's shape, as below (forgebench attn_breakdown:
        // matmul_large(.., Q_weight_buffer[32][32], ..) with a [8][32]
        // parameter; OPEN_ISSUES O167 in the hls-est tree).
        mlir::Value staticSrc;
        if (cgeist::hls::keepMisfitInlineOff())
          for (auto cast = callerArg.getDefiningOp<mlir::memref::CastOp>();
               cast; cast = cast.getSource()
                                .getDefiningOp<mlir::memref::CastOp>()) {
            auto t = dyn_cast<MemRefType>(cast.getSource().getType());
            if (t && t.hasStaticShape()) {
              // Only a whole array (a global's memref.get_global, a local
              // alloca, through casts that keep its type): a row view keeps
              // its cast, as before.
              mlir::Value whole = cast.getSource();
              while (auto same = whole.getDefiningOp<mlir::memref::CastOp>()) {
                if (same.getSource().getType() != whole.getType())
                  break;
                whole = same.getSource();
              }
              if (whole.getDefiningOp<mlir::memref::GetGlobalOp>() ||
                  whole.getDefiningOp<mlir::memref::AllocaOp>())
                staticSrc = whole;
              break;
            }
          }
        if (staticSrc) {
          auto staticTy = cast<MemRefType>(staticSrc.getType());
          if (dstTy.hasStaticShape() &&
              staticTy.getElementType() == dstTy.getElementType() &&
              staticTy.getRank() == dstTy.getRank() &&
              staticTy.getShape()[0] != dstShape[0] &&
              staticTy.getLayout().isIdentity() &&
              dstTy.getLayout().isIdentity()) {
            b.setInsertionPointAfterValue(callerArg);
            SmallVector<int64_t> strides(dstShape.size(), 1);
            for (int k = static_cast<int>(dstShape.size()) - 2; k >= 0; --k)
              strides[k] = strides[k + 1] * dstShape[k + 1];
            return b.create<mlir::memref::ReinterpretCastOp>(
                callerArg.getLoc(), dstTy, staticSrc, /*offset*/ 0,
                dstShape, strides);
          }
        }
        b.setInsertionPointAfterValue(callerArg);

        return b.create<mlir::memref::CastOp>(callerArg.getLoc(), calleeArgType,
                                              callerArg);
      }
      // A C array argument whose static shape differs from the parameter's
      // (C ignores the first dimension of an array parameter, and a `T p[]`
      // or `T *p` parameter takes any array): `g(x)` with `short x[4096]` and
      // `void g(short a[])`, or `f(z)` with `short z[16]` and
      // `void f(short a[4096])` (W8b, hls-doc/hlsfactory-v2.md class
      // cgeist-invalid-mlir). A cast towards a dynamic shape is a valid
      // memref.cast. Two static shapes that differ are not cast-compatible,
      // and a chain of two casts through a dynamic shape is folded back into
      // that invalid cast by memref.cast's folder, so the argument is
      // reinterpreted with the parameter's own shape and row-major strides,
      // which is how C addresses the parameter. Once the call is inlined, the
      // accesses go back to the argument itself when only the first
      // dimension differs (populateArrayArgumentPatterns).
      // A pointer to an array, `(T (*)[8][8]) p`, is a memref of the array's
      // own shape (memref<8x8xT>: cgeist's type for a pointer to an array is
      // the pointee's), passed to a parameter `T a[4][8][8]` one rank higher
      // with the same trailing shape. C addresses the parameter through the
      // pointer, row-major from its first element, so the argument is
      // re-typed through its address (class_ece8893_2023_spring GradCAM
      // src_hls, W8b; it was an invalid func.call).
      if (srcTy && srcShape.size() + 1 == dstShape.size() &&
          std::equal(srcShape.begin(), srcShape.end(),
                     std::next(dstShape.begin())) &&
          srcTy.getLayout().isIdentity() && dstTy.getLayout().isIdentity()) {
        b.setInsertionPointAfterValue(callerArg);
        mlir::Location loc = callerArg.getLoc();
        mlir::Value ptr = b.create<polygeist::Memref2PointerOp>(
            loc, LLVM::LLVMPointerType::get(b.getI8Type()), callerArg);
        return b.create<polygeist::Pointer2MemrefOp>(loc, dstTy, ptr);
      }
      if (srcTy && srcShape.size() == dstShape.size() && !srcShape.empty() &&
          srcTy.getLayout().isIdentity() && dstTy.getLayout().isIdentity()) {
        b.setInsertionPointAfterValue(callerArg);
        mlir::Location loc = callerArg.getLoc();
        if (mlir::memref::CastOp::areCastCompatible(srcTy, dstTy))
          return b.create<mlir::memref::CastOp>(loc, dstTy, callerArg);
        if (srcTy.hasStaticShape() && dstTy.hasStaticShape()) {
          SmallVector<int64_t> strides(dstShape.size(), 1);
          for (int k = static_cast<int>(dstShape.size()) - 2; k >= 0; --k)
            strides[k] = strides[k + 1] * dstShape[k + 1];
          return b.create<mlir::memref::ReinterpretCastOp>(
              loc, dstTy, callerArg, /*offset*/ 0, dstShape, strides);
        }
      }
    }
  }

  // Return the original value when casting fails.
  return callerArg;
}

/// Typecast the caller args to match the callee's signature. Mismatches that
/// cannot be resolved by given rules won't raise exceptions, e.g., if the
/// expected type for an arg is memref<10xi8> while the provided is
/// memref<20xf32>, we will simply ignore the case in this function and wait for
/// the rest of the pipeline to detect it.
static void castCallerArgs(mlir::func::FuncOp callee,
                           llvm::SmallVectorImpl<mlir::Value> &args,
                           mlir::OpBuilder &b) {
  mlir::FunctionType funcTy = callee.getFunctionType();
  assert(args.size() == funcTy.getNumInputs() &&
         "The caller arguments should have the same size as the number of "
         "callee arguments as the interface.");

  for (unsigned i = 0; i < args.size(); ++i) {
    mlir::Type calleeArgType = funcTy.getInput(i);
    mlir::Type callerArgType = args[i].getType();

    if (calleeArgType == callerArgType)
      continue;

    if (calleeArgType.isa<MemRefType>())
      args[i] = castCallerMemRefArg(args[i], calleeArgType, b);
  }
}

ValueCategory MLIRScanner::CallHelper(
    mlir::func::FuncOp tocall, QualType objType,
    ArrayRef<std::pair<ValueCategory, clang::Expr *>> arguments,
    QualType retType, bool retReference, clang::Expr *expr) {
  SmallVector<mlir::Value, 4> args;
  auto fnType = tocall.getFunctionType();

  auto loc = getMLIRLocation(expr->getExprLoc());

  size_t i = 0;
  // map from declaration name to mlir::value
  std::map<std::string, mlir::Value> mapFuncOperands;

  for (auto pair : arguments) {

    ValueCategory arg = std::get<0>(pair);
    auto *a = std::get<1>(pair);
    if (!arg.val) {
      expr->dump();
      a->dump();
    }
    assert(arg.val && "expect not null");

    if (auto *ice = dyn_cast_or_null<ImplicitCastExpr>(a))
      if (auto *dre = dyn_cast<DeclRefExpr>(ice->getSubExpr()))
        mapFuncOperands.insert(
            make_pair(dre->getDecl()->getName().str(), arg.val));

    if (i >= fnType.getInputs().size() || (i != 0 && a == nullptr)) {
      expr->dump();
      tocall.dump();
      fnType.dump();
      for (auto a : arguments) {
        std::get<1>(a)->dump();
      }
      llvm_unreachable("too many arguments in calls");
    }
    bool isReference =
        (i == 0 && a == nullptr) || a->isLValue() || a->isXValue();

    bool isArray = false;
    QualType aType = (i == 0 && a == nullptr) ? objType : a->getType();

    auto expectedType = Glob.getMLIRType(aType, &isArray);
    if (auto PT = dyn_cast<LLVM::LLVMPointerType>(arg.val.getType())) {
      if (PT.getAddressSpace() == 5)
        arg.val = builder.create<LLVM::AddrSpaceCastOp>(
            loc, LLVM::LLVMPointerType::get(PT.getElementType(), 0), arg.val);
    }

    mlir::Value val = nullptr;
    if (!isReference) {
      if (isArray) {
        if (!arg.isReference) {
          expr->dump();
          a->dump();
          llvm::errs() << " v: " << arg.val << "\n";
        }
        assert(arg.isReference);

        auto mt = Glob.getMLIRType(
                          Glob.CGM.getContext().getLValueReferenceType(aType))
                      .cast<MemRefType>();
        auto shape = std::vector<int64_t>(mt.getShape());
        assert(shape.size() == 2);

        auto pshape = shape[0];
        if (pshape == ShapedType::kDynamic)
          shape[0] = 1;

        OpBuilder abuilder(builder.getContext());
        abuilder.setInsertionPointToStart(allocationScope);
        auto alloc = abuilder.create<mlir::memref::AllocaOp>(
            loc, mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace()));
        ValueCategory(alloc, /*isRef*/ true)
            .store(loc, builder, arg, /*isArray*/ isArray);
        shape[0] = pshape;
        val = builder.create<mlir::memref::CastOp>(
            loc,
            mlir::MemRefType::get(shape, mt.getElementType(),
                                  MemRefLayoutAttrInterface(),
                                  mt.getMemorySpace()),
            alloc);
      } else {
        val = arg.getValue(loc, builder);
        if (val.getType().isa<LLVM::LLVMPointerType>() &&
            expectedType.isa<MemRefType>()) {
          val = builder.create<polygeist::Pointer2MemrefOp>(loc, expectedType,
                                                            val);
        }
        if (auto prevTy = dyn_cast<mlir::IntegerType>(val.getType())) {
          // The argument is converted to the parameter's integer width
          // explicitly (narrowing truncates; widening extends by the C
          // signedness of the converted argument).
          // (--scalar-wrapper-abi; with the option off, any mismatch is a
          // trunci as before.)
          auto ipostTy = expectedType.cast<mlir::IntegerType>();
          if (!ScalarWrapperABI) {
            if (prevTy != ipostTy)
              val = builder.create<arith::TruncIOp>(loc, ipostTy, val);
          } else if (prevTy.getWidth() > ipostTy.getWidth())
            val = builder.create<arith::TruncIOp>(loc, ipostTy, val);
          else if (prevTy.getWidth() < ipostTy.getWidth()) {
            if (aType->isUnsignedIntegerType() || prevTy.getWidth() == 1)
              val = builder.create<arith::ExtUIOp>(loc, ipostTy, val);
            else
              val = builder.create<arith::ExtSIOp>(loc, ipostTy, val);
          }
        }
      }
    } else {
      assert(arg.isReference);

      expectedType =
          Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(aType));

      val = arg.val;
      if (arg.val.getType().isa<LLVM::LLVMPointerType>() &&
          expectedType.isa<MemRefType>()) {
        val =
            builder.create<polygeist::Pointer2MemrefOp>(loc, expectedType, val);
      }
    }
    assert(val);
    args.push_back(val);
    i++;
  }

  // handle lowerto pragma.
  if (LTInfo.SymbolTable.count(tocall.getName())) {
    SmallVector<mlir::Value> inputOperands;
    SmallVector<mlir::Value> outputOperands;
    for (StringRef input : LTInfo.InputSymbol)
      if (mapFuncOperands.find(input.str()) != mapFuncOperands.end())
        inputOperands.push_back(mapFuncOperands[input.str()]);
    for (StringRef output : LTInfo.OutputSymbol)
      if (mapFuncOperands.find(output.str()) != mapFuncOperands.end())
        outputOperands.push_back(mapFuncOperands[output.str()]);

    if (inputOperands.size() == 0)
      inputOperands.append(args);
    auto replaced = mlirclang::replaceFuncByOperation(
        tocall, LTInfo.SymbolTable[tocall.getName()], builder, inputOperands,
        outputOperands);
    if (replaced->getNumResults() == 0)
      return ValueCategory();
    else
      return ValueCategory(replaced->getResult(0),
                           /*isReference=*/false);
  }

  bool isArrayReturn = false;
  if (!retReference)
    Glob.getMLIRType(retType, &isArrayReturn);

  mlir::Value alloc;
  if (isArrayReturn) {
    auto mt =
        Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(retType))
            .cast<MemRefType>();

    auto shape = std::vector<int64_t>(mt.getShape());
    assert(shape.size() == 2);

    auto pshape = shape[0];
    if (pshape == ShapedType::kDynamic)
      shape[0] = 1;

    OpBuilder abuilder(builder.getContext());
    abuilder.setInsertionPointToStart(allocationScope);
    alloc = abuilder.create<mlir::memref::AllocaOp>(
        loc, mlir::MemRefType::get(shape, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace()));
    shape[0] = pshape;
    alloc = builder.create<mlir::memref::CastOp>(
        loc,
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace()),
        alloc);
    args.push_back(alloc);
  }

  if (auto *CU = dyn_cast<CUDAKernelCallExpr>(expr)) {
    auto l0 = Visit(CU->getConfig()->getArg(0));
    assert(l0.isReference);
    mlir::Value blocks[3];
    mlir::Value val = l0.val;
    if (auto MT = dyn_cast<MemRefType>(val.getType())) {
      if (MT.getElementType().isa<LLVM::LLVMStructType>() &&
          MT.getShape().size() == 1) {
        val = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(MT.getElementType(),
                                       MT.getMemorySpaceAsInt()),
            val);
      }
    }
    for (int i = 0; i < 3; i++) {
      if (auto MT = dyn_cast<MemRefType>(val.getType())) {
        mlir::Value idx[] = {getConstantIndex(0), getConstantIndex(i)};
        assert(MT.getShape().size() == 2);
        blocks[i] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<mlir::memref::LoadOp>(loc, val, idx));
      } else {
        mlir::Value idx[] = {builder.create<arith::ConstantIntOp>(loc, 0, 32),
                             builder.create<arith::ConstantIntOp>(loc, i, 32)};
        auto PT = val.getType().cast<LLVM::LLVMPointerType>();
        auto ET = PT.getElementType().cast<LLVM::LLVMStructType>().getBody()[i];
        blocks[i] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<LLVM::LoadOp>(
                loc,
                builder.create<LLVM::GEPOp>(
                    loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                    val, idx)));
      }
    }

    auto t0 = Visit(CU->getConfig()->getArg(1));
    assert(t0.isReference);
    mlir::Value threads[3];
    val = t0.val;
    if (auto MT = dyn_cast<MemRefType>(val.getType())) {
      if (MT.getElementType().isa<LLVM::LLVMStructType>() &&
          MT.getShape().size() == 1) {
        val = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(MT.getElementType(),
                                       MT.getMemorySpaceAsInt()),
            val);
      }
    }
    for (int i = 0; i < 3; i++) {
      if (auto MT = dyn_cast<MemRefType>(val.getType())) {
        mlir::Value idx[] = {getConstantIndex(0), getConstantIndex(i)};
        assert(MT.getShape().size() == 2);
        threads[i] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<mlir::memref::LoadOp>(loc, val, idx));
      } else {
        mlir::Value idx[] = {builder.create<arith::ConstantIntOp>(loc, 0, 32),
                             builder.create<arith::ConstantIntOp>(loc, i, 32)};
        auto PT = val.getType().cast<LLVM::LLVMPointerType>();
        auto ET = PT.getElementType().cast<LLVM::LLVMStructType>().getBody()[i];
        threads[i] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<LLVM::LoadOp>(
                loc,
                builder.create<LLVM::GEPOp>(
                    loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                    val, idx)));
      }
    }
    mlir::Value stream = nullptr;
    SmallVector<mlir::Value, 1> asyncDependencies;
    if (3 < CU->getConfig()->getNumArgs() &&
        !isa<CXXDefaultArgExpr>(CU->getConfig()->getArg(3))) {
      stream = Visit(CU->getConfig()->getArg(3)).getValue(loc, builder);
      stream = builder.create<polygeist::StreamToTokenOp>(
          loc, builder.getType<gpu::AsyncTokenType>(), stream);
      assert(stream);
      asyncDependencies.push_back(stream);
    }
    auto op = builder.create<mlir::gpu::LaunchOp>(
        loc, blocks[0], blocks[1], blocks[2], threads[0], threads[1],
        threads[2],
        /*dynamic shmem size*/ nullptr,
        /*token type*/ stream ? stream.getType() : nullptr,
        /*dependencies*/ asyncDependencies);
    auto oldpoint = builder.getInsertionPoint();
    auto *oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&op.getRegion().front());
    builder.create<CallOp>(loc, tocall, args);
    builder.create<gpu::TerminatorOp>(loc);
    builder.setInsertionPoint(oldblock, oldpoint);
    return nullptr;
  }

  // Try to rescue some mismatched types.
  castCallerArgs(tocall, args, builder);

  auto op = builder.create<CallOp>(loc, tocall, args);

  // (--xlx-headers) A compound assignment (`/=`, `*=`, `+=`, `-=`, `++`,
  // `--`) to an ap_fixed object: marked on the call, and by the inliner on
  // the stores of its body (HLSPragma.cc). Vitis keeps two stores per bank for it when the
  // array is partitioned and the bank index is not constant: the
  // `if (!op) V = 0` at the head of the cross-format ap_fixed_base::operator=
  // and the final store (CIRCT VITIS_NOTES N38 R-cmpd, kernels w9r_d1div /
  // d2mul / d3add; a plain `A = w * A` replicates, w9r_d4asg). After cgeist's
  // own store forwarding only the final store is left, so hls-est needs the
  // mark to tell the two apart.
  if (cgeist::hls::xlxHeaders())
    if (auto *oc = dyn_cast_or_null<clang::CXXOperatorCallExpr>(expr)) {
      clang::OverloadedOperatorKind k = oc->getOperator();
      if ((k == clang::OO_SlashEqual || k == clang::OO_StarEqual ||
           k == clang::OO_PlusEqual || k == clang::OO_MinusEqual ||
           // `++` / `--` go through operator+= / -= and the same
           // cross-format operator= (Vitis: "store ... of constant 0", II 2;
           // kernel w9r7rev_rc_inc, W9r7 review defect 5)
           k == clang::OO_PlusPlus || k == clang::OO_MinusMinus) &&
          oc->getNumArgs() > 0)
        if (const auto *rd =
                oc->getArg(0)->getType()->getAsCXXRecordDecl())
          if (cgeist::hls::isApFixedClass(rd))
            op->setAttr("hls.ap_compound_assign", builder.getUnitAttr());
    }

  // (--xlx-headers) A call of an hls:: math function of hls_math.h is a core
  // of the Vitis math library (XlxMathCore.h): marked on the call and on the
  // callee, which is then never inlined.
  if (auto *ce = dyn_cast_or_null<clang::CallExpr>(expr))
    if (auto core = cgeist::xlx::mathCoreOf(ce->getDirectCallee(),
                                            Glob.CGM.getContext())) {
      SmallVector<NamedAttribute, 2> fields = {builder.getNamedAttr(
          "function", builder.getStringAttr(core->function))};
      if (!core->type.empty())
        fields.push_back(
            builder.getNamedAttr("type", builder.getStringAttr(core->type)));
      auto dict = builder.getDictionaryAttr(fields);
      op->setAttr(cgeist::xlx::kMathCoreAttr, dict);
      tocall->setAttr(cgeist::xlx::kMathCoreAttr, dict);
    }

  if (isArrayReturn) {
    // TODO remedy return
    if (retReference)
      expr->dump();
    assert(!retReference);
    return ValueCategory(alloc, /*isReference*/ true);
  } else if (op->getNumResults()) {
    return ValueCategory(op->getResult(0),
                         /*isReference*/ retReference);
  } else
    return nullptr;
  llvm::errs() << "do not support indirecto call of " << tocall << "\n";
  llvm_unreachable("no indirect");
}

mlir::Value MLIRScanner::getLLVM(Expr *E, bool isRef) {
  auto loc = getMLIRLocation(E->getExprLoc());

  auto sub = Visit(E);
  if (!sub.val) {
    E->dump();
  }
  assert(sub.val);

  bool isReference = E->isLValue() || E->isXValue();
  if (isReference) {
    assert(sub.isReference);
    mlir::Value val = sub.val;
    if (auto mt = dyn_cast<MemRefType>(val.getType())) {
      val =
          builder.create<polygeist::Memref2PointerOp>(loc, getOpaquePtr(), val);
    } else if (auto pt = dyn_cast<LLVM::LLVMPointerType>(val.getType())) {
      if (!pt.isOpaque())
        val = builder.create<LLVM::BitcastOp>(loc, getOpaquePtr(), val);
    }
    return val;
  }

  bool isArray = false;
  Glob.getMLIRType(E->getType(), &isArray);

  if (isArray) {
    assert(sub.isReference);
    auto mt = Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                   E->getType()))
                  .cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    assert(shape.size() == 2);

    auto PT = LLVM::LLVMPointerType::get(
        Glob.typeTranslator.translateType(anonymize(getLLVMType(E->getType()))),
        0);
    if (true) {
      sub = ValueCategory(
          builder.create<polygeist::Memref2PointerOp>(loc, PT, sub.val),
          sub.isReference);
    } else {
      OpBuilder abuilder(builder.getContext());
      abuilder.setInsertionPointToStart(allocationScope);
      auto one = abuilder.create<ConstantIntOp>(loc, 1, 64);
      auto alloc = abuilder.create<mlir::LLVM::AllocaOp>(loc, PT, one, 0);
      ValueCategory(alloc, /*isRef*/ true)
          .store(loc, builder, sub, /*isArray*/ isArray);
      sub = ValueCategory(alloc, /*isRef*/ true);
    }
  }
  mlir::Value val;
  clang::QualType ct;
  if (!isRef) {
    val = sub.getValue(loc, builder);
    ct = E->getType();
  } else {
    if (!sub.isReference) {
      OpBuilder abuilder(builder.getContext());
      abuilder.setInsertionPointToStart(allocationScope);
      auto one = abuilder.create<ConstantIntOp>(loc, 1, 64);
      auto alloc = abuilder.create<mlir::LLVM::AllocaOp>(
          loc, LLVM::LLVMPointerType::get(builder.getContext()), one, 0);
      ValueCategory(alloc, /*isRef*/ true)
          .store(loc, builder, sub, /*isArray*/ isArray);
      sub = ValueCategory(alloc, /*isRef*/ true);
    }
    assert(sub.isReference);
    val = sub.val;
    ct = Glob.CGM.getContext().getLValueReferenceType(E->getType());
  }
  if (auto mt = dyn_cast<MemRefType>(val.getType())) {
    val = builder.create<polygeist::Memref2PointerOp>(loc, getOpaquePtr(), val);
  } else if (auto pt = dyn_cast<LLVM::LLVMPointerType>(val.getType())) {
    if (!pt.isOpaque())
      val = builder.create<LLVM::BitcastOp>(loc, getOpaquePtr(), val);
  }
  return val;
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitClangBuiltinCallExpr(clang::CallExpr *expr) {
  auto success = [&](auto v) { return make_pair(v, true); };
  auto failure = [&]() { return make_pair(ValueCategory(), false); };
  auto loc = getMLIRLocation(expr->getExprLoc());

  switch (expr->getBuiltinCallee()) {
  case clang::Builtin::BImove:
  case clang::Builtin::BImove_if_noexcept:
  case clang::Builtin::BIforward:
  case clang::Builtin::BIas_const: {
    auto V = Visit(expr->getArg(0));
    return make_pair(V, true);
  }
  case clang::Builtin::BIaddressof:
  case clang::Builtin::BI__addressof:
  case clang::Builtin::BI__builtin_addressof: {
    auto V = Visit(expr->getArg(0));
    assert(V.isReference);
    mlir::Value val = V.val;
    auto T = getMLIRType(expr->getType());
    if (T == val.getType())
      return make_pair(ValueCategory(val, /*isRef*/ false), true);
    if (T.isa<LLVM::LLVMPointerType>()) {
      if (val.getType().isa<MemRefType>())
        val = builder.create<polygeist::Memref2PointerOp>(loc, T, val);
      else if (T != val.getType())
        val = builder.create<LLVM::BitcastOp>(loc, T, val);
      return make_pair(ValueCategory(val, /*isRef*/ false), true);
    } else {
      assert(T.isa<MemRefType>());
      if (val.getType().isa<MemRefType>())
        val = builder.create<polygeist::Memref2PointerOp>(
            loc, LLVM::LLVMPointerType::get(builder.getI8Type()), val);
      if (val.getType().isa<LLVM::LLVMPointerType>())
        val = builder.create<polygeist::Pointer2MemrefOp>(loc, T, val);
      return make_pair(ValueCategory(val, /*isRef*/ false), true);
    }
    expr->dump();
    llvm::errs() << " val: " << val << " T: " << T << "\n";
    llvm_unreachable("unhandled builtin addressof");
  }
  case Builtin::BI__builtin_operator_new: {
    mlir::Value count = Visit(*expr->arg_begin()).getValue(loc, builder);
    count = builder.create<IndexCastOp>(
        loc, mlir::IndexType::get(builder.getContext()), count);
    auto ty = getMLIRType(expr->getType());
    mlir::Value alloc;
    if (auto mt = dyn_cast<mlir::MemRefType>(ty)) {
      auto shape = std::vector<int64_t>(mt.getShape());
      mlir::Value args[1] = {count};
      alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
    } else {
      auto PT = ty.cast<LLVM::LLVMPointerType>();
      alloc = builder.create<mlir::LLVM::BitcastOp>(
          loc, ty, Glob.CallMalloc(builder, loc, count));
    }
    return make_pair(ValueCategory(alloc, /*isRef*/ false), true);
  }
  case Builtin::BI__builtin_operator_delete: {
    mlir::Value toDelete = Visit(*expr->arg_begin()).getValue(loc, builder);
    if (toDelete.getType().isa<mlir::MemRefType>()) {
      builder.create<mlir::memref::DeallocOp>(loc, toDelete);
    } else {
      mlir::Value args[1] = {
          builder.create<LLVM::BitcastOp>(loc, getOpaquePtr(), toDelete)};
      builder.create<mlir::LLVM::CallOp>(loc, Glob.GetOrCreateFreeFunction(),
                                         args);
    }
    return make_pair(nullptr, true);
  }
  case Builtin::BI__builtin_constant_p: {
    auto resultType = getMLIRType(expr->getType());
    llvm::errs() << "warning: assuming __builtin_constant_p to be false\n";
    return make_pair(
        ValueCategory(builder.create<arith::ConstantIntOp>(loc, 0, resultType),
                      /*isRef*/ false),
        true);
  }
  case Builtin::BI__builtin_unreachable: {
    llvm::errs() << "warning: ignoring __builtin_unreachable\n";
    return make_pair(nullptr, true);
  }
  case Builtin::BI__builtin_is_constant_evaluated: {
    auto resultType = getMLIRType(expr->getType());
    llvm::errs()
        << "warning: assuming __builtin_is_constant_evaluated to be false\n";
    return success(
        ValueCategory(builder.create<arith::ConstantIntOp>(loc, 0, resultType),
                      /*isRef*/ false));
  }
  case Builtin::BIsqrt:
  case Builtin::BIsqrtf:
  case Builtin::BIsqrtl:
  case Builtin::BI__builtin_sqrt:
  case Builtin::BI__builtin_sqrtf:
  case Builtin::BI__builtin_sqrtf16:
  case Builtin::BI__builtin_sqrtl:
  case Builtin::BI__builtin_sqrtf128:
  case Builtin::BI__builtin_elementwise_sqrt: {
    auto v = Visit(expr->getArg(0));
    assert(!v.isReference);
    Value res = builder.create<math::SqrtOp>(loc, v.val);
    auto postTy = getMLIRType(expr->getType());
    return success(ValueCategory(res, /*isRef*/ false));
  }
  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll: {
    auto v = Visit(expr->getArg(0));
    assert(!v.isReference);
    Value res = builder.create<math::CountLeadingZerosOp>(loc, v.val);
    auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    return success(
        ValueCategory(castInteger(builder, loc, res, postTy), /*isRef*/ false));
  }
  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll: {
    auto v = Visit(expr->getArg(0));
    assert(!v.isReference);
    Value res = builder.create<math::CountTrailingZerosOp>(loc, v.val);
    auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    return success(
        ValueCategory(castInteger(builder, loc, res, postTy), /*isRef*/ false));
  }
  default:
    break;
  }
  return make_pair(ValueCategory(), false);
}

/// Whether `record` is std::ios_base or derives from it.
static bool derivesFromIosBase(const clang::CXXRecordDecl *record) {
  if (!record || !record->hasDefinition())
    return false;
  record = record->getDefinition();
  if (record->isInStdNamespace() && record->getIdentifier() &&
      record->getName() == "ios_base")
    return true;
  for (const clang::CXXBaseSpecifier &base : record->bases())
    if (derivesFromIosBase(base.getType()->getAsCXXRecordDecl()))
      return true;
  return false;
}

/// Whether `type` (references removed) is a C++ standard stream: a class
/// that is or derives from std::ios_base (std::ostream, std::istream, their
/// file and string variants).
static bool isStdStreamType(clang::QualType type) {
  return derivesFromIosBase(
      type.getNonReferenceType().getCanonicalType()->getAsCXXRecordDecl());
}

/// Whether `call` is stream I/O whose value nothing needs: it returns
/// nothing or the stream itself, and its object or first argument is a
/// standard stream (`os << x`, `os.flush()`, `std::endl(os)`).
static bool isStreamIOCall(clang::CallExpr *call, clang::ASTContext &ctx) {
  clang::QualType result = call->getCallReturnType(ctx);
  if (!result->isVoidType() && !isStdStreamType(result))
    return false;
  if (auto *member = dyn_cast<clang::CXXMemberCallExpr>(call))
    if (clang::Expr *object = member->getImplicitObjectArgument())
      return isStdStreamType(object->getType());
  return call->getNumArgs() > 0 && isStdStreamType(call->getArg(0)->getType());
}

ValueCategory MLIRScanner::VisitCallExpr(clang::CallExpr *expr) {

  auto loc = getMLIRLocation(expr->getExprLoc());
  /*
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__shfl_up_sync") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        builder.create<gpu::ShuffleOp>(loc, );
        llvm_unreachable("__shfl_up_sync unhandled");
        return nullptr;
      }
    }
  */

  // Console I/O through the C++ streams (`std::cout << x << std::endl;`,
  // `std::cerr.flush()`) is no hardware: Vitis' reflow removes these calls
  // (the Vitis 2025.1 run of class_ece8893_2023_spring YOLOv4-Tiny
  // Maxpool_Layer: 29 ostream references in a.g.ld.5.gdce.ll, none in
  // a.o.1.ll). cgeist used to emit them with the libstdc++ internals they
  // call, whose `llvm.load` of a memref does not verify outside Polygeist.
  // Only the arguments that have side effects are evaluated; a nested stream
  // call among them is dropped the same way.
  if (isStreamIOCall(expr, Glob.CGM.getContext())) {
    auto evaluate = [&](clang::Expr *e) {
      if (e && e->HasSideEffects(Glob.CGM.getContext()))
        Visit(e);
    };
    if (auto *member = dyn_cast<clang::CXXMemberCallExpr>(expr))
      evaluate(member->getImplicitObjectArgument());
    for (clang::Expr *arg : expr->arguments())
      evaluate(arg);
    return ValueCategory();
  }

  // A builtin that clang evaluates to a constant of arithmetic type
  // (`__builtin_inff()`, which <math.h>'s INFINITY expands to,
  // `__builtin_huge_val()`, `__builtin_nanf("")`): emit the constant. These
  // builtins have no library definition, so a call to them could not be
  // lowered or linked (W8b repro r_inff.c: `float m = -INFINITY;`).
  if (expr->getBuiltinCallee() && expr->getType()->isArithmeticType() &&
      !expr->HasSideEffects(Glob.CGM.getContext())) {
    clang::Expr::EvalResult res;
    if (expr->EvaluateAsRValue(res, Glob.CGM.getContext()) &&
        !res.HasSideEffects) {
      mlir::Type ty = getMLIRType(expr->getType());
      if (res.Val.isFloat())
        if (auto ft = dyn_cast<mlir::FloatType>(ty);
            ft && &ft.getFloatSemantics() == &res.Val.getFloat().getSemantics())
          return ValueCategory(
              builder.create<arith::ConstantFloatOp>(loc, res.Val.getFloat(),
                                                     ft),
              /*isReference*/ false);
      if (res.Val.isInt() && res.Val.getInt().getBitWidth() <= 64)
        if (auto it = dyn_cast<mlir::IntegerType>(ty))
          return ValueCategory(
              builder.create<arith::ConstantIntOp>(
                  loc, res.Val.getInt().getSExtValue(), it.getWidth()),
              /*isReference*/ false);
    }
  }

  // Only callees named like the Xilinx HLS builtins (declared by the
  // real-header prelude) are lowered here, and only under --xlx-prelude or
  // --xlx-headers: without them such a call is emitted as before.
  if (XlxPrelude || cgeist::hls::xlxHeaders()) {
    auto xlx = EmitXlxBuiltin(expr);
    if (xlx.second)
      return xlx.first;
  }

  auto valEmitted = EmitGPUCallExpr(expr);
  if (valEmitted.second)
    return valEmitted.first;

  valEmitted = EmitBuiltinOps(expr);
  if (valEmitted.second)
    return valEmitted.first;

  valEmitted = EmitClangBuiltinCallExpr(expr);
  if (valEmitted.second)
    return valEmitted.first;

  if (auto *oc = dyn_cast<CXXOperatorCallExpr>(expr)) {
    if (oc->getOperator() == clang::OO_EqualEqual) {
      if (auto *lhs = dyn_cast<CXXTypeidExpr>(expr->getArg(0))) {
        if (auto *rhs = dyn_cast<CXXTypeidExpr>(expr->getArg(1))) {
          QualType LT = lhs->isTypeOperand()
                            ? lhs->getTypeOperand(Glob.CGM.getContext())
                            : lhs->getExprOperand()->getType();
          QualType RT = rhs->isTypeOperand()
                            ? rhs->getTypeOperand(Glob.CGM.getContext())
                            : rhs->getExprOperand()->getType();
          llvm::Constant *LC = Glob.CGM.GetAddrOfRTTIDescriptor(LT);
          llvm::Constant *RC = Glob.CGM.GetAddrOfRTTIDescriptor(RT);
          auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
          return ValueCategory(
              builder.create<arith::ConstantIntOp>(loc, LC == RC, postTy),
              false);
        }
      }
    }
  }
  // `x = y` on scalar wrappers with the trivial (defaulted) assignment:
  // a load and a store, no call.
  if (auto *oc = dyn_cast<CXXOperatorCallExpr>(expr)) {
    if (oc->getOperator() == clang::OO_Equal && oc->getNumArgs() == 2)
      if (auto *MD = dyn_cast_or_null<CXXMethodDecl>(oc->getCalleeDecl()))
        if (MD->isTrivial() &&
            (MD->isCopyAssignmentOperator() ||
             MD->isMoveAssignmentOperator()) &&
            Glob.getScalarWrapperField(MD->getParent())) {
          ValueCategory lhs = Visit(oc->getArg(0));
          ValueCategory rhs = Visit(oc->getArg(1));
          assert(lhs.val && rhs.val && lhs.isReference);
          lhs.store(loc, builder, rhs.getValue(loc, builder));
          return lhs;
        }
  }
  if (auto *oc = dyn_cast<CXXMemberCallExpr>(expr)) {
    if (auto *lhs = dyn_cast<CXXTypeidExpr>(oc->getImplicitObjectArgument())) {
      expr->getCallee()->dump();
      if (auto *ic = dyn_cast<MemberExpr>(expr->getCallee()))
        if (auto *sr = dyn_cast<NamedDecl>(ic->getMemberDecl())) {
          if (sr->getIdentifier() && sr->getName() == "name") {
            QualType LT = lhs->isTypeOperand()
                              ? lhs->getTypeOperand(Glob.CGM.getContext())
                              : lhs->getExprOperand()->getType();
            llvm::Constant *LC = Glob.CGM.GetAddrOfRTTIDescriptor(LT);
            while (auto *CE = dyn_cast<llvm::ConstantExpr>(LC))
              LC = CE->getOperand(0);
            std::string val = cast<llvm::GlobalVariable>(LC)->getName().str();
            return CommonArrayToPointer(
                loc, ValueCategory(
                         Glob.GetOrCreateGlobalLLVMString(loc, builder, val),
                         /*isReference*/ true));
          }
        }
    }
  }

  if (auto *ps = dyn_cast<CXXPseudoDestructorExpr>(expr->getCallee())) {
    return Visit(ps);
  }

  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "atomicAdd" ||
           sr->getDecl()->getName() == "atomicOr" ||
           sr->getDecl()->getName() == "atomicAnd")) {
        std::vector<ValueCategory> args;
        for (auto *a : expr->arguments()) {
          args.push_back(Visit(a));
        }
        auto a0 = args[0].getValue(loc, builder);
        auto a1 = args[1].getValue(loc, builder);
        AtomicRMWKind op;
        LLVM::AtomicBinOp lop;
        if (sr->getDecl()->getName() == "atomicAdd") {
          if (a1.getType().isa<mlir::IntegerType>()) {
            op = AtomicRMWKind::addi;
            lop = LLVM::AtomicBinOp::add;
          } else {
            op = AtomicRMWKind::addf;
            lop = LLVM::AtomicBinOp::fadd;
          }
        } else if (sr->getDecl()->getName() == "atomicOr") {
          op = AtomicRMWKind::ori;
          lop = LLVM::AtomicBinOp::_or;
        } else if (sr->getDecl()->getName() == "atomicAnd") {
          op = AtomicRMWKind::andi;
          lop = LLVM::AtomicBinOp::_and;
        } else
          assert(0);

        if (a0.getType().isa<MemRefType>())
          return ValueCategory(
              builder.create<memref::AtomicRMWOp>(
                  loc, a1.getType(), op, a1, a0,
                  std::vector<mlir::Value>({getConstantIndex(0)})),
              /*isReference*/ false);
        else
          return ValueCategory(
              builder.create<LLVM::AtomicRMWOp>(loc, lop, a0, a1,
                                                LLVM::AtomicOrdering::acq_rel),
              /*isReference*/ false);
      }
    }

  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      // (--scalar-wrapper-abi: only for a floating first argument, so that
      // hls::pow on an ap_fixed class stays a call; unchanged when off.)
      if (sr->getDecl()->getIdentifier() &&
          (!ScalarWrapperABI ||
           (expr->getNumArgs() == 2 &&
            expr->getArg(0)->getType()->isRealFloatingType())) &&
          (sr->getDecl()->getName() == "__powf" ||
           sr->getDecl()->getName() == "pow" ||
           sr->getDecl()->getName() == "__nv_pow" ||
           sr->getDecl()->getName() == "__nv_powf" ||
           sr->getDecl()->getName() == "__powi" ||
           sr->getDecl()->getName() == "powi" ||
           sr->getDecl()->getName() == "__nv_powi" ||
           sr->getDecl()->getName() == "__nv_powi" ||
           sr->getDecl()->getName() == "powf")) {
        auto mlirType = getMLIRType(expr->getType());
        std::vector<mlir::Value> args;
        for (auto *a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        // A floating argument of another type than the result is converted
        // to it first: libstdc++'s `std::pow(float, int)` is the promoted
        // template whose parameters are (float, int) and whose result is
        // double (C++11 [c.math]), computed as pow((double)x, (double)n).
        // Without the conversion the op computed in float and claimed a
        // double result (llvm.intr.powi(f32, i32) -> f64; CIRCT W9r14
        // review, kernel w9r14rv_pw_f32, OPEN_ISSUES O145).
        auto toResultType = [&](mlir::Value v) -> mlir::Value {
          auto from = v.getType().dyn_cast<mlir::FloatType>();
          auto to = mlirType.dyn_cast<mlir::FloatType>();
          if (!from || !to || from == to)
            return v;
          if (from.getWidth() < to.getWidth())
            return builder.create<arith::ExtFOp>(loc, to, v);
          return builder.create<arith::TruncFOp>(loc, to, v);
        };
        args[0] = toResultType(args[0]);
        if (!args[1].getType().isa<mlir::IntegerType>())
          args[1] = toResultType(args[1]);
        if (args[1].getType().isa<mlir::IntegerType>())
          return ValueCategory(
              builder.create<LLVM::PowIOp>(loc, mlirType, args[0], args[1]),
              /*isReference*/ false);
        else
          return ValueCategory(
              builder.create<math::PowFOp>(loc, mlirType, args[0], args[1]),
              /*isReference*/ false);
      }
    }
  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_expect") {
        llvm::errs() << "warning: ignoring __builtin_expect\n";
        return Visit(expr->getArg(0));
      }
    }
  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_fabsf" ||
           sr->getDecl()->getName() == "__nv_fabs" ||
           sr->getDecl()->getName() == "__nv_abs" ||
           sr->getDecl()->getName() == "fabs" ||
           sr->getDecl()->getName() == "fabsf" ||
           sr->getDecl()->getName() == "__builtin_fabs" ||
           sr->getDecl()->getName() == "__builtin_fabsf") &&
          // Only the C library function on a number: an overload of the same
          // name for a class (`fabs(ap_fixed<W, I>)`, the Vitis hls_math
          // overloads) is an ordinary call (W8b, flowgnn dgn and
          // accelerators GP_selection: cast<IntegerType> of a class value).
          expr->getNumArgs() == 1 &&
          expr->getArg(0)->getType()->isArithmeticType()) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value Fabs;
        if (V.getType().isa<mlir::FloatType>())
          Fabs = builder.create<math::AbsFOp>(loc, V);
        else {
          auto zero = builder.create<arith::ConstantIntOp>(
              loc, 0, V.getType().cast<mlir::IntegerType>().getWidth());
          Fabs = builder.create<SelectOp>(
              loc,
              builder.create<arith::CmpIOp>(loc, CmpIPredicate::sge, V, zero),
              V, builder.create<arith::SubIOp>(loc, zero, V));
        }
        return ValueCategory(Fabs, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__nv_mul24") {
        mlir::Value V0 = getLLVM(expr->getArg(0));
        mlir::Value V1 = getLLVM(expr->getArg(1));
        auto c8 = builder.create<arith::ConstantIntOp>(loc, 8, 32);
        V0 = builder.create<arith::ShLIOp>(loc, V0, c8);
        V0 = builder.create<arith::ShRUIOp>(loc, V0, c8);
        V1 = builder.create<arith::ShLIOp>(loc, V1, c8);
        V1 = builder.create<arith::ShRUIOp>(loc, V1, c8);
        return ValueCategory(builder.create<MulIOp>(loc, V0, V1), false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__nv_umulhi") {
        mlir::Value V0 = getLLVM(expr->getArg(0));
        mlir::Value V1 = getLLVM(expr->getArg(1));
        auto I64 = builder.getIntegerType(64);
        auto I32 = builder.getIntegerType(32);
        V0 = builder.create<ExtUIOp>(loc, I64, V0);
        V1 = builder.create<ExtUIOp>(loc, I64, V1);
        mlir::Value R = builder.create<arith::MulIOp>(loc, V0, V1);
        auto c32 = builder.create<arith::ConstantIntOp>(loc, 32, 64);
        R = builder.create<arith::ShRUIOp>(loc, R, c32);
        R = builder.create<TruncIOp>(loc, I32, R);
        return ValueCategory(R, false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_frexp" ||
           sr->getDecl()->getName() == "__builtin_frexpf" ||
           sr->getDecl()->getName() == "__builtin_frexpl" ||
           sr->getDecl()->getName() == "__builtin_frexpf128")) {
        mlir::Value V0 = getLLVM(expr->getArg(0));
        mlir::Value V1 = getLLVM(expr->getArg(1));

        auto name = sr->getDecl()
                        ->getName()
                        .substr(std::string("__builtin_").length())
                        .str();

        if (Glob.functions.find(name) == Glob.functions.end()) {
          std::vector<mlir::Type> types{V0.getType(), V1.getType()};

          auto RT = getMLIRType(expr->getType());
          std::vector<mlir::Type> rettypes{RT};
          mlir::OpBuilder mbuilder(Glob.module->getContext());
          auto funcType = mbuilder.getFunctionType(types, rettypes);
          Glob.functions[name] = mlir::func::FuncOp(mlir::func::FuncOp::create(
              builder.getUnknownLoc(), name, funcType));
          SymbolTable::setSymbolVisibility(Glob.functions[name],
                                           SymbolTable::Visibility::Private);
          Glob.module->push_back(Glob.functions[name]);
        }

        mlir::Value vals[] = {V0, V1};
        return ValueCategory(
            builder.create<CallOp>(loc, Glob.functions[name], vals)
                .getResult(0),
            false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_strlen" ||
           sr->getDecl()->getName() == "strlen")) {
        mlir::Value V0 = getLLVM(expr->getArg(0));

        const auto *name = "strlen";

        if (Glob.functions.find(name) == Glob.functions.end()) {
          std::vector<mlir::Type> types{V0.getType()};

          auto RT = getMLIRType(expr->getType());
          std::vector<mlir::Type> rettypes{RT};
          mlir::OpBuilder mbuilder(Glob.module->getContext());
          auto funcType = mbuilder.getFunctionType(types, rettypes);
          Glob.functions[name] = mlir::func::FuncOp(mlir::func::FuncOp::create(
              builder.getUnknownLoc(), name, funcType));
          SymbolTable::setSymbolVisibility(Glob.functions[name],
                                           SymbolTable::Visibility::Private);
          Glob.module->push_back(Glob.functions[name]);
        }

        mlir::Value vals[] = {V0};
        return ValueCategory(
            builder.create<CallOp>(loc, Glob.functions[name], vals)
                .getResult(0),
            false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_isfinite" ||
           sr->getDecl()->getName() == "__builtin_isinf" ||
           sr->getDecl()->getName() == "__nv_isinff")) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        mlir::Value V = getLLVM(expr->getArg(0));
        auto Ty = V.getType().cast<mlir::FloatType>();
        mlir::Value Fabs = builder.create<math::AbsFOp>(loc, V);
        auto Infinity = builder.create<ConstantFloatOp>(
            loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
        auto Pred = (sr->getDecl()->getName() == "__builtin_isinf" ||
                     sr->getDecl()->getName() == "__nv_isinff")
                        ? CmpFPredicate::OEQ
                        : CmpFPredicate::ONE;
        mlir::Value FCmp = builder.create<CmpFOp>(loc, Pred, Fabs, Infinity);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, FCmp);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_isnan" ||
           sr->getDecl()->getName() == "__nv_isnanf")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value Eq = builder.create<CmpFOp>(loc, CmpFPredicate::UNO, V, V);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, Eq);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_isnormal")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        auto Ty = V.getType().cast<mlir::FloatType>();
        mlir::Value Eq = builder.create<CmpFOp>(loc, CmpFPredicate::OEQ, V, V);

        mlir::Value Abs = builder.create<math::AbsFOp>(loc, V);
        auto Infinity = builder.create<ConstantFloatOp>(
            loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
        mlir::Value IsLessThanInf =
            builder.create<CmpFOp>(loc, CmpFPredicate::ULT, Abs, Infinity);
        APFloat Smallest =
            APFloat::getSmallestNormalized(Ty.getFloatSemantics());
        auto SmallestV = builder.create<ConstantFloatOp>(loc, Smallest, Ty);
        mlir::Value IsNormal =
            builder.create<CmpFOp>(loc, CmpFPredicate::UGE, Abs, SmallestV);
        V = builder.create<AndIOp>(loc, Eq, IsLessThanInf);
        V = builder.create<AndIOp>(loc, V, IsNormal);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_signbit") {
        mlir::Value V = getLLVM(expr->getArg(0));
        auto Ty = V.getType().cast<mlir::FloatType>();
        auto ITy = builder.getIntegerType(Ty.getWidth());
        mlir::Value BC = builder.create<BitcastOp>(loc, ITy, V);
        auto ZeroV = builder.create<ConstantIntOp>(loc, 0, ITy);
        V = builder.create<CmpIOp>(loc, CmpIPredicate::slt, BC, ZeroV);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isgreater") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isgreaterequal") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OGE, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isless") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_islessequal") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OLE, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_islessgreater") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isunordered") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::UNO, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_pow" ||
           sr->getDecl()->getName() == "__builtin_powf" ||
           sr->getDecl()->getName() == "__builtin_powl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<math::PowFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_fmodf")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<mlir::LLVM::FRemOp>(loc, V.getType(), V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_atanh" ||
           sr->getDecl()->getName() == "__builtin_atanhf" ||
           sr->getDecl()->getName() == "__builtin_atanhl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::AtanOp>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_scalbn" ||
           sr->getDecl()->getName() == "__nv_scalbnf" ||
           sr->getDecl()->getName() == "__nv_scalbnl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        auto name = sr->getDecl()->getName().substr(5).str();
        std::vector<mlir::Type> types{V.getType(), V2.getType()};
        auto RT = getMLIRType(expr->getType());

        std::vector<mlir::Type> rettypes{RT};

        mlir::OpBuilder mbuilder(Glob.module->getContext());
        auto funcType = mbuilder.getFunctionType(types, rettypes);
        mlir::func::FuncOp function =
            mlir::func::FuncOp(mlir::func::FuncOp::create(
                builder.getUnknownLoc(), name, funcType));
        SymbolTable::setSymbolVisibility(function,
                                         SymbolTable::Visibility::Private);

        Glob.functions[name] = function;
        Glob.module->push_back(function);
        mlir::Value vals[] = {V, V2};
        V = builder.create<CallOp>(loc, function, vals).getResult(0);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_dmul_rn")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<MulFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_dadd_rn")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<AddFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_dsub_rn")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<SubFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_log2" ||
           sr->getDecl()->getName() == "__builtin_log2f" ||
           sr->getDecl()->getName() == "__builtin_log2l" ||
           sr->getDecl()->getName() == "__nv_log2" ||
           sr->getDecl()->getName() == "__nv_log2f" ||
           sr->getDecl()->getName() == "__nv_log2l")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::Log2Op>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_log10" ||
           sr->getDecl()->getName() == "__builtin_log10f" ||
           sr->getDecl()->getName() == "__builtin_log10l" ||
           sr->getDecl()->getName() == "__nv_log10" ||
           sr->getDecl()->getName() == "__nv_log10f" ||
           sr->getDecl()->getName() == "__nv_log10l")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::Log10Op>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_log1p" ||
           sr->getDecl()->getName() == "__builtin_log1pf" ||
           sr->getDecl()->getName() == "__builtin_log1pl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::Log1pOp>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_exp2" ||
           sr->getDecl()->getName() == "__builtin_exp2f" ||
           sr->getDecl()->getName() == "__builtin_exp2l")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::Exp2Op>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_expm1" ||
           sr->getDecl()->getName() == "__builtin_expm1f" ||
           sr->getDecl()->getName() == "__builtin_expm1l")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::ExpM1Op>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_copysign" ||
           sr->getDecl()->getName() == "__builtin_copysignf" ||
           sr->getDecl()->getName() == "__builtin_copysignl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<LLVM::CopySignOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_copysign" ||
           sr->getDecl()->getName() == "__builtin_copysignf" ||
           sr->getDecl()->getName() == "__builtin_copysignl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<LLVM::CopySignOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_fmax" ||
           sr->getDecl()->getName() == "__builtin_fmaxf" ||
           sr->getDecl()->getName() == "__builtin_fmaxl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<LLVM::MaxNumOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_fmin" ||
           sr->getDecl()->getName() == "__builtin_fminf" ||
           sr->getDecl()->getName() == "__builtin_fminl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<LLVM::MinNumOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_fma" ||
           sr->getDecl()->getName() == "__builtin_fmaf" ||
           sr->getDecl()->getName() == "__builtin_fmal")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        mlir::Value V3 = getLLVM(expr->getArg(2));
        V = builder.create<LLVM::FMAOp>(loc, V, V2, V3);
        return ValueCategory(V, /*isRef*/ false);
      }
    }

  if (!CStyleMemRef) {
    if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
      if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
        if ((sr->getDecl()->getIdentifier() &&
             (sr->getDecl()->getName() == "fscanf" ||
              sr->getDecl()->getName() == "scanf" ||
              sr->getDecl()->getName() == "__isoc99_sscanf" ||
              sr->getDecl()->getName() == "sscanf")) ||
            (isa<CXXOperatorCallExpr>(expr) &&
             cast<CXXOperatorCallExpr>(expr)->getOperator() ==
                 OO_GreaterGreater)) {
          const auto *tocall = EmitCallee(expr->getCallee());
          auto strcmpF = Glob.GetOrCreateLLVMFunction(tocall);

          std::vector<mlir::Value> args;
          std::vector<std::pair<mlir::Value, mlir::Value>> ops;
          std::map<const void *, size_t> counts;
          for (auto *a : expr->arguments()) {
            auto v = getLLVM(a);
            if (auto toptr = v.getDefiningOp<polygeist::Memref2PointerOp>()) {
              auto T = toptr.getType().cast<LLVM::LLVMPointerType>();
              auto idx = counts[T.getAsOpaquePointer()]++;
              auto aop = allocateBuffer(idx, T);
              args.push_back(aop.getResult());
              ops.emplace_back(aop.getResult(), toptr.getSource());
            } else
              args.push_back(v);
          }
          auto called = builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args);
          for (auto pair : ops) {
            auto lop = builder.create<mlir::LLVM::LoadOp>(loc, pair.first);
            builder.create<mlir::memref::StoreOp>(
                loc, lop, pair.second,
                std::vector<mlir::Value>({getConstantIndex(0)}));
          }
          return ValueCategory(called.getResult(), /*isReference*/ false);
        }
      }
  }

  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "memmove" ||
           sr->getDecl()->getName() == "__builtin_memmove")) {
        std::vector<mlir::Value> args = {getLLVM(expr->getArg(0)),
                                         getLLVM(expr->getArg(1)),
                                         getLLVM(expr->getArg(2))};
        builder.create<LLVM::MemmoveOp>(loc, args[0], args[1], args[2],
                                        /*isVolatile*/ false);
        return ValueCategory(args[0], /*isReference*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "memset" ||
           sr->getDecl()->getName() == "__builtin_memset")) {
        std::vector<mlir::Value> args = {getLLVM(expr->getArg(0)),
                                         getLLVM(expr->getArg(1)),
                                         getLLVM(expr->getArg(2))};

        args[1] = builder.create<TruncIOp>(loc, builder.getI8Type(), args[1]);
        builder.create<LLVM::MemsetOp>(loc, args[0], args[1], args[2],
                                       /*isVolatile*/ false);
        return ValueCategory(args[0], /*isReference*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "memcpy" ||
           sr->getDecl()->getName() == "__builtin_memcpy")) {
        std::vector<mlir::Value> args = {getLLVM(expr->getArg(0)),
                                         getLLVM(expr->getArg(1)),
                                         getLLVM(expr->getArg(2))};
        builder.create<LLVM::MemcpyOp>(loc, args[0], args[1], args[2], false);
        return ValueCategory(args[0], /*isReference*/ false);
      }
      // TODO this only sets a preference so it is not needed but if possible
      // implement it
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "cudaFuncSetCacheConfig")) {
        return ValueCategory();
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "cudaMemcpy" ||
           sr->getDecl()->getName() == "cudaMemcpyAsync" ||
           sr->getDecl()->getName() == "cudaMemcpyToSymbol" ||
           sr->getDecl()->getName() == "memcpy" ||
           sr->getDecl()->getName() == "__builtin_memcpy")) {
        auto *dstSub = expr->getArg(0);
        while (auto *BC = dyn_cast<clang::CastExpr>(dstSub))
          dstSub = BC->getSubExpr();
        auto *srcSub = expr->getArg(1);
        while (auto *BC = dyn_cast<clang::CastExpr>(srcSub))
          srcSub = BC->getSubExpr();

#if 0
        auto dstst = dstSub->getType()->getUnqualifiedDesugaredType();
        if (isa<clang::PointerType>(dstst) || isa<clang::ArrayType>(dstst)) {

          auto elem = isa<clang::PointerType>(dstst)
                          ? cast<clang::PointerType>(dstst)
                                ->getPointeeType()
                                ->getUnqualifiedDesugaredType()

                          : cast<clang::ArrayType>(dstst)
                                ->getElementType()
                                ->getUnqualifiedDesugaredType();
          auto melem = elem;
          if (auto BC = dyn_cast<clang::ArrayType>(melem))
            melem = BC->getElementType()->getUnqualifiedDesugaredType();

          auto srcst = srcSub->getType()->getUnqualifiedDesugaredType();
          auto selem = isa<clang::PointerType>(srcst)
                           ? cast<clang::PointerType>(srcst)
                                 ->getPointeeType()
                                 ->getUnqualifiedDesugaredType()

                           : cast<clang::ArrayType>(srcst)
                                 ->getElementType()
                                 ->getUnqualifiedDesugaredType();

          auto mselem = selem;
          if (auto BC = dyn_cast<clang::ArrayType>(mselem))
            mselem = BC->getElementType()->getUnqualifiedDesugaredType();

          if (melem == mselem) {
            mlir::Value dst;
            ValueCategory vdst = Visit(dstSub);
            if (isa<clang::PointerType>(dstst)) {
              dst = vdst.getValue(loc, builder);
            } else {
              assert(vdst.isReference);
              dst = vdst.val;
            }
            // if (dst.getType().isa<MemRefType>())
            {
              mlir::Value src;
              ValueCategory vsrc = Visit(srcSub);
              if (isa<clang::PointerType>(srcst)) {
                src = vsrc.getValue(loc, builder);
              } else {
                assert(vsrc.isReference);
                src = vsrc.val;
              }

              bool dstArray = false;
              Glob.getMLIRType(QualType(elem, 0), &dstArray);
              bool srcArray = false;
              Glob.getMLIRType(QualType(selem, 0), &srcArray);
              auto elemSize = getTypeSize(QualType(elem, 0));
              if (srcArray && !dstArray)
                elemSize = getTypeSize(QualType(selem, 0));
              mlir::Value size = builder.create<IndexCastOp>(
                  loc, Visit(expr->getArg(2)).getValue(loc, builder),
                  mlir::IndexType::get(builder.getContext()));
              size = builder.create<DivUIOp>(
                  loc, size, builder.create<ConstantIndexOp>(loc, elemSize));

              if (sr->getDecl()->getName() == "cudaMemcpyToSymbol") {
                mlir::Value offset = Visit(expr->getArg(3)).getValue(loc, builder);
                offset = builder.create<IndexCastOp>(
                    loc, offset, mlir::IndexType::get(builder.getContext()));
                offset = builder.create<DivUIOp>(
                    loc, offset,
                    builder.create<ConstantIndexOp>(loc, elemSize));
                // assert(!dstArray);
                if (auto mt = dyn_cast<MemRefType>(dst.getType())) {
                  auto shape = std::vector<int64_t>(mt.getShape());
                  shape[0] = ShapedType::kDynamic;
                  auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                                   MemRefLayoutAttrInterface(),
                                                   mt.getMemorySpace());
                  dst = builder.create<polygeist::SubIndexOp>(loc, mt0, dst,
                                                              offset);
                } else {
                  mlir::Value idxs[] = {offset};
                  dst = builder.create<LLVM::GEPOp>(loc, dst.getType(), dst,
                                                    idxs);
                }
              }

              auto affineOp = builder.create<scf::ForOp>(
                  loc, getConstantIndex(0), size, getConstantIndex(1));

              auto oldpoint = builder.getInsertionPoint();
              auto oldblock = builder.getInsertionBlock();

              std::vector<mlir::Value> dstargs = {affineOp.getInductionVar()};
              std::vector<mlir::Value> srcargs = {affineOp.getInductionVar()};

              builder.setInsertionPointToStart(&affineOp.getLoopBody().front());

              if (dstArray) {
                std::vector<mlir::Value> start = {getConstantIndex(0)};
                auto mt = Glob.getMLIRType(Glob.CGM.getContext().getPointerType(
                                               QualType(elem, 0)))
                              .cast<MemRefType>();
                auto shape = std::vector<int64_t>(mt.getShape());
                assert(shape.size() > 0 && shape.back() != ShapedType::kDynamic);
                auto affineOp = builder.create<scf::ForOp>(
                    loc, getConstantIndex(0), getConstantIndex(shape.back()),
                    getConstantIndex(1));
                dstargs.push_back(affineOp.getInductionVar());
                builder.setInsertionPointToStart(
                    &affineOp.getLoopBody().front());
                if (srcArray) {
                  auto smt =
                      Glob.getMLIRType(Glob.CGM.getContext().getPointerType(
                                           QualType(elem, 0)))
                          .cast<MemRefType>();
                  auto sshape = std::vector<int64_t>(smt.getShape());
                  assert(sshape.size() > 0 && sshape.back() != ShapedType::kDynamic);
                  assert(sshape.back() == shape.back());
                  srcargs.push_back(affineOp.getInductionVar());
                } else {
                  srcargs[0] = builder.create<AddIOp>(
                      loc,
                      builder.create<MulIOp>(loc, srcargs[0],
                                             getConstantIndex(shape.back())),
                      affineOp.getInductionVar());
                }
              } else {
                if (srcArray) {
                  auto smt =
                      Glob.getMLIRType(Glob.CGM.getContext().getPointerType(
                                           QualType(selem, 0)))
                          .cast<MemRefType>();
                  auto sshape = std::vector<int64_t>(smt.getShape());
                  assert(sshape.size() > 0 && sshape.back() != ShapedType::kDynamic);
                  auto affineOp = builder.create<scf::ForOp>(
                      loc, getConstantIndex(0), getConstantIndex(sshape.back()),
                      getConstantIndex(1));
                  srcargs.push_back(affineOp.getInductionVar());
                  builder.setInsertionPointToStart(
                      &affineOp.getLoopBody().front());
                  dstargs[0] = builder.create<AddIOp>(
                      loc,
                      builder.create<MulIOp>(loc, dstargs[0],
                                             getConstantIndex(sshape.back())),
                      affineOp.getInductionVar());
                }
              }

              mlir::Value loaded;
              if (src.getType().isa<MemRefType>())
                loaded = builder.create<memref::LoadOp>(loc, src, srcargs);
              else {
                auto opt = src.getType().cast<LLVM::LLVMPointerType>();
                auto elty = LLVM::LLVMPointerType::get(opt.getElementType(),
                                                       opt.getAddressSpace());
                for (auto &val : srcargs) {
                  val = builder.create<IndexCastOp>(val.getLoc(), val,
                                                    builder.getI32Type());
                }
                loaded = builder.create<LLVM::LoadOp>(
                    loc, builder.create<LLVM::GEPOp>(loc, elty, src, srcargs));
              }
              if (dst.getType().isa<MemRefType>()) {
                builder.create<memref::StoreOp>(loc, loaded, dst, dstargs);
              } else {
                auto opt = dst.getType().cast<LLVM::LLVMPointerType>();
                auto elty = LLVM::LLVMPointerType::get(opt.getElementType(),
                                                       opt.getAddressSpace());
                for (auto &val : dstargs) {
                  val = builder.create<IndexCastOp>(val.getLoc(), val,
                                                    builder.getI32Type());
                }
                builder.create<LLVM::StoreOp>(
                    loc, loaded,
                    builder.create<LLVM::GEPOp>(loc, elty, dst, dstargs));
              }

              // TODO: set the value of the iteration value to the final bound
              // at the end of the loop.
              builder.setInsertionPoint(oldblock, oldpoint);

              auto retTy = getMLIRType(expr->getType());
              if (sr->getDecl()->getName() == "__builtin_memcpy" ||
                  retTy.isa<LLVM::LLVMPointerType>()) {
                if (dst.getType().isa<MemRefType>())
                  dst = builder.create<polygeist::Memref2PointerOp>(loc, retTy,
                                                                    dst);
                else
                  dst = builder.create<LLVM::BitcastOp>(loc, retTy, dst);
                if (dst.getType() != retTy) {
                    expr->dump();
                    llvm::errs() << " retTy: " << retTy << " dst: " << dst << "\n";
                }
                assert(dst.getType() == retTy);
                return ValueCategory(dst, /*isReference*/ false);
              } else {
                if (!retTy.isa<mlir::IntegerType>()) {
                  expr->dump();
                  llvm::errs() << " retTy: " << retTy << "\n";
                }
                return ValueCategory(
                    builder.create<ConstantIntOp>(loc, 0, retTy),
                    /*isReference*/ false);
              }
            }
          }
          /*
          function.dump();
          expr->dump();
          dstSub->dump();
          elem->dump();
          srcSub->dump();
          mselem->dump();
          llvm_unreachable("unhandled cudaMemcpy");
          */
        }
#endif
      }
    }

#if 0
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "cudaMemset")) {
        if (auto IL = dyn_cast<clang::IntegerLiteral>(expr->getArg(1)))
          if (IL->getValue() == 0) {
            auto dstSub = expr->getArg(0);
            while (auto BC = dyn_cast<clang::CastExpr>(dstSub))
              dstSub = BC->getSubExpr();

            auto dstst = dstSub->getType()->getUnqualifiedDesugaredType();
            auto elem = isa<clang::PointerType>(dstst)
                            ? cast<clang::PointerType>(dstst)->getPointeeType()
                            : cast<clang::ArrayType>(dstst)->getElementType();
            mlir::Value dst;
            ValueCategory vdst = Visit(dstSub);
            if (isa<clang::PointerType>(dstst)) {
              dst = vdst.getValue(loc, builder);
            } else {
              assert(vdst.isReference);
              dst = vdst.val;
            }
            if (dst.getType().isa<MemRefType>()) {

              bool dstArray = false;
              auto melem = Glob.getMLIRType(elem, &dstArray);
              mlir::Value toStore;
              if (melem.isa<mlir::IntegerType>())
                toStore = builder.create<ConstantIntOp>(loc, 0, melem);
              else {
                auto ft = melem.cast<FloatType>();
                toStore = builder.create<ConstantFloatOp>(
                    loc, APFloat(ft.getFloatSemantics(), "0"), ft);
              }

              auto elemSize = getTypeSize(elem);
              mlir::Value size = builder.create<IndexCastOp>(
                  loc, Visit(expr->getArg(2)).getValue(loc, builder),
                  mlir::IndexType::get(builder.getContext()));
              size = builder.create<DivUIOp>(
                  loc, size, builder.create<ConstantIndexOp>(loc, elemSize));

              auto affineOp = builder.create<scf::ForOp>(
                  loc, getConstantIndex(0), size, getConstantIndex(1));

              auto oldpoint = builder.getInsertionPoint();
              auto oldblock = builder.getInsertionBlock();

              std::vector<mlir::Value> args = {affineOp.getInductionVar()};

              builder.setInsertionPointToStart(&affineOp.getLoopBody().front());

              if (dstArray) {
                std::vector<mlir::Value> start = {getConstantIndex(0)};
                auto mt =
                    Glob.getMLIRType(Glob.CGM.getContext().getPointerType(elem))
                        .cast<MemRefType>();
                auto shape = std::vector<int64_t>(mt.getShape());
                auto affineOp = builder.create<scf::ForOp>(
                    loc, getConstantIndex(0), getConstantIndex(shape[1]),
                    getConstantIndex(1));
                args.push_back(affineOp.getInductionVar());
                builder.setInsertionPointToStart(
                    &affineOp.getLoopBody().front());
              }

              builder.create<memref::StoreOp>(loc, toStore, dst, args);

              // TODO: set the value of the iteration value to the final bound
              // at the end of the loop.
              builder.setInsertionPoint(oldblock, oldpoint);

              auto retTy = getMLIRType(expr->getType());
              return ValueCategory(builder.create<ConstantIntOp>(loc, 0, retTy),
                                   /*isReference*/ false);
            }
          }
      }
    }
#endif

  if (auto BI = expr->getBuiltinCallee()) {
    if (Glob.CGM.getContext().BuiltinInfo.isLibFunction(BI)) {
      llvm::errs() << "warning: we fall back to libc call for "
                   << Glob.CGM.getContext().BuiltinInfo.getName(BI) << "\n";

      std::vector<mlir::Value> args;
      for (size_t i = 0; i < expr->getNumArgs(); i++) {
        args.push_back(getLLVM(expr->getArg(i)));
      }

      auto name = expr->getCalleeDecl()
                      ->getAsFunction()
                      ->getName()
                      .substr(std::string("__builtin_").length())
                      .str();

      if (Glob.functions.find(name) == Glob.functions.end()) {
        auto types = llvm::to_vector(
            llvm::map_range(args, [&](auto a) { return a.getType(); }));

        auto RT = getMLIRType(expr->getType());
        std::vector<mlir::Type> rettypes{RT};
        mlir::OpBuilder mbuilder(Glob.module->getContext());
        auto funcType = mbuilder.getFunctionType(types, rettypes);
        Glob.functions[name] = mlir::func::FuncOp(mlir::func::FuncOp::create(
            builder.getUnknownLoc(), name, funcType));
        SymbolTable::setSymbolVisibility(Glob.functions[name],
                                         SymbolTable::Visibility::Private);
        Glob.module->push_back(Glob.functions[name]);
      }
      return ValueCategory(
          builder.create<CallOp>(loc, Glob.functions[name], args).getResult(0),
          false);
    } else if (!Glob.CGM.getContext().BuiltinInfo.isPredefinedLibFunction(BI))
      llvm::errs() << "warning: we failed to emit call to builtin function "
                   << Glob.CGM.getContext().BuiltinInfo.getName(BI) << "\n";
  }

  const auto *callee = EmitCallee(expr->getCallee());

  std::set<std::string> funcs = {
      "fread",
      "read",
      "strcmp",
      "fputs",
      "puts",
      "memcpy",
      "getenv",
      "strrchr",
      "mkdir",
      "printf",
      "fprintf",
      "sprintf",
      "fwrite",
      "__builtin_memcpy",
      "cudaMemcpy",
      "cudaMemcpyAsync",
      "cudaMalloc",
      "cudaMallocHost",
      "cudaFree",
      "cudaFreeHost",
      "open",
      "gettimeofday",
      "fopen",
      "time",
      "memset",
      "cudaMemset",
      "strcpy",
      "close",
      "fclose",
      "atoi",
      "malloc",
      "calloc",
      "free",
      "fgets",
      "__errno_location",
      "__assert_fail",
      "cudaEventElapsedTime",
      "cudaEventSynchronize",
      "cudaDeviceGetAttribute",
      "cudaFuncGetAttributes",
      "cudaGetDevice",
      "cudaGetDeviceCount",
      "cudaMemGetInfo",
      "clock_gettime",
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
      "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
      "cudaEventRecord"};
  if (!CStyleMemRef) {
    if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
      if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
        StringRef name;
        if (auto *CC = dyn_cast<CXXConstructorDecl>(sr->getDecl()))
          name = Glob.CGM.getMangledName(
              GlobalDecl(CC, CXXCtorType::Ctor_Complete));
        else if (auto *CC = dyn_cast<CXXDestructorDecl>(sr->getDecl()))
          name = Glob.CGM.getMangledName(
              GlobalDecl(CC, CXXDtorType::Dtor_Complete));
        else if (sr->getDecl()->hasAttr<CUDAGlobalAttr>())
          name = Glob.CGM.getMangledName(GlobalDecl(
              cast<FunctionDecl>(sr->getDecl()), KernelReferenceKind::Kernel));
        else
          name = Glob.CGM.getMangledName(sr->getDecl());
        if (funcs.count(name.str()) || name.startswith("mkl_") ||
            name.startswith("MKL_") || name.startswith("cublas") ||
            name.startswith("cblas_")) {

          std::vector<mlir::Value> args;
          for (auto *a : expr->arguments()) {
            args.push_back(getLLVM(a, /*isRef*/ false));
          }
          mlir::Value called;

          if (callee) {
            auto strcmpF = Glob.GetOrCreateLLVMFunction(callee);
            called = builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args)
                         .getResult();
          } else {
            args.insert(args.begin(),
                        getLLVM(expr->getCallee(), /*isRef*/ false));
            SmallVector<mlir::Type> RTs = {Glob.typeTranslator.translateType(
                anonymize(getLLVMType(expr->getType())))};
            if (RTs[0].isa<LLVM::LLVMVoidType>())
              RTs.clear();
            called =
                builder.create<mlir::LLVM::CallOp>(loc, RTs, args).getResult();
          }
          return ValueCategory(called, /*isReference*/ expr->isLValue() ||
                                           expr->isXValue());
        }
      }
  }

  if (!callee || callee->isVariadic()) {
    bool isReference = expr->isLValue() || expr->isXValue();
    std::vector<mlir::Value> args;
    mlir::Value called;
    if (callee) {
      auto strcmpF = Glob.GetOrCreateLLVMFunction(callee);
      std::vector<clang::QualType> types;
      if (auto CC = dyn_cast<CXXMethodDecl>(callee)) {
        types.push_back(CC->getThisType());
      }
      for (auto parm : callee->parameters()) {
        types.push_back(parm->getOriginalType());
      }
      int i = 0;
      for (auto *a : expr->arguments()) {
        bool isRef = false;
        if (i < types.size())
          isRef = types[i]->isReferenceType();
        i++;
        args.push_back(getLLVM(a, isRef));
      }
      called =
          builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args).getResult();
    } else {
      mlir::Value fn = Visit(expr->getCallee()).getValue(loc, builder);
      if (auto MT = dyn_cast<MemRefType>(fn.getType())) {
        fn = builder.create<polygeist::Memref2PointerOp>(
            loc, LLVM::LLVMPointerType::get(MT.getElementType(), 0), fn);
      }
      auto PTF = fn.getType()
                     .cast<LLVM::LLVMPointerType>()
                     .getElementType()
                     .cast<LLVM::LLVMFunctionType>();
      SmallVector<mlir::Type, 1> argtys;
      bool needsChange = false;
      for (auto FT : PTF.getParams()) {
        if (auto mt = dyn_cast<MemRefType>(FT)) {
          argtys.push_back(LLVM::LLVMPointerType::get(mt.getElementType(), 0));
          needsChange = true;
        } else
          argtys.push_back(FT);
      }
      auto rt = PTF.getReturnType();
      if (auto mt = dyn_cast<MemRefType>(rt)) {
        rt = LLVM::LLVMPointerType::get(mt.getElementType(), 0);
        needsChange = true;
      }
      if (needsChange)
        fn = builder.create<LLVM::BitcastOp>(
            loc,
            LLVM::LLVMPointerType::get(
                LLVM::LLVMFunctionType::get(rt, argtys, PTF.isVarArg()), 0),
            fn);

      args.push_back(fn);
      auto CT = expr->getType();
      // if (isReference)
      //  CT = Glob.CGM.getContext().getLValueReferenceType(CT);
      SmallVector<mlir::Type> RTs = {rt};
      // getMLIRType(CT)};

      auto ft = args[0]
                    .getType()
                    .cast<LLVM::LLVMPointerType>()
                    .getElementType()
                    .cast<LLVM::LLVMFunctionType>();
      auto ETy = expr->getCallee()->getType()->getUnqualifiedDesugaredType();
      ETy = cast<clang::PointerType>(ETy)
                ->getPointeeType()
                ->getUnqualifiedDesugaredType();
      auto CFT = dyn_cast<clang::FunctionProtoType>(ETy);
      std::vector<clang::QualType> types;
      if (CFT) {
        for (auto t : CFT->getParamTypes())
          types.push_back(t);
      } else {
        assert(isa<clang::FunctionNoProtoType>(ETy));
      }

      auto ETy2 = ETy->getCanonicalTypeUnqualified();

      const clang::CodeGen::CGFunctionInfo *FI;
      if (const FunctionProtoType *FPT = dyn_cast<FunctionProtoType>(ETy2)) {
        FI = &Glob.CGM.getTypes().arrangeFreeFunctionType(
            CanQual<FunctionProtoType>::CreateUnsafe(QualType(FPT, 0)));
      } else {
        const FunctionNoProtoType *FNPT = cast<FunctionNoProtoType>(ETy2);
        FI = &Glob.CGM.getTypes().arrangeFreeFunctionType(
            CanQual<FunctionNoProtoType>::CreateUnsafe(QualType(FNPT, 0)));
      }

      int i = 0;
      for (auto *a : expr->arguments()) {
        bool isRef = false;
        bool isArray = false;
        if (i < types.size()) {
          isRef = types[i]->isReferenceType();
          // auto inf = FI->arguments()[i].info;
          // isRef |= inf.isIndirect();
          Glob.getMLIRType(types[i], &isArray);
          isRef |= isArray;
        }

        auto sub = Visit(a);
        mlir::Value v;
        if (isRef) {
          if (!sub.isReference) {
            OpBuilder abuilder(builder.getContext());
            abuilder.setInsertionPointToStart(allocationScope);
            auto one = abuilder.create<ConstantIntOp>(loc, 1, 64);
            auto alloc = abuilder.create<mlir::LLVM::AllocaOp>(
                loc, LLVM::LLVMPointerType::get(sub.val.getType()), one, 0);
            ValueCategory(alloc, /*isRef*/ true)
                .store(loc, builder, sub, /*isArray*/ false);
            sub = ValueCategory(alloc, /*isRef*/ true);
          }
          assert(sub.isReference);
          v = sub.val;
        } else {
          v = sub.getValue(loc, builder);
        }
        if (i < FI->arg_size()) {
          // TODO expand full calling conv
          /*
          auto inf = FI->arguments()[i].info;
          if (inf.isIgnore() || inf.isInAlloca()) {
            i++;
            continue;
          }
          if (inf.isExpand()) {
            i++;
            continue;
          }
          */
        }
        i++;
        if (auto mt = dyn_cast<MemRefType>(v.getType())) {
          v = builder.create<polygeist::Memref2PointerOp>(
              loc, LLVM::LLVMPointerType::get(mt.getElementType(), 0), v);
        }
        args.push_back(v);
      }
      if (RTs[0].isa<mlir::NoneType>() || RTs[0].isa<LLVM::LLVMVoidType>())
        RTs.clear();
      else
        assert(RTs[0] == ft.getReturnType());

      mlir::Block::iterator oldpoint;
      mlir::Block *oldblock = nullptr;

      if (auto *CU = dyn_cast<CUDAKernelCallExpr>(expr)) {
        auto l0 = Visit(CU->getConfig()->getArg(0));
        assert(l0.isReference);
        mlir::Value val = l0.val;
        mlir::Value blocks[3];
        if (auto MT = dyn_cast<MemRefType>(val.getType())) {
          if (MT.getElementType().isa<LLVM::LLVMStructType>() &&
              MT.getShape().size() == 1) {
            val = builder.create<polygeist::Memref2PointerOp>(
                loc,
                LLVM::LLVMPointerType::get(MT.getElementType(),
                                           MT.getMemorySpaceAsInt()),
                val);
          }
        }
        for (int i = 0; i < 3; i++) {
          if (auto MT = dyn_cast<MemRefType>(val.getType())) {
            mlir::Value idx[] = {getConstantIndex(0), getConstantIndex(i)};
            assert(MT.getShape().size() == 2);
            blocks[i] = builder.create<IndexCastOp>(
                loc, mlir::IndexType::get(builder.getContext()),
                builder.create<mlir::memref::LoadOp>(loc, val, idx));
          } else {
            mlir::Value idx[] = {
                builder.create<arith::ConstantIntOp>(loc, 0, 32),
                builder.create<arith::ConstantIntOp>(loc, i, 32)};
            auto PT = val.getType().cast<LLVM::LLVMPointerType>();
            auto ET =
                PT.getElementType().cast<LLVM::LLVMStructType>().getBody()[i];
            blocks[i] = builder.create<IndexCastOp>(
                loc, mlir::IndexType::get(builder.getContext()),
                builder.create<LLVM::LoadOp>(
                    loc,
                    builder.create<LLVM::GEPOp>(
                        loc,
                        LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                        val, idx)));
          }
        }

        auto t0 = Visit(CU->getConfig()->getArg(1));
        assert(t0.isReference);
        mlir::Value threads[3];
        val = t0.val;
        if (auto MT = dyn_cast<MemRefType>(val.getType())) {
          if (MT.getElementType().isa<LLVM::LLVMStructType>() &&
              MT.getShape().size() == 1) {
            val = builder.create<polygeist::Memref2PointerOp>(
                loc,
                LLVM::LLVMPointerType::get(MT.getElementType(),
                                           MT.getMemorySpaceAsInt()),
                val);
          }
        }
        for (int i = 0; i < 3; i++) {
          if (auto MT = dyn_cast<MemRefType>(val.getType())) {
            mlir::Value idx[] = {getConstantIndex(0), getConstantIndex(i)};
            assert(MT.getShape().size() == 2);
            threads[i] = builder.create<IndexCastOp>(
                loc, mlir::IndexType::get(builder.getContext()),
                builder.create<mlir::memref::LoadOp>(loc, val, idx));
          } else {
            mlir::Value idx[] = {
                builder.create<arith::ConstantIntOp>(loc, 0, 32),
                builder.create<arith::ConstantIntOp>(loc, i, 32)};
            auto PT = val.getType().cast<LLVM::LLVMPointerType>();
            auto ET =
                PT.getElementType().cast<LLVM::LLVMStructType>().getBody()[i];
            threads[i] = builder.create<IndexCastOp>(
                loc, mlir::IndexType::get(builder.getContext()),
                builder.create<LLVM::LoadOp>(
                    loc,
                    builder.create<LLVM::GEPOp>(
                        loc,
                        LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                        val, idx)));
          }
        }
        mlir::Value stream = nullptr;
        SmallVector<mlir::Value, 1> asyncDependencies;
        if (3 < CU->getConfig()->getNumArgs() &&
            !isa<CXXDefaultArgExpr>(CU->getConfig()->getArg(3))) {
          stream = Visit(CU->getConfig()->getArg(3)).getValue(loc, builder);
          stream = builder.create<polygeist::StreamToTokenOp>(
              loc, builder.getType<gpu::AsyncTokenType>(), stream);
          assert(stream);
          asyncDependencies.push_back(stream);
        }
        auto op = builder.create<mlir::gpu::LaunchOp>(
            loc, blocks[0], blocks[1], blocks[2], threads[0], threads[1],
            threads[2],
            /*dynamic shmem size*/ nullptr,
            /*token type*/ stream ? stream.getType() : nullptr,
            /*dependencies*/ asyncDependencies);
        oldpoint = builder.getInsertionPoint();
        oldblock = builder.getInsertionBlock();
        builder.setInsertionPointToStart(&op.getRegion().front());
      }

      called = builder.create<mlir::LLVM::CallOp>(loc, RTs, args).getResult();
      if (PTF.getReturnType() != ft.getReturnType()) {
        called = builder.create<polygeist::Pointer2MemrefOp>(
            loc, PTF.getReturnType(), called);
      }

      if (oldblock) {
        builder.create<gpu::TerminatorOp>(loc);
        builder.setInsertionPoint(oldblock, oldpoint);
        return ValueCategory();
      }
    }

    if (isReference) {
      if (!(called.getType().isa<LLVM::LLVMPointerType>() ||
            called.getType().isa<MemRefType>())) {
        expr->dump();
        expr->getType()->dump();
        llvm::errs() << " call: " << called << "\n";
      }
    }
    if (called)
      return ValueCategory(called, isReference);
    else
      return ValueCategory();
  }

  // (--xlx-headers) A member of hls::stream called on an object of a class
  // derived from the member's class (hls::stream<T, D> over
  // hls::stream<T, 0>) is named as a member of the object's class.
  const clang::CXXRecordDecl *xlxStreamObject = nullptr;
  if (const auto *md = dyn_cast<CXXMethodDecl>(callee);
      md && Glob.isXlxStreamClass(md->getParent())) {
    const clang::Expr *object = nullptr;
    if (auto *mc = dyn_cast<CXXMemberCallExpr>(expr))
      object = mc->getImplicitObjectArgument();
    else if (isa<CXXOperatorCallExpr>(expr) && expr->getNumArgs() > 0)
      object = expr->getArg(0);
    if (object) {
      QualType t = object->IgnoreParenImpCasts()->getType();
      if (t->isPointerType())
        t = t->getPointeeType();
      xlxStreamObject = t->getAsCXXRecordDecl();
    }
  }
  auto tocall = xlxStreamObject
                    ? Glob.GetOrCreateMLIRFunction(callee, false,
                                                   xlxStreamObject)
                    : EmitDirectCallee(callee);

  SmallVector<std::pair<ValueCategory, clang::Expr *>> args;
  QualType objType;

  if (auto *CC = dyn_cast<CXXMemberCallExpr>(expr)) {
    ValueCategory obj = Visit(CC->getImplicitObjectArgument());
    objType = CC->getObjectType();
    if (!obj.val) {
      function.dump();
      llvm::errs() << " objval: " << obj.val << "\n";
      expr->dump();
      CC->getImplicitObjectArgument()->dump();
    }
    if (cast<MemberExpr>(CC->getCallee()->IgnoreParens())->isArrow()) {
      obj = obj.dereference(loc, builder);
    }
    assert(obj.val);
    assert(obj.isReference);
    args.emplace_back(make_pair(obj, (clang::Expr *)nullptr));
  }
  for (auto *a : expr->arguments())
    args.push_back(make_pair(Visit(a), a));
  return CallHelper(tocall, objType, args, expr->getType(),
                    expr->isLValue() || expr->isXValue(), expr);
}

//===- clang-mlir.cc - Emit MLIR IRs by walking clang AST--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-mlir.h"
#include "utils.h"
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/FileSystemOptions.h>
#include <clang/Basic/LangStandard.h>
#include <clang/Basic/OperatorKinds.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Basic/Version.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Parse/Parser.h>
#include <clang/Sema/Sema.h>
#include <clang/Sema/SemaDiagnostic.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/SCF/SCF.h>

using namespace std;
using namespace clang;
using namespace llvm;
using namespace clang::driver;
using namespace llvm::opt;
using namespace mlir;
using namespace mlir::arith;

#define DEBUG_TYPE "clang-mlir"

static cl::opt<bool>
    memRefFullRank("memref-fullrank", cl::init(false),
                   cl::desc("Get the full rank of the memref."));

mlir::Attribute wrapIntegerMemorySpace(unsigned memorySpace, MLIRContext *ctx) {
  if (memorySpace == 0)
    return nullptr;

  return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), memorySpace);
}

/// Try to typecast the caller arg of type MemRef to fit the corresponding
/// callee arg type. We only deal with the cast where src and dst have the same
/// shape size and elem type, and just the first shape differs: src has -1 and
/// dst has a constant integer.
static mlir::Value castCallerMemRefArg(mlir::Value callerArg,
                                       mlir::Type calleeArgType,
                                       mlir::OpBuilder &b) {
  mlir::OpBuilder::InsertionGuard guard(b);
  mlir::Type callerArgType = callerArg.getType();

  if (MemRefType dstTy = calleeArgType.dyn_cast_or_null<MemRefType>()) {
    MemRefType srcTy = callerArgType.dyn_cast<MemRefType>();
    if (srcTy && dstTy.getElementType() == srcTy.getElementType() &&
        dstTy.getMemorySpace() == srcTy.getMemorySpace()) {
      auto srcShape = srcTy.getShape();
      auto dstShape = dstTy.getShape();

      if (srcShape.size() == dstShape.size() && !srcShape.empty() &&
          srcShape[0] == -1 &&
          std::equal(std::next(srcShape.begin()), srcShape.end(),
                     std::next(dstShape.begin()))) {
        b.setInsertionPointAfterValue(callerArg);

        return b.create<mlir::memref::CastOp>(callerArg.getLoc(), callerArg,
                                              calleeArgType);
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
static void castCallerArgs(mlir::FuncOp callee,
                           llvm::SmallVectorImpl<mlir::Value> &args,
                           mlir::OpBuilder &b) {
  mlir::FunctionType funcTy = callee.getType().cast<mlir::FunctionType>();
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

MLIRScanner::MLIRScanner(MLIRASTConsumer &Glob,
                         mlir::OwningOpRef<mlir::ModuleOp> &module,
                         LowerToInfo &LTInfo)
    : Glob(Glob), module(module), builder(module->getContext()),
      loc(builder.getUnknownLoc()), ThisCapture(nullptr), LTInfo(LTInfo) {}

void MLIRScanner::init(mlir::FuncOp function, const FunctionDecl *fd) {
  this->function = function;
  this->EmittingFunctionDecl = fd;

  if (ShowAST) {
    llvm::errs() << "Emitting fn: " << function.getName() << "\n";
    llvm::errs() << *fd << "\n";
  }

  setEntryAndAllocBlock(function.addEntryBlock());

  unsigned i = 0;
  if (auto CM = dyn_cast<CXXMethodDecl>(fd)) {
    if (CM->getParent()->isLambda()) {
      for (auto C : CM->getParent()->captures()) {
        if (C.capturesVariable()) {
          CaptureKinds[C.getCapturedVar()] = C.getCaptureKind();
        }
      }
      CM->getParent()->getCaptureFields(Captures, ThisCapture);
      if (ThisCapture) {
        llvm::errs() << " thiscapture:\n";
        ThisCapture->dump();
      }
    }

    if (CM->isInstance()) {
      mlir::Value val = function.getArgument(i);
      ThisVal = ValueCategory(val, /*isReference*/ false);
      i++;
    }
  }

  for (auto parm : fd->parameters()) {
    assert(i != function.getNumArguments());
    // function.getArgument(i).setName(name);
    bool isArray = false;
    bool LLVMABI = false;

    if (Glob.getMLIRType(Glob.CGM.getContext().getPointerType(parm->getType()))
            .isa<mlir::LLVM::LLVMPointerType>())
      LLVMABI = true;

    if (!LLVMABI) {
      Glob.getMLIRType(parm->getType(), &isArray);
    }
    if (!isArray && isa<clang::ReferenceType>(
                        parm->getType()->getUnqualifiedDesugaredType()))
      isArray = true;
    mlir::Value val = function.getArgument(i);
    assert(val);
    if (isArray) {
      params.emplace(parm, ValueCategory(val, /*isReference*/ true));
    } else {
      auto alloc = createAllocOp(val.getType(), parm, /*memspace*/ 0, isArray,
                                 /*LLVMABI*/ LLVMABI);
      ValueCategory(alloc, /*isReference*/ true).store(builder, val);
    }
    i++;
  }

  if (auto CC = dyn_cast<CXXConstructorDecl>(fd)) {
    const CXXRecordDecl *ClassDecl = CC->getParent();
    for (auto expr : CC->inits()) {
      if (ShowAST) {
        llvm::errs() << " init: - baseInit:" << (int)expr->isBaseInitializer()
                     << " memberInit:" << (int)expr->isMemberInitializer()
                     << " anyMember:" << (int)expr->isAnyMemberInitializer()
                     << " indirectMember:"
                     << (int)expr->isIndirectMemberInitializer()
                     << " isinClass:" << (int)expr->isInClassMemberInitializer()
                     << " delegating:" << (int)expr->isDelegatingInitializer()
                     << " isPack:" << (int)expr->isPackExpansion() << "\n";
        if (expr->getMember())
          expr->getMember()->dump();
        if (expr->getInit())
          expr->getInit()->dump();
      }
      assert(ThisVal.val);
      FieldDecl *field = expr->getMember();
      if (!field) {
        if (expr->isBaseInitializer()) {
          bool BaseIsVirtual = expr->isBaseVirtual();

          auto BaseType = expr->getBaseClass();

          auto BaseClassDecl =
              cast<CXXRecordDecl>(BaseType->castAs<RecordType>()->getDecl());

          clang::CharUnits Offset;
          const ASTRecordLayout &Layout =
              Glob.astContext.getASTRecordLayout(ClassDecl);
          if (BaseIsVirtual)
            Offset = Layout.getVBaseClassOffset(BaseClassDecl);
          else
            Offset = Layout.getBaseClassOffset(BaseClassDecl);

          // Shift and cast down to the base type.
          // TODO: for complete types, this should be possible with a GEP.
          mlir::Value V = ThisVal.val;
          if (!Offset.isZero()) {
            V = builder.create<LLVM::BitcastOp>(
                loc,
                LLVM::LLVMPointerType::get(builder.getI8Type(),
                                           V.getType()
                                               .cast<LLVM::LLVMPointerType>()
                                               .getAddressSpace()),
                V);
            mlir::Value idxs[] = {
                builder.create<ConstantIntOp>(loc, Offset.getQuantity(), 32)};
            V = builder.create<LLVM::GEPOp>(loc, V.getType(), V, idxs);
          }

          bool isArray = false;
          auto subType = LLVM::LLVMPointerType::get(
              Glob.getMLIRType(QualType(BaseType, 0), &isArray,
                               /*allowMerge*/ false),
              V.getType().cast<LLVM::LLVMPointerType>().getAddressSpace());
          assert(!isArray && "implicit reference not handled");

          V = builder.create<LLVM::BitcastOp>(loc, subType, V);

          isArray = false;
          auto subType2 =
              Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                   QualType(BaseType, 0)),
                               &isArray);
          if (subType2.isa<MemRefType>())
            V = builder.create<polygeist::Pointer2MemrefOp>(loc, subType2, V);

          Expr *init = expr->getInit();
          if (auto clean = dyn_cast<ExprWithCleanups>(init)) {
            llvm::errs() << "TODO: cleanup\n";
            init = clean->getSubExpr();
          }

          VisitConstructCommon(cast<clang::CXXConstructExpr>(init),
                               /*name*/ nullptr, /*space*/ 0, /*mem*/ V);
          continue;
        }
        if (expr->isDelegatingInitializer()) {

          Expr *init = expr->getInit();
          if (auto clean = dyn_cast<ExprWithCleanups>(init)) {
            llvm::errs() << "TODO: cleanup\n";
            init = clean->getSubExpr();
          }

          VisitConstructCommon(cast<clang::CXXConstructExpr>(init),
                               /*name*/ nullptr, /*space*/ 0,
                               /*mem*/ ThisVal.val);
          continue;
        }
      }
      assert(field && "initialiation expression must apply to a field");
      if (auto AILE = dyn_cast<ArrayInitLoopExpr>(expr->getInit())) {
        VisitArrayInitLoop(AILE,
                           CommonFieldLookup(CC->getThisObjectType(), field,
                                             ThisVal.val, /*isLValue*/ false));
        continue;
      }
      if (auto cons = dyn_cast<CXXConstructExpr>(expr->getInit())) {
        VisitConstructCommon(cons, /*name*/ nullptr, /*space*/ 0,
                             CommonFieldLookup(CC->getThisObjectType(), field,
                                               ThisVal.val, /*isLValue*/ false)
                                 .val);
        continue;
      }
      auto initexpr = Visit(expr->getInit());
      if (!initexpr.val) {
        expr->getInit()->dump();
        assert(initexpr.val);
      }
      bool isArray = false;
      Glob.getMLIRType(expr->getInit()->getType(), &isArray);

      auto cfl = CommonFieldLookup(CC->getThisObjectType(), field, ThisVal.val,
                                   /*isLValue*/ false);
      assert(cfl.val);
      cfl.store(builder, initexpr, isArray);
    }
  }
  if (auto CC = dyn_cast<CXXDestructorDecl>(fd)) {
    CC->dump();
    llvm::errs() << " warning, destructor not fully handled yet\n";
  }

  Stmt *stmt = fd->getBody();
  assert(stmt);
  if (ShowAST) {
    stmt->dump();
  }

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back(
      (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                    builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().keepRunning);
  if (function.getType().getResults().size()) {
    auto type =
        mlir::MemRefType::get({}, function.getType().getResult(0), {}, 0);
    returnVal = builder.create<mlir::memref::AllocaOp>(loc, type);
    if (type.getElementType().isa<mlir::IntegerType>()) {
      builder.create<mlir::memref::StoreOp>(
          loc, builder.create<mlir::LLVM::UndefOp>(loc, type.getElementType()),
          returnVal, std::vector<mlir::Value>({}));
    }
  }
  Visit(stmt);

  if (function.getType().getResults().size()) {
    mlir::Value vals[1] = {
        builder.create<mlir::memref::LoadOp>(loc, returnVal)};
    builder.create<mlir::ReturnOp>(loc, vals);
  } else
    builder.create<mlir::ReturnOp>(loc);

  assert(function->getParentOp() == Glob.module.get() &&
         "New function must be inserted into global module");
}

mlir::OpBuilder &MLIRScanner::getBuilder() { return builder; }

mlir::Value MLIRScanner::createAllocOp(mlir::Type t, VarDecl *name,
                                       uint64_t memspace, bool isArray = false,
                                       bool LLVMABI = false) {

  mlir::MemRefType mr;
  mlir::Value alloc = nullptr;
  OpBuilder abuilder(builder.getContext());
  abuilder.setInsertionPointToStart(allocationScope);
  auto varLoc = name ? getMLIRLocation(name->getBeginLoc()) : loc;
  if (!isArray) {
    if (LLVMABI) {
      if (name)
        if (auto var = dyn_cast<VariableArrayType>(
                name->getType()->getUnqualifiedDesugaredType())) {
          auto len = Visit(var->getSizeExpr()).getValue(builder);
          alloc = builder.create<mlir::LLVM::AllocaOp>(
              varLoc, LLVM::LLVMPointerType::get(t, memspace), len);
          builder.create<polygeist::TrivialUseOp>(varLoc, alloc);
          alloc = builder.create<mlir::LLVM::BitcastOp>(
              varLoc,
              LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(t, 0)),
              alloc);
        }

      if (!alloc) {
        alloc = abuilder.create<mlir::LLVM::AllocaOp>(
            varLoc, mlir::LLVM::LLVMPointerType::get(t, memspace),
            abuilder.create<ConstantIntOp>(varLoc, 1, 64), 0);
        if (t.isa<mlir::IntegerType>()) {
          abuilder.create<LLVM::StoreOp>(
              varLoc, abuilder.create<mlir::LLVM::UndefOp>(varLoc, t), alloc);
        }
        // alloc = builder.create<mlir::LLVM::BitcastOp>(varLoc,
        // LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(t, 1)), alloc);
      }
    } else {
      mr = mlir::MemRefType::get(1, t, {}, memspace);
      alloc = abuilder.create<mlir::memref::AllocaOp>(varLoc, mr);
      if (memspace != 0) {
        alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
            varLoc, mlir::MemRefType::get(-1, t, {}, memspace),
            abuilder.create<polygeist::Memref2PointerOp>(
                varLoc, LLVM::LLVMPointerType::get(t, 0), alloc));
      }
      alloc = abuilder.create<mlir::memref::CastOp>(
          varLoc, alloc, mlir::MemRefType::get(-1, t, {}, 0));
      if (t.isa<mlir::IntegerType>()) {
        mlir::Value idxs[] = {abuilder.create<ConstantIndexOp>(loc, 0)};
        abuilder.create<mlir::memref::StoreOp>(
            varLoc, abuilder.create<mlir::LLVM::UndefOp>(varLoc, t), alloc,
            idxs);
      }
    }
  } else {
    auto mt = t.cast<mlir::MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    auto pshape = shape[0];

    if (name)
      if (auto var = dyn_cast<VariableArrayType>(
              name->getType()->getUnqualifiedDesugaredType())) {
        assert(shape[0] == -1);
        mr = mlir::MemRefType::get(
            shape, mt.getElementType(), MemRefLayoutAttrInterface(),
            wrapIntegerMemorySpace(memspace, mt.getContext()));
        auto len = Visit(var->getSizeExpr()).getValue(builder);
        len = builder.create<IndexCastOp>(varLoc, len, builder.getIndexType());
        alloc = builder.create<mlir::memref::AllocaOp>(varLoc, mr, len);
        builder.create<polygeist::TrivialUseOp>(varLoc, alloc);
        if (memspace != 0) {
          alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
              varLoc, mlir::MemRefType::get(shape, mt.getElementType()),
              abuilder.create<polygeist::Memref2PointerOp>(
                  varLoc, LLVM::LLVMPointerType::get(mt.getElementType(), 0),
                  alloc));
        }
      }

    if (!alloc) {
      if (pshape == -1)
        shape[0] = 1;
      mr = mlir::MemRefType::get(
          shape, mt.getElementType(), MemRefLayoutAttrInterface(),
          wrapIntegerMemorySpace(memspace, mt.getContext()));
      alloc = abuilder.create<mlir::memref::AllocaOp>(varLoc, mr);
      if (memspace != 0) {
        alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
            varLoc, mlir::MemRefType::get(shape, mt.getElementType()),
            abuilder.create<polygeist::Memref2PointerOp>(
                varLoc, LLVM::LLVMPointerType::get(mt.getElementType(), 0),
                alloc));
      }
      shape[0] = pshape;
      alloc = abuilder.create<mlir::memref::CastOp>(
          varLoc, alloc, mlir::MemRefType::get(shape, mt.getElementType()));
    }
  }
  assert(alloc);
  // NamedAttribute attrs[] = {NamedAttribute("name", name)};
  if (name) {
    // if (name->getName() == "i")
    //  assert(0 && " not i");
    if (params.find(name) != params.end()) {
      name->dump();
    }
    assert(params.find(name) == params.end());
    params[name] = ValueCategory(alloc, /*isReference*/ true);
  }
  return alloc;
}

ValueCategory MLIRScanner::VisitConstantExpr(clang::ConstantExpr *expr) {
  auto sv = Visit(expr->getSubExpr());
  if (auto ty = getMLIRType(expr->getType()).dyn_cast<mlir::IntegerType>()) {
    if (expr->hasAPValueResult()) {
      return ValueCategory(builder.create<ConstantIntOp>(
                               getMLIRLocation(expr->getExprLoc()),
                               expr->getResultAsAPSInt().getExtValue(), ty),
                           /*isReference*/ false);
    }
  }
  assert(sv.val);
  return sv;
}

ValueCategory MLIRScanner::VisitTypeTraitExpr(clang::TypeTraitExpr *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                    expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitIntegerLiteral(clang::IntegerLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                    expr->getValue().getSExtValue(), ty),
      /*isReference*/ false);
}

ValueCategory
MLIRScanner::VisitCharacterLiteral(clang::CharacterLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                    expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitFloatingLiteral(clang::FloatingLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::FloatType>();
  return ValueCategory(
      builder.create<ConstantFloatOp>(getMLIRLocation(expr->getExprLoc()),
                                      expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory
MLIRScanner::VisitImaginaryLiteral(clang::ImaginaryLiteral *expr) {
  auto mt = getMLIRType(expr->getType()).cast<MemRefType>();
  auto ty = mt.getElementType().cast<FloatType>();

  OpBuilder abuilder(builder.getContext());
  abuilder.setInsertionPointToStart(allocationScope);
  auto iloc = getMLIRLocation(expr->getExprLoc());
  auto alloc = abuilder.create<mlir::memref::AllocaOp>(iloc, mt);
  builder.create<mlir::memref::StoreOp>(
      iloc,
      builder.create<ConstantFloatOp>(iloc,
                                      APFloat(ty.getFloatSemantics(), "0"), ty),
      alloc, getConstantIndex(0));
  builder.create<mlir::memref::StoreOp>(
      iloc, Visit(expr->getSubExpr()).getValue(builder), alloc,
      getConstantIndex(1));
  return ValueCategory(alloc,
                       /*isReference*/ true);
}

ValueCategory
MLIRScanner::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                    expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitStringLiteral(clang::StringLiteral *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  return ValueCategory(
      Glob.GetOrCreateGlobalLLVMString(loc, builder, expr->getString()),
      /*isReference*/ true);
}

ValueCategory MLIRScanner::VisitParenExpr(clang::ParenExpr *expr) {
  return Visit(expr->getSubExpr());
}

ValueCategory
MLIRScanner::VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *decl) {
  auto Mty = getMLIRType(decl->getType());

  if (auto FT = Mty.dyn_cast<mlir::FloatType>())
    return ValueCategory(builder.create<ConstantFloatOp>(
                             loc, APFloat(FT.getFloatSemantics(), "0"), FT),
                         /*isReference*/ false);
  if (auto IT = Mty.dyn_cast<mlir::IntegerType>())
    return ValueCategory(builder.create<ConstantIntOp>(loc, 0, IT),
                         /*isReference*/ false);
  if (auto MT = Mty.dyn_cast<mlir::MemRefType>())
    return ValueCategory(
        builder.create<polygeist::Pointer2MemrefOp>(
            loc, MT,
            builder.create<mlir::LLVM::NullOp>(
                loc, LLVM::LLVMPointerType::get(builder.getI8Type()))),
        false);
  if (auto PT = Mty.dyn_cast<mlir::LLVM::LLVMPointerType>())
    return ValueCategory(builder.create<mlir::LLVM::NullOp>(loc, PT), false);
  for (auto child : decl->children()) {
    child->dump();
  }
  decl->dump();
  llvm::errs() << " mty: " << Mty << "\n";
  assert(0 && "bad");
}

/// Construct corresponding MLIR operations to initialize the given value by a
/// provided InitListExpr.
mlir::Attribute MLIRScanner::InitializeValueByInitListExpr(mlir::Value toInit,
                                                           clang::Expr *expr) {
  // Struct initializan requires an extra 0, since the first index
  // is the pointer index, and then the struct index.
  auto PTT = expr->getType()->getUnqualifiedDesugaredType();

  bool inner = false;
  if (isa<RecordType>(PTT) || isa<clang::ComplexType>(PTT)) {
    if (auto mt = toInit.getType().dyn_cast<MemRefType>()) {
      inner = true;
    }
  }

  while (auto CO = toInit.getDefiningOp<memref::CastOp>())
    toInit = CO.source();

  // Recursively visit the initialization expression following the linear
  // increment of the memory address.
  std::function<mlir::DenseElementsAttr(Expr *, mlir::Value, bool)> helper =
      [&](Expr *expr, mlir::Value toInit,
          bool inner) -> mlir::DenseElementsAttr {
    Location loc = toInit.getLoc();
    if (InitListExpr *initListExpr = dyn_cast<InitListExpr>(expr)) {

      if (inner) {
        if (auto mt = toInit.getType().dyn_cast<MemRefType>()) {
          auto shape = std::vector<int64_t>(mt.getShape());
          shape.erase(shape.begin());
          auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                           MemRefLayoutAttrInterface(),
                                           mt.getMemorySpace());
          toInit = builder.create<polygeist::SubIndexOp>(loc, mt0, toInit,
                                                         getConstantIndex(0));
        }
      }

      unsigned num = 0;
      if (initListExpr->hasArrayFiller()) {
        if (auto MT = toInit.getType().dyn_cast<MemRefType>()) {
          auto shape = MT.getShape();
          assert(shape.size() > 0);
          assert(shape[0] != -1);
          num = shape[0];
        } else
          assert(0 && "TODO get number of values in array filler expression");
      } else {
        num = initListExpr->getNumInits();
      }

      SmallVector<char> attrs;
      bool allSub = true;
      for (unsigned i = 0, e = num; i < e; ++i) {

        mlir::Value next;
        if (auto mt = toInit.getType().dyn_cast<MemRefType>()) {
          auto shape = std::vector<int64_t>(mt.getShape());
          shape[0] = -1;
          auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                           MemRefLayoutAttrInterface(),
                                           mt.getMemorySpace());
          next = builder.create<polygeist::SubIndexOp>(loc, mt0, toInit,
                                                       getConstantIndex(i));
        } else {
          auto PT = toInit.getType().cast<LLVM::LLVMPointerType>();
          auto ET = PT.getElementType();
          mlir::Type nextType;
          if (auto ST = ET.dyn_cast<LLVM::LLVMStructType>())
            nextType = ST.getBody()[i];
          else if (auto AT = ET.dyn_cast<LLVM::LLVMArrayType>())
            nextType = AT.getElementType();
          else
            assert(0 && "unknown inner type");

          mlir::Value idxs[] = {
              builder.create<ConstantIntOp>(loc, 0, 32),
              builder.create<ConstantIntOp>(loc, i, 32),
          };
          next = builder.create<LLVM::GEPOp>(
              loc, LLVM::LLVMPointerType::get(nextType, PT.getAddressSpace()),
              toInit, idxs);
        }

        auto sub =
            helper(initListExpr->hasArrayFiller() ? initListExpr->getInit(0)
                                                  : initListExpr->getInit(i),
                   next, true);
        if (sub) {
          size_t n = 1;
          if (sub.isSplat())
            n = sub.size();
          for (size_t i = 0; i < n; i++)
            for (auto ea : sub.getRawData())
              attrs.push_back(ea);
        } else {
          allSub = false;
        }
      }
      if (!allSub)
        return mlir::DenseElementsAttr();
      if (auto mt = toInit.getType().dyn_cast<MemRefType>()) {
        return DenseElementsAttr::getFromRawBuffer(
            RankedTensorType::get(mt.getShape(), mt.getElementType()), attrs,
            false);
      }
      return mlir::DenseElementsAttr();
    } else {
      bool isArray = false;
      Glob.getMLIRType(expr->getType(), &isArray);
      ValueCategory sub = Visit(expr);
      ValueCategory(toInit, /*isReference*/ true).store(builder, sub, isArray);
      if (!sub.isReference)
        if (auto mt = toInit.getType().dyn_cast<MemRefType>()) {
          if (auto cop = sub.val.getDefiningOp<ConstantIntOp>())
            return DenseElementsAttr::get(
                RankedTensorType::get(std::vector<int64_t>({1}),
                                      mt.getElementType()),
                cop.getValue());
          if (auto cop = sub.val.getDefiningOp<ConstantFloatOp>())
            return DenseElementsAttr::get(
                RankedTensorType::get(std::vector<int64_t>({1}),
                                      mt.getElementType()),
                cop.getValue());
        }
      return mlir::DenseElementsAttr();
    }
  };

  return helper(expr, toInit, inner);
}

ValueCategory MLIRScanner::VisitVarDecl(clang::VarDecl *decl) {
  mlir::Type subType = getMLIRType(decl->getType());
  ValueCategory inite = nullptr;
  unsigned memtype = decl->hasAttr<CUDASharedAttr>() ? 5 : 0;
  bool LLVMABI = false;
  bool isArray = false;

  if (Glob.getMLIRType(
              Glob.CGM.getContext().getLValueReferenceType(decl->getType()))
          .isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else
    Glob.getMLIRType(decl->getType(), &isArray);

  if (!LLVMABI && isArray) {
    subType = Glob.getMLIRType(
        Glob.CGM.getContext().getLValueReferenceType(decl->getType()));
  }

  if (auto init = decl->getInit()) {
    if (!isa<InitListExpr>(init) && !isa<CXXConstructExpr>(init)) {
      auto visit = Visit(init);
      if (!visit.val) {
        decl->dump();
        assert(visit.val);
      }
      bool isReference = init->isLValue() || init->isXValue();
      if (isReference) {
        assert(visit.isReference);
        builder.create<polygeist::TrivialUseOp>(loc, visit.val);
        return params[decl] = visit;
      }
      if (isArray) {
        assert(visit.isReference);
        inite = visit;
      } else {
        inite = ValueCategory(visit.getValue(builder), /*isRef*/ false);
        if (!inite.val) {
          init->dump();
          assert(0 && inite.val);
        }
        subType = inite.val.getType();
      }
    }
  } else if (auto ava = decl->getAttr<AlignValueAttr>()) {
    if (auto algn = dyn_cast<clang::ConstantExpr>(ava->getAlignment())) {
      for (auto a : algn->children()) {
        if (auto IL = dyn_cast<IntegerLiteral>(a)) {
          if (IL->getValue() == 8192) {
            llvm::Type *T = Glob.CGM.getTypes().ConvertType(decl->getType());
            subType = Glob.typeTranslator.translateType(T);
            LLVMABI = true;
            break;
          }
        }
      }
    }
  } else if (auto ava = decl->getAttr<InitPriorityAttr>()) {
    if (ava->getPriority() == 8192) {
      llvm::Type *T = Glob.CGM.getTypes().ConvertType(decl->getType());
      subType = Glob.typeTranslator.translateType(T);
      LLVMABI = true;
    }
  }

  mlir::Value op;

  Block *block = nullptr;
  Block::iterator iter;

  if (decl->isStaticLocal() && memtype == 0) {
    auto gv = Glob.GetOrCreateGlobal(
        decl, (function.getName() + "@static@").str(), /*tryInit*/ false);
    OpBuilder abuilder(builder.getContext());
    abuilder.setInsertionPointToStart(allocationScope);
    auto varLoc = getMLIRLocation(decl->getBeginLoc());
    op = abuilder.create<memref::GetGlobalOp>(varLoc, gv.first.type(),
                                              gv.first.getName());
    params[decl] = ValueCategory(op, /*isReference*/ true);
    if (decl->getInit()) {
      auto mr = MemRefType::get({1}, builder.getI1Type());
      bool inits[1] = {true};
      auto rtt = RankedTensorType::get({1}, builder.getI1Type());
      auto init_value = DenseIntElementsAttr::get(rtt, inits);
      OpBuilder gbuilder(builder.getContext());
      gbuilder.setInsertionPointToStart(module->getBody());
      auto name = Glob.CGM.getMangledName(decl);
      auto globalOp = gbuilder.create<mlir::memref::GlobalOp>(
          module->getLoc(),
          builder.getStringAttr(function.getName() + "@static@" + name +
                                "@init"),
          /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(mr),
          init_value, mlir::UnitAttr(), /*alignment*/ nullptr);
      SymbolTable::setSymbolVisibility(globalOp,
                                       mlir::SymbolTable::Visibility::Private);

      auto boolop =
          builder.create<memref::GetGlobalOp>(varLoc, mr, globalOp.getName());
      auto cond = builder.create<memref::LoadOp>(
          varLoc, boolop, std::vector<mlir::Value>({getConstantIndex(0)}));

      auto ifOp = builder.create<scf::IfOp>(varLoc, cond, /*hasElse*/ false);
      block = builder.getInsertionBlock();
      iter = builder.getInsertionPoint();
      builder.setInsertionPointToStart(&ifOp.getThenRegion().back());
      builder.create<memref::StoreOp>(
          varLoc, builder.create<ConstantIntOp>(varLoc, false, 1), boolop,
          std::vector<mlir::Value>({getConstantIndex(0)}));
    }
  } else
    op = createAllocOp(subType, decl, memtype, isArray, LLVMABI);

  if (inite.val) {
    ValueCategory(op, /*isReference*/ true).store(builder, inite, isArray);
  } else if (auto init = decl->getInit()) {
    if (isa<InitListExpr>(init)) {
      InitializeValueByInitListExpr(op, init);
    } else if (auto CE = dyn_cast<CXXConstructExpr>(init)) {
      VisitConstructCommon(CE, decl, memtype, op);
    } else
      assert(0 && "unknown init list");
  }
  if (block)
    builder.setInsertionPoint(block, iter);
  return ValueCategory(op, /*isReference*/ true);
}

ValueCategory
MLIRScanner::VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr) {
  return Visit(expr->getExpr());
}

ValueCategory MLIRScanner::VisitCXXThisExpr(clang::CXXThisExpr *expr) {
  return ThisVal;
}

ValueCategory MLIRScanner::VisitPredefinedExpr(clang::PredefinedExpr *expr) {
  return VisitStringLiteral(expr->getFunctionName());
}

ValueCategory MLIRScanner::VisitInitListExpr(clang::InitListExpr *expr) {
  mlir::Type subType = getMLIRType(expr->getType());
  bool isArray = false;
  bool LLVMABI = false;

  if (Glob.getMLIRType(
              Glob.CGM.getContext().getLValueReferenceType(expr->getType()))
          .isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else {
    Glob.getMLIRType(expr->getType(), &isArray);
    if (isArray)
      subType = Glob.getMLIRType(
          Glob.CGM.getContext().getLValueReferenceType(expr->getType()));
  }
  auto op = createAllocOp(subType, nullptr, /*memtype*/ 0, isArray, LLVMABI);
  InitializeValueByInitListExpr(op, expr);
  return ValueCategory(op, true);
}

ValueCategory
MLIRScanner::VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *expr) {
  assert(arrayinit.size());
  return ValueCategory(builder.create<IndexCastOp>(
                           loc, getMLIRType(expr->getType()), arrayinit.back()),
                       /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitArrayInitLoop(clang::ArrayInitLoopExpr *expr,
                                              ValueCategory tostore) {
  auto CAT = dyn_cast<clang::ConstantArrayType>(expr->getType());
  llvm::errs() << "warning recomputing common in arrayinitloopexpr\n";
  std::vector<mlir::Value> start = {getConstantIndex(0)};
  std::vector<mlir::Value> sizes = {
      getConstantIndex(CAT->getSize().getLimitedValue())};
  AffineMap map = builder.getSymbolIdentityMap();
  auto affineOp = builder.create<AffineForOp>(loc, start, map, sizes, map);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&affineOp.getLoopBody().front());

  arrayinit.push_back(affineOp.getInductionVar());

  auto alu =
      CommonArrayLookup(CommonArrayToPointer(tostore),
                        affineOp.getInductionVar(), /*isImplicitRef*/ false);

  if (auto AILE = dyn_cast<ArrayInitLoopExpr>(expr->getSubExpr())) {
    VisitArrayInitLoop(AILE, alu);
  } else {
    auto val = Visit(expr->getSubExpr());
    if (!val.val) {
      expr->dump();
      expr->getSubExpr()->dump();
    }
    assert(val.val);
    assert(tostore.isReference);
    bool isArray = false;
    Glob.getMLIRType(expr->getSubExpr()->getType(), &isArray);
    alu.store(builder, val, isArray);
  }

  arrayinit.pop_back();

  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory
MLIRScanner::VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr) {
  if (expr->getType()->isVoidType()) {
    Visit(expr->getSubExpr());
    return nullptr;
  }
  if (expr->getCastKind() == clang::CastKind::CK_NoOp)
    return Visit(expr->getSubExpr());
  if (expr->getCastKind() == clang::CastKind::CK_ConstructorConversion)
    return Visit(expr->getSubExpr());
  if (expr->getCastKind() == clang::CastKind::CK_IntegralCast) {
    auto scalar = Visit(expr->getSubExpr()).getValue(builder);
    assert(scalar);
    auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    if (scalar.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return ValueCategory(
          builder.create<mlir::LLVM::PtrToIntOp>(loc, postTy, scalar),
          /*isReference*/ false);
    }
    if (!scalar.getType().isa<mlir::IntegerType>()) {
      expr->dump();
      llvm::errs() << " scalar: " << scalar << "\n";
    }
    auto prevTy = scalar.getType().cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*expr->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }

    if (prevTy == postTy)
      return ValueCategory(scalar, /*isReference*/ false);
    if (prevTy.getWidth() < postTy.getWidth()) {
      if (signedType) {
        return ValueCategory(builder.create<ExtSIOp>(loc, scalar, postTy),
                             /*isReference*/ false);
      } else {
        return ValueCategory(builder.create<ExtUIOp>(loc, scalar, postTy),
                             /*isReference*/ false);
      }
    } else {
      return ValueCategory(
          builder.create<mlir::arith::TruncIOp>(loc, scalar, postTy),
          /*isReference*/ false);
    }
  }
  expr->dump();
  assert(0 && "unhandled functional cast type");
}

ValueCategory
MLIRScanner::VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr) {
  return Visit(expr->getSubExpr());
}

ValueCategory MLIRScanner::VisitLambdaExpr(clang::LambdaExpr *expr) {

  // llvm::DenseMap<const VarDecl *, FieldDecl *> InnerCaptures;
  // FieldDecl *ThisCapture = nullptr;

  // expr->getLambdaClass()->getCaptureFields(InnerCaptures, ThisCapture);

  bool LLVMABI = false;
  mlir::Type t = Glob.getMLIRType(expr->getCallOperator()->getThisType());

  bool isArray =
      false; // isa<clang::ArrayType>(expr->getCallOperator()->getThisType());
  Glob.getMLIRType(expr->getCallOperator()->getThisObjectType(), &isArray);

  if (auto PT = t.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    LLVMABI = true;
    t = PT.getElementType();
  }
  if (auto mt = t.dyn_cast<MemRefType>()) {
    auto shape = std::vector<int64_t>(mt.getShape());
    if (!isArray)
      shape[0] = 1;
    t = mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
  }
  auto op = createAllocOp(t, nullptr, /*memtype*/ 0, isArray, LLVMABI);

  for (auto tup : llvm::zip(expr->getLambdaClass()->captures(),
                            expr->getLambdaClass()->fields())) {
    auto C = std::get<0>(tup);
    auto field = std::get<1>(tup);
    if (C.capturesThis())
      continue;
    else if (!C.capturesVariable())
      continue;

    auto CK = C.getCaptureKind();
    auto var = C.getCapturedVar();

    ValueCategory result;

    if (params.find(var) != params.end()) {
      result = params[var];
    } else {
      if (auto VD = dyn_cast<VarDecl>(var)) {
        if (Captures.find(VD) != Captures.end()) {
          FieldDecl *field = Captures[VD];
          result = CommonFieldLookup(
              cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(),
              field, ThisVal.val, /*isLValue*/ false);
          assert(CaptureKinds.find(VD) != CaptureKinds.end());
          if (CaptureKinds[VD] == LambdaCaptureKind::LCK_ByRef)
            result = result.dereference(builder);
          goto endp;
        }
      }
      EmittingFunctionDecl->dump();
      expr->dump();
      function.dump();
      llvm::errs() << "<pairs>\n";
      for (auto p : params)
        p.first->dump();
      llvm::errs() << "</pairs>";
      var->dump();
    }
  endp:

    bool isArray = false;
    Glob.getMLIRType(field->getType(), &isArray);

    if (CK == LambdaCaptureKind::LCK_ByCopy)
      CommonFieldLookup(expr->getCallOperator()->getThisObjectType(), field, op,
                        /*isLValue*/ false)
          .store(builder, result, isArray);
    else {
      assert(CK == LambdaCaptureKind::LCK_ByRef);
      assert(result.isReference);

      auto val = result.val;

      if (auto mt = val.getType().dyn_cast<MemRefType>()) {
        auto shape = std::vector<int64_t>(mt.getShape());
        shape[0] = -1;
        val = builder.create<memref::CastOp>(
            loc,
            MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace()),
            val);
      }

      CommonFieldLookup(expr->getCallOperator()->getThisObjectType(), field, op,
                        /*isLValue*/ false)
          .store(builder, val);
    }
  }
  return ValueCategory(op, /*isReference*/ true);
}

// TODO actually deallocate
ValueCategory MLIRScanner::VisitMaterializeTemporaryExpr(
    clang::MaterializeTemporaryExpr *expr) {
  auto v = Visit(expr->getSubExpr());
  if (!v.val) {
    expr->dump();
  }
  assert(v.val);

  bool isArray = false;
  bool LLVMABI = false;
  if (Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                           expr->getSubExpr()->getType()))
          .isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else {
    Glob.getMLIRType(expr->getSubExpr()->getType(), &isArray);
  }
  if (isArray)
    return v;

  llvm::errs() << "cleanup of materialized not handled";
  auto op = createAllocOp(getMLIRType(expr->getSubExpr()->getType()), nullptr,
                          0, /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);

  ValueCategory(op, /*isRefererence*/ true).store(builder, v, isArray);
  return ValueCategory(op, /*isRefererence*/ true);
}

ValueCategory MLIRScanner::VisitCXXNewExpr(clang::CXXNewExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());

  mlir::Value count;

  if (expr->isArray()) {
    count = Visit(*expr->raw_arg_begin()).getValue(builder);
    count = builder.create<IndexCastOp>(
        loc, count, mlir::IndexType::get(builder.getContext()));
  } else {
    count = getConstantIndex(1);
  }
  assert(count);

  auto ty = getMLIRType(expr->getType());

  mlir::Value alloc;
  if (auto mt = ty.dyn_cast<mlir::MemRefType>()) {
    auto shape = std::vector<int64_t>(mt.getShape());
    mlir::Value args[1] = {count};
    alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
  } else {
    auto i64 = mlir::IntegerType::get(count.getContext(), 64);
    auto typeSize = builder.create<ConstantIntOp>(
        loc, getTypeSize(expr->getAllocatedType()), i64);
    count = builder.create<IndexCastOp>(loc, count, i64);
    mlir::Value args[1] = {builder.create<arith::MulIOp>(loc, count, typeSize)};
    alloc = builder.create<mlir::LLVM::BitcastOp>(
        loc, ty,
        builder
            .create<mlir::LLVM::CallOp>(loc, Glob.GetOrCreateMallocFunction(),
                                        args)
            ->getResult(0));
  }
  assert(alloc);

  if (expr->getConstructExpr()) {
    VisitConstructCommon(
        const_cast<CXXConstructExpr *>(expr->getConstructExpr()),
        /*name*/ nullptr, /*memtype*/ 0, alloc, count);
  }
  return ValueCategory(alloc, /*isRefererence*/ false);
}

mlir::Value add(MLIRScanner &sc, mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value lhs, mlir::Value rhs) {
  assert(lhs);
  assert(rhs);
  if (auto op = lhs.getDefiningOp<ConstantIntOp>()) {
    if (op.value() == 0) {
      return rhs;
    }
  }

  if (auto op = lhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.value() == 0) {
      return rhs;
    }
  }

  if (auto op = rhs.getDefiningOp<ConstantIntOp>()) {
    if (op.value() == 0) {
      return lhs;
    }
  }

  if (auto op = rhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.value() == 0) {
      return lhs;
    }
  }
  return builder.create<AddIOp>(loc, lhs, rhs);
}

mlir::Value MLIRScanner::castToIndex(mlir::Location loc, mlir::Value val) {
  assert(val && "Expect non-null value");

  if (auto op = val.getDefiningOp<ConstantIntOp>())
    return getConstantIndex(op.value());

  return builder.create<IndexCastOp>(loc, val,
                                     mlir::IndexType::get(val.getContext()));
}

ValueCategory
MLIRScanner::VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());

  bool isArray = false;
  mlir::Type melem = Glob.getMLIRType(expr->getType(), &isArray);
  assert(!isArray);

  if (melem.isa<mlir::IntegerType>())
    return ValueCategory(builder.create<ConstantIntOp>(loc, 0, melem), false);
  else if (auto MT = melem.dyn_cast<mlir::MemRefType>())
    return ValueCategory(
        builder.create<polygeist::Pointer2MemrefOp>(
            loc, MT,
            builder.create<mlir::LLVM::NullOp>(
                loc, LLVM::LLVMPointerType::get(builder.getI8Type()))),
        false);
  else if (auto PT = melem.dyn_cast<mlir::LLVM::LLVMPointerType>())
    return ValueCategory(builder.create<mlir::LLVM::NullOp>(loc, PT), false);
  else {
    if (!melem.isa<FloatType>())
      expr->dump();
    auto ft = melem.cast<FloatType>();
    return ValueCategory(builder.create<ConstantFloatOp>(
                             loc, APFloat(ft.getFloatSemantics(), "0"), ft),
                         false);
  }
}

ValueCategory MLIRScanner::VisitCXXPseudoDestructorExpr(
    clang::CXXPseudoDestructorExpr *expr) {
  Visit(expr->getBase());
  llvm::errs() << "not running pseudo destructor\n";
  return nullptr;
}

ValueCategory
MLIRScanner::VisitCXXConstructExpr(clang::CXXConstructExpr *cons) {
  return VisitConstructCommon(cons, /*name*/ nullptr, /*space*/ 0);
}

ValueCategory MLIRScanner::VisitConstructCommon(clang::CXXConstructExpr *cons,
                                                VarDecl *name, unsigned memtype,
                                                mlir::Value op,
                                                mlir::Value count) {
  auto loc = getMLIRLocation(cons->getExprLoc());

  bool isArray = false;
  mlir::Type subType = Glob.getMLIRType(cons->getType(), &isArray);

  bool LLVMABI = false;
  auto ptrty = Glob.getMLIRType(
      Glob.CGM.getContext().getLValueReferenceType(cons->getType()));
  if (ptrty.isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else if (isArray) {
    subType = ptrty;
    isArray = true;
  }
  if (op == nullptr)
    op = createAllocOp(subType, name, memtype, isArray, LLVMABI);

  auto decl = cons->getConstructor();
  if (cons->requiresZeroInitialization()) {
    mlir::Value val = op;
    size_t size;
    if (val.getType().isa<MemRefType>()) {
      val = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(builder.getI8Type()), val);
    } else {
      val = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              builder.getI8Type(),
              val.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          val);
      size = Glob.CGM.getContext()
                 .getTypeSizeInChars(cons->getType())
                 .getQuantity();
    }

    auto i8_0 = builder.create<ConstantIntOp>(loc, 0, 8);
    auto sizev = builder.create<ConstantIntOp>(loc, size, 32);

    auto falsev = builder.create<ConstantIntOp>(loc, false, 1);
    builder.create<LLVM::MemsetOp>(loc, val, i8_0, sizev, falsev);
  }

  if (decl->isTrivial() && decl->isDefaultConstructor())
    return ValueCategory(op, /*isReference*/ true);

  mlir::Block::iterator oldpoint;
  mlir::Block *oldblock;
  ValueCategory endobj(op, /*isReference*/ true);

  ValueCategory obj(op, /*isReference*/ true);
  QualType innerType = cons->getType();
  if (auto arrayType = Glob.CGM.getContext().getAsArrayType(cons->getType())) {
    innerType = arrayType->getElementType();
    mlir::Value size;
    if (count)
      size = count;
    else {
      auto CAT = cast<clang::ConstantArrayType>(arrayType);
      size = getConstantIndex(CAT->getSize().getLimitedValue());
    }
    auto forOp = builder.create<scf::ForOp>(loc, getConstantIndex(0), size,
                                            getConstantIndex(1));
    oldpoint = builder.getInsertionPoint();
    oldblock = builder.getInsertionBlock();

    builder.setInsertionPointToStart(&forOp.getLoopBody().front());
    assert(obj.isReference);
    obj = CommonArrayToPointer(obj);
    obj = CommonArrayLookup(obj, forOp.getInductionVar(),
                            /*isImplicitRef*/ false, /*removeIndex*/ false);
    assert(obj.isReference);
  }

  auto tocall = Glob.GetOrCreateMLIRFunction(cons->getConstructor());

  SmallVector<std::pair<ValueCategory, clang::Expr *>> args;
  args.emplace_back(make_pair(obj, (clang::Expr *)nullptr));
  for (auto a : cons->arguments())
    args.push_back(make_pair(Visit(a), a));
  CallHelper(tocall, innerType, args,
             /*retType*/ Glob.CGM.getContext().VoidTy, false, cons);

  if (Glob.CGM.getContext().getAsArrayType(cons->getType())) {
    builder.setInsertionPoint(oldblock, oldpoint);
  }
  return endobj;
}

ValueCategory MLIRScanner::CommonArrayToPointer(ValueCategory scalar) {
  assert(scalar.val);
  assert(scalar.isReference);
  if (auto PT = scalar.val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    if (PT.getElementType().isa<mlir::LLVM::LLVMPointerType>())
      return ValueCategory(scalar.val, /*isRef*/ false);
    mlir::Value vec[3] = {scalar.val, builder.create<ConstantIntOp>(loc, 0, 32),
                          builder.create<ConstantIntOp>(loc, 0, 32)};
    if (!PT.getElementType().isa<mlir::LLVM::LLVMArrayType>()) {
      EmittingFunctionDecl->dump();
      function.dump();
      llvm::errs() << " sval: " << scalar.val << "\n";
      llvm::errs() << PT << "\n";
    }
    auto ET =
        PT.getElementType().cast<mlir::LLVM::LLVMArrayType>().getElementType();
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(
            loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
            vec),
        /*isReference*/ false);
  }

  auto mt = scalar.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  // if (shape.size() > 1) {
  //  shape.erase(shape.begin());
  //} else {
  shape[0] = -1;
  //}
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());

  auto post = builder.create<memref::CastOp>(loc, mt0, scalar.val);
  return ValueCategory(post, /*isReference*/ false);
}

ValueCategory MLIRScanner::CommonArrayLookup(ValueCategory array,
                                             mlir::Value idx,
                                             bool isImplicitRefResult,
                                             bool removeIndex) {
  mlir::Value val = array.getValue(builder);
  assert(val);

  if (val.getType().isa<LLVM::LLVMPointerType>()) {

    std::vector<mlir::Value> vals = {val};
    idx = builder.create<IndexCastOp>(loc, idx, builder.getIntegerType(64));
    vals.push_back(idx);
    // TODO sub
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(loc, val.getType(), vals),
        /*isReference*/ true);
  }
  if (!val.getType().isa<MemRefType>()) {
    EmittingFunctionDecl->dump();
    builder.getInsertionBlock()->dump();
    function.dump();
    llvm::errs() << "value: " << val << "\n";
  }

  ValueCategory dref;
  {
    auto mt = val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    shape[0] = -1;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    auto post = builder.create<polygeist::SubIndexOp>(loc, mt0, val, idx);
    // TODO sub
    dref = ValueCategory(post, /*isReference*/ true);
  }
  assert(dref.isReference);
  if (!removeIndex)
    return dref;

  auto mt = dref.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() == 1 || (shape.size() == 2 && isImplicitRefResult)) {
    shape[0] = -1;
  } else {
    shape.erase(shape.begin());
  }
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());
  auto post = builder.create<polygeist::SubIndexOp>(loc, mt0, dref.val,
                                                    getConstantIndex(0));
  return ValueCategory(post, /*isReference*/ true);
}

ValueCategory
MLIRScanner::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr) {
  auto moo = Visit(expr->getLHS());

  auto rhs = Visit(expr->getRHS()).getValue(builder);
  // Check the RHS has been successfully emitted
  assert(rhs);
  auto idx = castToIndex(getMLIRLocation(expr->getRBracketLoc()), rhs);
  if (isa<clang::VectorType>(
          expr->getLHS()->getType()->getUnqualifiedDesugaredType())) {
    assert(moo.isReference);
    moo.isReference = false;
    auto mt = moo.val.getType().cast<MemRefType>();

    auto shape = std::vector<int64_t>(mt.getShape());
    shape.erase(shape.begin());
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    moo.val = builder.create<polygeist::SubIndexOp>(loc, mt0, moo.val,
                                                    getConstantIndex(0));
  }
  bool isArray = false;
  if (!Glob.CGM.getContext().getAsArrayType(expr->getType()))
    Glob.getMLIRType(expr->getType(), &isArray);
  return CommonArrayLookup(moo, idx, isArray);
}

bool isRecursiveStruct(llvm::Type *T, llvm::Type *Meta,
                       SmallPtrSetImpl<llvm::Type *> &seen) {
  if (seen.count(T))
    return false;
  seen.insert(T);
  if (T->isVoidTy() || T->isFPOrFPVectorTy() || T->isIntOrIntVectorTy())
    return false;
  if (T == Meta) {
    return true;
  }
  for (auto ST : T->subtypes()) {
    if (isRecursiveStruct(ST, Meta, seen)) {
      return true;
    }
  }
  return false;
}

llvm::Type *anonymize(llvm::Type *T) {
  if (auto PT = dyn_cast<llvm::PointerType>(T))
    return llvm::PointerType::get(anonymize(PT->getElementType()),
                                  PT->getAddressSpace());
  if (auto AT = dyn_cast<llvm::ArrayType>(T))
    return llvm::ArrayType::get(anonymize(AT->getElementType()),
                                AT->getNumElements());
  if (auto FT = dyn_cast<llvm::FunctionType>(T)) {
    SmallVector<llvm::Type *, 4> V;
    for (auto t : FT->params())
      V.push_back(anonymize(t));
    return llvm::FunctionType::get(anonymize(FT->getReturnType()), V,
                                   FT->isVarArg());
  }
  if (auto ST = dyn_cast<llvm::StructType>(T)) {
    if (ST->isLiteral())
      return ST;
    SmallVector<llvm::Type *, 4> V;

    for (auto t : ST->elements()) {
      SmallPtrSet<llvm::Type *, 4> Seen;
      if (isRecursiveStruct(t, ST, Seen))
        V.push_back(t);
      else
        V.push_back(anonymize(t));
    }
    return llvm::StructType::get(ST->getContext(), V, ST->isPacked());
  }
  return T;
}

const clang::FunctionDecl *MLIRScanner::EmitCallee(const Expr *E) {
  E = E->IgnoreParens();
  // Look through function-to-pointer decay.
  if (auto ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CK_FunctionToPointerDecay ||
        ICE->getCastKind() == CK_BuiltinFnToFnPtr) {
      return EmitCallee(ICE->getSubExpr());
    }

    // Resolve direct calls.
  } else if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    if (auto FD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
      return FD;
    }

  } else if (auto ME = dyn_cast<MemberExpr>(E)) {
    if (auto FD = dyn_cast<FunctionDecl>(ME->getMemberDecl())) {
      // TODO EmitIgnoredExpr(ME->getBase());
      return FD;
    }

    // Look through template substitutions.
  } else if (auto NTTP = dyn_cast<SubstNonTypeTemplateParmExpr>(E)) {
    return EmitCallee(NTTP->getReplacement());
  }

  return nullptr;
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitBuiltinOps(clang::CallExpr *expr) {
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee())) {
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__log2f") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::Log2Op>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "log") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::LogOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "ceil")) {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return make_pair(
            ValueCategory(builder.create<math::CeilOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "sqrtf" ||
           sr->getDecl()->getName() == "sqrt")) {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::SqrtOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "expf" ||
           sr->getDecl()->getName() == "exp")) {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::ExpOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "sin") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::SinOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }

      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "cos") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::CosOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
    }
  }

  return make_pair(ValueCategory(), false);
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitGPUCallExpr(clang::CallExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee())) {
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__syncthreads") {
        builder.create<mlir::NVVM::Barrier0Op>(loc);
        return make_pair(ValueCategory(), true);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "cudaFuncSetCacheConfig") {
        llvm::errs() << " Not emitting GPU option: cudaFuncSetCacheConfig\n";
        return make_pair(ValueCategory(), true);
      }
      // TODO move free out.
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "free" ||
           sr->getDecl()->getName() == "cudaFree" ||
           sr->getDecl()->getName() == "cudaFreeHost")) {

        auto sub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(sub))
          sub = BC->getSubExpr();
        mlir::Value arg = Visit(sub).getValue(builder);

        if (arg.getType().isa<mlir::LLVM::LLVMPointerType>()) {
          auto callee = EmitCallee(expr->getCallee());
          auto strcmpF = Glob.GetOrCreateLLVMFunction(callee);
          mlir::Value args[] = {builder.create<LLVM::BitcastOp>(
              loc, LLVM::LLVMPointerType::get(builder.getIntegerType(8)), arg)};
          builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args);
        } else {
          builder.create<mlir::memref::DeallocOp>(loc, arg);
        }
        if (sr->getDecl()->getName() == "cudaFree" ||
            sr->getDecl()->getName() == "cudaFreeHost") {
          auto ty = getMLIRType(expr->getType());
          auto op = builder.create<ConstantIntOp>(loc, 0, ty);
          return make_pair(ValueCategory(op, /*isReference*/ false), true);
        }
        // TODO remove me when the free is removed.
        return make_pair(ValueCategory(), true);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "cudaMalloc" ||
           sr->getDecl()->getName() == "cudaMallocHost" ||
           sr->getDecl()->getName() == "cudaMallocPitch")) {
        auto sub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(sub))
          sub = BC->getSubExpr();
        {
          auto dst = Visit(sub).getValue(builder);
          if (auto omt = dst.getType().dyn_cast<MemRefType>()) {
            if (auto mt = omt.getElementType().dyn_cast<MemRefType>()) {
              auto shape = std::vector<int64_t>(mt.getShape());

              auto elemSize = getTypeSize(
                  cast<clang::PointerType>(
                      cast<clang::PointerType>(
                          sub->getType()->getUnqualifiedDesugaredType())
                          ->getPointeeType())
                      ->getPointeeType());
              mlir::Value allocSize;
              if (sr->getDecl()->getName() == "cudaMallocPitch") {
                mlir::Value width = Visit(expr->getArg(2)).getValue(builder);
                mlir::Value height = Visit(expr->getArg(3)).getValue(builder);
                // Not changing pitch from provided width here
                // TODO can consider addition alignment considerations
                Visit(expr->getArg(1))
                    .dereference(builder)
                    .store(builder, width);
                allocSize = builder.create<MulIOp>(loc, width, height);
              } else
                allocSize = Visit(expr->getArg(1)).getValue(builder);
              auto idxType = mlir::IndexType::get(builder.getContext());
              mlir::Value args[1] = {builder.create<IndexCastOp>(
                  loc,
                  builder.create<DivUIOp>(
                      loc, allocSize,
                      builder.create<ConstantIntOp>(loc, elemSize,
                                                    allocSize.getType())),
                  idxType)};
              auto alloc = builder.create<mlir::memref::AllocOp>(
                  loc,
                  (sr->getDecl()->getName() != "cudaMallocHost" && !CudaLower)
                      ? mlir::MemRefType::get(
                            shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(),
                            wrapIntegerMemorySpace(1, mt.getContext()))
                      : mt,
                  args);
              ValueCategory(dst, /*isReference*/ true)
                  .store(builder,
                         builder.create<mlir::memref::CastOp>(loc, alloc, mt));
              auto retTy = getMLIRType(expr->getType());
              return make_pair(
                  ValueCategory(builder.create<ConstantIntOp>(loc, 0, retTy),
                                /*isReference*/ false),
                  true);
            }
          }
        }
      }
    }

    auto createBlockIdOp = [&](string str, mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc,
          builder.create<mlir::gpu::BlockIdOp>(
              loc, mlir::IndexType::get(builder.getContext()), str),
          mlirType);
    };

    auto createBlockDimOp = [&](string str,
                                mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc,
          builder.create<mlir::gpu::BlockDimOp>(
              loc, mlir::IndexType::get(builder.getContext()), str),
          mlirType);
    };

    auto createThreadIdOp = [&](string str,
                                mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc,
          builder.create<mlir::gpu::ThreadIdOp>(
              loc, mlir::IndexType::get(builder.getContext()), str),
          mlirType);
    };

    auto createGridDimOp = [&](string str, mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc,
          builder.create<mlir::gpu::GridDimOp>(
              loc, mlir::IndexType::get(builder.getContext()), str),
          mlirType);
    };

    if (auto ME = dyn_cast<MemberExpr>(ic->getSubExpr())) {
      auto memberName = ME->getMemberDecl()->getName();

      if (auto sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
        if (auto sr = dyn_cast<DeclRefExpr>(sr2->getSourceExpr())) {
          if (sr->getDecl()->getName() == "blockIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(ValueCategory(createBlockIdOp("x", mlirType),
                                             /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(ValueCategory(createBlockIdOp("y", mlirType),
                                             /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(ValueCategory(createBlockIdOp("z", mlirType),
                                             /*isReference*/ false),
                               true);
            }
          }
          if (sr->getDecl()->getName() == "blockDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(ValueCategory(createBlockDimOp("x", mlirType),
                                             /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(ValueCategory(createBlockDimOp("y", mlirType),
                                             /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(ValueCategory(createBlockDimOp("z", mlirType),
                                             /*isReference*/ false),
                               true);
            }
          }
          if (sr->getDecl()->getName() == "threadIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(ValueCategory(createThreadIdOp("x", mlirType),
                                             /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(ValueCategory(createThreadIdOp("y", mlirType),
                                             /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(ValueCategory(createThreadIdOp("z", mlirType),
                                             /*isReference*/ false),
                               true);
            }
          }
          if (sr->getDecl()->getName() == "gridDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(ValueCategory(createGridDimOp("x", mlirType),
                                             /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(ValueCategory(createGridDimOp("x", mlirType),
                                             /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(ValueCategory(createGridDimOp("z", mlirType),
                                             /*isReference*/ false),
                               true);
            }
          }
        }
      }
    }
  }
  return make_pair(ValueCategory(), false);
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
          args.push_back(Visit(a).getValue(builder));
        }
        builder.create<gpu::ShuffleOp>(loc, );
        assert(0 && "__shfl_up_sync unhandled");
        return nullptr;
      }
    }
  */

  auto valEmitted = EmitGPUCallExpr(expr);
  if (valEmitted.second)
    return valEmitted.first;

  valEmitted = EmitBuiltinOps(expr);
  if (valEmitted.second)
    return valEmitted.first;

  if (auto oc = dyn_cast<CXXOperatorCallExpr>(expr)) {
    if (oc->getOperator() == clang::OO_EqualEqual) {
      if (auto lhs = dyn_cast<CXXTypeidExpr>(expr->getArg(0))) {
        if (auto rhs = dyn_cast<CXXTypeidExpr>(expr->getArg(1))) {
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
  if (auto oc = dyn_cast<CXXMemberCallExpr>(expr)) {
    if (auto lhs = dyn_cast<CXXTypeidExpr>(oc->getImplicitObjectArgument())) {
      expr->getCallee()->dump();
      if (auto ic = dyn_cast<MemberExpr>(expr->getCallee()))
        if (auto sr = dyn_cast<NamedDecl>(ic->getMemberDecl())) {
          if (sr->getIdentifier() && sr->getName() == "name") {
            QualType LT = lhs->isTypeOperand()
                              ? lhs->getTypeOperand(Glob.CGM.getContext())
                              : lhs->getExprOperand()->getType();
            llvm::Constant *LC = Glob.CGM.GetAddrOfRTTIDescriptor(LT);
            while (auto CE = dyn_cast<llvm::ConstantExpr>(LC))
              LC = CE->getOperand(0);
            std::string val = cast<llvm::GlobalVariable>(LC)->getName().str();
            return CommonArrayToPointer(ValueCategory(
                Glob.GetOrCreateGlobalLLVMString(loc, builder, val),
                /*isReference*/ true));
          }
        }
    }
  }

  if (auto ps = dyn_cast<CXXPseudoDestructorExpr>(expr->getCallee())) {
    return Visit(ps);
  }

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "atomicAdd") {
        std::vector<ValueCategory> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a));
        }
        auto a1 = args[1].getValue(builder);
        if (a1.getType().isa<mlir::IntegerType>())
          return ValueCategory(
              builder.create<memref::AtomicRMWOp>(
                  loc, a1.getType(), AtomicRMWKind::addi, a1,
                  args[0].getValue(builder),
                  std::vector<mlir::Value>({getConstantIndex(0)})),
              /*isReference*/ false);
        else
          return ValueCategory(
              builder.create<memref::AtomicRMWOp>(
                  loc, a1.getType(), AtomicRMWKind::addf, a1,
                  args[0].getValue(builder),
                  std::vector<mlir::Value>({getConstantIndex(0)})),
              /*isReference*/ false);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "atomicOr") {
        std::vector<ValueCategory> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a));
        }
        auto a1 = args[1].getValue(builder);
        return ValueCategory(
            builder.create<memref::AtomicRMWOp>(
                loc, a1.getType(), AtomicRMWKind::ori, a1,
                args[0].getValue(builder),
                std::vector<mlir::Value>({getConstantIndex(0)})),
            /*isReference*/ false);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "atomicAnd") {
        std::vector<ValueCategory> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a));
        }
        auto a1 = args[1].getValue(builder);
        return ValueCategory(
            builder.create<memref::AtomicRMWOp>(
                loc, a1.getType(), AtomicRMWKind::andi, a1,
                args[0].getValue(builder),
                std::vector<mlir::Value>({getConstantIndex(0)})),
            /*isReference*/ false);
      }
    }

  auto getLLVM = [&](Expr *E) -> mlir::Value {
    auto sub = Visit(E);
    if (!sub.val) {
      expr->dump();
      E->dump();
    }
    assert(sub.val);

    bool isReference = E->isLValue() || E->isXValue();
    if (isReference) {
      assert(sub.isReference);
      mlir::Value val = sub.val;
      if (auto mt = val.getType().dyn_cast<MemRefType>()) {
        val = builder.create<polygeist::Memref2PointerOp>(
            loc, LLVM::LLVMPointerType::get(mt.getElementType()), val);
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

      OpBuilder abuilder(builder.getContext());
      abuilder.setInsertionPointToStart(allocationScope);
      auto one = abuilder.create<ConstantIntOp>(loc, 1, 64);
      auto alloc = abuilder.create<mlir::LLVM::AllocaOp>(
          loc,
          LLVM::LLVMPointerType::get(Glob.typeTranslator.translateType(
                                         anonymize(getLLVMType(E->getType()))),
                                     0),
          one, 0);
      ValueCategory(alloc, /*isRef*/ true)
          .store(builder, sub, /*isArray*/ isArray);
      sub = ValueCategory(alloc, /*isRef*/ true);
    }
    auto val = sub.getValue(builder);
    if (auto mt = val.getType().dyn_cast<MemRefType>()) {
      auto nt = Glob.typeTranslator
                    .translateType(anonymize(getLLVMType(E->getType())))
                    .cast<LLVM::LLVMPointerType>();
      val = builder.create<polygeist::Memref2PointerOp>(loc, nt, val);
    }
    return val;
  };

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
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
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
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
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_addressof") {
        auto V = Visit(expr->getArg(0));
        assert(V.isReference);
        mlir::Value val = V.val;
        auto T = getMLIRType(expr->getType());
        if (T == val.getType())
          return ValueCategory(val, /*isRef*/ false);
        if (T.isa<LLVM::LLVMPointerType>()) {
          if (val.getType().isa<MemRefType>())
            val = builder.create<polygeist::Memref2PointerOp>(loc, T, val);
          else if (T != val.getType())
            val = builder.create<LLVM::BitcastOp>(loc, T, val);
          return ValueCategory(val, /*isRef*/ false);
        } else {
          assert(T.isa<MemRefType>());
          if (val.getType().isa<MemRefType>())
            val = builder.create<polygeist::Memref2PointerOp>(
                loc, LLVM::LLVMPointerType::get(builder.getI8Type()), val);
          if (val.getType().isa<LLVM::LLVMPointerType>())
            val = builder.create<polygeist::Pointer2MemrefOp>(loc, T, val);
          return ValueCategory(val, /*isRef*/ false);
        }
        expr->dump();
        llvm::errs() << " val: " << val << " T: " << T << "\n";
        assert(0 && "unhandled builtin addressof");
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_fabsf" ||
           sr->getDecl()->getName() == "__nv_fabs" ||
           sr->getDecl()->getName() == "__nv_abs" ||
           sr->getDecl()->getName() == "fabs" ||
           sr->getDecl()->getName() == "fabsf" ||
           sr->getDecl()->getName() == "__builtin_fabs" ||
           sr->getDecl()->getName() == "__builtin_fabsf")) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value Fabs;
        if (V.getType().isa<mlir::FloatType>())
          Fabs = builder.create<math::AbsOp>(loc, V);
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
          Glob.functions[name] = mlir::FuncOp(
              mlir::FuncOp::create(builder.getUnknownLoc(), name, funcType));
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

        auto name = "strlen";

        if (Glob.functions.find(name) == Glob.functions.end()) {
          std::vector<mlir::Type> types{V0.getType()};

          auto RT = getMLIRType(expr->getType());
          std::vector<mlir::Type> rettypes{RT};
          mlir::OpBuilder mbuilder(Glob.module->getContext());
          auto funcType = mbuilder.getFunctionType(types, rettypes);
          Glob.functions[name] = mlir::FuncOp(
              mlir::FuncOp::create(builder.getUnknownLoc(), name, funcType));
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
           sr->getDecl()->getName() == "__builtin_isinf")) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        mlir::Value V = getLLVM(expr->getArg(0));
        auto Ty = V.getType().cast<mlir::FloatType>();
        mlir::Value Fabs = builder.create<math::AbsOp>(loc, V);
        auto Infinity = builder.create<ConstantFloatOp>(
            loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
        auto Pred = (sr->getDecl()->getName() == "__builtin_isinf")
                        ? CmpFPredicate::OEQ
                        : CmpFPredicate::ONE;
        mlir::Value FCmp = builder.create<CmpFOp>(loc, Pred, Fabs, Infinity);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, FCmp, postTy);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_isnan")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value Eq = builder.create<CmpFOp>(loc, CmpFPredicate::UNO, V, V);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, Eq, postTy);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_isnormal")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        auto Ty = V.getType().cast<mlir::FloatType>();
        mlir::Value Eq = builder.create<CmpFOp>(loc, CmpFPredicate::OEQ, V, V);

        mlir::Value Abs = builder.create<math::AbsOp>(loc, V);
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
        mlir::Value res = builder.create<ExtUIOp>(loc, V, postTy);
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
        mlir::Value res = builder.create<ExtUIOp>(loc, V, postTy);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isgreater") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, V, postTy);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isgreaterequal") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OGE, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, V, postTy);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isless") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, V, postTy);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_islessequal") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OLE, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, V, postTy);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_islessgreater") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, V, postTy);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isunordered") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::UNO, V, V2);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, V, postTy);
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
        mlir::FuncOp function = mlir::FuncOp(
            mlir::FuncOp::create(builder.getUnknownLoc(), name, funcType));
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

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if ((sr->getDecl()->getIdentifier() &&
           (sr->getDecl()->getName() == "fscanf" ||
            sr->getDecl()->getName() == "scanf" ||
            sr->getDecl()->getName() == "__isoc99_sscanf" ||
            sr->getDecl()->getName() == "sscanf")) ||
          (isa<CXXOperatorCallExpr>(expr) &&
           cast<CXXOperatorCallExpr>(expr)->getOperator() ==
               OO_GreaterGreater)) {
        auto tocall = EmitCallee(expr->getCallee());
        auto strcmpF = Glob.GetOrCreateLLVMFunction(tocall);

        std::vector<mlir::Value> args;
        std::vector<std::pair<mlir::Value, mlir::Value>> ops;
        std::map<const void *, size_t> counts;
        for (auto a : expr->arguments()) {
          auto v = getLLVM(a);
          if (auto toptr = v.getDefiningOp<polygeist::Memref2PointerOp>()) {
            auto T = toptr.getType().cast<LLVM::LLVMPointerType>();
            auto idx = counts[T.getAsOpaquePointer()]++;
            auto aop = allocateBuffer(idx, T);
            args.push_back(aop.getResult());
            ops.emplace_back(aop.getResult(), toptr.source());
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
        return ValueCategory(called.getResult(0), /*isReference*/ false);
      }
    }

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "memmove" ||
           sr->getDecl()->getName() == "__builtin_memmove")) {
        std::vector<mlir::Value> args = {
            getLLVM(expr->getArg(0)), getLLVM(expr->getArg(1)),
            getLLVM(expr->getArg(2)), /*isVolatile*/
            builder.create<ConstantIntOp>(loc, false, 1)};
        builder.create<LLVM::MemmoveOp>(loc, args[0], args[1], args[2],
                                        args[3]);
        return ValueCategory(args[0], /*isReference*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "memset" ||
           sr->getDecl()->getName() == "__builtin_memset")) {
        std::vector<mlir::Value> args = {
            getLLVM(expr->getArg(0)), getLLVM(expr->getArg(1)),
            getLLVM(expr->getArg(2)), /*isVolatile*/
            builder.create<ConstantIntOp>(loc, false, 1)};

        args[1] = builder.create<TruncIOp>(loc, builder.getI8Type(), args[1]);
        builder.create<LLVM::MemsetOp>(loc, args[0], args[1], args[2], args[3]);
        return ValueCategory(args[0], /*isReference*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "memcpy" ||
           sr->getDecl()->getName() == "__builtin_memcpy")) {
        std::vector<mlir::Value> args = {
            getLLVM(expr->getArg(0)), getLLVM(expr->getArg(1)),
            getLLVM(expr->getArg(2)), /*isVolatile*/
            builder.create<ConstantIntOp>(loc, false, 1)};
        builder.create<LLVM::MemcpyOp>(loc, args[0], args[1], args[2], args[3]);
        return ValueCategory(args[0], /*isReference*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "cudaMemcpy" ||
           sr->getDecl()->getName() == "cudaMemcpyAsync" ||
           sr->getDecl()->getName() == "cudaMemcpyToSymbol" ||
           sr->getDecl()->getName() == "memcpy" ||
           sr->getDecl()->getName() == "__builtin_memcpy")) {
        auto dstSub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(dstSub))
          dstSub = BC->getSubExpr();
        auto srcSub = expr->getArg(1);
        while (auto BC = dyn_cast<clang::CastExpr>(srcSub))
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
              dst = vdst.getValue(builder);
            } else {
              assert(vdst.isReference);
              dst = vdst.val;
            }
            // if (dst.getType().isa<MemRefType>())
            {
              mlir::Value src;
              ValueCategory vsrc = Visit(srcSub);
              if (isa<clang::PointerType>(srcst)) {
                src = vsrc.getValue(builder);
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
                  loc, Visit(expr->getArg(2)).getValue(builder),
                  mlir::IndexType::get(builder.getContext()));
              size = builder.create<DivUIOp>(
                  loc, size, builder.create<ConstantIndexOp>(loc, elemSize));

              if (sr->getDecl()->getName() == "cudaMemcpyToSymbol") {
                mlir::Value offset = Visit(expr->getArg(3)).getValue(builder);
                offset = builder.create<IndexCastOp>(
                    loc, offset, mlir::IndexType::get(builder.getContext()));
                offset = builder.create<DivUIOp>(
                    loc, offset,
                    builder.create<ConstantIndexOp>(loc, elemSize));
                // assert(!dstArray);
                if (auto mt = dst.getType().dyn_cast<MemRefType>()) {
                  auto shape = std::vector<int64_t>(mt.getShape());
                  shape[0] = -1;
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
                assert(shape.size() > 0 && shape.back() != -1);
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
                  assert(sshape.size() > 0 && sshape.back() != -1);
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
                  assert(sshape.size() > 0 && sshape.back() != -1);
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
          assert(0 && "unhandled cudaMemcpy");
          */
        }
#endif
      }
    }

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
              dst = vdst.getValue(builder);
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
                  loc, Visit(expr->getArg(2)).getValue(builder),
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

  auto callee = EmitCallee(expr->getCallee());

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
      "clock_gettime",
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
      "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
      "cudaEventRecord"};
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      StringRef name;
      if (auto CC = dyn_cast<CXXConstructorDecl>(sr->getDecl()))
        name =
            Glob.CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete));
      else if (auto CC = dyn_cast<CXXDestructorDecl>(sr->getDecl()))
        name =
            Glob.CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete));
      else if (sr->getDecl()->hasAttr<CUDAGlobalAttr>())
        name = Glob.CGM.getMangledName(GlobalDecl(
            cast<FunctionDecl>(sr->getDecl()), KernelReferenceKind::Kernel));
      else
        name = Glob.CGM.getMangledName(sr->getDecl());
      if (funcs.count(name.str()) || name.startswith("mkl_") ||
          name.startswith("MKL_") || name.startswith("cublas") ||
          name.startswith("cblas_")) {

        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(getLLVM(a));
        }
        mlir::Value called;

        if (callee) {
          auto strcmpF = Glob.GetOrCreateLLVMFunction(callee);
          called = builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args)
                       .getResult(0);
        } else {
          args.insert(args.begin(), getLLVM(expr->getCallee()));
          SmallVector<mlir::Type> RTs = {Glob.typeTranslator.translateType(
              anonymize(getLLVMType(expr->getType())))};
          if (RTs[0].isa<LLVM::LLVMVoidType>())
            RTs.clear();
          called =
              builder.create<mlir::LLVM::CallOp>(loc, RTs, args).getResult(0);
        }
        return ValueCategory(called, /*isReference*/ expr->isLValue() ||
                                         expr->isXValue());
      }
    }

  if (!callee || callee->isVariadic()) {
    bool isReference = expr->isLValue() || expr->isXValue();
    std::vector<mlir::Value> args;
    for (auto a : expr->arguments()) {
      args.push_back(getLLVM(a));
    }
    mlir::Value called;
    if (callee) {
      auto strcmpF = Glob.GetOrCreateLLVMFunction(callee);
      called =
          builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args).getResult(0);
    } else {
      args.insert(args.begin(), getLLVM(expr->getCallee()));
      auto CT = expr->getType();
      if (isReference)
        CT = Glob.CGM.getContext().getLValueReferenceType(CT);
      SmallVector<mlir::Type> RTs = {
          Glob.typeTranslator.translateType(anonymize(getLLVMType(CT)))};
      auto ft = args[0]
                    .getType()
                    .cast<LLVM::LLVMPointerType>()
                    .getElementType()
                    .cast<LLVM::LLVMFunctionType>();
      assert(RTs[0] == ft.getReturnType());
      if (RTs[0].isa<LLVM::LLVMVoidType>())
        RTs.clear();
      called = builder.create<mlir::LLVM::CallOp>(loc, RTs, args).getResult(0);
    }
    if (isReference) {
      if (!(called.getType().isa<LLVM::LLVMPointerType>() ||
            called.getType().isa<MemRefType>())) {
        expr->dump();
        expr->getType()->dump();
        llvm::errs() << " call: " << called << "\n";
      }
    }
    return ValueCategory(called, isReference);
  }

  auto tocall = EmitDirectCallee(callee);

  SmallVector<std::pair<ValueCategory, clang::Expr *>> args;
  QualType objType;

  if (auto CC = dyn_cast<CXXMemberCallExpr>(expr)) {
    ValueCategory obj = Visit(CC->getImplicitObjectArgument());
    objType = CC->getObjectType();
    if (!obj.val) {
      function.dump();
      llvm::errs() << " objval: " << obj.val << "\n";
      expr->dump();
      CC->getImplicitObjectArgument()->dump();
    }
    if (cast<MemberExpr>(CC->getCallee()->IgnoreParens())->isArrow()) {
      obj = obj.dereference(builder);
    }
    assert(obj.val);
    assert(obj.isReference);
    args.emplace_back(make_pair(obj, (clang::Expr *)nullptr));
  }
  for (auto a : expr->arguments())
    args.push_back(make_pair(Visit(a), a));
  return CallHelper(tocall, objType, args, expr->getType(),
                    expr->isLValue() || expr->isXValue(), expr);
}

ValueCategory MLIRScanner::CallHelper(
    mlir::FuncOp tocall, QualType objType,
    ArrayRef<std::pair<ValueCategory, clang::Expr *>> arguments,
    QualType retType, bool retReference, clang::Expr *expr) {
  SmallVector<mlir::Value, 4> args;
  auto fnType = tocall.getType();

  size_t i = 0;
  // map from declaration name to mlir::value
  std::map<std::string, mlir::Value> mapFuncOperands;

  for (auto pair : arguments) {

    ValueCategory arg = std::get<0>(pair);
    auto a = std::get<1>(pair);
    if (!arg.val) {
      expr->dump();
      a->dump();
    }
    assert(arg.val && "expect not null");

    if (auto ice = dyn_cast_or_null<ImplicitCastExpr>(a))
      if (auto dre = dyn_cast<DeclRefExpr>(ice->getSubExpr()))
        mapFuncOperands.insert(
            make_pair(dre->getDecl()->getName().str(), arg.val));

    if (i >= fnType.getInputs().size() || (i != 0 && a == nullptr)) {
      expr->dump();
      tocall.dump();
      fnType.dump();
      for (auto a : arguments) {
        std::get<1>(a)->dump();
      }
      assert(0 && "too many arguments in calls");
    }
    bool isReference =
        (i == 0 && a == nullptr) || a->isLValue() || a->isXValue();

    bool isArray = false;
    QualType aType = (i == 0 && a == nullptr) ? objType : a->getType();

    auto expectedType = Glob.getMLIRType(aType, &isArray);
    if (auto PT = arg.val.getType().dyn_cast<LLVM::LLVMPointerType>()) {
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
        if (pshape == -1)
          shape[0] = 1;

        OpBuilder abuilder(builder.getContext());
        abuilder.setInsertionPointToStart(allocationScope);
        auto alloc = abuilder.create<mlir::memref::AllocaOp>(
            loc, mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace()));
        ValueCategory(alloc, /*isRef*/ true)
            .store(builder, arg, /*isArray*/ isArray);
        shape[0] = pshape;
        val = builder.create<mlir::memref::CastOp>(
            loc, alloc,
            mlir::MemRefType::get(shape, mt.getElementType(),
                                  MemRefLayoutAttrInterface(),
                                  mt.getMemorySpace()));
      } else {
        val = arg.getValue(builder);
        if (val.getType().isa<LLVM::LLVMPointerType>() &&
            expectedType.isa<MemRefType>()) {
          val = builder.create<polygeist::Pointer2MemrefOp>(loc, expectedType,
                                                            val);
        }
        if (auto prevTy = val.getType().dyn_cast<mlir::IntegerType>()) {
          auto ipostTy = expectedType.cast<mlir::IntegerType>();
          if (prevTy != ipostTy)
            val = builder.create<arith::TruncIOp>(loc, val, ipostTy);
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

    return ValueCategory(mlirclang::replaceFuncByOperation(
                             tocall, LTInfo.SymbolTable[tocall.getName()],
                             builder, inputOperands, outputOperands)
                             ->getResult(0),
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
    if (pshape == -1)
      shape[0] = 1;

    OpBuilder abuilder(builder.getContext());
    abuilder.setInsertionPointToStart(allocationScope);
    alloc = abuilder.create<mlir::memref::AllocaOp>(
        loc, mlir::MemRefType::get(shape, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace()));
    shape[0] = pshape;
    alloc = builder.create<mlir::memref::CastOp>(
        loc, alloc,
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(),
                              mt.getMemorySpace()));
    args.push_back(alloc);
  }

  if (auto CU = dyn_cast<CUDAKernelCallExpr>(expr)) {
    auto l0 = Visit(CU->getConfig()->getArg(0));
    assert(l0.isReference);
    mlir::Value blocks[3];
    for (int i = 0; i < 3; i++) {
      std::vector<mlir::Value> idx = {getConstantIndex(0), getConstantIndex(i)};
      assert(l0.val.getType().cast<MemRefType>().getShape().size() == 2);
      blocks[i] = builder.create<IndexCastOp>(
          loc, builder.create<mlir::memref::LoadOp>(loc, l0.val, idx),
          mlir::IndexType::get(builder.getContext()));
    }

    auto t0 = Visit(CU->getConfig()->getArg(1));
    assert(t0.isReference);
    mlir::Value threads[3];
    for (int i = 0; i < 3; i++) {
      std::vector<mlir::Value> idx = {getConstantIndex(0), getConstantIndex(i)};
      assert(t0.val.getType().cast<MemRefType>().getShape().size() == 2);
      threads[i] = builder.create<IndexCastOp>(
          loc, builder.create<mlir::memref::LoadOp>(loc, t0.val, idx),
          mlir::IndexType::get(builder.getContext()));
    }
    auto op = builder.create<mlir::gpu::LaunchOp>(loc, blocks[0], blocks[1],
                                                  blocks[2], threads[0],
                                                  threads[1], threads[2]);
    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&op.getRegion().front());
    builder.create<mlir::CallOp>(loc, tocall, args);
    builder.create<gpu::TerminatorOp>(loc);
    builder.setInsertionPoint(oldblock, oldpoint);
    return nullptr;
  }

  // Try to rescue some mismatched types.
  castCallerArgs(tocall, args, builder);

  auto op = builder.create<mlir::CallOp>(loc, tocall, args);

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
  assert(0 && "no indirect");
}

mlir::Value MLIRScanner::getConstantIndex(int x) {
  if (constants.find(x) != constants.end()) {
    return constants[x];
  }
  mlir::OpBuilder subbuilder(builder.getContext());
  subbuilder.setInsertionPointToStart(entryBlock);
  return constants[x] = subbuilder.create<ConstantIndexOp>(loc, x);
}

ValueCategory MLIRScanner::VisitMSPropertyRefExpr(MSPropertyRefExpr *expr) {
  assert(0 && "unhandled ms propertyref");
  // TODO obviously fake
  return nullptr;
}

ValueCategory
MLIRScanner::VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr) {
  return Visit(expr->getResultExpr());
}

ValueCategory MLIRScanner::VisitUnaryOperator(clang::UnaryOperator *U) {
  auto loc = getMLIRLocation(U->getExprLoc());
  auto sub = Visit(U->getSubExpr());

  switch (U->getOpcode()) {
  case clang::UnaryOperator::Opcode::UO_Extension: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_LNot: {
    assert(sub.val);
    mlir::Value val = sub.getValue(builder);

    if (auto MT = val.getType().dyn_cast<mlir::MemRefType>()) {
      val = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(builder.getI8Type()), val);
    }

    if (auto LT = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::eq, val, nullptr_llvm);
      return ValueCategory(ne, /*isReference*/ false);
    }

    if (!val.getType().isa<mlir::IntegerType>()) {
      U->dump();
      val.dump();
    }
    auto ty = val.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      val = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, val,
          builder.create<ConstantIntOp>(loc, 0, ty));
    }
    auto c1 = builder.create<ConstantIntOp>(loc, 1, val.getType());
    mlir::Value res = builder.create<XOrIOp>(loc, val, c1);

    auto postTy = getMLIRType(U->getType()).cast<mlir::IntegerType>();
    if (postTy.getWidth() > 1)
      res = builder.create<ExtUIOp>(loc, res, postTy);
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Not: {
    assert(sub.val);
    mlir::Value val = sub.getValue(builder);

    if (!val.getType().isa<mlir::IntegerType>()) {
      U->dump();
      val.dump();
    }
    auto ty = val.getType().cast<mlir::IntegerType>();
    auto c1 = builder.create<ConstantIntOp>(
        loc, APInt::getAllOnesValue(ty.getWidth()).getSExtValue(), ty);
    return ValueCategory(builder.create<XOrIOp>(loc, val, c1),
                         /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Deref: {
    auto dref = sub.dereference(builder);
    return dref;
  }
  case clang::UnaryOperator::Opcode::UO_AddrOf: {
    assert(sub.isReference);
    if (sub.val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return ValueCategory(sub.val, /*isReference*/ false);
    }

    bool isArray = false;
    Glob.getMLIRType(U->getSubExpr()->getType(), &isArray);
    auto mt = sub.val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    mlir::Value res;
    shape[0] = -1;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    res = builder.create<memref::CastOp>(loc, sub.val, mt0);
    return ValueCategory(res,
                         /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Plus: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_Minus: {
    mlir::Value val = sub.getValue(builder);
    auto ty = val.getType();
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      if (auto CI = val.getDefiningOp<ConstantFloatOp>()) {
        auto api = CI.getValue().cast<FloatAttr>().getValue();
        return ValueCategory(builder.create<arith::ConstantOp>(
                                 loc, ty, mlir::FloatAttr::get(ty, -api)),
                             /*isReference*/ false);
      }
      return ValueCategory(builder.create<NegFOp>(loc, val),
                           /*isReference*/ false);
    } else {
      if (auto CI = val.getDefiningOp<ConstantIntOp>()) {
        auto api = CI.getValue().cast<IntegerAttr>().getValue();
        return ValueCategory(builder.create<arith::ConstantOp>(
                                 loc, ty, mlir::IntegerAttr::get(ty, -api)),
                             /*isReference*/ false);
      }
      return ValueCategory(
          builder.create<SubIOp>(loc,
                                 builder.create<ConstantIntOp>(
                                     loc, 0, ty.cast<mlir::IntegerType>()),
                                 val),
          /*isReference*/ false);
    }
  }
  case clang::UnaryOperator::Opcode::UO_PreInc:
  case clang::UnaryOperator::Opcode::UO_PostInc: {
    assert(sub.isReference);
    auto prev = sub.getValue(builder);
    auto ty = prev.getType();

    mlir::Value next;
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      if (prev.getType() != ty) {
        U->dump();
        llvm::errs() << " ty: " << ty << "prev: " << prev << "\n";
      }
      assert(prev.getType() == ty);
      next = builder.create<AddFOp>(
          loc, prev,
          builder.create<ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else if (auto mt = ty.dyn_cast<MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(1));
    } else if (auto pt = ty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      next = builder.create<LLVM::GEPOp>(
          loc, pt,
          std::vector<mlir::Value>(
              {prev, builder.create<ConstantIntOp>(loc, 1, ity)}));
    } else {
      if (!ty.isa<mlir::IntegerType>()) {
        llvm::errs() << ty << " - " << prev << "\n";
        U->dump();
      }
      if (prev.getType() != ty) {
        U->dump();
        llvm::errs() << " ty: " << ty << "prev: " << prev << "\n";
      }
      assert(prev.getType() == ty);
      next = builder.create<AddIOp>(
          loc, prev,
          builder.create<ConstantIntOp>(loc, 1, ty.cast<mlir::IntegerType>()));
    }
    sub.store(builder, next);

    if (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc)
      return sub;
    else
      return ValueCategory(prev, /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_PreDec:
  case clang::UnaryOperator::Opcode::UO_PostDec: {
    auto ty = getMLIRType(U->getType());
    assert(sub.isReference);
    auto prev = sub.getValue(builder);

    mlir::Value next;
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      next = builder.create<SubFOp>(
          loc, prev,
          builder.create<ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else if (auto pt = ty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      next = builder.create<LLVM::GEPOp>(
          loc, pt,
          std::vector<mlir::Value>(
              {prev, builder.create<ConstantIntOp>(loc, -1, ity)}));
    } else if (auto mt = ty.dyn_cast<MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(-1));
    } else {
      if (!ty.isa<mlir::IntegerType>()) {
        llvm::errs() << ty << " - " << prev << "\n";
        U->dump();
      }
      next = builder.create<SubIOp>(
          loc, prev,
          builder.create<ConstantIntOp>(loc, 1, ty.cast<mlir::IntegerType>()));
    }
    sub.store(builder, next);
    return ValueCategory(
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? prev
                                                                     : next,
        /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Real:
  case clang::UnaryOperator::Opcode::UO_Imag: {
    int fnum =
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_Real) ? 0 : 1;
    auto lhs_v = sub.val;
    assert(sub.isReference);
    if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      return ValueCategory(builder.create<polygeist::SubIndexOp>(
                               loc, mt0, lhs_v, getConstantIndex(fnum)),
                           /*isReference*/ true);
    } else if (auto PT =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      mlir::Type ET;
      if (auto ST =
              PT.getElementType().dyn_cast<mlir::LLVM::LLVMStructType>()) {
        ET = ST.getBody()[fnum];
      } else {
        ET = PT.getElementType()
                 .cast<mlir::LLVM::LLVMArrayType>()
                 .getElementType();
      }
      mlir::Value vec[3] = {lhs_v, builder.create<ConstantIntOp>(loc, 0, 32),
                            builder.create<ConstantIntOp>(loc, fnum, 32)};
      return ValueCategory(
          builder.create<mlir::LLVM::GEPOp>(
              loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
              vec),
          /*isReference*/ true);
    }

    llvm::errs() << "lhs_v: " << lhs_v << "\n";
    U->dump();
    assert(0 && "unhandled real");
  }
  default: {
    U->dump();
    assert(0 && "unhandled opcode");
  }
  }
}

ValueCategory MLIRScanner::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *expr) {
  return Visit(expr->getReplacement());
}

ValueCategory
MLIRScanner::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Uop) {
  switch (Uop->getKind()) {
  case UETT_SizeOf: {
    auto value = getTypeSize(Uop->getTypeOfArgument());
    auto ty = getMLIRType(Uop->getType()).cast<mlir::IntegerType>();
    return ValueCategory(builder.create<ConstantIntOp>(loc, value, ty),
                         /*isReference*/ false);
  }
  default:
    Uop->dump();
    assert(0 && "unhandled VisitUnaryExprOrTypeTraitExpr");
  }
}

bool hasAffineArith(Operation *op, AffineExpr &expr,
                    mlir::Value &affineForIndVar) {
  // skip IndexCastOp
  if (isa<IndexCastOp>(op))
    return hasAffineArith(op->getOperand(0).getDefiningOp(), expr,
                          affineForIndVar);

  // induction variable are modelled as memref<1xType>
  // %1 = index_cast %induction : index to i32
  // %2 = alloca() : memref<1xi32>
  // store %1, %2[0] : memref<1xi32>
  // ...
  // %5 = load %2[0] : memref<1xf32>
  if (isa<mlir::memref::LoadOp>(op)) {
    auto load = cast<mlir::memref::LoadOp>(op);
    auto loadOperand = load.getOperand(0);
    if (loadOperand.getType().cast<MemRefType>().getShape().size() != 1)
      return false;
    auto maybeAllocaOp = loadOperand.getDefiningOp();
    if (!isa<mlir::memref::AllocaOp>(maybeAllocaOp))
      return false;
    auto allocaUsers = maybeAllocaOp->getUsers();
    if (llvm::none_of(allocaUsers, [](mlir::Operation *op) {
          if (isa<mlir::memref::StoreOp>(op))
            return true;
          return false;
        }))
      return false;
    for (auto user : allocaUsers)
      if (auto storeOp = dyn_cast<mlir::memref::StoreOp>(user)) {
        auto storeOperand = storeOp.getOperand(0);
        auto maybeIndexCast = storeOperand.getDefiningOp();
        if (!isa<IndexCastOp>(maybeIndexCast))
          return false;
        auto indexCastOperand = maybeIndexCast->getOperand(0);
        if (auto blockArg = indexCastOperand.dyn_cast<mlir::BlockArgument>()) {
          if (auto affineForOp = dyn_cast<mlir::AffineForOp>(
                  blockArg.getOwner()->getParentOp()))
            affineForIndVar = affineForOp.getInductionVar();
          else
            return false;
        }
      }
    return true;
  }

  // at this point we expect only AddIOp or MulIOp
  if ((!isa<AddIOp>(op)) && (!isa<MulIOp>(op))) {
    return false;
  }

  // make sure that the current op has at least one constant operand
  // (ConstantIndexOp or ConstantIntOp)
  if (llvm::none_of(op->getOperands(), [](mlir::Value operand) {
        return (isa<ConstantIndexOp>(operand.getDefiningOp()) ||
                isa<ConstantIntOp>(operand.getDefiningOp()));
      }))
    return false;

  // build affine expression by adding or multiplying constants.
  // and keep iterating on the non-constant index
  mlir::Value nonCstOperand = nullptr;
  for (auto operand : op->getOperands()) {
    if (auto constantIndexOp =
            dyn_cast<ConstantIndexOp>(operand.getDefiningOp())) {
      if (isa<AddIOp>(op))
        expr = expr + constantIndexOp.value();
      else
        expr = expr * constantIndexOp.value();
    } else if (auto constantIntOp =
                   dyn_cast<ConstantIntOp>(operand.getDefiningOp())) {
      if (isa<AddIOp>(op))
        expr = expr + constantIntOp.value();
      else
        expr = expr * constantIntOp.value();
    } else
      nonCstOperand = operand;
  }
  return hasAffineArith(nonCstOperand.getDefiningOp(), expr, affineForIndVar);
}

ValueCategory MLIRScanner::VisitBinaryOperator(clang::BinaryOperator *BO) {
  auto loc = getMLIRLocation(BO->getExprLoc());

  auto fixInteger = [&](mlir::Value res) {
    auto prevTy = res.getType().cast<mlir::IntegerType>();
    auto postTy = getMLIRType(BO->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (postTy != prevTy) {
      if (signedType) {
        res = builder.create<ExtSIOp>(loc, res, postTy);
      } else {
        res = builder.create<ExtUIOp>(loc, res, postTy);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  };

  auto lhs = Visit(BO->getLHS());
  if (!lhs.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->dump();
    BO->getLHS()->dump();
    assert(lhs.val);
  }

  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_LAnd: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto cond = lhs.getValue(builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    if (!cond.getType().isa<mlir::IntegerType>()) {
      BO->dump();
      BO->getType()->dump();
      llvm::errs() << "cond: " << cond << "\n";
    }
    auto prevTy = cond.getType().cast<mlir::IntegerType>();
    if (!prevTy.isInteger(1)) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, 0, prevTy));
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

    auto rhs = Visit(BO->getRHS()).getValue(builder);
    assert(rhs != nullptr);
    if (auto LT = rhs.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      rhs = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, rhs, nullptr_llvm);
    }
    if (!rhs.getType().cast<mlir::IntegerType>().isInteger(1)) {
      rhs = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, rhs,
          builder.create<ConstantIntOp>(loc, 0, rhs.getType()));
    }
    mlir::Value truearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().back());
    mlir::Value falsearray[] = {
        builder.create<ConstantIntOp>(loc, 0, types[0])};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);
    return fixInteger(ifOp.getResult(0));
  }
  case clang::BinaryOperator::Opcode::BO_LOr: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto cond = lhs.getValue(builder);
    auto prevTy = cond.getType().cast<mlir::IntegerType>();
    if (!prevTy.isInteger(1)) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, 0, prevTy));
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

    mlir::Value truearray[] = {builder.create<ConstantIntOp>(loc, 1, types[0])};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().back());
    auto rhs = Visit(BO->getRHS()).getValue(builder);
    if (!rhs.getType().cast<mlir::IntegerType>().isInteger(1)) {
      rhs = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, rhs,
          builder.create<ConstantIntOp>(loc, 0, rhs.getType()));
    }
    assert(rhs != nullptr);
    mlir::Value falsearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);

    return fixInteger(ifOp.getResult(0));
  }
  default:
    break;
  }
  auto rhs = Visit(BO->getRHS());
  if (!rhs.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->getRHS()->dump();
    assert(rhs.val);
  }
  // TODO note assumptions made here about unsigned / unordered
  bool signedType = true;
  if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
    if (bit->isUnsignedInteger())
      signedType = false;
    if (bit->isSignedInteger())
      signedType = true;
  }
  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_Shr: {
    auto lhsv = lhs.getValue(builder);
    auto rhsv = rhs.getValue(builder);
    auto prevTy = rhsv.getType().cast<mlir::IntegerType>();
    auto postTy = lhsv.getType().cast<mlir::IntegerType>();
    if (prevTy.getWidth() < postTy.getWidth())
      rhsv = builder.create<arith::ExtUIOp>(loc, rhsv, postTy);
    if (prevTy.getWidth() > postTy.getWidth())
      rhsv = builder.create<arith::TruncIOp>(loc, rhsv, postTy);
    assert(lhsv.getType() == rhsv.getType());
    if (signedType)
      return ValueCategory(builder.create<ShRSIOp>(loc, lhsv, rhsv),
                           /*isReference*/ false);
    else
      return ValueCategory(builder.create<ShRUIOp>(loc, lhsv, rhsv),
                           /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Shl: {
    auto lhsv = lhs.getValue(builder);
    auto rhsv = rhs.getValue(builder);
    auto prevTy = rhsv.getType().cast<mlir::IntegerType>();
    auto postTy = lhsv.getType().cast<mlir::IntegerType>();
    if (prevTy.getWidth() < postTy.getWidth())
      rhsv = builder.create<arith::ExtUIOp>(loc, rhsv, postTy);
    if (prevTy.getWidth() > postTy.getWidth())
      rhsv = builder.create<arith::TruncIOp>(loc, rhsv, postTy);
    assert(lhsv.getType() == rhsv.getType());
    return ValueCategory(builder.create<ShLIOp>(loc, lhsv, rhsv),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_And: {
    return ValueCategory(builder.create<AndIOp>(loc, lhs.getValue(builder),
                                                rhs.getValue(builder)),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Xor: {
    return ValueCategory(builder.create<XOrIOp>(loc, lhs.getValue(builder),
                                                rhs.getValue(builder)),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Or: {
    // TODO short circuit
    return ValueCategory(builder.create<OrIOp>(loc, lhs.getValue(builder),
                                               rhs.getValue(builder)),
                         /*isReference*/ false);
  }
    {
      auto lhs_v = lhs.getValue(builder);
      mlir::Value res;
      if (lhs_v.getType().isa<mlir::FloatType>()) {
        res = builder.create<CmpFOp>(loc, CmpFPredicate::UGT, lhs_v,
                                     rhs.getValue(builder));
      } else {
        res = builder.create<CmpIOp>(
            loc, signedType ? CmpIPredicate::sgt : CmpIPredicate::ugt, lhs_v,
            rhs.getValue(builder));
      }
      return fixInteger(res);
    }
  case clang::BinaryOperator::Opcode::BO_GT:
  case clang::BinaryOperator::Opcode::BO_GE:
  case clang::BinaryOperator::Opcode::BO_LT:
  case clang::BinaryOperator::Opcode::BO_LE:
  case clang::BinaryOperator::Opcode::BO_EQ:
  case clang::BinaryOperator::Opcode::BO_NE: {
    signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getLHS()->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    CmpFPredicate FPred;
    CmpIPredicate IPred;
    LLVM::ICmpPredicate LPred;
    switch (BO->getOpcode()) {
    case clang::BinaryOperator::Opcode::BO_GT:
      FPred = CmpFPredicate::UGT;
      IPred = signedType ? CmpIPredicate::sgt : CmpIPredicate::ugt,
      LPred = LLVM::ICmpPredicate::ugt;
      break;
    case clang::BinaryOperator::Opcode::BO_GE:
      FPred = CmpFPredicate::UGE;
      IPred = signedType ? CmpIPredicate::sge : CmpIPredicate::uge,
      LPred = LLVM::ICmpPredicate::uge;
      break;
    case clang::BinaryOperator::Opcode::BO_LT:
      FPred = CmpFPredicate::ULT;
      IPred = signedType ? CmpIPredicate::slt : CmpIPredicate::ult,
      LPred = LLVM::ICmpPredicate::ult;
      break;
    case clang::BinaryOperator::Opcode::BO_LE:
      FPred = CmpFPredicate::ULE;
      IPred = signedType ? CmpIPredicate::sle : CmpIPredicate::ule,
      LPred = LLVM::ICmpPredicate::ule;
      break;
    case clang::BinaryOperator::Opcode::BO_EQ:
      FPred = CmpFPredicate::UEQ;
      IPred = CmpIPredicate::eq;
      LPred = LLVM::ICmpPredicate::eq;
      break;
    case clang::BinaryOperator::Opcode::BO_NE:
      FPred = CmpFPredicate::UNE;
      IPred = CmpIPredicate::ne;
      LPred = LLVM::ICmpPredicate::ne;
      break;
    default:
      llvm_unreachable("Unknown op in binary comparision switch");
    }

    auto lhs_v = lhs.getValue(builder);
    auto rhs_v = rhs.getValue(builder);
    if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      lhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getElementType()), lhs_v);
    }
    if (auto mt = rhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      rhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getElementType()), rhs_v);
    }
    mlir::Value res;
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      res = builder.create<arith::CmpFOp>(loc, FPred, lhs_v, rhs_v);
    } else if (lhs_v.getType().isa<LLVM::LLVMPointerType>()) {
      res = builder.create<LLVM::ICmpOp>(loc, LPred, lhs_v, rhs_v);
    } else {
      res = builder.create<arith::CmpIOp>(loc, IPred, lhs_v, rhs_v);
    }
    return fixInteger(res);
  }
  case clang::BinaryOperator::Opcode::BO_Mul: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(
          builder.create<arith::MulFOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    } else {
      return ValueCategory(
          builder.create<arith::MulIOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Div: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(
          builder.create<arith::DivFOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
      ;
    } else {
      if (signedType)
        return ValueCategory(
            builder.create<arith::DivSIOp>(loc, lhs_v, rhs.getValue(builder)),
            /*isReference*/ false);
      else
        return ValueCategory(
            builder.create<arith::DivUIOp>(loc, lhs_v, rhs.getValue(builder)),
            /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Rem: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(
          builder.create<arith::RemFOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    } else {
      if (signedType)
        return ValueCategory(
            builder.create<arith::RemSIOp>(loc, lhs_v, rhs.getValue(builder)),
            /*isReference*/ false);
      else
        return ValueCategory(
            builder.create<arith::RemUIOp>(loc, lhs_v, rhs.getValue(builder)),
            /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Add: {
    auto lhs_v = lhs.getValue(builder);
    auto rhs_v = rhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(builder.create<AddFOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    } else if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      auto ptradd = rhs_v;
      ptradd = castToIndex(loc, ptradd);
      return ValueCategory(
          builder.create<polygeist::SubIndexOp>(loc, mt0, lhs_v, ptradd),
          /*isReference*/ false);
    } else if (auto pt =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      return ValueCategory(
          builder.create<LLVM::GEPOp>(loc, pt,
                                      std::vector<mlir::Value>({lhs_v, rhs_v})),
          /*isReference*/ false);
    } else {
      if (auto lhs_c = lhs_v.getDefiningOp<ConstantIntOp>()) {
        if (auto rhs_c = rhs_v.getDefiningOp<ConstantIntOp>()) {
          return ValueCategory(
              builder.create<arith::ConstantIntOp>(
                  loc, lhs_c.value() + rhs_c.value(), lhs_c.getType()),
              false);
        }
      }
      return ValueCategory(builder.create<AddIOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Sub: {
    auto lhs_v = lhs.getValue(builder);
    auto rhs_v = rhs.getValue(builder);
    if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      lhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getElementType()), lhs_v);
    }
    if (auto mt = rhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      rhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getElementType()), rhs_v);
    }
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      assert(rhs_v.getType() == lhs_v.getType());
      return ValueCategory(builder.create<SubFOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    } else if (auto pt =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      if (auto IT = rhs_v.getType().dyn_cast<mlir::IntegerType>()) {
        mlir::Value vals[1] = {builder.create<SubIOp>(
            loc, builder.create<ConstantIntOp>(loc, 0, IT.getWidth()), rhs_v)};
        return ValueCategory(
            builder.create<LLVM::GEPOp>(loc, lhs_v.getType(), lhs_v,
                                        ArrayRef<mlir::Value>(vals)),
            false);
      }
      return ValueCategory(
          builder.create<SubIOp>(loc,
                                 builder.create<LLVM::PtrToIntOp>(
                                     loc, getMLIRType(BO->getType()), lhs_v),
                                 builder.create<LLVM::PtrToIntOp>(
                                     loc, getMLIRType(BO->getType()), rhs_v)),
          /*isReference*/ false);
    } else {
      return ValueCategory(builder.create<SubIOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Assign: {
    assert(lhs.isReference);
    mlir::Value tostore = rhs.getValue(builder);
    mlir::Type subType;
    if (auto PT = lhs.val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>())
      subType = PT.getElementType();
    else
      subType = lhs.val.getType().cast<MemRefType>().getElementType();
    if (tostore.getType() != subType) {
      if (auto prevTy = tostore.getType().dyn_cast<mlir::IntegerType>()) {
        if (auto postTy = subType.dyn_cast<mlir::IntegerType>()) {
          bool signedType = true;
          if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
            if (bit->isUnsignedInteger())
              signedType = false;
            if (bit->isSignedInteger())
              signedType = true;
          }

          if (prevTy.getWidth() < postTy.getWidth()) {
            if (signedType) {
              tostore = builder.create<arith::ExtSIOp>(loc, tostore, postTy);
            } else {
              tostore = builder.create<arith::ExtUIOp>(loc, tostore, postTy);
            }
          } else if (prevTy.getWidth() > postTy.getWidth()) {
            tostore = builder.create<arith::TruncIOp>(loc, tostore, postTy);
          }
        }
      }
    }
    lhs.store(builder, tostore);
    return lhs;
  }

  case clang::BinaryOperator::Opcode::BO_Comma: {
    return rhs;
  }

  case clang::BinaryOperator::Opcode::BO_AddAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (auto postTy = prev.getType().dyn_cast<mlir::FloatType>()) {
      mlir::Value rhsV = rhs.getValue(builder);
      auto prevTy = rhsV.getType().cast<mlir::FloatType>();
      if (prevTy == postTy) {
      } else if (prevTy.getWidth() < postTy.getWidth()) {
        rhsV = builder.create<ExtFOp>(loc, rhsV, postTy);
      } else {
        rhsV = builder.create<TruncFOp>(loc, rhsV, postTy);
      }
      assert(rhsV.getType() == prev.getType());
      result = builder.create<AddFOp>(loc, prev, rhsV);
    } else if (auto pt =
                   prev.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      result = builder.create<LLVM::GEPOp>(
          loc, pt, std::vector<mlir::Value>({prev, rhs.getValue(builder)}));
    } else {
      auto postTy = prev.getType().dyn_cast<mlir::IntegerType>();
      mlir::Value rhsV = rhs.getValue(builder);
      auto prevTy = rhsV.getType().cast<mlir::IntegerType>();
      if (prevTy == postTy) {
      } else if (prevTy.getWidth() < postTy.getWidth()) {
        if (signedType) {
          rhsV = builder.create<ExtSIOp>(loc, rhsV, postTy);
        } else {
          rhsV = builder.create<ExtUIOp>(loc, rhsV, postTy);
        }
      } else {
        rhsV = builder.create<TruncIOp>(loc, rhsV, postTy);
      }
      assert(rhsV.getType() == prev.getType());
      result = builder.create<AddIOp>(loc, prev, rhsV);
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_SubAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      auto right = rhs.getValue(builder);
      if (right.getType() != prev.getType()) {
        auto prevTy = right.getType().cast<mlir::FloatType>();
        auto postTy = getMLIRType(BO->getType()).cast<mlir::FloatType>();

        if (prevTy.getWidth() < postTy.getWidth()) {
          right = builder.create<ExtFOp>(loc, right, postTy);
        } else {
          right = builder.create<TruncFOp>(loc, right, postTy);
        }
      }
      if (right.getType() != prev.getType()) {
        BO->dump();
        llvm::errs() << " p:" << prev << " r:" << right << "\n";
      }
      assert(right.getType() == prev.getType());
      result = builder.create<SubFOp>(loc, prev, right);
    } else {
      result = builder.create<SubIOp>(loc, prev, rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_MulAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      auto right = rhs.getValue(builder);
      if (right.getType() != prev.getType()) {
        auto prevTy = right.getType().cast<mlir::FloatType>();
        auto postTy = getMLIRType(BO->getType()).cast<mlir::FloatType>();

        if (prevTy.getWidth() < postTy.getWidth()) {
          right = builder.create<ExtFOp>(loc, right, postTy);
        } else {
          right = builder.create<TruncFOp>(loc, right, postTy);
        }
      }
      if (right.getType() != prev.getType()) {
        BO->dump();
        llvm::errs() << " p:" << prev << " r:" << right << "\n";
      }
      assert(right.getType() == prev.getType());
      result = builder.create<MulFOp>(loc, prev, right);
    } else {
      result = builder.create<MulIOp>(loc, prev, rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_DivAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      mlir::Value val = rhs.getValue(builder);
      auto prevTy = val.getType().cast<mlir::FloatType>();
      auto postTy = prev.getType().cast<mlir::FloatType>();

      if (prevTy.getWidth() < postTy.getWidth()) {
        val = builder.create<ExtFOp>(loc, val, postTy);
      } else if (prevTy.getWidth() > postTy.getWidth()) {
        val = builder.create<TruncFOp>(loc, val, postTy);
      }
      result = builder.create<DivFOp>(loc, prev, val);
    } else {
      if (signedType)
        result = builder.create<DivSIOp>(loc, prev, rhs.getValue(builder));
      else
        result = builder.create<DivUIOp>(loc, prev, rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_ShrAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;

    if (signedType)
      result = builder.create<ShRSIOp>(loc, prev, rhs.getValue(builder));
    else
      result = builder.create<ShRUIOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_ShlAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result =
        builder.create<ShLIOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_RemAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;

    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<RemFOp>(loc, prev, rhs.getValue(builder));
    } else {
      if (signedType)
        result = builder.create<RemSIOp>(loc, prev, rhs.getValue(builder));
      else
        result = builder.create<RemUIOp>(loc, prev, rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_AndAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result =
        builder.create<AndIOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_OrAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result =
        builder.create<OrIOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_XorAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result =
        builder.create<XOrIOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }

  default: {
    BO->dump();
    assert(0 && "unhandled opcode");
  }
  }
}

ValueCategory MLIRScanner::VisitExprWithCleanups(ExprWithCleanups *E) {
  auto ret = Visit(E->getSubExpr());
  for (auto &child : E->children()) {
    child->dump();
    llvm::errs() << "cleanup not handled\n";
  }
  return ret;
}

ValueCategory MLIRScanner::CommonFieldLookup(clang::QualType CT,
                                             const FieldDecl *FD,
                                             mlir::Value val, bool isLValue) {
  assert(FD && "Attempting to lookup field of nullptr");
  auto rd = FD->getParent();

  auto ST = cast<llvm::StructType>(getLLVMType(CT));

  size_t fnum = 0;

  auto CXRD = dyn_cast<CXXRecordDecl>(rd);

  if (rd->isUnion() ||
      (CXRD && (!CXRD->hasDefinition() || CXRD->isPolymorphic() ||
                CXRD->getDefinition()->getNumBases() > 0)) ||
      (!ST->isLiteral() && (ST->getName() == "struct._IO_FILE" ||
                            ST->getName() == "class.std::basic_ifstream" ||
                            ST->getName() == "class.std::basic_istream" ||
                            ST->getName() == "class.std::basic_ostream" ||
                            ST->getName() == "class.std::basic_ofstream"))) {
    auto &layout = Glob.CGM.getTypes().getCGRecordLayout(rd);
    fnum = layout.getLLVMFieldNo(FD);
  } else {
    fnum = 0;
    for (auto field : rd->fields()) {
      if (field == FD) {
        break;
      }
      fnum++;
    }
  }

  if (auto PT = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    mlir::Value vec[3] = {val, builder.create<ConstantIntOp>(loc, 0, 32),
                          builder.create<ConstantIntOp>(loc, fnum, 32)};
    if (!PT.getElementType()
             .isa<mlir::LLVM::LLVMStructType, mlir::LLVM::LLVMArrayType>()) {
      llvm::errs() << "function: " << function << "\n";
      // rd->dump();
      FD->dump();
      FD->getType()->dump();
      llvm::errs() << " val: " << val << " - pt: " << PT << " fn: " << fnum
                   << " ST: " << *ST << "\n";
    }
    mlir::Type ET;
    if (auto ST = PT.getElementType().dyn_cast<mlir::LLVM::LLVMStructType>()) {
      ET = ST.getBody()[fnum];
    } else {
      ET = PT.getElementType()
               .cast<mlir::LLVM::LLVMArrayType>()
               .getElementType();
    }
    mlir::Value commonGEP = builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()), vec);
    if (rd->isUnion()) {
      auto subType =
          Glob.typeTranslator.translateType(getLLVMType(FD->getType()));
      commonGEP = builder.create<mlir::LLVM::BitcastOp>(
          loc, mlir::LLVM::LLVMPointerType::get(subType, PT.getAddressSpace()),
          commonGEP);
    }
    if (isLValue)
      commonGEP =
          ValueCategory(commonGEP, /*isReference*/ true).getValue(builder);
    return ValueCategory(commonGEP, /*isReference*/ true);
  }
  auto mt = val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() > 1) {
    shape.erase(shape.begin());
  } else {
    shape[0] = -1;
  }
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());
  shape[0] = -1;
  auto mt1 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());
  mlir::Value sub0 =
      builder.create<polygeist::SubIndexOp>(loc, mt0, val, getConstantIndex(0));
  mlir::Value sub1 = builder.create<polygeist::SubIndexOp>(
      loc, mt1, sub0, getConstantIndex(fnum));
  if (isLValue)
    sub1 = ValueCategory(sub1, /*isReference*/ true).getValue(builder);
  return ValueCategory(sub1, /*isReference*/ true);
}

ValueCategory MLIRScanner::VisitDeclRefExpr(DeclRefExpr *E) {
  auto name = E->getDecl()->getName().str();

  if (auto tocall = dyn_cast<FunctionDecl>(E->getDecl()))
    return ValueCategory(builder.create<LLVM::AddressOfOp>(
                             loc, Glob.GetOrCreateLLVMFunction(tocall)),
                         /*isReference*/ true);

  if (auto VD = dyn_cast<VarDecl>(E->getDecl())) {
    if (Captures.find(VD) != Captures.end()) {
      FieldDecl *field = Captures[VD];
      auto res = CommonFieldLookup(
          cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(), field,
          ThisVal.val,
          isa<clang::ReferenceType>(
              field->getType()->getUnqualifiedDesugaredType()));
      assert(CaptureKinds.find(VD) != CaptureKinds.end());
      return res;
    }
  }

  if (auto PD = dyn_cast<VarDecl>(E->getDecl())) {
    auto found = params.find(PD);
    if (found != params.end()) {
      auto res = found->second;
      assert(res.val);
      return res;
    }
  }
  if (auto ED = dyn_cast<EnumConstantDecl>(E->getDecl())) {
    auto ty = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    return ValueCategory(
        builder.create<ConstantIntOp>(loc, ED->getInitVal().getExtValue(), ty),
        /*isReference*/ false);

    if (!ED->getInitExpr())
      ED->dump();
    return Visit(ED->getInitExpr());
  }
  if (auto VD = dyn_cast<ValueDecl>(E->getDecl())) {
    if (Glob.getMLIRType(Glob.CGM.getContext().getPointerType(E->getType()))
            .isa<mlir::LLVM::LLVMPointerType>() ||
        name == "stderr" || name == "stdout" || name == "stdin" ||
        (E->hasQualifier())) {
      return ValueCategory(builder.create<mlir::LLVM::AddressOfOp>(
                               loc, Glob.GetOrCreateLLVMGlobal(VD)),
                           /*isReference*/ true);
    }

    auto gv = Glob.GetOrCreateGlobal(VD, /*prefix=*/"");
    auto gv2 = builder.create<memref::GetGlobalOp>(loc, gv.first.type(),
                                                   gv.first.getName());
    bool isArray = gv.second;
    // TODO check reference
    if (isArray)
      return ValueCategory(gv2, /*isReference*/ true);
    else
      return ValueCategory(gv2, /*isReference*/ true);
    // return gv2;
  }
  E->dump();
  E->getDecl()->dump();
  llvm::errs() << "couldn't find " << name << "\n";
  assert(0 && "couldnt find value");
  return nullptr;
}

ValueCategory MLIRScanner::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  if (!E->getSourceExpr()) {
    E->dump();
    assert(E->getSourceExpr());
  }
  auto res = Visit(E->getSourceExpr());
  if (!res.val) {
    E->dump();
    E->getSourceExpr()->dump();
    assert(res.val);
  }
  return res;
}

ValueCategory MLIRScanner::VisitCXXTypeidExpr(clang::CXXTypeidExpr *E) {
  QualType T;
  if (E->isTypeOperand())
    T = E->getTypeOperand(Glob.CGM.getContext());
  else
    T = E->getExprOperand()->getType();
  llvm::Constant *C = Glob.CGM.GetAddrOfRTTIDescriptor(T);
  llvm::errs() << *C << "\n";
  auto ty = getMLIRType(E->getType());
  llvm::errs() << ty << "\n";
  assert(0 && "unhandled typeid");
}

ValueCategory
MLIRScanner::VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr) {
  assert(ThisVal.val);
  auto toset = Visit(expr->getExpr());
  assert(!ThisVal.isReference);
  assert(toset.val);

  bool isArray = false;
  Glob.getMLIRType(expr->getExpr()->getType(), &isArray);

  auto cfl = CommonFieldLookup(
      cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(),
      expr->getField(), ThisVal.val, /*isLValue*/ false);
  assert(cfl.val);
  cfl.store(builder, toset, isArray);
  return cfl;
}

ValueCategory MLIRScanner::VisitMemberExpr(MemberExpr *ME) {
  auto memberName = ME->getMemberDecl()->getName();
  if (auto sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
    if (auto sr = dyn_cast<DeclRefExpr>(sr2->getSourceExpr())) {
      if (sr->getDecl()->getName() == "blockIdx") {
        if (memberName == "__fetch_builtin_x") {
        }
        llvm::errs() << "known block index";
      }
      if (sr->getDecl()->getName() == "blockDim") {
        llvm::errs() << "known block dim";
      }
      if (sr->getDecl()->getName() == "threadIdx") {
        llvm::errs() << "known thread index";
      }
      if (sr->getDecl()->getName() == "gridDim") {
        llvm::errs() << "known grid index";
      }
    }
  }
  auto base = Visit(ME->getBase());
  clang::QualType OT = ME->getBase()->getType();
  if (ME->isArrow()) {
    if (!base.val) {
      ME->dump();
    }
    base = base.dereference(builder);
    OT = cast<clang::PointerType>(OT->getUnqualifiedDesugaredType())
             ->getPointeeType();
  }
  if (!base.isReference) {
    EmittingFunctionDecl->dump();
    function.dump();
    ME->dump();
    llvm::errs() << "base value: " << base.val << "\n";
  }
  assert(base.isReference);
  const FieldDecl *field = cast<FieldDecl>(ME->getMemberDecl());
  return CommonFieldLookup(
      OT, field, base.val,
      isa<clang::ReferenceType>(
          field->getType()->getUnqualifiedDesugaredType()));
}

ValueCategory MLIRScanner::VisitCastExpr(CastExpr *E) {
  auto loc = getMLIRLocation(E->getExprLoc());
  switch (E->getCastKind()) {

  case clang::CastKind::CK_NullToPointer: {
    auto llvmType = getMLIRType(E->getType());
    if (llvmType.isa<LLVM::LLVMPointerType>())
      return ValueCategory(builder.create<mlir::LLVM::NullOp>(loc, llvmType),
                           /*isReference*/ false);
    else
      return ValueCategory(
          builder.create<polygeist::Pointer2MemrefOp>(
              loc, llvmType,
              builder.create<mlir::LLVM::NullOp>(
                  loc, LLVM::LLVMPointerType::get(builder.getI8Type()))),
          false);
  }
  case clang::CastKind::CK_UserDefinedConversion: {
    return Visit(E->getSubExpr());
  }
  case clang::CastKind::CK_BaseToDerived:
  case clang::CastKind::CK_DerivedToBase:
  case clang::CastKind::CK_UncheckedDerivedToBase: {
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    assert(se.val);
    if (auto opt = se.val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      mlir::Type nt = getMLIRType(
          (E->isLValue() || E->isXValue())
              ? Glob.CGM.getContext().getLValueReferenceType(E->getType())
              : E->getType());
      auto pt = nt.dyn_cast<mlir::LLVM::LLVMPointerType>();
      if (!pt) {
        if (!nt.isa<MemRefType>()) {
          E->dump();
          E->getType()->dump();
          llvm::errs() << " nt: " << nt << "\n";
          assert(nt.isa<MemRefType>());
        }
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(loc, nt, se.val),
            se.isReference);
      }
      auto nval = builder.create<mlir::LLVM::BitcastOp>(loc, pt, se.val);
      return ValueCategory(nval, /*isReference*/ se.isReference);
    }
    if (!se.val.getType().isa<mlir::MemRefType>() || se.isReference) {
      E->dump();
      E->getType()->dump();
      llvm::errs() << se.val << " isref: " << (int)se.isReference << "\n";
    }

    // No reason this can't be handled, just isn't implemented yet.
    assert(!se.isReference);
    auto ut = se.val.getType().cast<mlir::MemRefType>();
    auto mt = getMLIRType(E->getType()).cast<mlir::MemRefType>();
    if (ut.getShape().size() != mt.getShape().size()) {
      E->dump();
      llvm::errs() << " se.val: " << se.val << " ut: " << ut << " mt: " << mt
                   << "\n";
    }
    assert(ut.getShape().size() == mt.getShape().size());
    auto ty =
        mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                              MemRefLayoutAttrInterface(), ut.getMemorySpace());
    return ValueCategory(builder.create<mlir::memref::CastOp>(loc, se.val, ty),
                         /*isReference*/ se.isReference);
  }
  case clang::CastKind::CK_BitCast: {

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getIdentifier() &&
              sr->getDecl()->getName() == "polybench_alloc_data") {
            if (auto mt =
                    getMLIRType(E->getType()).dyn_cast<mlir::MemRefType>()) {
              auto shape = std::vector<int64_t>(mt.getShape());
              // shape.erase(shape.begin());
              auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                               MemRefLayoutAttrInterface(),
                                               mt.getMemorySpace());

              auto alloc = builder.create<mlir::memref::AllocOp>(loc, mt0);
              return ValueCategory(alloc, /*isReference*/ false);
            }
          }
        }

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getIdentifier() &&
              (sr->getDecl()->getName() == "malloc" ||
               sr->getDecl()->getName() == "calloc"))
            if (auto mt =
                    getMLIRType(E->getType()).dyn_cast<mlir::MemRefType>()) {
              auto shape = std::vector<int64_t>(mt.getShape());

              auto elemSize =
                  getTypeSize(cast<clang::PointerType>(
                                  E->getType()->getUnqualifiedDesugaredType())
                                  ->getPointeeType());
              mlir::Value allocSize = builder.create<IndexCastOp>(
                  loc, Visit(CI->getArg(0)).getValue(builder),
                  mlir::IndexType::get(builder.getContext()));
              if (sr->getDecl()->getName() == "calloc") {
                allocSize = builder.create<MulIOp>(
                    loc, allocSize,
                    builder.create<IndexCastOp>(
                        loc, Visit(CI->getArg(1)).getValue(builder),
                        mlir::IndexType::get(builder.getContext())));
              }
              mlir::Value args[1] = {builder.create<DivUIOp>(
                  loc, allocSize,
                  builder.create<ConstantIndexOp>(loc, elemSize))};
              auto alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
              if (sr->getDecl()->getName() == "calloc") {
                mlir::Value val = alloc;
                if (val.getType().isa<MemRefType>()) {
                  val = builder.create<polygeist::Memref2PointerOp>(
                      loc, LLVM::LLVMPointerType::get(builder.getI8Type()), val);
                } else {
                  val = builder.create<LLVM::BitcastOp>(
                      loc,
                      LLVM::LLVMPointerType::get(
                          builder.getI8Type(),
                          val.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
                      val);
                }
                auto i8_0 = builder.create<ConstantIntOp>(loc, 0, 8);
                auto sizev = builder.create<IndexCastOp>(loc, allocSize, builder.getIndexType());
                auto falsev = builder.create<ConstantIntOp>(loc, false, 1);
                builder.create<LLVM::MemsetOp>(loc, val, i8_0, sizev, falsev);
              }
              return ValueCategory(alloc, /*isReference*/ false);
            }
        }
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    auto scalar = se.getValue(builder);
    if (auto spt = scalar.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nt = getMLIRType(E->getType());
      LLVM::LLVMPointerType pt = nt.dyn_cast<LLVM::LLVMPointerType>();
      if (!pt) {
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(loc, nt, scalar),
            false);
      }
      pt = LLVM::LLVMPointerType::get(pt.getElementType(),
                                      spt.getAddressSpace());
      auto nval = builder.create<mlir::LLVM::BitcastOp>(loc, pt, scalar);
      return ValueCategory(nval, /*isReference*/ false);
    }
    if (!scalar.getType().isa<mlir::MemRefType>()) {
      E->dump();
      E->getType()->dump();
      llvm::errs() << "scalar: " << scalar << "\n";
    }
    auto ut = scalar.getType().cast<mlir::MemRefType>();
    auto mlirty = getMLIRType(E->getType());

    if (auto PT = mlirty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      return ValueCategory(
          builder.create<mlir::polygeist::Memref2PointerOp>(loc, PT, scalar),
          /*isReference*/ false);
    } else if (auto mt = mlirty.dyn_cast<mlir::MemRefType>()) {
      auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      ut.getMemorySpace());
      if (ut.getShape().size() == mt.getShape().size() + 1) {
        return ValueCategory(builder.create<mlir::polygeist::SubIndexOp>(
                                 loc, ty, scalar, getConstantIndex(0)),
                             /*isReference*/ false);
      }
      if (ut.getShape().size() != mt.getShape().size()) {
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(
                loc, ty,
                builder.create<polygeist::Memref2PointerOp>(
                    loc, LLVM::LLVMPointerType::get(builder.getI8Type()),
                    scalar)),
            /*isReference*/ false);
      }
      return ValueCategory(
          builder.create<mlir::memref::CastOp>(loc, scalar, ty),
          /*isReference*/ false);
    } else {
      E->dump();
      E->getType()->dump();
      llvm::errs() << " scalar: " << scalar << " mlirty: " << mlirty << "\n";
      assert(0 && "illegal type for cast");
      llvm_unreachable("illegal type for cast");
    }
  }
  case clang::CastKind::CK_LValueToRValue: {
    if (auto dr = dyn_cast<DeclRefExpr>(E->getSubExpr())) {
      if (dr->getDecl()->getIdentifier() &&
          dr->getDecl()->getName() == "warpSize") {
        auto mlirType = getMLIRType(E->getType());
        return ValueCategory(
            builder.create<mlir::NVVM::WarpSizeOp>(loc, mlirType),
            /*isReference*/ false);
      }
      /*
      if (dr->isNonOdrUseReason() == clang::NonOdrUseReason::NOUR_Constant) {
        dr->dump();
        auto VD = cast<VarDecl>(dr->getDecl());
        assert(VD->getInit());
      }
      */
    }
    auto prev = Visit(E->getSubExpr());

    bool isArray = false;
    Glob.getMLIRType(E->getType(), &isArray);
    if (isArray)
      return prev;

    auto lres = prev.getValue(builder);
    if (!prev.isReference) {
      E->dump();
      lres.dump();
    }
    assert(prev.isReference);
    return ValueCategory(lres, /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralToFloating: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    auto ty = getMLIRType(E->getType()).cast<mlir::FloatType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getSubExpr()->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (signedType)
      return ValueCategory(builder.create<SIToFPOp>(loc, scalar, ty),
                           /*isReference*/ false);
    else
      return ValueCategory(builder.create<UIToFPOp>(loc, scalar, ty),
                           /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingToIntegral: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    auto ty = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (signedType)
      return ValueCategory(builder.create<FPToSIOp>(loc, scalar, ty),
                           /*isReference*/ false);
    else
      return ValueCategory(builder.create<FPToUIOp>(loc, scalar, ty),
                           /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    assert(scalar);
    auto postTy = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    if (scalar.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return ValueCategory(
          builder.create<mlir::LLVM::PtrToIntOp>(loc, postTy, scalar),
          /*isReference*/ false);
    }
    if (scalar.getType().isa<mlir::IndexType>() ||
        postTy.isa<mlir::IndexType>()) {
      return ValueCategory(builder.create<IndexCastOp>(loc, scalar, postTy),
                           false);
    }
    if (!scalar.getType().isa<mlir::IntegerType>()) {
      E->dump();
      llvm::errs() << " scalar: " << scalar << "\n";
    }
    auto prevTy = scalar.getType().cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }

    if (prevTy == postTy)
      return ValueCategory(scalar, /*isReference*/ false);
    if (prevTy.getWidth() < postTy.getWidth()) {
      if (signedType) {
        if (auto CI = scalar.getDefiningOp<ConstantIntOp>()) {
          return ValueCategory(
              builder.create<arith::ConstantOp>(
                  loc, postTy,
                  mlir::IntegerAttr::get(
                      postTy, CI.getValue().cast<IntegerAttr>().getValue().sext(
                                  postTy.getWidth()))),
              /*isReference*/ false);
        }
        return ValueCategory(
            builder.create<arith::ExtSIOp>(loc, scalar, postTy),
            /*isReference*/ false);
      } else {
        if (auto CI = scalar.getDefiningOp<ConstantIntOp>()) {
          return ValueCategory(
              builder.create<arith::ConstantOp>(
                  loc, postTy,
                  mlir::IntegerAttr::get(
                      postTy, CI.getValue().cast<IntegerAttr>().getValue().zext(
                                  postTy.getWidth()))),
              /*isReference*/ false);
        }
        return ValueCategory(
            builder.create<arith::ExtUIOp>(loc, scalar, postTy),
            /*isReference*/ false);
      }
    } else {
      if (auto CI = scalar.getDefiningOp<ConstantIntOp>()) {
        return ValueCategory(
            builder.create<arith::ConstantOp>(
                loc, postTy,
                mlir::IntegerAttr::get(
                    postTy, CI.getValue().cast<IntegerAttr>().getValue().trunc(
                                postTy.getWidth()))),
            /*isReference*/ false);
      }
      return ValueCategory(builder.create<arith::TruncIOp>(loc, scalar, postTy),
                           /*isReference*/ false);
    }
  }
  case clang::CastKind::CK_FloatingCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    if (!scalar.getType().isa<mlir::FloatType>()) {
      E->dump();
      llvm::errs() << "scalar: " << scalar << "\n";
    }
    auto prevTy = scalar.getType().cast<mlir::FloatType>();
    auto postTy = getMLIRType(E->getType()).cast<mlir::FloatType>();

    if (prevTy == postTy)
      return ValueCategory(scalar, /*isReference*/ false);
    if (auto c = scalar.getDefiningOp<ConstantFloatOp>()) {
      APFloat Val = c.getValue().cast<FloatAttr>().getValue();
      bool ignored;
      Val.convert(postTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                  &ignored);
      return ValueCategory(builder.create<arith::ConstantOp>(
                               loc, postTy, mlir::FloatAttr::get(postTy, Val)),
                           false);
    }
    if (prevTy.getWidth() < postTy.getWidth()) {
      return ValueCategory(builder.create<arith::ExtFOp>(loc, scalar, postTy),
                           /*isReference*/ false);
    } else {
      return ValueCategory(builder.create<arith::TruncFOp>(loc, scalar, postTy),
                           /*isReference*/ false);
    }
  }
  case clang::CastKind::CK_ArrayToPointerDecay: {
    return CommonArrayToPointer(Visit(E->getSubExpr()));

#if 0
    auto mt = scalar.val.getType().cast<mlir::MemRefType>();
    auto shape2 = std::vector<int64_t>(mt.getShape());
    if (shape2.size() == 0) {
      E->dump();
      //nex.dump();
      assert(0);
    }
    shape2[0] = -1;
    auto nex = mlir::MemRefType::get(shape2, mt.getElementType(),
                                     mt.getLayout(), mt.getMemorySpace());
    auto cst = builder.create<mlir::MemRefCastOp>(loc, scalar.val, nex);
    //llvm::errs() << "<ArrayToPtrDecay>\n";
    //E->dump();
    //llvm::errs() << cst << " - " << scalar.val << "\n";
    //auto offs = scalar.offsets;
    //offs.push_back(getConstantIndex(0));
    return ValueCategory(cst, scalar.isReference);
#endif
  }
  case clang::CastKind::CK_FunctionToPointerDecay: {
    auto scalar = Visit(E->getSubExpr());
    assert(scalar.isReference);
    return ValueCategory(scalar.val, /*isReference*/ false);
  }
  case clang::CastKind::CK_ConstructorConversion:
  case clang::CastKind::CK_NoOp: {
    return Visit(E->getSubExpr());
  }
  case clang::CastKind::CK_ToVoid: {
    Visit(E->getSubExpr());
    return nullptr;
  }
  case clang::CastKind::CK_PointerToBoolean: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    if (auto mt = scalar.getType().dyn_cast<mlir::MemRefType>()) {
      scalar = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getElementType()), scalar);
    }
    if (auto LT = scalar.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, scalar, nullptr_llvm);
      return ValueCategory(ne, /*isReference*/ false);
    }
    function.dump();
    llvm::errs() << "scalar: " << scalar << "\n";
    E->dump();
    assert(0 && "unhandled ptrtobool cast");
  }
  case clang::CastKind::CK_PointerToIntegral: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    if (auto mt = scalar.getType().dyn_cast<mlir::MemRefType>()) {
      scalar = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getElementType()), scalar);
    }
    if (auto LT = scalar.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto mlirType = getMLIRType(E->getType());
      auto val = builder.create<mlir::LLVM::BitcastOp>(loc, mlirType, scalar);
      return ValueCategory(val, /*isReference*/ false);
    }
    function.dump();
    llvm::errs() << "scalar: " << scalar << "\n";
    E->dump();
    assert(0 && "unhandled ptrtoint cast");
  }
  case clang::CastKind::CK_IntegralToBoolean: {
    auto res = Visit(E->getSubExpr()).getValue(builder);
    auto prevTy = res.getType().cast<mlir::IntegerType>();
    res = builder.create<arith::CmpIOp>(
        loc, CmpIPredicate::ne, res,
        builder.create<ConstantIntOp>(loc, 0, prevTy));
    auto postTy = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (postTy.getWidth() > 1) {
      if (signedType) {
        res = builder.create<ExtSIOp>(loc, res, postTy);
      } else {
        res = builder.create<ExtUIOp>(loc, res, postTy);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingToBoolean: {
    auto res = Visit(E->getSubExpr()).getValue(builder);
    auto prevTy = res.getType().cast<mlir::FloatType>();
    auto postTy = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    auto Zero = builder.create<ConstantFloatOp>(
        loc, APFloat::getZero(prevTy.getFloatSemantics()), prevTy);
    res = builder.create<arith::CmpFOp>(loc, CmpFPredicate::UNE, res, Zero);
    if (1 < postTy.getWidth()) {
      if (signedType) {
        res = builder.create<ExtSIOp>(loc, res, postTy);
      } else {
        res = builder.create<ExtUIOp>(loc, res, postTy);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralToPointer: {
    auto res = Visit(E->getSubExpr()).getValue(builder);
    auto postTy = getMLIRType(E->getType()).cast<LLVM::LLVMPointerType>();
    res = builder.create<LLVM::BitcastOp>(loc, postTy, res);
    return ValueCategory(res, /*isReference*/ false);
  }

  default:
    EmittingFunctionDecl->dump();
    E->dump();
    assert(0 && "unhandled cast");
  }
}

ValueCategory
MLIRScanner::VisitConditionalOperator(clang::ConditionalOperator *E) {
  auto cond = Visit(E->getCond()).getValue(builder);
  assert(cond != nullptr);
  if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
    cond = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
  }
  auto prevTy = cond.getType().cast<mlir::IntegerType>();
  if (!prevTy.isInteger(1)) {
    cond = builder.create<arith::CmpIOp>(
        loc, CmpIPredicate::ne, cond,
        builder.create<ConstantIntOp>(loc, 0, prevTy));
  }
  std::vector<mlir::Type> types;
  if (!E->getType()->isVoidType())
    types.push_back(getMLIRType(E->getType()));
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                              /*hasElseRegion*/ true);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();
  builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

  auto trueExpr = Visit(E->getTrueExpr());

  bool isReference = E->isLValue() || E->isXValue();

  std::vector<mlir::Value> truearray;
  if (!E->getType()->isVoidType()) {
    if (!trueExpr.val) {
      E->dump();
    }
    assert(trueExpr.val);
    mlir::Value truev;
    if (isReference) {
      assert(trueExpr.isReference);
      truev = trueExpr.val;
    } else {
      if (trueExpr.isReference)
        if (auto mt = trueExpr.val.getType().dyn_cast<MemRefType>())
          if (mt.getShape().size() != 1) {
            E->dump();
            E->getTrueExpr()->dump();
            llvm::errs() << " trueExpr: " << trueExpr.val << "\n";
            assert(0);
          }
      truev = trueExpr.getValue(builder);
    }
    assert(truev != nullptr);
    truearray.push_back(truev);
    builder.create<mlir::scf::YieldOp>(loc, truearray);
  }

  builder.setInsertionPointToStart(&ifOp.getElseRegion().back());

  auto falseExpr = Visit(E->getFalseExpr());
  std::vector<mlir::Value> falsearray;
  if (!E->getType()->isVoidType()) {
    mlir::Value falsev;
    if (isReference) {
      assert(falseExpr.isReference);
      falsev = falseExpr.val;
    } else
      falsev = falseExpr.getValue(builder);
    assert(falsev != nullptr);
    falsearray.push_back(falsev);
    builder.create<mlir::scf::YieldOp>(loc, falsearray);
  }

  builder.setInsertionPoint(oldblock, oldpoint);

  for (size_t i = 0; i < truearray.size(); i++)
    types[i] = truearray[i].getType();
  auto newIfOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                 /*hasElseRegion*/ true);
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  ifOp.erase();
  return ValueCategory(newIfOp.getResult(0), /*isReference*/ isReference);
}

ValueCategory MLIRScanner::VisitStmtExpr(clang::StmtExpr *stmt) {
  ValueCategory off = nullptr;
  for (auto a : stmt->getSubStmt()->children()) {
    off = Visit(a);
  }
  return off;
}

mlir::LLVM::LLVMFuncOp MLIRASTConsumer::GetOrCreateMallocFunction() {
  std::string name = "malloc";
  if (llvmFunctions.find(name) != llvmFunctions.end()) {
    return llvmFunctions[name];
  }
  auto ctx = module->getContext();
  mlir::Type types[] = {mlir::IntegerType::get(ctx, 64)};
  auto llvmFnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMPointerType::get(mlir::IntegerType::get(ctx, 8)), types, false);

  LLVM::Linkage lnk = LLVM::Linkage::External;
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module->getLoc(), name, llvmFnType, lnk);
}

mlir::LLVM::LLVMFuncOp
MLIRASTConsumer::GetOrCreateLLVMFunction(const FunctionDecl *FD) {
  std::string name;
  if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
  else if (auto CC = dyn_cast<CXXDestructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
  else
    name = CGM.getMangledName(FD).str();

  if (llvmFunctions.find(name) != llvmFunctions.end()) {
    return llvmFunctions[name];
  }

  std::vector<mlir::Type> types;
  if (auto CC = dyn_cast<CXXMethodDecl>(FD)) {
    types.push_back(typeTranslator.translateType(
        anonymize(getLLVMType(CC->getThisType()))));
  }
  for (auto parm : FD->parameters()) {
    types.push_back(typeTranslator.translateType(
        anonymize(getLLVMType(parm->getOriginalType()))));
  }

  auto rt =
      typeTranslator.translateType(anonymize(getLLVMType(FD->getReturnType())));

  auto llvmFnType = LLVM::LLVMFunctionType::get(rt, types,
                                                /*isVarArg=*/FD->isVariadic());

  LLVM::Linkage lnk;
  switch (CGM.getFunctionLinkage(FD)) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
    lnk = LLVM::Linkage::Internal;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    lnk = LLVM::Linkage::External;
    break;
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    // Available Externally not supported in MLIR LLVM Dialect
    // lnk = LLVM::Linkage::AvailableExternally;
    lnk = LLVM::Linkage::External;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    lnk = LLVM::Linkage::Linkonce;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    lnk = LLVM::Linkage::Weak;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    lnk = LLVM::Linkage::WeakODR;
    break;
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
    lnk = LLVM::Linkage::Common;
    break;
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    lnk = LLVM::Linkage::Appending;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    lnk = LLVM::Linkage::ExternWeak;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    lnk = LLVM::Linkage::LinkonceODR;
    break;
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    lnk = LLVM::Linkage::Private;
    break;
  }
  // Insert the function into the body of the parent module.
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module->getLoc(), name, llvmFnType, lnk);
}

mlir::LLVM::GlobalOp
MLIRASTConsumer::GetOrCreateLLVMGlobal(const ValueDecl *FD) {
  std::string name = CGM.getMangledName(FD).str();

  if (llvmGlobals.find(name) != llvmGlobals.end()) {
    return llvmGlobals[name];
  }

  LLVM::Linkage lnk;
  if (!isa<VarDecl>(FD))
    FD->dump();
  switch (CGM.getLLVMLinkageVarDefinition(cast<VarDecl>(FD),
                                          /*isConstant*/ false)) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
    lnk = LLVM::Linkage::Internal;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    lnk = LLVM::Linkage::External;
    break;
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    lnk = LLVM::Linkage::AvailableExternally;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    lnk = LLVM::Linkage::Linkonce;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    lnk = LLVM::Linkage::Weak;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    lnk = LLVM::Linkage::WeakODR;
    break;
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
    lnk = LLVM::Linkage::Common;
    break;
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    lnk = LLVM::Linkage::Appending;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    lnk = LLVM::Linkage::ExternWeak;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    lnk = LLVM::Linkage::LinkonceODR;
    break;
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    lnk = LLVM::Linkage::Private;
    break;
  }

  if (cast<VarDecl>(FD)->getInit())
    cast<VarDecl>(FD)->getInit()->dump();

  auto rt = getMLIRType(FD->getType());

  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());

  auto glob = builder.create<LLVM::GlobalOp>(
      module->getLoc(), rt, /*constant*/ false, lnk, name, mlir::Attribute());

  if (cast<VarDecl>(FD)->isThisDeclarationADefinition() ==
          VarDecl::Definition ||
      cast<VarDecl>(FD)->isThisDeclarationADefinition() ==
          VarDecl::TentativeDefinition) {
    Block *blk = new Block();
    glob.getInitializerRegion().push_back(blk);
    builder.setInsertionPointToStart(blk);
    builder.create<LLVM::ReturnOp>(
        module->getLoc(),
        std::vector<mlir::Value>(
            {builder.create<LLVM::UndefOp>(module->getLoc(), rt)}));
  }
  return llvmGlobals[name] = glob;
}

std::pair<mlir::memref::GlobalOp, bool>
MLIRASTConsumer::GetOrCreateGlobal(const ValueDecl *FD, std::string prefix,
                                   bool tryInit) {
  std::string name = prefix + CGM.getMangledName(FD).str();

  if (globals.find(name) != globals.end()) {
    return globals[name];
  }

  auto rt = getMLIRType(FD->getType());
  unsigned memspace = 0;
  bool isArray = isa<clang::ArrayType>(FD->getType());

  mlir::MemRefType mr;
  if (!isArray) {
    mr = mlir::MemRefType::get(1, rt, {}, memspace);
  } else {
    auto mt = rt.cast<mlir::MemRefType>();
    mr = mlir::MemRefType::get(
        mt.getShape(), mt.getElementType(), MemRefLayoutAttrInterface(),
        wrapIntegerMemorySpace(memspace, mt.getContext()));
  }

  mlir::SymbolTable::Visibility lnk;
  mlir::Attribute initial_value;

  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());

  if (cast<VarDecl>(FD)->isThisDeclarationADefinition() ==
      VarDecl::Definition) {
    initial_value = builder.getUnitAttr();
  } else if (cast<VarDecl>(FD)->isThisDeclarationADefinition() ==
             VarDecl::TentativeDefinition) {
    initial_value = builder.getUnitAttr();
  }

  switch (CGM.getLLVMLinkageVarDefinition(cast<VarDecl>(FD),
                                          /*isConstant*/ false)) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
    lnk = mlir::SymbolTable::Visibility::Private;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    lnk = mlir::SymbolTable::Visibility::Private;
    break;
  }

  auto globalOp = builder.create<mlir::memref::GlobalOp>(
      module->getLoc(), builder.getStringAttr(name),
      /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(mr),
      initial_value, mlir::UnitAttr(), /*alignment*/ nullptr);
  SymbolTable::setSymbolVisibility(globalOp, lnk);

  globals[name] = std::make_pair(globalOp, isArray);

  if (tryInit)
    if (auto init = cast<VarDecl>(FD)->getInit()) {
      MLIRScanner ms(*this, module, LTInfo);
      mlir::Block *B = new Block();
      ms.setEntryAndAllocBlock(B);
      OpBuilder builder(module->getContext());
      builder.setInsertionPointToEnd(B);
      auto op = builder.create<memref::AllocaOp>(module->getLoc(), mr);

      bool initialized = false;
      if (isa<InitListExpr>(init)) {
        if (auto A = ms.InitializeValueByInitListExpr(
                op, const_cast<clang::Expr *>(init))) {
          initialized = true;
          initial_value = A;
        }
      } else {
        mlir::Value val =
            ms.Visit(const_cast<clang::Expr *>(init)).getValue(builder);
        if (auto cop = val.getDefiningOp<arith::ConstantOp>()) {
          initial_value = cop.getValue();
          initial_value = SplatElementsAttr::get(
              RankedTensorType::get(mr.getShape(), mr.getElementType()),
              initial_value);
          initialized = true;
        }
      }

      if (!initialized) {
        FD->dump();
        init->dump();
        llvm::errs() << " warning not initializing global: " << name << "\n";
      } else {
        globalOp.initial_valueAttr(initial_value);
      }
      delete B;
    }

  return globals[name];
}

mlir::Value MLIRASTConsumer::GetOrCreateGlobalLLVMString(
    mlir::Location loc, mlir::OpBuilder &builder, StringRef value) {
  using namespace mlir;
  // Create the global at the entry of the module.
  if (llvmStringGlobals.find(value.str()) == llvmStringGlobals.end()) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module->getBody());
    auto type = LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
    llvmStringGlobals[value.str()] = builder.create<LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
        "str" + std::to_string(llvmStringGlobals.size()),
        builder.getStringAttr(value.str() + '\0'));
  }

  LLVM::GlobalOp global = llvmStringGlobals[value.str()];
  // Get the pointer to the first character in the global string.
  mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
  return globalPtr;
}

mlir::FuncOp MLIRASTConsumer::GetOrCreateMLIRFunction(const FunctionDecl *FD) {
  assert(FD->getTemplatedKind() !=
         FunctionDecl::TemplatedKind::TK_FunctionTemplate);
  assert(
      FD->getTemplatedKind() !=
      FunctionDecl::TemplatedKind::TK_DependentFunctionTemplateSpecialization);
  std::string name;
  if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
  else if (auto CC = dyn_cast<CXXDestructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
  else
    name = CGM.getMangledName(FD).str();

  assert(name != "free");

  llvm::GlobalValue::LinkageTypes LV;
  if (!FD->hasBody())
    LV = llvm::GlobalValue::LinkageTypes::ExternalLinkage;
  else if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
    LV = CGM.getFunctionLinkage(GlobalDecl(CC, CXXCtorType::Ctor_Complete));
  else if (auto CC = dyn_cast<CXXDestructorDecl>(FD))
    LV = CGM.getFunctionLinkage(GlobalDecl(CC, CXXDtorType::Dtor_Complete));
  else
    LV = CGM.getFunctionLinkage(FD);

  LLVM::Linkage lnk;
  switch (LV) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
    lnk = LLVM::Linkage::Internal;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    lnk = LLVM::Linkage::External;
    break;
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    lnk = LLVM::Linkage::AvailableExternally;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    lnk = LLVM::Linkage::Linkonce;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    lnk = LLVM::Linkage::Weak;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    lnk = LLVM::Linkage::WeakODR;
    break;
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
    lnk = LLVM::Linkage::Common;
    break;
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    lnk = LLVM::Linkage::Appending;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    lnk = LLVM::Linkage::ExternWeak;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    lnk = LLVM::Linkage::LinkonceODR;
    break;
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    lnk = LLVM::Linkage::Private;
    break;
  }

  const FunctionDecl *Def = nullptr;
  if (!FD->isDefined(Def, /*checkforfriend*/ true))
    Def = FD;

  if (functions.find(name) != functions.end()) {
    auto function = functions[name];

    if (Def->isThisDeclarationADefinition()) {
      if (LV == llvm::GlobalValue::InternalLinkage ||
          LV == llvm::GlobalValue::PrivateLinkage || !Def->isDefined() ||
          Def->hasAttr<CUDAGlobalAttr>() || Def->hasAttr<CUDADeviceAttr>()) {
        SymbolTable::setSymbolVisibility(function,
                                         SymbolTable::Visibility::Private);
      } else {
        SymbolTable::setSymbolVisibility(function,
                                         SymbolTable::Visibility::Public);
      }
      mlir::OpBuilder builder(module->getContext());
      NamedAttrList attrs(function->getAttrDictionary());
      attrs.set("llvm.linkage",
                mlir::LLVM::LinkageAttr::get(builder.getContext(), lnk));
      function->setAttrs(attrs.getDictionary(builder.getContext()));
      functionsToEmit.push_back(Def);
    }
    assert(function->getParentOp() == module.get());
    return function;
  }

  std::vector<mlir::Type> types;
  std::vector<std::string> names;

  if (auto CC = dyn_cast<CXXMethodDecl>(FD)) {
    if (CC->isInstance()) {
      auto t = getMLIRType(CC->getThisType());

      bool isArray = false; // isa<clang::ArrayType>(CC->getThisType());
      getMLIRType(CC->getThisObjectType(), &isArray);
      if (auto mt = t.dyn_cast<MemRefType>()) {
        auto shape = std::vector<int64_t>(mt.getShape());
        // shape[0] = 1;
        t = mlir::MemRefType::get(shape, mt.getElementType(),
                                  MemRefLayoutAttrInterface(),
                                  mt.getMemorySpace());
      }
      if (!t.isa<LLVM::LLVMPointerType, MemRefType>()) {
        FD->dump();
        CC->getThisType()->dump();
        llvm::errs() << " t: " << t << " isArray: " << (int)isArray
                     << " LLTy: " << *getLLVMType(CC->getThisType())
                     << " mlirty: " << getMLIRType(CC->getThisType()) << "\n";
      }
      assert(((bool)t.isa<LLVM::LLVMPointerType, MemRefType>()));
      types.push_back(t);
      names.push_back("this");
    }
  }
  for (auto parm : FD->parameters()) {
    bool llvmType = name == "main" && types.size() == 1;
    if (auto ava = parm->getAttr<AlignValueAttr>()) {
      if (auto algn = dyn_cast<clang::ConstantExpr>(ava->getAlignment())) {
        for (auto a : algn->children()) {
          if (auto IL = dyn_cast<IntegerLiteral>(a)) {
            if (IL->getValue() == 8192) {
              llvmType = true;
              break;
            }
          }
        }
      }
    }
    if (llvmType) {
      types.push_back(typeTranslator.translateType(
          anonymize(getLLVMType(parm->getType()))));
    } else {
      bool ArrayStruct = false;
      auto t = getMLIRType(parm->getType(), &ArrayStruct);
      if (ArrayStruct) {
        t = getMLIRType(
            CGM.getContext().getLValueReferenceType(parm->getType()));
      }

      types.push_back(t);
    }
    names.push_back(parm->getName().str());
  }

  bool isArrayReturn = false;
  getMLIRType(FD->getReturnType(), &isArrayReturn);

  std::vector<mlir::Type> rettypes;

  if (isArrayReturn) {
    auto mt = getMLIRType(
                  CGM.getContext().getLValueReferenceType(FD->getReturnType()))
                  .cast<MemRefType>();

    auto shape = std::vector<int64_t>(mt.getShape());
    assert(shape.size() == 2);

    types.push_back(mt);
  } else {
    auto rt = getMLIRType(FD->getReturnType());
    if (!rt.isa<mlir::NoneType>()) {
      rettypes.push_back(rt);
    }
  }
  mlir::OpBuilder builder(module->getContext());
  auto funcType = builder.getFunctionType(types, rettypes);
  mlir::FuncOp function = mlir::FuncOp(
      mlir::FuncOp::create(getMLIRLocation(FD->getLocation()), name, funcType));

  if (LV == llvm::GlobalValue::InternalLinkage ||
      LV == llvm::GlobalValue::PrivateLinkage || !FD->isDefined() ||
      FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
    SymbolTable::setSymbolVisibility(function,
                                     SymbolTable::Visibility::Private);
  } else {
    SymbolTable::setSymbolVisibility(function, SymbolTable::Visibility::Public);
  }
  NamedAttrList attrs(function->getAttrDictionary());
  attrs.set("llvm.linkage",
            mlir::LLVM::LinkageAttr::get(builder.getContext(), lnk));
  function->setAttrs(attrs.getDictionary(builder.getContext()));

  functions[name] = function;
  module->push_back(function);

  if (Def->isThisDeclarationADefinition()) {
    assert(Def->getTemplatedKind() !=
           FunctionDecl::TemplatedKind::TK_FunctionTemplate);
    assert(Def->getTemplatedKind() !=
           FunctionDecl::TemplatedKind::
               TK_DependentFunctionTemplateSpecialization);
    functionsToEmit.push_back(Def);
  } else {
    emitIfFound.insert(name);
  }
  assert(function->getParentOp() == module.get());
  return function;
}

void MLIRASTConsumer::run() {
  while (functionsToEmit.size()) {
    const FunctionDecl *FD = functionsToEmit.front();
    assert(FD->getBody());
    functionsToEmit.pop_front();
    assert(FD->getTemplatedKind() != FunctionDecl::TK_FunctionTemplate);
    assert(FD->getTemplatedKind() !=
           FunctionDecl::TemplatedKind::
               TK_DependentFunctionTemplateSpecialization);
    std::string name;
    if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
    else if (auto CC = dyn_cast<CXXDestructorDecl>(FD))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
    else
      name = CGM.getMangledName(FD).str();

    if (done.count(name))
      continue;
    done.insert(name);
    MLIRScanner ms(*this, module, LTInfo);
    ms.init(GetOrCreateMLIRFunction(FD), FD);
  }
}

void MLIRASTConsumer::HandleDeclContext(DeclContext *DC) {

  for (auto D : DC->decls()) {
    if (auto NS = dyn_cast<clang::NamespaceDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::ExternCContextDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::LinkageSpecDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(D);
    if (!fd) {
      continue;
    }
    if (!fd->doesThisDeclarationHaveABody()) {
      if (!fd->doesDeclarationForceExternallyVisibleDefinition()) {
        continue;
      }
    }
    if (!fd->hasBody())
      continue;

    if (fd->isTemplated()) {
      continue;
    }

    bool externLinkage = true;
    /*
    auto LV = CGM.getFunctionLinkage(fd);
    if (LV == llvm::GlobalValue::InternalLinkage || LV ==
    llvm::GlobalValue::PrivateLinkage) externLinkage = false; if
    (fd->isInlineSpecified()) externLinkage = false;
    */
    if (!CGM.getContext().DeclMustBeEmitted(fd))
      externLinkage = false;

    std::string name;
    if (auto CC = dyn_cast<CXXConstructorDecl>(fd))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
    else if (auto CC = dyn_cast<CXXDestructorDecl>(fd))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
    else
      name = CGM.getMangledName(fd).str();

    // Don't create std functions unless necessary
    if (StringRef(name).startswith("_ZNKSt"))
      continue;
    if (StringRef(name).startswith("_ZSt"))
      continue;
    if (StringRef(name).startswith("_ZNSt"))
      continue;
    if (StringRef(name).startswith("_ZN9__gnu"))
      continue;
    if (name == "cudaGetDevice" || name == "cudaMalloc")
      continue;

    if ((emitIfFound.count("*") && name != "fpclassify" && !fd->isStatic() &&
         externLinkage) ||
        emitIfFound.count(name)) {
      functionsToEmit.push_back(fd);
    } else {
    }
  }
}

bool MLIRASTConsumer::HandleTopLevelDecl(DeclGroupRef dg) {
  DeclGroupRef::iterator it;

  if (error)
    return true;

  for (it = dg.begin(); it != dg.end(); ++it) {
    if (auto NS = dyn_cast<clang::NamespaceDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::ExternCContextDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::LinkageSpecDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(*it);
    if (!fd) {
      continue;
    }
    if (!fd->doesThisDeclarationHaveABody()) {
      if (!fd->doesDeclarationForceExternallyVisibleDefinition()) {
        continue;
      }
    }
    if (!fd->hasBody())
      continue;
    if (fd->isTemplated()) {
      continue;
    }

    bool externLinkage = true;
    /*
    auto LV = CGM.getFunctionLinkage(fd);
    if (LV == llvm::GlobalValue::InternalLinkage || LV ==
    llvm::GlobalValue::PrivateLinkage) externLinkage = false; if
    (fd->isInlineSpecified()) externLinkage = false;
    */
    if (!CGM.getContext().DeclMustBeEmitted(fd))
      externLinkage = false;

    std::string name;
    if (auto CC = dyn_cast<CXXConstructorDecl>(fd))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
    else if (auto CC = dyn_cast<CXXDestructorDecl>(fd))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
    else
      name = CGM.getMangledName(fd).str();

    // Don't create std functions unless necessary
    if (StringRef(name).startswith("_ZNKSt"))
      continue;
    if (StringRef(name).startswith("_ZSt"))
      continue;
    if (StringRef(name).startswith("_ZNSt"))
      continue;
    if (StringRef(name).startswith("_ZN9__gnu"))
      continue;
    if (name == "cudaGetDevice" || name == "cudaMalloc")
      continue;

    if ((emitIfFound.count("*") && name != "fpclassify" && !fd->isStatic() &&
         externLinkage) ||
        emitIfFound.count(name)) {
      functionsToEmit.push_back(fd);
    } else {
    }
  }

  return true;
}

// Wait until Sema has instantiated all the relevant code
// before running codegen on the selected functions.
void MLIRASTConsumer::HandleTranslationUnit(ASTContext &C) { run(); }

mlir::Location MLIRASTConsumer::getMLIRLocation(clang::SourceLocation loc) {
  auto spellingLoc = SM.getSpellingLoc(loc);
  auto lineNumber = SM.getSpellingLineNumber(spellingLoc);
  auto colNumber = SM.getSpellingColumnNumber(spellingLoc);
  auto fileId = SM.getFilename(spellingLoc);

  auto ctx = module->getContext();
  return FileLineColLoc::get(ctx, fileId, lineNumber, colNumber);
}

/// Iteratively get the size of each dim of the given ConstantArrayType inst.
static void getConstantArrayShapeAndElemType(const clang::QualType &ty,
                                             SmallVectorImpl<int64_t> &shape,
                                             clang::QualType &elemTy) {
  shape.clear();

  clang::QualType curTy = ty;
  while (curTy->isConstantArrayType()) {
    auto cstArrTy = cast<clang::ConstantArrayType>(curTy);
    shape.push_back(cstArrTy->getSize().getSExtValue());
    curTy = cstArrTy->getElementType();
  }

  elemTy = curTy;
}

mlir::Type MLIRASTConsumer::getMLIRType(clang::QualType qt, bool *implicitRef,
                                        bool allowMerge) {
  if (auto ET = dyn_cast<clang::ElaboratedType>(qt)) {
    return getMLIRType(ET->getNamedType(), implicitRef, allowMerge);
  }
  if (auto ET = dyn_cast<clang::UsingType>(qt)) {
    return getMLIRType(ET->getUnderlyingType(), implicitRef, allowMerge);
  }
  if (auto ET = dyn_cast<clang::DeducedType>(qt)) {
    return getMLIRType(ET->getDeducedType(), implicitRef, allowMerge);
  }
  if (auto ST = dyn_cast<clang::SubstTemplateTypeParmType>(qt)) {
    return getMLIRType(ST->getReplacementType(), implicitRef, allowMerge);
  }
  if (auto ST = dyn_cast<clang::TemplateSpecializationType>(qt)) {
    return getMLIRType(ST->desugar(), implicitRef, allowMerge);
  }
  if (auto ST = dyn_cast<clang::TypedefType>(qt)) {
    return getMLIRType(ST->desugar(), implicitRef, allowMerge);
  }
  if (auto DT = dyn_cast<clang::DecltypeType>(qt)) {
    return getMLIRType(DT->desugar(), implicitRef, allowMerge);
  }

  if (auto DT = dyn_cast<clang::DecayedType>(qt)) {
    bool assumeRef = false;
    auto mlirty = getMLIRType(DT->getOriginalType(), &assumeRef, allowMerge);
    if (assumeRef) {
      // Constant array types like `int A[30][20]` will be converted to LLVM
      // type `[20 x i32]* %0`, which has the outermost dimension size erased,
      // and we can only recover to `memref<?x20xi32>` from there. This prevents
      // us from doing more comprehensive analysis. Here we specifically handle
      // this case by unwrapping the clang-adjusted type, to get the
      // corresponding ConstantArrayType with the full dimensions.
      if (memRefFullRank) {
        clang::QualType origTy = DT->getOriginalType();
        if (origTy->isConstantArrayType()) {
          SmallVector<int64_t, 4> shape;
          clang::QualType elemTy;
          getConstantArrayShapeAndElemType(origTy, shape, elemTy);

          return mlir::MemRefType::get(shape, getMLIRType(elemTy));
        }
      }

      // If -memref-fullrank is unset or it cannot be fulfilled.
      auto mt = mlirty.dyn_cast<MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2[0] = -1;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    } else {
      return getMLIRType(DT->getAdjustedType(), implicitRef, allowMerge);
    }
    return mlirty;
  }
  if (auto CT = dyn_cast<clang::ComplexType>(qt)) {
    bool assumeRef = false;
    auto subType =
        getMLIRType(CT->getElementType(), &assumeRef, /*allowMerge*/ false);
    if (allowMerge) {
      assert(!assumeRef);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(2, subType);
    }
    mlir::Type types[2] = {subType, subType};
    return mlir::LLVM::LLVMStructType::getLiteral(module->getContext(), types);
  }
  if (auto RT = dyn_cast<clang::RecordType>(qt)) {
    if (RT->getDecl()->isInvalidDecl()) {
      RT->getDecl()->dump();
      RT->dump();
    }
    assert(!RT->getDecl()->isInvalidDecl());
    if (typeCache.find(RT) != typeCache.end())
      return typeCache[RT];
    llvm::Type *LT = CGM.getTypes().ConvertType(qt);
    if (!isa<llvm::StructType>(LT)) {
      qt->dump();
      llvm::errs() << "LT: " << *LT << "\n";
    }
    llvm::StructType *ST = cast<llvm::StructType>(LT);

    if (RT->getDecl()->field_empty())
      if (ST->getNumElements() == 1 && ST->getElementType(0U)->isIntegerTy(8))
        return typeTranslator.translateType(anonymize(ST));

    SmallPtrSet<llvm::Type *, 4> Seen;
    bool notAllSame = false;
    bool recursive = false;
    for (size_t i = 0; i < ST->getNumElements(); i++) {
      if (isRecursiveStruct(ST->getTypeAtIndex(i), ST, Seen)) {
        recursive = true;
      }
      if (ST->getTypeAtIndex(i) != ST->getTypeAtIndex(0U)) {
        notAllSame = true;
      }
    }

    auto CXRD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (RT->getDecl()->isUnion() ||
        (CXRD && (!CXRD->hasDefinition() || CXRD->isPolymorphic() ||
                  CXRD->getDefinition()->getNumBases() > 0)) ||
        (!ST->isLiteral() && (ST->getName() == "struct._IO_FILE" ||
                              ST->getName() == "class.std::basic_ifstream" ||
                              ST->getName() == "class.std::basic_istream" ||
                              ST->getName() == "class.std::basic_ostream" ||
                              ST->getName() == "class.std::basic_ofstream"))) {
      return typeTranslator.translateType(anonymize(ST));
    }

    /* TODO
    if (ST->getNumElements() == 1 && !recursive &&
        !RT->getDecl()->fields().empty() && ++RT->getDecl()->field_begin() ==
    RT->getDecl()->field_end()) { auto subT =
    getMLIRType((*RT->getDecl()->field_begin())->getType(), implicitRef,
    allowMerge); return subT;
    }
    */
    if (recursive)
      typeCache[RT] = LLVM::LLVMStructType::getIdentified(
          module->getContext(), ("polygeist@mlir@" + ST->getName()).str());

    SmallVector<mlir::Type, 4> types;

    bool innerLLVM = false;
    for (auto f : RT->getDecl()->fields()) {
      bool subRef = false;
      auto ty = getMLIRType(f->getType(), &subRef, /*allowMerge*/ false);
      assert(!subRef);
      innerLLVM |= ty.isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                          LLVM::LLVMArrayType>();
      types.push_back(ty);
    }

    if (recursive) {
      auto LR = typeCache[RT].setBody(types, /*isPacked*/ false);
      assert(LR.succeeded());
      return typeCache[RT];
    }

    if (notAllSame || !allowMerge || innerLLVM) {
      return mlir::LLVM::LLVMStructType::getLiteral(module->getContext(),
                                                    types);
    }

    if (!types.size()) {
      RT->dump();
      llvm::errs() << "ST: " << *ST << "\n";
      llvm::errs() << "fields\n";
      for (auto f : RT->getDecl()->fields()) {
        llvm::errs() << " +++ ";
        f->getType()->dump();
        llvm::errs() << " @@@ " << *CGM.getTypes().ConvertType(f->getType())
                     << "\n";
      }
      llvm::errs() << "types\n";
      for (auto t : types)
        llvm::errs() << " --- " << t << "\n";
    }
    assert(types.size());
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get(types.size(), types[0]);
  }

  auto t = qt->getUnqualifiedDesugaredType();
  if (t->isVoidType()) {
    mlir::OpBuilder builder(module->getContext());
    return builder.getNoneType();
  }

  // if (auto AT = dyn_cast<clang::VariableArrayType>(t)) {
  //   return getMLIRType(AT->getElementType(), implicitRef, allowMerge);
  // }

  if (auto AT = dyn_cast<clang::ArrayType>(t)) {
    auto PTT = AT->getElementType()->getUnqualifiedDesugaredType();
    if (PTT->isCharType()) {
      llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
      return typeTranslator.translateType(T);
    }
    bool subRef = false;
    auto ET = getMLIRType(AT->getElementType(), &subRef, allowMerge);
    int64_t size = -1;
    if (auto CAT = dyn_cast<clang::ConstantArrayType>(AT))
      size = CAT->getSize().getZExtValue();
    if (subRef) {
      auto mt = ET.cast<MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2.insert(shape2.begin(), size);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    }
    if (!allowMerge || ET.isa<LLVM::LLVMPointerType, LLVM::LLVMArrayType,
                              LLVM::LLVMFunctionType, LLVM::LLVMStructType>())
      return LLVM::LLVMArrayType::get(ET, (size == -1) ? 0 : size);
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get({size}, ET);
  }

  if (auto AT = dyn_cast<clang::VectorType>(t)) {
    bool subRef = false;
    auto ET = getMLIRType(AT->getElementType(), &subRef, allowMerge);
    int64_t size = AT->getNumElements();
    if (subRef) {
      auto mt = ET.cast<MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2.insert(shape2.begin(), size);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    }
    if (!allowMerge || ET.isa<LLVM::LLVMPointerType, LLVM::LLVMArrayType,
                              LLVM::LLVMFunctionType, LLVM::LLVMStructType>())
      return LLVM::LLVMFixedVectorType::get(ET, size);
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get({size}, ET);
  }

  if (auto FT = dyn_cast<clang::FunctionProtoType>(t)) {
    auto RT = getMLIRType(FT->getReturnType());
    if (RT.isa<mlir::NoneType>())
      RT = LLVM::LLVMVoidType::get(RT.getContext());
    SmallVector<mlir::Type> Args;
    for (auto T : FT->getParamTypes()) {
      Args.push_back(getMLIRType(T));
    }
    return LLVM::LLVMFunctionType::get(RT, Args, FT->isVariadic());
  }

  if (isa<clang::PointerType, clang::ReferenceType>(t)) {
    int64_t outer = (isa<clang::PointerType>(t)) ? -1 : -1;
    auto PTT = isa<clang::PointerType>(t) ? cast<clang::PointerType>(t)
                                                ->getPointeeType()
                                                ->getUnqualifiedDesugaredType()
                                          : cast<clang::ReferenceType>(t)
                                                ->getPointeeType()
                                                ->getUnqualifiedDesugaredType();

    if (PTT->isCharType() || PTT->isVoidType()) {
      llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
      return typeTranslator.translateType(T);
    }
    bool subRef = false;
    auto subType =
        getMLIRType(isa<clang::PointerType>(t)
                        ? cast<clang::PointerType>(t)->getPointeeType()
                        : cast<clang::ReferenceType>(t)->getPointeeType(),
                    &subRef, /*allowMerge*/ true);

    if (subType.isa<LLVM::LLVMArrayType, LLVM::LLVMStructType,
                    LLVM::LLVMPointerType, LLVM::LLVMFunctionType>())
      return LLVM::LLVMPointerType::get(subType);

    if (isa<clang::ArrayType>(PTT)) {
      if (subType.isa<MemRefType>()) {
        assert(subRef);
        return subType;
      } else
        return LLVM::LLVMPointerType::get(subType);
    }

    if (isa<clang::VectorType>(PTT) || isa<clang::ComplexType>(PTT)) {
      if (subType.isa<MemRefType>()) {
        assert(subRef);
        auto mt = subType.cast<MemRefType>();
        auto shape2 = std::vector<int64_t>(mt.getShape());
        shape2.insert(shape2.begin(), outer);
        return mlir::MemRefType::get(shape2, mt.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     mt.getMemorySpace());
      } else
        return LLVM::LLVMPointerType::get(subType);
    }

    if (isa<clang::RecordType>(PTT))
      if (subRef) {
        auto mt = subType.cast<MemRefType>();
        auto shape2 = std::vector<int64_t>(mt.getShape());
        shape2.insert(shape2.begin(), outer);
        return mlir::MemRefType::get(shape2, mt.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     mt.getMemorySpace());
      }

    assert(!subRef);
    return mlir::MemRefType::get({outer}, subType);
  }

  if (t->isBuiltinType() || isa<clang::EnumType>(t)) {
    if (t->isBooleanType()) {
      OpBuilder builder(module->getContext());
      return builder.getIntegerType(8);
    }
    llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
    mlir::OpBuilder builder(module->getContext());
    if (T->isVoidTy()) {
      return builder.getNoneType();
    }
    if (T->isFloatTy()) {
      return builder.getF32Type();
    }
    if (T->isDoubleTy()) {
      return builder.getF64Type();
    }
    if (T->isX86_FP80Ty())
      return builder.getF80Type();
    if (T->isFP128Ty())
      return builder.getF128Type();

    if (auto IT = dyn_cast<llvm::IntegerType>(T)) {
      return builder.getIntegerType(IT->getBitWidth());
    }
  }
  qt->dump();
  assert(0 && "unhandled type");
}

llvm::Type *MLIRASTConsumer::getLLVMType(clang::QualType t) {
  if (t->isVoidType()) {
    return llvm::Type::getVoidTy(llvmMod.getContext());
  }
  llvm::Type *T = CGM.getTypes().ConvertType(t);
  return T;
}

#include "llvm/Support/Host.h"

#include "clang/Frontend/FrontendAction.h"
class MLIRAction : public clang::ASTFrontendAction {
public:
  std::set<std::string> emitIfFound;
  std::set<std::string> done;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> globals;
  std::map<std::string, mlir::FuncOp> functions;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> llvmFunctions;
  MLIRAction(std::string fn, mlir::OwningOpRef<mlir::ModuleOp> &module)
      : module(module) {
    emitIfFound.insert(fn);
  }
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    return std::unique_ptr<clang::ASTConsumer>(new MLIRASTConsumer(
        emitIfFound, done, llvmStringGlobals, globals, functions, llvmGlobals,
        llvmFunctions, CI.getPreprocessor(), CI.getASTContext(), module,
        CI.getSourceManager()));
  }
};

mlir::FuncOp MLIRScanner::EmitDirectCallee(GlobalDecl GD) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());
  return Glob.GetOrCreateMLIRFunction(FD);
}

mlir::Location MLIRScanner::getMLIRLocation(clang::SourceLocation loc) {
  return Glob.getMLIRLocation(loc);
}

mlir::Type MLIRScanner::getMLIRType(clang::QualType t) {
  return Glob.getMLIRType(t);
}

llvm::Type *MLIRScanner::getLLVMType(clang::QualType t) {
  return Glob.getLLVMType(t);
}

size_t MLIRScanner::getTypeSize(clang::QualType t) {
  llvm::Type *T = Glob.CGM.getTypes().ConvertType(t);
  return (Glob.llvmMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
}

#include "clang/Frontend/TextDiagnosticBuffer.h"
static bool parseMLIR(const char *Argv0, std::vector<std::string> filenames,
                      std::string fn, std::vector<std::string> includeDirs,
                      std::vector<std::string> defines,
                      mlir::OwningOpRef<mlir::ModuleOp> &module,
                      llvm::Triple &triple, llvm::DataLayout &DL) {

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  bool Success;
  //{
  const char *binary = Argv0; // CudaLower ? "clang++" : "clang";
  const unique_ptr<Driver> driver(
      new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags));
  std::vector<const char *> Argv;
  Argv.push_back(binary);
  for (auto a : filenames) {
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (FOpenMP)
    Argv.push_back("-fopenmp");
  if (Standard != "") {
    auto a = "-std=" + Standard;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    char *chars = (char *)malloc(ResourceDir.length() + 1);
    memcpy(chars, ResourceDir.data(), ResourceDir.length());
    chars[ResourceDir.length()] = 0;
    Argv.push_back(chars);
  }
  if (Verbose) {
    Argv.push_back("-v");
  }
  if (CUDAGPUArch != "") {
    auto a = "--cuda-gpu-arch=" + CUDAGPUArch;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (CUDAPath != "") {
    auto a = "--cuda-path=" + CUDAPath;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (MArch != "") {
    auto a = "-march=" + MArch;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  for (auto a : includeDirs) {
    Argv.push_back("-I");
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  for (auto a : defines) {
    char *chars = (char *)malloc(a.length() + 3);
    chars[0] = '-';
    chars[1] = 'D';
    memcpy(chars + 2, a.data(), a.length());
    chars[2 + a.length()] = 0;
    Argv.push_back(chars);
  }
  for (auto a : Includes) {
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back("-include");
    Argv.push_back(chars);
  }

  Argv.push_back("-emit-ast");

  const unique_ptr<Compilation> compilation(
      driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv)));
  JobList &Jobs = compilation->getJobs();
  if (Jobs.size() < 1)
    return false;

  MLIRAction Act(fn, module);

  for (auto &job : Jobs) {
    std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

    Command *cmd = cast<Command>(&job);
    if (strcmp(cmd->getCreator().getName(), "clang"))
      return false;

    const ArgStringList *args = &cmd->getArguments();

    Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(), *args,
                                                 Diags);
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    void *GetExecutablePathVP = (void *)(intptr_t)GetExecutablePath;
    // Infer the builtin include path if unspecified.
    if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
        Clang->getHeaderSearchOpts().ResourceDir.size() == 0)
      Clang->getHeaderSearchOpts().ResourceDir =
          CompilerInvocation::GetResourcesPath(Argv0, GetExecutablePathVP);

    //}
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    // Create the actual diagnostics engine.
    Clang->createDiagnostics();
    if (!Clang->hasDiagnostics())
      return false;

    DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
    if (!Success)
      return false;

    // Create and execute the frontend action.

    // Create the target instance.
    Clang->setTarget(TargetInfo::CreateTargetInfo(
        Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
    if (!Clang->hasTarget())
      return false;

    // Create TargetInfo for the other side of CUDA and OpenMP compilation.
    if ((Clang->getLangOpts().CUDA || Clang->getLangOpts().OpenMPIsDevice) &&
        !Clang->getFrontendOpts().AuxTriple.empty()) {
      auto TO = std::make_shared<clang::TargetOptions>();
      TO->Triple = llvm::Triple::normalize(Clang->getFrontendOpts().AuxTriple);
      TO->HostTriple = Clang->getTarget().getTriple().str();
      Clang->setAuxTarget(
          TargetInfo::CreateTargetInfo(Clang->getDiagnostics(), TO));
    }

    // Inform the target of the language options.
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    Clang->getTarget().adjust(Clang->getDiagnostics(), Clang->getLangOpts());

    // Adjust target options based on codegen options.
    Clang->getTarget().adjustTargetOptions(Clang->getCodeGenOpts(),
                                           Clang->getTargetOpts());

    module.get()->setAttr(
        LLVM::LLVMDialect::getDataLayoutAttrName(),
        StringAttr::get(module->getContext(),
                        Clang->getTarget().getDataLayoutString()));
    module.get()->setAttr(
        LLVM::LLVMDialect::getTargetTripleAttrName(),
        StringAttr::get(module->getContext(),
                        Clang->getTarget().getTriple().getTriple()));

    for (const auto &FIF : Clang->getFrontendOpts().Inputs) {
      // Reset the ID tables if we are reusing the SourceManager and parsing
      // regular files.
      if (Clang->hasSourceManager() && !Act.isModelParsingAction())
        Clang->getSourceManager().clearIDTables();
      if (Act.BeginSourceFile(*Clang, FIF)) {

        llvm::Error err = Act.Execute();
        if (err) {
          llvm::errs() << "saw error: " << err << "\n";
          return false;
        }
        assert(Clang->hasSourceManager());

        Act.EndSourceFile();
      }
    }
    DL = llvm::DataLayout(Clang->getTarget().getDataLayoutString());
    triple = Clang->getTarget().getTriple();
  }
  return true;
}

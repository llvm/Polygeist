#include "clang-mlir.h"
#include "utils.h"
#include "llvm/Support/Debug.h"
#include <clang/AST/Decl.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/FileSystemOptions.h>
#include <clang/Basic/LangStandard.h>
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

using namespace std;
using namespace clang;
using namespace clang::driver;
using namespace llvm::opt;
using namespace mlir;

#define DEBUG_TYPE "clang-mlir"

class IfScope {
public:
  MLIRScanner &scanner;
  IfScope(MLIRScanner &scanner) : scanner(scanner) { scanner.pushLoopIf(); }
  ~IfScope() { scanner.popLoopIf(); }
};

mlir::Value MLIRScanner::createAllocOp(mlir::Type t, VarDecl *name,
                                       uint64_t memspace, bool isArray = false,
                                       bool LLVMABI = false) {

  mlir::MemRefType mr;
  mlir::Value alloc = nullptr;
  OpBuilder abuilder(builder.getContext());
  abuilder.setInsertionPointToStart(allocationScope);
  if (!isArray) {
    if (LLVMABI) {
      auto iTy = builder.getIntegerType(64);
      alloc = abuilder.create<mlir::LLVM::AllocaOp>(
          loc, mlir::LLVM::LLVMPointerType::get(t, memspace),
          builder.create<mlir::ConstantOp>(loc, iTy,
                                           builder.getIntegerAttr(iTy, 1)),
          0);
    } else {
      mr = mlir::MemRefType::get(1, t, {}, memspace);
      alloc = abuilder.create<mlir::memref::AllocaOp>(loc, mr);
      alloc = abuilder.create<mlir::memref::CastOp>(
          loc, alloc, mlir::MemRefType::get(-1, t, {}, memspace));
    }
  } else {
    auto mt = t.cast<mlir::MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    auto pshape = shape[0];

    if (name)
      if (auto var = dyn_cast<VariableArrayType>(
              name->getType()->getUnqualifiedDesugaredType())) {
        assert(shape[0] == -1);
        mr = mlir::MemRefType::get(shape, mt.getElementType(),
                                   mt.getAffineMaps(), memspace);
        auto len = Visit(var->getSizeExpr()).getValue(builder);
        len = builder.create<IndexCastOp>(loc, len, builder.getIndexType());
        alloc = builder.create<mlir::memref::AllocaOp>(loc, mr, len);
        builder.create<polygeist::TrivialUseOp>(loc, alloc);
      }

    if (!alloc) {
      if (pshape == -1)
        shape[0] = 1;
      mr = mlir::MemRefType::get(shape, mt.getElementType(), mt.getAffineMaps(),
                                 memspace);
      alloc = abuilder.create<mlir::memref::AllocaOp>(loc, mr);
      shape[0] = pshape;
      alloc = abuilder.create<mlir::memref::CastOp>(
          loc, alloc,
          mlir::MemRefType::get(shape, mt.getElementType(), mt.getAffineMaps(),
                                memspace));
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
    params[name] = ValueWithOffsets(alloc, /*isReference*/ true);
  }
  return alloc;
}

ValueWithOffsets MLIRScanner::VisitDeclStmt(clang::DeclStmt *decl) {
  IfScope scope(*this);
  for (auto sub : decl->decls()) {
    if (auto vd = dyn_cast<VarDecl>(sub)) {
      VisitVarDecl(vd);
    } else if (isa<TypeAliasDecl, RecordDecl, StaticAssertDecl, TypedefDecl,
                   UsingDecl>(sub)) {
    } else {
      llvm::errs() << " + visiting unknonwn sub decl stmt\n";
      sub->dump();
      assert(0 && "unknown sub decl");
    }
  }
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitConstantExpr(clang::ConstantExpr *expr) {
  if (expr->hasAPValueResult()) {
    auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    return ValueWithOffsets(
        builder.create<mlir::ConstantOp>(
            loc, ty,
            builder.getIntegerAttr(ty, expr->getAPValueResult().getInt())),
        /*isReference*/ false);
  }
  return Visit(expr->getSubExpr());
}
ValueWithOffsets MLIRScanner::VisitIntegerLiteral(clang::IntegerLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueWithOffsets(
      builder.create<mlir::ConstantOp>(
          loc, ty, builder.getIntegerAttr(ty, expr->getValue())),
      /*isReference*/ false);
}
ValueWithOffsets
MLIRScanner::VisitCharacterLiteral(clang::CharacterLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueWithOffsets(
      builder.create<mlir::ConstantOp>(
          loc, ty, builder.getIntegerAttr(ty, expr->getValue())),
      /*isReference*/ false);
}

ValueWithOffsets
MLIRScanner::VisitFloatingLiteral(clang::FloatingLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::FloatType>();
  return ValueWithOffsets(
      builder.create<mlir::ConstantOp>(
          loc, ty, builder.getFloatAttr(ty, expr->getValue())),
      /*isReference*/ false);
}

ValueWithOffsets
MLIRScanner::VisitImaginaryLiteral(clang::ImaginaryLiteral *expr) {
  auto mt = getMLIRType(expr->getType()).cast<MemRefType>();
  auto ty = mt.getElementType();

  OpBuilder abuilder(builder.getContext());
  abuilder.setInsertionPointToStart(allocationScope);
  auto alloc = abuilder.create<mlir::memref::AllocaOp>(loc, mt);
  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::ConstantOp>(loc, ty, builder.getFloatAttr(ty, 0.0)),
      alloc, getConstantIndex(0));
  builder.create<mlir::memref::StoreOp>(
      loc, Visit(expr->getSubExpr()).getValue(builder), alloc,
      getConstantIndex(1));
  return ValueWithOffsets(alloc,
                          /*isReference*/ true);
}

ValueWithOffsets
MLIRScanner::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueWithOffsets(
      builder.create<mlir::ConstantOp>(
          loc, ty, builder.getIntegerAttr(ty, expr->getValue())),
      /*isReference*/ false);
}

ValueWithOffsets MLIRScanner::VisitStringLiteral(clang::StringLiteral *expr) {
  auto ty = getMLIRType(expr->getType());
  return ValueWithOffsets(builder.create<mlir::ConstantOp>(
                              loc, ty, builder.getStringAttr(expr->getBytes())),
                          /*isReference*/ true);
}

ValueWithOffsets MLIRScanner::VisitParenExpr(clang::ParenExpr *expr) {
  return Visit(expr->getSubExpr());
}

ValueWithOffsets
MLIRScanner::VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *decl) {
  auto Mty = getMLIRType(decl->getType());

  if (Mty.isa<mlir::FloatType>())
    return ValueWithOffsets(builder.create<mlir::ConstantOp>(
                                loc, Mty, builder.getFloatAttr(Mty, 0.0)),
                            /*isReference*/ false);
  ;
  for (auto child : decl->children()) {
    child->dump();
  }
  decl->dump();
  llvm::errs() << " mty: " << Mty << "\n";
  assert(0 && "bad");
}

#include "clang/AST/Attr.h"
ValueWithOffsets MLIRScanner::VisitVarDecl(clang::VarDecl *decl) {
  auto loc = getMLIRLocation(decl->getLocation());
  unsigned memtype = 0;

  if (decl->hasAttr<CUDASharedAttr>()) {
    memtype = 5;
  }
  mlir::Type subType = getMLIRType(decl->getType());
  ValueWithOffsets inite = nullptr;
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
    isArray = true;
  }

  if (auto init = decl->getInit()) {
    if (auto CE = dyn_cast<CXXConstructExpr>(init)) {
      return VisitConstructCommon(CE, decl, memtype);
    }
    if (!isa<InitListExpr>(init)) {
      auto visit = Visit(init);
      if (!visit.val) {
        decl->dump();
      }
      bool isReference = init->isLValue() || init->isXValue();
      if (isReference) {
        assert(visit.isReference);
        return params[decl] = visit;
      }
      if (isArray) {
        assert(visit.isReference);
        inite = visit;
      } else {
        inite = ValueWithOffsets(visit.getValue(builder), /*isRef*/ false);
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

  auto op = createAllocOp(subType, decl, memtype, isArray, LLVMABI);
  if (inite.val) {
    ValueWithOffsets(op, /*isReference*/ true).store(builder, inite, isArray);
  } else if (auto init = decl->getInit()) {
    if (auto il = dyn_cast<InitListExpr>(init)) {
      if (il->hasArrayFiller()) {
        auto visit = Visit(il->getInit(0));
        mlir::Value inite = visit.getValue(builder);
        if (!inite) {
          il->getArrayFiller()->dump();
          assert(inite);
        }
        assert(subType.cast<MemRefType>().getShape().size() == 1);
        for (ssize_t i = 0; i < subType.cast<MemRefType>().getShape()[0]; i++) {
          assert(inite.getType() ==
                 op.getType().cast<MemRefType>().getElementType());
          assert(op.getType().cast<MemRefType>().getShape().size() == 1);
          builder.create<mlir::memref::StoreOp>(loc, inite, op,
                                                getConstantIndex(i));
        }
      } else {
        init->dump();
        assert(0 && "init list expr unhandled");
      }
    } else
      assert(0 && "unknown init list");
  }
  return ValueWithOffsets(op, /*isReference*/ true);
}

ValueWithOffsets
MLIRScanner::VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr) {
  return Visit(expr->getExpr());
}

ValueWithOffsets MLIRScanner::VisitCXXThisExpr(clang::CXXThisExpr *expr) {
  return ThisVal;
}

ValueWithOffsets MLIRScanner::VisitPredefinedExpr(clang::PredefinedExpr *expr) {
  return VisitStringLiteral(expr->getFunctionName());
}

ValueWithOffsets MLIRScanner::VisitInitListExpr(clang::InitListExpr *expr) {
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
  if (expr->hasArrayFiller()) {
    auto visit = Visit(expr->getInit(0));
    mlir::Value inite = visit.getValue(builder);
    if (!inite) {
      expr->getArrayFiller()->dump();
      assert(inite);
    }
    assert(subType.cast<MemRefType>().getShape().size() == 1);
    for (ssize_t i = 0; i < subType.cast<MemRefType>().getShape()[0]; i++) {
      assert(inite.getType() ==
             op.getType().cast<MemRefType>().getElementType());
      assert(op.getType().cast<MemRefType>().getShape().size() == 1);
      builder.create<mlir::memref::StoreOp>(loc, inite, op,
                                            getConstantIndex(i));
    }
    return ValueWithOffsets(op, true);
  }
  if (expr->getNumInits()) {
    assert(op.getType().cast<MemRefType>().getShape().size() == 1);
    assert(op.getType().cast<MemRefType>().getShape()[0] ==
           expr->getNumInits());
    for (size_t i = 0; i < expr->getNumInits(); ++i) {
      auto inite = Visit(expr->getInit(i)).getValue(builder);
      assert(inite.getType() ==
             op.getType().cast<MemRefType>().getElementType());
      builder.create<mlir::memref::StoreOp>(loc, inite, op,
                                            getConstantIndex(i));
    }
  }
  return ValueWithOffsets(op, true);
}

ValueWithOffsets
MLIRScanner::VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *expr) {
  assert(arrayinit.size());
  return ValueWithOffsets(
      builder.create<IndexCastOp>(loc, getMLIRType(expr->getType()),
                                  arrayinit.back()),
      /*isReference*/ false);
}

ValueWithOffsets MLIRScanner::VisitArrayInitLoop(clang::ArrayInitLoopExpr *expr,
                                                 ValueWithOffsets tostore) {
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

  auto alu = CommonArrayLookup(CommonArrayToPointer(tostore),
                               affineOp.getInductionVar());

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

ValueWithOffsets
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
    auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    if (scalar.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return ValueWithOffsets(
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
      return ValueWithOffsets(scalar, /*isReference*/ false);
    if (prevTy.getWidth() < postTy.getWidth()) {
      if (signedType) {
        return ValueWithOffsets(
            builder.create<mlir::SignExtendIOp>(loc, scalar, postTy),
            /*isReference*/ false);
      } else {
        return ValueWithOffsets(
            builder.create<mlir::ZeroExtendIOp>(loc, scalar, postTy),
            /*isReference*/ false);
      }
    } else {
      return ValueWithOffsets(
          builder.create<mlir::TruncateIOp>(loc, scalar, postTy),
          /*isReference*/ false);
    }
  }
  expr->dump();
  assert(0 && "unhandled functional cast type");
}

ValueWithOffsets
MLIRScanner::VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr) {
  return Visit(expr->getSubExpr());
}

ValueWithOffsets MLIRScanner::VisitLambdaExpr(clang::LambdaExpr *expr) {

  llvm::DenseMap<const VarDecl *, FieldDecl *> Captures;
  FieldDecl *ThisCapture = nullptr;

  expr->getLambdaClass()->getCaptureFields(Captures, ThisCapture);

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
    t = mlir::MemRefType::get(shape, mt.getElementType(), mt.getAffineMaps(),
                              mt.getMemorySpace());
  }
  auto op = createAllocOp(t, nullptr, /*memtype*/ 0, isArray, LLVMABI);

  llvm::DenseMap<const VarDecl *, LambdaCaptureKind> CaptureKinds;
  for (auto C : expr->getLambdaClass()->captures()) {
    if (C.capturesVariable()) {
      CaptureKinds[C.getCapturedVar()] = C.getCaptureKind();
    }
  }

  for (auto pair : Captures) {
    assert(params.find(pair.first) != params.end());
    assert(CaptureKinds.find(pair.first) != CaptureKinds.end());

    bool isArray = false;
    Glob.getMLIRType(pair.second->getType(), &isArray);

    if (CaptureKinds[pair.first] == LambdaCaptureKind::LCK_ByCopy)
      CommonFieldLookup(expr->getCallOperator()->getThisObjectType(),
                        pair.second, op)
          .store(builder, params[pair.first], isArray);
    else {
      assert(CaptureKinds[pair.first] == LambdaCaptureKind::LCK_ByRef);
      assert(params[pair.first].isReference);
      auto mt = params[pair.first].val.getType().cast<MemRefType>();
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;

      CommonFieldLookup(expr->getCallOperator()->getThisObjectType(),
                        pair.second, op)
          .store(builder,
                 builder.create<memref::CastOp>(
                     loc,
                     MemRefType::get(shape, mt.getElementType(),
                                     mt.getAffineMaps(), mt.getMemorySpace()),
                     params[pair.first].val));
    }
  }
  return ValueWithOffsets(op, /*isReference*/ true);
}

// TODO actually deallocate
ValueWithOffsets MLIRScanner::VisitMaterializeTemporaryExpr(
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

  ValueWithOffsets(op, /*isRefererence*/ true).store(builder, v, isArray);
  return ValueWithOffsets(op, /*isRefererence*/ true);
}

ValueWithOffsets MLIRScanner::VisitCXXNewExpr(clang::CXXNewExpr *expr) {
  assert(!expr->getConstructExpr());
  assert(expr->isArray());
  // assert(expr->isGlobalNew());
  auto v = Visit(*expr->raw_arg_begin());
  assert(v.val);

  auto mt = getMLIRType(expr->getType()).cast<mlir::MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());

  auto allocSize = builder.create<mlir::IndexCastOp>(
      loc, v.getValue(builder), mlir::IndexType::get(builder.getContext()));
  mlir::Value args[1] = {allocSize};
  auto alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
  return ValueWithOffsets(alloc, /*isRefererence*/ false);
}

bool MLIRScanner::getLowerBound(clang::ForStmt *fors,
                                AffineLoopDescriptor &descr) {
  auto init = fors->getInit();
  if (auto declStmt = dyn_cast<DeclStmt>(init))
    if (declStmt->isSingleDecl()) {
      auto decl = declStmt->getSingleDecl();
      if (auto varDecl = dyn_cast<VarDecl>(decl)) {
        if (varDecl->hasInit()) {
          mlir::Value val = VisitVarDecl(varDecl).getValue(builder);
          descr.setName(varDecl);
          descr.setType(val.getType());
          LLVM_DEBUG(descr.getType().print(llvm::dbgs()));

          if (descr.getForwardMode())
            descr.setLowerBound(val);
          else {
            val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
            descr.setUpperBound(val);
          }
          return true;
        }
      }
    }

  // BinaryOperator 0x7ff7aa17e938 'int' '='
  // |-DeclRefExpr 0x7ff7aa17e8f8 'int' lvalue Var 0x7ff7aa17e758 'i' 'int'
  // -IntegerLiteral 0x7ff7aa17e918 'int' 0
  if (auto binOp = dyn_cast<clang::BinaryOperator>(init))
    if (binOp->getOpcode() == clang::BinaryOperator::Opcode::BO_Assign)
      if (auto declRefStmt = dyn_cast<DeclRefExpr>(binOp->getLHS())) {
        mlir::Value val = Visit(binOp->getRHS()).getValue(builder);
        val = builder.create<mlir::IndexCastOp>(
            loc, val, mlir::IndexType::get(builder.getContext()));
        descr.setName(cast<VarDecl>(declRefStmt->getDecl()));
        descr.setType(getMLIRType(declRefStmt->getDecl()->getType()));
        if (descr.getForwardMode())
          descr.setLowerBound(val);
        else {
          val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
          descr.setUpperBound(val);
        }
        return true;
      }
  return false;
}

// Make sure that the induction variable initialized in
// the for is the same as the one used in the condition.
bool matchIndvar(const Expr *expr, VarDecl *indVar) {
  while (auto IC = dyn_cast<ImplicitCastExpr>(expr)) {
    expr = IC->getSubExpr();
  }
  if (auto declRef = dyn_cast<DeclRefExpr>(expr)) {
    auto declRefName = declRef->getDecl();
    if (declRefName == indVar)
      return true;
  }
  return false;
}

bool MLIRScanner::getUpperBound(clang::ForStmt *fors,
                                AffineLoopDescriptor &descr) {
  auto cond = fors->getCond();
  if (auto binaryOp = dyn_cast<clang::BinaryOperator>(cond)) {
    auto lhs = binaryOp->getLHS();
    if (!matchIndvar(lhs, descr.getName()))
      return false;

    if (descr.getForwardMode()) {
      if (binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_LT &&
          binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_LE)
        return false;

      auto rhs = binaryOp->getRHS();
      mlir::Value val = Visit(rhs).getValue(builder);
      val = builder.create<IndexCastOp>(loc, val,
                                        mlir::IndexType::get(val.getContext()));
      if (binaryOp->getOpcode() == clang::BinaryOperator::Opcode::BO_LE)
        val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
      descr.setUpperBound(val);
      return true;
    } else {
      if (binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_GT &&
          binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_GE)
        return false;

      auto rhs = binaryOp->getRHS();
      mlir::Value val = Visit(rhs).getValue(builder);
      val = builder.create<IndexCastOp>(loc, val,
                                        mlir::IndexType::get(val.getContext()));
      if (binaryOp->getOpcode() == clang::BinaryOperator::Opcode::BO_GT)
        val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
      descr.setLowerBound(val);
      return true;
    }
  }
  return false;
}

bool MLIRScanner::getConstantStep(clang::ForStmt *fors,
                                  AffineLoopDescriptor &descr) {
  auto inc = fors->getInc();
  if (auto unaryOp = dyn_cast<clang::UnaryOperator>(inc))
    if (unaryOp->isPrefix() || unaryOp->isPostfix()) {
      bool forwardLoop =
          unaryOp->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc ||
          unaryOp->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc;
      descr.setStep(1);
      descr.setForwardMode(forwardLoop);
      return true;
    }
  return false;
}

bool MLIRScanner::isTrivialAffineLoop(clang::ForStmt *fors,
                                      AffineLoopDescriptor &descr) {
  if (!getConstantStep(fors, descr)) {
    LLVM_DEBUG(dbgs() << "getConstantStep -> false\n");
    return false;
  }
  if (!getLowerBound(fors, descr)) {
    LLVM_DEBUG(dbgs() << "getLowerBound -> false\n");
    return false;
  }
  if (!getUpperBound(fors, descr)) {
    LLVM_DEBUG(dbgs() << "getUpperBound -> false\n");
    return false;
  }
  LLVM_DEBUG(dbgs() << "isTrivialAffineLoop -> true\n");
  return true;
}

void MLIRScanner::buildAffineLoopImpl(clang::ForStmt *fors, mlir::Location loc,
                                      mlir::Value lb, mlir::Value ub,
                                      const AffineLoopDescriptor &descr) {
  auto affineOp = builder.create<AffineForOp>(
      loc, lb, builder.getSymbolIdentityMap(), ub,
      builder.getSymbolIdentityMap(), descr.getStep(),
      /*iterArgs=*/llvm::None);

  auto &reg = affineOp.getLoopBody();

  auto val = (mlir::Value)affineOp.getInductionVar();

  reg.front().clear();

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToEnd(&reg.front());

  auto er = builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  er.region().push_back(new Block());
  builder.setInsertionPointToStart(&er.region().back());
  builder.create<scf::YieldOp>(loc);
  builder.setInsertionPointToStart(&er.region().back());

  if (!descr.getForwardMode()) {
    val = builder.create<mlir::SubIOp>(loc, val, lb);
    val = builder.create<mlir::SubIOp>(
        loc, builder.create<mlir::SubIOp>(loc, ub, getConstantIndex(1)), val);
  }
  auto idx = builder.create<mlir::IndexCastOp>(loc, val, descr.getType());
  assert(params.find(descr.getName()) != params.end());
  params[descr.getName()].store(builder, idx);

  // TODO: set loop context.
  Visit(fors->getBody());

  builder.setInsertionPointToEnd(&reg.front());
  builder.create<AffineYieldOp>(loc);

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(oldblock, oldpoint);
}

static bool isTerminator(Operation *op) {
  return op->mightHaveTrait<OpTrait::IsTerminator>();
}

void MLIRScanner::buildAffineLoop(clang::ForStmt *fors, mlir::Location loc,
                                  const AffineLoopDescriptor &descr) {
  mlir::Value lb = descr.getLowerBound();
  mlir::Value ub = descr.getUpperBound();
  buildAffineLoopImpl(fors, loc, lb, ub, descr);
  return;
}

ValueWithOffsets MLIRScanner::VisitWhileStmt(clang::WhileStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getLParenLoc());

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<mlir::ConstantOp>(
      loc, i1Ty, builder.getIntegerAttr(i1Ty, 1));
  loops.push_back(
      (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                    builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);

  auto toadd = builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  builder.create<mlir::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&condB);

  if (auto s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      ty = builder.getIntegerType(1);
      cond = builder.create<mlir::TruncateIOp>(loc, cond, ty);
    }
    auto nb = builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                                   std::vector<mlir::Value>());
    cond = builder.create<mlir::AndOp>(loc, cond, nb);
    builder.create<mlir::CondBranchOp>(loc, cond, &bodyB, &exitB);
  }

  builder.setInsertionPointToStart(&bodyB);
  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                           std::vector<mlir::Value>()),
      loops.back().keepRunning, std::vector<mlir::Value>());

  Visit(fors->getBody());
  loops.pop_back();
  // if (builder.getInsertionBlock()->empty() ||
  //    builder.getInsertionBlock()->back().isKnownNonTerminator()) {
  builder.create<mlir::BranchOp>(loc, &condB);
  //}

  builder.setInsertionPointToStart(&exitB);

  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitDoStmt(clang::DoStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getDoLoc());

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<mlir::ConstantOp>(
      loc, i1Ty, builder.getIntegerAttr(i1Ty, 1));
  loops.push_back(
      (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                    builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);

  auto toadd = builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  builder.create<mlir::BranchOp>(loc, &bodyB);

  builder.setInsertionPointToStart(&condB);

  if (auto s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      ty = builder.getIntegerType(1);
      cond = builder.create<mlir::TruncateIOp>(loc, cond, ty);
    }
    auto nb = builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                                   std::vector<mlir::Value>());
    cond = builder.create<mlir::AndOp>(loc, cond, nb);
    builder.create<mlir::CondBranchOp>(loc, cond, &bodyB, &exitB);
  }

  builder.setInsertionPointToStart(&bodyB);
  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                           std::vector<mlir::Value>()),
      loops.back().keepRunning, std::vector<mlir::Value>());

  Visit(fors->getBody());
  loops.pop_back();
  // if (builder.getInsertionBlock()->empty() ||
  //    builder.getInsertionBlock()->back().isKnownNonTerminator()) {
  builder.create<mlir::BranchOp>(loc, &condB);
  //}

  builder.setInsertionPointToStart(&exitB);

  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitOMPParallelForDirective(
    clang::OMPParallelForDirective *fors) {
  IfScope scope(*this);
  fors->dump();

  Visit(fors->getPreInits());

  SmallVector<mlir::Value> inits;
  for (auto f : fors->inits()) {
    llvm::errs() << " init: ";
    f->dump();
    llvm::errs() << "\n";
    inits.push_back(builder.create<IndexCastOp>(loc, Visit(f).getValue(builder),
                                                builder.getIndexType()));
  }
  inits.clear();
  inits.push_back(getConstantIndex(0));

  SmallVector<mlir::Value> finals;
  for (auto f : fors->finals()) {
    llvm::errs() << " final: ";
    f->dump();
    llvm::errs() << "\n";
    finals.push_back(builder.create<IndexCastOp>(
        loc, Visit(f).getValue(builder), builder.getIndexType()));
  }
  finals.clear();
  finals.push_back(builder.create<IndexCastOp>(
      loc, Visit(fors->getNumIterations()).getValue(builder),
      builder.getIndexType()));

  SmallVector<mlir::Value> counters;
  mlir::Value incs[] = {getConstantIndex(1)};

  for (auto m : fors->getInnermostCapturedStmt()->captures()) {
    llvm::errs() << " cap: ";
    m.getCapturedVar()->dump();
    llvm::errs() << "\n";
    /*
    if (m->getCaptureKind() == LambdaCaptureKind::LCK_ByCopy)
      CommonFieldLookup(expr->getCallOperator()->getThisObjectType(),
                        pair.second, op)
          .store(builder, params[pair.first], isArray);
    else {
      assert(m->getCaptureKind() == LambdaCaptureKind::LCK_ByRef);
      assert(params[pair.first].isReference);
      auto mt = params[pair.first].val.getType().cast<MemRefType>();
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
    }
    */
  }

  for (auto f : fors->counters()) {
    llvm::errs() << " counter: ";
    f->dump();
    llvm::errs() << "\n";
    cast<DeclRefExpr>(f)->getDecl()->dump();
    // counters.push_back(builder.create<IndexCastOp>(loc,
    // Visit(f).getValue(builder), builder.getIndexType()));
  }

  llvm::errs() << " assoc:";
  fors->getAssociatedStmt()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " sblock:";
  fors->getStructuredBlock()->dump();
  llvm::errs() << "\n";

  llvm::errs() << " preinit: ";
  fors->getPreInits()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " init: ";
  fors->getInit()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " lb: ";
  fors->getLowerBoundVariable()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " ub: ";
  fors->getUpperBoundVariable()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " precond: ";
  fors->getPreCond()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " cond: ";
  fors->getCond()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " inc: ";
  fors->getInc()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " last: ";
  fors->getLastIteration()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " stride: ";
  fors->getStrideVariable()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " iters: ";
  fors->getNumIterations()->dump();
  llvm::errs() << "\n";
  llvm::errs() << " body: ";
  fors->getBody()->dump();
  llvm::errs() << "\n";

  auto affineOp = builder.create<scf::ParallelOp>(loc, inits, finals, incs);

  fors->getIterationVariable()->dump();

  auto inds = affineOp.getInductionVars();

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&affineOp.region().front());

  auto er = builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  er.region().push_back(new Block());
  builder.setInsertionPointToStart(&er.region().back());
  builder.create<scf::YieldOp>(loc);
  builder.setInsertionPointToStart(&er.region().back());

  auto idx = builder.create<mlir::IndexCastOp>(
      loc, inds[0], getMLIRType(fors->getIterationVariable()->getType()));
  VarDecl *name =
      cast<VarDecl>(cast<DeclRefExpr>(fors->counters()[0])->getDecl());
  assert(params.find(name) == params.end());
  params[name] = ValueWithOffsets(idx, false);

  // TODO: set loop context.
  Visit(fors->getBody());

  builder.create<scf::YieldOp>(loc);

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitForStmt(clang::ForStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getForLoc());

  AffineLoopDescriptor affineLoopDescr;
  if (Glob.scopLocList.isInScop(fors->getForLoc()) &&
      isTrivialAffineLoop(fors, affineLoopDescr)) {
    buildAffineLoop(fors, loc, affineLoopDescr);
  } else {

    if (auto s = fors->getInit()) {
      Visit(s);
    }

    auto i1Ty = builder.getIntegerType(1);
    auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
    auto truev = builder.create<mlir::ConstantOp>(
        loc, i1Ty, builder.getIntegerAttr(i1Ty, 1));
    loops.push_back(
        (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                      builder.create<mlir::memref::AllocaOp>(loc, type)});
    builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);

    auto toadd = builder.getInsertionBlock()->getParent();
    auto &condB = *(new Block());
    toadd->getBlocks().push_back(&condB);
    auto &bodyB = *(new Block());
    toadd->getBlocks().push_back(&bodyB);
    auto &exitB = *(new Block());
    toadd->getBlocks().push_back(&exitB);

    builder.create<mlir::BranchOp>(loc, &condB);

    builder.setInsertionPointToStart(&condB);

    if (auto s = fors->getCond()) {
      auto condRes = Visit(s);
      auto cond = condRes.getValue(builder);
      if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
        auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
        cond = builder.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
      }
      auto ty = cond.getType().cast<mlir::IntegerType>();
      if (ty.getWidth() != 1) {
        ty = builder.getIntegerType(1);
        cond = builder.create<mlir::TruncateIOp>(loc, cond, ty);
      }
      auto nb = builder.create<mlir::memref::LoadOp>(
          loc, loops.back().noBreak, std::vector<mlir::Value>());
      cond = builder.create<mlir::AndOp>(loc, cond, nb);
      builder.create<mlir::CondBranchOp>(loc, cond, &bodyB, &exitB);
    }

    builder.setInsertionPointToStart(&bodyB);
    builder.create<mlir::memref::StoreOp>(
        loc,
        builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                             std::vector<mlir::Value>()),
        loops.back().keepRunning, std::vector<mlir::Value>());

    Visit(fors->getBody());
    if (auto s = fors->getInc()) {
      Visit(s);
    }
    loops.pop_back();
    if (builder.getInsertionBlock()->empty() ||
        !isTerminator(&builder.getInsertionBlock()->back())) {
      builder.create<mlir::BranchOp>(loc, &condB);
    }

    builder.setInsertionPointToStart(&exitB);
  }
  return nullptr;
}

mlir::Value add(MLIRScanner &sc, mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value lhs, mlir::Value rhs) {
  assert(lhs);
  assert(rhs);
  if (auto op = lhs.getDefiningOp<mlir::ConstantOp>()) {
    if (op.getValue().cast<IntegerAttr>().getInt() == 0) {
      return rhs;
    }
  }

  if (auto op = lhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.getValue() == 0) {
      return rhs;
    }
  }

  if (auto op = rhs.getDefiningOp<mlir::ConstantOp>()) {
    if (op.getValue().cast<IntegerAttr>().getInt() == 0) {
      return lhs;
    }
  }

  if (auto op = rhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.getValue() == 0) {
      return lhs;
    }
  }
  return builder.create<mlir::AddIOp>(loc, lhs, rhs);
}

mlir::Value MLIRScanner::castToIndex(mlir::Location loc, mlir::Value val) {
  assert(val && "Expect non-null value");

  if (auto op = val.getDefiningOp<mlir::ConstantOp>())
    return getConstantIndex(op.getValue().cast<IntegerAttr>().getInt());

  return builder.create<mlir::IndexCastOp>(
      loc, val, mlir::IndexType::get(val.getContext()));
}

ValueWithOffsets
MLIRScanner::VisitCXXConstructExpr(clang::CXXConstructExpr *cons) {
  return VisitConstructCommon(cons, /*name*/ nullptr, /*space*/ 0);
}

ValueWithOffsets
MLIRScanner::VisitConstructCommon(clang::CXXConstructExpr *cons, VarDecl *name,
                                  unsigned memtype, mlir::Value op) {
  if (cons->getConstructionKind() !=
      clang::CXXConstructExpr::ConstructionKind::CK_Complete) {
    cons->dump();
  }
  assert(cons->getConstructionKind() ==
         clang::CXXConstructExpr::ConstructionKind::CK_Complete);

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

  SmallVector<mlir::Value> args;
  args.push_back(op);
  auto getLLVM = [&](Expr *E) -> mlir::Value {
    if (auto IC1 = dyn_cast<ImplicitCastExpr>(E)) {
      if (auto IC2 = dyn_cast<ImplicitCastExpr>(IC1->getSubExpr())) {
        if (auto slit = dyn_cast<clang::StringLiteral>(IC2->getSubExpr())) {
          return Glob.GetOrCreateGlobalLLVMString(loc, builder,
                                                  slit->getString());
        }
      }
      if (auto IC1 = dyn_cast<ImplicitCastExpr>(E)) {
        if (auto IC2 = dyn_cast<PredefinedExpr>(IC1->getSubExpr())) {
          if (auto slit =
                  dyn_cast<clang::StringLiteral>(IC2->getFunctionName())) {
            return Glob.GetOrCreateGlobalLLVMString(loc, builder,
                                                    slit->getString());
          }
        }
      }
      if (auto slit = dyn_cast<clang::StringLiteral>(IC1->getSubExpr())) {
        return Glob.GetOrCreateGlobalLLVMString(loc, builder,
                                                slit->getString());
      }
    }

    auto sub = Visit(E);
    if (!sub.val) {
      cons->dump();
      E->dump();
    }
    assert(sub.val);

    bool isReference = E->isLValue() || E->isXValue();
    if (isReference) {
      if (!sub.isReference) {
        cons->dump();
        E->dump();
        llvm::errs() << " val: " << sub.val
                     << " isRef: " << (int)sub.isReference << "\n";
      }
      assert(sub.isReference);
      return sub.val;
    }
    return sub.getValue(builder);
  };
  std::vector<std::pair<ValueWithOffsets, ValueWithOffsets>> toRestore;
  for (auto a : cons->arguments()) {
    if (LLVMABI)
      args.push_back(getLLVM(a));
    else {
      auto arg = Visit(a);
      if (!arg.val) {
        cons->dump();
        a->dump();
      }
      assert(arg.val);
      bool isReference = a->isLValue() || a->isXValue();

      bool isArray = false;

      Glob.getMLIRType(a->getType(), &isArray);

      mlir::Value val;
      if (!isReference) {
        if (isArray) {
          assert(arg.isReference);

          auto mt =
              Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                   a->getType()))
                  .cast<MemRefType>();
          auto shape = std::vector<int64_t>(mt.getShape());
          auto pshape = shape[0];
          if (pshape == -1)
            shape[0] = 1;
          assert(shape.size() == 2);

          OpBuilder abuilder(builder.getContext());
          abuilder.setInsertionPointToStart(allocationScope);
          auto alloc = abuilder.create<mlir::memref::AllocaOp>(
              loc,
              mlir::MemRefType::get(shape, mt.getElementType(),
                                    mt.getAffineMaps(), mt.getMemorySpace()));

          ValueWithOffsets(alloc, /*isRef*/ true)
              .store(builder, arg, /*isArray*/ isArray);
          shape[0] = pshape;
          val = builder.create<mlir::memref::CastOp>(
              loc, alloc,
              mlir::MemRefType::get(shape, mt.getElementType(),
                                    mt.getAffineMaps(), mt.getMemorySpace()));
        } else
          val = arg.getValue(builder);
      } else {
        assert(arg.isReference);

        if (isArray && arg.val.getType().isa<LLVM::LLVMPointerType>()) {
          auto mt =
              Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                   a->getType()))
                  .cast<MemRefType>();
          auto shape = std::vector<int64_t>(mt.getShape());
          auto pshape = shape[0];
          if (pshape == -1)
            shape[0] = 1;
          assert(shape.size() == 2);

          OpBuilder abuilder(builder.getContext());
          abuilder.setInsertionPointToStart(allocationScope);
          auto alloc = abuilder.create<mlir::memref::AllocaOp>(loc, mt);
          ValueWithOffsets(alloc, /*isRef*/ true)
              .store(builder, arg, /*isArray*/ isArray);
          toRestore.emplace_back(ValueWithOffsets(alloc, /*isRef*/ true), arg);
          shape[0] = pshape;
          val = builder.create<memref::CastOp>(
              loc, alloc,
              MemRefType::get(shape, mt.getElementType(), mt.getAffineMaps(),
                              mt.getMemorySpace()));
        } else
          val = arg.val;

        /*
        if (!isArray)
          if (auto mt = val.getType().dyn_cast<MemRefType>()) {
            auto shape = std::vector<int64_t>(mt.getShape());
            shape[0] = 1;
            val = builder.create<memref::CastOp>(
                loc, val,
                mlir::MemRefType::get(shape, mt.getElementType(),
                                      mt.getAffineMaps(), mt.getMemorySpace()));
          }
        */
      }
      args.push_back(val);
    }
  }

  auto tocall = Glob.GetOrCreateMLIRFunction(cons->getConstructor());
  builder.create<mlir::CallOp>(loc, tocall, args);
  for (auto pair : toRestore) {
    pair.second.store(builder, pair.first, /*isArray*/ true);
  }
  return ValueWithOffsets(op, /*isReference*/ true);
}

ValueWithOffsets MLIRScanner::CommonArrayToPointer(ValueWithOffsets scalar) {
  assert(scalar.val);
  assert(scalar.isReference);
  if (auto PT = scalar.val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    if (PT.getElementType().isa<mlir::LLVM::LLVMPointerType>())
      return ValueWithOffsets(scalar.val, /*isRef*/ false);
    auto iTy = builder.getIntegerType(32);
    mlir::Value vec[3] = {scalar.val,
                          builder.create<mlir::ConstantOp>(
                              loc, iTy, builder.getIntegerAttr(iTy, 0)),
                          builder.create<mlir::ConstantOp>(
                              loc, iTy, builder.getIntegerAttr(iTy, 0))};
    if (!PT.getElementType().isa<mlir::LLVM::LLVMArrayType>()) {
      EmittingFunctionDecl->dump();
      function.dump();
      llvm::errs() << " sval: " << scalar.val << "\n";
      llvm::errs() << PT << "\n";
    }
    auto ET =
        PT.getElementType().cast<mlir::LLVM::LLVMArrayType>().getElementType();
    return ValueWithOffsets(
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
  auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                   mt.getAffineMaps(), mt.getMemorySpace());

  auto post = builder.create<memref::CastOp>(loc, mt0, scalar.val);
  return ValueWithOffsets(post, /*isReference*/ false);
}

ValueWithOffsets MLIRScanner::CommonArrayLookup(ValueWithOffsets array,
                                                mlir::Value idx) {
  mlir::Value val = array.getValue(builder);
  assert(val);

  if (val.getType().isa<LLVM::LLVMPointerType>()) {

    std::vector<mlir::Value> vals = {val};
    idx =
        builder.create<mlir::IndexCastOp>(loc, idx, builder.getIntegerType(64));
    vals.push_back(idx);
    // TODO sub
    return ValueWithOffsets(
        builder.create<mlir::LLVM::GEPOp>(loc, val.getType(), vals),
        /*isReference*/ true);
  }
  if (!val.getType().isa<MemRefType>()) {
    EmittingFunctionDecl->dump();
    builder.getInsertionBlock()->dump();
    function.dump();
    llvm::errs() << "value: " << val << "\n";
  }

  ValueWithOffsets dref;
  {
    auto mt = val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    shape[0] = -1;
    auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                     mt.getAffineMaps(), mt.getMemorySpace());
    auto post = builder.create<polygeist::SubIndexOp>(loc, mt0, val, idx);
    // TODO sub
    dref = ValueWithOffsets(post, /*isReference*/ true);
  }
  assert(dref.isReference);

  auto mt = dref.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() > 1) {
    shape.erase(shape.begin());
  } else {
    shape[0] = -1;
  }
  auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                   mt.getAffineMaps(), mt.getMemorySpace());
  auto post = builder.create<polygeist::SubIndexOp>(loc, mt0, dref.val,
                                                    getConstantIndex(0));
  return ValueWithOffsets(post, /*isReference*/ true);
}

ValueWithOffsets
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
    auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                     mt.getAffineMaps(), mt.getMemorySpace());
    moo.val = builder.create<polygeist::SubIndexOp>(loc, mt0, moo.val,
                                                    getConstantIndex(0));
  }
  return CommonArrayLookup(moo, idx);
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

std::pair<ValueWithOffsets, bool>
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
            ValueWithOffsets(builder.create<mlir::math::Log2Op>(loc, args[0]),
                             /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "log") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return make_pair(
            ValueWithOffsets(builder.create<mlir::math::LogOp>(loc, args[0]),
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
            ValueWithOffsets(builder.create<mlir::CeilFOp>(loc, args[0]),
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
            ValueWithOffsets(builder.create<mlir::math::SqrtOp>(loc, args[0]),
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
            ValueWithOffsets(builder.create<mlir::math::ExpOp>(loc, args[0]),
                             /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "sin") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return make_pair(
            ValueWithOffsets(builder.create<mlir::math::SinOp>(loc, args[0]),
                             /*isReference*/ false),
            true);
      }

      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "cos") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return make_pair(
            ValueWithOffsets(builder.create<mlir::math::CosOp>(loc, args[0]),
                             /*isReference*/ false),
            true);
      }
    }
  }

  return make_pair(ValueWithOffsets(), false);
}

std::pair<ValueWithOffsets, bool>
MLIRScanner::EmitGPUCallExpr(clang::CallExpr *expr) {
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee())) {
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__syncthreads") {
        builder.create<mlir::NVVM::Barrier0Op>(loc);
        return make_pair(ValueWithOffsets(), true);
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
           auto tocall = EmitDirectCallee(callee);
           mlir::Value args[] = { arg };
           builder.create<mlir::CallOp>(loc, tocall, args);
        } else { 
          builder.create<mlir::memref::DeallocOp>(loc, arg);
        }
        if (sr->getDecl()->getName() == "cudaFree" ||
            sr->getDecl()->getName() == "cudaFreeHost") {
          auto ty = getMLIRType(expr->getType());
          auto op = builder.create<mlir::ConstantOp>(
              loc, ty, builder.getIntegerAttr(ty, /*cudaSuccess*/ 0));
          return make_pair(ValueWithOffsets(op, /*isReference*/ false), true);
        }
        // TODO remove me when the free is removed.
        return make_pair(ValueWithOffsets(), true);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "cudaMalloc" ||
           sr->getDecl()->getName() == "cudaMallocHost")) {
        auto sub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(sub))
          sub = BC->getSubExpr();
        {
          auto dst = Visit(sub).getValue(builder);
          if (auto omt = dst.getType().dyn_cast<MemRefType>()) {
            auto mt = omt.getElementType().dyn_cast<MemRefType>();
            auto shape = std::vector<int64_t>(mt.getShape());

            auto elemSize =
                getTypeSize(cast<clang::PointerType>(
                                cast<clang::PointerType>(
                                    sub->getType()
                                        ->getUnqualifiedDesugaredType())
                                    ->getPointeeType())
                                ->getPointeeType());
            mlir::Value allocSize = builder.create<mlir::IndexCastOp>(
                loc, Visit(expr->getArg(1)).getValue(builder),
                mlir::IndexType::get(builder.getContext()));
            mlir::Value args[1] = {builder.create<mlir::UnsignedDivIOp>(
                loc, allocSize,
                builder.create<mlir::ConstantOp>(
                    loc, allocSize.getType(),
                    builder.getIntegerAttr(allocSize.getType(), elemSize)))};
            auto alloc = builder.create<mlir::memref::AllocOp>(
                loc,
                (sr->getDecl()->getName() == "cudaMalloc" && !CudaLower)
                    ? mlir::MemRefType::get(shape, mt.getElementType(),
                                            mt.getAffineMaps(), 1)
                    : mt,
                args);
            ValueWithOffsets(dst, /*isReference*/ true)
                .store(builder,
                       builder.create<mlir::memref::CastOp>(loc, alloc, mt));
            auto retTy = getMLIRType(expr->getType());
            return make_pair(
                ValueWithOffsets(
                    builder.create<mlir::ConstantOp>(
                        loc, retTy, builder.getIntegerAttr(retTy, 0)),
                    /*isReference*/ false),
                true);
          }
        }
      }
    }

    auto createBlockIdOp = [&](string str, mlir::Type mlirType) -> mlir::Value {
      return builder.create<mlir::IndexCastOp>(
          loc,
          builder.create<mlir::gpu::BlockIdOp>(
              loc, mlir::IndexType::get(builder.getContext()), str),
          mlirType);
    };

    auto createBlockDimOp = [&](string str,
                                mlir::Type mlirType) -> mlir::Value {
      return builder.create<mlir::IndexCastOp>(
          loc,
          builder.create<mlir::gpu::BlockDimOp>(
              loc, mlir::IndexType::get(builder.getContext()), str),
          mlirType);
    };

    auto createThreadIdOp = [&](string str,
                                mlir::Type mlirType) -> mlir::Value {
      return builder.create<mlir::IndexCastOp>(
          loc,
          builder.create<mlir::gpu::ThreadIdOp>(
              loc, mlir::IndexType::get(builder.getContext()), str),
          mlirType);
    };

    auto createGridDimOp = [&](string str, mlir::Type mlirType) -> mlir::Value {
      return builder.create<mlir::IndexCastOp>(
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
              return make_pair(ValueWithOffsets(createBlockIdOp("x", mlirType),
                                                /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(ValueWithOffsets(createBlockIdOp("y", mlirType),
                                                /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(ValueWithOffsets(createBlockIdOp("z", mlirType),
                                                /*isReference*/ false),
                               true);
            }
          }
          if (sr->getDecl()->getName() == "blockDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(ValueWithOffsets(createBlockDimOp("x", mlirType),
                                                /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(ValueWithOffsets(createBlockDimOp("y", mlirType),
                                                /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(ValueWithOffsets(createBlockDimOp("z", mlirType),
                                                /*isReference*/ false),
                               true);
            }
          }
          if (sr->getDecl()->getName() == "threadIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(ValueWithOffsets(createThreadIdOp("x", mlirType),
                                                /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(ValueWithOffsets(createThreadIdOp("y", mlirType),
                                                /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(ValueWithOffsets(createThreadIdOp("z", mlirType),
                                                /*isReference*/ false),
                               true);
            }
          }
          if (sr->getDecl()->getName() == "gridDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(ValueWithOffsets(createGridDimOp("x", mlirType),
                                                /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(ValueWithOffsets(createGridDimOp("x", mlirType),
                                                /*isReference*/ false),
                               true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(ValueWithOffsets(createGridDimOp("z", mlirType),
                                                /*isReference*/ false),
                               true);
            }
          }
        }
      }
    }
  }
  return make_pair(ValueWithOffsets(), false);
}

ValueWithOffsets MLIRScanner::VisitCallExpr(clang::CallExpr *expr) {

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

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "atomicAdd") {
        std::vector<ValueWithOffsets> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a));
        }
        auto a1 = args[1].getValue(builder);
        if (a1.getType().isa<mlir::IntegerType>())
          return ValueWithOffsets(
              builder.create<mlir::AtomicRMWOp>(
                  loc, a1.getType(), AtomicRMWKind::addi, a1,
                  args[0].getValue(builder),
                  std::vector<mlir::Value>({getConstantIndex(0)})),
              /*isReference*/ false);
        else
          return ValueWithOffsets(
              builder.create<mlir::AtomicRMWOp>(
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
#if 1
        llvm_unreachable("atomicOr unhandled");
        assert(0 && "atomicOr unhandled");
#else
        std::vector<ValueWithOffsets> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a));
        }
        auto a1 = args[1].getValue(builder);
        return ValueWithOffsets(
            builder.create<mlir::AtomicRMWOp>(
                loc, a1.getType(), AtomicRMWKind::ori, a1,
                args[0].getValue(builder),
                std::vector<mlir::Value>({getConstantIndex(0)})),
            /*isReference*/ false);
#endif
      }
    }

  auto getLLVM = [&](Expr *E) -> mlir::Value {
    if (auto IC1 = dyn_cast<ImplicitCastExpr>(E)) {
      if (auto IC2 = dyn_cast<ImplicitCastExpr>(IC1->getSubExpr())) {
        if (auto slit = dyn_cast<clang::StringLiteral>(IC2->getSubExpr())) {
          return Glob.GetOrCreateGlobalLLVMString(loc, builder,
                                                  slit->getString());
        }
      }
      if (auto IC1 = dyn_cast<ImplicitCastExpr>(E)) {
        if (auto IC2 = dyn_cast<PredefinedExpr>(IC1->getSubExpr())) {
          if (auto slit =
                  dyn_cast<clang::StringLiteral>(IC2->getFunctionName())) {
            return Glob.GetOrCreateGlobalLLVMString(loc, builder,
                                                    slit->getString());
          }
        }
      }
      if (auto IC1 = dyn_cast<ImplicitCastExpr>(E)) {
        if (auto U0 = dyn_cast<clang::UnaryOperator>(IC1->getSubExpr()))
          if (auto IC2 = dyn_cast<PredefinedExpr>(U0->getSubExpr())) {
            if (auto slit =
                    dyn_cast<clang::StringLiteral>(IC2->getFunctionName())) {
              return Glob.GetOrCreateGlobalLLVMString(loc, builder,
                                                      slit->getString());
            }
          }
      }
      if (auto slit = dyn_cast<clang::StringLiteral>(IC1->getSubExpr())) {
        return Glob.GetOrCreateGlobalLLVMString(loc, builder,
                                                slit->getString());
      }
    }
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
    auto val = sub.getValue(builder);
    if (auto mt = val.getType().dyn_cast<MemRefType>()) {
      val = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getElementType()), val);
    }
    return val;
  };

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      // TODO add pow to standard dialect
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__powf" ||
           sr->getDecl()->getName() == "pow" ||
           sr->getDecl()->getName() == "powf")) {
        auto mlirType = getMLIRType(expr->getType());
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return ValueWithOffsets(
            builder.create<mlir::math::PowFOp>(loc, mlirType, args[0], args[1]),
            /*isReference*/ false);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "fprintf" ||
           sr->getDecl()->getName() == "printf")) {

        auto tocall = EmitCallee(expr->getCallee());
        auto fprintfF = Glob.GetOrCreateLLVMFunction(tocall);

        std::vector<mlir::Value> args;
        size_t i = 0;
        for (auto a : expr->arguments()) {
          args.push_back(getLLVM(a));
          i++;
        }

        builder.create<mlir::LLVM::CallOp>(loc, fprintfF, args);

        return nullptr;
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__nv_fabsf" ||
                  sr->getDecl()->getName() == "__nv_fabs"
                  )) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value Fabs = builder.create<mlir::AbsFOp>(loc, V);
        return ValueWithOffsets(Fabs, /*isRef*/false);
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_isfinite" ||
                  sr->getDecl()->getName() == "__builtin_isinf"
                  )) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        mlir::Value V = getLLVM(expr->getArg(0));
        auto Ty = V.getType().cast<mlir::FloatType>();
        mlir::Value Fabs = builder.create<mlir::AbsFOp>(loc, V);
        auto Infinity = builder.create<mlir::ConstantFloatOp>(loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
        auto Pred = ( sr->getDecl()->getName() == "__builtin_isinf" )
                                        ? mlir::CmpFPredicate::OEQ
                                        : mlir::CmpFPredicate::ONE;
        mlir::Value FCmp = builder.create<CmpFOp>(loc, Pred, Fabs, Infinity);
        auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
        mlir::Value res = builder.create<mlir::ZeroExtendIOp>(loc, FCmp, postTy);
        return ValueWithOffsets(res, /*isRef*/false);
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_isnormal"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          auto Ty = V.getType().cast<mlir::FloatType>();
          mlir::Value Eq = builder.create<CmpFOp>(loc, mlir::CmpFPredicate::OEQ, V, V);

          mlir::Value Abs = builder.create<mlir::AbsFOp>(loc, V);
          auto Infinity = builder.create<mlir::ConstantFloatOp>(loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
          mlir::Value IsLessThanInf = builder.create<CmpFOp>(loc, mlir::CmpFPredicate::ULT, Abs, Infinity);
          APFloat Smallest = APFloat::getSmallestNormalized(Ty.getFloatSemantics());
          auto SmallestV = builder.create<mlir::ConstantFloatOp>(loc, Smallest, Ty);
          mlir::Value IsNormal = builder.create<CmpFOp>(loc, mlir::CmpFPredicate::UGE, Abs, SmallestV);
          V = builder.create<AndOp>(loc, Eq, IsLessThanInf);
          V = builder.create<AndOp>(loc, V, IsNormal);
          auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
          mlir::Value res = builder.create<mlir::ZeroExtendIOp>(loc, V, postTy);
          return ValueWithOffsets(res, /*isRef*/false);
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "__builtin_signbit") {
          mlir::Value V = getLLVM(expr->getArg(0));
          auto Ty = V.getType().cast<mlir::FloatType>();
          auto ITy = builder.getIntegerType(Ty.getWidth());
          mlir::Value BC = builder.create<BitcastOp>(loc, ITy, V);
          auto ZeroV = builder.create<mlir::ConstantIntOp>(loc, 0, ITy);
          V = builder.create<CmpIOp>(loc, mlir::CmpIPredicate::slt, BC, ZeroV);
          auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
          mlir::Value res = builder.create<mlir::ZeroExtendIOp>(loc, V, postTy);
          return ValueWithOffsets(res, /*isRef*/false);
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "__builtin_isgreater") {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<CmpFOp>(loc, mlir::CmpFPredicate::OGT, V, V2);
          auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
          mlir::Value res = builder.create<mlir::ZeroExtendIOp>(loc, V, postTy);
          return ValueWithOffsets(res, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "__builtin_isgreaterequal") {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<CmpFOp>(loc, mlir::CmpFPredicate::OGE, V, V2);
          auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
          mlir::Value res = builder.create<mlir::ZeroExtendIOp>(loc, V, postTy);
          return ValueWithOffsets(res, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "__builtin_isless") {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<CmpFOp>(loc, mlir::CmpFPredicate::OLT, V, V2);
          auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
          mlir::Value res = builder.create<mlir::ZeroExtendIOp>(loc, V, postTy);
          return ValueWithOffsets(res, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "__builtin_islessequal") {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<CmpFOp>(loc, mlir::CmpFPredicate::OLE, V, V2);
          auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
          mlir::Value res = builder.create<mlir::ZeroExtendIOp>(loc, V, postTy);
          return ValueWithOffsets(res, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "__builtin_islessgreater") {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<CmpFOp>(loc, mlir::CmpFPredicate::ONE, V, V2);
          auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
          mlir::Value res = builder.create<mlir::ZeroExtendIOp>(loc, V, postTy);
          return ValueWithOffsets(res, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "__builtin_isunordered") {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<CmpFOp>(loc, mlir::CmpFPredicate::UNO, V, V2);
          auto postTy = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
          mlir::Value res = builder.create<mlir::ZeroExtendIOp>(loc, V, postTy);
          return ValueWithOffsets(res, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_atanh" ||
                  sr->getDecl()->getName() == "__builtin_atanhf" ||
                  sr->getDecl()->getName() == "__builtin_atanhl"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          V = builder.create<math::AtanOp>(loc, V);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__nv_scalbn" ||
                  sr->getDecl()->getName() == "__nv_scalbnf" ||
                  sr->getDecl()->getName() == "__nv_scalbnl"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          auto name = sr->getDecl()->getName().substr(5).str();
          std::vector<mlir::Type> types { V.getType(), V2.getType() };
          auto RT = getMLIRType(expr->getType());

          std::vector<mlir::Type> rettypes { RT };

          mlir::OpBuilder mbuilder(Glob.module.getContext());
          auto funcType = mbuilder.getFunctionType(types, rettypes);
          mlir::FuncOp function = mlir::FuncOp(
              mlir::FuncOp::create(builder.getUnknownLoc(), name, funcType));
          SymbolTable::setSymbolVisibility(function,
                                             SymbolTable::Visibility::Private);

          Glob.functions[name] = function;
          Glob.module.push_back(function);
          mlir::Value vals[] = {V, V2};
          V = builder.create<CallOp>(loc, function, vals).getResult(0);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__nv_dmul_rn"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<MulFOp>(loc, V, V2);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__nv_dadd_rn"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<AddFOp>(loc, V, V2);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__nv_dsub_rn"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<SubFOp>(loc, V, V2);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_log2" ||
                  sr->getDecl()->getName() == "__builtin_log2f" ||
                  sr->getDecl()->getName() == "__builtin_log2l" ||
                  sr->getDecl()->getName() == "__nv_log2" ||
                  sr->getDecl()->getName() == "__nv_log2f" ||
                  sr->getDecl()->getName() == "__nv_log2l"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          V = builder.create<math::Log2Op>(loc, V);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_log1p" ||
                  sr->getDecl()->getName() == "__builtin_log1pf" ||
                  sr->getDecl()->getName() == "__builtin_log1pl"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          V = builder.create<math::Log1pOp>(loc, V);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_exp2" ||
                  sr->getDecl()->getName() == "__builtin_exp2f" ||
                  sr->getDecl()->getName() == "__builtin_exp2l"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          V = builder.create<math::Exp2Op>(loc, V);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_expm1" ||
                  sr->getDecl()->getName() == "__builtin_expm1f" ||
                  sr->getDecl()->getName() == "__builtin_expm1l"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          V = builder.create<math::ExpM1Op>(loc, V);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_copysign" ||
                  sr->getDecl()->getName() == "__builtin_copysignf" ||
                  sr->getDecl()->getName() == "__builtin_copysignl"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<LLVM::CopySignOp>(loc, V, V2);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_copysign" ||
                  sr->getDecl()->getName() == "__builtin_copysignf" ||
                  sr->getDecl()->getName() == "__builtin_copysignl"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<LLVM::CopySignOp>(loc, V, V2);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_fmax" ||
                  sr->getDecl()->getName() == "__builtin_fmaxf" ||
                  sr->getDecl()->getName() == "__builtin_fmaxl"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<LLVM::MaxNumOp>(loc, V, V2);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_fmin" ||
                  sr->getDecl()->getName() == "__builtin_fminf" ||
                  sr->getDecl()->getName() == "__builtin_fminl"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          V = builder.create<LLVM::MinNumOp>(loc, V, V2);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
      if (sr->getDecl()->getIdentifier() && (
                  sr->getDecl()->getName() == "__builtin_fma" ||
                  sr->getDecl()->getName() == "__builtin_fmaf" ||
                  sr->getDecl()->getName() == "__builtin_fmal"
                  )) {
          mlir::Value V = getLLVM(expr->getArg(0));
          mlir::Value V2 = getLLVM(expr->getArg(1));
          mlir::Value V3 = getLLVM(expr->getArg(2));
          V = builder.create<LLVM::FMAOp>(loc, V, V2, V3);
          return ValueWithOffsets(V, /*isRef*/false); 
      }
    }

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "fscanf" ||
           sr->getDecl()->getName() == "scanf" ||
           sr->getDecl()->getName() == "__isoc99_sscanf" ||
           sr->getDecl()->getName() == "sscanf"
           ) || (isa<CXXOperatorCallExpr>(expr) && cast<CXXOperatorCallExpr>(expr)->getOperator() == OO_GreaterGreater)) {
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
        return ValueWithOffsets(called.getResult(0), /*isReference*/ false);
      }
    }

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "cudaMemcpy" ||
           sr->getDecl()->getName() == "cudaMemcpyToSymbol" ||
           sr->getDecl()->getName() == "memcpy")) {
        auto dstSub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(dstSub))
          dstSub = BC->getSubExpr();
        auto srcSub = expr->getArg(1);
        while (auto BC = dyn_cast<clang::CastExpr>(srcSub))
          srcSub = BC->getSubExpr();

        auto dstst = dstSub->getType()->getUnqualifiedDesugaredType();
        auto elem = isa<clang::PointerType>(dstst)
                        ? cast<clang::PointerType>(dstst)->getPointeeType()->getUnqualifiedDesugaredType()

                        : cast<clang::ArrayType>(dstst)->getElementType()->getUnqualifiedDesugaredType();
        auto melem = elem;
        if (auto BC = dyn_cast<clang::ArrayType>(melem)) melem = BC->getElementType()->getUnqualifiedDesugaredType();

        auto srcst = srcSub->getType()->getUnqualifiedDesugaredType();
        auto selem = isa<clang::PointerType>(srcst)
                         ? cast<clang::PointerType>(srcst)->getPointeeType()->getUnqualifiedDesugaredType()

                         : cast<clang::ArrayType>(srcst)->getElementType()->getUnqualifiedDesugaredType();

        auto mselem = selem;
        if (auto BC = dyn_cast<clang::ArrayType>(mselem)) mselem = BC->getElementType()->getUnqualifiedDesugaredType();

        if (melem == mselem) {
          mlir::Value dst;
          ValueWithOffsets vdst = Visit(dstSub);
          if (isa<clang::PointerType>(dstst)) {
            dst = vdst.getValue(builder);
          } else {
            assert(vdst.isReference);
            dst = vdst.val;
          }
          if (dst.getType().isa<MemRefType>()) {
            mlir::Value src;
            ValueWithOffsets vsrc = Visit(srcSub);
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
            mlir::Value size = builder.create<mlir::IndexCastOp>(
                loc,
                Visit(
                    expr->getArg(sr->getDecl()->getName() == "memcpy" ? 1 : 2))
                    .getValue(builder),
                mlir::IndexType::get(builder.getContext()));
            size = builder.create<mlir::UnsignedDivIOp>(
                loc, size,
                builder.create<mlir::ConstantIndexOp>(loc, elemSize));

            if (sr->getDecl()->getName() == "cudaMemcpyToSymbol") {
              mlir::Value offset = Visit(expr->getArg(3)).getValue(builder);
              offset = builder.create<mlir::IndexCastOp>(
                  loc, offset, mlir::IndexType::get(builder.getContext()));
              offset = builder.create<mlir::UnsignedDivIOp>(
                  loc, offset,
                  builder.create<mlir::ConstantIndexOp>(loc, elemSize));
              // assert(!dstArray);
              auto mt = dst.getType().cast<MemRefType>();
              auto shape = std::vector<int64_t>(mt.getShape());
              shape[0] = -1;
              auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                               mt.getAffineMaps(),
                                               mt.getMemorySpace());
              dst =
                  builder.create<polygeist::SubIndexOp>(loc, mt0, dst, offset);
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
              auto mt =
                  Glob.getMLIRType(Glob.CGM.getContext().getPointerType(QualType(elem, 0)))
                      .cast<MemRefType>();
              auto shape = std::vector<int64_t>(mt.getShape());
              assert(shape.size() > 0 && shape.back() != -1);
              auto affineOp = builder.create<scf::ForOp>(
                  loc, getConstantIndex(0), getConstantIndex(shape.back()),
                  getConstantIndex(1));
              dstargs.push_back(affineOp.getInductionVar());
              builder.setInsertionPointToStart(&affineOp.getLoopBody().front());
              if (srcArray) {
                auto smt =
                  Glob.getMLIRType(Glob.CGM.getContext().getPointerType(QualType(elem, 0)))
                      .cast<MemRefType>();
                auto sshape = std::vector<int64_t>(smt.getShape());
                assert(sshape.size() > 0 && sshape.back() != -1);
                assert(sshape.back() == shape.back());
                srcargs.push_back(affineOp.getInductionVar());
              } else {
                srcargs[0] = builder.create<AddIOp>(loc, builder.create<MulIOp>(loc, srcargs[0], getConstantIndex(shape.back())), affineOp.getInductionVar());
              }
            } else {
                if (srcArray) {
                    auto smt =
                      Glob.getMLIRType(Glob.CGM.getContext().getPointerType(QualType(selem, 0)))
                          .cast<MemRefType>();
                    auto sshape = std::vector<int64_t>(smt.getShape());
                    assert(sshape.size() > 0 && sshape.back() != -1);
                  auto affineOp = builder.create<scf::ForOp>(
                      loc, getConstantIndex(0), getConstantIndex(sshape.back()),
                      getConstantIndex(1));
                  srcargs.push_back(affineOp.getInductionVar());
                  builder.setInsertionPointToStart(&affineOp.getLoopBody().front());
                  dstargs[0] = builder.create<AddIOp>(loc, builder.create<MulIOp>(loc, dstargs[0], getConstantIndex(sshape.back())), affineOp.getInductionVar());
                }
            }

            auto loaded = builder.create<memref::LoadOp>(loc, src, srcargs);
            builder.create<memref::StoreOp>(loc, loaded, dst, dstargs);

            // TODO: set the value of the iteration value to the final bound
            // at the end of the loop.
            builder.setInsertionPoint(oldblock, oldpoint);

            auto retTy = getMLIRType(expr->getType());
            return ValueWithOffsets(
                builder.create<mlir::ConstantOp>(
                    loc, retTy, builder.getIntegerAttr(retTy, 0)),
                /*isReference*/ false);
          }
        }
        function.dump();
        expr->dump();
        dstSub->dump();
        elem->dump();
        srcSub->dump();
        mselem->dump();
        assert(0 && "unhandled cudaMemcpy");
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
            ValueWithOffsets vdst = Visit(dstSub);
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
                toStore = builder.create<mlir::ConstantOp>(
                    loc, melem, builder.getIntegerAttr(melem, 0));
              else
                toStore = builder.create<mlir::ConstantOp>(
                    loc, melem, builder.getFloatAttr(melem, 0));

              auto elemSize = getTypeSize(elem);
              mlir::Value size = builder.create<mlir::IndexCastOp>(
                  loc, Visit(expr->getArg(2)).getValue(builder),
                  mlir::IndexType::get(builder.getContext()));
              size = builder.create<mlir::UnsignedDivIOp>(
                  loc, size,
                  builder.create<mlir::ConstantIndexOp>(loc, elemSize));

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
              return ValueWithOffsets(
                  builder.create<mlir::ConstantOp>(
                      loc, retTy, builder.getIntegerAttr(retTy, 0)),
                  /*isReference*/ false);
            }
          }
      }
    }

  auto callee = EmitCallee(expr->getCallee());

  std::set<std::string> funcs = {"strcmp",
                                 "sprintf",
                                 "fputs",
                                 "puts",
                                 "memcpy",
                                 "cudaMalloc",
                                 "open",
                                 "fopen",
                                 "memset",
                                 "cudaMemset",
                                 "strcpy",
                                 "close",
                                 "fclose",
                                 "atoi",
                                 "malloc",
                                 "calloc",
                                 "fgets",
                                 "__assert_fail",
                                 "cudaEventElapsedTime",
                                 "cudaEventSynchronize",
                                 "cudaEventRecord"};
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if ((sr->getDecl()->getIdentifier() &&
           funcs.count(sr->getDecl()->getName().str()))) {

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
          called = builder
                       .create<mlir::LLVM::CallOp>(
                           loc,
                           std::vector<mlir::Type>(
                               {Glob.typeTranslator.translateType(
                                   anonymize(getLLVMType(expr->getType())))}),
                           args)
                       .getResult(0);
        }
        return ValueWithOffsets(called, /*isReference*/ expr->isLValue() ||
                                            expr->isXValue());
      }
    }

  if (!callee || callee->isVariadic()) {
    std::vector<mlir::Value> args;
    for (auto a : expr->arguments()) {
      args.push_back(getLLVM(a));
    }
    mlir::Value called;
    args.insert(args.begin(), getLLVM(expr->getCallee()));
    called = builder
                 .create<mlir::LLVM::CallOp>(
                     loc,
                     std::vector<mlir::Type>({Glob.typeTranslator.translateType(
                         anonymize(getLLVMType(expr->getType())))}),
                     args)
                 .getResult(0);
    return ValueWithOffsets(called, /*isReference*/ expr->isLValue() ||
                                        expr->isXValue());
  }

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "gettimeofday") {
        auto tocall = EmitCallee(expr->getCallee());
        auto fprintfF = Glob.GetOrCreateLLVMFunction(tocall);
        std::vector<mlir::Value> args;
        size_t i = 0;
        mlir::Value tostore = nullptr;
        mlir::Value alloc;
        for (auto a : expr->arguments()) {

          if (i == 0) {
            tostore = Visit(a).getValue(builder);
            auto mt = tostore.getType().cast<MemRefType>();
            auto shape = std::vector<int64_t>(mt.getShape());
            mlir::Value res;
            shape.erase(shape.begin());
            auto mt0 =
                mlir::MemRefType::get(shape, mt.getElementType(),
                                      mt.getAffineMaps(), mt.getMemorySpace());
            tostore = builder.create<polygeist::SubIndexOp>(
                loc, mt0, tostore, getConstantIndex(0));
            i++;
            auto indexType = mlir::IntegerType::get(module.getContext(), 64);
            auto one = builder.create<mlir::ConstantOp>(
                loc, indexType,
                builder.getIntegerAttr(builder.getIntegerType(64), 1));
            OpBuilder abuilder(builder.getContext());
            abuilder.setInsertionPointToStart(allocationScope);
            alloc = abuilder.create<mlir::LLVM::AllocaOp>(
                loc,
                Glob.typeTranslator.translateType(
                    anonymize(getLLVMType(a->getType()))),
                one, 0);
            args.push_back(alloc);
            continue;
          }
          auto llvmType = Glob.typeTranslator.translateType(
              anonymize(getLLVMType(a->getType())));

          if (auto IC1 = dyn_cast<ImplicitCastExpr>(a)) {
            if (IC1->getCastKind() == clang::CastKind::CK_NullToPointer) {
              args.push_back(builder.create<mlir::LLVM::NullOp>(loc, llvmType));
              i++;
              continue;
            }
          }
          mlir::Value val = Visit(a).getValue(builder);
          args.push_back(val);
          i++;
        }
        assert(alloc);

        auto co = builder.create<mlir::LLVM::CallOp>(loc, fprintfF, args)
                      .getResult(0);
        // co = builder.create<mlir::IndexCastOp>( // was DialectCastOp
        //   loc, getMLIRType(expr->getType()), co);
        auto ret = co;

        auto allV = ValueWithOffsets(alloc, /*isReference*/ true);
        ValueWithOffsets(tostore, /*isReference*/ true)
            .store(builder, allV, /*isArray*/ true);
        return ValueWithOffsets(ret, /*isReference*/ false);
      }
    }

  auto tocall = EmitDirectCallee(callee);
  SmallVector<mlir::Value, 4> args;
  auto fnType = tocall.getType();
  size_t i = 0;
  if (auto CC = dyn_cast<CXXMemberCallExpr>(expr)) {
    auto arg = Visit(CC->getImplicitObjectArgument());
    if (!arg.val) {
      function.dump();
      llvm::errs() << " av: " << arg.val << "\n";
      expr->dump();
      CC->getImplicitObjectArgument()->dump();
    }
    if (cast<MemberExpr>(CC->getCallee()->IgnoreParens())->isArrow()) {
        /*
      function.dump();
      llvm::errs() << " tav: " << arg.val << "\n";
      expr->dump();
      CC->getImplicitObjectArgument()->dump();
      */
      arg = arg.dereference(builder);
    }
    assert(arg.val);
    assert(arg.isReference);
    args.push_back(arg.val);
    i++;
  }

  std::vector<std::pair<ValueWithOffsets, ValueWithOffsets>> toRestore;

  // map from declaration name to mlir::value
  std::map<std::string, mlir::Value> mapFuncOperands;

  for (clang::Expr *a : expr->arguments()) {
    ValueWithOffsets arg = Visit(a);
    if (!arg.val) {
      expr->dump();
      a->dump();
    }
    assert(arg.val && "expect not null");
    if (auto ice = dyn_cast<ImplicitCastExpr>(a))
      if (auto dre = dyn_cast<DeclRefExpr>(ice->getSubExpr()))
        mapFuncOperands.insert(
            make_pair(dre->getDecl()->getName().str(), arg.val));

    if (i >= fnType.getInputs().size()) {
      expr->dump();
      tocall.dump();
      fnType.dump();
      for (auto a : expr->arguments()) {
        a->dump();
      }
      assert(0 && "too many arguments in calls");
    }
    bool isReference = a->isLValue() || a->isXValue();

    bool isArray = false;
    Glob.getMLIRType(a->getType(), &isArray);

    mlir::Value val = nullptr;
    if (!isReference) {
      if (isArray) {
        if (!arg.isReference) {
          expr->dump();
          a->dump();
          llvm::errs() << " v: " << arg.val << "\n";
        }
        assert(arg.isReference);

        auto mt = Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                       a->getType()))
                      .cast<MemRefType>();
        auto shape = std::vector<int64_t>(mt.getShape());
        assert(shape.size() == 2);

        auto pshape = shape[0];
        if (pshape == -1)
          shape[0] = 1;

        OpBuilder abuilder(builder.getContext());
        abuilder.setInsertionPointToStart(allocationScope);
        auto alloc = abuilder.create<mlir::memref::AllocaOp>(
            loc,
            mlir::MemRefType::get(shape, mt.getElementType(),
                                  mt.getAffineMaps(), mt.getMemorySpace()));
        ValueWithOffsets(alloc, /*isRef*/ true)
            .store(builder, arg, /*isArray*/ isArray);
        shape[0] = pshape;
        val = builder.create<mlir::memref::CastOp>(
            loc, alloc,
            mlir::MemRefType::get(shape, mt.getElementType(),
                                  mt.getAffineMaps(), mt.getMemorySpace()));
      } else
        val = arg.getValue(builder);
    } else {
      assert(arg.isReference);

      if (isArray && arg.val.getType().isa<LLVM::LLVMPointerType>()) {
        auto mt = Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                       a->getType()))
                      .cast<MemRefType>();

        auto shape = std::vector<int64_t>(mt.getShape());
        if (shape.size() != 2) {
          expr->dump();
          a->dump();
          a->getType()->dump();
          llvm::errs() << " arg.val: " << arg.val << "\n";
          llvm::errs() << " mt: " << mt << "\n";
        }
        auto pshape = shape[0];
        if (shape.size() == 2)
          if (pshape == -1)
            shape[0] = 1;

        OpBuilder abuilder(builder.getContext());
        abuilder.setInsertionPointToStart(allocationScope);
        auto alloc = abuilder.create<mlir::memref::AllocaOp>(
            loc,
            mlir::MemRefType::get(shape, mt.getElementType(),
                                  mt.getAffineMaps(), mt.getMemorySpace()));

        ValueWithOffsets(alloc, /*isRef*/ true)
            .store(builder, arg, /*isArray*/ isArray);
        toRestore.emplace_back(ValueWithOffsets(alloc, /*isRef*/ true), arg);
        if (shape.size() == 2) {
          shape[0] = pshape;
          val = builder.create<memref::CastOp>(
              loc, alloc,
              MemRefType::get(shape, mt.getElementType(), mt.getAffineMaps(),
                              mt.getMemorySpace()));
        } else {
          val = alloc;
        }
      } else
        val = arg.val;
    }
    assert(val);
    /*
    if (val.getType() != fnType.getInput(i)) {
      if (auto MR1 = val.getType().dyn_cast<MemRefType>()) {
        if (auto MR2 = fnType.getInput(i).dyn_cast<MemRefType>()) {
          val = builder.create<mlir::memref::CastOp>(loc, val, MR2);
        }
      }
    }
    */
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

    return ValueWithOffsets(mlirclang::replaceFuncByOperation(
                                tocall, LTInfo.SymbolTable[tocall.getName()],
                                builder, inputOperands, outputOperands)
                                ->getResult(0),
                            /*isReference=*/false);
  }

  bool isArrayReturn = false;
  if (!(expr->isLValue() || expr->isXValue()))
    Glob.getMLIRType(expr->getType(), &isArrayReturn);

  mlir::Value alloc;
  if (isArrayReturn) {
    auto mt = Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                   expr->getType()))
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
                                   mt.getAffineMaps(), mt.getMemorySpace()));
    shape[0] = pshape;
    alloc = builder.create<mlir::memref::CastOp>(
        loc, alloc,
        mlir::MemRefType::get(shape, mt.getElementType(), mt.getAffineMaps(),
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
    for (auto pair : toRestore) {
      pair.second.store(builder, pair.first, /*isArray*/ true);
    }
    builder.create<gpu::TerminatorOp>(loc);
    builder.setInsertionPoint(oldblock, oldpoint);
    return nullptr;
  }

  auto op = builder.create<mlir::CallOp>(loc, tocall, args);
  for (auto pair : toRestore) {
    pair.second.store(builder, pair.first, /*isArray*/ true);
  }

  if (isArrayReturn) {
    // TODO remedy return
    if (expr->isLValue() || expr->isXValue())
      expr->dump();
    assert(!(expr->isLValue() || expr->isXValue()));
    return ValueWithOffsets(alloc, /*isReference*/ true);
  } else if (op->getNumResults()) {
    return ValueWithOffsets(
        op->getResult(0), /*isReference*/ expr->isLValue() || expr->isXValue());
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
  return constants[x] = subbuilder.create<mlir::ConstantIndexOp>(loc, x);
}

ValueWithOffsets MLIRScanner::VisitMSPropertyRefExpr(MSPropertyRefExpr *expr) {
  assert(0 && "unhandled ms propertyref");
  // TODO obviously fake
  return nullptr;
}

ValueWithOffsets
MLIRScanner::VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr) {
  return Visit(expr->getResultExpr());
}

ValueWithOffsets MLIRScanner::VisitUnaryOperator(clang::UnaryOperator *U) {
  auto sub = Visit(U->getSubExpr());

  switch (U->getOpcode()) {
  case clang::UnaryOperator::Opcode::UO_Extension: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_LNot: {
    assert(sub.val);
    mlir::Value val = sub.getValue(builder);

    if (auto LT = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::eq, val, nullptr_llvm);
      return ValueWithOffsets(ne, /*isReference*/ false);
    }

    if (!val.getType().isa<mlir::IntegerType>()) {
      U->dump();
      val.dump();
    }
    auto ty = val.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      ty = builder.getIntegerType(1);
      val = builder.create<mlir::TruncateIOp>(loc, val, ty);
    }
    auto c1 = builder.create<mlir::ConstantOp>(loc, ty,
                                               builder.getIntegerAttr(ty, 1));
    return ValueWithOffsets(builder.create<mlir::XOrOp>(loc, val, c1),
                            /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Not: {
    assert(sub.val);
    mlir::Value val = sub.getValue(builder);

    if (!val.getType().isa<mlir::IntegerType>()) {
      U->dump();
      val.dump();
    }
    auto ty = val.getType().cast<mlir::IntegerType>();
    auto c1 = builder.create<mlir::ConstantOp>(
        loc, ty,
        builder.getIntegerAttr(ty, APInt::getAllOnesValue(ty.getWidth())));
    return ValueWithOffsets(builder.create<mlir::XOrOp>(loc, val, c1),
                            /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Deref: {
    auto dref = sub.dereference(builder);
    return dref;
  }
  case clang::UnaryOperator::Opcode::UO_AddrOf: {
    assert(sub.isReference);
    if (sub.val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return ValueWithOffsets(sub.val, /*isReference*/ false);
    }

    bool isArray = false;
    Glob.getMLIRType(U->getSubExpr()->getType(), &isArray);
    auto mt = sub.val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    mlir::Value res;
    shape[0] = -1;
    auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                     mt.getAffineMaps(), mt.getMemorySpace());
    res = builder.create<memref::CastOp>(loc, sub.val, mt0);
    return ValueWithOffsets(res,
                            /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Plus: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_Minus: {
    auto ty = getMLIRType(U->getType());
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      return ValueWithOffsets(
          builder.create<mlir::NegFOp>(loc, sub.getValue(builder)),
          /*isReference*/ false);
    } else {
      return ValueWithOffsets(builder.create<mlir::SubIOp>(
                                  loc,
                                  builder.create<mlir::ConstantIntOp>(
                                      loc, 0, ty.cast<mlir::IntegerType>()),
                                  sub.getValue(builder)),
                              /*isReference*/ false);
    }
  }
  case clang::UnaryOperator::Opcode::UO_PreInc:
  case clang::UnaryOperator::Opcode::UO_PostInc: {
    auto ty = getMLIRType(U->getType());
    assert(sub.isReference);
    auto prev = sub.getValue(builder);

    mlir::Value next;
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      if (prev.getType() != ty) {
        U->dump();
        llvm::errs() << " ty: " << ty << "prev: " << prev << "\n";
      }
      assert(prev.getType() == ty);
      next = builder.create<mlir::AddFOp>(
          loc, prev,
          builder.create<mlir::ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else if (auto mt = ty.dyn_cast<MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       mt.getAffineMaps(), mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(1));
    } else if (auto pt = ty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      next = builder.create<LLVM::GEPOp>(
          loc, pt,
          std::vector<mlir::Value>(
              {prev, builder.create<mlir::ConstantOp>(
                         loc, ity, builder.getIntegerAttr(ity, 1))}));
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
      next = builder.create<mlir::AddIOp>(
          loc, prev,
          builder.create<mlir::ConstantIntOp>(loc, 1,
                                              ty.cast<mlir::IntegerType>()));
    }
    sub.store(builder, next);
    return ValueWithOffsets(
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? prev
                                                                     : next,
        /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_PreDec:
  case clang::UnaryOperator::Opcode::UO_PostDec: {
    auto ty = getMLIRType(U->getType());
    assert(sub.isReference);
    auto prev = sub.getValue(builder);

    mlir::Value next;
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      next = builder.create<mlir::SubFOp>(
          loc, prev,
          builder.create<mlir::ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else if (auto pt = ty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      next = builder.create<LLVM::GEPOp>(
          loc, pt,
          std::vector<mlir::Value>(
              {prev, builder.create<mlir::ConstantOp>(
                         loc, ity, builder.getIntegerAttr(ity, -1))}));
    } else if (auto mt = ty.dyn_cast<MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      // Technically not legal per the -1
      llvm::errs() << "postdec of memref subindex doing illegal op\n";
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       mt.getAffineMaps(), mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(-1));
    } else {
      if (!ty.isa<mlir::IntegerType>()) {
        llvm::errs() << ty << " - " << prev << "\n";
        U->dump();
      }
      next = builder.create<mlir::SubIOp>(
          loc, prev,
          builder.create<mlir::ConstantIntOp>(loc, 1,
                                              ty.cast<mlir::IntegerType>()));
    }
    sub.store(builder, next);
    return ValueWithOffsets(
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? prev
                                                                     : next,
        /*isReference*/ false);
  }
  default: {
    U->dump();
    assert(0 && "unhandled opcode");
  }
  }
}

ValueWithOffsets MLIRScanner::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *expr) {
  return Visit(expr->getReplacement());
}

ValueWithOffsets
MLIRScanner::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Uop) {
  switch (Uop->getKind()) {
  case UETT_SizeOf: {
    auto value = getTypeSize(Uop->getTypeOfArgument());
    auto ty = getMLIRType(Uop->getType()).cast<mlir::IntegerType>();
    return ValueWithOffsets(builder.create<mlir::ConstantOp>(
                                loc, ty, builder.getIntegerAttr(ty, value)),
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
  if (isa<mlir::IndexCastOp>(op))
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
        if (!isa<mlir::IndexCastOp>(maybeIndexCast))
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
  if ((!isa<mlir::AddIOp>(op)) && (!isa<mlir::MulIOp>(op))) {
    return false;
  }

  // make sure that the current op has at least one constant operand
  // (ConstantIndexOp or ConstantIntOp)
  if (llvm::none_of(op->getOperands(), [](mlir::Value operand) {
        return (isa<mlir::ConstantIndexOp>(operand.getDefiningOp()) ||
                isa<mlir::ConstantIntOp>(operand.getDefiningOp()));
      }))
    return false;

  // build affine expression by adding or multiplying constants.
  // and keep iterating on the non-constant index
  mlir::Value nonCstOperand = nullptr;
  for (auto operand : op->getOperands()) {
    if (auto constantIndexOp =
            dyn_cast<mlir::ConstantIndexOp>(operand.getDefiningOp())) {
      if (isa<mlir::AddIOp>(op))
        expr = expr + constantIndexOp.getValue();
      else
        expr = expr * constantIndexOp.getValue();
    } else if (auto constantIntOp =
                   dyn_cast<mlir::ConstantIntOp>(operand.getDefiningOp())) {
      if (isa<mlir::AddIOp>(op))
        expr = expr + constantIntOp.getValue();
      else
        expr = expr * constantIntOp.getValue();
    } else
      nonCstOperand = operand;
  }
  return hasAffineArith(nonCstOperand.getDefiningOp(), expr, affineForIndVar);
}

ValueWithOffsets MLIRScanner::VisitBinaryOperator(clang::BinaryOperator *BO) {
  auto lhs = Visit(BO->getLHS());
  if (!lhs.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->dump();
    BO->getLHS()->dump();
    assert(lhs.val);
  }
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
        res = builder.create<mlir::SignExtendIOp>(loc, res, postTy);
      } else {
        res = builder.create<mlir::ZeroExtendIOp>(loc, res, postTy);
      }
    }
    return ValueWithOffsets(res, /*isReference*/ false);
  };

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
      auto postTy = builder.getI1Type();
      cond = builder.create<mlir::TruncateIOp>(loc, cond, postTy);
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.thenRegion().back());

    auto rhs = Visit(BO->getRHS()).getValue(builder);
    assert(rhs != nullptr);
    if (auto LT = rhs.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      rhs = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, rhs, nullptr_llvm);
    }
    if (!rhs.getType().cast<mlir::IntegerType>().isInteger(1)) {
      auto postTy = builder.getI1Type();
      rhs = builder.create<mlir::TruncateIOp>(loc, rhs, postTy);
    }
    mlir::Value truearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.elseRegion().back());
    mlir::Value falsearray[] = {builder.create<mlir::ConstantOp>(
        loc, types[0], builder.getIntegerAttr(types[0], 0))};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);
    return fixInteger(ifOp.getResult(0));
  }
  case clang::BinaryOperator::Opcode::BO_LOr: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto cond = lhs.getValue(builder);
    auto prevTy = cond.getType().cast<mlir::IntegerType>();
    if (!prevTy.isInteger(1)) {
      auto postTy = builder.getI1Type();
      cond = builder.create<mlir::TruncateIOp>(loc, cond, postTy);
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.thenRegion().back());

    mlir::Value truearray[] = {builder.create<mlir::ConstantOp>(
        loc, types[0], builder.getIntegerAttr(types[0], 1))};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.elseRegion().back());
    auto rhs = Visit(BO->getRHS()).getValue(builder);
    if (!rhs.getType().cast<mlir::IntegerType>().isInteger(1)) {
      auto postTy = builder.getI1Type();
      rhs = builder.create<mlir::TruncateIOp>(loc, rhs, postTy);
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
    if (signedType)
      return ValueWithOffsets(
          builder.create<mlir::SignedShiftRightOp>(loc, lhs.getValue(builder),
                                                   rhs.getValue(builder)),
          /*isReference*/ false);
    else
      return ValueWithOffsets(
          builder.create<mlir::UnsignedShiftRightOp>(loc, lhs.getValue(builder),
                                                     rhs.getValue(builder)),
          /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Shl: {
    return ValueWithOffsets(
        builder.create<mlir::ShiftLeftOp>(loc, lhs.getValue(builder),
                                          rhs.getValue(builder)),
        /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_And: {
    return ValueWithOffsets(builder.create<mlir::AndOp>(loc,
                                                        lhs.getValue(builder),
                                                        rhs.getValue(builder)),
                            /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Xor: {
    return ValueWithOffsets(builder.create<mlir::XOrOp>(loc,
                                                        lhs.getValue(builder),
                                                        rhs.getValue(builder)),
                            /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Or: {
    // TODO short circuit
    return ValueWithOffsets(builder.create<mlir::OrOp>(loc,
                                                       lhs.getValue(builder),
                                                       rhs.getValue(builder)),
                            /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_GT: {
    auto lhs_v = lhs.getValue(builder);
    mlir::Value res;
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      res = builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::UGT, lhs_v,
                                         rhs.getValue(builder));
    } else {
      res = builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::sgt : mlir::CmpIPredicate::ugt,
          lhs_v, rhs.getValue(builder));
    }
    return fixInteger(res);
  }
  case clang::BinaryOperator::Opcode::BO_GE: {
    auto lhs_v = lhs.getValue(builder);
    mlir::Value res;
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      res = builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::UGE, lhs_v,
                                         rhs.getValue(builder));
    } else {
      res = builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::sge : mlir::CmpIPredicate::uge,
          lhs_v, rhs.getValue(builder));
    }
    return fixInteger(res);
  }
  case clang::BinaryOperator::Opcode::BO_LT: {
    auto lhs_v = lhs.getValue(builder);
    mlir::Value res;
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      res = builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::ULT, lhs_v,
                                         rhs.getValue(builder));
    } else if (auto pt =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      res = builder.create<LLVM::ICmpOp>( 
          loc, signedType ? mlir::LLVM::ICmpPredicate::slt : mlir::LLVM::ICmpPredicate::ult,
          lhs_v, rhs.getValue(builder));
    } else {
      res = builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::slt : mlir::CmpIPredicate::ult,
          lhs_v, rhs.getValue(builder));
    }
    return fixInteger(res);
  }
  case clang::BinaryOperator::Opcode::BO_LE: {
    auto lhs_v = lhs.getValue(builder);
    mlir::Value res;
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      res = builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::ULE, lhs_v,
                                         rhs.getValue(builder));
    } else if (auto pt =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      res = builder.create<LLVM::ICmpOp>( 
          loc, signedType ? mlir::LLVM::ICmpPredicate::sle : mlir::LLVM::ICmpPredicate::ule,
          lhs_v, rhs.getValue(builder));
    } else {
      res = builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::sle : mlir::CmpIPredicate::ule,
          lhs_v, rhs.getValue(builder));
    }
    return fixInteger(res);
  }
  case clang::BinaryOperator::Opcode::BO_EQ: {
    auto lhs_v = lhs.getValue(builder);
    mlir::Value res;
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      res = builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::UEQ, lhs_v,
                                         rhs.getValue(builder));
    } else if (auto pt =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      res = builder.create<LLVM::ICmpOp>( 
          loc, mlir::LLVM::ICmpPredicate::eq,
          lhs_v, rhs.getValue(builder));
    } else {
      res = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, lhs_v,
                                         rhs.getValue(builder));
    }
    return fixInteger(res);
  }
  case clang::BinaryOperator::Opcode::BO_NE: {
    auto lhs_v = lhs.getValue(builder);
    mlir::Value res;
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      res = builder.create<mlir::CmpFOp>(loc, mlir::CmpFPredicate::UNE, lhs_v,
                                         rhs.getValue(builder));
    } else if (auto pt =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      res = builder.create<LLVM::ICmpOp>( 
          loc, mlir::LLVM::ICmpPredicate::ne,
          lhs_v, rhs.getValue(builder));
    } else {
      res = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ne, lhs_v,
                                         rhs.getValue(builder));
    }
    return fixInteger(res);
  }
  case clang::BinaryOperator::Opcode::BO_Mul: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(
          builder.create<mlir::MulFOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    } else {
      return ValueWithOffsets(
          builder.create<mlir::MulIOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Div: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(
          builder.create<mlir::DivFOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
      ;
    } else {
      if (signedType)
        return ValueWithOffsets(builder.create<mlir::SignedDivIOp>(
                                    loc, lhs_v, rhs.getValue(builder)),
                                /*isReference*/ false);
      else
        return ValueWithOffsets(builder.create<mlir::UnsignedDivIOp>(
                                    loc, lhs_v, rhs.getValue(builder)),
                                /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Rem: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(
          builder.create<mlir::RemFOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    } else {
      if (signedType)
        return ValueWithOffsets(builder.create<mlir::SignedRemIOp>(
                                    loc, lhs_v, rhs.getValue(builder)),
                                /*isReference*/ false);
      else
        return ValueWithOffsets(builder.create<mlir::UnsignedRemIOp>(
                                    loc, lhs_v, rhs.getValue(builder)),
                                /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Add: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(
          builder.create<mlir::AddFOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    } else if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       mt.getAffineMaps(), mt.getMemorySpace());
      auto ptradd = rhs.getValue(builder);
      ptradd = castToIndex(loc, ptradd);
      return ValueWithOffsets(
          builder.create<polygeist::SubIndexOp>(loc, mt0, lhs_v, ptradd),
          /*isReference*/ false);
    } else if (auto pt =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      return ValueWithOffsets(
          builder.create<LLVM::GEPOp>(
              loc, pt,
              std::vector<mlir::Value>({lhs_v, rhs.getValue(builder)})),
          /*isReference*/ false);
    } else {
      return ValueWithOffsets(
          builder.create<mlir::AddIOp>(loc, lhs_v, rhs.getValue(builder)),
          /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Sub: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      auto right = rhs.getValue(builder);
      assert(right.getType() == lhs_v.getType());
      return ValueWithOffsets(builder.create<mlir::SubFOp>(loc, lhs_v, right),
                              /*isReference*/ false);
    } else if (auto pt =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      return ValueWithOffsets(
          builder.create<mlir::SubIOp>(
              loc,
              builder.create<LLVM::PtrToIntOp>(loc, getMLIRType(BO->getType()),
                                               lhs_v),
              builder.create<LLVM::PtrToIntOp>(loc, getMLIRType(BO->getType()),
                                               rhs.getValue(builder))),
          /*isReference*/ false);
    } else if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      llvm::errs() << " memref ptrtoint: " << mt << "\n";
      return ValueWithOffsets(
          builder.create<mlir::SubIOp>(
              loc,
              builder.create<LLVM::PtrToIntOp>(loc, getMLIRType(BO->getType()),
                                               lhs_v),
              builder.create<LLVM::PtrToIntOp>(loc, getMLIRType(BO->getType()),
                                               rhs.getValue(builder))),
          /*isReference*/ false);
    } else {
      return ValueWithOffsets(
          builder.create<mlir::SubIOp>(loc, lhs_v, rhs.getValue(builder)),
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
              tostore =
                  builder.create<mlir::SignExtendIOp>(loc, tostore, postTy);
            } else {
              tostore =
                  builder.create<mlir::ZeroExtendIOp>(loc, tostore, postTy);
            }
          } else if (prevTy.getWidth() > postTy.getWidth()) {
            tostore = builder.create<mlir::TruncateIOp>(loc, tostore, postTy);
          }
        }
      }
    }
    if (tostore.getType() != subType ||
        (lhs.val.getType().isa<MemRefType>() &&
         lhs.val.getType().cast<MemRefType>().getShape().size() != 1)) {
      BO->dump();
      function.dump();
      llvm::errs() << " lhs.val: " << lhs.val << "\n";
      llvm::errs() << " tostore: " << tostore << "\n";
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
        rhsV = builder.create<mlir::FPExtOp>(loc, rhsV, postTy);
      } else {
        rhsV = builder.create<mlir::FPTruncOp>(loc, rhsV, postTy);
      }
      assert(rhsV.getType() == prev.getType());
      result = builder.create<mlir::AddFOp>(loc, prev, rhsV);
    } else if (auto pt =
                   prev.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      result = builder.create<LLVM::GEPOp>(
          loc, pt, std::vector<mlir::Value>({prev, rhs.getValue(builder)}));
    } else {
      result = builder.create<mlir::AddIOp>(loc, prev, rhs.getValue(builder));
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
          right = builder.create<mlir::FPExtOp>(loc, right, postTy);
        } else {
          right = builder.create<mlir::FPTruncOp>(loc, right, postTy);
        }
      }
      if (right.getType() != prev.getType()) {
        BO->dump();
        llvm::errs() << " p:" << prev << " r:" << right << "\n";
      }
      assert(right.getType() == prev.getType());
      result = builder.create<mlir::SubFOp>(loc, prev, right);
    } else {
      result = builder.create<mlir::SubIOp>(loc, prev, rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_MulAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<mlir::MulFOp>(loc, prev, rhs.getValue(builder));
    } else {
      result = builder.create<mlir::MulIOp>(loc, prev, rhs.getValue(builder));
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
          val = builder.create<mlir::FPExtOp>(loc, val, postTy);
        } else if (prevTy.getWidth() > postTy.getWidth()) {
          val = builder.create<mlir::FPTruncOp>(loc, val, postTy);
        }
      result = builder.create<mlir::DivFOp>(loc, prev, val);
    } else {
      if (signedType)
        result = builder.create<mlir::SignedDivIOp>(loc, prev,
                                                    rhs.getValue(builder));
      else
        result = builder.create<mlir::UnsignedDivIOp>(loc, prev,
                                                      rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_ShrAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;

    if (signedType)
      result = builder.create<mlir::SignedShiftRightOp>(loc, prev,
                                                        rhs.getValue(builder));
    else
      result = builder.create<mlir::UnsignedShiftRightOp>(
          loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_ShlAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result =
        builder.create<mlir::ShiftLeftOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_RemAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;

    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<mlir::RemFOp>(loc, prev, rhs.getValue(builder));
    } else {
      if (signedType)
        result = builder.create<mlir::SignedRemIOp>(loc, prev,
                                                    rhs.getValue(builder));
      else
        result = builder.create<mlir::UnsignedRemIOp>(loc, prev,
                                                      rhs.getValue(builder));
    }
    lhs.store(builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_OrAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result =
        builder.create<mlir::OrOp>(loc, prev, rhs.getValue(builder));
    lhs.store(builder, result);
    return lhs;
  }

  default: {
    BO->dump();
    assert(0 && "unhandled opcode");
  }
  }
}

ValueWithOffsets MLIRScanner::VisitAttributedStmt(AttributedStmt *AS) {
  llvm::errs() << "warning ignoring attributes\n";
  return Visit(AS->getSubStmt());
}

ValueWithOffsets MLIRScanner::VisitExprWithCleanups(ExprWithCleanups *E) {
  auto ret = Visit(E->getSubExpr());
  for (auto &child : E->children()) {
    child->dump();
    llvm::errs() << "cleanup not handled\n";
  }
  return ret;
}

ValueWithOffsets MLIRScanner::CommonFieldLookup(clang::QualType CT,
                                                const FieldDecl *FD,
                                                mlir::Value val) {
  auto rd = FD->getParent();

  auto ST = cast<llvm::StructType>(getLLVMType(CT));

  size_t fnum = 0;
  auto CXRD = dyn_cast<CXXRecordDecl>(rd);
  if ((CXRD && (!CXRD->hasDefinition() || CXRD->isPolymorphic() ||
                CXRD->getNumBases() > 0)) ||
      rd->isUnion()) {
    auto &layout = Glob.CGM.getTypes().getCGRecordLayout(rd);
    fnum = layout.getLLVMFieldNo(FD);
  } else {
// TODO
#if 0
    bool recursive = false;
    SmallPtrSet<llvm::Type*, 4> Seen;
    for (size_t i = 0; i < ST->getNumElements(); i++) {
      if (isRecursiveStruct(ST->getTypeAtIndex(i), ST, Seen)) {
        recursive = true;
      }
    }

      if (ST->getNumElements() == 1 && !recursive &&
            !rd->fields().empty() && ++rd->field_begin() == rd->field_end()) {
        return ValueWithOffsets(val, /*isReference*/true);
      }
#endif
    fnum = 0;
    for (auto field : rd->fields()) {
      if (field == FD) {
        break;
      }
      fnum++;
    }
  }

  if (auto PT = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    auto iTy = builder.getIntegerType(32);
    mlir::Value vec[3] = {val,
                          builder.create<mlir::ConstantOp>(
                              loc, iTy, builder.getIntegerAttr(iTy, 0)),
                          builder.create<mlir::ConstantOp>(
                              loc, iTy, builder.getIntegerAttr(iTy, fnum))};
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
      auto subType = Glob.typeTranslator.translateType(getLLVMType(FD->getType()));
      commonGEP = builder.create<mlir::LLVM::BitcastOp>(
          loc,
          mlir::LLVM::LLVMPointerType::get(subType, PT.getAddressSpace()),
          commonGEP);
    }
    return ValueWithOffsets(commonGEP, /*isReference*/ true);
  }
  auto mt = val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() > 1) {
    shape.erase(shape.begin());
  } else {
    shape[0] = -1;
  }
  auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                   mt.getAffineMaps(), mt.getMemorySpace());
  shape[0] = -1;
  auto mt1 = mlir::MemRefType::get(shape, mt.getElementType(),
                                   mt.getAffineMaps(), mt.getMemorySpace());
  mlir::Value sub0 =
      builder.create<polygeist::SubIndexOp>(loc, mt0, val, getConstantIndex(0));
  mlir::Value sub1 = builder.create<polygeist::SubIndexOp>(
      loc, mt1, sub0, getConstantIndex(fnum));
  return ValueWithOffsets(sub1, /*isReference*/ true);
}

ValueWithOffsets MLIRScanner::VisitDeclRefExpr(DeclRefExpr *E) {
  auto name = E->getDecl()->getName().str();
  assert(!isa<FunctionDecl>(E->getDecl()));

  if (auto VD = dyn_cast<VarDecl>(E->getDecl())) {
    if (Captures.find(VD) != Captures.end()) {
      FieldDecl *field = Captures[VD];
      auto res = CommonFieldLookup(
          cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(), field,
          ThisVal.val);
      assert(CaptureKinds.find(VD) != CaptureKinds.end());
      if (CaptureKinds[VD] == LambdaCaptureKind::LCK_ByRef)
        res = res.dereference(builder);
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
    return ValueWithOffsets(
        builder.create<mlir::ConstantOp>(
            loc, ty, builder.getIntegerAttr(ty, ED->getInitVal())),
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
      return ValueWithOffsets(builder.create<mlir::LLVM::AddressOfOp>(
                                  loc, Glob.GetOrCreateLLVMGlobal(VD)),
                              /*isReference*/ true);
    }

    auto gv = Glob.GetOrCreateGlobal(VD);
    auto gv2 = builder.create<memref::GetGlobalOp>(loc, gv.first.type(),
                                                   gv.first.getName());
    bool isArray = gv.second;
    // TODO check reference
    if (isArray)
      return ValueWithOffsets(gv2, /*isReference*/ true);
    else
      return ValueWithOffsets(gv2, /*isReference*/ true);
    // return gv2;
  }
  E->dump();
  E->getDecl()->dump();
  llvm::errs() << "couldn't find " << name << "\n";
  assert(0 && "couldnt find value");
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
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

ValueWithOffsets
MLIRScanner::VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr) {
  assert(ThisVal.val);
  auto toset = Visit(expr->getExpr());
  assert(!ThisVal.isReference);
  assert(toset.val);

  bool isArray = false;
  Glob.getMLIRType(expr->getExpr()->getType(), &isArray);

  auto cfl = CommonFieldLookup(
      cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(),
      expr->getField(), ThisVal.val);
  assert(cfl.val);
  cfl.store(builder, toset, isArray);
  return cfl;
}

ValueWithOffsets MLIRScanner::VisitMemberExpr(MemberExpr *ME) {
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
  return CommonFieldLookup(OT, field, base.val);
}

ValueWithOffsets MLIRScanner::VisitCastExpr(CastExpr *E) {
  switch (E->getCastKind()) {

  case clang::CastKind::CK_NullToPointer: {
    auto llvmType =
        Glob.typeTranslator.translateType(anonymize(getLLVMType(E->getType())));
    return ValueWithOffsets(builder.create<mlir::LLVM::NullOp>(loc, llvmType),
                            /*isReference*/ false);
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
      auto pt = Glob.typeTranslator.translateType(
          anonymize(getLLVMType(E->getType())));
      if (se.isReference)
        pt = mlir::LLVM::LLVMPointerType::get(pt, opt.getAddressSpace());
      auto nval = builder.create<mlir::LLVM::BitcastOp>(loc, pt, se.val);
      return ValueWithOffsets(nval, /*isReference*/ se.isReference);
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
    auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                    ut.getAffineMaps(), ut.getMemorySpace());
    return ValueWithOffsets(
        builder.create<mlir::memref::CastOp>(loc, se.val, ty),
        /*isReference*/ se.isReference);
  }
  case clang::CastKind::CK_BitCast: {

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getName() == "polybench_alloc_data") {
            auto mt = getMLIRType(E->getType()).cast<mlir::MemRefType>();

            auto shape = std::vector<int64_t>(mt.getShape());
            // shape.erase(shape.begin());
            auto mt0 =
                mlir::MemRefType::get(shape, mt.getElementType(),
                                      mt.getAffineMaps(), mt.getMemorySpace());

            auto alloc = builder.create<mlir::memref::AllocOp>(loc, mt0);
            return ValueWithOffsets(alloc, /*isReference*/ false);
          }
        }

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getName() == "malloc" ||
              sr->getDecl()->getName() == "calloc")
            if (auto mt =
                    getMLIRType(E->getType()).dyn_cast<mlir::MemRefType>()) {
              auto shape = std::vector<int64_t>(mt.getShape());

              auto elemSize =
                  getTypeSize(cast<clang::PointerType>(
                                  E->getType()->getUnqualifiedDesugaredType())
                                  ->getPointeeType());
              mlir::Value allocSize = builder.create<mlir::IndexCastOp>(
                  loc, Visit(CI->getArg(0)).getValue(builder),
                  mlir::IndexType::get(builder.getContext()));
              if (sr->getDecl()->getName() == "calloc") {
                allocSize = builder.create<mlir::MulIOp>(
                    loc, allocSize,
                    builder.create<mlir::IndexCastOp>(
                        loc, Visit(CI->getArg(1)).getValue(builder),
                        mlir::IndexType::get(builder.getContext())));
              }
              mlir::Value args[1] = {builder.create<mlir::UnsignedDivIOp>(
                  loc, allocSize,
                  builder.create<mlir::ConstantOp>(
                      loc, allocSize.getType(),
                      builder.getIntegerAttr(allocSize.getType(), elemSize)))};
              auto alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
              if (sr->getDecl()->getName() == "calloc") {
                  mlir::Value toStore;
                  auto melem = mt.getElementType();
                  if (melem.isa<mlir::IntegerType>())
                    toStore = builder.create<mlir::ConstantOp>(
                        loc, melem, builder.getIntegerAttr(melem, 0));
                  else
                    toStore = builder.create<mlir::ConstantOp>(
                        loc, melem, builder.getFloatAttr(melem, 0));
                  auto affineOp = builder.create<scf::ForOp>(
                      loc, getConstantIndex(0), args[0], getConstantIndex(1));

                  auto oldpoint = builder.getInsertionPoint();
                  auto oldblock = builder.getInsertionBlock();

                  std::vector<mlir::Value> stargs = {affineOp.getInductionVar()};

                  builder.setInsertionPointToStart(&affineOp.getLoopBody().front());
                  builder.create<memref::StoreOp>(loc, toStore, alloc, stargs);

                  builder.setInsertionPoint(oldblock, oldpoint);
              }
              return ValueWithOffsets(alloc, /*isReference*/ false);
            }
        }
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    auto scalar = se.getValue(builder);
    if (scalar.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      auto pt = Glob.typeTranslator.translateType(
          anonymize(getLLVMType(E->getType())));
      auto nval = builder.create<mlir::LLVM::BitcastOp>(loc, pt, scalar);
      return ValueWithOffsets(nval, /*isReference*/ false);
    }
    if (!scalar.getType().isa<mlir::MemRefType>()) {
      E->dump();
      E->getType()->dump();
      llvm::errs() << "scalar: " << scalar << "\n";
    }
    auto ut = scalar.getType().cast<mlir::MemRefType>();
    auto mlirty = getMLIRType(E->getType());

    if (auto PT = mlirty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      mlirty = mlir::MemRefType::get(-1, PT.getElementType());
    } else if (!mlirty.isa<mlir::MemRefType>()) {
      E->dump();
      E->getType()->dump();
      llvm::errs() << " scalar: " << scalar << " mlirty: " << mlirty << "\n";
    }
    auto mt = mlirty.cast<mlir::MemRefType>();
    auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                    ut.getAffineMaps(), ut.getMemorySpace());
    if (ut.getShape().size() == mt.getShape().size() + 1) {
      return ValueWithOffsets(builder.create<mlir::polygeist::SubIndexOp>(
                                  loc, ty, scalar, getConstantIndex(0)),
                              /*isReference*/ false);
    }
    return ValueWithOffsets(
        builder.create<mlir::memref::CastOp>(loc, scalar, ty),
        /*isReference*/ false);
  }
  case clang::CastKind::CK_LValueToRValue: {
    if (auto dr = dyn_cast<DeclRefExpr>(E->getSubExpr())) {
      if (dr->getDecl()->getName() == "warpSize") {
        bool foundVal = false;
        /*
        for (int i = scopes.size() - 1; i >= 0; i--) {
          auto found = scopes[i].find("warpSize");
          if (found != scopes[i].end()) {
            foundVal = true;
            break;
          }
        }
        */
        if (!foundVal) {
          auto mlirType = getMLIRType(E->getType());
          return ValueWithOffsets(
              builder.create<mlir::NVVM::WarpSizeOp>(loc, mlirType),
              /*isReference*/ false);
        }
      }
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
    return ValueWithOffsets(lres, /*isReference*/ false);
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
      return ValueWithOffsets(builder.create<mlir::SIToFPOp>(loc, scalar, ty),
                              /*isReference*/ false);
    else
      return ValueWithOffsets(builder.create<mlir::UIToFPOp>(loc, scalar, ty),
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
      return ValueWithOffsets(builder.create<mlir::FPToSIOp>(loc, scalar, ty),
                              /*isReference*/ false);
    else
      return ValueWithOffsets(builder.create<mlir::FPToUIOp>(loc, scalar, ty),
                              /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    auto postTy = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    if (scalar.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return ValueWithOffsets(
          builder.create<mlir::LLVM::PtrToIntOp>(loc, postTy, scalar),
          /*isReference*/ false);
    }
    if (scalar.getType().isa<mlir::IndexType>() ||
        postTy.isa<mlir::IndexType>()) {
      return ValueWithOffsets(builder.create<IndexCastOp>(loc, scalar, postTy),
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
      return ValueWithOffsets(scalar, /*isReference*/ false);
    if (prevTy.getWidth() < postTy.getWidth()) {
      if (signedType) {
        return ValueWithOffsets(
            builder.create<mlir::SignExtendIOp>(loc, scalar, postTy),
            /*isReference*/ false);
      } else {
        return ValueWithOffsets(
            builder.create<mlir::ZeroExtendIOp>(loc, scalar, postTy),
            /*isReference*/ false);
      }
    } else {
      return ValueWithOffsets(
          builder.create<mlir::TruncateIOp>(loc, scalar, postTy),
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
      return ValueWithOffsets(scalar, /*isReference*/ false);
    if (prevTy.getWidth() < postTy.getWidth()) {
      return ValueWithOffsets(
          builder.create<mlir::FPExtOp>(loc, scalar, postTy),
          /*isReference*/ false);
    } else {
      return ValueWithOffsets(
          builder.create<mlir::FPTruncOp>(loc, scalar, postTy),
          /*isReference*/ false);
    }
  }
  case clang::CastKind::CK_ArrayToPointerDecay: {
    if (auto SR = dyn_cast<clang::StringLiteral>(E->getSubExpr())) {
      return ValueWithOffsets(
          Glob.GetOrCreateGlobalLLVMString(loc, builder, SR->getString()),
          /*isReference*/ false);
    }
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
                                     mt.getAffineMaps(), mt.getMemorySpace());
    auto cst = builder.create<mlir::MemRefCastOp>(loc, scalar.val, nex);
    //llvm::errs() << "<ArrayToPtrDecay>\n";
    //E->dump();
    //llvm::errs() << cst << " - " << scalar.val << "\n";
    //auto offs = scalar.offsets;
    //offs.push_back(getConstantIndex(0));
    return ValueWithOffsets(cst, scalar.isReference);
#endif
  }
  case clang::CastKind::CK_FunctionToPointerDecay: {
    auto scalar = Visit(E->getSubExpr());
    assert(scalar.isReference);
    return ValueWithOffsets(scalar.val, /*isReference*/ false);
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
      return ValueWithOffsets(ne, /*isReference*/ false);
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
      return ValueWithOffsets(val, /*isReference*/ false);
    }
    function.dump();
    llvm::errs() << "scalar: " << scalar << "\n";
    E->dump();
    assert(0 && "unhandled ptrtoint cast");
  }
  case clang::CastKind::CK_IntegralToBoolean: {
    auto res = Visit(E->getSubExpr()).getValue(builder);
    auto prevTy = res.getType().cast<mlir::IntegerType>();
    auto postTy = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (prevTy.getWidth() < postTy.getWidth()) {
      if (signedType) {
        res = builder.create<mlir::SignExtendIOp>(loc, res, postTy);
      } else {
        res = builder.create<mlir::ZeroExtendIOp>(loc, res, postTy);
      }
    } else if (prevTy.getWidth() > postTy.getWidth()) {
      res = builder.create<mlir::TruncateIOp>(loc, res, postTy);
    }
    return ValueWithOffsets(res, /*isReference*/ false);
  }
  default:
    EmittingFunctionDecl->dump();
    E->dump();
    assert(0 && "unhandled cast");
  }
}

ValueWithOffsets MLIRScanner::VisitIfStmt(clang::IfStmt *stmt) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(stmt->getIfLoc());
  auto cond = Visit(stmt->getCond()).getValue(builder);
  assert(cond != nullptr);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();
  if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
    cond = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
  }
  auto prevTy = cond.getType().cast<mlir::IntegerType>();
  if (!prevTy.isInteger(1)) {
    auto postTy = builder.getI1Type();
    cond = builder.create<mlir::TruncateIOp>(loc, cond, postTy);
  }
  bool hasElseRegion = stmt->getElse();
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond, hasElseRegion);

  ifOp.thenRegion().back().clear();
  builder.setInsertionPointToStart(&ifOp.thenRegion().back());
  Visit(stmt->getThen());
  builder.create<scf::YieldOp>(loc);
  if (hasElseRegion) {
    ifOp.elseRegion().back().clear();
    builder.setInsertionPointToStart(&ifOp.elseRegion().back());
    Visit(stmt->getElse());
    builder.create<scf::YieldOp>(loc);
  }

  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitSwitchStmt(clang::SwitchStmt *stmt) {
  IfScope scope(*this);
  auto cond = Visit(stmt->getCond()).getValue(builder);
  assert(cond != nullptr);
  SmallVector<int32_t> caseVals;

  auto er = builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  er.region().push_back(new Block());
  auto oldpoint2 = builder.getInsertionPoint();
  auto oldblock2 = builder.getInsertionBlock();

  auto &exitB = *(new Block());
  builder.setInsertionPointToStart(&exitB);
  builder.create<scf::YieldOp>(loc);

  SmallVector<Block *> blocks;
  bool inCase = false;

  for (auto cse : stmt->getBody()->children()) {
    if (auto cses = dyn_cast<CaseStmt>(cse)) {
      auto &condB = *(new Block());

      caseVals.push_back((int32_t)Visit(cses->getLHS())
                             .getValue(builder)
                             .getDefiningOp<ConstantIntOp>()
                             .getValue());

      if (inCase) {
        auto noBreak =
            builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak);
        builder.create<mlir::CondBranchOp>(loc, noBreak, &condB, &exitB);
        loops.pop_back();
      }

      inCase = true;
      er.region().getBlocks().push_back(&condB);
      blocks.push_back(&condB);
      builder.setInsertionPointToStart(&condB);

      auto i1Ty = builder.getIntegerType(1);
      auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
      auto truev = builder.create<mlir::ConstantIntOp>(loc, true, 1);
      loops.push_back(
          (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                        builder.create<mlir::memref::AllocaOp>(loc, type)});
      builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);
      builder.create<mlir::memref::StoreOp>(loc, truev,
                                            loops.back().keepRunning);
      Visit(cses->getSubStmt());
    } else {
      Visit(cse);
    }
  }

  if (inCase)
    loops.pop_back();
  builder.create<mlir::BranchOp>(loc, &exitB);

  er.region().getBlocks().push_back(&exitB);

  DenseIntElementsAttr caseValuesAttr;
  ShapedType caseValueType = mlir::VectorType::get(
      static_cast<int64_t>(caseVals.size()), cond.getType());
  caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals);

  builder.setInsertionPointToStart(&er.region().front());
  builder.create<mlir::SwitchOp>(
      loc, cond, &exitB, ArrayRef<mlir::Value>(), caseValuesAttr, blocks,
      SmallVector<mlir::ValueRange>(caseVals.size(), ArrayRef<mlir::Value>()));
  builder.setInsertionPoint(oldblock2, oldpoint2);
  return nullptr;
}

ValueWithOffsets
MLIRScanner::VisitConditionalOperator(clang::ConditionalOperator *E) {
  auto cond = Visit(E->getCond()).getValue(builder);
  assert(cond != nullptr);
  auto prevTy = cond.getType().cast<mlir::IntegerType>();
  if (!prevTy.isInteger(1)) {
    auto postTy = builder.getI1Type();
    cond = builder.create<mlir::TruncateIOp>(loc, cond, postTy);
  }
  std::vector<mlir::Type> types;
  if (!E->getType()->isVoidType())
    types.push_back(getMLIRType(E->getType()));
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                              /*hasElseRegion*/ true);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();
  builder.setInsertionPointToStart(&ifOp.thenRegion().back());

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
    } else
      truev = trueExpr.getValue(builder);
    assert(truev != nullptr);
    truearray.push_back(truev);
    builder.create<mlir::scf::YieldOp>(loc, truearray);
  }

  builder.setInsertionPointToStart(&ifOp.elseRegion().back());

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
  newIfOp.thenRegion().takeBody(ifOp.thenRegion());
  newIfOp.elseRegion().takeBody(ifOp.elseRegion());
  return ValueWithOffsets(newIfOp.getResult(0), /*isReference*/ isReference);
  // return ifOp;
}

ValueWithOffsets MLIRScanner::VisitCompoundStmt(clang::CompoundStmt *stmt) {
  for (auto a : stmt->children()) {
    IfScope scope(*this);
    Visit(a);
  }
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitStmtExpr(clang::StmtExpr *stmt) {
  ValueWithOffsets off = nullptr;
  for (auto a : stmt->getSubStmt()->children()) {
    IfScope scope(*this);
    off = Visit(a);
  }
  return off;
}

ValueWithOffsets MLIRScanner::VisitBreakStmt(clang::BreakStmt *stmt) {
  IfScope scope(*this);
  assert(loops.size());
  assert(loops.back().keepRunning);
  assert(loops.back().noBreak);
  auto i1Ty = builder.getI1Type();
  auto vfalse = builder.create<mlir::ConstantOp>(
      builder.getUnknownLoc(), i1Ty, builder.getIntegerAttr(i1Ty, 0));
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().keepRunning);
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().noBreak);

  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitContinueStmt(clang::ContinueStmt *stmt) {
  IfScope scope(*this);
  assert(loops.size());
  assert(loops.back().keepRunning);
  auto i1Ty = builder.getI1Type();
  auto vfalse = builder.create<mlir::ConstantOp>(
      builder.getUnknownLoc(), i1Ty, builder.getIntegerAttr(i1Ty, 0));
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().keepRunning);
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitReturnStmt(clang::ReturnStmt *stmt) {
  IfScope scope(*this);
  bool isArrayReturn = false;
  Glob.getMLIRType(EmittingFunctionDecl->getReturnType(), &isArrayReturn);

  if (isArrayReturn) {
    auto rv = Visit(stmt->getRetValue());
    assert(rv.val);
    assert(rv.isReference);
    auto op = function.getArgument(function.getNumArguments() - 1);
    assert(rv.val.getType().cast<MemRefType>().getElementType() ==
           op.getType().cast<MemRefType>().getElementType());
    assert(op.getType().cast<MemRefType>().getShape().size() == 2);
    assert(rv.val.getType().cast<MemRefType>().getShape().size() == 2);
    assert(rv.val.getType().cast<MemRefType>().getShape()[1] ==
           op.getType().cast<MemRefType>().getShape()[1]);

    for (int i = 0; i < op.getType().cast<MemRefType>().getShape()[1]; i++) {
      std::vector<mlir::Value> idx = {getConstantIndex(0), getConstantIndex(i)};
      assert(rv.val.getType().cast<MemRefType>().getShape().size() == 2);
      builder.create<mlir::memref::StoreOp>(
          loc, builder.create<mlir::memref::LoadOp>(loc, rv.val, idx), op, idx);
    }
  } else if (stmt->getRetValue()) {
    auto rv = Visit(stmt->getRetValue());
    if (!stmt->getRetValue()->getType()->isVoidType()) {
      if (!rv.val) {
        stmt->dump();
      }
      assert(rv.val);
      if (stmt->getRetValue()->isLValue() || stmt->getRetValue()->isXValue()) {
        assert(rv.isReference);
        builder.create<mlir::memref::StoreOp>(loc, rv.val, returnVal);
      } else {
        builder.create<mlir::memref::StoreOp>(loc, rv.getValue(builder),
                                              returnVal);
      }
    }
  }

  assert(loops.size());
  auto i1Ty = builder.getI1Type();
  auto vfalse = builder.create<mlir::ConstantOp>(
      builder.getUnknownLoc(), i1Ty, builder.getIntegerAttr(i1Ty, 0));
  for (auto l : loops) {
    builder.create<mlir::memref::StoreOp>(loc, vfalse, l.keepRunning);
    builder.create<mlir::memref::StoreOp>(loc, vfalse, l.noBreak);
  }

  return nullptr;
}

mlir::LLVM::LLVMFuncOp
MLIRASTConsumer::GetOrCreateLLVMFunction(const FunctionDecl *FD) {
  std::string name;
  if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
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

  // Insert the function into the body of the parent module.
  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module.getLoc(), name, llvmFnType);
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

  auto rt = typeTranslator.translateType(anonymize(getLLVMType(FD->getType())));

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());

  auto glob = builder.create<LLVM::GlobalOp>(
      module.getLoc(), rt, /*constant*/ false, lnk, name, mlir::Attribute());

  if (cast<VarDecl>(FD)->isThisDeclarationADefinition() ==
          VarDecl::Definition ||
      cast<VarDecl>(FD)->isThisDeclarationADefinition() ==
          VarDecl::TentativeDefinition) {
    Block *blk = new Block();
    glob.getInitializerRegion().push_back(blk);
    builder.setInsertionPointToStart(blk);
    builder.create<LLVM::ReturnOp>(
        module.getLoc(),
        std::vector<mlir::Value>(
            {builder.create<LLVM::UndefOp>(module.getLoc(), rt)}));
  }
  return llvmGlobals[name] = glob;
}

std::pair<mlir::memref::GlobalOp, bool>
MLIRASTConsumer::GetOrCreateGlobal(const ValueDecl *FD) {
  std::string name = CGM.getMangledName(FD).str();

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
    mr = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                               mt.getAffineMaps(), memspace);
  }

  mlir::SymbolTable::Visibility lnk;
  mlir::Attribute initial_value;

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());

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
      module.getLoc(), builder.getStringAttr(FD->getName()),
      /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(mr),
      initial_value, mlir::UnitAttr());
  SymbolTable::setSymbolVisibility(globalOp, lnk);

  return globals[name] = std::make_pair(globalOp, isArray);
}

mlir::Value MLIRASTConsumer::GetOrCreateGlobalLLVMString(
    mlir::Location loc, mlir::OpBuilder &builder, StringRef value) {
  using namespace mlir;
  // Create the global at the entry of the module.
  if (llvmStringGlobals.find(value.str()) == llvmStringGlobals.end()) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
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
  mlir::Value cst0 = builder.create<mlir::ConstantOp>(
      loc, mlir::IntegerType::get(builder.getContext(), 64),
      builder.getIntegerAttr(builder.getIntegerType(64), 0));
  return builder.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(
          mlir::IntegerType::get(builder.getContext(), 8)),
      globalPtr, ArrayRef<mlir::Value>({cst0, cst0}));
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
  else
    name = CGM.getMangledName(FD).str();

  if (functions.find(name) != functions.end()) {
    return functions[name];
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
                                  mt.getAffineMaps(), mt.getMemorySpace());
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
        auto mt = getMLIRType(
                      CGM.getContext().getLValueReferenceType(parm->getType()))
                      .cast<MemRefType>();
        t = mt; // auto shape = std::vector<int64_t>(mt.getShape());
        // t = mlir::MemRefType::get(shape, mt.getElementType(),
        //                                mt.getAffineMaps(),
        //                                mt.getMemorySpace());
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

    types.push_back(mlir::MemRefType::get(
        shape, mt.getElementType(), mt.getAffineMaps(), mt.getMemorySpace()));
  } else {
    auto rt = getMLIRType(FD->getReturnType());
    if (!rt.isa<mlir::NoneType>()) {
      rettypes.push_back(rt);
    }
  }
  mlir::OpBuilder builder(module.getContext());
  auto funcType = builder.getFunctionType(types, rettypes);
  mlir::FuncOp function = mlir::FuncOp(
      mlir::FuncOp::create(builder.getUnknownLoc(), name, funcType));
  if (FD->getLinkageInternal() == clang::Linkage::InternalLinkage ||
      !FD->isDefined() || FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
    SymbolTable::setSymbolVisibility(function,
                                     SymbolTable::Visibility::Private);
  } else {
    SymbolTable::setSymbolVisibility(function, SymbolTable::Visibility::Public);
  }

  functions[name] = function;
  module.push_back(function);
  const FunctionDecl *Def = nullptr;
  if (FD->isDefined(Def, /*checkforfriend*/ true) && Def->getBody()) {
    assert(Def->getTemplatedKind() !=
           FunctionDecl::TemplatedKind::TK_FunctionTemplate);
    assert(Def->getTemplatedKind() !=
           FunctionDecl::TemplatedKind::
               TK_DependentFunctionTemplateSpecialization);
    functionsToEmit.push_back(Def);
  } else if (FD->getIdentifier())
    emitIfFound.insert(name);
  /*else {
    for (const FunctionDecl *s_FD : FD->redecls()) {
        s_FD->dump();
    }
    FD->dump();
    assert(0 && "unlinkable instruction");
  }*/
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
    else
      name = CGM.getMangledName(FD).str();

    if (done.count(name))
      continue;
    done.insert(name);
    MLIRScanner ms(*this, GetOrCreateMLIRFunction(FD), FD, module, LTInfo);
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
      if (!fd->doesDeclarationForceExternallyVisibleDefinition())
        continue;
    }
    if (!fd->hasBody())
      continue;
    if (fd->getIdentifier() == nullptr)
      continue;

    if (fd->isTemplated()) continue;
    
    bool externLinkage = true;
    LinkageInfo LV = fd->getLinkageAndVisibility();
    if (LV.getLinkage() == clang::Linkage::NoLinkage || LV.getLinkage() == clang::Linkage::InternalLinkage)
        externLinkage = false;
    if (fd->isInlineSpecified()) externLinkage = false;
    
    std::string name;
    if (auto CC = dyn_cast<CXXConstructorDecl>(fd))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
    else
      name = CGM.getMangledName(fd).str();

    // Don't create std functions unless necessary
    if (StringRef(name).startswith("_ZNKSt")) continue;
    if (StringRef(name).startswith("_ZSt")) continue;
    if (StringRef(name).startswith("_ZNSt")) continue;
    if (StringRef(name).startswith("_ZN9__gnu")) continue;

    if ((emitIfFound.count("*") && fd->getName() != "fpclassify" &&
         fd->getName() != "isnan" &&
         !fd->isStatic() && externLinkage) ||
        emitIfFound.count(name)) {
      if (fd->getTemplatedKind() != FunctionDecl::TK_FunctionTemplate) {
        functionsToEmit.push_back(fd);
      }
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
      if (!fd->doesDeclarationForceExternallyVisibleDefinition())
        continue;
    }
    if (!fd->hasBody())
      continue;
    if (fd->getIdentifier() == nullptr)
      continue;
    if (fd->isTemplated()) continue;

    bool externLinkage = true;
    LinkageInfo LV = fd->getLinkageAndVisibility();
    if (LV.getLinkage() == clang::Linkage::NoLinkage || LV.getLinkage() == clang::Linkage::InternalLinkage)
        externLinkage = false;
    if (fd->isInlineSpecified()) externLinkage = false;
    
    std::string name;
    if (auto CC = dyn_cast<CXXConstructorDecl>(fd))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
    else
      name = CGM.getMangledName(fd).str();
    
    // Don't create std functions unless necessary
    if (StringRef(name).startswith("_ZNKSt")) continue;
    if (StringRef(name).startswith("_ZSt")) continue;
    if (StringRef(name).startswith("_ZNSt")) continue;
    if (StringRef(name).startswith("_ZN9__gnu")) continue;

    if ((emitIfFound.count("*") && fd->getName() != "fpclassify" &&
         fd->getName() != "isnan" &&
         !fd->isStatic() && externLinkage) ||
        emitIfFound.count(name)) {
      if (fd->getTemplatedKind() != FunctionDecl::TK_FunctionTemplate) {
        functionsToEmit.push_back(fd);
      }
    } else {
    }
  }

  run();

  return true;
}

mlir::Location MLIRASTConsumer::getMLIRLocation(clang::SourceLocation loc) {
  auto spellingLoc = SM.getSpellingLoc(loc);
  auto lineNumber = SM.getSpellingLineNumber(spellingLoc);
  auto colNumber = SM.getSpellingColumnNumber(spellingLoc);
  auto fileId = SM.getFilename(spellingLoc);

  auto ctx = module.getContext();
  auto mlirIdentifier = Identifier::get(fileId, ctx);
  mlir::OpBuilder builder(ctx);
  return builder.getUnknownLoc();
  // return builder.getFileLineColLoc(mlirIdentifier, lineNumber, colNumber);
}

mlir::Type MLIRASTConsumer::getMLIRType(clang::QualType qt, bool *implicitRef,
                                        bool allowMerge) {
  if (auto ET = dyn_cast<clang::ElaboratedType>(qt)) {
    return getMLIRType(ET->getNamedType(), implicitRef, allowMerge);
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
      auto mt = mlirty.dyn_cast<MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2[0] = -1;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   mt.getAffineMaps(), mt.getMemorySpace());
    } else {
      return getMLIRType(DT->getAdjustedType(), implicitRef, allowMerge);
    }
    return mlirty;
  }
  if (auto RT = dyn_cast<clang::RecordType>(qt)) {
    llvm::StructType *ST =
        cast<llvm::StructType>(CGM.getTypes().ConvertType(qt));

    if (RT->getDecl()->field_empty())
      if (ST->getNumElements() == 1 && ST->getElementType(0U)->isIntegerTy(8))
        return typeTranslator.translateType(
            llvm::StructType::get(ST->getContext()));

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
        ST->getNumElements() == 0 || recursive ||
        (!ST->isLiteral() && (ST->getName().contains("SmallVector") ||
                              ST->getName() == "struct._IO_FILE" ||
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

    if (notAllSame || !allowMerge || innerLLVM) {
      return mlir::LLVM::LLVMStructType::getLiteral(module.getContext(), types);
    }

    if (!types.size()) {
      RT->dump();
      llvm::errs() << "ST: " << *ST << "\n";
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
    mlir::OpBuilder builder(module.getContext());
    return builder.getNoneType();
  }

  // if (auto AT = dyn_cast<clang::VariableArrayType>(t)) {
  //   return getMLIRType(AT->getElementType(), implicitRef, allowMerge);
  // }

  if (auto AT = dyn_cast<clang::ArrayType>(t)) {
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
                                   mt.getAffineMaps(), mt.getMemorySpace());
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
                                   mt.getAffineMaps(), mt.getMemorySpace());
    }
    if (!allowMerge || ET.isa<LLVM::LLVMPointerType, LLVM::LLVMArrayType,
                              LLVM::LLVMFunctionType, LLVM::LLVMStructType>())
      return LLVM::LLVMFixedVectorType::get(ET, size);
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get({size}, ET);
  }

  if (isa<clang::PointerType, clang::ReferenceType>(t)) {
    int64_t outer = (isa<clang::PointerType>(t)) ? -1 : -1;
    auto PTT = isa<clang::PointerType>(t) ? cast<clang::PointerType>(t)
                                                ->getPointeeType()
                                                ->getUnqualifiedDesugaredType()
                                          : cast<clang::ReferenceType>(t)
                                                ->getPointeeType()
                                                ->getUnqualifiedDesugaredType();

    if (PTT->isCharType() || PTT->isVoidType() || PTT->isFunctionType()) {
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
                    LLVM::LLVMPointerType>())
      return LLVM::LLVMPointerType::get(subType);

    if (PTT->isBooleanType()) {
      OpBuilder builder(module);
      return MemRefType::get(outer, builder.getIntegerType(1), {});
    }

    if (isa<clang::ArrayType>(PTT)) {
      if (subType.isa<MemRefType>()) {
        assert(subRef);
        return subType;
      } else
        return LLVM::LLVMPointerType::get(subType);
    }

    if (isa<clang::VectorType>(PTT)) {
      if (subType.isa<MemRefType>()) {
        assert(subRef);
        auto mt = subType.cast<MemRefType>();
        auto shape2 = std::vector<int64_t>(mt.getShape());
        shape2.insert(shape2.begin(), outer);
        return mlir::MemRefType::get(shape2, mt.getElementType(),
                                     mt.getAffineMaps(), mt.getMemorySpace());
      } else
        return LLVM::LLVMPointerType::get(subType);
    }

    if (isa<clang::RecordType>(PTT))
      if (subRef) {
        auto mt = subType.cast<MemRefType>();
        auto shape2 = std::vector<int64_t>(mt.getShape());
        shape2.insert(shape2.begin(), outer);
        return mlir::MemRefType::get(shape2, mt.getElementType(),
                                     mt.getAffineMaps(), mt.getMemorySpace());
      }

    assert(!subRef);
    return mlir::MemRefType::get({outer}, subType);
  }

  if (t->isBuiltinType() || isa<clang::EnumType>(t)) {
    llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
    return getMLIRType(T);
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

mlir::Type MLIRASTConsumer::getMLIRType(llvm::Type *t) {
  mlir::OpBuilder builder(module.getContext());
  if (t->isVoidTy()) {
    return builder.getNoneType();
  }
  if (t->isFloatTy()) {
    return builder.getF32Type();
  }
  if (t->isDoubleTy()) {
    return builder.getF64Type();
  }
  if (t->isX86_FP80Ty())
    return builder.getF80Type();
  if (t->isFP128Ty())
    return builder.getF128Type();

  if (auto IT = dyn_cast<llvm::IntegerType>(t)) {
    return builder.getIntegerType(IT->getBitWidth());
  }
  if (auto pt = dyn_cast<llvm::PointerType>(t)) {
    if (isa<llvm::FunctionType>(pt->getElementType())) {
      return typeTranslator.translateType(t);
    }
    if (auto ST = dyn_cast<llvm::StructType>(pt->getElementType())) {
      if (ST->getNumElements() == 0 ||
          (!ST->isLiteral() &&
           (ST->getName().contains("SmallVector") ||
            ST->getName() == "struct._IO_FILE" ||
            ST->getName() == "class.std::basic_ifstream" ||
            ST->getName() == "class.std::basic_istream" ||
            ST->getName() == "class.std::basic_ostream" ||
            ST->getName() == "class.std::basic_ofstream"))) {
        return typeTranslator.translateType(t);
      }
      bool notAllSame = false;
      SmallPtrSet<llvm::Type *, 4> Seen;
      bool recursive = false;
      for (size_t i = 0; i < ST->getNumElements(); i++) {
        if (isRecursiveStruct(ST->getTypeAtIndex(i), ST, Seen)) {
          recursive = true;
        }
        if (ST->getTypeAtIndex(i) != ST->getTypeAtIndex(0U)) {
          notAllSame = true;
        }
      }
      if (!recursive && ST->getNumElements() == 1)
        return getMLIRType(ST->getTypeAtIndex(0U));

      if (recursive) {
        return typeTranslator.translateType(t);
      }
      if (!notAllSame) {
        auto subT = getMLIRType(ST->getTypeAtIndex(0U));
        if (subT.isa<LLVM::LLVMArrayType, LLVM::LLVMStructType,
                     LLVM::LLVMPointerType>())
          return LLVM::LLVMPointerType::get(subT);
        return mlir::MemRefType::get({-1, ST->getNumElements()}, subT, {},
                                     pt->getAddressSpace());
      }
    }
    if (auto AT = dyn_cast<llvm::ArrayType>(pt->getElementType())) {
      auto under = getMLIRType(AT);
      auto mt = under.cast<mlir::MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      // shape2[0] = -1;//
      // shape2.insert(shape2.begin(), -1);
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   mt.getAffineMaps(), mt.getMemorySpace());
    }
    return mlir::MemRefType::get(-1, getMLIRType(pt->getElementType()), {},
                                 pt->getAddressSpace());
  }
  if (auto pt = dyn_cast<llvm::ArrayType>(t)) {
    auto under = getMLIRType(pt->getElementType());
    if (under.isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                  LLVM::LLVMArrayType>())
      return LLVM::LLVMArrayType::get(under, pt->getNumElements());
    if (auto mt = under.dyn_cast<mlir::MemRefType>()) {
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2.insert(shape2.begin(), (int64_t)pt->getNumElements());
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   mt.getAffineMaps(), mt.getMemorySpace());
    }
    return mlir::MemRefType::get({(int64_t)pt->getNumElements()}, under);
  }
  if (auto ST = dyn_cast<llvm::StructType>(t)) {
    bool notAllSame = false;
    SmallPtrSet<llvm::Type *, 4> Seen;
    bool recursive = false;
    for (size_t i = 0; i < ST->getNumElements(); i++) {
      if (isRecursiveStruct(ST->getTypeAtIndex(i), ST, Seen)) {
        recursive = true;
      }
      if (ST->getTypeAtIndex(i) != ST->getTypeAtIndex(0U)) {
        notAllSame = true;
      }
    }
    if (!recursive && ST->getNumElements() == 1)
      return getMLIRType(ST->getTypeAtIndex(0U));
    if (ST->getNumElements() == 0 || recursive ||
        (!ST->isLiteral() && (ST->getName().contains("SmallVector") ||
                              ST->getName() == "struct._IO_FILE" ||
                              ST->getName() == "class.std::basic_ifstream" ||
                              ST->getName() == "class.std::basic_istream" ||
                              ST->getName() == "class.std::basic_ostream" ||
                              ST->getName() == "class.std::basic_ofstream"))) {
      return typeTranslator.translateType(t);
    }

    if (notAllSame) {
      SmallVector<mlir::Type> types;
      for (size_t i = 0; i < ST->getNumElements(); i++) {
        auto st = getMLIRType(ST->getTypeAtIndex(i));
        if (st.isa<MemRefType>()) {
          types.push_back(typeTranslator.translateType(ST->getTypeAtIndex(i)));
        } else
          types.push_back(st);
      }
      return mlir::LLVM::LLVMStructType::getLiteral(module.getContext(), types);
    }

    return mlir::MemRefType::get(ST->getNumElements(),
                                 getMLIRType(ST->getTypeAtIndex(0U)));
  }
  llvm::errs() << *t << "\n";
  assert(0 && "unknown type to convert");
  return nullptr;
}

void MLIRScanner::pushLoopIf() {
  if (loops.size() && loops.back().keepRunning) {
    auto lop = builder.create<memref::LoadOp>(loc, loops.back().keepRunning);
    auto ifOp = builder.create<scf::IfOp>(loc, lop,
                                          /*hasElse*/ false);
    prevBlock.push_back(builder.getInsertionBlock());
    prevIterator.push_back(builder.getInsertionPoint());
    ifOp.thenRegion().back().clear();
    builder.setInsertionPointToStart(&ifOp.thenRegion().back());
    auto er = builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
    builder.create<scf::YieldOp>(loc);
    er.region().push_back(new Block());
    builder.setInsertionPointToStart(&er.region().back());
  }
}

void MLIRScanner::popLoopIf() {
  if (loops.size() && loops.back().keepRunning) {
    builder.create<scf::YieldOp>(loc);
    builder.setInsertionPoint(prevBlock.back(), prevIterator.back());
    prevBlock.pop_back();
    prevIterator.pop_back();
  }
}

#include "llvm/Support/Host.h"

#include "clang/Frontend/FrontendAction.h"
class MLIRAction : public clang::ASTFrontendAction {
public:
  std::set<std::string> emitIfFound;
  std::set<std::string> done;
  mlir::ModuleOp &module;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> globals;
  std::map<std::string, mlir::FuncOp> functions;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> llvmFunctions;
  MLIRAction(std::string fn, mlir::ModuleOp &module) : module(module) {
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
                      std::vector<std::string> defines, mlir::ModuleOp &module,
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

    module->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                    StringAttr::get(module.getContext(),
                                    Clang->getTarget().getDataLayoutString()));
    module->setAttr(
        LLVM::LLVMDialect::getTargetTripleAttrName(),
        StringAttr::get(module.getContext(),
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

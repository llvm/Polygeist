#include "clang-mlir.h"

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
#include <clang/Lex/Pragma.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Parse/Parser.h>
#include <clang/Sema/Sema.h>
#include <clang/Sema/SemaDiagnostic.h>

using namespace std;
using namespace clang;
using namespace clang::driver;
using namespace llvm::opt;
using namespace mlir;

// from clang codegenmodule
static std::string getMangledNameImpl(MangleContext &MC, GlobalDecl GD,
                                      const NamedDecl *ND,
                                      bool OmitMultiVersionMangling = false) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  // MangleContext &MC = CGM.getCXXABI().getMangleContext();
  if (MC.shouldMangleDeclName(ND))
    MC.mangleName(GD.getWithDecl(ND), Out);
  else {
    IdentifierInfo *II = ND->getIdentifier();
    assert(II && "Attempt to mangle unnamed decl.");
    const auto *FD = dyn_cast<FunctionDecl>(ND);

    if (FD && FD->getType()->castAs<clang::FunctionType>()->getCallConv() ==
                  CC_X86RegCall) {
      Out << "__regcall3__" << II->getName();
    } else if (FD && FD->hasAttr<CUDAGlobalAttr>() &&
               GD.getKernelReferenceKind() == KernelReferenceKind::Stub) {
      Out << "__device_stub__" << II->getName();
    } else {
      Out << II->getName();
    }
  }

  return std::string(Out.str());
}

void MLIRScanner::setValue(std::string name, ValueWithOffsets &&val) {
  auto z = scopes.back().emplace(name, val);
  assert(z.second);
  assert(val.offsets.size() == z.first->second.offsets.size());
}

ValueWithOffsets MLIRScanner::getValue(std::string name) {
  for (int i = scopes.size() - 1; i >= 0; i--) {
    auto found = scopes[i].find(name);
    if (found != scopes[i].end()) {
      return found->second;
    }
  }
  llvm::errs() << "couldn't find " << name << "\n";
  assert(0 && "couldnt find value");
}

mlir::Type MLIRScanner::getLLVMTypeFromMLIRType(mlir::Type t) {
  if (auto it = t.dyn_cast<mlir::IntegerType>()) {
    return mlir::LLVM::LLVMIntegerType::get(t.getContext(), it.getWidth());
  }
  assert(0 && "unhandled mlir=>llvm type");
}

mlir::Value MLIRScanner::createAllocOp(mlir::Type t, std::string name,
                                       uint64_t memspace,
                                       bool isArray = false) {
  mlir::MemRefType mr;
  if (!isArray) {
    mr = mlir::MemRefType::get(1, t, {}, memspace);
  } else {
    auto mt = t.cast<mlir::MemRefType>();
    mr = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                               mt.getAffineMaps(), memspace);
  }
  // NamedAttribute attrs[] = {NamedAttribute("name", name)};
  auto alloc = builder.create<mlir::AllocaOp>(loc, mr);
  if (isArray)
    setValue(name, ValueWithOffsets(alloc, {}));
  else
    setValue(name, ValueWithOffsets(alloc, {getConstantIndex(0)}));
  if (!isArray)
    assert(getValue(name).offsets.size() == 1);
  return alloc;
}

mlir::Value MLIRScanner::createAndSetAllocOp(std::string name, mlir::Value v,
                                             uint64_t memspace) {
  auto op = createAllocOp(v.getType(), name, memspace);
  mlir::Value zeroIndex = getConstantIndex(0);
  builder.create<mlir::StoreOp>(loc, v, op, zeroIndex);
  return op;
}

ValueWithOffsets MLIRScanner::VisitDeclStmt(clang::DeclStmt *decl) {
  for (auto sub : decl->decls()) {
    if (auto vd = dyn_cast<VarDecl>(sub)) {
      VisitVarDecl(vd);
    } else {
      llvm::errs() << " + visiting unknonwn sub decl stmt\n";
      sub->dump();
      assert(0 && "unknown sub decl");
    }
  }
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitIntegerLiteral(clang::IntegerLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return (mlir::Value)builder.create<mlir::ConstantOp>(
      loc, ty, builder.getIntegerAttr(ty, expr->getValue()));
}

ValueWithOffsets
MLIRScanner::VisitFloatingLiteral(clang::FloatingLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::FloatType>();
  return (mlir::Value)builder.create<mlir::ConstantOp>(
      loc, ty, builder.getFloatAttr(ty, expr->getValue()));
}

ValueWithOffsets
MLIRScanner::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return (mlir::Value)builder.create<mlir::ConstantOp>(
      loc, ty, builder.getIntegerAttr(ty, expr->getValue()));
}

ValueWithOffsets MLIRScanner::VisitStringLiteral(clang::StringLiteral *expr) {
  auto ty = getMLIRType(expr->getType());
  return (mlir::Value)builder.create<mlir::ConstantOp>(
      loc, ty, builder.getStringAttr(expr->getBytes()));
}

ValueWithOffsets MLIRScanner::VisitParenExpr(clang::ParenExpr *expr) {
  return Visit(expr->getSubExpr());
}

ValueWithOffsets MLIRScanner::VisitVarDecl(clang::VarDecl *decl) {
  unsigned memtype = 0;

  // if (decl->hasAttr<CUDADeviceAttr>() || decl->hasAttr<CUDAConstantAttr>()
  // ||
  if (decl->hasAttr<CUDASharedAttr>()) {
    memtype = 5;
  }
  mlir::Type subType = getMLIRType(decl->getType());
  mlir::Value inite = nullptr;
  if (auto init = decl->getInit()) {
    auto visit = Visit(init);
    inite = (mlir::Value)visit;
    if (!inite) {
      init->dump();
    }
    subType = inite.getType();
  }
  auto op = createAllocOp(subType, decl->getName().str(), memtype,
                          /*isArray*/ isa<clang::ArrayType>(decl->getType()));
  mlir::Value zeroIndex = getConstantIndex(0);
  if (inite) {
    builder.create<mlir::StoreOp>(loc, inite, op, zeroIndex);
  }
  return ValueWithOffsets(op, {zeroIndex});
}

ValueWithOffsets MLIRScanner::VisitForStmt(clang::ForStmt *fors) {
  scopes.emplace_back();

  auto loc = getMLIRLocation(fors->getForLoc());

  if (auto s = fors->getInit()) {
    Visit(s);
  }

  auto &condB = *function.addBlock();
  auto &bodyB = *function.addBlock();

  auto &exitB = *function.addBlock();

  builder.create<mlir::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&condB);

  if (auto s = fors->getCond()) {
    auto condRes = Visit(s);
    builder.create<mlir::CondBranchOp>(loc, (mlir::Value)condRes, &bodyB,
                                       &exitB);
  }

  loops.push_back((LoopContext){&condB, &exitB});
  builder.setInsertionPointToStart(&bodyB);
  Visit(fors->getBody());
  if (auto s = fors->getInc()) {
    Visit(s);
  }
  loops.pop_back();
  if (builder.getInsertionBlock()->empty() ||
      builder.getInsertionBlock()->back().isKnownNonTerminator()) {
    builder.create<mlir::BranchOp>(loc, &condB);
  }

  builder.setInsertionPointToStart(&exitB);
  scopes.pop_back();
  return nullptr;
}

ValueWithOffsets
MLIRScanner::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr) {
  auto lhs = Visit(expr->getLHS());
  auto rhs = (mlir::Value)Visit(expr->getRHS());
  auto offsets = lhs.offsets;
  auto idx = builder.create<mlir::IndexCastOp>(
      loc, rhs, mlir::IndexType::get(rhs.getContext()));
  assert(offsets.size() > 0);
  offsets[offsets.size() - 1] =
      builder.create<mlir::AddIOp>(loc, (mlir::Value)lhs.offsets.back(), idx);
  return ValueWithOffsets(lhs.val, offsets);
}

mlir::FuncOp MLIRScanner::EmitCallee(const Expr *E) {
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
      return EmitDirectCallee(FD);
    }
  } else if (auto ME = dyn_cast<MemberExpr>(E)) {
    if (auto FD = dyn_cast<FunctionDecl>(ME->getMemberDecl())) {
      // TODO EmitIgnoredExpr(ME->getBase());
      return EmitDirectCallee(FD);
    }

    // Look through template substitutions.
  } else if (auto NTTP = dyn_cast<SubstNonTypeTemplateParmExpr>(E)) {
    return EmitCallee(NTTP->getReplacement());

    // Treat pseudo-destructor calls differently.
    //} else if (auto PDE = dyn_cast<CXXPseudoDestructorExpr>(E)) {
    //    return CGCallee::forPseudoDestructor(PDE);
  }

  assert(0 && "indirect references not handled");
}

ValueWithOffsets MLIRScanner::VisitCallExpr(clang::CallExpr *expr) {
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto ME = dyn_cast<MemberExpr>(ic->getSubExpr())) {
      auto memberName = ME->getMemberDecl()->getName();

      if (auto sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
        if (auto sr = dyn_cast<DeclRefExpr>(sr2->getSourceExpr())) {
          if (sr->getDecl()->getName() == "blockIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "x"),
                  mlirType);
            }
            if (memberName == "__fetch_builtin_y") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "y"),
                  mlirType);
            }
            if (memberName == "__fetch_builtin_z") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "z"),
                  mlirType);
            }
          }
          if (sr->getDecl()->getName() == "blockDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "x"),
                  mlirType);
            }
            if (memberName == "__fetch_builtin_y") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "y"),
                  mlirType);
            }
            if (memberName == "__fetch_builtin_z") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "z"),
                  mlirType);
            }
          }
          if (sr->getDecl()->getName() == "threadIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::ThreadIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "x"),
                  mlirType);
            }
            if (memberName == "__fetch_builtin_y") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::ThreadIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "y"),
                  mlirType);
            }
            if (memberName == "__fetch_builtin_z") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::ThreadIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "z"),
                  mlirType);
            }
          }
          if (sr->getDecl()->getName() == "gridDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::GridDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "x"),
                  mlirType);
            }
            if (memberName == "__fetch_builtin_y") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::GridDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "y"),
                  mlirType);
            }
            if (memberName == "__fetch_builtin_z") {
              return (mlir::Value)builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::GridDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "z"),
                  mlirType);
            }
          }
        }
      }
    }

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "__syncthreads") {
        builder.create<mlir::NVVM::Barrier0Op>(loc);
        return nullptr;
      }
    }

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "__shfl_up_sync") {
        assert(0 && "__shfl_up_sync unhandled");
        // builder.create<mlir::NVVM::ShflBflyOp>(loc);
        return nullptr;
      }
    }

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "__log2f") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back((mlir::Value)Visit(a));
        }
        return (mlir::Value)builder.create<mlir::Log2Op>(loc, args[0]);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "sqrtf" ||
          sr->getDecl()->getName() == "sqrt") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back((mlir::Value)Visit(a));
        }
        return (mlir::Value)builder.create<mlir::SqrtOp>(loc, args[0]);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "atomicAdd") {
        std::vector<ValueWithOffsets> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a));
        }
        if (args[1].val.getType().isa<mlir::IntegerType>())
          return (mlir::Value)builder.create<mlir::AtomicRMWOp>(
              loc, args[1].val.getType(), AtomicRMWKind::addi,
              (mlir::Value)args[1], args[0].val, args[0].offsets);
        else
          return (mlir::Value)builder.create<mlir::AtomicRMWOp>(
              loc, args[1].val.getType(), AtomicRMWKind::addf,
              (mlir::Value)args[1], args[0].val, args[0].offsets);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      // TODO add pow to standard dialect
      if (sr->getDecl()->getName() == "__powf") {
        auto mlirType = getMLIRType(expr->getType());
        mlir::Type llvmType =
            mlir::LLVM::TypeFromLLVMIRTranslator(*mlirType.getContext())
                .translateType(getLLVMType(expr->getType()));
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back((mlir::Value)Visit(a));
        }
        auto arg0 =
            builder.create<mlir::LLVM::DialectCastOp>(loc, llvmType, args[0]);
        auto arg1 =
            builder.create<mlir::LLVM::DialectCastOp>(loc, llvmType, args[1]);
        return (mlir::Value)builder.create<mlir::LLVM::DialectCastOp>(
            loc, mlirType,
            builder.create<mlir::LLVM::PowOp>(loc, llvmType, arg0, arg1));
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      // TODO add pow to standard dialect
      if (sr->getDecl()->getName() == "fprintf") {
        llvm::errs() << "warning skipping fprintf\n";
        // std::vector<mlir::Value> args;
        // for(auto a : expr->arguments()) {
        //    args.push_back((mlir::Value)Visit(a));
        //}
        return nullptr;
      }
    }

  auto tocall = EmitCallee(expr->getCallee());
  std::vector<mlir::Value> args;
  auto fnType = tocall.getType();
  size_t i = 0;
  for (auto a : expr->arguments()) {
    mlir::Value val = (mlir::Value)Visit(a);
    if (val.getType() != fnType.getInput(i)) {
      if (auto MR1 = val.getType().dyn_cast<MemRefType>()) {
        if (auto MR2 = fnType.getInput(i).dyn_cast<MemRefType>()) {
          val = builder.create<mlir::MemRefCastOp>(loc, val, MR2);
        }
      }
    }
    args.push_back(val);
    i++;
  }
  auto op = builder.create<mlir::CallOp>(loc, tocall, args);
  if (op.getNumResults())
    return op.getResult(0);
  else
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
  return getConstantIndex(0);
}

ValueWithOffsets
MLIRScanner::VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr) {
  // if (auto syn = dyn_cast<MSPropertyRefExpr>(expr->getSyntacticForm ())) {
  //    return Visit(syn);
  //}
  return Visit(expr->getResultExpr());
}

ValueWithOffsets MLIRScanner::VisitUnaryOperator(clang::UnaryOperator *U) {
  auto sub = Visit(U->getSubExpr());
  // TODO note assumptions made here about unsigned / unordered
  bool signedType = true;
  if (auto bit = dyn_cast<clang::BuiltinType>(&*U->getType())) {
    if (bit->isUnsignedInteger())
      signedType = false;
    if (bit->isSignedInteger())
      signedType = true;
  }
  auto ty = getMLIRType(U->getType());

  switch (U->getOpcode()) {
  case clang::UnaryOperator::Opcode::UO_LNot: {
    mlir::Value val = (mlir::Value)sub;
    auto ty = val.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      ty = builder.getIntegerType(1);
      val = (mlir::Value)builder.create<mlir::TruncateIOp>(loc, val, ty);
    }
    auto c1 = (mlir::Value)builder.create<mlir::ConstantOp>(
        loc, ty, builder.getIntegerAttr(ty, 1));
    return (mlir::Value)builder.create<mlir::XOrOp>(loc, val, c1);
  }
  case clang::UnaryOperator::Opcode::UO_Deref: {
    auto off = sub.offsets;
    /*
    U->getType().desugar()->dump();
    if (isa<clang::ArrayType>(U->getType().desugar())) {
        assert(off.size() != 0);
        assert(sub.val.getType().cast<MemRefType>().getShape().size() ==
    off.size());
        auto val = builder.create<mlir::LoadOp>(loc, sub.val, off);
        if (U->getType()->isPointerType())
            return ValueWithOffsets(val, {getConstantIndex(0)});
        else
            return ValueWithOffsets(val, {});
    }
    */

    return ValueWithOffsets(sub.val, off);
  }
  case clang::UnaryOperator::Opcode::UO_AddrOf: {
    auto v = sub.offsets;
    return ValueWithOffsets(sub.val, v);
  }
  case clang::UnaryOperator::Opcode::UO_Minus: {
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::NegFOp>(loc, (mlir::Value)sub);
    } else {
      return (mlir::Value)builder.create<mlir::SubIOp>(
          loc,
          builder.create<mlir::ConstantIntOp>(loc, 0,
                                              ty.cast<mlir::IntegerType>()),
          (mlir::Value)sub);
    }
  }
  case clang::UnaryOperator::Opcode::UO_PreInc:
  case clang::UnaryOperator::Opcode::UO_PostInc: {
    auto off = sub.offsets;
    assert(off.size() != 0);
    auto prev = builder.create<mlir::LoadOp>(loc, sub.val, off);

    mlir::Value next;
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      next = builder.create<mlir::AddFOp>(
          loc, prev,
          builder.create<mlir::ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else {
      next = builder.create<mlir::AddIOp>(
          loc, prev,
          builder.create<mlir::ConstantIntOp>(loc, 1,
                                              ty.cast<mlir::IntegerType>()));
    }
    builder.create<mlir::StoreOp>(loc, next, sub.val, off);
    return ValueWithOffsets(
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? prev
                                                                     : next,
        {});
  }
  case clang::UnaryOperator::Opcode::UO_PreDec:
  case clang::UnaryOperator::Opcode::UO_PostDec: {
    auto off = sub.offsets;
    assert(off.size() != 0);
    auto prev = builder.create<mlir::LoadOp>(loc, sub.val, off);

    mlir::Value next;
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      next = builder.create<mlir::SubFOp>(
          loc, prev,
          builder.create<mlir::ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else {
      next = builder.create<mlir::SubIOp>(
          loc, prev,
          builder.create<mlir::ConstantIntOp>(loc, 1,
                                              ty.cast<mlir::IntegerType>()));
    }
    builder.create<mlir::StoreOp>(loc, next, sub.val, off);
    return ValueWithOffsets(
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? prev
                                                                     : next,
        {});
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
    return (mlir::Value)builder.create<mlir::ConstantOp>(
        loc, ty, builder.getIntegerAttr(ty, value));
  }
  default:
    Uop->dump();
    assert(0 && "unhandled VisitUnaryExprOrTypeTraitExpr");
  }
}

ValueWithOffsets MLIRScanner::VisitBinaryOperator(clang::BinaryOperator *BO) {
  auto lhs = Visit(BO->getLHS());
  if (!lhs.val) {
    BO->getLHS()->dump();
  }
  assert(lhs.val);
  auto rhs = Visit(BO->getRHS());
  if (!rhs.val) {
    BO->getRHS()->dump();
  }
  assert(rhs.val);
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
      return (mlir::Value)builder.create<mlir::SignedShiftRightOp>(
          loc, (mlir::Value)lhs, (mlir::Value)rhs);
    else
      return (mlir::Value)builder.create<mlir::UnsignedShiftRightOp>(
          loc, (mlir::Value)lhs, (mlir::Value)rhs);
  }
  case clang::BinaryOperator::Opcode::BO_Shl: {
    return (mlir::Value)builder.create<mlir::ShiftLeftOp>(loc, (mlir::Value)lhs,
                                                          (mlir::Value)rhs);
  }
  case clang::BinaryOperator::Opcode::BO_LAnd: {
    return (mlir::Value)builder.create<mlir::AndOp>(loc, (mlir::Value)lhs,
                                                    (mlir::Value)rhs);
  }
  case clang::BinaryOperator::Opcode::BO_LOr: {
    return (mlir::Value)builder.create<mlir::OrOp>(loc, (mlir::Value)lhs,
                                                   (mlir::Value)rhs);
  }
  case clang::BinaryOperator::Opcode::BO_GT: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::UGT, (mlir::Value)lhs, (mlir::Value)rhs);
    } else {
      return (mlir::Value)builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::sgt : mlir::CmpIPredicate::ugt,
          (mlir::Value)lhs, (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_GE: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::UGE, (mlir::Value)lhs, (mlir::Value)rhs);
    } else {
      return (mlir::Value)builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::sge : mlir::CmpIPredicate::uge,
          (mlir::Value)lhs, (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_LT: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::ULT, (mlir::Value)lhs, (mlir::Value)rhs);
    } else {
      return (mlir::Value)builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::slt : mlir::CmpIPredicate::ult,
          (mlir::Value)lhs, (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_LE: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::ULE, (mlir::Value)lhs, (mlir::Value)rhs);
    } else {
      return (mlir::Value)builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::sle : mlir::CmpIPredicate::ule,
          (mlir::Value)lhs, (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_EQ: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::UEQ, (mlir::Value)lhs, (mlir::Value)rhs);
    } else {
      return (mlir::Value)builder.create<mlir::CmpIOp>(
          loc, mlir::CmpIPredicate::eq, (mlir::Value)lhs, (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_NE: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::UNE, (mlir::Value)lhs, (mlir::Value)rhs);
    } else {
      return (mlir::Value)(mlir::Value)builder.create<mlir::CmpIOp>(
          loc, mlir::CmpIPredicate::ne, (mlir::Value)lhs, (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Mul: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::MulFOp>(loc, (mlir::Value)lhs,
                                                       (mlir::Value)rhs);
    } else {
      return (mlir::Value)builder.create<mlir::MulIOp>(loc, (mlir::Value)lhs,
                                                       (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Div: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::DivFOp>(loc, (mlir::Value)lhs,
                                                       (mlir::Value)rhs);
    } else {
      if (signedType)
        return (mlir::Value)builder.create<mlir::SignedDivIOp>(
            loc, (mlir::Value)lhs, (mlir::Value)rhs);
      else
        return (mlir::Value)builder.create<mlir::UnsignedDivIOp>(
            loc, (mlir::Value)lhs, (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Rem: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::RemFOp>(loc, (mlir::Value)lhs,
                                                       (mlir::Value)rhs);
    } else {
      if (signedType)
        return (mlir::Value)builder.create<mlir::SignedRemIOp>(
            loc, (mlir::Value)lhs, (mlir::Value)rhs);
      else
        return (mlir::Value)builder.create<mlir::UnsignedRemIOp>(
            loc, (mlir::Value)lhs, (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Add: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::AddFOp>(loc, (mlir::Value)lhs,
                                                       (mlir::Value)rhs);
    } else if (lhs.getType().isa<mlir::MemRefType>()) {
      auto offsets = lhs.offsets;
      auto idx = builder.create<mlir::IndexCastOp>(
          loc, (mlir::Value)rhs, mlir::IndexType::get(rhs.val.getContext()));
      assert(offsets.size() > 0);
      offsets[offsets.size() - 1] = builder.create<mlir::AddIOp>(
          loc, (mlir::Value)lhs.offsets.back(), idx);
      return ValueWithOffsets(lhs.val, offsets);
    } else {
      return (mlir::Value)builder.create<mlir::AddIOp>(loc, (mlir::Value)lhs,
                                                       (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Sub: {
    if (lhs.getType().isa<mlir::FloatType>()) {
      return (mlir::Value)builder.create<mlir::SubFOp>(loc, (mlir::Value)lhs,
                                                       (mlir::Value)rhs);
    } else {
      return (mlir::Value)builder.create<mlir::SubIOp>(loc, (mlir::Value)lhs,
                                                       (mlir::Value)rhs);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Assign: {
    auto off = lhs.offsets;
    if (off.size() == 0) {
      BO->dump();
    }
    assert(off.size() != 0);

    if (lhs.val.getType().cast<MemRefType>().getShape().size() != off.size()) {
      BO->dump();
      llvm::errs() << "{\n";
      for (auto a : off)
        a.dump();
      llvm::errs() << "}\n";
    }
    assert(lhs.val.getType().cast<MemRefType>().getShape().size() ==
           off.size());
    builder.create<mlir::StoreOp>(loc, (mlir::Value)rhs, lhs.val, off);
    return lhs;
  }

  case clang::BinaryOperator::Opcode::BO_Comma: {
    return rhs;
  }

  case clang::BinaryOperator::Opcode::BO_AddAssign: {
    auto off = lhs.offsets;
    assert(off.size() != 0);
    auto prev = builder.create<mlir::LoadOp>(loc, lhs.val, off);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<mlir::AddFOp>(loc, (mlir::Value)prev,
                                            (mlir::Value)rhs);
    } else {
      result = builder.create<mlir::AddIOp>(loc, (mlir::Value)prev,
                                            (mlir::Value)rhs);
    }
    builder.create<mlir::StoreOp>(loc, result, lhs.val, off);
    return ValueWithOffsets(prev, {});
  }
  case clang::BinaryOperator::Opcode::BO_SubAssign: {
    auto off = lhs.offsets;
    assert(off.size() != 0);
    auto prev = builder.create<mlir::LoadOp>(loc, lhs.val, off);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<mlir::SubFOp>(loc, (mlir::Value)prev,
                                            (mlir::Value)rhs);
    } else {
      result = builder.create<mlir::SubIOp>(loc, (mlir::Value)prev,
                                            (mlir::Value)rhs);
    }
    builder.create<mlir::StoreOp>(loc, result, lhs.val, off);
    return ValueWithOffsets(prev, {});
  }
  case clang::BinaryOperator::Opcode::BO_MulAssign: {
    auto off = lhs.offsets;
    assert(off.size() != 0);
    auto prev = builder.create<mlir::LoadOp>(loc, lhs.val, off);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<mlir::MulFOp>(loc, (mlir::Value)prev,
                                            (mlir::Value)rhs);
    } else {
      result = builder.create<mlir::MulIOp>(loc, (mlir::Value)prev,
                                            (mlir::Value)rhs);
    }
    builder.create<mlir::StoreOp>(loc, result, lhs.val, off);
    return ValueWithOffsets(prev, {});
  }
  case clang::BinaryOperator::Opcode::BO_DivAssign: {
    auto off = lhs.offsets;
    assert(off.size() != 0);
    auto prev = builder.create<mlir::LoadOp>(loc, lhs.val, off);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<mlir::DivFOp>(loc, (mlir::Value)prev,
                                            (mlir::Value)rhs);
    } else {
      if (signedType)
        return (mlir::Value)builder.create<mlir::SignedDivIOp>(
            loc, (mlir::Value)prev, (mlir::Value)rhs);
      else
        return (mlir::Value)builder.create<mlir::UnsignedDivIOp>(
            loc, (mlir::Value)prev, (mlir::Value)rhs);
    }
    builder.create<mlir::StoreOp>(loc, result, lhs.val, off);
    return ValueWithOffsets(prev, {});
  }
  case clang::BinaryOperator::Opcode::BO_ShrAssign: {
    auto off = lhs.offsets;
    assert(off.size() != 0);
    auto prev = builder.create<mlir::LoadOp>(loc, lhs.val, off);

    mlir::Value result;

    if (signedType)
      return (mlir::Value)builder.create<mlir::SignedShiftRightOp>(
          loc, (mlir::Value)prev, (mlir::Value)rhs);
    else
      return (mlir::Value)builder.create<mlir::UnsignedShiftRightOp>(
          loc, (mlir::Value)prev, (mlir::Value)rhs);
    builder.create<mlir::StoreOp>(loc, result, lhs.val, off);
    return ValueWithOffsets(prev, {});
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
    assert(0 && "cleanup not handled");
  }
  return ret;
}

ValueWithOffsets MLIRScanner::VisitDeclRefExpr(DeclRefExpr *E) {
  if (auto FD = dyn_cast<FunctionDecl>(E->getDecl())) {
  }
  return getValue(E->getDecl()->getName().str());
}

ValueWithOffsets MLIRScanner::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  assert(E->getSourceExpr());
  for (auto c : E->children()) {
    c->dump();
  }
  auto res = Visit(E->getSourceExpr());
  if (!res.val) {
    E->dump();
    E->getSourceExpr()->dump();
  }
  assert(res.val);
  return res;
}

ValueWithOffsets MLIRScanner::VisitMemberExpr(MemberExpr *ME) {
  auto memberName = ME->getMemberDecl()->getName();
  llvm::errs() << "md name: " << memberName << "\n";
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
  assert(0 && "memberexpr unhandled");
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitCastExpr(CastExpr *E) {
  switch (E->getCastKind()) {
  case clang::CastKind::CK_BitCast: {
    auto scalar = Visit(E->getSubExpr());
    auto ut = scalar.getType().cast<mlir::MemRefType>();
    auto mt = getMLIRType(E->getType()).cast<mlir::MemRefType>();

    auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                    ut.getAffineMaps(), ut.getMemorySpace());
    auto offs = scalar.offsets;
    for (auto &off : offs) {
      if (auto op = off.getDefiningOp<ConstantIndexOp>()) {
        assert(op.getValue() == 0);
      } else {
        assert(0 && "cast of nonconstant op is not handled");
      }
    }
    return ValueWithOffsets(
        builder.create<mlir::MemRefCastOp>(loc, scalar.val, ty), offs);
  }
  case clang::CastKind::CK_LValueToRValue: {
    if (auto dr = dyn_cast<DeclRefExpr>(E->getSubExpr())) {
      if (dr->getDecl()->getName() == "warpSize") {
        bool foundVal = false;
        for (int i = scopes.size() - 1; i >= 0; i--) {
          auto found = scopes[i].find("warpSize");
          if (found != scopes[i].end()) {
            foundVal = true;
            break;
          }
        }
        if (!foundVal) {
          auto mlirType = getMLIRType(E->getType());
          auto llvmType = getLLVMTypeFromMLIRType(mlirType);
          return (mlir::Value)builder.create<mlir::LLVM::DialectCastOp>(
              loc, mlirType,
              builder.create<mlir::NVVM::WarpSizeOp>(loc, llvmType));
        }
      }
    }
    auto scalar = Visit(E->getSubExpr());
    auto off = scalar.offsets;
    if (off.size() == 0) {
      E->dump();
    }
    assert(off.size() != 0);
    if (!scalar.val) {
      E->getSubExpr()->dump();
    }
    assert(scalar.val);
    if (scalar.val.getType().cast<MemRefType>().getShape().size() !=
        off.size()) {

      E->getSubExpr()->dump();
      llvm::errs() << "{\n";
      for (auto a : off)
        a.dump();
      llvm::errs() << "}\n";
    }
    assert(scalar.val.getType().cast<MemRefType>().getShape().size() ==
           off.size());

    auto val = builder.create<mlir::LoadOp>(loc, scalar.val, off);
    if (E->getType()->isPointerType())
      return ValueWithOffsets(val, {getConstantIndex(0)});
    else
      return ValueWithOffsets(val, {});
  }
  case clang::CastKind::CK_IntegralToFloating: {
    auto scalar = Visit(E->getSubExpr());
    auto ty = getMLIRType(E->getType()).cast<mlir::FloatType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getSubExpr()->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (signedType)
      return (mlir::Value)builder.create<mlir::SIToFPOp>(
          loc, (mlir::Value)scalar, ty);
    else
      return (mlir::Value)builder.create<mlir::UIToFPOp>(
          loc, (mlir::Value)scalar, ty);
  }
  case clang::CastKind::CK_FloatingToIntegral: {
    auto scalar = Visit(E->getSubExpr());
    auto ty = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (signedType)
      return (mlir::Value)builder.create<mlir::FPToSIOp>(
          loc, (mlir::Value)scalar, ty);
    else
      return (mlir::Value)builder.create<mlir::FPToUIOp>(
          loc, (mlir::Value)scalar, ty);
  }
  case clang::CastKind::CK_IntegralCast: {
    auto scalar = Visit(E->getSubExpr());
    auto prevTy = scalar.getType().cast<mlir::IntegerType>();
    auto postTy = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }

    if (prevTy == postTy)
      return scalar;
    if (prevTy.getWidth() < postTy.getWidth()) {
      if (signedType) {
        return (mlir::Value)builder.create<mlir::SignExtendIOp>(
            loc, (mlir::Value)scalar, postTy);
      } else {
        return (mlir::Value)builder.create<mlir::ZeroExtendIOp>(
            loc, (mlir::Value)scalar, postTy);
      }
    } else {
      return (mlir::Value)builder.create<mlir::TruncateIOp>(
          loc, (mlir::Value)scalar, postTy);
    }
  }
  case clang::CastKind::CK_FloatingCast: {
    auto scalar = Visit(E->getSubExpr());
    auto prevTy = scalar.getType().cast<mlir::FloatType>();
    auto postTy = getMLIRType(E->getType()).cast<mlir::FloatType>();

    if (prevTy == postTy)
      return scalar;
    if (prevTy.getWidth() < postTy.getWidth()) {
      return (mlir::Value)builder.create<mlir::FPExtOp>(
          loc, (mlir::Value)scalar, postTy);
    } else {
      return (mlir::Value)builder.create<mlir::FPTruncOp>(
          loc, (mlir::Value)scalar, postTy);
    }
  }
  case clang::CastKind::CK_ArrayToPointerDecay: {
    auto scalar = Visit(E->getSubExpr());
    if (!scalar.val) {
      E->dump();
    }

    if (auto UO = dyn_cast<clang::UnaryOperator>(E->getSubExpr())) {
      if (UO->getOpcode() == clang::UnaryOperator::Opcode::UO_Deref) {
        auto off = scalar.offsets;
        assert(off.size() != 0);
        assert(scalar.val.getType().cast<MemRefType>().getShape().size() ==
               off.size());

        auto val = builder.create<mlir::LoadOp>(loc, scalar.val, off);
        if (UO->getType()->isPointerType())
          scalar = ValueWithOffsets(val, {getConstantIndex(0)});
        else
          scalar = ValueWithOffsets(val, {});
      }
    }

    assert(scalar.val);
    auto mt = scalar.val.getType().cast<mlir::MemRefType>();
    auto shape2 = std::vector<int64_t>(mt.getShape());
    shape2[0] = -1;
    auto nex = mlir::MemRefType::get(shape2, mt.getElementType(),
                                     mt.getAffineMaps(), mt.getMemorySpace());
    auto offs = scalar.offsets;
    offs.push_back(getConstantIndex(0));
    if (shape2.size() < offs.size()) {
      E->dump();
      llvm::errs() << "{\n";
      for (auto a : offs)
        a.dump();
      llvm::errs() << "}\n";
      nex.dump();
      assert(0);
    }
    return ValueWithOffsets(
        builder.create<mlir::MemRefCastOp>(loc, scalar.val, nex), offs);
  }
  case clang::CastKind::CK_FunctionToPointerDecay: {
    auto scalar = Visit(E->getSubExpr());
    return scalar;
  }
  case clang::CastKind::CK_NoOp: {
    return Visit(E->getSubExpr());
  }
  default:
    E->dump();
    assert(0 && "unhandled cast");
  }
}

ValueWithOffsets MLIRScanner::VisitIfStmt(clang::IfStmt *stmt) {
  auto cond = (mlir::Value)Visit(stmt->getCond());
  assert(cond != nullptr);

  bool hasElseRegion = stmt->getElse();
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond, hasElseRegion);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();
  builder.setInsertionPointToStart(&ifOp.thenRegion().back());
  Visit(stmt->getThen());
  if (hasElseRegion) {
    builder.setInsertionPointToStart(&ifOp.elseRegion().back());
    Visit(stmt->getElse());
  }

  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
  // return ifOp;
}

// todo visit break statement
// ValueWithOffsets

ValueWithOffsets
MLIRScanner::VisitConditionalOperator(clang::ConditionalOperator *E) {
  auto cond = (mlir::Value)Visit(E->getCond());
  assert(cond != nullptr);

  mlir::Type types[] = {getMLIRType(E->getType())};
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                              /*hasElseRegion*/ true);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();
  builder.setInsertionPointToStart(&ifOp.thenRegion().back());

  auto truev = Visit(E->getTrueExpr());
  assert(truev.val != nullptr);
  mlir::Value truearray[] = {truev.val};
  builder.create<mlir::scf::YieldOp>(loc, truearray);

  builder.setInsertionPointToStart(&ifOp.elseRegion().back());
  auto falsev = Visit(E->getFalseExpr());
  assert(truev.offsets == falsev.offsets);
  assert(falsev.val != nullptr);
  mlir::Value falsearray[] = {falsev.val};
  builder.create<mlir::scf::YieldOp>(loc, falsearray);

  builder.setInsertionPoint(oldblock, oldpoint);

  types[0] = truev.val.getType();
  auto newIfOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                 /*hasElseRegion*/ true);
  newIfOp.thenRegion().takeBody(ifOp.thenRegion());
  newIfOp.elseRegion().takeBody(ifOp.elseRegion());
  ifOp.erase();

  return ValueWithOffsets(newIfOp.getResult(0), truev.offsets);
  // return ifOp;
}

ValueWithOffsets MLIRScanner::VisitCompoundStmt(clang::CompoundStmt *stmt) {
  for (auto a : stmt->children())
    Visit(a);
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitBreakStmt(clang::BreakStmt *stmt) {
  assert(loops.size());
  builder.create<mlir::BranchOp>(loc, loops.back().exitB);
  return nullptr;
}
ValueWithOffsets MLIRScanner::VisitContinueStmt(clang::ContinueStmt *stmt) {
  assert(loops.size());
  builder.create<mlir::BranchOp>(loc, loops.back().condB);
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitReturnStmt(clang::ReturnStmt *stmt) {
  if (stmt->getRetValue()) {
    auto rv = (mlir::Value)Visit(stmt->getRetValue());
    assert(rv);
    builder.create<mlir::ReturnOp>(loc, rv);
  } else {
    builder.create<mlir::ReturnOp>(loc);
  }
  return nullptr;
}

std::map<const FunctionDecl *, mlir::FuncOp> functions;
mlir::FuncOp MLIRASTConsumer::GetOrCreateMLIRFunction(const FunctionDecl *FD) {
  if (functions.find(FD) != functions.end()) {
    return functions[FD];
  }
  std::string name = CGM.getMangledName(FD).str();

  std::vector<mlir::Type> types;
  std::vector<std::string> names;
  for (auto parm : FD->parameters()) {
    types.push_back(getMLIRType(parm->getOriginalType()));
    names.push_back(parm->getName().str());
  }

  // auto argTypes = getFunctionArgumentsTypes(mcg.getContext(),
  // inputTensors);
  auto rt = getMLIRType(FD->getReturnType());
  std::vector<mlir::Type> rettypes;
  if (!rt.isa<mlir::NoneType>()) {
    rettypes.push_back(rt);
  }
  mlir::OpBuilder builder(module.getContext());
  auto funcType = builder.getFunctionType(types, rettypes);
  mlir::FuncOp function = mlir::FuncOp(
      mlir::FuncOp::create(builder.getUnknownLoc(), name, funcType));
  functions[FD] = function;
  module.push_back(function);
  if (FD->isDefined())
    functionsToEmit.push_back(FD);
  return function;
}

void MLIRASTConsumer::run() {
  while (functionsToEmit.size()) {
    const FunctionDecl *todo = functionsToEmit.front();
    functionsToEmit.pop_front();
    if (done.count(todo))
      continue;
    done.insert(todo);
    MLIRScanner ms(*this, GetOrCreateMLIRFunction(todo), todo, module);
  }
}

bool MLIRASTConsumer::HandleTopLevelDecl(DeclGroupRef dg) {
  DeclGroupRef::iterator it;

  if (error)
    return true;
  for (it = dg.begin(); it != dg.end(); ++it) {
    FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(*it);
    if (!fd)
      continue;
    if (!fd->hasBody())
      continue;
    if (fd->getIdentifier() == nullptr)
      continue;
    // llvm::errs() << *fd << "  " << fd->isGlobal() << "\n";

    if (fd->getName() == fn) {
      functionsToEmit.push_back(fd);
    }
  }

  run();

  return true;
}

mlir::Location MLIRASTConsumer::getMLIRLocation(clang::SourceLocation loc) {
  auto lineNumber = SM.getSpellingLineNumber(loc);
  auto colNumber = SM.getSpellingColumnNumber(loc);
  auto fileId = SM.getFilename(loc);

  auto ctx = module.getContext();
  auto mlirIdentifier = Identifier::get(fileId, ctx);
  mlir::OpBuilder builder(ctx);
  return builder.getFileLineColLoc(mlirIdentifier, lineNumber, colNumber);
}

mlir::Type MLIRASTConsumer::getMLIRType(clang::QualType t) {
  if (t->isVoidType()) {
    mlir::OpBuilder builder(module.getContext());
    return builder.getNoneType();
  }
  llvm::Type *T = CGM.getTypes().ConvertType(t);
  return getMLIRType(T);
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
  if (auto IT = dyn_cast<llvm::IntegerType>(t)) {
    return builder.getIntegerType(IT->getBitWidth());
  }
  if (auto pt = dyn_cast<llvm::PointerType>(t)) {
    return mlir::MemRefType::get(-1, getMLIRType(pt->getElementType()), {},
                                 pt->getAddressSpace());
  }
  if (auto pt = dyn_cast<llvm::ArrayType>(t)) {
    auto under = getMLIRType(pt->getElementType());
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
    for (size_t i = 1; i < ST->getNumElements(); i++) {
      if (ST->getTypeAtIndex(i) != ST->getTypeAtIndex((size_t)0ULL)) {
        notAllSame = true;
        break;
      }
    }
    if (!notAllSame) {
      return mlir::MemRefType::get(
          ST->getNumElements(), getMLIRType(ST->getTypeAtIndex((size_t)0ULL)));
    }
  }
  // if (auto pt = dyn_cast<clang::RecordType>(t)) {
  //    llvm::errs() << " thing: " << pt->getName() << "\n";
  //}
  llvm::errs() << *t << "\n";
  assert(0 && "unknown type to convert");
  return nullptr;
}

#include "llvm/Support/Host.h"

#include "clang/Frontend/FrontendAction.h"
class MLIRAction : public clang::ASTFrontendAction {
public:
  std::string fn;
  mlir::ModuleOp &module;
  MLIRAction(std::string fn, mlir::ModuleOp &module) : fn(fn), module(module) {}
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    return std::unique_ptr<clang::ASTConsumer>(
        new MLIRASTConsumer(fn, CI.getPreprocessor(), CI.getASTContext(),
                            module, CI.getSourceManager()));
  }
};

mlir::FuncOp MLIRScanner::EmitDirectCallee(GlobalDecl GD) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());

  /*
  if (auto builtinID = FD->getBuiltinID()) {
      // Replaceable builtin provide their own implementation of a builtin.
  Unless
      // we are in the builtin implementation itself, don't call the actual
      // builtin. If we are in the builtin implementation, avoid trivial
  infinite
      // recursion.
      if (!FD->isInlineBuiltinDeclaration() ||
          CGF.CurFn->getName() == FD->getName())
      return CGCallee::forBuiltin(builtinID, FD);
  }
  */
  mlir::FuncOp V = Glob.GetOrCreateMLIRFunction(FD);
  /*
  if (!FD->hasPrototype()) {
      if (const FunctionProtoType *Proto =
              FD->getType()->getAs<FunctionProtoType>()) {
      // Ugly case: for a K&R-style definition, the type of the definition
      // isn't the same as the type of a use.  Correct for this with a
      // bitcast.
      QualType NoProtoType =
          CGM.getContext().getFunctionNoProtoType(Proto->getReturnType());
      NoProtoType = CGM.getContext().getPointerType(NoProtoType);
      V = llvm::ConstantExpr::getBitCast(V,
                                      CGM.getTypes().ConvertType(NoProtoType));
      }
  }
  */
  return V;
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
  return Glob.llvmMod.getDataLayout().getTypeSizeInBits(T) / 8;
}

// -cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu -S
// -disable-free -main-file-name saxpy.cu -mrelocation-model static
// -mframe-pointer=all -fno-rounding-math -fno-verbose-asm -no-integrated-as
// -aux-target-cpu x86-64 -fcuda-is-device -mlink-builtin-bitcode
// /usr/local/cuda/nvvm/libdevice/libdevice.10.bc -target-feature +ptx70
// -target-sdk-version=11.0 -target-cpu sm_35 -fno-split-dwarf-inlining
// -debugger-tuning=gdb -v -resource-dir lib/clang/12.0.0 -internal-isystem
// lib/clang/12.0.0/include/cuda_wrappers -internal-isystem
// /usr/local/cuda/include -include __clang_cuda_runtime_wrapper.h
// -internal-isystem
// /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0
// -internal-isystem
// /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0
// -internal-isystem
// /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0
// -internal-isystem
// /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward
// -internal-isystem
// /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0
// -internal-isystem
// /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0
// -internal-isystem
// /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0
// -internal-isystem
// /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward
// -internal-isystem /usr/local/include -internal-isystem
// lib/clang/12.0.0/include -internal-externc-isystem
// /usr/include/x86_64-linux-gnu -internal-externc-isystem /include
// -internal-externc-isystem /usr/include -internal-isystem /usr/local/include
// -internal-isystem lib/clang/12.0.0/include -internal-externc-isystem
// /usr/include/x86_64-linux-gnu -internal-externc-isystem /include
// -internal-externc-isystem /usr/include -fdeprecated-macro
// -fno-dwarf-directory-asm -fno-autolink -fdebug-compilation-dir
// /mnt/Data/git/MLIR-GPU/build -ferror-limit 19 -fgnuc-version=4.2.1
// -fcxx-exceptions -fexceptions -o /tmp/saxpy-a8baec.s -x cuda bin/saxpy.cu

#include "clang/Frontend/TextDiagnosticBuffer.h"
static bool parseMLIR(const char *filename, std::string fn,
                      std::vector<std::string> includeDirs,
                      mlir::ModuleOp &module) {
  std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Register the support for object-file-wrapped Clang modules.
  // auto PCHOps = Clang->getPCHContainerOperations();
  // PCHOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  // PCHOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);
  // if (invocation)
  //    Clang->setInvocation(std::shared_ptr<CompilerInvocation>(invocation));
  bool Success;
  {
    const char *binary = "clang";
    const unique_ptr<Driver> driver(
        new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags));
    std::vector<const char *> Argv;
    Argv.push_back(binary);
    Argv.push_back(filename);
    if (CudaLower)
      Argv.push_back("--cuda-gpu-arch=sm_35");
    for (auto a : includeDirs) {
      Argv.push_back("-I");
      char *chars = (char *)malloc(a.length() + 1);
      memcpy(chars, a.data(), a.length());
      chars[a.length()] = 0;
      Argv.push_back(chars);
    }

    const unique_ptr<Compilation> compilation(
        driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv)));
    JobList &Jobs = compilation->getJobs();
    if (Jobs.size() < 1)
      return false;

    Command *cmd = cast<Command>(&*Jobs.begin());
    if (strcmp(cmd->getCreator().getName(), "clang"))
      return false;

    const ArgStringList *args = &cmd->getArguments();

    Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(), *args,
                                                 Diags);
  }
  Clang->getInvocation().getFrontendOpts().DisableFree = false;

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
        LLVM_OBJ_ROOT "/lib/clang/" CLANG_VERSION_STRING;

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
  Clang->getTarget().adjust(Clang->getLangOpts());

  // Adjust target options based on codegen options.
  Clang->getTarget().adjustTargetOptions(Clang->getCodeGenOpts(),
                                         Clang->getTargetOpts());

  MLIRAction Act(fn, module);

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
      // llvm::errs() << "ended source file\n";
    }
  }
  return true;
}

#include "clang-mlir.h"

#include "llvm/Support/Debug.h"
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

#define DEBUG_TYPE "clang-mlir"

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
  if (Glob.globalVariables.find(name) != Glob.globalVariables.end()) {
    auto gv = Glob.GetOrCreateGlobal(Glob.globalVariables[name]);
    auto gv2 = builder.create<GetGlobalMemrefOp>(loc, gv.first.type(),
                                                 gv.first.getName());
    bool isArray = gv.second;
    if (isArray)
      return ValueWithOffsets(gv2, {});
    else
      return ValueWithOffsets(gv2, {getConstantIndex(0)});
  }
  if (Glob.globalFunctions.find(name) != Glob.globalFunctions.end()) {
    auto gv = Glob.GetOrCreateMLIRFunction(Glob.globalFunctions[name]);
    // TODO, how to represent?
    // return ValueWithOffsets(gv, std::vector<mlir::Value>());
  }
  llvm::errs() << "couldn't find " << name << "\n";
  assert(0 && "couldnt find value");
}

mlir::Type MLIRScanner::getLLVMTypeFromMLIRType(mlir::Type t) {
  if (auto it = t.dyn_cast<mlir::IntegerType>()) {
    return mlir::LLVM::LLVMIntegerType::get(t.getContext(), it.getWidth());
  }
  // t.dump();
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

ValueWithOffsets
MLIRScanner::VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *decl) {
  for (auto child : decl->children()) {
    child->dump();
  }
  decl->dump();
  assert(0 && "bad");
}

ValueWithOffsets MLIRScanner::VisitVarDecl(clang::VarDecl *decl) {
  auto loc = getMLIRLocation(decl->getLocation());
  unsigned memtype = 0;

  // if (decl->hasAttr<CUDADeviceAttr>() || decl->hasAttr<CUDAConstantAttr>()
  // ||
  if (decl->hasAttr<CUDASharedAttr>()) {
    memtype = 5;
  }

  mlir::Type subType = getMLIRType(decl->getType());
  mlir::Value inite = nullptr;
  if (auto init = decl->getInit()) {
    if (!isa<InitListExpr>(init)) {
      auto visit = Visit(init);
      inite = (mlir::Value)visit;
      if (!inite) {
        init->dump();
      }
      subType = inite.getType();
    }
  }
  bool isArray = isa<clang::ArrayType>(decl->getType());
  if (isa<llvm::StructType>(getLLVMType(decl->getType()))) {
    isArray = true;
  }
  auto op = createAllocOp(subType, decl->getName().str(), memtype, isArray);
  mlir::Value zeroIndex = getConstantIndex(0);
  if (inite) {
    builder.create<mlir::StoreOp>(loc, inite, op, zeroIndex);
  } else if (auto init = decl->getInit()) {
    if (auto il = dyn_cast<InitListExpr>(init)) {
      if (il->hasArrayFiller()) {
        auto visit = Visit(il->getInit(0));
        inite = (mlir::Value)visit;
        if (!inite) {
          il->getArrayFiller()->dump();
          assert(inite);
        }
        assert(subType.cast<MemRefType>().getShape().size() == 1);
        for (size_t i = 0; i < subType.cast<MemRefType>().getShape()[0]; i++) {
          builder.create<mlir::StoreOp>(loc, inite, op, getConstantIndex(i));
        }
      } else {
        init->dump();
        assert(0 && "init list expr unhandled");
      }
    }
  }
  return ValueWithOffsets(op, {zeroIndex});
}

bool MLIRScanner::getLowerBound(clang::ForStmt *fors,
                                AffineLoopDescriptor &descr) {
  auto init = fors->getInit();
  if (auto declStmt = dyn_cast<DeclStmt>(init))
    if (declStmt->isSingleDecl()) {
      auto decl = declStmt->getSingleDecl();
      if (auto varDecl = dyn_cast<VarDecl>(decl)) {
        if (varDecl->hasInit()) {
          auto init = varDecl->getInit();

          mlir::Value val = (mlir::Value)Visit(init);
          descr.setName(varDecl->getName().str());
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
        mlir::Value val = (mlir::Value)Visit(binOp->getRHS());
        val = builder.create<mlir::IndexCastOp>(
            loc, val, mlir::IndexType::get(builder.getContext()));
        descr.setName(declRefStmt->getNameInfo().getAsString());
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
bool matchIndvar(const Expr *expr, std::string indVar) {
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr))
    if (auto declRef = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      auto declRefName = declRef->getDecl()->getName().str();
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
      mlir::Value val = (mlir::Value)Visit(rhs);
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
      mlir::Value val = (mlir::Value)Visit(rhs);
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

  auto val = affineOp.getInductionVar();

  reg.front().clear();

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToEnd(&reg.front());

  if (!descr.getForwardMode()) {
    val = builder.create<mlir::SubIOp>(loc, val, lb);
    val = builder.create<mlir::SubIOp>(
        loc, builder.create<mlir::SubIOp>(loc, ub, getConstantIndex(1)), val);
  }
  auto idx = builder.create<mlir::IndexCastOp>(loc, val, descr.getType());
  createAndSetAllocOp(descr.getName(), idx, 0);

  // TODO: set loop context.
  Visit(fors->getBody());
  builder.create<AffineYieldOp>(loc);

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(oldblock, oldpoint);
}

void MLIRScanner::buildAffineLoop(clang::ForStmt *fors, mlir::Location loc,
                                  const AffineLoopDescriptor &descr) {
  mlir::Value lb = descr.getLowerBound();
  mlir::Value ub = descr.getUpperBound();
  buildAffineLoopImpl(fors, loc, lb, ub, descr);
  return;
}

ValueWithOffsets MLIRScanner::VisitForStmt(clang::ForStmt *fors) {
  // fors->dump();
  scopes.emplace_back();

  auto loc = getMLIRLocation(fors->getForLoc());

  AffineLoopDescriptor affineLoopDescr;
  if (Glob.scopLocList.isInScop(fors->getForLoc()) &&
      isTrivialAffineLoop(fors, affineLoopDescr)) {
    buildAffineLoop(fors, loc, affineLoopDescr);
  } else {

    if (auto s = fors->getInit()) {
      Visit(s);
    }

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
  }
  scopes.pop_back();
  return nullptr;
}

mlir::Value add(MLIRScanner &sc, mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value lhs, mlir::Value rhs) {
  assert(lhs);
  assert(rhs);
  if (auto op = lhs.getDefiningOp<ConstantOp>()) {
    if (op.getValue().cast<IntegerAttr>().getInt() == 0) {
      return rhs;
    }
  }

  if (auto op = lhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.getValue() == 0) {
      return rhs;
    }
  }

  if (auto op = rhs.getDefiningOp<ConstantOp>()) {
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

  if (auto op = val.getDefiningOp<ConstantOp>())
    return getConstantIndex(op.getValue().cast<IntegerAttr>().getInt());

  return builder.create<mlir::IndexCastOp>(
      loc, val, mlir::IndexType::get(val.getContext()));
}

ValueWithOffsets
MLIRScanner::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr) {
  auto lhs = Visit(expr->getLHS());
  // Check the LHS has been successfully emitted
  assert(lhs.val);
  auto rhs = (mlir::Value)Visit(expr->getRHS());
  // Check the RHS has been successfully emitted
  assert(rhs);
  auto offsets = lhs.offsets;
  auto idx = castToIndex(getMLIRLocation(expr->getRBracketLoc()), rhs);
  assert(offsets.size() > 0);
  offsets[offsets.size() - 1] =
      add(*this, builder, loc, lhs.offsets.back(), idx);
  return ValueWithOffsets(lhs.val, offsets);
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
      if (sr->getDecl()->getName() == "expf" ||
          sr->getDecl()->getName() == "exp") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back((mlir::Value)Visit(a));
        }
        return (mlir::Value)builder.create<mlir::ExpOp>(loc, args[0]);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "sin") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back((mlir::Value)Visit(a));
        }
        return (mlir::Value)builder.create<mlir::SinOp>(loc, args[0]);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "cos") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back((mlir::Value)Visit(a));
        }
        return (mlir::Value)builder.create<mlir::CosOp>(loc, args[0]);
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
      if (sr->getDecl()->getName() == "__powf" ||
          sr->getDecl()->getName() == "pow" ||
          sr->getDecl()->getName() == "powf") {
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
      if (sr->getDecl()->getName() == "fprintf" ||
          sr->getDecl()->getName() == "printf") {

        auto tocall = EmitCallee(expr->getCallee());
        auto fprintfF = Glob.GetOrCreateLLVMFunction(tocall);

        std::vector<mlir::Value> args;
        size_t i = 0;
        for (auto a : expr->arguments()) {
          if (i == 0 && sr->getDecl()->getName() == "fprintf") {
            auto decl =
                cast<DeclRefExpr>(cast<ImplicitCastExpr>(a)->getSubExpr());
            args.push_back(builder.create<mlir::LLVM::LoadOp>(
                loc, builder.create<mlir::LLVM::AddressOfOp>(
                         loc, Glob.GetOrCreateLLVMGlobal(
                                  cast<VarDecl>(decl->getDecl())))));
            i++;
            continue;
          }

          if (auto IC1 = dyn_cast<ImplicitCastExpr>(a)) {
            if (auto IC2 = dyn_cast<ImplicitCastExpr>(IC1->getSubExpr())) {
              if (auto slit =
                      dyn_cast<clang::StringLiteral>(IC2->getSubExpr())) {
                args.push_back(Glob.GetOrCreateGlobalLLVMString(
                    loc, builder, slit->getString()));
                i++;
                continue;
              }
            }
            if (auto slit = dyn_cast<clang::StringLiteral>(IC1->getSubExpr())) {
              args.push_back(Glob.GetOrCreateGlobalLLVMString(
                  loc, builder, slit->getString()));
              i++;
              continue;
            }
          }

          mlir::Value val = (mlir::Value)Visit(a);
          auto llvmType =
              Glob.typeTranslator.translateType(getLLVMType(a->getType()));
          val = builder.create<mlir::LLVM::DialectCastOp>(loc, llvmType, val);
          args.push_back(val);
          i++;
        }

        builder.create<mlir::LLVM::CallOp>(loc, fprintfF, args);

        return nullptr;
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "strcmp") {
        auto tocall = EmitCallee(expr->getCallee());
        auto strcmpF = Glob.GetOrCreateLLVMFunction(tocall);

        std::vector<mlir::Value> args;
        size_t i = 0;
        for (auto a : expr->arguments()) {
          if (auto IC1 = dyn_cast<ImplicitCastExpr>(a)) {
            if (auto IC2 = dyn_cast<ImplicitCastExpr>(IC1->getSubExpr())) {
              if (auto slit =
                      dyn_cast<clang::StringLiteral>(IC2->getSubExpr())) {
                args.push_back(Glob.GetOrCreateGlobalLLVMString(
                    loc, builder, slit->getString()));
                i++;
                continue;
              }
            }
            if (auto slit = dyn_cast<clang::StringLiteral>(IC1->getSubExpr())) {
              args.push_back(Glob.GetOrCreateGlobalLLVMString(
                  loc, builder, slit->getString()));
              i++;
              continue;
            }
          }
          mlir::Value val = (mlir::Value)Visit(a);
          auto llvmType =
              Glob.typeTranslator.translateType(getLLVMType(a->getType()));
          if (val.getType() != llvmType)
            val = builder.create<mlir::LLVM::DialectCastOp>(loc, llvmType, val);
          args.push_back(val);
          i++;
        }

        return ValueWithOffsets(
            builder.create<mlir::LLVM::DialectCastOp>(
                loc, getMLIRType(expr->getType()),
                builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args)
                    .getResult(0)),
            {});
      }
    }

  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "gettimeofday") {
        auto tocall = EmitCallee(expr->getCallee());
        auto fprintfF = Glob.GetOrCreateLLVMFunction(tocall);

        std::vector<mlir::Value> args;
        size_t i = 0;
        mlir::Value tostore = nullptr;
        mlir::Value alloc;
        for (auto a : expr->arguments()) {

          if (i == 0) {
            tostore = (mlir::Value)Visit(a);
            i++;
            LLVM::LLVMType indexType =
                LLVM::LLVMType::getIntNTy(module.getContext(), 64);
            auto one = builder.create<LLVM::ConstantOp>(
                loc, indexType,
                builder.getIntegerAttr(builder.getIndexType(), 1));
            alloc = builder.create<mlir::LLVM::AllocaOp>(
                loc,
                Glob.typeTranslator.translateType((getLLVMType(a->getType()))),
                one, 0);
            args.push_back(alloc);
            continue;
          }
          auto llvmType =
              Glob.typeTranslator.translateType(getLLVMType(a->getType()));

          if (auto IC1 = dyn_cast<ImplicitCastExpr>(a)) {
            if (IC1->getCastKind() == clang::CastKind::CK_NullToPointer) {
              args.push_back(builder.create<mlir::LLVM::NullOp>(loc, llvmType));
              i++;
              continue;
            }
          }
          mlir::Value val = (mlir::Value)Visit(a);
          if (!isa<mlir::LLVM::NullOp>(val.getDefiningOp()))
            val = builder.create<mlir::LLVM::DialectCastOp>(loc, llvmType, val);
          args.push_back(val);
          i++;
        }
        assert(alloc);

        auto co = builder.create<mlir::LLVM::CallOp>(loc, fprintfF, args)
                      .getResult(0);
        co = builder.create<mlir::LLVM::DialectCastOp>(
            loc, getMLIRType(expr->getType()), co);
        auto ret = ValueWithOffsets(co, {});

        auto loaded = builder.create<mlir::LLVM::LoadOp>(loc, alloc);

        auto st = loaded.getType().cast<LLVM::LLVMType>();
        for (size_t i = 0; i < st.getStructNumElements(); i++) {
          mlir::Value ev = builder.create<mlir::LLVM::ExtractValueOp>(
              loc, st.getStructElementType(i), loaded,
              builder.getI64ArrayAttr(i));
          ev = builder.create<mlir::LLVM::DialectCastOp>(
              loc,
              Glob.getMLIRType(Glob.reverseTypeTranslator.translateType(
                  ev.getType().cast<LLVM::LLVMType>())),
              ev);
          builder.create<mlir::StoreOp>(
              loc, ev, tostore,
              std::vector<mlir::Value>({getConstantIndex(i)}));
        }

        return ret;
      }
    }

  auto tocall = EmitDirectCallee(EmitCallee(expr->getCallee()));
  std::vector<mlir::Value> args;
  auto fnType = tocall.getType();
  size_t i = 0;
  for (auto a : expr->arguments()) {
    mlir::Value val = (mlir::Value)Visit(a);
    if (i >= fnType.getInputs().size()) {
      expr->dump();
      tocall.dump();
      fnType.dump();
      assert(0 && "too many arguments in calls");
    }
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
  case clang::UnaryOperator::Opcode::UO_Extension: {
    return sub;
  }
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

bool isInAffineScope(Operation *op) {
  auto *curOp = op;
  while (auto *parentOp = curOp->getParentOp()) {
    if (isa<mlir::AffineForOp>(parentOp))
      return true;
    curOp = parentOp;
  }
  return false;
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
  if (isa<mlir::LoadOp>(op)) {
    auto load = cast<mlir::LoadOp>(op);
    auto loadOperand = load.getOperand(0);
    if (loadOperand.getType().cast<MemRefType>().getShape().size() != 1)
      return false;
    auto maybeAllocaOp = loadOperand.getDefiningOp();
    if (!isa<mlir::AllocaOp>(maybeAllocaOp))
      return false;
    auto allocaUsers = maybeAllocaOp->getUsers();
    if (llvm::none_of(allocaUsers, [](mlir::Operation *op) {
          if (isa<mlir::StoreOp>(op))
            return true;
          return false;
        }))
      return false;
    for (auto user : allocaUsers)
      if (auto storeOp = dyn_cast<mlir::StoreOp>(user)) {
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

  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_LAnd: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, (mlir::Value)lhs,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.thenRegion().back());

    auto rhs = Visit(BO->getRHS());
    assert(rhs.val != nullptr);
    mlir::Value truearray[] = {rhs.val};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.elseRegion().back());
    mlir::Value falsearray[] = {builder.create<mlir::ConstantOp>(
        loc, types[0], builder.getIntegerAttr(types[0], 0))};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);
    return ValueWithOffsets(ifOp.getResult(0), {});
  }
  case clang::BinaryOperator::Opcode::BO_LOr: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, (mlir::Value)lhs,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.thenRegion().back());

    mlir::Value truearray[] = {builder.create<mlir::ConstantOp>(
        loc, types[0], builder.getIntegerAttr(types[0], 1))};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.elseRegion().back());
    auto rhs = Visit(BO->getRHS());
    assert(rhs.val != nullptr);
    mlir::Value falsearray[] = {rhs.val};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);

    return ValueWithOffsets(ifOp.getResult(0), {});
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
  case clang::BinaryOperator::Opcode::BO_And: {
    return (mlir::Value)builder.create<mlir::AndOp>(loc, (mlir::Value)lhs,
                                                    (mlir::Value)rhs);
  }
  case clang::BinaryOperator::Opcode::BO_Or: {
    // TODO short circuit
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
    mlir::Value tostore = nullptr;

    if (auto BO2 = dyn_cast<clang::BinaryOperator>(BO->getRHS())) {
      if (BO2->getOpcode() == clang::BinaryOperator::Opcode::BO_Assign) {

        auto off = rhs.offsets;
        assert(off.size() != 0);
        assert(rhs.val.getType().cast<MemRefType>().getShape().size() ==
               off.size());

        tostore = builder.create<mlir::LoadOp>(loc, rhs.val, off);
      }
    }

    if (tostore == nullptr) {
      tostore = (mlir::Value)rhs;
    }
    builder.create<mlir::StoreOp>(loc, tostore, lhs.val, off);
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
  return getValue(E->getDecl()->getName().str());
}

ValueWithOffsets MLIRScanner::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  if (!E->getSourceExpr()) {
    E->dump();
  }
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
  // ME->getBase()->getType().getDesugaredType(Glob.astContext)->dump();
  auto rd = cast<RecordType>(
                ME->getBase()->getType().getDesugaredType(Glob.astContext))
                ->getDecl();
  auto &layout = Glob.CGM.getTypes().getCGRecordLayout(rd);
  const FieldDecl *field = nullptr;
  for (auto f : rd->fields()) {
    if (f->getName() == memberName) {
      field = f;
    }
  }
  // llvm::errs() << "md name: " << memberName << "\n";
  assert(field);
  return ValueWithOffsets((mlir::Value)base,
                          {getConstantIndex(layout.getLLVMFieldNo(field))});
}

ValueWithOffsets MLIRScanner::VisitCastExpr(CastExpr *E) {
  switch (E->getCastKind()) {

  case clang::CastKind::CK_NullToPointer: {
    auto llvmType =
        Glob.typeTranslator.translateType(getLLVMType(E->getType()));
    return ValueWithOffsets(builder.create<mlir::LLVM::NullOp>(loc, llvmType));
  }

  case clang::CastKind::CK_BitCast: {

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getName() == "polybench_alloc_data") {
            auto mt = getMLIRType(E->getType()).cast<mlir::MemRefType>();
            auto inner = mt.getElementType().cast<mlir::MemRefType>();

            auto shape = std::vector<int64_t>(mt.getShape());
            shape[0] = 1;
            auto mt0 =
                mlir::MemRefType::get(shape, mt.getElementType(),
                                      mt.getAffineMaps(), mt.getMemorySpace());

            auto alloc = builder.create<mlir::AllocOp>(loc, inner);
            auto alloc2 = builder.create<mlir::MemRefCastOp>(
                loc, builder.create<mlir::AllocaOp>(loc, mt0), mt);

            mlir::Value zeroIndex = getConstantIndex(0);
            builder.create<mlir::StoreOp>(loc, alloc, alloc2, zeroIndex);
            return ValueWithOffsets(alloc2, {getConstantIndex(0)});

            // std::vector<mlir::Value> args;
            // for (auto a : expr->arguments()) {
            //  args.push_back((mlir::Value)Visit(a));
            //}
            // return (mlir::Value)builder.create<mlir::SqrtOp>(loc, args[0]);
          }
        }

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
    if (scalar.val.getType().isa<mlir::LLVM::LLVMType>()) {
      assert(off.size() == 1);
      mlir::Value val = nullptr;

      if (auto op = off[0].getDefiningOp<ConstantOp>()) {
        if (op.getValue().cast<IntegerAttr>().getInt() == 0) {
          val = scalar.val;
        }
      }

      if (auto op = off[0].getDefiningOp<ConstantIndexOp>()) {
        if (op.getValue() == 0) {
          val = scalar.val;
        }
      }

      if (val == nullptr) {
        llvm::errs() << "doing the nullptr variant: " << off[0] << "\n";
        std::vector<mlir::Value> vals = {scalar.val};
        for (auto v : off) {
          auto llvmType = LLVM::LLVMType::getInt64Ty(builder.getContext());
          v = builder.create<mlir::IndexCastOp>(loc, v,
                                                builder.getIntegerType(64));
          vals.push_back((mlir::Value)builder.create<mlir::LLVM::DialectCastOp>(
              loc, llvmType, v));
        }
        val =
            builder.create<mlir::LLVM::GEPOp>(loc, scalar.val.getType(), vals);
      }
      val = builder.create<mlir::LLVM::LoadOp>(loc, val);
      if (E->getType()->isPointerType())
        return ValueWithOffsets(val, {getConstantIndex(0)});
      else
        return ValueWithOffsets(val, {});
    }
    if (scalar.val.getType().cast<MemRefType>().getShape().size() !=
        off.size()) {

      E->getSubExpr()->dump();
      llvm::errs() << "scalar.val: " << scalar.val << "\n";
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

    auto Sub = E->getSubExpr();
    while (auto SE = dyn_cast<ParenExpr>(Sub)) {
      Sub = SE->getSubExpr();
    }

    if (auto UO = dyn_cast<clang::UnaryOperator>(Sub)) {
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
  case clang::CastKind::CK_ToVoid: {
    Visit(E->getSubExpr());
    return nullptr;
  }
  default:
    E->dump();
    assert(0 && "unhandled cast");
  }
}

ValueWithOffsets MLIRScanner::VisitIfStmt(clang::IfStmt *stmt) {
  auto loc = getMLIRLocation(stmt->getIfLoc());
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

mlir::LLVM::LLVMFuncOp
MLIRASTConsumer::GetOrCreateLLVMFunction(const FunctionDecl *FD) {
  if (llvmFunctions.find(FD) != llvmFunctions.end()) {
    return llvmFunctions[FD];
  }
  std::string name = CGM.getMangledName(FD).str();
  std::vector<mlir::LLVM::LLVMType> types;
  for (auto parm : FD->parameters()) {
    types.push_back(
        typeTranslator.translateType(getLLVMType(parm->getOriginalType())));
  }

  auto rt = typeTranslator.translateType(getLLVMType(FD->getReturnType()));

  auto llvmFnType =
      LLVM::LLVMType::getFunctionTy(rt, types,
                                    /*isVarArg=*/FD->isVariadic());

  // Insert the function into the body of the parent module.

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  return llvmFunctions[FD] = builder.create<LLVM::LLVMFuncOp>(module.getLoc(),
                                                              name, llvmFnType);
}

mlir::LLVM::GlobalOp MLIRASTConsumer::GetOrCreateLLVMGlobal(const VarDecl *FD) {
  if (llvmGlobals.find(FD) != llvmGlobals.end()) {
    return llvmGlobals[FD];
  }

  auto rt = typeTranslator.translateType(getLLVMType(FD->getType()));

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  // auto lnk = CGM.getLLVMLinkageVarDefinition(FD, /*isConstant*/false);
  // TODO handle proper global linkage
  auto lnk = LLVM::Linkage::External;
  return llvmGlobals[FD] = builder.create<LLVM::GlobalOp>(
             module.getLoc(), rt, /*constant*/ false, lnk, FD->getName(),
             mlir::Attribute());
}

std::pair<mlir::GlobalMemrefOp, bool>
MLIRASTConsumer::GetOrCreateGlobal(const VarDecl *FD) {
  if (globals.find(FD) != globals.end()) {
    return globals[FD];
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

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  // auto lnk = CGM.getLLVMLinkageVarDefinition(FD, /*isConstant*/false);
  // TODO handle proper global linkage
  // builder.getStringAttr("public")
  auto globalOp = builder.create<mlir::GlobalMemrefOp>(
      module.getLoc(), FD->getName(), mlir::StringAttr(),
      mlir::TypeAttr::get(mr), mlir::Attribute(), false);
  // Private == internal, Public == External [in lowering]
  SymbolTable::setSymbolVisibility(globalOp, SymbolTable::Visibility::Private);
  return globals[FD] = std::make_pair(globalOp, isArray);
}

mlir::Value MLIRASTConsumer::GetOrCreateGlobalLLVMString(
    mlir::Location loc, mlir::OpBuilder &builder, StringRef value) {
  using namespace mlir;
  // Create the global at the entry of the module.
  if (llvmStringGlobals.find(value.str()) == llvmStringGlobals.end()) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMType::getArrayTy(
        LLVM::LLVMType::getInt8Ty(builder.getContext()), value.size() + 1);
    llvmStringGlobals[value.str()] = builder.create<LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
        "str" + std::to_string(llvmStringGlobals.size()),
        builder.getStringAttr(value.str() + '\0'));
  }

  LLVM::GlobalOp global = llvmStringGlobals[value.str()];
  // Get the pointer to the first character in the global string.
  mlir::Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  mlir::Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt64Ty(builder.getContext()),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMType::getInt8PtrTy(builder.getContext()), globalPtr,
      ArrayRef<mlir::Value>({cst0, cst0}));
}

mlir::FuncOp MLIRASTConsumer::GetOrCreateMLIRFunction(const FunctionDecl *FD) {
  std::string name = CGM.getMangledName(FD).str();
  if (functions.find(name) != functions.end()) {
    return functions[name];
  }

  std::vector<mlir::Type> types;
  std::vector<std::string> names;
  for (auto parm : FD->parameters()) {
    if (name == "main" && types.size() == 1) {
      types.push_back(
          typeTranslator.translateType(getLLVMType(parm->getOriginalType())));
    } else {
      types.push_back(getMLIRType(parm->getOriginalType()));
    }
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
  if (FD->getLinkageInternal() == clang::Linkage::InternalLinkage ||
      !FD->isDefined()) {
    SymbolTable::setSymbolVisibility(function,
                                     SymbolTable::Visibility::Private);
  } else {
    SymbolTable::setSymbolVisibility(function, SymbolTable::Visibility::Public);
  }

  functions[name] = function;
  module.push_back(function);
  if (FD->isDefined())
    functionsToEmit.push_back(FD);
  else
    emitIfFound.insert(FD->getName().str());
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
    if (VarDecl *fd = dyn_cast<clang::VarDecl>(*it)) {
      globalVariables[fd->getName().str()] = fd;
    }
    if (FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(*it)) {
      globalFunctions[fd->getName().str()] = fd;
    }
  }

  for (it = dg.begin(); it != dg.end(); ++it) {
    FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(*it);
    if (!fd)
      continue;
    if (!fd->hasBody())
      continue;
    if (fd->getIdentifier() == nullptr)
      continue;
    // llvm::errs() << *fd << "  " << fd->isGlobal() << "\n";
    if (emitIfFound.count(fd->getName().str())) {
      functionsToEmit.push_back(fd);
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
      if (ST->getTypeAtIndex(i) != ST->getTypeAtIndex(0U)) {
        notAllSame = true;
        break;
      }
    }
    if (!notAllSame) {
      return mlir::MemRefType::get(ST->getNumElements(),
                                   getMLIRType(ST->getTypeAtIndex(0U)));
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
  std::set<std::string> emitIfFound;
  mlir::ModuleOp &module;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmStringGlobals;
  std::map<std::string, mlir::FuncOp> functions;
  MLIRAction(std::string fn, mlir::ModuleOp &module) : module(module) {
    emitIfFound.insert(fn);
  }
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    return std::unique_ptr<clang::ASTConsumer>(new MLIRASTConsumer(
        emitIfFound, llvmStringGlobals, functions, CI.getPreprocessor(),
        CI.getASTContext(), module, CI.getSourceManager()));
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

#include "clang/Frontend/TextDiagnosticBuffer.h"
static bool parseMLIR(std::vector<std::string> filenames, std::string fn,
                      std::vector<std::string> includeDirs,
                      std::vector<std::string> defines, mlir::ModuleOp &module,
                      llvm::Triple &triple, llvm::DataLayout &DL) {

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
  //{
  const char *binary = "clang";
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
  if (CudaLower)
    Argv.push_back("--cuda-gpu-arch=sm_35");
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
    //}
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

    module.setAttr(
        LLVM::LLVMDialect::getDataLayoutAttrName(),
        StringAttr::get(
            Clang->getTarget().getDataLayout().getStringRepresentation(),
            module.getContext()));
    module.setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                   StringAttr::get(Clang->getTarget().getTriple().getTriple(),
                                   module.getContext()));
    // module.llvmModule->setDataLayout(DL);
    // llvmModule->setTargetTriple(triple.getTriple());

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
    DL = Clang->getTarget().getDataLayout();
    triple = Clang->getTarget().getTriple();
  }
  return true;
}

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

class IfScope {
public:
  MLIRScanner &scanner;
  IfScope(MLIRScanner &scanner) : scanner(scanner) { scanner.pushLoopIf(); }
  ~IfScope() { scanner.popLoopIf(); }
};
void MLIRScanner::setValue(std::string name, ValueWithOffsets &&val) {
  auto z = scopes.back().emplace(name, val);
  assert(z.second);
  //assert(val.offsets.size() == z.first->second.offsets.size());
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
    // TODO check reference
    if (isArray)
      return ValueWithOffsets(gv2, /*isReference*/true);
    else
      return ValueWithOffsets(gv2, /*isReference*/true);
    //return gv2;
  }
  if (Glob.globalFunctions.find(name) != Glob.globalFunctions.end()) {
    auto gv = Glob.GetOrCreateMLIRFunction(Glob.globalFunctions[name]);
    // TODO, how to represent?
    // return ValueWithOffsets(gv, std::vector<mlir::Value>());
  }
  llvm::errs() << "couldn't find " << name << "\n";
  assert(0 && "couldnt find value");
  return nullptr;
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
    setValue(name, ValueWithOffsets(alloc, /*isReference*/true));
  else
    setValue(name, ValueWithOffsets(alloc, /*isReference*/true));
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
  IfScope scope(*this);
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
  return ValueWithOffsets(builder.create<mlir::ConstantOp>(
      loc, ty, builder.getIntegerAttr(ty, expr->getValue())), /*isReference*/false);
}

ValueWithOffsets
MLIRScanner::VisitFloatingLiteral(clang::FloatingLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::FloatType>();
  return ValueWithOffsets(builder.create<mlir::ConstantOp>(
      loc, ty, builder.getFloatAttr(ty, expr->getValue())), /*isReference*/false);
}

ValueWithOffsets
MLIRScanner::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueWithOffsets(builder.create<mlir::ConstantOp>(
      loc, ty, builder.getIntegerAttr(ty, expr->getValue())), /*isReference*/false);
}

ValueWithOffsets MLIRScanner::VisitStringLiteral(clang::StringLiteral *expr) {
  auto ty = getMLIRType(expr->getType());
  return ValueWithOffsets(builder.create<mlir::ConstantOp>(
      loc, ty, builder.getStringAttr(expr->getBytes())), /*isReference*/false);
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
      inite = visit.getValue(builder);
      if (!inite) {
        init->dump();
        assert(0 && inite);
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
        inite = visit.getValue(builder);
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
  return ValueWithOffsets(op, /*isReference*/true);
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

          mlir::Value val = Visit(init).getValue(builder);
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
        mlir::Value val = Visit(binOp->getRHS()).getValue(builder);
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

ValueWithOffsets MLIRScanner::VisitWhileStmt(clang::WhileStmt *fors) {
  IfScope scope(*this);
  scopes.emplace_back();

  auto loc = getMLIRLocation(fors->getLParenLoc());

    auto i1Ty = builder.getIntegerType(1);
    auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
    auto truev = builder.create<mlir::ConstantOp>(
      loc, i1Ty, builder.getIntegerAttr(i1Ty, 1));
    loops.push_back((LoopContext){builder.create<mlir::AllocaOp>(loc, type), builder.create<mlir::AllocaOp>(loc, type)});
    builder.create<mlir::StoreOp>(loc, truev, loops.back().noBreak);

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
      builder.create<mlir::CondBranchOp>(loc, condRes.getValue(builder), &bodyB,
                                         &exitB);
    }

    builder.setInsertionPointToStart(&bodyB);
    builder.create<mlir::StoreOp>(loc, 
      builder.create<mlir::LoadOp>(loc, loops.back().noBreak, std::vector<mlir::Value>()),
      loops.back().keepRunning, 
      std::vector<mlir::Value>());
    
    Visit(fors->getBody());
    loops.pop_back();
    if (builder.getInsertionBlock()->empty() ||
        builder.getInsertionBlock()->back().isKnownNonTerminator()) {
      builder.create<mlir::BranchOp>(loc, &condB);
    }

    builder.setInsertionPointToStart(&exitB);

  scopes.pop_back();
  return nullptr;
}


ValueWithOffsets MLIRScanner::VisitForStmt(clang::ForStmt *fors) {
  IfScope scope(*this);
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

    auto i1Ty = builder.getIntegerType(1);
    auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
    auto truev = builder.create<mlir::ConstantOp>(
        loc, i1Ty, builder.getIntegerAttr(i1Ty, 1));
    loops.push_back((LoopContext){builder.create<mlir::AllocaOp>(loc, type),
                                  builder.create<mlir::AllocaOp>(loc, type)});
    builder.create<mlir::StoreOp>(loc, truev, loops.back().noBreak);

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
      builder.create<mlir::CondBranchOp>(loc, condRes.getValue(builder), &bodyB,
                                         &exitB);
    }

    builder.setInsertionPointToStart(&bodyB);
    builder.create<mlir::StoreOp>(
        loc,
        builder.create<mlir::LoadOp>(loc, loops.back().noBreak,
                                     std::vector<mlir::Value>()),
        loops.back().keepRunning, std::vector<mlir::Value>());

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
  auto moo = Visit(expr->getLHS());

  auto lhs = moo.getValue(builder);
  // Check the LHS has been successfully emitted
  assert(lhs);
  //mlir::Value val = lhs.getValue(builder);
  
  mlir::Value val = lhs;
  
  auto rhs = Visit(expr->getRHS()).getValue(builder);
  // Check the RHS has been successfully emitted
  assert(rhs);
  auto idx = castToIndex(getMLIRLocation(expr->getRBracketLoc()), rhs);

  if (val.getType().isa<LLVM::LLVMType>()) {

    std::vector<mlir::Value> vals = {val};
    auto llvmType = LLVM::LLVMType::getInt64Ty(builder.getContext());
    idx = builder.create<mlir::IndexCastOp>(loc, idx,
                                          builder.getIntegerType(64));
    vals.push_back(builder.create<mlir::LLVM::DialectCastOp>(
        loc, llvmType, idx));
    // TODO sub
    return ValueWithOffsets(builder.create<mlir::LLVM::GEPOp>(loc, val.getType(), vals), /*isReference*/false).dereference(builder);
  }
  if (!val.getType().isa<MemRefType>()) {
    builder.getInsertionBlock()->dump();
    function.dump();
    expr->dump();
    llvm::errs() << val << "\n";
  }

  ValueWithOffsets dref;
  {
    auto mt = val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    //if (shape.size() > 1) {
    //  shape.erase(shape.begin());
    //} else {
      shape[0] = -1;
    //}
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              mt.getAffineMaps(), mt.getMemorySpace());
    auto post = builder.create<SubIndexOp>(loc, mt0, val, idx);
    // TODO sub
    dref = ValueWithOffsets(post, /*isReference*/false).dereference(builder);
  }
  assert(dref.isReference);

  auto mt = dref.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() > 1) {
    shape.erase(shape.begin());
  } else {
    shape[0] = -1;
  }
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            mt.getAffineMaps(), mt.getMemorySpace());
  auto post = builder.create<SubIndexOp>(loc, mt0, dref.val, getConstantIndex(0));
  return ValueWithOffsets(post, /*isReference*/true);
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
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "x"),
                  mlirType), /*isReference*/false);
            }
            if (memberName == "__fetch_builtin_y") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "y"),
                  mlirType), /*isReference*/false);
            }
            if (memberName == "__fetch_builtin_z") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "z"),
                  mlirType), /*isReference*/false);
            }
          }
          if (sr->getDecl()->getName() == "blockDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "x"),
                  mlirType), /*isReference*/false);
            }
            if (memberName == "__fetch_builtin_y") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "y"),
                  mlirType), /*isReference*/false);
            }
            if (memberName == "__fetch_builtin_z") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::BlockDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "z"),
                  mlirType), /*isReference*/false);
            }
          }
          if (sr->getDecl()->getName() == "threadIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::ThreadIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "x"),
                  mlirType), /*isReference*/false);
            }
            if (memberName == "__fetch_builtin_y") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::ThreadIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "y"),
                  mlirType), /*isReference*/false);
            }
            if (memberName == "__fetch_builtin_z") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::ThreadIdOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "z"),
                  mlirType), /*isReference*/false);
            }
          }
          if (sr->getDecl()->getName() == "gridDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::GridDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "x"),
                  mlirType), /*isReference*/false);
            }
            if (memberName == "__fetch_builtin_y") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::GridDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "y"),
                  mlirType), /*isReference*/false);
            }
            if (memberName == "__fetch_builtin_z") {
              return ValueWithOffsets(builder.create<mlir::IndexCastOp>(
                  loc,
                  builder.create<mlir::gpu::GridDimOp>(
                      loc, mlir::IndexType::get(builder.getContext()), "z"),
                  mlirType), /*isReference*/false);
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
          args.push_back(Visit(a).getValue(builder));
        }
        return ValueWithOffsets(builder.create<mlir::Log2Op>(loc, args[0]), /*isReference*/false);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "sqrtf" ||
          sr->getDecl()->getName() == "sqrt") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return ValueWithOffsets(builder.create<mlir::SqrtOp>(loc, args[0]), /*isReference*/false);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "expf" ||
          sr->getDecl()->getName() == "exp") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return ValueWithOffsets(builder.create<mlir::ExpOp>(loc, args[0]), /*isReference*/false);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "sin") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return ValueWithOffsets(builder.create<mlir::SinOp>(loc, args[0]), /*isReference*/false);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "cos") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return ValueWithOffsets(builder.create<mlir::CosOp>(loc, args[0]), /*isReference*/false);
      }
    }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getName() == "atomicAdd") {
        std::vector<ValueWithOffsets> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a));
        }
        assert(args[0].isReference);
        auto a1 = args[1].getValue(builder);
        if (a1.getType().isa<mlir::IntegerType>())
          return ValueWithOffsets(builder.create<mlir::AtomicRMWOp>(
              loc, a1.getType(), AtomicRMWKind::addi,
              a1, args[0].val, std::vector<mlir::Value>({getConstantIndex(0)})), /*isReference*/false);
        else
          return ValueWithOffsets(builder.create<mlir::AtomicRMWOp>(
              loc, a1.getType(), AtomicRMWKind::addf,
              a1, args[0].val, std::vector<mlir::Value>({getConstantIndex(0)})), /*isReference*/false);
      }
    }

  auto getLLVM = [&](Expr* E) -> mlir::Value {
    if (auto IC1 = dyn_cast<ImplicitCastExpr>(E)) {
      if (auto IC2 = dyn_cast<ImplicitCastExpr>(IC1->getSubExpr())) {
        if (auto slit =
                dyn_cast<clang::StringLiteral>(IC2->getSubExpr())) {
          return Glob.GetOrCreateGlobalLLVMString(
              loc, builder, slit->getString());
        }
      }
      if (auto slit = dyn_cast<clang::StringLiteral>(IC1->getSubExpr())) {
        return Glob.GetOrCreateGlobalLLVMString(
            loc, builder, slit->getString());
      }
    }
    auto V = Visit(E).getValue(builder);
    auto mlirType = getMLIRType(E->getType());
    mlir::Type llvmType =
        mlir::LLVM::TypeFromLLVMIRTranslator(*mlirType.getContext())
            .translateType(getLLVMType(E->getType()));
    if (V.getType() == llvmType) return V;
    if (V.getType().isa<mlir::LLVM::LLVMType>()) {
      E->dump();
      llvm::errs() << mlirType << "\n";
      llvm::errs() << llvmType << "\n";
      llvm::errs() << V << "\n";
    }
    assert(!V.getType().isa<mlir::LLVM::LLVMType>());
    return builder.create<mlir::LLVM::DialectCastOp>(loc, llvmType, V);
  };

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
          args.push_back(Visit(a).getValue(builder));
        }
        auto arg0 =
            builder.create<mlir::LLVM::DialectCastOp>(loc, llvmType, args[0]);
        auto arg1 =
            builder.create<mlir::LLVM::DialectCastOp>(loc, llvmType, args[1]);
        return ValueWithOffsets(builder.create<mlir::LLVM::DialectCastOp>(
            loc, mlirType,
            builder.create<mlir::LLVM::PowOp>(loc, llvmType, arg0, arg1)), /*isReference*/false);
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

          mlir::Value val = Visit(a).getValue(builder);
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
        if (sr->getDecl()->getName() == "free") {

          std::vector<mlir::Value> args;
          for (auto a : expr->arguments()) {
            args.push_back(Visit(a).getValue(builder));
          }

          builder.create<mlir::DeallocOp>(loc, args[0]);
          return nullptr;
        }
      }

  std::set<std::string> funcs = {"strcmp", "open", "fopen", "close", "fclose", "atoi", "fscanf"};
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (funcs.count(sr->getDecl()->getName().str())) {
        if (sr->getDecl()->getName().str() == "fscanf") {
          llvm::errs() << "warning fscanf uses implicit case of memref to pointer which may use the wrong location\n";
        }
        auto tocall = EmitCallee(expr->getCallee());
        auto strcmpF = Glob.GetOrCreateLLVMFunction(tocall);

        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(getLLVM(a));
        }
        auto called = builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args);

        return ValueWithOffsets(builder.create<mlir::LLVM::DialectCastOp>(
                loc, getMLIRType(expr->getType()),
                called.getResult(0)), /*isReference*/false);
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
            tostore = Visit(a).getValue(builder);
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
          mlir::Value val = Visit(a).getValue(builder);
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
        auto ret = co;

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

        return ValueWithOffsets(ret, /*isReference*/false);
      }
    }

  auto tocall = EmitDirectCallee(EmitCallee(expr->getCallee()));
  std::vector<mlir::Value> args;
  auto fnType = tocall.getType();
  size_t i = 0;
  for (auto a : expr->arguments()) {
    mlir::Value val = Visit(a).getValue(builder);
    if (i >= fnType.getInputs().size()) {
      expr->dump();
      tocall.dump();
      fnType.dump();
      assert(0 && "too many arguments in calls");
    }
    /*
    if (val.getType() != fnType.getInput(i)) {
      if (auto MR1 = val.getType().dyn_cast<MemRefType>()) {
        if (auto MR2 = fnType.getInput(i).dyn_cast<MemRefType>()) {
          val = builder.create<mlir::MemRefCastOp>(loc, val, MR2);
        }
      }
    }
    */
    args.push_back(val);
    i++;
  }
  auto op = builder.create<mlir::CallOp>(loc, tocall, args);
  if (op.getNumResults())
    return ValueWithOffsets(op.getResult(0), /*isReference*/false);
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
  return nullptr;
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
  auto ty = getMLIRType(U->getType());

  switch (U->getOpcode()) {
  case clang::UnaryOperator::Opcode::UO_Extension: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_LNot: {
    mlir::Value val = sub.getValue(builder);
    auto ty = val.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      ty = builder.getIntegerType(1);
      val = builder.create<mlir::TruncateIOp>(loc, val, ty);
    }
    auto c1 = builder.create<mlir::ConstantOp>(
        loc, ty, builder.getIntegerAttr(ty, 1));
    return ValueWithOffsets(builder.create<mlir::XOrOp>(loc, val, c1), /*isReference*/false);
  }
  case clang::UnaryOperator::Opcode::UO_Deref: {
    auto dref = sub.dereference(builder);
    assert(dref.isReference);

    auto mt = dref.val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    if (shape.size() > 1) {
      shape.erase(shape.begin());
    } else {
      shape[0] = -1;
    }
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              mt.getAffineMaps(), mt.getMemorySpace());
    auto post = builder.create<SubIndexOp>(loc, mt0, dref.val, getConstantIndex(0));
    return ValueWithOffsets(post, /*isReference*/true);
  }
  case clang::UnaryOperator::Opcode::UO_AddrOf: {
    assert(sub.isReference);
    auto mt = sub.val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    shape[0] = -1;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              mt.getAffineMaps(), mt.getMemorySpace());

    return ValueWithOffsets(builder.create<MemRefCastOp>(loc, sub.val, mt0), /*isReference*/false);
  }
  case clang::UnaryOperator::Opcode::UO_Minus: {
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::NegFOp>(loc, sub.getValue(builder)), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::SubIOp>(
          loc,
          builder.create<mlir::ConstantIntOp>(loc, 0,
                                              ty.cast<mlir::IntegerType>()),
          sub.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::UnaryOperator::Opcode::UO_PreInc:
  case clang::UnaryOperator::Opcode::UO_PostInc: {
    assert(sub.isReference);
    auto prev = sub.getValue(builder);
    
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
    builder.create<mlir::StoreOp>(loc, next, sub.val, std::vector<mlir::Value>({getConstantIndex(0)}));
    return ValueWithOffsets((U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? prev
                                                                     : next, /*isReference*/false);
  }
  case clang::UnaryOperator::Opcode::UO_PreDec:
  case clang::UnaryOperator::Opcode::UO_PostDec: {
    assert(sub.isReference);
    auto prev = sub.getValue(builder);

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
    builder.create<mlir::StoreOp>(loc, next, sub.val, std::vector<mlir::Value>({getConstantIndex(0)}));
    return ValueWithOffsets((U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? prev
                                                                     : next, /*isReference*/false);
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
        loc, ty, builder.getIntegerAttr(ty, value)), /*isReference*/false);
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
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, lhs.getValue(builder),
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.thenRegion().back());

    auto rhs = Visit(BO->getRHS()).getValue(builder);
    assert(rhs != nullptr);
    mlir::Value truearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.elseRegion().back());
    mlir::Value falsearray[] = {builder.create<mlir::ConstantOp>(
        loc, types[0], builder.getIntegerAttr(types[0], 0))};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);
    return ValueWithOffsets(ifOp.getResult(0), /*isReference*/false);
  }
  case clang::BinaryOperator::Opcode::BO_LOr: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, lhs.getValue(builder),
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.thenRegion().back());

    mlir::Value truearray[] = {builder.create<mlir::ConstantOp>(
        loc, types[0], builder.getIntegerAttr(types[0], 1))};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.elseRegion().back());
    auto rhs = Visit(BO->getRHS()).getValue(builder);
    assert(rhs != nullptr);
    mlir::Value falsearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);

    return ValueWithOffsets(ifOp.getResult(0), /*isReference*/false);
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
      return ValueWithOffsets(builder.create<mlir::SignedShiftRightOp>(
          loc, lhs.getValue(builder), rhs.getValue(builder)), /*isReference*/false);
    else
      return ValueWithOffsets(builder.create<mlir::UnsignedShiftRightOp>(
          loc, lhs.getValue(builder), rhs.getValue(builder)), /*isReference*/false);
  }
  case clang::BinaryOperator::Opcode::BO_Shl: {
    return ValueWithOffsets(builder.create<mlir::ShiftLeftOp>(loc, lhs.getValue(builder),
                                                          rhs.getValue(builder)), /*isReference*/false);
  }
  case clang::BinaryOperator::Opcode::BO_And: {
    return ValueWithOffsets(builder.create<mlir::AndOp>(loc, lhs.getValue(builder),
                                                    rhs.getValue(builder)), /*isReference*/false);
  }
  case clang::BinaryOperator::Opcode::BO_Or: {
    // TODO short circuit
    return ValueWithOffsets(builder.create<mlir::OrOp>(loc, lhs.getValue(builder),
                                                   rhs.getValue(builder)), /*isReference*/false);
  }
  case clang::BinaryOperator::Opcode::BO_GT: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::UGT, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::sgt : mlir::CmpIPredicate::ugt,
          lhs_v, rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_GE: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::UGE, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::sge : mlir::CmpIPredicate::uge,
          lhs_v, rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_LT: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::ULT, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::slt : mlir::CmpIPredicate::ult,
          lhs_v, rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_LE: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::ULE, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::CmpIOp>(
          loc, signedType ? mlir::CmpIPredicate::sle : mlir::CmpIPredicate::ule,
          lhs_v, rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_EQ: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::UEQ, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::CmpIOp>(
          loc, mlir::CmpIPredicate::eq, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_NE: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::CmpFOp>(
          loc, mlir::CmpFPredicate::UNE, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::CmpIOp>(
          loc, mlir::CmpIPredicate::ne, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Mul: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::MulFOp>(loc, lhs_v,
                                                       rhs.getValue(builder)), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::MulIOp>(loc, lhs_v,
                                                       rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Div: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::DivFOp>(loc, lhs_v,
                                                       rhs.getValue(builder)), /*isReference*/false);;
    } else {
      if (signedType)
        return ValueWithOffsets(builder.create<mlir::SignedDivIOp>(
            loc, lhs_v, rhs.getValue(builder)), /*isReference*/false);
      else
        return ValueWithOffsets(builder.create<mlir::UnsignedDivIOp>(
            loc, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Rem: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::RemFOp>(loc, lhs_v,
                                                       rhs.getValue(builder)), /*isReference*/false);
    } else {
      if (signedType)
        return ValueWithOffsets(builder.create<mlir::SignedRemIOp>(
            loc, lhs_v, rhs.getValue(builder)), /*isReference*/false);
      else
        return ValueWithOffsets(builder.create<mlir::UnsignedRemIOp>(
            loc, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Add: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::AddFOp>(loc, lhs_v,
                                                       rhs.getValue(builder)), /*isReference*/false);
    } else if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 =
          mlir::MemRefType::get(shape, mt.getElementType(),
                                mt.getAffineMaps(), mt.getMemorySpace());
      return ValueWithOffsets(builder.create<SubIndexOp>(loc, mt0, lhs_v, rhs.getValue(builder)), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::AddIOp>(loc, lhs_v,
                                                       rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Sub: {
    auto lhs_v = lhs.getValue(builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueWithOffsets(builder.create<mlir::SubFOp>(loc, lhs_v,
                                                       rhs.getValue(builder)), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::SubIOp>(loc, lhs_v,
                                                       rhs.getValue(builder)), /*isReference*/false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Assign: {
    assert(lhs.isReference);
    mlir::Value tostore = rhs.getValue(builder);
    builder.create<mlir::StoreOp>(loc, tostore, lhs.val, std::vector<mlir::Value>({getConstantIndex(0)}));
    return lhs;
  }

  case clang::BinaryOperator::Opcode::BO_Comma: {
    return rhs;
  }

  case clang::BinaryOperator::Opcode::BO_AddAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<mlir::AddFOp>(loc, prev,
                                            rhs.getValue(builder));
    } else {
      result = builder.create<mlir::AddIOp>(loc, prev,
                                            rhs.getValue(builder));
    }
    builder.create<mlir::StoreOp>(loc, result, lhs.val, std::vector<mlir::Value>({getConstantIndex(0)}));
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_SubAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<mlir::SubFOp>(loc, prev,
                                            rhs.getValue(builder));
    } else {
      result = builder.create<mlir::SubIOp>(loc, prev,
                                            rhs.getValue(builder));
    }
    builder.create<mlir::StoreOp>(loc, result, lhs.val, std::vector<mlir::Value>({getConstantIndex(0)}));
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_MulAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<mlir::MulFOp>(loc, prev,
                                            rhs.getValue(builder));
    } else {
      result = builder.create<mlir::MulIOp>(loc, prev,
                                            rhs.getValue(builder));
    }
    builder.create<mlir::StoreOp>(loc, result, lhs.val, std::vector<mlir::Value>({getConstantIndex(0)}));
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_DivAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;
    if (prev.getType().isa<mlir::FloatType>()) {
      result = builder.create<mlir::DivFOp>(loc, prev,
                                            rhs.getValue(builder));
    } else {
      if (signedType)
        return ValueWithOffsets(builder.create<mlir::SignedDivIOp>(
            loc, prev, rhs.getValue(builder)), /*isReference*/true);
      else
        return ValueWithOffsets(builder.create<mlir::UnsignedDivIOp>(
            loc, prev, rhs.getValue(builder)), /*isReference*/true);
    }
    builder.create<mlir::StoreOp>(loc, result, lhs.val, std::vector<mlir::Value>({getConstantIndex(0)}));
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_ShrAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(builder);

    mlir::Value result;

    if (signedType)
      return ValueWithOffsets(builder.create<mlir::SignedShiftRightOp>(
          loc, prev, rhs.getValue(builder)), /*isReference*/true);
    else
      return ValueWithOffsets(builder.create<mlir::UnsignedShiftRightOp>(
          loc, prev, rhs.getValue(builder)), /*isReference*/true);
    builder.create<mlir::StoreOp>(loc, result, lhs.val, std::vector<mlir::Value>({getConstantIndex(0)}));
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
    assert(0 && "cleanup not handled");
  }
  return ret;
}

ValueWithOffsets MLIRScanner::VisitDeclRefExpr(DeclRefExpr *E) {
  auto val = getValue(E->getDecl()->getName().str());
  //E->dump();
  //llvm::errs() << "DeclRefExpr: " << val.val << " - isReference: " << val.isReference << "\n";
  return val;
}

ValueWithOffsets MLIRScanner::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  if (!E->getSourceExpr()) {
    E->dump();
    assert(E->getSourceExpr());
  }
  for (auto c : E->children()) {
    //c->dump();
  }
  auto res = Visit(E->getSourceExpr());
  if (!res.val) {
    E->dump();
    E->getSourceExpr()->dump();
    assert(res.val);
  }
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
  assert(base.isReference);
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

  auto mt = base.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() > 1) {
    shape.erase(shape.begin());
  } else {
    shape[0] = -1;
  }
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            mt.getAffineMaps(), mt.getMemorySpace());
  return ValueWithOffsets(builder.create<SubIndexOp>(loc, mt0, base.val, getConstantIndex(layout.getLLVMFieldNo(field))), /*isReference*/true);
}

ValueWithOffsets MLIRScanner::VisitCastExpr(CastExpr *E) {
  switch (E->getCastKind()) {

  case clang::CastKind::CK_NullToPointer: {
    auto llvmType =
        Glob.typeTranslator.translateType(getLLVMType(E->getType()));
    return ValueWithOffsets(builder.create<mlir::LLVM::NullOp>(loc, llvmType), /*isReference*/ false);
  }

  case clang::CastKind::CK_BitCast: {

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getName() == "polybench_alloc_data") {
            auto mt = getMLIRType(E->getType()).cast<mlir::MemRefType>();

            auto shape = std::vector<int64_t>(mt.getShape());
            shape.erase(shape.begin());
            auto mt0 =
                mlir::MemRefType::get(shape, mt.getElementType(),
                                      mt.getAffineMaps(), mt.getMemorySpace());

            auto alloc = builder.create<mlir::AllocOp>(loc, mt0);
            auto alloc2 = builder.create<mlir::SubIndexOp>(
                loc, mt, alloc, getConstantIndex(-1));

            //mlir::Value zeroIndex = getConstantIndex(0);
            //builder.create<mlir::StoreOp>(loc, alloc, alloc2, zeroIndex);
            return ValueWithOffsets(alloc2, /*isReference*/false);
          }
        }

    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    auto ut = scalar.getType().cast<mlir::MemRefType>();
    auto mt = getMLIRType(E->getType()).cast<mlir::MemRefType>();

    auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                    ut.getAffineMaps(), ut.getMemorySpace());
    return ValueWithOffsets(builder.create<mlir::MemRefCastOp>(loc, scalar, ty), /*isReference*/false);
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
          return ValueWithOffsets(builder.create<mlir::LLVM::DialectCastOp>(
              loc, mlirType,
              builder.create<mlir::NVVM::WarpSizeOp>(loc, llvmType)), /*isReference*/true);
        }
      }
    }
    auto prev = Visit(E->getSubExpr());

    //E->dump();
    //llvm::errs() << prev.val << " - " << prev.isReference << "\n";
    //return prev;
    auto lres  = prev.getValue(builder);
    //llvm::errs() << " - lres: " << lres <<  " mt: " << getMLIRType(E->getType()) << " " << *Glob.CGM.getTypes().ConvertType(E->getType()) << "\n";
    assert(prev.isReference);
    //assert(lres.getType() == getMLIRType(E->getType()));
    return ValueWithOffsets(lres, /*isReference*/false);

    if (prev.val.getType().isa<mlir::LLVM::LLVMType>()) {
      return ValueWithOffsets(prev.val, /*isReference*/true);
    }
    auto c0 = builder.create<mlir::ConstantIndexOp>(loc, 0);
    if (!prev.val.getType().isa<mlir::MemRefType>()) {
      builder.getInsertionBlock()->dump();
      function.dump();
      E->dump();
      llvm::errs() << prev.val << "\n";
    }
    auto mt = prev.val.getType().cast<mlir::MemRefType>();

    auto shape = std::vector<int64_t>(mt.getShape());
    if (shape.size() == 1)
      return ValueWithOffsets(builder.create<LoadOp>(loc, prev.val, std::vector<mlir::Value>({c0})), /*isReference*/false);

    shape.erase(shape.begin());
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              mt.getAffineMaps(), mt.getMemorySpace());
    return ValueWithOffsets(builder.create<SubIndexOp>(loc, mt0, prev.val, c0), /*isReference*/true);
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
      return ValueWithOffsets(builder.create<mlir::SIToFPOp>(
          loc, scalar, ty), /*isReference*/false);
    else
      return ValueWithOffsets(builder.create<mlir::UIToFPOp>(
          loc, scalar, ty), /*isReference*/false);
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
      return ValueWithOffsets(builder.create<mlir::FPToSIOp>(
          loc, scalar, ty), /*isReference*/false);
    else
      return ValueWithOffsets(builder.create<mlir::FPToUIOp>(
          loc, scalar, ty), /*isReference*/false);
  }
  case clang::CastKind::CK_IntegralCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
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
      return ValueWithOffsets(scalar, /*isReference*/false);
    if (prevTy.getWidth() < postTy.getWidth()) {
      if (signedType) {
        return ValueWithOffsets(builder.create<mlir::SignExtendIOp>(
            loc, scalar, postTy), /*isReference*/false);
      } else {
        return ValueWithOffsets(builder.create<mlir::ZeroExtendIOp>(
            loc, scalar, postTy), /*isReference*/false);
      }
    } else {
      return ValueWithOffsets(builder.create<mlir::TruncateIOp>(
          loc, scalar, postTy), /*isReference*/false);
    }
  }
  case clang::CastKind::CK_FloatingCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    if (!scalar.getType().isa<mlir::FloatType>()) {
      E->dump();
      llvm::errs() << scalar << "\n";
    }
    auto prevTy = scalar.getType().cast<mlir::FloatType>();
    auto postTy = getMLIRType(E->getType()).cast<mlir::FloatType>();

    if (prevTy == postTy)
      return ValueWithOffsets(scalar, /*isReference*/false);
    if (prevTy.getWidth() < postTy.getWidth()) {
      return ValueWithOffsets(builder.create<mlir::FPExtOp>(
          loc, scalar, postTy), /*isReference*/false);
    } else {
      return ValueWithOffsets(builder.create<mlir::FPTruncOp>(
          loc, scalar, postTy), /*isReference*/false);
    }
  }
  case clang::CastKind::CK_ArrayToPointerDecay: {
    auto scalar = Visit(E->getSubExpr());
    if (!scalar.val) {
      E->dump();
    }
    //if (!scalar.isReference) {
    //}
    assert(scalar.isReference);

    auto mt = scalar.val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    //if (shape.size() > 1) {
    //  shape.erase(shape.begin());
    //} else {
      shape[0] = -1;
    //}
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              mt.getAffineMaps(), mt.getMemorySpace());

    auto post = builder.create<MemRefCastOp>(loc, mt0, scalar.val);
    return ValueWithOffsets(post, /*isReference*/false);

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
    return scalar;
  }
  case clang::CastKind::CK_NoOp: {
    return Visit(E->getSubExpr());
  }
  case clang::CastKind::CK_ToVoid: {
    Visit(E->getSubExpr());
    return nullptr;
  }
  case clang::CastKind::CK_PointerToBoolean: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    if (auto LT = scalar.getType().dyn_cast<mlir::LLVM::LLVMType>()) {
      if (LT.isPointerTy()) {
        auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
        auto ne = builder.create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, scalar, nullptr_llvm);
        auto mlirType = getMLIRType(E->getType());
        mlir::Value val = builder.create<mlir::LLVM::DialectCastOp>(loc, mlirType, ne);
        return ValueWithOffsets(val, /*isReference*/false);
      }
    }
    function.dump();
    llvm::errs() << scalar << "\n";
    E->dump();
    assert(0 && "unhandled ptrtobool cast");
  }
  default:
    E->dump();
    assert(0 && "unhandled cast");
  }
}

ValueWithOffsets MLIRScanner::VisitIfStmt(clang::IfStmt *stmt) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(stmt->getIfLoc());
  auto cond = Visit(stmt->getCond()).getValue(builder);
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
  auto cond = Visit(E->getCond()).getValue(builder);
  assert(cond != nullptr);

  mlir::Type types[] = {getMLIRType(E->getType())};
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                              /*hasElseRegion*/ true);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();
  builder.setInsertionPointToStart(&ifOp.thenRegion().back());

  auto truev = Visit(E->getTrueExpr()).getValue(builder);
  assert(truev != nullptr);
  mlir::Value truearray[] = {truev};
  builder.create<mlir::scf::YieldOp>(loc, truearray);

  builder.setInsertionPointToStart(&ifOp.elseRegion().back());
  auto falsev = Visit(E->getFalseExpr()).getValue(builder);
  assert(falsev != nullptr);
  mlir::Value falsearray[] = {falsev};
  builder.create<mlir::scf::YieldOp>(loc, falsearray);

  builder.setInsertionPoint(oldblock, oldpoint);

  types[0] = truev.getType();
  auto newIfOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                 /*hasElseRegion*/ true);
  newIfOp.thenRegion().takeBody(ifOp.thenRegion());
  newIfOp.elseRegion().takeBody(ifOp.elseRegion());
  ifOp.erase();

  return ValueWithOffsets(newIfOp.getResult(0), /*isReference*/false);
  // return ifOp;
}

ValueWithOffsets MLIRScanner::VisitCompoundStmt(clang::CompoundStmt *stmt) {
  for (auto a : stmt->children()) {
    IfScope scope(*this);
    Visit(a);
  }
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitBreakStmt(clang::BreakStmt *stmt) {
  IfScope scope(*this);
  assert(loops.size());
  assert(loops.back().keepRunning);
  assert(loops.back().noBreak);
  auto i1Ty = builder.getI1Type();
  auto vfalse = builder.create<mlir::ConstantOp>(
      builder.getUnknownLoc(), i1Ty, builder.getIntegerAttr(i1Ty, 0));
  builder.create<mlir::StoreOp>(loc, vfalse, loops.back().keepRunning);
  builder.create<mlir::StoreOp>(loc, vfalse, loops.back().noBreak);

  return nullptr;
}
ValueWithOffsets MLIRScanner::VisitContinueStmt(clang::ContinueStmt *stmt) {
  IfScope scope(*this);
  assert(loops.size());
  assert(loops.back().keepRunning);
  auto i1Ty = builder.getI1Type();
  auto vfalse = builder.create<mlir::ConstantOp>(
      builder.getUnknownLoc(), i1Ty, builder.getIntegerAttr(i1Ty, 0));
  builder.create<mlir::StoreOp>(loc, vfalse, loops.back().keepRunning);
  return nullptr;
}

ValueWithOffsets MLIRScanner::VisitReturnStmt(clang::ReturnStmt *stmt) {
  IfScope scope(*this);
  if (stmt->getRetValue()) {
    auto rv = Visit(stmt->getRetValue()).getValue(builder);
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
          typeTranslator.translateType(getLLVMType(parm->getType())));
    } else {
      types.push_back(getMLIRType(parm->getType()));
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

  std::function<void(Decl*)> handle = [&](Decl* D) {
    if (auto lsd = dyn_cast<clang::LinkageSpecDecl>(D)) {
      for(auto e : lsd->decls()) handle(e);
    }
    if (VarDecl *fd = dyn_cast<clang::VarDecl>(D)) {
      globalVariables[fd->getName().str()] = fd;
    }
    if (FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(D)) {
      if (fd->getIdentifier()) {
        globalFunctions[fd->getName().str()] = fd;
      }
    }
  };
  for (it = dg.begin(); it != dg.end(); ++it) {
    handle(*it);
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
    if (auto ST = dyn_cast<llvm::StructType>(pt->getElementType())) {
      if (ST->getName() == "struct._IO_FILE") {
        return typeTranslator.translateType(t);
      }
      bool notAllSame = false;
      for (size_t i = 1; i < ST->getNumElements(); i++) {
        if (ST->getTypeAtIndex(i) != ST->getTypeAtIndex(0U)) {
          notAllSame = true;
          break;
        }
      }
      if (!notAllSame) {
        return mlir::MemRefType::get({-1, ST->getNumElements()},
                                    getMLIRType(ST->getTypeAtIndex(0U)), {}, pt->getAddressSpace());
      }
    }
    if (auto AT = dyn_cast<llvm::ArrayType>(pt->getElementType())) {
      auto under = getMLIRType(AT);
      auto mt = under.cast<mlir::MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2.insert(shape2.begin(), -1);
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   mt.getAffineMaps(), mt.getMemorySpace());
    }
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
    if (ST->getName() == "struct._IO_FILE") {
      return typeTranslator.translateType(t);
    }
    SmallVector<mlir::Type> types;
    for (size_t i = 0; i < ST->getNumElements(); i++) {
      types.push_back(getMLIRType(ST->getTypeAtIndex(i)));
    }
    //return mlir::LLVM::StructType::get(ST->getName(), types);
  }
  // if (auto pt = dyn_cast<clang::RecordType>(t)) {
  //    llvm::errs() << " thing: " << pt->getName() << "\n";
  //}
  llvm::errs() << *t << "\n";
  assert(0 && "unknown type to convert");
  return nullptr;
}

void MLIRScanner::pushLoopIf() {
  if (loops.size() && loops.back().keepRunning) {
    auto ifOp = builder.create<scf::IfOp>(
        loc, builder.create<LoadOp>(loc, loops.back().keepRunning),
        /*hasElse*/ false);
    prevBlock.push_back(builder.getInsertionBlock());
    prevIterator.push_back(builder.getInsertionPoint());
    ifOp.thenRegion().back().clear();
    builder.setInsertionPointToStart(&ifOp.thenRegion().back());
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

std::string GetExecutablePath(const char *Argv0, bool CanonicalPrefixes) {
  if (!CanonicalPrefixes) {
    SmallString<128> ExecutablePath(Argv0);
    // Do a PATH lookup if Argv0 isn't a valid path.
    if (!llvm::sys::fs::exists(ExecutablePath))
      if (llvm::ErrorOr<std::string> P =
              llvm::sys::findProgramByName(ExecutablePath))
        ExecutablePath = *P;
    return std::string(ExecutablePath.str());
  }

  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *P = (void*) (intptr_t) GetExecutablePath;
  return llvm::sys::fs::getMainExecutable(Argv0, P);
}

#include "clang/Frontend/TextDiagnosticBuffer.h"
static bool parseMLIR(const char* Argv0, std::vector<std::string> filenames, std::string fn,
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
  const char *binary = Argv0; //CudaLower ? "clang++" : "clang";
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
  if (FOpenMP)
    Argv.push_back("-fopenmp");
  if (Standard != "") {
    auto a = "-std=" + Standard;
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

    void *GetExecutablePathVP = (void *)(intptr_t)GetExecutablePath;
    // Infer the builtin include path if unspecified.
    if (Clang->getHeaderSearchOpts().UseBuiltinIncludes && Clang->getHeaderSearchOpts().ResourceDir.size() == 0)
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
      }
    }
    DL = Clang->getTarget().getDataLayout();
    triple = Clang->getTarget().getTriple();
  }
  return true;
}

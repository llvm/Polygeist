//===- clang-mlir.h - Emit MLIR IRs by walking clang AST---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_MLIR_H
#define CLANG_MLIR_H

#include "clang/AST/StmtVisitor.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>

#include "AffineUtils.h"
#include "ValueCategory.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "polygeist/Ops.h"
#include "pragmaHandler.h"
#include "llvm/IR/DerivedTypes.h"

#include "clang/../../lib/CodeGen/CGRecordLayout.h"
#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "clang/AST/Mangle.h"

using namespace clang;
using namespace mlir;

struct LoopContext {
  mlir::Value keepRunning;
  mlir::Value noBreak;
};

struct MLIRASTConsumer : public ASTConsumer {
  std::set<std::string> &emitIfFound;
  std::set<std::string> &done;
  std::map<std::string, mlir::LLVM::GlobalOp> &llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &globals;
  std::map<std::string, mlir::FuncOp> &functions;
  std::map<std::string, mlir::LLVM::GlobalOp> &llvmGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> &llvmFunctions;
  Preprocessor &PP;
  ASTContext &astContext;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  clang::SourceManager &SM;
  LLVMContext lcontext;
  llvm::Module llvmMod;
  CodeGenOptions codegenops;
  CodeGen::CodeGenModule CGM;
  bool error;
  ScopLocList scopLocList;
  LowerToInfo LTInfo;

  /// The stateful type translator (contains named structs).
  LLVM::TypeFromLLVMIRTranslator typeTranslator;
  LLVM::TypeToLLVMIRTranslator reverseTypeTranslator;

  MLIRASTConsumer(
      std::set<std::string> &emitIfFound, std::set<std::string> &done,
      std::map<std::string, mlir::LLVM::GlobalOp> &llvmStringGlobals,
      std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &globals,
      std::map<std::string, mlir::FuncOp> &functions,
      std::map<std::string, mlir::LLVM::GlobalOp> &llvmGlobals,
      std::map<std::string, mlir::LLVM::LLVMFuncOp> &llvmFunctions,
      Preprocessor &PP, ASTContext &astContext,
      mlir::OwningOpRef<mlir::ModuleOp> &module, clang::SourceManager &SM)
      : emitIfFound(emitIfFound), done(done),
        llvmStringGlobals(llvmStringGlobals), globals(globals),
        functions(functions), llvmGlobals(llvmGlobals),
        llvmFunctions(llvmFunctions), PP(PP), astContext(astContext),
        module(module), SM(SM), lcontext(), llvmMod("tmp", lcontext),
        codegenops(),
        CGM(astContext, PP.getHeaderSearchInfo().getHeaderSearchOpts(),
            PP.getPreprocessorOpts(), codegenops, llvmMod, PP.getDiagnostics()),
        error(false), typeTranslator(*module->getContext()),
        reverseTypeTranslator(lcontext) {
    addPragmaScopHandlers(PP, scopLocList);
    addPragmaEndScopHandlers(PP, scopLocList);
    addPragmaLowerToHandlers(PP, LTInfo);
  }

  ~MLIRASTConsumer() {}

  mlir::FuncOp GetOrCreateMLIRFunction(const FunctionDecl *FD);

  mlir::LLVM::LLVMFuncOp GetOrCreateLLVMFunction(const FunctionDecl *FD);

  mlir::LLVM::GlobalOp GetOrCreateLLVMGlobal(const ValueDecl *VD);

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  mlir::Value GetOrCreateGlobalLLVMString(mlir::Location loc,
                                          mlir::OpBuilder &builder,
                                          StringRef value);

  std::pair<mlir::memref::GlobalOp, bool> GetOrCreateGlobal(const ValueDecl *VD,
                                                            std::string prefix);

  std::deque<const FunctionDecl *> functionsToEmit;

  void run();

  bool HandleTopLevelDecl(DeclGroupRef dg) override;

  void HandleDeclContext(DeclContext *DC);

  mlir::Type getMLIRType(clang::QualType t, bool *implicitRef = nullptr,
                         bool allowMerge = true);

  llvm::Type *getLLVMType(clang::QualType t);

  mlir::Type getMLIRType(llvm::Type *t);

  mlir::Location getMLIRLocation(clang::SourceLocation loc);
};

struct MLIRScanner : public StmtVisitor<MLIRScanner, ValueCategory> {
private:
  MLIRASTConsumer &Glob;
  mlir::FuncOp function;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::Block *entryBlock;
  std::vector<LoopContext> loops;
  mlir::Block *allocationScope;

  // ValueCategory getValue(std::string name);

  std::map<const void *, std::vector<mlir::LLVM::AllocaOp>> bufs;
  mlir::LLVM::AllocaOp allocateBuffer(size_t i, mlir::LLVM::LLVMPointerType t) {
    auto &vec = bufs[t.getAsOpaquePointer()];
    if (i < vec.size())
      return vec[i];

    mlir::OpBuilder subbuilder(builder.getContext());
    subbuilder.setInsertionPointToStart(allocationScope);

    auto indexType = subbuilder.getIntegerType(64);
    auto one = subbuilder.create<mlir::ConstantOp>(
        loc, indexType,
        subbuilder.getIntegerAttr(subbuilder.getIntegerType(64), 1));
    auto rs = subbuilder.create<mlir::LLVM::AllocaOp>(loc, t, one, 0);
    vec.push_back(rs);
    return rs;
  }

  mlir::Location getMLIRLocation(clang::SourceLocation loc);

  llvm::Type *getLLVMType(clang::QualType t);
  mlir::Type getMLIRType(clang::QualType t);

  size_t getTypeSize(clang::QualType t);

  mlir::Value createAllocOp(mlir::Type t, VarDecl *name, uint64_t memspace,
                            bool isArray, bool LLVMABI);

  const clang::FunctionDecl *EmitCallee(const Expr *E);

  mlir::FuncOp EmitDirectCallee(GlobalDecl GD);

  std::map<int, mlir::Value> constants;
  mlir::Value getConstantIndex(int x);

  mlir::Value castToIndex(mlir::Location loc, mlir::Value val);

  bool isTrivialAffineLoop(clang::ForStmt *fors,
                           mlirclang::AffineLoopDescriptor &descr);

  bool getUpperBound(clang::ForStmt *fors,
                     mlirclang::AffineLoopDescriptor &descr);

  bool getLowerBound(clang::ForStmt *fors,
                     mlirclang::AffineLoopDescriptor &descr);

  bool getConstantStep(clang::ForStmt *fors,
                       mlirclang::AffineLoopDescriptor &descr);

  void buildAffineLoop(clang::ForStmt *fors, mlir::Location loc,
                       const mlirclang::AffineLoopDescriptor &descr);

  void buildAffineLoopImpl(clang::ForStmt *fors, mlir::Location loc,
                           mlir::Value lb, mlir::Value ub,
                           const mlirclang::AffineLoopDescriptor &descr);
  std::vector<Block *> prevBlock;
  std::vector<Block::iterator> prevIterator;

public:
  void pushLoopIf();
  void popLoopIf();

public:
  const FunctionDecl *EmittingFunctionDecl;
  std::map<const VarDecl *, ValueCategory> params;
  llvm::DenseMap<const VarDecl *, FieldDecl *> Captures;
  llvm::DenseMap<const VarDecl *, LambdaCaptureKind> CaptureKinds;
  FieldDecl *ThisCapture;
  std::vector<mlir::Value> arrayinit;
  ValueCategory ThisVal;
  mlir::Value returnVal;
  LowerToInfo &LTInfo;

  MLIRScanner(MLIRASTConsumer &Glob, mlir::FuncOp function,
              const FunctionDecl *fd, mlir::OwningOpRef<mlir::ModuleOp> &module,
              LowerToInfo &LTInfo)
      : Glob(Glob), function(function), module(module),
        builder(module->getContext()), loc(builder.getUnknownLoc()),
        EmittingFunctionDecl(fd), ThisCapture(nullptr), LTInfo(LTInfo) {

    if (ShowAST) {
      llvm::errs() << "Emitting fn: " << function.getName() << "\n";
      llvm::errs() << *fd << "\n";
    }

    allocationScope = entryBlock = function.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

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
      auto LLTy = getLLVMType(parm->getType());
      while (auto ST = dyn_cast<llvm::StructType>(LLTy)) {
        if (ST->getNumElements() == 1)
          LLTy = ST->getTypeAtIndex(0U);
        else
          break;
      }
      bool LLVMABI = false;

      if (Glob.getMLIRType(
                  Glob.CGM.getContext().getPointerType(parm->getType()))
              .isa<mlir::LLVM::LLVMPointerType>())
        LLVMABI = true;

      if (!LLVMABI) {
        Glob.getMLIRType(parm->getType(), &isArray);
      }
      if (!isArray && (isa<clang::RValueReferenceType>(parm->getType()) ||
                       isa<clang::LValueReferenceType>(parm->getType())))
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

      for (auto expr : CC->inits()) {
        if (ShowAST) {
          if (expr->getMember())
            expr->getMember()->dump();
          if (expr->getInit())
            expr->getInit()->dump();
        }
        assert(ThisVal.val);
        FieldDecl *field = expr->getMember();
        if (auto AILE = dyn_cast<ArrayInitLoopExpr>(expr->getInit())) {
          VisitArrayInitLoop(AILE, CommonFieldLookup(CC->getThisObjectType(),
                                                     field, ThisVal.val));
          continue;
        }
        auto initexpr = Visit(expr->getInit());
        if (!initexpr.val) {
          expr->getInit()->dump();
          assert(initexpr.val);
        }
        bool isArray = false;
        Glob.getMLIRType(expr->getInit()->getType(), &isArray);

        auto cfl =
            CommonFieldLookup(CC->getThisObjectType(), field, ThisVal.val);
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
    auto truev = builder.create<mlir::ConstantIntOp>(loc, true, 1);
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
            loc,
            builder.create<mlir::LLVM::UndefOp>(loc, type.getElementType()),
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

  ValueCategory VisitDeclStmt(clang::DeclStmt *decl);

  ValueCategory VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *decl);

  ValueCategory VisitConstantExpr(clang::ConstantExpr *expr);

  ValueCategory VisitIntegerLiteral(clang::IntegerLiteral *expr);

  ValueCategory VisitCharacterLiteral(clang::CharacterLiteral *expr);

  ValueCategory VisitFloatingLiteral(clang::FloatingLiteral *expr);

  ValueCategory VisitImaginaryLiteral(clang::ImaginaryLiteral *expr);

  ValueCategory VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr);
  ValueCategory VisitCXXTypeidExpr(clang::CXXTypeidExpr *expr);

  ValueCategory VisitStringLiteral(clang::StringLiteral *expr);

  ValueCategory VisitParenExpr(clang::ParenExpr *expr);

  ValueCategory VisitVarDecl(clang::VarDecl *decl);

  ValueCategory VisitForStmt(clang::ForStmt *fors);

  ValueCategory
  VisitOMPParallelForDirective(clang::OMPParallelForDirective *fors);

  ValueCategory VisitWhileStmt(clang::WhileStmt *fors);

  ValueCategory VisitDoStmt(clang::DoStmt *fors);

  ValueCategory VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr);

  ValueCategory VisitCallExpr(clang::CallExpr *expr);

  std::pair<ValueCategory, bool> EmitGPUCallExpr(clang::CallExpr *expr);

  std::pair<ValueCategory, bool> EmitBuiltinOps(clang::CallExpr *expr);

  ValueCategory VisitCXXConstructExpr(clang::CXXConstructExpr *expr);

  ValueCategory VisitConstructCommon(clang::CXXConstructExpr *expr,
                                     VarDecl *name, unsigned space,
                                     mlir::Value mem = nullptr);

  ValueCategory VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *expr);

  ValueCategory VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr);

  ValueCategory VisitUnaryOperator(clang::UnaryOperator *U);

  ValueCategory
  VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *expr);

  ValueCategory
  VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *Uop);

  ValueCategory VisitBinaryOperator(clang::BinaryOperator *BO);

  ValueCategory VisitAttributedStmt(clang::AttributedStmt *AS);

  ValueCategory VisitExprWithCleanups(clang::ExprWithCleanups *E);

  ValueCategory VisitDeclRefExpr(clang::DeclRefExpr *E);

  ValueCategory VisitOpaqueValueExpr(clang::OpaqueValueExpr *E);

  ValueCategory VisitMemberExpr(clang::MemberExpr *ME);

  ValueCategory VisitCastExpr(clang::CastExpr *E);

  ValueCategory VisitIfStmt(clang::IfStmt *stmt);

  ValueCategory VisitSwitchStmt(clang::SwitchStmt *stmt);

  ValueCategory VisitConditionalOperator(clang::ConditionalOperator *E);

  ValueCategory VisitCompoundStmt(clang::CompoundStmt *stmt);

  ValueCategory VisitBreakStmt(clang::BreakStmt *stmt);

  ValueCategory VisitContinueStmt(clang::ContinueStmt *stmt);

  ValueCategory VisitReturnStmt(clang::ReturnStmt *stmt);

  ValueCategory VisitStmtExpr(clang::StmtExpr *stmt);

  ValueCategory VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr);

  ValueCategory
  VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *expr);

  ValueCategory VisitCXXNewExpr(clang::CXXNewExpr *expr);

  ValueCategory VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr);

  ValueCategory VisitCXXThisExpr(clang::CXXThisExpr *expr);

  ValueCategory VisitPredefinedExpr(clang::PredefinedExpr *expr);

  ValueCategory VisitLambdaExpr(clang::LambdaExpr *expr);

  ValueCategory VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr);

  ValueCategory VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr);

  ValueCategory VisitInitListExpr(clang::InitListExpr *expr);

  ValueCategory VisitArrayInitLoop(clang::ArrayInitLoopExpr *expr,
                                   ValueCategory tostore);

  ValueCategory VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *expr);

  ValueCategory CommonFieldLookup(clang::QualType OT, const FieldDecl *FD,
                                  mlir::Value val);

  ValueCategory CommonArrayLookup(ValueCategory val, mlir::Value idx);

  ValueCategory CommonArrayToPointer(ValueCategory val);
};

#endif

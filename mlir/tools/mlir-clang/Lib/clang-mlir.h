#ifndef CLANG_MLIR_H
#define CLANG_MLIR_H

#include "clang/AST/StmtVisitor.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Target/LLVMIR/TypeTranslation.h"

#include "../../../../clang/lib/CodeGen/CodeGenModule.h"
#include "clang/AST/Mangle.h"

using namespace std;
using namespace clang;
using namespace mlir;

struct LoopContext {
  mlir::Block *condB;
  mlir::Block *exitB;
};

struct ValueWithOffsets {
  mlir::Value val;
  std::vector<mlir::Value> offsets;
  ValueWithOffsets() = default;
  ValueWithOffsets(const ValueWithOffsets &r) = default;
  ValueWithOffsets(ValueWithOffsets &&r) = default;

  ValueWithOffsets &operator=(ValueWithOffsets &rhs) {
    val = rhs.val;
    offsets = rhs.offsets;
    return *this;
  }
  ValueWithOffsets &operator=(ValueWithOffsets &&rhs) {
    val = rhs.val;
    offsets = rhs.offsets;
    return *this;
  }

  ValueWithOffsets(mlir::Value val, std::vector<mlir::Value> offsets)
      : val(val), offsets(offsets) {
    if (auto MT = val.getType().dyn_cast<MemRefType>())
      assert(offsets.size() <= MT.getShape().size());
  }
  ValueWithOffsets(mlir::Value sval) : val(sval), offsets() {
    // if (val)
    //    assert(!val.getType().isa<mlir::MemRefType>());
  }
  ValueWithOffsets(std::nullptr_t sval) : val(sval), offsets() {
    // if (val)
    //    assert(!val.getType().isa<mlir::MemRefType>());
  }
  explicit operator mlir::Value() const {
    // This is needed because a pointer can be used as an lvalue but it still
    // has an offset
    if (offsets.size() != 0) {
      assert(offsets.size() == 1);
      if (auto op = offsets[0].getDefiningOp<ConstantIndexOp>()) {
        assert(op.getValue() == 0);
      } else {
        assert(0 && "unable to get value");
      }
    }
    return val;
  }
  mlir::Type getType() { return val.getType(); }
};

struct MLIRASTConsumer : public ASTConsumer {
  std::string fn;
  Preprocessor &PP;
  ASTContext &astContext;
  mlir::ModuleOp &module;
  clang::SourceManager &SM;
  MangleContext &MC;
  LLVMContext lcontext;
  llvm::Module llvmMod;
  CodeGenOptions codegenops;
  CodeGen::CodeGenModule CGM;
  bool error;

  MLIRASTConsumer(std::string fn, Preprocessor &PP, ASTContext &astContext,
                  mlir::ModuleOp &module, clang::SourceManager &SM)
      : fn(fn), PP(PP), astContext(astContext), module(module), SM(SM),
        MC(*astContext.createMangleContext()), lcontext(),
        llvmMod("tmp", lcontext), codegenops(),
        CGM(astContext, PP.getHeaderSearchInfo().getHeaderSearchOpts(),
            PP.getPreprocessorOpts(), codegenops, llvmMod, PP.getDiagnostics()),
        error(false) {}

  ~MLIRASTConsumer() {}

  mlir::FuncOp GetOrCreateMLIRFunction(const FunctionDecl *FD);

  std::deque<const FunctionDecl *> functionsToEmit;
  std::set<const FunctionDecl *> done;

  void run();

  virtual bool HandleTopLevelDecl(DeclGroupRef dg);

  mlir::Type getMLIRType(clang::QualType t);

  llvm::Type *getLLVMType(clang::QualType t);

  mlir::Type getMLIRType(llvm::Type *t);

  mlir::Location getMLIRLocation(clang::SourceLocation loc);
};

struct MLIRScanner : public StmtVisitor<MLIRScanner, ValueWithOffsets> {
public:
  MLIRASTConsumer &Glob;
  mlir::FuncOp function;
  mlir::ModuleOp &module;
  mlir::OpBuilder builder;
  mlir::Location loc;

  mlir::Block *entryBlock;

  std::vector<std::map<std::string, ValueWithOffsets>> scopes;
  std::vector<LoopContext> loops;

  void setValue(std::string name, ValueWithOffsets &&val);

  ValueWithOffsets getValue(std::string name);

  mlir::Type getLLVMTypeFromMLIRType(mlir::Type t);

  mlir::Location getMLIRLocation(clang::SourceLocation loc);

  mlir::Type getMLIRType(clang::QualType t);

  llvm::Type *getLLVMType(clang::QualType t);

  size_t getTypeSize(clang::QualType t);

  mlir::Value createAllocOp(mlir::Type t, std::string name, uint64_t memspace,
                            bool isArray);

  mlir::Value createAndSetAllocOp(std::string name, mlir::Value v,
                                  uint64_t memspace);

  MLIRScanner(MLIRASTConsumer &Glob, mlir::FuncOp function,
              const FunctionDecl *fd, mlir::ModuleOp &module)
      : Glob(Glob), function(function), module(module),
        builder(module.getContext()), loc(builder.getUnknownLoc()) {
    // llvm::errs() << *fd << "\n";

    scopes.emplace_back();
    std::vector<std::string> names;
    std::vector<bool> isReference;
    for (auto parm : fd->parameters()) {
      names.push_back(parm->getName().str());
      isReference.push_back(isa<LValueReferenceType>(parm->getType()) ||
                            isa<clang::ArrayType>(parm->getType()));
    }

    entryBlock = function.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    for (unsigned i = 0, e = function.getNumArguments(); i != e; ++i) {
      // function.getArgument(i).setName(names[i]);
      if (isReference[i])
        setValue(names[i], ValueWithOffsets(function.getArgument(i),
                                            {getConstantIndex(0)}));
      else
        createAndSetAllocOp(names[i], function.getArgument(i), 0);
    }

    scopes.emplace_back();

    Stmt *stmt = fd->getBody();
    // stmt->dump();
    Visit(stmt);

    auto endBlock = builder.getInsertionBlock();
    if (endBlock->empty() || endBlock->back().isKnownNonTerminator()) {
      builder.create<mlir::ReturnOp>(loc);
    }
    // function.dump();
  }

  ValueWithOffsets VisitDeclStmt(clang::DeclStmt *decl);

  ValueWithOffsets VisitIntegerLiteral(clang::IntegerLiteral *expr);

  ValueWithOffsets VisitFloatingLiteral(clang::FloatingLiteral *expr);

  ValueWithOffsets VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr);

  ValueWithOffsets VisitStringLiteral(clang::StringLiteral *expr);

  ValueWithOffsets VisitParenExpr(clang::ParenExpr *expr);

  ValueWithOffsets VisitVarDecl(clang::VarDecl *decl);

  ValueWithOffsets VisitForStmt(clang::ForStmt *fors);

  ValueWithOffsets VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr);

  mlir::FuncOp EmitDirectCallee(GlobalDecl GD);

  mlir::FuncOp EmitCallee(const Expr *E);

  ValueWithOffsets VisitCallExpr(clang::CallExpr *expr);

  std::map<int, mlir::Value> constants;
  mlir::Value getConstantIndex(int x);

  ValueWithOffsets VisitMSPropertyRefExpr(MSPropertyRefExpr *expr);

  ValueWithOffsets VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr);

  ValueWithOffsets VisitUnaryOperator(clang::UnaryOperator *U);

  ValueWithOffsets
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *expr);

  ValueWithOffsets VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Uop);

  ValueWithOffsets VisitBinaryOperator(clang::BinaryOperator *BO);

  ValueWithOffsets VisitAttributedStmt(AttributedStmt *AS);

  ValueWithOffsets VisitExprWithCleanups(ExprWithCleanups *E);

  ValueWithOffsets VisitDeclRefExpr(DeclRefExpr *E);

  ValueWithOffsets VisitOpaqueValueExpr(OpaqueValueExpr *E);

  ValueWithOffsets VisitMemberExpr(MemberExpr *ME);

  ValueWithOffsets VisitCastExpr(CastExpr *E);

  ValueWithOffsets VisitIfStmt(clang::IfStmt *stmt);

  ValueWithOffsets VisitConditionalOperator(clang::ConditionalOperator *E);

  ValueWithOffsets VisitCompoundStmt(clang::CompoundStmt *stmt);

  ValueWithOffsets VisitBreakStmt(clang::BreakStmt *stmt);

  ValueWithOffsets VisitContinueStmt(clang::ContinueStmt *stmt);

  ValueWithOffsets VisitReturnStmt(clang::ReturnStmt *stmt);
};

#endif

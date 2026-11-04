//===- clang-mlir.h - Emit MLIR IRs by walking clang AST---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_MLIR_H
#define CLANG_MLIR_H

#include "AffineUtils.h"
#include "ValueCategory.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "polygeist/Ops.h"
#include "pragmaHandler.h"
#include "HLSPragma.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/CommandLine.h"

#include "clang/../../lib/CodeGen/CGRecordLayout.h"
#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "clang/AST/Mangle.h"

using namespace clang;
using namespace mlir;

extern llvm::cl::opt<std::string> PrefixABI;

extern llvm::cl::opt<bool> HLSAnnotate;

// --scalar-wrapper-abi (default off), defined in clang-mlir.cc.
extern llvm::cl::opt<bool> ScalarWrapperABI;

struct LoopContext {
  mlir::Value keepRunning;
  mlir::Value noBreak;
};

struct MLIRASTConsumer : public ASTConsumer {
  std::set<std::string> &emitIfFound;
  std::set<std::string> &done;
  std::map<std::string, mlir::LLVM::GlobalOp> &llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &globals;
  std::map<std::string, mlir::func::FuncOp> &functions;
  std::map<std::string, mlir::LLVM::GlobalOp> &llvmGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> &llvmFunctions;
  Preprocessor &PP;
  ASTContext &astContext;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  clang::SourceManager &SM;
  llvm::LLVMContext lcontext;
  llvm::Module llvmMod;
  CodeGenOptions &codegenops;
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
      std::map<std::string, mlir::func::FuncOp> &functions,
      std::map<std::string, mlir::LLVM::GlobalOp> &llvmGlobals,
      std::map<std::string, mlir::LLVM::LLVMFuncOp> &llvmFunctions,
      Preprocessor &PP, ASTContext &astContext,
      mlir::OwningOpRef<mlir::ModuleOp> &module, clang::SourceManager &SM,
      CodeGenOptions &codegenops)
      : emitIfFound(emitIfFound), done(done),
        llvmStringGlobals(llvmStringGlobals), globals(globals),
        functions(functions), llvmGlobals(llvmGlobals),
        llvmFunctions(llvmFunctions), PP(PP), astContext(astContext),
        module(module), SM(SM), lcontext(), llvmMod("tmp", lcontext),
        codegenops(codegenops),
        CGM(astContext, nullptr, PP.getHeaderSearchInfo().getHeaderSearchOpts(),
            PP.getPreprocessorOpts(), codegenops, llvmMod, PP.getDiagnostics()),
        error(false), typeTranslator(*module->getContext()),
        reverseTypeTranslator(lcontext) {
   
    addPragmaScopHandlers(PP, scopLocList);
    addPragmaEndScopHandlers(PP, scopLocList);
    addPragmaLowerToHandlers(PP, LTInfo);
    cgeist::hls::beginTranslationUnit(PP, module->getContext());
  }

  ~MLIRASTConsumer() {}

  mlir::Type getVoidMemRefTy() const {
    return MemRefType::get({ShapedType::kDynamic},
                           mlir::OpBuilder(module->getContext()).getI8Type());
  }

  /// `xlxStreamObject` (--xlx-headers): the static class of the object a
  /// member of hls::stream is called on, when that is a class derived from
  /// the member's own class (hls::stream<T, D> over hls::stream<T, 0>); the
  /// external function is then named as a member of that class.
  mlir::func::FuncOp
  GetOrCreateMLIRFunction(const FunctionDecl *FD, bool getDeviceStub = false,
                          const clang::CXXRecordDecl *xlxStreamObject = nullptr);

  /// --xlx-headers: whether `RD` is a specialisation of the Vitis HLS class
  /// template hls::stream (hls_stream_39.h, ap_axi_sdata.h). Such an object
  /// is an i32 handle and its members are external functions, as with the
  /// stand-in headers (shim/exactclass hls_stream.h), whose IR form hls-est's
  /// FIFO model reads (StreamFifos.cpp, VITIS_NOTES N31).
  bool isXlxStreamClass(const clang::RecordDecl *RD) const;
  /// A member function of an hls::stream class (isXlxStreamClass): never
  /// emitted, every call stays an external call.
  bool isXlxStreamMember(const clang::FunctionDecl *FD) const;
  /// The bits Vitis packs a value of `T` into on a stream (VITIS_NOTES N58):
  /// a struct is the sum of its members' (bases first, no padding), `bool`
  /// is 1 bit, an array n x its element, anything else the width of the
  /// MLIR type cgeist gives it. None for a type with no such width.
  std::optional<int64_t> packedStreamBits(clang::QualType T);

  mlir::LLVM::LLVMFuncOp GetOrCreateLLVMFunction(const FunctionDecl *FD);
  mlir::LLVM::LLVMFuncOp GetOrCreateFreeFunction();
  mlir::Value CallMalloc(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value arg);

  mlir::LLVM::GlobalOp GetOrCreateLLVMGlobal(const ValueDecl *VD,
                                             std::string prefix = "");

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  mlir::Value GetOrCreateGlobalLLVMString(mlir::Location loc,
                                          mlir::OpBuilder &builder,
                                          StringRef value);

  std::pair<mlir::memref::GlobalOp, bool>
  GetOrCreateGlobal(const ValueDecl *VD, std::string prefix,
                    bool tryInit = true);

  /// A global whose initializer is not a constant cgeist can evaluate: a
  /// function `__hls_global_init.<name>` that computes it and stores it
  /// (C++'s dynamic initialization, which runs before main), marked
  /// `hls.global_init` with the declaration's source position. False when
  /// the initializer cannot be emitted that way.
  bool emitDynamicInitializer(const clang::VarDecl *VD, const clang::Expr *init,
                              mlir::memref::GlobalOp global, bool isArray);

  std::deque<const FunctionDecl *> functionsToEmit;

  void run();

  void HandleTranslationUnit(clang::ASTContext &Context) override;

  // Several translation units: the global definitions of this unit that no
  // emitted function used, so that a unit that only declares them (`extern`)
  // still gets their storage and initializer (CIRCT OPEN_ISSUES O75; W9r7
  // review defect 2). Unused ones are removed after the last unit
  // (eraseUnusedDefinitions).
  void emitUnusedDefinitions(clang::DeclContext *DC);

  bool HandleTopLevelDecl(DeclGroupRef dg) override;

  void HandleDeclContext(DeclContext *DC);

  std::map<const clang::RecordType *, mlir::LLVM::LLVMStructType> typeCache;

  /// Scalar wrappers (--scalar-wrapper-abi): a record whose only data member,
  /// through any chain of single-base classes without members of their own,
  /// is one scalar (integer, _BitInt, enum, floating). Such a record is
  /// lowered to that scalar: ap_fixed<16,5> is an i16, `this` / references
  /// are memref<?xi16>, arrays memref<...xi16>. Returns the member, or null
  /// when the record is not a wrapper. Memoised.
  llvm::DenseMap<const clang::RecordDecl *, const clang::FieldDecl *>
      scalarWrapperCache;
  const clang::FieldDecl *getScalarWrapperField(const clang::RecordDecl *RD);

  /// --scalar-wrapper-abi: the initial value of the variable `VD` of memref
  /// type `mr` as clang's constant evaluation of its initialiser (scalars,
  /// arrays, scalar wrappers and their arrays), or null when it is not a
  /// compile-time constant of that shape (or the option is off).
  mlir::DenseElementsAttr evaluateConstantInitializer(const clang::VarDecl *VD,
                                                      mlir::MemRefType mr);
  /// --xlx-headers (with --scalar-wrapper-abi): the initialiser `init` of an
  /// object of type `type` as the dense value of `mr`, when its leaves are
  /// scalar wrappers constructed from string literals (folded as by the Vitis
  /// front end, XlxStringBits.cc), ap types constructed from a numeric
  /// constant (foldNumericConstruct) or integer constants, and at least one
  /// is not a plain integer constant; null otherwise or when the options are
  /// off.
  mlir::DenseElementsAttr evaluateXlxStringInitializer(const clang::Expr *init,
                                                       clang::QualType type,
                                                       mlir::MemRefType mr);
  /// --scalar-wrapper-abi: the initial value of the global `VD` whose
  /// memref `mr` has llvm.struct elements, as one attribute per scalar in
  /// memory order (clang's constant evaluation, or the --xlx-headers folds of
  /// ap constructions); null when it is not such a constant.
  mlir::ArrayAttr evaluateStructInitializer(const clang::VarDecl *VD,
                                            const clang::Expr *init,
                                            mlir::MemRefType mr);
  mlir::Type getMLIRType(clang::QualType t, bool *implicitRef = nullptr,
                         bool allowMerge = true);

  llvm::Type *getLLVMType(clang::QualType t);

  mlir::Location getMLIRLocation(clang::SourceLocation loc);
};

class MLIRScanner : public StmtVisitor<MLIRScanner, ValueCategory> {
private:
  friend class IfScope;
  MLIRASTConsumer &Glob;
  mlir::func::FuncOp function;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  mlir::OpBuilder builder;
  mlir::Block *entryBlock;
  std::vector<LoopContext> loops;
  mlir::Block *allocationScope;

  std::map<const void *, std::vector<mlir::LLVM::AllocaOp>> bufs;
  mlir::LLVM::AllocaOp allocateBuffer(size_t i, mlir::LLVM::LLVMPointerType t) {
    auto &vec = bufs[t.getAsOpaquePointer()];
    if (i < vec.size())
      return vec[i];

    mlir::OpBuilder subbuilder(builder.getContext());
    subbuilder.setInsertionPointToStart(allocationScope);
    auto loc = subbuilder.getUnknownLoc();

    auto one = subbuilder.create<arith::ConstantIntOp>(loc, 1, 64);
    auto rs = subbuilder.create<mlir::LLVM::AllocaOp>(loc, t, one, 0);
    vec.push_back(rs);
    return rs;
  }

  LLVM::LLVMPointerType getOpaquePtr() {
    return LLVM::LLVMPointerType::get(builder.getContext());
  }

  mlir::Location getMLIRLocation(clang::SourceLocation loc);

  llvm::Type *getLLVMType(clang::QualType t);
  mlir::Type getMLIRType(clang::QualType t);

  mlir::Value getTypeSize(mlir::Location loc, clang::QualType t);
  mlir::Value getTypeAlign(mlir::Location loc, clang::QualType t);

  /// The constant a load of the const variable `VD` yields (clang's
  /// constant evaluation of its initialiser), as an arith.constant of the
  /// MLIR type of `resultType`, or null when it is not a compile-time
  /// constant scalar.
  mlir::Value tryEmitConstVarValue(mlir::Location loc, const VarDecl *VD,
                                   clang::QualType resultType);

  /// Value-initialisation of scalar-wrapper objects of type `T` at the
  /// reference `op` (--scalar-wrapper-abi): one zero store per object, a
  /// loop per array dimension of `T` (the outermost runs `count` times
  /// when given, as for `new T[n]()`).
  void storeZeroToWrapperObjects(mlir::Location loc, mlir::Value op,
                                 clang::QualType T, mlir::Value count);

  mlir::Value createAllocOp(mlir::Type t, VarDecl *name, uint64_t memspace,
                            bool isArray, bool LLVMABI);

  const clang::FunctionDecl *EmitCallee(const Expr *E);

  mlir::func::FuncOp EmitDirectCallee(const FunctionDecl *FD);

  std::map<int64_t, mlir::Value> constants;

  mlir::Value castToIndex(mlir::Location loc, mlir::Value val,
                          bool isUnsigned = false);

  mlir::Value getLLVM(Expr *E, bool isRef = false);

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

public:
  const FunctionDecl *EmittingFunctionDecl;
  std::map<const ValueDecl *, ValueCategory> params;
  llvm::DenseMap<const ValueDecl *, FieldDecl *> Captures;
  llvm::DenseMap<const ValueDecl *, LambdaCaptureKind> CaptureKinds;
  FieldDecl *ThisCapture;
  std::vector<mlir::Value> arrayinit;
  ValueCategory ThisVal;
  mlir::Value returnVal;
  LowerToInfo &LTInfo;

  MLIRScanner(MLIRASTConsumer &Glob, mlir::OwningOpRef<mlir::ModuleOp> &module,
              LowerToInfo &LTInfo);

  void init(mlir::func::FuncOp function, const FunctionDecl *fd);

  void setEntryAndAllocBlock(mlir::Block *B) {
    allocationScope = entryBlock = B;
    builder.setInsertionPointToStart(B);
  }

  mlir::OpBuilder &getBuilder();

  mlir::Value getConstantIndex(int64_t x);

  ValueCategory createComplexFloat(mlir::Location loc, mlir::Value real,
                                   mlir::Value imag, clang::QualType cty);
  mlir::Value getComplexPart(mlir::Location loc, mlir::Value complex, int fnum);
  ValueCategory getComplexPartRef(mlir::Location loc, mlir::Value complex,
                                  int fnum);

  ValueCategory VisitDeclStmt(clang::DeclStmt *decl);

  ValueCategory VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *decl);

  ValueCategory VisitExtVectorElementExpr(clang::ExtVectorElementExpr *expr);

  ValueCategory VisitConstantExpr(clang::ConstantExpr *expr);

  ValueCategory VisitAtomicExpr(clang::AtomicExpr *expr);

  ValueCategory VisitTypeTraitExpr(clang::TypeTraitExpr *expr);

  ValueCategory VisitGNUNullExpr(clang::GNUNullExpr *expr);
  ValueCategory VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr *expr);
  ValueCategory VisitIntegerLiteral(clang::IntegerLiteral *expr);

  ValueCategory VisitCharacterLiteral(clang::CharacterLiteral *expr);

  ValueCategory VisitFloatingLiteral(clang::FloatingLiteral *expr);

  ValueCategory VisitImaginaryLiteral(clang::ImaginaryLiteral *expr);

  ValueCategory VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr);

  ValueCategory VisitCXXTypeidExpr(clang::CXXTypeidExpr *expr);

  ValueCategory VisitCXXTryStmt(clang::CXXTryStmt *stmt);

  ValueCategory VisitStringLiteral(clang::StringLiteral *expr);

  ValueCategory VisitParenExpr(clang::ParenExpr *expr);

  ValueCategory VisitVarDecl(clang::VarDecl *decl);

  ValueCategory VisitForStmt(clang::ForStmt *fors);

  ValueCategory VisitCXXForRangeStmt(clang::CXXForRangeStmt *fors);

  ValueCategory VisitOMPSingleDirective(clang::OMPSingleDirective *);

  ValueCategory VisitOMPForDirective(clang::OMPForDirective *);

  ValueCategory VisitOMPParallelDirective(clang::OMPParallelDirective *);

  ValueCategory
  VisitOMPParallelForDirective(clang::OMPParallelForDirective *fors);

  ValueCategory VisitWhileStmt(clang::WhileStmt *fors);

  ValueCategory VisitDoStmt(clang::DoStmt *fors);

  ValueCategory VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr);

  ValueCategory VisitCallExpr(clang::CallExpr *expr);

  ValueCategory
  CallHelper(mlir::func::FuncOp tocall, QualType objType,
             ArrayRef<std::pair<ValueCategory, clang::Expr *>> arguments,
             QualType retType, bool retReference, clang::Expr *expr);

  std::pair<ValueCategory, bool>
  EmitClangBuiltinCallExpr(clang::CallExpr *expr);

  std::pair<ValueCategory, bool> EmitGPUCallExpr(clang::CallExpr *expr);

  std::pair<ValueCategory, bool> EmitBuiltinOps(clang::CallExpr *expr);

  /// The Xilinx HLS bit builtins of the Vitis headers (CGXlxBuiltin.cc).
  std::pair<ValueCategory, bool> EmitXlxBuiltin(clang::CallExpr *expr);

  ValueCategory
  VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr);

  ValueCategory
  VisitCXXPseudoDestructorExpr(clang::CXXPseudoDestructorExpr *expr);

  ValueCategory VisitCXXConstructExpr(clang::CXXConstructExpr *expr);

  ValueCategory VisitConstructCommon(clang::CXXConstructExpr *expr,
                                     VarDecl *name, unsigned space,
                                     mlir::Value mem = nullptr,
                                     mlir::Value count = nullptr);

  ValueCategory VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *expr);

  ValueCategory VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr);

  ValueCategory VisitUnaryOperator(clang::UnaryOperator *U);

  ValueCategory
  VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *expr);

  ValueCategory
  VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *Uop);

  ValueCategory VisitBinaryOperator(clang::BinaryOperator *BO);

  ValueCategory VisitCXXNoexceptExpr(clang::CXXNoexceptExpr *AS);

  ValueCategory VisitAttributedStmt(clang::AttributedStmt *AS);

  ValueCategory VisitExprWithCleanups(clang::ExprWithCleanups *E);

  ValueCategory VisitDeclRefExpr(clang::DeclRefExpr *E);

  ValueCategory VisitOpaqueValueExpr(clang::OpaqueValueExpr *E);

  ValueCategory VisitMemberExpr(clang::MemberExpr *ME);

  ValueCategory VisitCastExpr(clang::CastExpr *E);

  mlir::Value GetAddressOfBaseClass(mlir::Location loc, mlir::Value obj,
                                    const CXXRecordDecl *DerivedClass,
                                    ArrayRef<const clang::Type *> BaseTypes,
                                    ArrayRef<bool> BaseVirtuals);

  mlir::Value GetAddressOfDerivedClass(mlir::Location loc, mlir::Value obj,
                                       const CXXRecordDecl *DerivedClass,
                                       CastExpr::path_const_iterator Start,
                                       CastExpr::path_const_iterator End);

  ValueCategory VisitIfStmt(clang::IfStmt *stmt);

  ValueCategory VisitSwitchStmt(clang::SwitchStmt *stmt);

  ValueCategory VisitConditionalOperator(clang::ConditionalOperator *E);

  ValueCategory VisitCompoundStmt(clang::CompoundStmt *stmt);

  ValueCategory VisitBreakStmt(clang::BreakStmt *stmt);

  ValueCategory VisitContinueStmt(clang::ContinueStmt *stmt);

  ValueCategory VisitReturnStmt(clang::ReturnStmt *stmt);

  std::map<LabelStmt *, Block *> labels;
  ValueCategory VisitLabelStmt(clang::LabelStmt *stmt);

  ValueCategory VisitGotoStmt(clang::GotoStmt *stmt);

  ValueCategory VisitStmtExpr(clang::StmtExpr *stmt);

  ValueCategory VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr);

  ValueCategory
  VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *expr);

  ValueCategory VisitCXXNewExpr(clang::CXXNewExpr *expr);

  ValueCategory VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr);

  ValueCategory VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr);

  ValueCategory VisitCXXThisExpr(clang::CXXThisExpr *expr);

  ValueCategory VisitPredefinedExpr(clang::PredefinedExpr *expr);

  ValueCategory VisitLambdaExpr(clang::LambdaExpr *expr);

  ValueCategory VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr);

  ValueCategory VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr);

  mlir::Attribute InitializeValueByInitListExpr(mlir::Value toInit,
                                                clang::Expr *expr);
  ValueCategory VisitInitListExpr(clang::InitListExpr *expr);
  ValueCategory
  VisitCXXStdInitializerListExpr(clang::CXXStdInitializerListExpr *expr);

  ValueCategory VisitArrayInitLoop(clang::ArrayInitLoopExpr *expr,
                                   ValueCategory tostore);

  ValueCategory VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *expr);

  ValueCategory VisitSizeOfPackExpr(clang::SizeOfPackExpr *e);

  ValueCategory CommonFieldLookup(mlir::Location loc, clang::QualType OT,
                                  const FieldDecl *FD, mlir::Value val,
                                  bool isLValue);

  ValueCategory CommonArrayLookup(mlir::Location loc, ValueCategory val,
                                  mlir::Value idx, bool isImplicitRefResult,
                                  bool removeIndex = true);

  ValueCategory CommonArrayToPointer(mlir::Location loc, ValueCategory val);
};

#endif

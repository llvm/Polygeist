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

#include "../../llvm-project/clang/lib/CodeGen/CGRecordLayout.h"
#include "../../llvm-project/clang/lib/CodeGen/CodeGenModule.h"
#include "clang/AST/Mangle.h"

using namespace clang;
using namespace mlir;

struct LoopContext {
  mlir::Value keepRunning;
  mlir::Value noBreak;
};

struct AffineLoopDescriptor {
private:
  mlir::Value upperBound;
  mlir::Value lowerBound;
  int64_t step;
  mlir::Type indVarType;
  VarDecl *indVar;
  bool forwardMode;

public:
  AffineLoopDescriptor()
      : upperBound(nullptr), lowerBound(nullptr),
        step(std::numeric_limits<int64_t>::max()), indVarType(nullptr),
        indVar(nullptr), forwardMode(true){};
  AffineLoopDescriptor(const AffineLoopDescriptor &) = delete;

  void setLowerBound(mlir::Value value) { lowerBound = value; }
  void setUpperBound(mlir::Value value) { upperBound = value; }

  void setStep(int value) { step = value; };
  void setType(mlir::Type type) { indVarType = type; }
  void setName(VarDecl *value) { indVar = value; }

  VarDecl *getName() const { return indVar; }
  mlir::Type getType() const { return indVarType; }
  int getStep() const { return step; }

  auto getLowerBound() const { return lowerBound; }

  auto getUpperBound() const { return upperBound; }

  void setForwardMode(bool value) { forwardMode = value; };
  bool getForwardMode() const { return forwardMode; }
};

struct ValueWithOffsets {
  mlir::Value val;
  bool isReference;
  ValueWithOffsets() : val(nullptr), isReference(false){};
  ValueWithOffsets(std::nullptr_t) : val(nullptr), isReference(false){};
  ValueWithOffsets(mlir::Value val, bool isReference)
      : val(val), isReference(isReference) {
    if (isReference) {
      if (val.getType().isa<mlir::LLVM::LLVMPointerType>()) {

      } else if (val.getType().isa<mlir::MemRefType>()) {

      } else {
        val.getDefiningOp()->getParentOfType<FuncOp>().dump();
        llvm::errs() << val << "\n";
        assert(val.getType().isa<mlir::MemRefType>());
      }
    }
  };

  mlir::Value getValue(OpBuilder &builder) const {
    assert(val);
    if (!isReference)
      return val;
    auto loc = builder.getUnknownLoc();
    if (val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return builder.create<mlir::LLVM::LoadOp>(loc, val);
    }
    auto c0 = builder.create<mlir::ConstantIndexOp>(loc, 0);
    if (!val.getType().isa<mlir::MemRefType>()) {
      llvm::errs() << val << "\n";
    }
    assert(val.getType().isa<mlir::MemRefType>());
    // return ValueWithOffsets(builder.create<memref::SubIndexOp>(loc, mt0, val,
    // c0), /*isReference*/true);
    if (val.getType().cast<mlir::MemRefType>().getShape().size() != 1) {
      llvm::errs() << " val: " << val << " ty: " << val.getType() << "\n";
    }
    assert(val.getType().cast<mlir::MemRefType>().getShape().size() == 1);
    return builder.create<memref::LoadOp>(loc, val,
                                          std::vector<mlir::Value>({c0}));
  }

  void store(OpBuilder &builder, ValueWithOffsets toStore, bool isArray) const {
    assert(toStore.val);
    if (isArray) {
      if (!toStore.isReference) {
        llvm::errs() << " toStore.val: " << toStore.val << " isref "
                     << toStore.isReference << " isar" << isArray << "\n";
      }
      assert(toStore.isReference);
      auto loc = builder.getUnknownLoc();
      auto zeroIndex = builder.create<mlir::ConstantIndexOp>(loc, 0);

      if (auto smt = toStore.val.getType().dyn_cast<MemRefType>()) {
        assert(smt.getShape().size() <= 2);

        if (auto mt = val.getType().dyn_cast<MemRefType>()) {
          assert(smt.getElementType() == mt.getElementType());
          if (mt.getShape().size() != smt.getShape().size()) {
            llvm::errs() << " val: " << val << " tsv: " << toStore.val << "\n";
            llvm::errs() << " mt: " << mt << " smt: " << smt << "\n";
          }
          assert(mt.getShape().size() == smt.getShape().size());
          assert(smt.getShape().back() == mt.getShape().back());

          for (ssize_t i = 0; i < smt.getShape().back(); i++) {
            SmallVector<mlir::Value, 2> idx;
            if (smt.getShape().size() == 2)
              idx.push_back(zeroIndex);
            idx.push_back(builder.create<mlir::ConstantIndexOp>(loc, i));
            builder.create<mlir::memref::StoreOp>(
                loc,
                builder.create<mlir::memref::LoadOp>(loc, toStore.val, idx),
                val, idx);
          }
        } else {
          auto pt = val.getType().cast<LLVM::LLVMPointerType>();
          mlir::Type elty;
          if (auto at = pt.getElementType().dyn_cast<LLVM::LLVMArrayType>()) {
            elty = at.getElementType();
            assert(smt.getShape().back() == at.getNumElements());
          } else {
            auto st = pt.getElementType().dyn_cast<LLVM::LLVMStructType>();
            elty = st.getBody()[0];
            assert(smt.getShape().back() == (ssize_t)st.getBody().size());
          }
          assert(elty == smt.getElementType());
          elty = LLVM::LLVMPointerType::get(elty, pt.getAddressSpace());

          auto iTy = builder.getIntegerType(32);
          auto zero32 = builder.create<mlir::ConstantOp>(
              loc, iTy, builder.getIntegerAttr(iTy, 0));
          for (ssize_t i = 0; i < smt.getShape().back(); i++) {
            SmallVector<mlir::Value, 2> idx;
            if (smt.getShape().size() == 2)
              idx.push_back(zeroIndex);
            idx.push_back(builder.create<mlir::ConstantIndexOp>(loc, i));
            mlir::Value lidx[] = {
                val, zero32,
                builder.create<mlir::ConstantOp>(
                    loc, iTy, builder.getIntegerAttr(iTy, i))};
            builder.create<mlir::LLVM::StoreOp>(
                loc,
                builder.create<mlir::memref::LoadOp>(loc, toStore.val, idx),
                builder.create<mlir::LLVM::GEPOp>(loc, elty, lidx));
          }
        }
      } else if (auto smt = val.getType().dyn_cast<MemRefType>()) {
        assert(smt.getShape().size() <= 2);

        auto pt = toStore.val.getType().cast<LLVM::LLVMPointerType>();
        mlir::Type elty;
        if (auto at = pt.getElementType().dyn_cast<LLVM::LLVMArrayType>()) {
          elty = at.getElementType();
          assert(smt.getShape().back() == at.getNumElements());
        } else {
          auto st = pt.getElementType().dyn_cast<LLVM::LLVMStructType>();
          elty = st.getBody()[0];
          assert(smt.getShape().back() == (ssize_t)st.getBody().size());
        }
        assert(elty == smt.getElementType());
        elty = LLVM::LLVMPointerType::get(elty, pt.getAddressSpace());

        auto iTy = builder.getIntegerType(32);
        auto zero32 = builder.create<mlir::ConstantOp>(
            loc, iTy, builder.getIntegerAttr(iTy, 0));
        for (ssize_t i = 0; i < smt.getShape().back(); i++) {
          SmallVector<mlir::Value, 2> idx;
          if (smt.getShape().size() == 2)
            idx.push_back(zeroIndex);
          idx.push_back(builder.create<mlir::ConstantIndexOp>(loc, i));
          mlir::Value lidx[] = {toStore.val, zero32,
                                builder.create<mlir::ConstantOp>(
                                    loc, iTy, builder.getIntegerAttr(iTy, i))};
          builder.create<mlir::memref::StoreOp>(
              loc,
              builder.create<mlir::LLVM::LoadOp>(
                  loc, builder.create<mlir::LLVM::GEPOp>(loc, elty, lidx)),
              val, idx);
        }
      } else
        store(builder, toStore.getValue(builder));
    } else {
      store(builder, toStore.getValue(builder));
    }
  }

  void store(OpBuilder &builder, mlir::Value toStore) const {
    assert(isReference);
    assert(val);
    auto loc = builder.getUnknownLoc();
    if (auto PT = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      if (toStore.getType() != PT.getElementType()) {
        if (auto mt = toStore.getType().dyn_cast<MemRefType>()) {
          if (auto spt =
                  PT.getElementType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
            if (mt.getElementType() == spt.getElementType()) {
              toStore = builder.create<polygeist::Memref2PointerOp>(loc, spt,
                                                                    toStore);
            }
          }
        }
      }
      if (toStore.getType() != PT.getElementType()) {
        llvm::errs() << " toStore: " << toStore << " PT: " << PT
                     << " val: " << val << "\n";
      }
      assert(toStore.getType() == PT.getElementType());
      builder.create<mlir::LLVM::StoreOp>(loc, toStore, val);
    } else {
      assert(val.getType().cast<MemRefType>().getShape().size() == 1);
      if (toStore.getType() !=
          val.getType().cast<MemRefType>().getElementType()) {
        llvm::errs() << " toStore: " << toStore
                     << " PT: " << val.getType().cast<MemRefType>()
                     << " val: " << val << "\n";
      }
      assert(toStore.getType() ==
             val.getType().cast<MemRefType>().getElementType());
      auto c0 = builder.create<mlir::ConstantIndexOp>(loc, 0);
      builder.create<mlir::memref::StoreOp>(loc, toStore, val,
                                            std::vector<mlir::Value>({c0}));
    }
  }

  ValueWithOffsets dereference(OpBuilder &builder) const {
    assert(val && "val must be not-null");

    auto loc = builder.getUnknownLoc();
    if (val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      if (!isReference)
        return ValueWithOffsets(val, /*isReference*/ true);
      else
        return ValueWithOffsets(builder.create<mlir::LLVM::LoadOp>(loc, val),
                                /*isReference*/ true);
    }

    auto c0 = builder.create<mlir::ConstantIndexOp>(loc, 0);
    if (!val.getType().isa<mlir::MemRefType>())
      llvm::errs() << val << "\n";
    auto mt = val.getType().cast<mlir::MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());

    if (isReference) {
      if (shape.size() > 1) {
        shape.erase(shape.begin());
        auto mt0 =
            mlir::MemRefType::get(shape, mt.getElementType(),
                                  mt.getAffineMaps(), mt.getMemorySpace());
        return ValueWithOffsets(
            builder.create<polygeist::SubIndexOp>(loc, mt0, val, c0),
            /*isReference*/ true);
      } else {
        // shape[0] = -1;
        return ValueWithOffsets(builder.create<memref::LoadOp>(
                                    loc, val, std::vector<mlir::Value>({c0})),
                                /*isReference*/ true);
      }
    }
    // return ValueWithOffsets(val, /*isReference*/true);
    // assert(shape.size() == 1);
    return ValueWithOffsets(val, /*isReference*/ true);
  }
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
  mlir::ModuleOp &module;
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
      Preprocessor &PP, ASTContext &astContext, mlir::ModuleOp &module,
      clang::SourceManager &SM)
      : emitIfFound(emitIfFound), done(done),
        llvmStringGlobals(llvmStringGlobals), globals(globals),
        functions(functions), llvmGlobals(llvmGlobals),
        llvmFunctions(llvmFunctions), PP(PP), astContext(astContext),
        module(module), SM(SM), lcontext(), llvmMod("tmp", lcontext),
        codegenops(),
        CGM(astContext, PP.getHeaderSearchInfo().getHeaderSearchOpts(),
            PP.getPreprocessorOpts(), codegenops, llvmMod, PP.getDiagnostics()),
        error(false), typeTranslator(*module.getContext()),
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

  std::pair<mlir::memref::GlobalOp, bool>
  GetOrCreateGlobal(const ValueDecl *VD);

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

struct MLIRScanner : public StmtVisitor<MLIRScanner, ValueWithOffsets> {
private:
  MLIRASTConsumer &Glob;
  mlir::FuncOp function;
  mlir::ModuleOp &module;
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::Block *entryBlock;
  std::vector<LoopContext> loops;
  mlir::Block *allocationScope;

  // ValueWithOffsets getValue(std::string name);

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

  bool isTrivialAffineLoop(clang::ForStmt *fors, AffineLoopDescriptor &descr);

  bool getUpperBound(clang::ForStmt *fors, AffineLoopDescriptor &descr);

  bool getLowerBound(clang::ForStmt *fors, AffineLoopDescriptor &descr);

  bool getConstantStep(clang::ForStmt *fors, AffineLoopDescriptor &descr);

  void buildAffineLoop(clang::ForStmt *fors, mlir::Location loc,
                       const AffineLoopDescriptor &descr);

  void buildAffineLoopImpl(clang::ForStmt *fors, mlir::Location loc,
                           mlir::Value lb, mlir::Value ub,
                           const AffineLoopDescriptor &descr);
  std::vector<Block *> prevBlock;
  std::vector<Block::iterator> prevIterator;

public:
  void pushLoopIf();
  void popLoopIf();

public:
  const FunctionDecl *EmittingFunctionDecl;
  std::map<const VarDecl *, ValueWithOffsets> params;
  llvm::DenseMap<const VarDecl *, FieldDecl *> Captures;
  llvm::DenseMap<const VarDecl *, LambdaCaptureKind> CaptureKinds;
  FieldDecl *ThisCapture;
  std::vector<mlir::Value> arrayinit;
  ValueWithOffsets ThisVal;
  mlir::Value returnVal;
  LowerToInfo &LTInfo;

  MLIRScanner(MLIRASTConsumer &Glob, mlir::FuncOp function,
              const FunctionDecl *fd, mlir::ModuleOp &module,
              LowerToInfo &LTInfo)
      : Glob(Glob), function(function), module(module),
        builder(module.getContext()), loc(builder.getUnknownLoc()),
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
        ThisVal = ValueWithOffsets(val, /*isReference*/ false);
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
        params.emplace(parm, ValueWithOffsets(val, /*isReference*/ true));
      } else {
        auto alloc = createAllocOp(val.getType(), parm, /*memspace*/ 0, isArray,
                                   /*LLVMABI*/ LLVMABI);
        ValueWithOffsets(alloc, /*isReference*/ true).store(builder, val);
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
    }
    Visit(stmt);

    if (function.getType().getResults().size()) {
      mlir::Value vals[1] = {
          builder.create<mlir::memref::LoadOp>(loc, returnVal)};
      builder.create<mlir::ReturnOp>(loc, vals);
    } else
      builder.create<mlir::ReturnOp>(loc);
    // function.dump();
  }

  ValueWithOffsets VisitDeclStmt(clang::DeclStmt *decl);

  ValueWithOffsets
  VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *decl);

  ValueWithOffsets VisitConstantExpr(clang::ConstantExpr *expr);

  ValueWithOffsets VisitIntegerLiteral(clang::IntegerLiteral *expr);

  ValueWithOffsets VisitCharacterLiteral(clang::CharacterLiteral *expr);

  ValueWithOffsets VisitFloatingLiteral(clang::FloatingLiteral *expr);

  ValueWithOffsets VisitImaginaryLiteral(clang::ImaginaryLiteral *expr);

  ValueWithOffsets VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr);
  ValueWithOffsets VisitCXXTypeidExpr(clang::CXXTypeidExpr *expr);

  ValueWithOffsets VisitStringLiteral(clang::StringLiteral *expr);

  ValueWithOffsets VisitParenExpr(clang::ParenExpr *expr);

  ValueWithOffsets VisitVarDecl(clang::VarDecl *decl);

  ValueWithOffsets VisitForStmt(clang::ForStmt *fors);

  ValueWithOffsets
  VisitOMPParallelForDirective(clang::OMPParallelForDirective *fors);

  ValueWithOffsets VisitWhileStmt(clang::WhileStmt *fors);

  ValueWithOffsets VisitDoStmt(clang::DoStmt *fors);

  ValueWithOffsets VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr);

  ValueWithOffsets VisitCallExpr(clang::CallExpr *expr);

  std::pair<ValueWithOffsets, bool> EmitGPUCallExpr(clang::CallExpr *expr);

  std::pair<ValueWithOffsets, bool> EmitBuiltinOps(clang::CallExpr *expr);

  ValueWithOffsets VisitCXXConstructExpr(clang::CXXConstructExpr *expr);

  ValueWithOffsets VisitConstructCommon(clang::CXXConstructExpr *expr,
                                        VarDecl *name, unsigned space,
                                        mlir::Value mem = nullptr);

  ValueWithOffsets VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *expr);

  ValueWithOffsets VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr);

  ValueWithOffsets VisitUnaryOperator(clang::UnaryOperator *U);

  ValueWithOffsets
  VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *expr);

  ValueWithOffsets
  VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *Uop);

  ValueWithOffsets VisitBinaryOperator(clang::BinaryOperator *BO);

  ValueWithOffsets VisitAttributedStmt(clang::AttributedStmt *AS);

  ValueWithOffsets VisitExprWithCleanups(clang::ExprWithCleanups *E);

  ValueWithOffsets VisitDeclRefExpr(clang::DeclRefExpr *E);

  ValueWithOffsets VisitOpaqueValueExpr(clang::OpaqueValueExpr *E);

  ValueWithOffsets VisitMemberExpr(clang::MemberExpr *ME);

  ValueWithOffsets VisitCastExpr(clang::CastExpr *E);

  ValueWithOffsets VisitIfStmt(clang::IfStmt *stmt);

  ValueWithOffsets VisitSwitchStmt(clang::SwitchStmt *stmt);

  ValueWithOffsets VisitConditionalOperator(clang::ConditionalOperator *E);

  ValueWithOffsets VisitCompoundStmt(clang::CompoundStmt *stmt);

  ValueWithOffsets VisitBreakStmt(clang::BreakStmt *stmt);

  ValueWithOffsets VisitContinueStmt(clang::ContinueStmt *stmt);

  ValueWithOffsets VisitReturnStmt(clang::ReturnStmt *stmt);

  ValueWithOffsets VisitStmtExpr(clang::StmtExpr *stmt);

  ValueWithOffsets VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr);

  ValueWithOffsets
  VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *expr);

  ValueWithOffsets VisitCXXNewExpr(clang::CXXNewExpr *expr);

  ValueWithOffsets VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr);

  ValueWithOffsets VisitCXXThisExpr(clang::CXXThisExpr *expr);

  ValueWithOffsets VisitPredefinedExpr(clang::PredefinedExpr *expr);

  ValueWithOffsets VisitLambdaExpr(clang::LambdaExpr *expr);

  ValueWithOffsets VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr);

  ValueWithOffsets
  VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr);

  ValueWithOffsets VisitInitListExpr(clang::InitListExpr *expr);

  ValueWithOffsets VisitArrayInitLoop(clang::ArrayInitLoopExpr *expr,
                                      ValueWithOffsets tostore);

  ValueWithOffsets VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *expr);

  ValueWithOffsets CommonFieldLookup(clang::QualType OT, const FieldDecl *FD,
                                     mlir::Value val);

  ValueWithOffsets CommonArrayLookup(ValueWithOffsets val, mlir::Value idx);

  ValueWithOffsets CommonArrayToPointer(ValueWithOffsets val);
};

#endif

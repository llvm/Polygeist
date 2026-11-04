//===- clang-mlir.cc - Emit MLIR IRs by walking clang AST--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-mlir.h"
#include "polygeist/Passes/Passes.h"
#include "../ArgumentList.h"
#include "TypeUtils.h"
#include "XlxStringBits.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "utils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace std;
using namespace clang;
using namespace llvm;
using namespace clang::driver;
using namespace llvm::opt;
using namespace mlir;
using namespace mlir::arith;
using namespace mlirclang;

#define DEBUG_TYPE "clang-mlir"

static cl::opt<bool>
    memRefFullRank("memref-fullrank", cl::init(false),
                   cl::desc("Get the full rank of the memref."));

static cl::opt<bool> memRefABI("memref-abi", cl::init(true),
                               cl::desc("Use memrefs when possible"));

static cl::opt<bool> HLSDynamicGlobalInit(
    "hls-dynamic-global-init", cl::init(true),
    cl::desc("Emit a global whose initializer is not a constant (C++ dynamic "
             "initialization, `static const T s = 1.0 / sqrt(128.0)`) as a "
             "function __hls_global_init.<name> marked hls.global_init that "
             "computes and stores it, instead of leaving the global "
             "uninitialized (check_semantics runs these before the top; "
             "hls-est does not reach them)"));

cl::opt<bool> HLSAnnotate("hls-annotate", cl::init(false),
                              cl::desc("annotate hls pragmas"));

cl::opt<std::string> PrefixABI("prefix-abi", cl::init(""),
                               cl::desc("Prefix for emitted symbols"));

cl::opt<bool> CStyleMemRef("c-style-memref", cl::init(true),
                           cl::desc("Use c style memrefs when possible"));

static cl::opt<bool>
    CombinedStructABI("struct-abi", cl::init(true),
                      cl::desc("Use literal LLVM ABI for structs"));

// Opt-in (default off): with the option off, every code path below behaves
// exactly as before the option existed.
cl::opt<bool> ScalarWrapperABI(
    "scalar-wrapper-abi", cl::init(false),
    cl::desc("Lower a class whose only data member (through any chain of "
             "single-base classes) is one scalar to that scalar: ap_fixed / "
             "ap_int style wrappers become iN / fN values, references and "
             "memref elements (also enables the constant evaluation of const "
             "initialisers, explicit integer argument / return width "
             "conversion, per-dimension construction of N-D class arrays and "
             "__builtin_bit_cast that such wrappers need)"));

ValueCategory MLIRScanner::createComplexFloat(mlir::Location loc,
                                              mlir::Value real,
                                              mlir::Value imag,
                                              clang::QualType cty) {
  bool ref;
  mlir::Type elty = getMLIRType(cty);
  mlir::Type mt;
  OpBuilder abuilder(builder.getContext());
  abuilder.setInsertionPointToStart(allocationScope);
  mlir::Value result;
  if (auto MT = dyn_cast<MemRefType>(elty)) {
    mt = elty;
    elty = MT.getElementType();
    ref = true;
    result = abuilder.create<mlir::memref::AllocaOp>(loc, MT);
  } else {
    ref = false;
  }

  if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(elty)) {
    mlir::Value str = builder.create<polygeist::UndefOp>(loc, ST);
    str = builder.create<LLVM::InsertValueOp>(loc, ST, str, real,
                                              builder.getDenseI64ArrayAttr(0));
    str = builder.create<LLVM::InsertValueOp>(loc, ST, str, imag,
                                              builder.getDenseI64ArrayAttr(1));
    if (ref) {
      builder.create<mlir::memref::StoreOp>(loc, str, result,
                                            getConstantIndex(0));
    } else {
      result = str;
    }
  } else {
    builder.create<mlir::memref::StoreOp>(loc, real, result,
                                          getConstantIndex(0));
    builder.create<mlir::memref::StoreOp>(loc, imag, result,
                                          getConstantIndex(1));
  }
  return ValueCategory(result, /*isReference*/ ref);
}

ValueCategory MLIRScanner::getComplexPartRef(mlir::Location loc,
                                             mlir::Value val, int fnum) {
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
  if (auto mt = dyn_cast<mlir::MemRefType>(val.getType());
      mt && mt.getRank() == 1 && mt.getElementType().isa<mlir::FloatType>()) {
    // A temporary of createComplexFloat (memref<2xT>): the parts are its
    // elements 0 and 1.
    auto mt1 = mlir::MemRefType::get({ShapedType::kDynamic},
                                     mt.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     mt.getMemorySpace());
    return ValueCategory(builder.create<polygeist::SubIndexOp>(
                             loc, mt1, val, getConstantIndex(fnum)),
                         /*isReference*/ true);
  }
  if (auto mt = dyn_cast<mlir::MemRefType>(val.getType())) {
    auto shape = std::vector<int64_t>(mt.getShape());
    assert(shape.size() == 2);
    shape.erase(shape.begin());
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    shape[0] = ShapedType::kDynamic;
    auto mt1 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    mlir::Value si = builder.create<polygeist::SubIndexOp>(loc, mt0, val,
                                                           getConstantIndex(0));
    si = builder.create<polygeist::SubIndexOp>(loc, mt1, si,
                                               getConstantIndex(fnum));
    return ValueCategory(si, /*isReference*/ true);
  } else if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(val.getType())) {
    mlir::Type ET;
    if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(PT.getElementType())) {
      ET = ST.getBody()[fnum];
    } else {
      ET = PT.getElementType()
               .cast<mlir::LLVM::LLVMArrayType>()
               .getElementType();
    }
    mlir::Value vec[2] = {builder.create<ConstantIntOp>(loc, 0, 32),
                          builder.create<ConstantIntOp>(loc, fnum, 32)};
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(
            loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
            val, vec),
        /*isReference*/ true);
  } else {
    llvm_unreachable("unexpected complex type");
  }
}

/// Get real (fnum = 0) or imaginary (fnum = 1) part of a complex float
mlir::Value MLIRScanner::getComplexPart(mlir::Location loc, mlir::Value complex,
                                        int fnum) {
  if (auto ft = dyn_cast<mlir::FloatType>(complex.getType())) {
    if (fnum == 0)
      return complex;
    else
      return builder.create<ConstantFloatOp>(
          loc, APFloat::getZero(ft.getFloatSemantics()), ft);
  } else if (auto ST =
                 dyn_cast<mlir::LLVM::LLVMStructType>(complex.getType())) {
    return builder.create<LLVM::ExtractValueOp>(loc, complex, fnum);
  }
  auto ref = getComplexPartRef(loc, complex, fnum).val;
  if (auto mt = dyn_cast<mlir::MemRefType>(ref.getType())) {
    // A memref<?xT> view of the part: element 0 (a load without indices is
    // invalid on it and asserted in SimplifySubIndexUsers).
    return ValueCategory(ref, /*isReference*/ true).getValue(loc, builder);
  } else if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(ref.getType())) {
    return builder.create<mlir::LLVM::LoadOp>(loc, ref);
  } else {
    assert(0);
  }
}

bool isLLVMStructABI(const RecordDecl *RD, llvm::StructType *ST) {
  if (!CombinedStructABI)
    return true;
  if (RD->isUnion())
    return true;
  if (auto CXRD = dyn_cast<CXXRecordDecl>(RD)) {
    if (!CXRD->hasDefinition())
      return true;
    if (CXRD->getNumVBases())
      return true;
    for (auto m : CXRD->methods()) {
      if (m->isVirtualAsWritten() || m->isPure())
        return true;
    }
  }
  if (ST) {
    if (!ST->isLiteral() && (ST->getName() == "struct._IO_FILE" ||
                             ST->getName() == "class.std::basic_ifstream" ||
                             ST->getName() == "class.std::basic_istream" ||
                             ST->getName() == "class.std::basic_ostream" ||
                             ST->getName() == "class.std::basic_ofstream"))
      return true;
  }
  return false;
}

bool MLIRASTConsumer::isXlxStreamMember(const clang::FunctionDecl *FD) const {
  const auto *md = dyn_cast_or_null<clang::CXXMethodDecl>(FD);
  return md && isXlxStreamClass(md->getParent());
}

bool MLIRASTConsumer::isXlxStreamClass(const clang::RecordDecl *RD) const {
  if (!cgeist::hls::xlxHeaders() || !RD)
    return false;
  auto *spec = dyn_cast<clang::ClassTemplateSpecializationDecl>(RD);
  if (!spec || spec->getName() != "stream")
    return false;
  auto *ns = dyn_cast<clang::NamespaceDecl>(
      spec->getDeclContext()->getRedeclContext());
  return ns && ns->getName() == "hls" &&
         ns->getDeclContext()->getRedeclContext()->isTranslationUnit();
}

std::optional<int64_t>
MLIRASTConsumer::packedStreamBits(clang::QualType T) {
  T = T.getCanonicalType();
  if (T->isBooleanType())
    return 1;
  if (const auto *AT = CGM.getContext().getAsConstantArrayType(T)) {
    std::optional<int64_t> e = packedStreamBits(AT->getElementType());
    if (!e)
      return std::nullopt;
    return *e * static_cast<int64_t>(AT->getSize().getZExtValue());
  }
  // A scalar (also an ap_int / ap_fixed class cgeist lowers to one iW).
  auto scalarBits = [&](mlir::Type mt) -> std::optional<int64_t> {
    if (mt.isIntOrFloat())
      return mt.getIntOrFloatBitWidth();
    return std::nullopt;
  };
  const auto *RD = T->getAsRecordDecl();
  if (!RD) {
    if (T->isPointerType() || T->isReferenceType())
      return std::nullopt;
    return scalarBits(getMLIRType(T));
  }
  if (getScalarWrapperField(RD))
    if (std::optional<int64_t> b = scalarBits(getMLIRType(T)))
      return b;
  RD = RD->getDefinition();
  if (!RD || RD->isUnion() || RD->isInvalidDecl() || RD->isDependentType())
    return std::nullopt;
  int64_t sum = 0;
  if (const auto *CXRD = dyn_cast<clang::CXXRecordDecl>(RD)) {
    if (CXRD->isPolymorphic() || CXRD->getNumVBases() != 0)
      return std::nullopt;
    for (const clang::CXXBaseSpecifier &B : CXRD->bases()) {
      std::optional<int64_t> b = packedStreamBits(B.getType());
      if (!b)
        return std::nullopt;
      sum += *b;
    }
  }
  for (const clang::FieldDecl *F : RD->fields()) {
    if (F->isBitField())
      return std::nullopt;
    std::optional<int64_t> b = packedStreamBits(F->getType());
    if (!b)
      return std::nullopt;
    sum += *b;
  }
  return sum;
}

const clang::FieldDecl *
MLIRASTConsumer::getScalarWrapperField(const clang::RecordDecl *RD) {
  if (!ScalarWrapperABI || !RD)
    return nullptr;
  RD = RD->getDefinition();
  if (!RD)
    return nullptr;
  auto it = scalarWrapperCache.find(RD);
  if (it != scalarWrapperCache.end())
    return it->second;
  scalarWrapperCache[RD] = nullptr; // recursion guard
  const clang::FieldDecl *result = nullptr;
  // (--xlx-headers) An hls::stream is one i32 handle (getMLIRType) whatever
  // its element type: the payload member `V` (hls::stream<T, 0>, also through
  // the base of hls::stream<T, D>) stands for it, so that the base / derived
  // conversions of the object are the identity. Only the stream's own
  // members, which cgeist never emits, name `V`.
  if (isXlxStreamClass(RD)) {
    auto *CXRD = dyn_cast<CXXRecordDecl>(RD);
    if (CXRD && CXRD->getNumBases() == 1 && RD->field_empty()) {
      if (auto *BRT =
              CXRD->bases_begin()->getType()->getAs<clang::RecordType>())
        result = getScalarWrapperField(BRT->getDecl());
    } else if (!RD->field_empty()) {
      result = *RD->field_begin();
    }
    scalarWrapperCache[RD] = result;
    return result;
  }
  do {
    if (RD->isUnion() || RD->isInvalidDecl() || RD->isDependentType() ||
        RD->hasAttr<clang::PackedAttr>() || RD->hasAttr<clang::AlignedAttr>())
      break;
    auto *CXRD = dyn_cast<CXXRecordDecl>(RD);
    if (CXRD && (CXRD->isLambda() || CXRD->isPolymorphic() ||
                 CXRD->getNumVBases() != 0 || CXRD->getNumBases() > 1))
      break;
    const clang::FieldDecl *field = nullptr;
    unsigned nfields = 0;
    for (auto *f : RD->fields()) {
      field = f;
      nfields++;
    }
    if (CXRD && CXRD->getNumBases() == 1) {
      // Chain: a class without members of its own over a wrapper base
      // (ap_fixed over ap_fixed_base, ap_uint over ap_int_base).
      if (nfields != 0)
        break;
      auto *BRT = CXRD->bases_begin()->getType()->getAs<clang::RecordType>();
      if (BRT)
        result = getScalarWrapperField(BRT->getDecl());
      break;
    }
    if (nfields != 1 || field->isBitField())
      break;
    clang::QualType FT = field->getType();
    if (FT->isReferenceType())
      break;
    const clang::Type *T = FT->getUnqualifiedDesugaredType();
    if (isa<clang::BitIntType>(T) || isa<clang::EnumType>(T) ||
        (T->isBuiltinType() &&
         (T->isIntegerType() || T->isRealFloatingType())))
      result = field;
  } while (false);
  if (result) {
    // The record must have exactly the scalar's object representation: an
    // explicitly aligned member (`alignas(16) int v`) or anything else that
    // adds padding or alignment keeps the record an aggregate, so that
    // sizeof / alignof and the memory layout stay those of the record.
    clang::ASTContext &Ctx = CGM.getContext();
    clang::QualType RecTy = Ctx.getRecordType(RD);
    clang::QualType ScalarTy = result->getType();
    if (Ctx.getTypeSize(RecTy) != Ctx.getTypeSize(ScalarTy) ||
        Ctx.getTypeAlign(RecTy) != Ctx.getTypeAlign(ScalarTy))
      result = nullptr;
  }
  scalarWrapperCache[RD] = result;
  return result;
}

/// The reference `value` to a scalar-wrapper object, as the reference type
/// `nt` of a base or derived class of the same wrapper chain: the same
/// scalar, so only the memref shape / pointer type may need a cast.
static mlir::Value castScalarWrapperRef(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value value,
                                        mlir::Type nt) {
  if (value.getType() == nt)
    return value;
  auto smt = dyn_cast<MemRefType>(value.getType());
  auto dmt = dyn_cast<MemRefType>(nt);
  if (smt && dmt && smt.getElementType() == dmt.getElementType() &&
      smt.getRank() == dmt.getRank() &&
      smt.getMemorySpace() == dmt.getMemorySpace())
    return builder.create<memref::CastOp>(loc, dmt, value);
  auto spt = dyn_cast<LLVM::LLVMPointerType>(value.getType());
  auto dpt = dyn_cast<LLVM::LLVMPointerType>(nt);
  if (spt && dpt)
    return builder.create<LLVM::BitcastOp>(loc, dpt, value);
  return value;
}

/// Flattens the constant-evaluated initialiser `V` into attributes of the
/// scalar type `elemTy`, in memory order (array elements, then for a record
/// its bases and fields; a scalar wrapper contributes its one member).
static bool apValueToElements(const clang::APValue &V, mlir::Type elemTy,
                              SmallVectorImpl<mlir::Attribute> &out) {
  switch (V.getKind()) {
  case clang::APValue::Int: {
    auto it = dyn_cast<mlir::IntegerType>(elemTy);
    if (!it)
      return false;
    llvm::APSInt v = V.getInt();
    APInt r = v.isUnsigned() ? v.zextOrTrunc(it.getWidth())
                             : v.sextOrTrunc(it.getWidth());
    out.push_back(mlir::IntegerAttr::get(it, r));
    return true;
  }
  case clang::APValue::Float: {
    auto ft = dyn_cast<mlir::FloatType>(elemTy);
    if (!ft)
      return false;
    llvm::APFloat f = V.getFloat();
    bool lossy = false;
    f.convert(ft.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
              &lossy);
    out.push_back(mlir::FloatAttr::get(ft, f));
    return true;
  }
  case clang::APValue::Array: {
    for (unsigned i = 0, e = V.getArrayInitializedElts(); i < e; i++)
      if (!apValueToElements(V.getArrayInitializedElt(i), elemTy, out))
        return false;
    if (V.hasArrayFiller())
      for (unsigned i = V.getArrayInitializedElts(), e = V.getArraySize();
           i < e; i++)
        if (!apValueToElements(V.getArrayFiller(), elemTy, out))
          return false;
    return true;
  }
  case clang::APValue::Struct: {
    for (unsigned i = 0, e = V.getStructNumBases(); i < e; i++)
      if (!apValueToElements(V.getStructBase(i), elemTy, out))
        return false;
    for (unsigned i = 0, e = V.getStructNumFields(); i < e; i++)
      if (!apValueToElements(V.getStructField(i), elemTy, out))
        return false;
    return true;
  }
  default:
    return false;
  }
}

mlir::DenseElementsAttr
MLIRASTConsumer::evaluateConstantInitializer(const clang::VarDecl *VD,
                                             mlir::MemRefType mr) {
  if (!ScalarWrapperABI || !VD || !mr.hasStaticShape())
    return nullptr;
  const VarDecl *InitVD = nullptr;
  const Expr *AnyInit = VD->getAnyInitializer(InitVD);
  if (!AnyInit || !InitVD)
    return nullptr;
  clang::APValue *cv = InitVD->evaluateValue();
  if (!cv)
    return nullptr;
  SmallVector<mlir::Attribute> elts;
  if (!apValueToElements(*cv, mr.getElementType(), elts) ||
      (int64_t)elts.size() != mr.getNumElements())
    return nullptr;
  return DenseElementsAttr::get(
      RankedTensorType::get(mr.getShape(), mr.getElementType()), elts);
}

mlir::DenseElementsAttr
MLIRASTConsumer::evaluateXlxStringInitializer(const clang::Expr *init,
                                              clang::QualType type,
                                              mlir::MemRefType mr) {
  if (!ScalarWrapperABI || !cgeist::hls::xlxHeaders() || !init ||
      !mr.hasStaticShape())
    return nullptr;
  auto it = mr.getElementType().dyn_cast<mlir::IntegerType>();
  if (!it)
    return nullptr;
  clang::ASTContext &Ctx = CGM.getContext();
  SmallVector<APInt> elts;
  bool anyFolded = false;
  std::function<bool(const clang::Expr *, clang::QualType)> walk =
      [&](const clang::Expr *e, clang::QualType T) -> bool {
    if (!e)
      return false;
    e = e->IgnoreParens();
    if (auto *EWC = dyn_cast<ExprWithCleanups>(e))
      e = EWC->getSubExpr()->IgnoreParens();
    if (auto *IL = dyn_cast<InitListExpr>(e)) {
      if (const auto *AT = Ctx.getAsConstantArrayType(T)) {
        uint64_t n = AT->getSize().getZExtValue();
        for (uint64_t i = 0; i < n; ++i) {
          const clang::Expr *el = i < IL->getNumInits() ? IL->getInit(i)
                                                        : IL->getArrayFiller();
          if (!walk(el, AT->getElementType()))
            return false;
        }
        return true;
      }
      return IL->getNumInits() == 1 && walk(IL->getInit(0), T);
    }
    // A default-constructed ap object, or an array of them (`ap_fixed<16,5>
    // buf[N];`, the filler of `buf[N] = {0}`): zero in static storage (the
    // headers' default constructors are `= default` and set nothing, but
    // are not constexpr with the Xilinx bitwidth types, so clang's constant
    // evaluation gives up).
    if (const auto *AT = Ctx.getAsConstantArrayType(T)) {
      if (auto *CE = dyn_cast<CXXConstructExpr>(e);
          CE && CE->getNumArgs() == 0 &&
          CE->getConstructor()->isDefaultConstructor()) {
        uint64_t n = AT->getSize().getZExtValue();
        clang::QualType ET = AT->getElementType();
        while (const auto *inner = Ctx.getAsConstantArrayType(ET)) {
          n *= inner->getSize().getZExtValue();
          ET = inner->getElementType();
        }
        // The construction of every element (the element's own default
        // constructor).
        auto zero = cgeist::xlx::foldDefaultConstruct(e, Ctx);
        if (!zero || zero->getBitWidth() != it.getWidth())
          return false;
        for (uint64_t i = 0; i < n; ++i)
          elts.push_back(*zero);
        anyFolded = true;
        return true;
      }
    }
    if (auto v = cgeist::xlx::foldDefaultConstruct(e, Ctx)) {
      if (v->getBitWidth() != it.getWidth())
        return false;
      elts.push_back(*v);
      anyFolded = true;
      return true;
    }
    if (auto v = cgeist::xlx::foldStringConstruct(e, Ctx)) {
      if (v->getBitWidth() != it.getWidth())
        return false;
      elts.push_back(*v);
      anyFolded = true;
      return true;
    }
    // An ap type constructed from a number: the real headers' constructors
    // are not constexpr, so clang has no constant for it (OPEN_ISSUES O63);
    // the Vitis front end folds the constructor (its global initialiser).
    if (auto v = cgeist::xlx::foldNumericConstruct(e, Ctx)) {
      if (v->getBitWidth() != it.getWidth())
        return false;
      elts.push_back(*v);
      anyFolded = true;
      return true;
    }
    if (T->isIntegerType()) {
      clang::Expr::EvalResult R;
      if (!e->isValueDependent() && e->EvaluateAsInt(R, Ctx)) {
        llvm::APSInt v = R.Val.getInt();
        elts.push_back(v.isUnsigned() ? v.zextOrTrunc(it.getWidth())
                                      : v.sextOrTrunc(it.getWidth()));
        return true;
      }
    }
    return false;
  };
  if (!walk(init, type) || !anyFolded ||
      (int64_t)elts.size() != mr.getNumElements())
    return nullptr;
  return DenseElementsAttr::get(
      RankedTensorType::get(mr.getShape(), mr.getElementType()), elts);
}

/// The scalars of an (LLVM-typed) memref element `t`, all zero, in memory
/// order: a scalar, the elements of an array, the members of a struct.
static bool zeroScalars(mlir::Type t, SmallVectorImpl<mlir::Attribute> &out) {
  if (auto it = dyn_cast<mlir::IntegerType>(t)) {
    out.push_back(mlir::IntegerAttr::get(it, 0));
    return true;
  }
  if (auto ft = dyn_cast<mlir::FloatType>(t)) {
    out.push_back(mlir::FloatAttr::get(ft, 0.0));
    return true;
  }
  if (auto at = dyn_cast<mlir::LLVM::LLVMArrayType>(t)) {
    for (unsigned i = 0, n = at.getNumElements(); i < n; ++i)
      if (!zeroScalars(at.getElementType(), out))
        return false;
    return true;
  }
  if (auto st = dyn_cast<mlir::LLVM::LLVMStructType>(t)) {
    for (mlir::Type m : st.getBody())
      if (!zeroScalars(m, out))
        return false;
    return true;
  }
  return false;
}

/// The constant-evaluated value `V` of an object whose memref element type
/// is `t` (an llvm.struct / llvm.array of scalars, or a scalar, which a
/// scalar wrapper class stands for under --scalar-wrapper-abi), flattened
/// into scalar attributes in memory order.
static bool apValueToTyped(const clang::APValue &V, mlir::Type t,
                           SmallVectorImpl<mlir::Attribute> &out) {
  if (isa<mlir::IntegerType, mlir::FloatType>(t)) {
    const clang::APValue *v = &V;
    for (unsigned guard = 0; v->isStruct() && guard < 8; ++guard) {
      if (v->getStructNumFields() == 1 && v->getStructNumBases() == 0)
        v = &v->getStructField(0);
      else if (v->getStructNumBases() == 1 && v->getStructNumFields() == 0)
        v = &v->getStructBase(0);
      else
        return false;
    }
    if (!v->isInt() && !v->isFloat())
      return false;
    return apValueToElements(*v, t, out);
  }
  if (auto at = dyn_cast<mlir::LLVM::LLVMArrayType>(t)) {
    if (!V.isArray() || V.getArraySize() != at.getNumElements())
      return false;
    for (unsigned i = 0, n = at.getNumElements(); i < n; ++i) {
      const clang::APValue &e = i < V.getArrayInitializedElts()
                                    ? V.getArrayInitializedElt(i)
                                    : V.getArrayFiller();
      if (!apValueToTyped(e, at.getElementType(), out))
        return false;
    }
    return true;
  }
  if (auto st = dyn_cast<mlir::LLVM::LLVMStructType>(t)) {
    if (!V.isStruct() || V.getStructNumBases() != 0 ||
        V.getStructNumFields() != st.getBody().size())
      return false;
    for (auto [i, m] : llvm::enumerate(st.getBody()))
      if (!apValueToTyped(V.getStructField(i), m, out))
        return false;
    return true;
  }
  return false;
}

/// The initialiser `e` of an object of type `T` whose memref element type
/// is `t`, flattened like apValueToTyped, for the initialisers clang cannot
/// evaluate (--xlx-headers: ap members constructed from numbers or strings,
/// default-constructed ap members). `folded` is set when a leaf needed one
/// of those folds.
static bool initExprToTyped(const clang::Expr *e, clang::QualType T,
                            mlir::Type t, clang::ASTContext &Ctx,
                            SmallVectorImpl<mlir::Attribute> &out,
                            bool &folded) {
  if (!e || isa<clang::ImplicitValueInitExpr>(e->IgnoreParens()))
    return zeroScalars(t, out);
  e = e->IgnoreParens();
  if (auto *EWC = dyn_cast<ExprWithCleanups>(e))
    e = EWC->getSubExpr()->IgnoreParens();
  auto *IL = dyn_cast<InitListExpr>(e);
  if (auto it = dyn_cast<mlir::IntegerType>(t)) {
    std::optional<APInt> v = cgeist::xlx::foldNumericConstruct(e, Ctx);
    if (!v)
      v = cgeist::xlx::foldStringConstruct(e, Ctx);
    if (!v)
      v = cgeist::xlx::foldDefaultConstruct(e, Ctx);
    if (v) {
      if (v->getBitWidth() != it.getWidth())
        return false;
      out.push_back(mlir::IntegerAttr::get(it, *v));
      folded = true;
      return true;
    }
    if (IL && IL->getNumInits() == 1)
      return initExprToTyped(IL->getInit(0), T, t, Ctx, out, folded);
    clang::Expr::EvalResult R;
    if (T->isIntegerType() && !e->isValueDependent() &&
        e->EvaluateAsInt(R, Ctx)) {
      llvm::APSInt v = R.Val.getInt();
      out.push_back(mlir::IntegerAttr::get(
          it, v.isUnsigned() ? v.zextOrTrunc(it.getWidth())
                             : v.sextOrTrunc(it.getWidth())));
      return true;
    }
    return false;
  }
  if (auto ft = dyn_cast<mlir::FloatType>(t)) {
    if (IL && IL->getNumInits() == 1)
      return initExprToTyped(IL->getInit(0), T, t, Ctx, out, folded);
    llvm::APFloat f(0.0);
    if (!T->isRealFloatingType() || e->isValueDependent() ||
        !e->EvaluateAsFloat(f, Ctx))
      return false;
    bool lossy = false;
    f.convert(ft.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
              &lossy);
    out.push_back(mlir::FloatAttr::get(ft, f));
    return true;
  }
  if (auto at = dyn_cast<mlir::LLVM::LLVMArrayType>(t)) {
    const auto *CAT = Ctx.getAsConstantArrayType(T);
    if (!CAT || CAT->getSize().getZExtValue() != at.getNumElements() || !IL)
      return false;
    for (unsigned i = 0, n = at.getNumElements(); i < n; ++i) {
      const clang::Expr *el =
          i < IL->getNumInits() ? IL->getInit(i) : IL->getArrayFiller();
      if (!initExprToTyped(el, CAT->getElementType(), at.getElementType(),
                           Ctx, out, folded))
        return false;
    }
    return true;
  }
  if (auto st = dyn_cast<mlir::LLVM::LLVMStructType>(t)) {
    const auto *RD = T->getAsCXXRecordDecl();
    if (!RD || !IL || RD->getNumBases() != 0 || RD->isUnion())
      return false;
    SmallVector<const clang::FieldDecl *> fields(RD->fields());
    if (fields.size() != st.getBody().size() ||
        llvm::any_of(fields,
                     [](const clang::FieldDecl *f) { return f->isBitField(); }))
      return false;
    for (auto [i, f] : llvm::enumerate(fields)) {
      const clang::Expr *fi = i < IL->getNumInits() ? IL->getInit(i) : nullptr;
      if (!initExprToTyped(fi, f->getType(), st.getBody()[i], Ctx, out,
                           folded))
        return false;
    }
    return true;
  }
  return false;
}

mlir::ArrayAttr
MLIRASTConsumer::evaluateStructInitializer(const clang::VarDecl *VD,
                                           const clang::Expr *init,
                                           mlir::MemRefType mr) {
  if (!ScalarWrapperABI || !VD || !mr.hasStaticShape() ||
      !isa<mlir::LLVM::LLVMStructType>(mr.getElementType()))
    return nullptr;
  // The whole variable as nested arrays of the element struct.
  mlir::Type whole = mr.getElementType();
  for (int64_t d : llvm::reverse(mr.getShape()))
    whole = mlir::LLVM::LLVMArrayType::get(whole, d);
  clang::ASTContext &Ctx = CGM.getContext();
  // A scalar (not an array) variable: memref<1 x struct>.
  bool scalarVar = !Ctx.getAsConstantArrayType(VD->getType());
  SmallVector<mlir::Attribute> flat;
  const VarDecl *InitVD = nullptr;
  if (VD->getAnyInitializer(InitVD) && InitVD) {
    if (clang::APValue *cv = InitVD->evaluateValue()) {
      bool ok = scalarVar
                    ? apValueToTyped(*cv, mr.getElementType(), flat)
                    : apValueToTyped(*cv, whole, flat);
      if (ok)
        return mlir::ArrayAttr::get(mr.getContext(), flat);
      flat.clear();
    }
  }
  bool folded = false;
  bool ok = scalarVar ? initExprToTyped(init, VD->getType(),
                                        mr.getElementType(), Ctx, flat, folded)
                      : initExprToTyped(init, VD->getType(), whole, Ctx, flat,
                                        folded);
  if (!ok)
    return nullptr;
  return mlir::ArrayAttr::get(mr.getContext(), flat);
}

mlir::Attribute wrapIntegerMemorySpace(unsigned memorySpace, MLIRContext *ctx) {
  if (memorySpace == 0)
    return nullptr;

  return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), memorySpace);
}

MLIRScanner::MLIRScanner(MLIRASTConsumer &Glob,
                         mlir::OwningOpRef<mlir::ModuleOp> &module,
                         LowerToInfo &LTInfo)
    : Glob(Glob), module(module), builder(module->getContext()),
      ThisCapture(nullptr), LTInfo(LTInfo) {}

namespace {
/// Finds an access to a volatile-qualified object in a function body: a
/// glvalue expression of volatile-qualified type (reads, writes, compound
/// assignments, ++ / --, and conservatively also `&v` and unevaluated
/// operands). cgeist does not represent volatile on memref accesses, so the
/// function is marked instead (polygeist.volatile_access, Passes.h).
class VolatileAccessFinder
    : public clang::RecursiveASTVisitor<VolatileAccessFinder> {
public:
  bool found = false;
  bool VisitExpr(clang::Expr *e) {
    // Some expressions carry no type (e.g. a pack or an unresolved
    // placeholder in an uninstantiated context); they access nothing.
    clang::QualType t = e->getType();
    if (!t.isNull() && e->isGLValue() && t.isVolatileQualified()) {
      found = true;
      return false;
    }
    return true;
  }
};
} // namespace

/// Whether the body of `fd` (with a constructor's member initialisers)
/// accesses a volatile-qualified object (VolatileAccessFinder).
static bool accessesVolatile(const FunctionDecl *fd) {
  VolatileAccessFinder finder;
  if (auto *ctor = dyn_cast<CXXConstructorDecl>(fd))
    for (auto *init : ctor->inits())
      if (init->getInit() && !finder.found)
        finder.TraverseStmt(init->getInit());
  if (Stmt *body = fd->getBody(); body && !finder.found)
    finder.TraverseStmt(body);
  return finder.found;
}

void MLIRScanner::init(mlir::func::FuncOp function, const FunctionDecl *fd) {
  this->function = function;
  if (accessesVolatile(fd))
    function->setAttr(mlir::polygeist::kVolatileAccessAttrName,
                      mlir::UnitAttr::get(function->getContext()));
  this->EmittingFunctionDecl = fd;

  if (ShowAST) {
    llvm::errs() << "Emitting fn: " << function.getName() << "\n";
    llvm::errs() << *fd << "\n";
  }

  setEntryAndAllocBlock(function.addEntryBlock());

  {
    auto *MD = dyn_cast<CXXMethodDecl>(fd);
    cgeist::hls::enterFunction(fd, function,
                               /*argOffset=*/MD && MD->isInstance() ? 1 : 0);
  }

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
      ValueCategory(alloc, /*isReference*/ true)
          .store(getMLIRLocation(parm->getBeginLoc()), builder, val);
    }
    i++;
  }

  auto loc = getMLIRLocation(fd->getBeginLoc());
  if (fd->hasAttr<CUDAGlobalAttr>() && Glob.CGM.getLangOpts().CUDA &&
      !Glob.CGM.getLangOpts().CUDAIsDevice) {
    auto deviceStub =
        Glob.GetOrCreateMLIRFunction(fd, /* getDeviceStub */ true);
    builder.create<func::CallOp>(loc, deviceStub, function.getArguments());
    builder.create<func::ReturnOp>(loc);
    return;
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

          // Shift and cast down to the base type.
          // TODO: for complete types, this should be possible with a GEP.
          mlir::Value V = ThisVal.val;

          const clang::Type *BaseTypes[] = {BaseType};
          bool BaseVirtual[] = {BaseIsVirtual};

          V = GetAddressOfBaseClass(loc, V, /*derived*/ ClassDecl, BaseTypes,
                                    BaseVirtual);

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
        VisitArrayInitLoop(AILE, CommonFieldLookup(loc, CC->getThisObjectType(),
                                                   field, ThisVal.val,
                                                   /*isLValue*/ false));
        continue;
      }
      if (auto cons = dyn_cast<CXXConstructExpr>(expr->getInit())) {
        VisitConstructCommon(cons, /*name*/ nullptr, /*space*/ 0,
                             CommonFieldLookup(loc, CC->getThisObjectType(),
                                               field, ThisVal.val,
                                               /*isLValue*/ false)
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

      if (field->getType()->isReferenceType()) {
        assert(initexpr.isReference);
        initexpr.isReference = false;
        isArray = false;
      }

      auto cfl =
          CommonFieldLookup(loc, CC->getThisObjectType(), field, ThisVal.val,
                            /*isLValue*/ false);
      assert(cfl.val);
      cfl.store(loc, builder, initexpr, isArray);
    }
  }
  if (auto CC = dyn_cast<CXXDestructorDecl>(fd)) {
    CC->dump();
    llvm::errs() << " warning, destructor not fully handled yet\n";
  }

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back({builder.create<mlir::memref::AllocaOp>(loc, type),
                   builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().keepRunning);
  if (function.getFunctionType().getResults().size()) {
    auto type = mlir::MemRefType::get(
        {}, function.getFunctionType().getResult(0), {}, 0);
    returnVal = builder.create<mlir::memref::AllocaOp>(loc, type);
    if (type.getElementType().isa<mlir::IntegerType, mlir::FloatType>()) {
      builder.create<mlir::memref::StoreOp>(
          loc, builder.create<polygeist::UndefOp>(loc, type.getElementType()),
          returnVal, std::vector<mlir::Value>({}));
    }
  }

  if (auto D = dyn_cast<CXXMethodDecl>(fd)) {
    // ClangAST incorrectly does not contain the correct definition
    // of a union move operation and as such we _must_ emit a memcpy
    // for a defaulted union copy or move.
    if (D->getParent()->isUnion() && D->isDefaulted()) {
      mlir::Value V = ThisVal.val;
      assert(V);
      if (auto MT = dyn_cast<MemRefType>(V.getType())) {
        V = builder.create<polygeist::Memref2PointerOp>(
            loc, LLVM::LLVMPointerType::get(MT.getElementType()), V);
      }
      mlir::Value src = function.getArgument(1);
      if (auto MT = dyn_cast<MemRefType>(src.getType())) {
        src = builder.create<polygeist::Memref2PointerOp>(
            loc, LLVM::LLVMPointerType::get(MT.getElementType()), src);
      }
      mlir::Value typeSize = builder.create<polygeist::TypeSizeOp>(
          loc, builder.getIndexType(),
          mlir::TypeAttr::get(
              V.getType().cast<LLVM::LLVMPointerType>().getElementType()));
      typeSize = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                    typeSize);
      V = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              builder.getI8Type(),
              V.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          V);
      src = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              builder.getI8Type(),
              src.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          src);
      builder.create<LLVM::MemcpyOp>(loc, V, src, typeSize,
                                     /*isVolatile*/ false);
    }
  }

  Stmt *stmt = fd->getBody();
  if (stmt) {
    if (ShowAST) {
      stmt->dump();
    }
    Visit(stmt);

    loc = getMLIRLocation(stmt->getEndLoc());
  } else {
    loc = getMLIRLocation(fd->getEndLoc());
  }

  if (function.getFunctionType().getResults().size()) {
    mlir::Value vals[1] = {
        builder.create<mlir::memref::LoadOp>(loc, returnVal)};
    builder.create<func::ReturnOp>(loc, vals);
  } else
    builder.create<func::ReturnOp>(loc);

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
  auto varLoc =
      name ? getMLIRLocation(name->getBeginLoc()) : builder.getUnknownLoc();
  if (!isArray) {
    if (LLVMABI) {
      if (name)
        if (auto var = dyn_cast<VariableArrayType>(
                name->getType()->getUnqualifiedDesugaredType())) {
          auto len = Visit(var->getSizeExpr()).getValue(varLoc, builder);
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
            abuilder.create<arith::ConstantIntOp>(varLoc, 1, 64), 0);
        if (t.isa<mlir::IntegerType, mlir::FloatType>() && memspace == 0) {
          abuilder.create<LLVM::StoreOp>(
              varLoc, abuilder.create<polygeist::UndefOp>(varLoc, t), alloc);
        }
        // alloc = builder.create<mlir::LLVM::BitcastOp>(varLoc,
        // LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(t, 1)), alloc);
      }
    } else {
      mr = mlir::MemRefType::get(1, t, {}, memspace);
      alloc = abuilder.create<mlir::memref::AllocaOp>(varLoc, mr);
      if (memspace != 0) {
        alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
            varLoc,
            mlir::MemRefType::get(ShapedType::kDynamic, t, {}, memspace),
            abuilder.create<polygeist::Memref2PointerOp>(
                varLoc, LLVM::LLVMPointerType::get(t, 0), alloc));
      }
      alloc = abuilder.create<mlir::memref::CastOp>(
          varLoc, mlir::MemRefType::get(ShapedType::kDynamic, t, {}, 0), alloc);
      if (t.isa<mlir::IntegerType, mlir::FloatType>() && memspace == 0) {
        mlir::Value idxs[] = {abuilder.create<ConstantIndexOp>(varLoc, 0)};
        abuilder.create<mlir::memref::StoreOp>(
            varLoc, abuilder.create<polygeist::UndefOp>(varLoc, t), alloc,
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
        assert(shape[0] == ShapedType::kDynamic);
        mr = mlir::MemRefType::get(
            shape, mt.getElementType(), MemRefLayoutAttrInterface(),
            wrapIntegerMemorySpace(memspace, mt.getContext()));
        auto len = Visit(var->getSizeExpr()).getValue(varLoc, builder);
        len = builder.create<IndexCastOp>(varLoc, builder.getIndexType(), len);
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
      if (pshape == ShapedType::kDynamic)
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
          varLoc, mlir::MemRefType::get(shape, mt.getElementType()), alloc);
    }
  }
  assert(alloc);
  // NamedAttribute attrs[] = {NamedAttribute("name", name)};
  if (name) {
    if (params.find(name) != params.end()) {
      name->dump();
    }
    assert(params.find(name) == params.end());
    params[name] = ValueCategory(alloc, /*isReference*/ true);

    // Tag the underlying memref.alloca / llvm.alloca with the C variable name
    // (consumed by hls-est) and hand it to the HLS pragma machinery.
    mlir::Operation *defining = alloc.getDefiningOp();
    while (auto castOp =
               llvm::dyn_cast_or_null<mlir::memref::CastOp>(defining))
      defining = castOp.getSource().getDefiningOp();
    if (HLSAnnotate && defining &&
        (llvm::isa<mlir::memref::AllocaOp>(defining) ||
         llvm::isa<mlir::LLVM::AllocaOp>(defining))) {
      defining->setAttr(
          "polygeist.varname",
          mlir::StringAttr::get(builder.getContext(), name->getNameAsString()));

      // Preserve genuine on-chip buffers (multi-element arrays) as hardware
      // memory resources: this attr blocks store-to-load forwarding and
      // dead-alloca elimination in the Polygeist canonicalizer, so the buffer
      // survives for BRAM estimation. Scalars are deliberately left alone:
      // they must still mem2reg to SSA, so we do NOT tag them.
      if (auto memrefAlloca =
              llvm::dyn_cast<mlir::memref::AllocaOp>(defining)) {
        auto mt = llvm::cast<mlir::MemRefType>(memrefAlloca.getType());
        bool isBuffer = mt.hasStaticShape() && mt.getNumElements() > 1;
        if (isBuffer)
          defining->setAttr("hls.preserve",
                            mlir::UnitAttr::get(builder.getContext()));
      }
      cgeist::hls::onLocalStorage(name, defining);
    }
  }
  return alloc;
}

ValueCategory
MLIRScanner::VisitExtVectorElementExpr(clang::ExtVectorElementExpr *expr) {
  auto base = Visit(expr->getBase());

  SmallVector<uint32_t, 4> indices;
  expr->getEncodedElementAccess(indices);
  assert(indices.size() == 1 &&
         "The support for higher dimensions to be implemented.");

  assert(base.isReference);
  base.isReference = false;

  const auto et = base.val.getType();
  assert(et.isa<LLVM::LLVMPointerType>() || et.isa<MemRefType>());

  ValueCategory result = nullptr;
  const auto exprLoc = getMLIRLocation(expr->getExprLoc());
  const auto accLoc = getMLIRLocation(expr->getAccessorLoc());
  const mlir::Value idxs[2] = {
      builder.create<ConstantIntOp>(exprLoc, 0, 32),
      builder.create<ConstantIntOp>(exprLoc, indices[0], 32),
  };

  if (const auto pt = dyn_cast<LLVM::LLVMPointerType>(et)) {
    auto pt0 =
        pt.getElementType().cast<mlir::LLVM::LLVMArrayType>().getElementType();
    base.val = builder.create<mlir::LLVM::GEPOp>(
        exprLoc, mlir::LLVM::LLVMPointerType::get(pt0, pt.getAddressSpace()),
        base.val, idxs);

    result = ValueCategory(base.val, true);
  } else if (const auto mt = dyn_cast<MemRefType>(et)) {
    auto shape = std::vector<int64_t>(mt.getShape());

    if (shape.size() == 1) {
      shape[0] = ShapedType::kDynamic;
    } else {
      shape.erase(shape.begin());
    }

    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    base.val = builder.create<polygeist::SubIndexOp>(
        exprLoc, mt0, base.val, castToIndex(accLoc, idxs[0]));

    result = CommonArrayLookup(exprLoc, base, castToIndex(accLoc, idxs[1]),
                               base.isReference);
  } else {
    llvm_unreachable("Unexpected MLIR type received");
  }

  return result;
}

ValueCategory MLIRScanner::VisitConstantExpr(clang::ConstantExpr *expr) {
  auto sv = Visit(expr->getSubExpr());
  if (auto ty = dyn_cast<mlir::IntegerType>(getMLIRType(expr->getType()))) {
    if (expr->hasAPValueResult()) {
      return ValueCategory(
          builder.create<arith::ConstantOp>(
              getMLIRLocation(expr->getExprLoc()), ty,
              mlir::IntegerAttr::get(
                  ty, expr->getResultAsAPSInt().extOrTrunc(ty.getWidth()))),
          /*isReference*/ false);
    }
  }
  assert(sv.val);
  return sv;
}

ValueCategory MLIRScanner::VisitTypeTraitExpr(clang::TypeTraitExpr *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<arith::ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                           expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory
MLIRScanner::VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr *expr) {
  return ValueCategory(
      builder.create<mlir::LLVM::ZeroOp>(getMLIRLocation(expr->getExprLoc()),
                                         getMLIRType(expr->getType())),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitGNUNullExpr(clang::GNUNullExpr *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(builder.create<arith::ConstantIntOp>(
                           getMLIRLocation(expr->getExprLoc()), 0, ty),
                       /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitIntegerLiteral(clang::IntegerLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  // The literal's APInt has the width of its type; build the attribute from
  // it directly, so a value that does not fit in 64 bits (a _BitInt literal)
  // is kept as it is.
  return ValueCategory(
      builder.create<arith::ConstantOp>(
          getMLIRLocation(expr->getExprLoc()), ty,
          mlir::IntegerAttr::get(ty,
                                 expr->getValue().zextOrTrunc(ty.getWidth()))),
      /*isReference*/ false);
}

ValueCategory
MLIRScanner::VisitCharacterLiteral(clang::CharacterLiteral *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<arith::ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
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
  auto loc = getMLIRLocation(expr->getExprLoc());
  auto convertedType = getMLIRType(expr->getType());
  mlir::FloatType fty;
  if (auto mt = dyn_cast<MemRefType>(convertedType)) {
    auto elty = mt.getElementType();
    if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(elty)) {
      fty = ST.getBody()[0].cast<FloatType>();
    } else {
      fty = elty.cast<FloatType>();
    }
  } else if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(convertedType)) {
    fty = ST.getBody()[0].cast<FloatType>();
  } else {
    llvm_unreachable("unexpected complex type\n");
  }

  auto zero = builder.create<ConstantFloatOp>(
      loc, APFloat(fty.getFloatSemantics(), "0"), fty);
  auto imag = Visit(expr->getSubExpr()).getValue(loc, builder);
  return createComplexFloat(loc, zero, imag, expr->getType());
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
  auto loc = getMLIRLocation(decl->getExprLoc());

  if (auto FT = dyn_cast<mlir::FloatType>(Mty))
    return ValueCategory(builder.create<ConstantFloatOp>(
                             loc, APFloat(FT.getFloatSemantics(), "0"), FT),
                         /*isReference*/ false);
  if (auto IT = dyn_cast<mlir::IntegerType>(Mty))
    return ValueCategory(builder.create<ConstantIntOp>(loc, 0, IT),
                         /*isReference*/ false);
  if (auto MT = dyn_cast<mlir::MemRefType>(Mty))
    return ValueCategory(
        builder.create<polygeist::Pointer2MemrefOp>(
            loc, MT,
            builder.create<mlir::LLVM::ZeroOp>(
                loc, LLVM::LLVMPointerType::get(builder.getI8Type()))),
        false);
  if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(Mty))
    return ValueCategory(builder.create<mlir::LLVM::ZeroOp>(loc, PT), false);
  for (auto child : decl->children()) {
    child->dump();
  }
  decl->dump();
  llvm::errs() << " mty: " << Mty << "\n";
  llvm_unreachable("bad");
}

/// Whether clang's constant value `v` is all zeros (integers 0, floats +0.0,
/// arrays and structs of such).
static bool isZeroValue(const clang::APValue &v) {
  switch (v.getKind()) {
  case clang::APValue::Int:
    return v.getInt().isZero();
  case clang::APValue::Float:
    return v.getFloat().isPosZero();
  case clang::APValue::Array:
    for (unsigned i = 0, e = v.getArrayInitializedElts(); i < e; ++i)
      if (!isZeroValue(v.getArrayInitializedElt(i)))
        return false;
    return !v.hasArrayFiller() || isZeroValue(v.getArrayFiller());
  case clang::APValue::Struct:
    for (unsigned i = 0, e = v.getStructNumBases(); i < e; ++i)
      if (!isZeroValue(v.getStructBase(i)))
        return false;
    for (unsigned i = 0, e = v.getStructNumFields(); i < e; ++i)
      if (!isZeroValue(v.getStructField(i)))
        return false;
    return true;
  default:
    return false;
  }
}

/// Whether the braced initialiser `e` sets every scalar to zero (`= {0}`,
/// `= {}`), as clang's constant evaluation shows.
static bool isZeroInitializer(clang::Expr *e, clang::ASTContext &ctx) {
  if (isa<clang::ImplicitValueInitExpr>(e))
    return true;
  clang::Expr::EvalResult res;
  return !e->isValueDependent() && e->EvaluateAsRValue(res, ctx) &&
         !res.HasSideEffects && isZeroValue(res.Val);
}

/// Stores zero into every scalar of the aggregate `toInit` refers to, for an
/// aggregate member or element that a braced initialiser leaves without a
/// value (`window w = {};` with `struct window { unsigned char win[4][4]; }`,
/// clang's ImplicitValueInitExpr of the member's array type, W8b repro
/// r_local_struct_zero.cpp). `toInit` is a memref of scalars (a view whose
/// dynamic leading dimension selects one element, or the whole array) or an
/// LLVM pointer to an aggregate. Returns false when it is neither; else sets
/// `dense` to the splat zero of the scalars written into a memref (null for
/// an LLVM pointer), which a global's initial value is built from.
static bool zeroFillAggregate(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value toInit,
                              mlir::DenseElementsAttr &dense) {
  dense = mlir::DenseElementsAttr();
  if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
    mlir::Type et = mt.getElementType();
    if (!et.isa<mlir::IntegerType, mlir::FloatType>() || mt.getRank() == 0)
      return false;
    for (auto [d, n] : llvm::enumerate(mt.getShape()))
      if (n == ShapedType::kDynamic && d != 0)
        return false;
    mlir::TypedAttr zero = builder.getZeroAttr(et).cast<mlir::TypedAttr>();
    mlir::Value zeroV = builder.create<arith::ConstantOp>(loc, zero);
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    // A dynamic leading dimension is a view of one element: index 0 only.
    SmallVector<mlir::Value> idx;
    int64_t count = 1;
    mlir::OpBuilder::InsertionGuard guard(builder);
    for (int64_t n : mt.getShape()) {
      if (n == ShapedType::kDynamic) {
        idx.push_back(c0);
        continue;
      }
      count *= n;
      auto ub = builder.create<arith::ConstantIndexOp>(loc, n);
      auto loop = builder.create<scf::ForOp>(loc, c0, ub, c1);
      builder.setInsertionPointToStart(loop.getBody());
      idx.push_back(loop.getInductionVar());
    }
    builder.create<memref::StoreOp>(loc, zeroV, toInit, idx);
    // A view of one element contributes `count` scalars to its parent's
    // value; the whole array (static leading dimension) gets its own shape,
    // as a global's initial value needs.
    std::vector<int64_t> shape({count});
    if (mt.getShape()[0] != ShapedType::kDynamic)
      shape.assign(mt.getShape().begin(), mt.getShape().end());
    dense = SplatElementsAttr::get(RankedTensorType::get(shape, et), zero);
    return true;
  }
  if (auto pt = dyn_cast<LLVM::LLVMPointerType>(toInit.getType())) {
    mlir::Type et = pt.getElementType();
    if (!et || !et.isa<LLVM::LLVMArrayType, LLVM::LLVMStructType>())
      return false;
    builder.create<LLVM::StoreOp>(loc, builder.create<LLVM::ZeroOp>(loc, et),
                                  toInit);
    return true;
  }
  return false;
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
    if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
      inner = true;
    }
  }

  while (auto CO = toInit.getDefiningOp<memref::CastOp>())
    toInit = CO.getSource();

  // Recursively visit the initialization expression following the linear
  // increment of the memory address.
  std::function<mlir::DenseElementsAttr(Expr *, mlir::Value, bool)> helper =
      [&](Expr *expr, mlir::Value toInit,
          bool inner) -> mlir::DenseElementsAttr {
    Location loc = toInit.getLoc();
    // A member that a braced initialiser leaves out takes its default member
    // initialiser (`struct S { int n = 4; float w[2]; }; S s = {};`, C++11).
    // It is the initialiser's own expression, evaluated here; visiting the
    // CXXDefaultInitExpr itself needs a `this` the aggregate does not have
    // (W8b, class_ece8893_2023_spring FocusedACO: assertion `ThisVal.val`).
    if (auto *defaultInit = dyn_cast<clang::CXXDefaultInitExpr>(expr))
      expr = defaultInit->getExpr();
    // A braced initialiser of a scalar (`bool r{false}`, the default member
    // initialiser a braced aggregate initialiser leaves to a member) is its
    // one element: `struct P { int x{0}; bool r{false}; }; P p = {a};` hit
    // "unknown inner type" below with the member's scalar pointer
    // (OPEN_ISSUES O152, FocusedACO three-opt `Segment seg[3] = {...}`).
    if (auto *scalarList = dyn_cast<InitListExpr>(expr);
        scalarList && scalarList->getType()->isScalarType() &&
        scalarList->getNumInits() == 1 && !scalarList->hasArrayFiller())
      return helper(scalarList->getInit(0), toInit, inner);
    if (InitListExpr *initListExpr = dyn_cast<InitListExpr>(expr)) {
      // An all-zero braced initialiser of an array of scalars (`fm_t
      // buf[64][112][112] = {0};`) is one zero-filling loop nest, not one
      // store per element: element by element, a large array took cgeist
      // longer than 10 minutes (W8b: YOLOv4-Tiny, DSC_Acceleration).
      if (isa<MemRefType>(toInit.getType()) &&
          isZeroInitializer(initListExpr, Glob.CGM.getContext())) {
        mlir::DenseElementsAttr dense;
        if (zeroFillAggregate(builder, loc, toInit, dense))
          return dense;
      }

      if (inner) {
        if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
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
        if (auto MT = dyn_cast<MemRefType>(toInit.getType())) {
          auto shape = MT.getShape();
          assert(shape.size() > 0);
          assert(shape[0] != ShapedType::kDynamic);
          num = shape[0];
        } else if (auto PT =
                       dyn_cast<LLVM::LLVMPointerType>(toInit.getType())) {
          if (auto AT = dyn_cast<LLVM::LLVMArrayType>(PT.getElementType())) {
            num = AT.getNumElements();
          } else if (auto AT =
                         dyn_cast<LLVM::LLVMStructType>(PT.getElementType())) {
            num = AT.getBody().size();
          } else {
            toInit.getType().dump();
            llvm_unreachable(
                "TODO get number of values in array filler expression");
          }
        } else {
          toInit.getType().dump();
          llvm_unreachable(
              "TODO get number of values in array filler expression");
        }
      } else {
        num = initListExpr->getNumInits();
      }

      SmallVector<char> attrs;
      bool allSub = true;

      if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
        auto shape = std::vector<int64_t>(mt.getShape());
        // --scalar-wrapper-abi keeps a struct of one scalar member as that
        // scalar (memref<i32> for `struct V { int x; }`): its braced
        // initialiser `V{a.x + b.x}` initialises the scalar itself (before,
        // assert(0) below; OPEN_ISSUES O91, reproducer c_initlist.cpp).
        if (shape.empty() && !initListExpr->hasArrayFiller() &&
            initListExpr->getNumInits() == 1 &&
            mt.getElementType().isIntOrIndexOrFloat()) {
          mlir::Value v =
              Visit(initListExpr->getInit(0)).getValue(loc, builder);
          if (v.getType() == mt.getElementType()) {
            builder.create<memref::StoreOp>(loc, v, toInit, ValueRange());
            return mlir::DenseElementsAttr();
          }
        }
        if (shape.size() == 0) {
          auto ET = mt.getElementType();
          if (ET.isa<mlir::LLVM::LLVMStructType>() ||
              ET.isa<mlir::LLVM::LLVMArrayType>())
            toInit = builder.create<polygeist::Memref2PointerOp>(
                loc, LLVM::LLVMPointerType::get(ET, mt.getMemorySpaceAsInt()),
                toInit);
          // else if (ET.isa<MemRefType>())
          //   toInit = builder.create<memref::LoadOp>(loc, toInit);
          else
            assert(0);
        }
      }

      for (unsigned i = 0, e = num; i < e; ++i) {

        mlir::Value next;
        if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
          auto shape = std::vector<int64_t>(mt.getShape());
          assert(shape.size() > 0);
          shape[0] = ShapedType::kDynamic;
          auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                           MemRefLayoutAttrInterface(),
                                           mt.getMemorySpace());
          next = builder.create<polygeist::SubIndexOp>(loc, mt0, toInit,
                                                       getConstantIndex(i));
        } else {
          auto PT = toInit.getType().cast<LLVM::LLVMPointerType>();
          auto ET = PT.getElementType();
          mlir::Type nextType;
          if (auto ST = dyn_cast<LLVM::LLVMStructType>(ET))
            nextType = ST.getBody()[i];
          else if (auto AT = dyn_cast<LLVM::LLVMArrayType>(ET))
            nextType = AT.getElementType();
          else
            llvm_unreachable("unknown inner type");

          mlir::Value idxs[] = {
              builder.create<ConstantIntOp>(loc, 0, 32),
              builder.create<ConstantIntOp>(loc, i, 32),
          };
          next = builder.create<LLVM::GEPOp>(
              loc, LLVM::LLVMPointerType::get(nextType, PT.getAddressSpace()),
              toInit, idxs);
        }

        auto expr =
            (initListExpr->hasArrayFiller() && i >= initListExpr->getNumInits())
                ? initListExpr->getArrayFiller()
                : initListExpr->getInit(i);
        auto sub = helper(expr, next, true);
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
      if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
        std::vector<int64_t> shape(mt.getShape());
        assert(shape.size() > 0);
        shape[0] = num;
        return DenseElementsAttr::getFromRawBuffer(
            RankedTensorType::get(shape, mt.getElementType()), attrs);
      }
      return mlir::DenseElementsAttr();
    } else {
      // (--xlx-headers) An element constructed from a string literal is a
      // constant (XlxStringBits.cc); no constructor call or temporary.
      if (cgeist::hls::xlxHeaders() && ScalarWrapperABI)
        if (auto mt = dyn_cast<MemRefType>(toInit.getType()))
          if (auto et = dyn_cast<mlir::IntegerType>(mt.getElementType()))
            if (auto v = cgeist::xlx::foldStringConstruct(
                    expr, Glob.CGM.getContext());
                v && v->getBitWidth() == et.getWidth()) {
              mlir::Attribute attr = builder.getIntegerAttr(et, *v);
              ValueCategory(toInit, /*isReference*/ true)
                  .store(loc, builder,
                         builder.create<arith::ConstantOp>(
                             loc, et, attr.cast<mlir::IntegerAttr>()));
              return DenseElementsAttr::get(
                  RankedTensorType::get(std::vector<int64_t>({1}), et),
                  llvm::ArrayRef<mlir::Attribute>(attr));
            }
      bool isArray = false;
      Glob.getMLIRType(expr->getType(), &isArray);
      if (isa<clang::ImplicitValueInitExpr>(expr) &&
          (isArray || expr->getType()->isArrayType() ||
           expr->getType()->isRecordType())) {
        mlir::DenseElementsAttr dense;
        if (zeroFillAggregate(builder, loc, toInit, dense))
          return dense;
      }
      ValueCategory sub = Visit(expr);
      ValueCategory(toInit, /*isReference*/ true)
          .store(loc, builder, sub, isArray);
      if (!sub.isReference)
        if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
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

/// The initialiser of `decl`, with an ExprWithCleanups around a braced
/// initialiser removed: `fm_t buf[2][2][2] = {0}` with a class element type
/// wraps its InitListExpr in one (the element constructors bind temporaries),
/// which the init-list paths below would not see; visited as an expression it
/// became a rank-3 temporary that the array store cannot copy (W8b repro
/// r_init3d.cpp). The temporaries are trivially destructible scalars here.
static clang::Expr *initSkippingCleanups(const clang::VarDecl *decl) {
  auto *init = const_cast<clang::Expr *>(decl->getInit());
  if (auto *ewc = dyn_cast_or_null<clang::ExprWithCleanups>(init))
    if (isa<clang::InitListExpr>(ewc->getSubExpr()))
      return ewc->getSubExpr();
  return init;
}

ValueCategory MLIRScanner::VisitVarDecl(clang::VarDecl *decl) {
  decl = decl->getCanonicalDecl();
  auto varLoc = getMLIRLocation(decl->getBeginLoc());
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

  // (--scalar-wrapper-abi) A static local with constant initialisation
  // (C++ [basic.start.static], e.g. `static const ap_uint<12> lt[4] =
  // {...}`) is initialised before the program runs: its global gets clang's
  // evaluated value as dense initial value and nothing is emitted at run
  // time (no initialiser code, no guard flag).
  // (--xlx-headers) The same for an initialiser of string constructions
  // (`const static ap_ufixed<25,11> tbl[32] = {"0x1.0000p0", ...}` in the
  // Vitis math headers), which the Vitis front end folds at compile time
  // (XlxStringBits.cc).
  bool staticConstInit = false;
  if (ScalarWrapperABI && decl->isStaticLocal() && memtype == 0 &&
      decl->getInit() &&
      (decl->hasConstantInitialization() || cgeist::hls::xlxHeaders()) &&
      !Glob.getMLIRType(
               Glob.CGM.getContext().getPointerType(decl->getType()))
           .isa<mlir::LLVM::LLVMPointerType>()) {
    auto gv = Glob.GetOrCreateGlobal(
        decl, (function.getName() + "@static@").str(), /*tryInit*/ false);
    auto gmt = gv.first.getType().cast<MemRefType>();
    if (auto dense = decl->hasConstantInitialization()
                         ? Glob.evaluateConstantInitializer(decl, gmt)
                         : Glob.evaluateXlxStringInitializer(
                               decl->getInit(), decl->getType(), gmt)) {
      gv.first.setInitialValueAttr(dense);
      staticConstInit = true;
    }
  }

  if (auto init = initSkippingCleanups(decl)) {
    if (!staticConstInit && !isa<InitListExpr>(init) &&
        !isa<CXXConstructExpr>(init)) {
      auto visit = Visit(init);
      if (!visit.val) {
        decl->dump();
        assert(visit.val);
      }
      bool isReference = init->isLValue() || init->isXValue();
      if (isReference) {
        assert(visit.isReference);
        builder.create<polygeist::TrivialUseOp>(varLoc, visit.val);
        return params[decl] = visit;
      }
      if (isArray) {
        assert(visit.isReference);
        inite = visit;
      } else {
        inite = ValueCategory(visit.getValue(varLoc, builder), /*isRef*/ false);
        if (!inite.val) {
          init->dump();
          llvm_unreachable("?");
        }
        if (HLSAnnotate)
          cgeist::hls::onAssignedValue(decl, inite.val);
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
  mlir::Value staticInitTarget; // --scalar-wrapper-abi, see below

  Block *block = nullptr;
  Block::iterator iter;

  if (decl->isStaticLocal() && memtype == 0) {
    OpBuilder abuilder(builder.getContext());
    abuilder.setInsertionPointToStart(allocationScope);

    if (Glob.getMLIRType(Glob.CGM.getContext().getPointerType(decl->getType()))
            .isa<mlir::LLVM::LLVMPointerType>()) {
      op = abuilder.create<mlir::LLVM::AddressOfOp>(
          varLoc, Glob.GetOrCreateLLVMGlobal(
                      decl, (function.getName() + "@static@").str()));
    } else {
      auto gv = Glob.GetOrCreateGlobal(
          decl, (function.getName() + "@static@").str(), /*tryInit*/ false);
      op = abuilder.create<memref::GetGlobalOp>(varLoc, gv.first.getType(),
                                                gv.first.getName());
      // (--scalar-wrapper-abi) A run-time initialiser value of array type,
      // or an init list, is stored through the global's static shape: the
      // dynamic-shape view below does not match an array value, and an init
      // list with an array filler needs the static size. (A constructor
      // call keeps the dynamic-shape view as its `this`.)
      if (ScalarWrapperABI && decl->getInit())
        staticInitTarget = op;
      auto mt = gv.first.getType().cast<MemRefType>();
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = ShapedType::kDynamic;
      op = abuilder.create<memref::CastOp>(
          varLoc,
          MemRefType::get(shape, mt.getElementType(),
                          MemRefLayoutAttrInterface(), mt.getMemorySpace()),
          op);
    }
    params[decl] = ValueCategory(op, /*isReference*/ true);
    if (staticConstInit)
      return ValueCategory(op, /*isReference*/ true);
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
    ValueCategory(staticInitTarget && isArray ? staticInitTarget : op,
                  /*isReference*/ true)
        .store(varLoc, builder, inite, isArray);
  } else if (auto init = initSkippingCleanups(decl)) {
    if (isa<InitListExpr>(init)) {
      InitializeValueByInitListExpr(staticInitTarget ? staticInitTarget : op,
                                    init);
    } else if (auto CE = dyn_cast<CXXConstructExpr>(init)) {
      VisitConstructCommon(CE, decl, memtype, op);
    } else
      llvm_unreachable("unknown init list");
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

ValueCategory MLIRScanner::VisitCXXStdInitializerListExpr(
    clang::CXXStdInitializerListExpr *expr) {

  auto loc = getMLIRLocation(expr->getExprLoc());

  auto ArrayPtr = Visit(expr->getSubExpr());

  const ConstantArrayType *ArrayType =
      Glob.CGM.getContext().getAsConstantArrayType(
          expr->getSubExpr()->getType());
  assert(ArrayType && "std::initializer_list constructed from non-array");

  // FIXME: Perform the checks on the field types in SemaInit.
  RecordDecl *Record = expr->getType()->castAs<RecordType>()->getDecl();
  auto Field = Record->field_begin();

  mlir::Type subType = getMLIRType(expr->getType());

  mlir::Value res = builder.create<polygeist::UndefOp>(loc, subType);

  ArrayPtr = CommonArrayToPointer(loc, ArrayPtr);

  res = builder.create<LLVM::InsertValueOp>(loc, res.getType(), res,
                                            ArrayPtr.getValue(loc, builder),
                                            builder.getDenseI64ArrayAttr(0));
  Field++;
  auto iTy = getMLIRType(Field->getType()).cast<mlir::IntegerType>();
  res = builder.create<LLVM::InsertValueOp>(
      loc, res.getType(), res,
      builder.create<arith::ConstantIntOp>(
          loc, ArrayType->getSize().getZExtValue(), iTy.getWidth()),
      builder.getDenseI64ArrayAttr(1));
  return ValueCategory(res, /*isRef*/ false);
}

ValueCategory
MLIRScanner::VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *expr) {
  assert(arrayinit.size());
  auto loc = getMLIRLocation(expr->getExprLoc());
  return ValueCategory(builder.create<IndexCastOp>(
                           loc, getMLIRType(expr->getType()), arrayinit.back()),
                       /*isReference*/ false);
}

static const clang::ConstantArrayType *getCAT(const clang::Type *T) {
  const clang::Type *Child;
  if (auto CAT = dyn_cast<clang::ConstantArrayType>(T)) {
    return CAT;
  } else if (auto ET = dyn_cast<clang::ElaboratedType>(T)) {
    Child = ET->getNamedType().getTypePtr();
  } else if (auto TypeDefT = dyn_cast<clang::TypedefType>(T)) {
    Child = TypeDefT->getUnqualifiedDesugaredType();
  } else {
    llvm_unreachable("Unhandled case\n");
  }
  return getCAT(Child);
}

ValueCategory MLIRScanner::VisitArrayInitLoop(clang::ArrayInitLoopExpr *expr,
                                              ValueCategory tostore) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  const clang::ConstantArrayType *CAT = getCAT(expr->getType().getTypePtr());
  llvm::errs() << "warning recomputing common in arrayinitloopexpr\n";
  std::vector<mlir::Value> start = {getConstantIndex(0)};
  std::vector<mlir::Value> sizes = {
      getConstantIndex(CAT->getSize().getLimitedValue())};
  AffineMap map = builder.getSymbolIdentityMap();
  auto affineOp =
      builder.create<affine::AffineForOp>(loc, start, map, sizes, map);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&affineOp.getRegion().getBlocks().front());

  arrayinit.push_back(affineOp.getInductionVar());

  auto alu =
      CommonArrayLookup(loc, CommonArrayToPointer(loc, tostore),
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
    alu.store(loc, builder, val, isArray);
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
  return VisitCastExpr(expr);
}

ValueCategory
MLIRScanner::VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr) {
  return Visit(expr->getSubExpr());
}

ValueCategory MLIRScanner::VisitLambdaExpr(clang::LambdaExpr *expr) {

  // llvm::DenseMap<const VarDecl *, FieldDecl *> InnerCaptures;
  // FieldDecl *ThisCapture = nullptr;

  // expr->getLambdaClass()->getCaptureFields(InnerCaptures, ThisCapture);
  auto loc = getMLIRLocation(expr->getExprLoc());

  bool LLVMABI = false;
  mlir::Type t = Glob.getMLIRType(expr->getCallOperator()->getThisType());

  bool isArray = false;
  Glob.getMLIRType(expr->getCallOperator()->getThisObjectType(), &isArray);

  if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(t)) {
    LLVMABI = true;
    t = PT.getElementType();
  }
  if (auto mt = dyn_cast<MemRefType>(t)) {
    auto shape = std::vector<int64_t>(mt.getShape());
    if (!isArray) {
      t = mt.getElementType();
    }
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
              loc,
              cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(),
              field, ThisVal.val, /*isLValue*/ false);
          assert(CaptureKinds.find(VD) != CaptureKinds.end());
          if (CaptureKinds[VD] == LambdaCaptureKind::LCK_ByRef)
            result = result.dereference(loc, builder);
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
      CommonFieldLookup(loc, expr->getCallOperator()->getThisObjectType(),
                        field, op,
                        /*isLValue*/ false)
          .store(loc, builder, result, isArray);
    else {
      assert(CK == LambdaCaptureKind::LCK_ByRef);
      assert(result.isReference);

      auto val = result.val;

      if (auto mt = dyn_cast<MemRefType>(val.getType())) {
        auto shape = std::vector<int64_t>(mt.getShape());
        shape[0] = ShapedType::kDynamic;
        val = builder.create<memref::CastOp>(
            loc,
            MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace()),
            val);
      }

      CommonFieldLookup(loc, expr->getCallOperator()->getThisObjectType(),
                        field, op,
                        /*isLValue*/ false)
          .store(loc, builder, val);
    }
  }
  return ValueCategory(op, /*isReference*/ true);
}

// TODO actually deallocate
ValueCategory MLIRScanner::VisitMaterializeTemporaryExpr(
    clang::MaterializeTemporaryExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
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

  ValueCategory(op, /*isRefererence*/ true).store(loc, builder, v, isArray);
  return ValueCategory(op, /*isRefererence*/ true);
}

ValueCategory MLIRScanner::VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  expr->dump();
  llvm::errs() << "warning not calling destructor on delete\n";

  mlir::Value toDelete = Visit(expr->getArgument()).getValue(loc, builder);

  if (toDelete.getType().isa<mlir::MemRefType>()) {
    builder.create<mlir::memref::DeallocOp>(loc, toDelete);
  } else {
    mlir::Value args[1] = {
        builder.create<LLVM::BitcastOp>(loc, getOpaquePtr(), toDelete)};
    builder.create<mlir::LLVM::CallOp>(loc, Glob.GetOrCreateFreeFunction(),
                                       args);
  }

  return nullptr;
}
ValueCategory MLIRScanner::VisitCXXNewExpr(clang::CXXNewExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());

  mlir::Value count;

  if (expr->isArray()) {
    count = Visit(*expr->raw_arg_begin()).getValue(loc, builder);
    count = builder.create<IndexCastOp>(
        loc, mlir::IndexType::get(builder.getContext()), count);
  } else {
    count = getConstantIndex(1);
  }
  assert(count);

  auto ty = getMLIRType(expr->getType());

  mlir::Value alloc;
  mlir::Value arrayCons;
  if (!expr->placement_arguments().empty()) {
    mlir::Value val =
        Visit(*expr->placement_arg_begin()).getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(ty)) {
      if (auto mtin = dyn_cast<mlir::MemRefType>(val.getType())) {
        val = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(mtin.getElementType(),
                                       mtin.getMemorySpaceAsInt()),
            val);
      }
      arrayCons = alloc =
          builder.create<polygeist::Pointer2MemrefOp>(loc, mt, val);
    } else {
      arrayCons = alloc = builder.create<mlir::LLVM::BitcastOp>(loc, ty, val);
      auto PT = ty.cast<LLVM::LLVMPointerType>();
      if (expr->isArray())
        arrayCons = builder.create<mlir::LLVM::BitcastOp>(
            loc,
            LLVM::LLVMPointerType::get(
                LLVM::LLVMArrayType::get(PT.getElementType(), 0),
                PT.getAddressSpace()),
            alloc);
    }
  } else if (auto mt = dyn_cast<mlir::MemRefType>(ty)) {
    auto shape = std::vector<int64_t>(mt.getShape());
    mlir::Value args[1] = {count};
    arrayCons = alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
  } else {
    auto typeSize = getTypeSize(loc, expr->getAllocatedType());
    mlir::Value arg = builder.create<arith::MulIOp>(loc, typeSize, count);
    arrayCons = alloc = builder.create<mlir::LLVM::BitcastOp>(
        loc, ty, Glob.CallMalloc(builder, loc, arg));
    auto PT = ty.cast<LLVM::LLVMPointerType>();
    if (expr->isArray())
      arrayCons = builder.create<mlir::LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              LLVM::LLVMArrayType::get(PT.getElementType(), 0),
              PT.getAddressSpace()),
          alloc);
  }
  assert(alloc);
  if (expr->hasInitializer()) {
    auto init = expr->getInitializer();
    if (isa<InitListExpr>(init)) {
      (void)InitializeValueByInitListExpr(alloc, init);
    } else if (isa<CXXConstructExpr>(init)) {
      assert(arrayCons);
      VisitConstructCommon(
          const_cast<CXXConstructExpr *>(expr->getConstructExpr()),
          /*name*/ nullptr, /*memtype*/ 0, arrayCons, count);
    } else {
      ValueCategory val = Visit(init);
      ValueCategory(alloc, /* isReference */ true)
          .store(loc, builder, val, /* isArray */ false);
    }
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

/// Whether the sign-dependent ops emitted for a value of clang type `qt`
/// (extsi/extui, shrsi/shrui, divsi/divui, cmpi s*/u*, sitofp/uitofp,
/// index_cast/index_castui, ...) are signed. Decided on the canonical type,
/// so a typedef or alias template (uint32_t, size_t, ap_uint<W>) and an
/// unsigned _BitInt are seen through; bool and an enum with an unsigned
/// underlying type count as unsigned, as in clang. Non-integer types default
/// to signed.
static bool isSignedIntegerQualType(clang::QualType qt) {
  return !qt->isUnsignedIntegerType();
}

/// The type of the operand `e` of an arithmetic operator as written, before
/// the integer promotions and the usual arithmetic conversions.
static clang::QualType operandTypeAsWritten(const clang::Expr *e) {
  return e->IgnoreParenImpCasts()->getType();
}

/// Marks `v` polygeist.nsw (Passes.h) when it is an arith.addi / subi / muli
/// that cgeist built in the width of `compQT` for a C integer +, -, * (or
/// unary minus, `rhsQT` null) computed in `compQT` on operands written with
/// types `lhsQT` / `rhsQT`, and that operation is undefined when its exact
/// result does not fit: `compQT` is a signed integer type at least as wide as
/// int (C11 6.5p5, C++ [expr.pre]p4) and the program is not compiled with
/// -fwrapv. Unsigned arithmetic, which wraps, is not marked, nor is arithmetic
/// on an operand promoted from a type narrower than int (char, short, bool,
/// an enum, a narrow _BitInt), nor an op built in a narrower type than
/// `compQT` (the `+=` / `-=` of a narrow LHS). AffineCFG raises a marked op to
/// an affine expression without proving that it keeps the value. Returns `v`.
static mlir::Value markNoSignedWrap(clang::CodeGen::CodeGenModule &CGM,
                                    mlir::Value v, clang::QualType compQT,
                                    clang::QualType lhsQT,
                                    clang::QualType rhsQT = clang::QualType()) {
  mlir::Operation *op = v.getDefiningOp();
  if (!isa_and_nonnull<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op))
    return v;
  clang::ASTContext &ctx = CGM.getContext();
  if (CGM.getLangOpts().isSignedOverflowDefined() ||
      !compQT->isSignedIntegerType())
    return v;
  unsigned intWidth = ctx.getIntWidth(ctx.IntTy);
  unsigned compWidth = ctx.getIntWidth(compQT);
  auto it = dyn_cast<mlir::IntegerType>(v.getType());
  if (compWidth < intWidth || !it || it.getWidth() != compWidth)
    return v;
  auto promotedFromNarrow = [&](clang::QualType t) {
    if (t.isNull())
      return false;
    if (!t->isIntegerType() || t->isEnumeralType())
      return true;
    return ctx.isPromotableIntegerType(t) || ctx.getIntWidth(t) < intWidth;
  };
  if (promotedFromNarrow(lhsQT) || promotedFromNarrow(rhsQT))
    return v;
  op->setAttr(mlir::polygeist::kNoSignedWrapAttrName,
              mlir::UnitAttr::get(op->getContext()));
  return v;
}

mlir::Value MLIRScanner::castToIndex(mlir::Location loc, mlir::Value val,
                                     bool isUnsigned) {
  assert(val && "Expect non-null value");

  if (auto op = val.getDefiningOp<ConstantIntOp>()) {
    // A constant index is extended to the index width as its C type says
    // (`ap_uint<8> j = 200` is 200, not -56), when it fits.
    APInt v = op.getValue().cast<IntegerAttr>().getValue();
    if (isUnsigned ? v.getActiveBits() < 64 : v.getSignificantBits() <= 64)
      return getConstantIndex(isUnsigned ? (int64_t)v.getZExtValue()
                                         : v.getSExtValue());
  }

  if (isUnsigned)
    return builder.create<arith::IndexCastUIOp>(
        loc, mlir::IndexType::get(val.getContext()), val);
  return builder.create<arith::IndexCastOp>(
      loc, mlir::IndexType::get(val.getContext()), val);
}

ValueCategory
MLIRScanner::VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());

  bool isArray = false;
  mlir::Type melem = Glob.getMLIRType(expr->getType(), &isArray);
  assert(!isArray);

  if (melem.isa<mlir::IntegerType>())
    return ValueCategory(builder.create<ConstantIntOp>(loc, 0, melem), false);
  else if (auto MT = dyn_cast<mlir::MemRefType>(melem))
    return ValueCategory(
        builder.create<polygeist::Pointer2MemrefOp>(
            loc, MT,
            builder.create<mlir::LLVM::ZeroOp>(
                loc, LLVM::LLVMPointerType::get(builder.getI8Type()))),
        false);
  else if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(melem))
    return ValueCategory(builder.create<mlir::LLVM::ZeroOp>(loc, PT), false);
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
  const clang::FieldDecl *wrapperField = Glob.getScalarWrapperField(
      Glob.CGM.getContext().getBaseElementType(cons->getType())->getAsRecordDecl());
  // (--xlx-headers) A scalar wrapper constructed from a string literal
  // (`ap_fixed<16,5> c = "1.5"`): the value _ssdm_string2bits would give,
  // folded at compile time as by the Vitis front end (XlxStringBits.cc).
  if (cgeist::hls::xlxHeaders() && wrapperField &&
      !Glob.CGM.getContext().getAsArrayType(cons->getType()))
    if (auto mt = dyn_cast<MemRefType>(op.getType()))
      if (auto et = dyn_cast<mlir::IntegerType>(mt.getElementType()))
        if (auto v = cgeist::xlx::foldStringConstruct(
                cons, Glob.CGM.getContext());
            v && v->getBitWidth() == et.getWidth()) {
          ValueCategory(op, /*isReference*/ true)
              .store(loc, builder,
                     builder.create<arith::ConstantOp>(
                         loc, et, builder.getIntegerAttr(et, *v)));
          return ValueCategory(op, /*isReference*/ true);
        }
  if (cons->requiresZeroInitialization() && wrapperField &&
      op.getType().isa<MemRefType>()) {
    // Scalar wrapper(s): plain zero stores instead of a memset through a
    // pointer (which the HLS analyses cannot read), one per object
    // constructed here (not per element of the memref `op` refers into:
    // `this` of a base or member subobject is memref<?xT>).
    storeZeroToWrapperObjects(loc, op, cons->getType(), count);
  } else if (cons->requiresZeroInitialization()) {
    mlir::Value val = op;
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
    mlir::Value size = getTypeSize(loc, cons->getType());

    auto i8_0 = builder.create<ConstantIntOp>(loc, 0, 8);
    auto sizev =
        builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), size);

    auto falsev = builder.create<ConstantIntOp>(loc, false, 1);
    builder.create<LLVM::MemsetOp>(loc, val, i8_0, sizev, falsev);
  }

  if (decl->isTrivial() && decl->isDefaultConstructor())
    return ValueCategory(op, /*isReference*/ true);

  // A trivial copy / move of a scalar wrapper is a load and a store.
  if (wrapperField && decl->isTrivial() &&
      (decl->isCopyConstructor() || decl->isMoveConstructor()) &&
      cons->getNumArgs() == 1 &&
      !Glob.CGM.getContext().getAsArrayType(cons->getType())) {
    ValueCategory src = Visit(cons->getArg(0));
    assert(src.val && "copy constructor source");
    ValueCategory(op, /*isReference*/ true)
        .store(loc, builder, src.getValue(loc, builder));
    return ValueCategory(op, /*isReference*/ true);
  }

  mlir::Block::iterator oldpoint;
  mlir::Block *oldblock;
  ValueCategory endobj(op, /*isReference*/ true);

  ValueCategory obj(op, /*isReference*/ true);
  QualType innerType = cons->getType();
  if (!ScalarWrapperABI) {
    // Unchanged lowering (option off): one loop over the outermost dimension.
    if (auto arrayType =
            Glob.CGM.getContext().getAsArrayType(cons->getType())) {
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

      builder.setInsertionPointToStart(&forOp.getRegion().getBlocks().front());
      assert(obj.isReference);
      obj = CommonArrayToPointer(loc, obj);
      obj = CommonArrayLookup(loc, obj, forOp.getInductionVar(),
                              /*isImplicitRef*/ false, /*removeIndex*/ false);
      assert(obj.isReference);
    }
  } else if (Glob.CGM.getContext().getAsArrayType(cons->getType())) {
    // One loop per array dimension, so that the constructor's `this` is the
    // element reference (a constant array of class objects `T a[N][M]` was
    // only looped over its outermost dimension before).
    oldpoint = builder.getInsertionPoint();
    oldblock = builder.getInsertionBlock();
    bool outermost = true;
    while (auto *AT = Glob.CGM.getContext().getAsArrayType(innerType)) {
      innerType = AT->getElementType();
      mlir::Value size;
      if (outermost && count)
        size = count;
      else {
        auto CAT = cast<clang::ConstantArrayType>(AT);
        size = getConstantIndex(CAT->getSize().getLimitedValue());
      }
      auto forOp = builder.create<scf::ForOp>(loc, getConstantIndex(0), size,
                                              getConstantIndex(1));
      builder.setInsertionPointToStart(&forOp.getRegion().getBlocks().front());
      assert(obj.isReference);
      bool last = !Glob.CGM.getContext().getAsArrayType(innerType);
      obj = CommonArrayToPointer(loc, obj);
      obj = CommonArrayLookup(loc, obj, forOp.getInductionVar(),
                              /*isImplicitRef*/ false, /*removeIndex*/ !last);
      assert(obj.isReference);
      outermost = false;
    }
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

void MLIRScanner::storeZeroToWrapperObjects(mlir::Location loc,
                                            mlir::Value op, clang::QualType T,
                                            mlir::Value count) {
  auto &Ctx = Glob.CGM.getContext();
  OpBuilder::InsertionGuard guard(builder);
  // One loop per array dimension of the constructed type (the outermost one
  // runs `count` times for `new T[n]()`), as for the constructor calls, so
  // the store goes to exactly the objects being value-initialised.
  ValueCategory obj(op, /*isReference*/ true);
  bool outermost = true;
  while (auto *AT = Ctx.getAsArrayType(T)) {
    T = AT->getElementType();
    mlir::Value size;
    if (outermost && count)
      size = count;
    else
      size = getConstantIndex(
          cast<clang::ConstantArrayType>(AT)->getSize().getLimitedValue());
    auto forOp = builder.create<scf::ForOp>(loc, getConstantIndex(0), size,
                                            getConstantIndex(1));
    builder.setInsertionPointToStart(forOp.getBody());
    bool last = !Ctx.getAsArrayType(T);
    obj = CommonArrayToPointer(loc, obj);
    obj = CommonArrayLookup(loc, obj, forOp.getInductionVar(),
                            /*isImplicitRef*/ false, /*removeIndex*/ !last);
    outermost = false;
  }
  mlir::Type et = getMLIRType(T); // the wrapper's scalar
  mlir::Value zero;
  if (auto it = dyn_cast<mlir::IntegerType>(et))
    zero = builder.create<ConstantIntOp>(loc, 0, it);
  else if (auto ft = dyn_cast<mlir::FloatType>(et))
    zero = builder.create<ConstantFloatOp>(
        loc, APFloat::getZero(ft.getFloatSemantics()), ft);
  else
    llvm_unreachable("zero store of a non-scalar wrapper");
  obj.store(loc, builder, zero);
}

mlir::Value MLIRScanner::tryEmitConstVarValue(mlir::Location loc,
                                              const VarDecl *VD,
                                              clang::QualType resultType) {
  if (!ScalarWrapperABI || !VD || isa<ParmVarDecl>(VD))
    return nullptr;
  clang::QualType VT = VD->getType();
  if (!VT.isConstQualified() || VT.isVolatileQualified())
    return nullptr;
  const VarDecl *InitVD = nullptr;
  const Expr *Init = VD->getAnyInitializer(InitVD);
  if (!Init || !InitVD)
    return nullptr;
  // Only an initialiser clang evaluated as a constant expression (static
  // members, globals, constexpr; a non-constant `const` local stays a load).
  if (!InitVD->hasConstantInitialization() &&
      !InitVD->isUsableInConstantExpressions(Glob.CGM.getContext()))
    return nullptr;
  clang::APValue *val = InitVD->evaluateValue();
  if (!val)
    return nullptr;
  mlir::Type ty = getMLIRType(resultType);
  if (!ty.isa<mlir::IntegerType, mlir::FloatType>())
    return nullptr;
  SmallVector<mlir::Attribute> elts;
  if (!apValueToElements(*val, ty, elts) || elts.size() != 1)
    return nullptr;
  if (auto ia = dyn_cast<mlir::IntegerAttr>(elts[0]))
    return builder.create<arith::ConstantOp>(loc, ty, ia);
  if (auto fa = dyn_cast<mlir::FloatAttr>(elts[0]))
    return builder.create<arith::ConstantOp>(loc, ty, fa);
  return nullptr;
}

ValueCategory MLIRScanner::CommonArrayToPointer(mlir::Location loc,
                                                ValueCategory scalar) {
  assert(scalar.val);
  assert(scalar.isReference);
  if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(scalar.val.getType())) {
    if (PT.getElementType().isa<mlir::LLVM::LLVMPointerType>())
      return ValueCategory(scalar.val, /*isRef*/ false);
    mlir::Value vec[2] = {builder.create<ConstantIntOp>(loc, 0, 32),
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
            scalar.val, vec),
        /*isReference*/ false);
  }

  auto mt = scalar.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  // if (shape.size() > 1) {
  //  shape.erase(shape.begin());
  //} else {
  shape[0] = ShapedType::kDynamic;
  //}
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());

  auto post = builder.create<memref::CastOp>(loc, mt0, scalar.val);
  return ValueCategory(post, /*isReference*/ false);
}

ValueCategory MLIRScanner::CommonArrayLookup(mlir::Location loc,
                                             ValueCategory array,
                                             mlir::Value idx,
                                             bool isImplicitRefResult,
                                             bool removeIndex) {
  mlir::Value val = array.getValue(loc, builder);
  assert(val);

  if (val.getType().isa<LLVM::LLVMPointerType>()) {

    mlir::Value vals[] = {
        builder.create<IndexCastOp>(loc, builder.getIntegerType(64), idx)};
    // TODO sub
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(loc, val.getType(), val, vals),
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
    shape[0] = ShapedType::kDynamic;
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
    shape[0] = ShapedType::kDynamic;
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
  auto loc = getMLIRLocation(expr->getExprLoc());
  auto moo = Visit(expr->getLHS());

  auto rhs = Visit(expr->getRHS()).getValue(loc, builder);
  // Check the RHS has been successfully emitted
  assert(rhs);
  auto idx = castToIndex(getMLIRLocation(expr->getRBracketLoc()), rhs,
                         !isSignedIntegerQualType(expr->getRHS()->getType()));
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
  return CommonArrayLookup(loc, moo, idx, isArray);
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
  } else if (auto UOp = dyn_cast<clang::UnaryOperator>(E)) {
    if (UOp->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
      return EmitCallee(UOp->getSubExpr());
    }
  }

  return nullptr;
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitBuiltinOps(clang::CallExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  // The name-based lowering below is for the C math library; an overload
  // of the same name on another type (hls::exp on an ap_fixed class)
  // stays a call (--scalar-wrapper-abi; unchanged when the option is off).
  bool floatArgs = true;
  if (ScalarWrapperABI) {
    floatArgs = expr->getNumArgs() > 0;
    for (auto *a : expr->arguments())
      floatArgs &= a->getType()->isRealFloatingType();
  }
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee())) {
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (floatArgs && sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__log2f") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::Log2Op>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (floatArgs && sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "log") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::LogOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (floatArgs && sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "ceil")) {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<math::CeilOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (floatArgs && sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "expf" ||
           sr->getDecl()->getName() == "exp")) {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::ExpOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (floatArgs && sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "sin") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::SinOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }

      if (floatArgs && sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "cos") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
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
      if (sr->getDecl()->getIdentifier() && CudaLower &&
          sr->getDecl()->getName() == "cudaFuncSetCacheConfig") {
        llvm::errs() << " Not emitting GPU option: cudaFuncSetCacheConfig\n";
        return make_pair(ValueCategory(), true);
      }
      // TODO move free out.
      std::string str;
      llvm::raw_string_ostream ss(str);
      ss.str();
      sr->getDecl()->printQualifiedName(ss);
      if (str == "free" || ((CudaLower && ToCPU.size() > 0) &&
                            (str == "cudaFree" || str == "cudaFreeHost"))) {

        auto sub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(sub))
          sub = BC->getSubExpr();
        mlir::Value arg = Visit(sub).getValue(loc, builder);

        if (arg.getType().isa<mlir::LLVM::LLVMPointerType>()) {
          auto callee = EmitCallee(expr->getCallee());
          auto strcmpF = Glob.GetOrCreateLLVMFunction(callee);
          mlir::Value args[] = {
              builder.create<LLVM::BitcastOp>(loc, getOpaquePtr(), arg)};
          builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args);
        } else {
          assert(arg.getType().isa<MemRefType>());
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
      if ((CudaLower && ToCPU.size() > 0) &&
          (str == "cudaMalloc" || str == "cudaMallocHost" ||
           str == "cudaMallocPitch")) {
        auto sub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(sub))
          sub = BC->getSubExpr();
        {
          auto dst = Visit(sub).getValue(loc, builder);
          if (auto omt = dyn_cast<MemRefType>(dst.getType())) {
            if (auto mt = dyn_cast<MemRefType>(omt.getElementType())) {
              auto shape = std::vector<int64_t>(mt.getShape());

              auto elemSize = getTypeSize(
                  loc, cast<clang::PointerType>(
                           cast<clang::PointerType>(
                               sub->getType()->getUnqualifiedDesugaredType())
                               ->getPointeeType())
                           ->getPointeeType());
              mlir::Value allocSize;
              if (str == "cudaMallocPitch") {
                mlir::Value width =
                    Visit(expr->getArg(2)).getValue(loc, builder);
                mlir::Value height =
                    Visit(expr->getArg(3)).getValue(loc, builder);
                // Not changing pitch from provided width here
                // TODO can consider addition alignment considerations
                Visit(expr->getArg(1))
                    .dereference(loc, builder)
                    .store(loc, builder, width);
                allocSize = builder.create<MulIOp>(loc, width, height);
              } else
                allocSize = Visit(expr->getArg(1)).getValue(loc, builder);
              auto idxType = mlir::IndexType::get(builder.getContext());
              mlir::Value args[1] = {builder.create<DivUIOp>(
                  loc, builder.create<IndexCastOp>(loc, idxType, allocSize),
                  elemSize)};
              auto alloc = builder.create<mlir::memref::AllocOp>(
                  loc,
                  (str != "cudaMallocHost" && !CudaLower)
                      ? mlir::MemRefType::get(shape, mt.getElementType(),
                                              MemRefLayoutAttrInterface())
                      : mt,
                  args);
              mlir::Value allocv = alloc;
              allocv = builder.create<polygeist::Memref2PointerOp>(
                  loc, LLVM::LLVMPointerType::get(mt.getElementType(), 1),
                  allocv);
              allocv =
                  builder.create<polygeist::Pointer2MemrefOp>(loc, mt, allocv);
              ValueCategory(dst, /*isReference*/ true)
                  .store(loc, builder, allocv);
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

    auto createBlockIdOp = [&](gpu::Dimension str,
                               mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::BlockIdOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    auto createBlockDimOp = [&](gpu::Dimension str,
                                mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::BlockDimOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    auto createThreadIdOp = [&](gpu::Dimension str,
                                mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::ThreadIdOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    auto createGridDimOp = [&](gpu::Dimension str,
                               mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::GridDimOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    if (auto ME = dyn_cast<MemberExpr>(ic->getSubExpr())) {
      auto memberName = ME->getMemberDecl()->getName();

      if (auto sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
        if (auto sr = dyn_cast<DeclRefExpr>(sr2->getSourceExpr())) {
          if (sr->getDecl()->getName() == "blockIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(
                  ValueCategory(createBlockIdOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(
                  ValueCategory(createBlockIdOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(
                  ValueCategory(createBlockIdOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (sr->getDecl()->getName() == "blockDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(
                  ValueCategory(createBlockDimOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(
                  ValueCategory(createBlockDimOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(
                  ValueCategory(createBlockDimOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (sr->getDecl()->getName() == "threadIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(
                  ValueCategory(createThreadIdOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(
                  ValueCategory(createThreadIdOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(
                  ValueCategory(createThreadIdOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (sr->getDecl()->getName() == "gridDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(
                  ValueCategory(createGridDimOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(
                  ValueCategory(createGridDimOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(
                  ValueCategory(createGridDimOp(gpu::Dimension::z, mlirType),
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

mlir::Value MLIRScanner::getConstantIndex(int64_t x) {
  if (constants.find(x) != constants.end()) {
    return constants[x];
  }
  mlir::OpBuilder subbuilder(builder.getContext());
  subbuilder.setInsertionPointToStart(entryBlock);
  return constants[x] =
             subbuilder.create<ConstantIndexOp>(subbuilder.getUnknownLoc(), x);
}

ValueCategory MLIRScanner::VisitMSPropertyRefExpr(MSPropertyRefExpr *expr) {
  llvm_unreachable("unhandled ms propertyref");
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
    mlir::Value val = sub.getValue(loc, builder);

    if (auto MT = dyn_cast<mlir::MemRefType>(val.getType())) {
      val = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(MT.getElementType(),
                                     MT.getMemorySpaceAsInt()),
          val);
    }
    auto postTy = getMLIRType(U->getType()).cast<mlir::IntegerType>();

    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(val.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
      mlir::Value ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::eq, val, nullptr_llvm);
      if (postTy.getWidth() > 1)
        ne = builder.create<arith::ExtUIOp>(loc, postTy, ne);
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

    if (postTy.getWidth() > 1)
      res = builder.create<arith::ExtUIOp>(loc, postTy, res);
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Not: {
    assert(sub.val);
    mlir::Value val = sub.getValue(loc, builder);

    if (!val.getType().isa<mlir::IntegerType>()) {
      U->dump();
      val.dump();
    }
    auto ty = val.getType().cast<mlir::IntegerType>();
    // The mask is built from the full-width APInt: ConstantIntOp(int64_t, iN)
    // zero-extends the 64-bit -1, so wider integers (ap_int<W>, W > 64)
    // would only have their low 64 bits inverted.
    auto c1 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(ty, APInt::getAllOnes(ty.getWidth())));
    return ValueCategory(builder.create<XOrIOp>(loc, val, c1),
                         /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Deref: {
    auto dref = sub.dereference(loc, builder);
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
    shape[0] = ShapedType::kDynamic;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    res = builder.create<memref::CastOp>(loc, mt0, sub.val);
    return ValueCategory(res,
                         /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Plus: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_Minus: {
    mlir::Value val = sub.getValue(loc, builder);
    auto ty = val.getType();
    if (auto ft = dyn_cast<mlir::FloatType>(ty)) {
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
          markNoSignedWrap(
              Glob.CGM,
              builder.create<SubIOp>(loc,
                                     builder.create<ConstantIntOp>(
                                         loc, 0, ty.cast<mlir::IntegerType>()),
                                     val),
              U->getType(), operandTypeAsWritten(U->getSubExpr())),
          /*isReference*/ false);
    }
  }
  case clang::UnaryOperator::Opcode::UO_PreInc:
  case clang::UnaryOperator::Opcode::UO_PostInc: {
    assert(sub.isReference);
    auto prev = sub.getValue(loc, builder);
    auto ty = prev.getType();

    mlir::Value next;
    if (auto ft = dyn_cast<mlir::FloatType>(ty)) {
      if (prev.getType() != ty) {
        U->dump();
        llvm::errs() << " ty: " << ty << "prev: " << prev << "\n";
      }
      assert(prev.getType() == ty);
      next = builder.create<AddFOp>(
          loc, prev,
          builder.create<ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else if (auto mt = dyn_cast<MemRefType>(ty)) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = ShapedType::kDynamic;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(1));
    } else if (auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(ty)) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      next = builder.create<LLVM::GEPOp>(
          loc, pt, prev,
          std::vector<mlir::Value>(
              {builder.create<ConstantIntOp>(loc, 1, ity)}));
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
      // ++x is x += 1 (C11 6.5.3.1): x's type, unless x is promoted.
      next = markNoSignedWrap(
          Glob.CGM,
          builder.create<AddIOp>(loc, prev,
                                 builder.create<ConstantIntOp>(
                                     loc, 1, ty.cast<mlir::IntegerType>())),
          U->getSubExpr()->getType(), U->getSubExpr()->getType());
    }
    sub.store(loc, builder, next);

    if (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc)
      return sub;
    else
      return ValueCategory(prev, /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_PreDec:
  case clang::UnaryOperator::Opcode::UO_PostDec: {
    auto ty = getMLIRType(U->getType());
    assert(sub.isReference);
    auto prev = sub.getValue(loc, builder);

    mlir::Value next;
    if (auto ft = dyn_cast<mlir::FloatType>(ty)) {
      next = builder.create<SubFOp>(
          loc, prev,
          builder.create<ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else if (auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(ty)) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      next = builder.create<LLVM::GEPOp>(
          loc, pt, prev,
          std::vector<mlir::Value>(
              {builder.create<ConstantIntOp>(loc, ShapedType::kDynamic, ity)}));
    } else if (auto mt = dyn_cast<MemRefType>(ty)) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = ShapedType::kDynamic;
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
      next = markNoSignedWrap(
          Glob.CGM,
          builder.create<SubIOp>(loc, prev,
                                 builder.create<ConstantIntOp>(
                                     loc, 1, ty.cast<mlir::IntegerType>())),
          U->getSubExpr()->getType(), U->getSubExpr()->getType());
    }
    sub.store(loc, builder, next);
    // In C++ `--i` is an lvalue, as `++i` (returned as `sub` above): a use
    // such as `for (i = 8; --i;)` reads it through an lvalue-to-rvalue cast,
    // which asserted on the value returned before (hlsyn-vitis aes, W8b).
    if (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PreDec &&
        U->isLValue())
      return sub;
    return ValueCategory(
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostDec) ? prev
                                                                     : next,
        /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Real:
  case clang::UnaryOperator::Opcode::UO_Imag: {
    int fnum =
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_Real) ? 0 : 1;
    assert(sub.isReference);
    return getComplexPartRef(loc, sub.val, fnum);
  }
  default: {
    U->dump();
    llvm_unreachable("unhandled opcode");
  }
  }
}

ValueCategory MLIRScanner::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *expr) {
  return Visit(expr->getReplacement());
}

ValueCategory
MLIRScanner::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Uop) {
  auto loc = getMLIRLocation(Uop->getExprLoc());
  switch (Uop->getKind()) {
  case UETT_SizeOf: {
    auto value = getTypeSize(loc, Uop->getTypeOfArgument());
    auto retTy = getMLIRType(Uop->getType()).cast<mlir::IntegerType>();
    return ValueCategory(builder.create<arith::IndexCastOp>(loc, retTy, value),
                         /*isReference*/ false);
  }
  case UETT_AlignOf: {
    auto value = getTypeAlign(loc, Uop->getTypeOfArgument());
    auto retTy = getMLIRType(Uop->getType()).cast<mlir::IntegerType>();
    return ValueCategory(builder.create<arith::IndexCastOp>(loc, retTy, value),
                         /*isReference*/ false);
  }
  default:
    Uop->dump();
    llvm_unreachable("unhandled VisitUnaryExprOrTypeTraitExpr");
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
        if (auto blockArg = dyn_cast<mlir::BlockArgument>(indexCastOperand)) {
          if (auto affineForOp = dyn_cast<mlir::affine::AffineForOp>(
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

ValueCategory MLIRScanner::VisitAtomicExpr(clang::AtomicExpr *BO) {
  auto loc = getMLIRLocation(BO->getExprLoc());

  switch (BO->getOp()) {
  case AtomicExpr::AtomicOp::AO__atomic_add_fetch: {
    auto a0 = Visit(BO->getPtr()).getValue(loc, builder);
    auto a1 = Visit(BO->getVal1()).getValue(loc, builder);
    auto ty = a1.getType();
    AtomicRMWKind op;
    LLVM::AtomicBinOp lop;
    if (ty.isa<mlir::IntegerType>()) {
      op = AtomicRMWKind::addi;
      lop = LLVM::AtomicBinOp::add;
    } else {
      op = AtomicRMWKind::addf;
      lop = LLVM::AtomicBinOp::fadd;
    }
    // TODO add atomic ordering
    mlir::Value v;
    if (a0.getType().isa<MemRefType>())
      v = builder.create<memref::AtomicRMWOp>(
          loc, op, a1, a0, std::vector<mlir::Value>({getConstantIndex(0)}));
    else
      v = builder.create<LLVM::AtomicRMWOp>(loc, lop, a0, a1,
                                            LLVM::AtomicOrdering::acq_rel);

    if (ty.isa<mlir::IntegerType>())
      v = builder.create<arith::AddIOp>(loc, v, a1);
    else
      v = builder.create<arith::AddFOp>(loc, v, a1);

    return ValueCategory(v, false);
  }
  case AtomicExpr::AtomicOp::AO__atomic_load: {
    // In the absence of an atomic load instruction, fall back to += 0.0
    auto a0 = Visit(BO->getPtr()).getValue(loc, builder);
    auto ret = Visit(BO->getVal1()).dereference(loc, builder);
    mlir::Type ty;
    if (auto MT = dyn_cast<MemRefType>(a0.getType())) {
      ty = MT.getElementType();
    } else {
      ty = a0.getType().cast<LLVM::LLVMPointerType>().getElementType();
    }
    AtomicRMWKind op;
    LLVM::AtomicBinOp lop;
    mlir::Value a1;
    if (ty.isa<mlir::IntegerType>()) {
      op = AtomicRMWKind::addi;
      lop = LLVM::AtomicBinOp::add;
      a1 = builder.create<ConstantIntOp>(loc, 0, ty);
    } else {
      op = AtomicRMWKind::addf;
      lop = LLVM::AtomicBinOp::fadd;
      a1 = builder.create<ConstantFloatOp>(
          loc, APFloat(ty.cast<mlir::FloatType>().getFloatSemantics(), "0"),
          ty.cast<mlir::FloatType>());
    }
    // TODO add atomic ordering
    mlir::Value v;
    if (a0.getType().isa<MemRefType>())
      v = builder.create<memref::AtomicRMWOp>(
          loc, op, a1, a0, std::vector<mlir::Value>({getConstantIndex(0)}));
    else
      v = builder.create<LLVM::AtomicRMWOp>(loc, lop, a0, a1,
                                            LLVM::AtomicOrdering::acq_rel);
    ret.store(loc, builder, v);
    return ValueCategory(v, false);
  }
  case AtomicExpr::AtomicOp::AO__atomic_store: {
    llvm::errs() << " TODO: implement atomic store with atomics\n";
    auto a0 = Visit(BO->getPtr()).dereference(loc, builder);
    auto a1 =
        Visit(BO->getVal1()).dereference(loc, builder).getValue(loc, builder);
    a0.store(loc, builder, a1);
    return ValueCategory();
  }
  case AtomicExpr::AtomicOp::AO__atomic_compare_exchange: {
    // In the absence of an atomic load instruction, fall back to += 0.0
    auto a0 = Visit(BO->getPtr()).getValue(loc, builder);
    auto a1 =
        Visit(BO->getVal1()).dereference(loc, builder).getValue(loc, builder);
    auto a2 =
        Visit(BO->getVal2()).dereference(loc, builder).getValue(loc, builder);
    if (auto mt = dyn_cast<MemRefType>(a0.getType())) {
      a0 = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getElementType(),
                                     mt.getMemorySpaceAsInt()),
          a0);
    }
    // TODO add atomic ordering
    mlir::Value v = builder.create<LLVM::AtomicCmpXchgOp>(
        loc, a0, a1, a2, LLVM::AtomicOrdering::seq_cst,
        LLVM::AtomicOrdering::seq_cst);
    v = builder.create<LLVM::ExtractValueOp>(loc, v, 1);
    auto postTy = getMLIRType(BO->getType()).cast<mlir::IntegerType>();
    if (postTy.getWidth() > 1)
      v = builder.create<arith::ExtUIOp>(loc, postTy, v);
    return ValueCategory(v, false);
  }
  default:
    llvm::errs() << "unhandled atomic:";
    BO->dump();
    assert(0);
  }
}

/// Reports the value assigned to a plain variable to `#pragma HLS bind_op`.
/// The variable an assignment writes: `x`, `x[i][j]`, `x.f`.
static const VarDecl *assignedVariable(clang::Expr *lhs) {
  lhs = lhs->IgnoreParenCasts();
  while (true) {
    if (auto *ASE = dyn_cast<ArraySubscriptExpr>(lhs))
      lhs = ASE->getBase()->IgnoreParenCasts();
    else if (auto *ME = dyn_cast<MemberExpr>(lhs))
      lhs = ME->getBase()->IgnoreParenCasts();
    else
      break;
  }
  if (auto DRE = dyn_cast<DeclRefExpr>(lhs))
    return dyn_cast<VarDecl>(DRE->getDecl());
  return nullptr;
}

/// Reports the value assigned to a variable to `#pragma HLS bind_op`.
static void notifyHLSAssign(clang::BinaryOperator *BO, mlir::Value value) {
  if (!HLSAnnotate)
    return;
  if (const VarDecl *VD = assignedVariable(BO->getLHS()))
    cgeist::hls::onAssignedValue(VD, value);
}

ValueCategory MLIRScanner::VisitBinaryOperator(clang::BinaryOperator *BO) {
  auto loc = getMLIRLocation(BO->getExprLoc());

  auto fixInteger = [&](mlir::Value res, bool forceUnsigned = false) {
    auto prevTy = res.getType().cast<mlir::IntegerType>();
    auto postTy = getMLIRType(BO->getType()).cast<mlir::IntegerType>();
    bool signedType = isSignedIntegerQualType(BO->getType());
    if (forceUnsigned)
      signedType = false;
    if (postTy != prevTy) {
      if (signedType) {
        res = builder.create<mlir::arith::ExtSIOp>(loc, postTy, res);
      } else {
        res = builder.create<mlir::arith::ExtUIOp>(loc, postTy, res);
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
    auto cond = lhs.getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(cond.getType())) {
      cond = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getElementType(),
                                     mt.getMemorySpaceAsInt()),
          cond);
    }
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
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

    auto rhs = Visit(BO->getRHS()).getValue(loc, builder);
    assert(rhs != nullptr);
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(rhs.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
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
    auto cond = lhs.getValue(loc, builder);
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
    auto rhs = Visit(BO->getRHS()).getValue(loc, builder);
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
  bool signedType = isSignedIntegerQualType(BO->getType());

  // Scalar `lhs op= rhs` (an integer or floating LHS, every compound
  // assignment): clang evaluates it in the computation type
  // (CompoundAssignOperator::getComputationLHSType(), e.g. `int` for an
  // `unsigned char` LHS, `double` for `int += 1.5` and for `float += double`,
  // the LHS type itself for a _BitInt), then converts the result back to the
  // LHS type. Both operands are brought to the computation type,
  // `mkInt(l, r, isSigned)` / `mkFloat(l, r)` build the op there, and the
  // result is truncated/extended (or converted back from a floating
  // computation type, or rounded to a narrower floating LHS) to the LHS
  // element type before the store; a bool LHS stores the result tested
  // against zero.
  auto emitScalarCompound =
      [&](llvm::function_ref<mlir::Value(mlir::Value, mlir::Value, bool)> mkInt,
          llvm::function_ref<mlir::Value(mlir::Value, mlir::Value)> mkFloat)
      -> ValueCategory {
    assert(lhs.isReference);
    auto *CAO = cast<clang::CompoundAssignOperator>(BO);
    clang::QualType compQT = CAO->getComputationLHSType();
    clang::QualType lhsQT = BO->getLHS()->getType();
    mlir::Type compTy = getMLIRType(compQT);
    bool compSigned = isSignedIntegerQualType(compQT);
    bool lhsSigned = isSignedIntegerQualType(lhsQT);
    mlir::Value prev = lhs.getValue(loc, builder);
    mlir::Value rhsV = rhs.getValue(loc, builder);
    mlir::Type lhsTy = prev.getType();
    // Convert the C value `v`, of signedness `vSigned`, to scalar type `to`,
    // whose C signedness is `toSigned` (it only decides fptosi vs fptoui: a
    // floating computation stored to an unsigned LHS is fptoui in C).
    auto convert = [&](mlir::Value v, mlir::Type to, bool vSigned,
                       bool toSigned) -> mlir::Value {
      mlir::Type from = v.getType();
      if (from == to)
        return v;
      if (auto fi = dyn_cast<mlir::IntegerType>(from)) {
        if (auto ti = dyn_cast<mlir::IntegerType>(to)) {
          if (fi.getWidth() > ti.getWidth())
            return builder.create<arith::TruncIOp>(loc, to, v);
          if (vSigned)
            return builder.create<arith::ExtSIOp>(loc, to, v);
          return builder.create<arith::ExtUIOp>(loc, to, v);
        }
        if (isa<mlir::FloatType>(to)) {
          if (vSigned)
            return builder.create<arith::SIToFPOp>(loc, to, v);
          return builder.create<arith::UIToFPOp>(loc, to, v);
        }
      } else if (auto ff = dyn_cast<mlir::FloatType>(from)) {
        if (auto tf = dyn_cast<mlir::FloatType>(to)) {
          if (ff.getWidth() < tf.getWidth())
            return builder.create<arith::ExtFOp>(loc, to, v);
          if (ff.getWidth() > tf.getWidth())
            return builder.create<arith::TruncFOp>(loc, to, v);
        } else if (isa<mlir::IntegerType>(to)) {
          if (toSigned)
            return builder.create<arith::FPToSIOp>(loc, to, v);
          return builder.create<arith::FPToUIOp>(loc, to, v);
        }
      }
      BO->dump();
      llvm::errs() << " from: " << v << " to: " << to << "\n";
      llvm_unreachable("unsupported compound assignment conversion");
    };
    rhsV = convert(rhsV, compTy,
                   isSignedIntegerQualType(BO->getRHS()->getType()),
                   compSigned);
    // `+` and `-` take the low bits of their result from the low bits of
    // their operands alone: with an integer computation type wider than an
    // integer (non-bool) LHS, `+=` / `-=` are built in the LHS type on the
    // RHS truncated to it, the same value as computing in the computation
    // type and truncating. A narrow counter's `i += 2` / `i -= 2` thus stays
    // an iN add of the counter, the update CanonicalizeFor raises to a for
    // loop. The iN op wraps modulo 2^N like the C assignment; AffineCFG
    // raises it to an affine expression only when it provably does not wrap.
    // The other compound assignments compute in the computation type.
    switch (BO->getOpcode()) {
    case clang::BinaryOperator::Opcode::BO_AddAssign:
    case clang::BinaryOperator::Opcode::BO_SubAssign:
      if (isa<mlir::IntegerType>(compTy) && isa<mlir::IntegerType>(lhsTy) &&
          !lhsQT->isBooleanType() &&
          lhsTy.getIntOrFloatBitWidth() < compTy.getIntOrFloatBitWidth()) {
        mlir::Value narrowRhs =
            builder.create<arith::TruncIOp>(loc, lhsTy, rhsV);
        mlir::Value result = mkInt(prev, narrowRhs, compSigned);
        notifyHLSAssign(BO, result);
        lhs.store(loc, builder, result);
        return lhs;
      }
      break;
    default:
      break;
    }
    prev = convert(prev, compTy, lhsSigned, compSigned);
    mlir::Value result;
    if (isa<mlir::FloatType>(compTy)) {
      assert(mkFloat && "compound assignment has no floating-point form");
      result = mkFloat(prev, rhsV);
    } else {
      result = markNoSignedWrap(Glob.CGM, mkInt(prev, rhsV, compSigned),
                                compQT, lhsQT,
                                operandTypeAsWritten(BO->getRHS()));
    }
    if (lhsQT->isBooleanType()) {
      // `b op= x` stores `(bool)(b op x)`: the result is tested against zero
      // (as CK_IntegralToBoolean / CK_FloatingToBoolean do), not truncated.
      if (auto ft = dyn_cast<mlir::FloatType>(compTy)) {
        auto zero = builder.create<ConstantFloatOp>(
            loc, APFloat::getZero(ft.getFloatSemantics()), ft);
        result = builder.create<arith::CmpFOp>(loc, CmpFPredicate::UNE, result,
                                               zero);
      } else {
        result = builder.create<arith::CmpIOp>(
            loc, CmpIPredicate::ne, result,
            builder.create<ConstantIntOp>(loc, 0, compTy));
      }
      if (lhsTy.getIntOrFloatBitWidth() > 1)
        result = builder.create<arith::ExtUIOp>(loc, lhsTy, result);
    } else {
      result = convert(result, lhsTy, compSigned, lhsSigned);
    }
    notifyHLSAssign(BO, result);
    lhs.store(loc, builder, result);
    return lhs;
  };

  // Floating complex `+ - * /` and their compound assignments, built the
  // way Vitis's front end does for its FPGA targets (hls-llvm-project
  // clang/lib/CodeGen/CGExprComplex.cpp EmitBinAdd / EmitBinSub /
  // EmitBinMul / EmitBinDiv under IsFPGAArch, Apache-2.0 WITH
  // LLVM-exception): a real operand contributes no imaginary part (C11
  // Annex G.5.1), the product is (ac - bd) + i(ad + bc) with no NaN
  // recovery libcall, and the quotient by a complex is ((ac + bd) + i(bc -
  // ad)) / (cc + dd) with no __divsc3. The type test looks through typedefs:
  // libstdc++'s std::complex<float> keeps its value as `_ComplexT _M_value`
  // and its operators are `_M_value op= __z.__rep()`, whose type is the
  // typedef, which the opcode cases below test with isa<ComplexType> and
  // missed (CIRCT W9 round 6 review, defect 2: `a[i] = a[i] + a[j]` and
  // `a[i] = a[i] * a[j]` asserted 'must have shape 1').
  if (const auto *cty = BO->getType()->getAs<clang::ComplexType>()) {
    clang::BinaryOperator::Opcode op = BO->getOpcode();
    bool assign = BO->isCompoundAssignmentOp();
    if (assign)
      op = clang::BinaryOperator::getOpForCompoundAssignment(op);
    bool arith = op == clang::BO_Add || op == clang::BO_Sub ||
                 op == clang::BO_Mul || op == clang::BO_Div;
    // `+` and `+=` of two complex operands on a type written as `_Complex`
    // itself already work in the opcode cases below (the `+=` case also
    // tells the HLS pragma layer about the assignment); they keep that path
    // so that the MLIR of code that compiled before is unchanged.
    if (op == clang::BO_Add && isa<clang::ComplexType>(BO->getType()) &&
        BO->getLHS()->getType()->isAnyComplexType() &&
        BO->getRHS()->getType()->isAnyComplexType())
      arith = false;
    clang::QualType compElt = cty->getElementType();
    if (assign)
      if (const auto *cc = cast<clang::CompoundAssignOperator>(BO)
                               ->getComputationResultType()
                               ->getAs<clang::ComplexType>())
        compElt = cc->getElementType();
    auto compTy = dyn_cast<mlir::FloatType>(getMLIRType(compElt));
    auto lhsElt = dyn_cast<mlir::FloatType>(getMLIRType(cty->getElementType()));
    if (arith && compTy && lhsElt) {
      auto toFloat = [&](mlir::Value v, mlir::FloatType t) -> mlir::Value {
        if (!v || v.getType() == t)
          return v;
        auto ft = cast<mlir::FloatType>(v.getType());
        if (ft.getWidth() < t.getWidth())
          return builder.create<arith::ExtFOp>(loc, t, v);
        return builder.create<arith::TruncFOp>(loc, t, v);
      };
      // (real, imaginary) of an operand; the imaginary part of a real one
      // is null.
      auto parts = [&](ValueCategory v, clang::QualType t)
          -> std::pair<mlir::Value, mlir::Value> {
        if (t->isAnyComplexType())
          return {toFloat(getComplexPart(loc, v.val, 0), compTy),
                  toFloat(getComplexPart(loc, v.val, 1), compTy)};
        mlir::Value r = v.getValue(loc, builder);
        if (auto it = dyn_cast<mlir::IntegerType>(r.getType()))
          r = t->isUnsignedIntegerType()
                  ? builder.create<arith::UIToFPOp>(loc, compTy, r).getResult()
                  : builder.create<arith::SIToFPOp>(loc, compTy, r)
                        .getResult();
        return {toFloat(r, compTy), mlir::Value()};
      };
      auto [a, b] = parts(lhs, BO->getLHS()->getType());
      auto [c, d] = parts(rhs, BO->getRHS()->getType());
      auto add = [&](mlir::Value x, mlir::Value y) -> mlir::Value {
        return builder.create<AddFOp>(loc, x, y);
      };
      auto sub = [&](mlir::Value x, mlir::Value y) -> mlir::Value {
        return builder.create<SubFOp>(loc, x, y);
      };
      auto mul = [&](mlir::Value x, mlir::Value y) -> mlir::Value {
        return builder.create<MulFOp>(loc, x, y);
      };
      auto div = [&](mlir::Value x, mlir::Value y) -> mlir::Value {
        return builder.create<arith::DivFOp>(loc, x, y);
      };
      mlir::Value re, im;
      switch (op) {
      case clang::BO_Add:
        re = add(a, c);
        im = b && d ? add(b, d) : (b ? b : d);
        break;
      case clang::BO_Sub:
        re = sub(a, c);
        im = b && d ? sub(b, d)
                    : (b ? b : builder.create<arith::NegFOp>(loc, d));
        break;
      case clang::BO_Mul:
        if (b && d) {
          re = sub(mul(a, c), mul(b, d));
          im = add(mul(a, d), mul(b, c));
        } else {
          re = mul(a, c);
          im = b ? mul(b, c) : mul(a, d);
        }
        break;
      default: // BO_Div
        if (d) {
          if (!b)
            b = builder.create<ConstantFloatOp>(
                loc, APFloat::getZero(compTy.getFloatSemantics()), compTy);
          mlir::Value den = add(mul(c, c), mul(d, d));
          re = div(add(mul(a, c), mul(b, d)), den);
          im = div(sub(mul(b, c), mul(a, d)), den);
        } else {
          re = div(a, c);
          im = div(b, c);
        }
        break;
      }
      if (!im) // both operands real: not a complex expression
        im = builder.create<ConstantFloatOp>(
            loc, APFloat::getZero(compTy.getFloatSemantics()), compTy);
      re = toFloat(re, lhsElt);
      im = toFloat(im, lhsElt);
      if (!assign)
        return createComplexFloat(loc, re, im, BO->getType());
      assert(lhs.isReference);
      getComplexPartRef(loc, lhs.val, 0).store(loc, builder, re);
      getComplexPartRef(loc, lhs.val, 1).store(loc, builder, im);
      return lhs;
    }
  }

  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_Shr: {
    auto lhsv = lhs.getValue(loc, builder);
    auto rhsv = rhs.getValue(loc, builder);
    auto prevTy = rhsv.getType().cast<mlir::IntegerType>();
    auto postTy = lhsv.getType().cast<mlir::IntegerType>();
    if (prevTy.getWidth() < postTy.getWidth())
      rhsv = builder.create<mlir::arith::ExtUIOp>(loc, postTy, rhsv);
    if (prevTy.getWidth() > postTy.getWidth())
      rhsv = builder.create<mlir::arith::TruncIOp>(loc, postTy, rhsv);
    assert(lhsv.getType() == rhsv.getType());
    if (signedType)
      return ValueCategory(builder.create<ShRSIOp>(loc, lhsv, rhsv),
                           /*isReference*/ false);
    else
      return ValueCategory(builder.create<ShRUIOp>(loc, lhsv, rhsv),
                           /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Shl: {
    auto lhsv = lhs.getValue(loc, builder);
    auto rhsv = rhs.getValue(loc, builder);
    auto prevTy = rhsv.getType().cast<mlir::IntegerType>();
    auto postTy = lhsv.getType().cast<mlir::IntegerType>();
    if (prevTy.getWidth() < postTy.getWidth())
      rhsv = builder.create<arith::ExtUIOp>(loc, postTy, rhsv);
    if (prevTy.getWidth() > postTy.getWidth())
      rhsv = builder.create<arith::TruncIOp>(loc, postTy, rhsv);
    assert(lhsv.getType() == rhsv.getType());
    return ValueCategory(builder.create<ShLIOp>(loc, lhsv, rhsv),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_And: {
    return ValueCategory(builder.create<AndIOp>(loc, lhs.getValue(loc, builder),
                                                rhs.getValue(loc, builder)),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Xor: {
    return ValueCategory(builder.create<XOrIOp>(loc, lhs.getValue(loc, builder),
                                                rhs.getValue(loc, builder)),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Or: {
    // TODO short circuit
    return ValueCategory(builder.create<OrIOp>(loc, lhs.getValue(loc, builder),
                                               rhs.getValue(loc, builder)),
                         /*isReference*/ false);
  }
    {
      auto lhs_v = lhs.getValue(loc, builder);
      mlir::Value res;
      if (lhs_v.getType().isa<mlir::FloatType>()) {
        res = builder.create<CmpFOp>(loc, CmpFPredicate::UGT, lhs_v,
                                     rhs.getValue(loc, builder));
      } else {
        res = builder.create<CmpIOp>(
            loc, signedType ? CmpIPredicate::sgt : CmpIPredicate::ugt, lhs_v,
            rhs.getValue(loc, builder));
      }
      return fixInteger(res);
    }
  case clang::BinaryOperator::Opcode::BO_GT:
  case clang::BinaryOperator::Opcode::BO_GE:
  case clang::BinaryOperator::Opcode::BO_LT:
  case clang::BinaryOperator::Opcode::BO_LE:
  case clang::BinaryOperator::Opcode::BO_EQ:
  case clang::BinaryOperator::Opcode::BO_NE: {
    signedType = isSignedIntegerQualType(BO->getLHS()->getType());
    CmpFPredicate FPred;
    CmpIPredicate IPred;
    LLVM::ICmpPredicate LPred;
    switch (BO->getOpcode()) {
    case clang::BinaryOperator::Opcode::BO_GT:
      FPred = CmpFPredicate::OGT;
      IPred = signedType ? CmpIPredicate::sgt : CmpIPredicate::ugt,
      LPred = LLVM::ICmpPredicate::ugt;
      break;
    case clang::BinaryOperator::Opcode::BO_GE:
      FPred = CmpFPredicate::OGE;
      IPred = signedType ? CmpIPredicate::sge : CmpIPredicate::uge,
      LPred = LLVM::ICmpPredicate::uge;
      break;
    case clang::BinaryOperator::Opcode::BO_LT:
      FPred = CmpFPredicate::OLT;
      IPred = signedType ? CmpIPredicate::slt : CmpIPredicate::ult,
      LPred = LLVM::ICmpPredicate::ult;
      break;
    case clang::BinaryOperator::Opcode::BO_LE:
      FPred = CmpFPredicate::OLE;
      IPred = signedType ? CmpIPredicate::sle : CmpIPredicate::ule,
      LPred = LLVM::ICmpPredicate::ule;
      break;
    case clang::BinaryOperator::Opcode::BO_EQ:
      FPred = CmpFPredicate::OEQ;
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

    auto lhs_v = lhs.getValue(loc, builder);
    auto rhs_v = rhs.getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(lhs_v.getType())) {
      lhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getElementType()), lhs_v);
    }
    if (auto mt = dyn_cast<mlir::MemRefType>(rhs_v.getType())) {
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
    return fixInteger(res, /*forceUnsigned*/ true);
  }
  case clang::BinaryOperator::Opcode::BO_Mul: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex mult");
    }
    auto lhs_v = lhs.getValue(loc, builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(
          builder.create<arith::MulFOp>(loc, lhs_v, rhs.getValue(loc, builder)),
          /*isReference*/ false);
    } else {
      return ValueCategory(
          markNoSignedWrap(Glob.CGM,
                           builder.create<arith::MulIOp>(
                               loc, lhs_v, rhs.getValue(loc, builder)),
                           BO->getType(), operandTypeAsWritten(BO->getLHS()),
                           operandTypeAsWritten(BO->getRHS())),
          /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Div: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex div");
    }
    auto lhs_v = lhs.getValue(loc, builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(
          builder.create<arith::DivFOp>(loc, lhs_v, rhs.getValue(loc, builder)),
          /*isReference*/ false);
      ;
    } else {
      if (signedType)
        return ValueCategory(builder.create<arith::DivSIOp>(
                                 loc, lhs_v, rhs.getValue(loc, builder)),
                             /*isReference*/ false);
      else
        return ValueCategory(builder.create<arith::DivUIOp>(
                                 loc, lhs_v, rhs.getValue(loc, builder)),
                             /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Rem: {
    auto lhs_v = lhs.getValue(loc, builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(
          builder.create<arith::RemFOp>(loc, lhs_v, rhs.getValue(loc, builder)),
          /*isReference*/ false);
    } else {
      if (signedType)
        return ValueCategory(builder.create<arith::RemSIOp>(
                                 loc, lhs_v, rhs.getValue(loc, builder)),
                             /*isReference*/ false);
      else
        return ValueCategory(builder.create<arith::RemUIOp>(
                                 loc, lhs_v, rhs.getValue(loc, builder)),
                             /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Add: {
    auto emitSubindex = [&](auto mr, auto ptradd, bool ptraddUnsigned) {
      auto mt = mr.getType().template dyn_cast<mlir::MemRefType>();
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = ShapedType::kDynamic;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      ptradd = castToIndex(loc, ptradd, ptraddUnsigned);
      return ValueCategory(
          builder.create<polygeist::SubIndexOp>(loc, mt0, mr, ptradd),
          /*isReference*/ false);
    };
    if (auto cty = dyn_cast<clang::ComplexType>(BO->getType())) {
      mlir::Value real =
          builder.create<AddFOp>(loc, getComplexPart(loc, lhs.val, 0),
                                 getComplexPart(loc, rhs.val, 0));
      mlir::Value imag =
          builder.create<AddFOp>(loc, getComplexPart(loc, lhs.val, 1),
                                 getComplexPart(loc, rhs.val, 1));
      return createComplexFloat(loc, real, imag, BO->getType());
    }

    auto lhs_v = lhs.getValue(loc, builder);
    auto rhs_v = rhs.getValue(loc, builder);
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      return ValueCategory(builder.create<AddFOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    } else if (lhs_v.getType().isa<mlir::MemRefType>()) {
      return emitSubindex(lhs_v, rhs_v,
                          !isSignedIntegerQualType(BO->getRHS()->getType()));
    } else if (rhs_v.getType().isa<mlir::MemRefType>()) {
      return emitSubindex(rhs_v, lhs_v,
                          !isSignedIntegerQualType(BO->getLHS()->getType()));
    } else if (auto pt =
                   dyn_cast<mlir::LLVM::LLVMPointerType>(lhs_v.getType())) {
      return ValueCategory(
          builder.create<LLVM::GEPOp>(loc, pt, lhs_v,
                                      std::vector<mlir::Value>({rhs_v})),
          /*isReference*/ false);
    } else {
      if (auto lhs_c = lhs_v.getDefiningOp<ConstantIntOp>()) {
        if (auto rhs_c = rhs_v.getDefiningOp<ConstantIntOp>()) {
          if (lhs_c.getType() == rhs_c.getType()) {
            // Fold in the constants' own width (wraps like the C type; works
            // past 64 bits).
            APInt sum = lhs_c.getValue().cast<IntegerAttr>().getValue() +
                        rhs_c.getValue().cast<IntegerAttr>().getValue();
            return ValueCategory(
                builder.create<arith::ConstantOp>(
                    loc, lhs_c.getType(),
                    mlir::IntegerAttr::get(lhs_c.getType(), sum)),
                false);
          }
        }
      }
      return ValueCategory(
          markNoSignedWrap(Glob.CGM, builder.create<AddIOp>(loc, lhs_v, rhs_v),
                           BO->getType(), operandTypeAsWritten(BO->getLHS()),
                           operandTypeAsWritten(BO->getRHS())),
          /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Sub: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex sub");
    }
    auto lhs_v = lhs.getValue(loc, builder);
    auto rhs_v = rhs.getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(lhs_v.getType())) {
      mlir::Type innerType = mt.getElementType();
      auto shape = mt.getShape();
      for (size_t i = 1; i < shape.size(); i++)
        innerType = LLVM::LLVMArrayType::get(innerType, shape[i]);
      lhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(innerType), lhs_v);
    }
    if (auto mt = dyn_cast<mlir::MemRefType>(rhs_v.getType())) {
      mlir::Type innerType = mt.getElementType();
      auto shape = mt.getShape();
      for (size_t i = 1; i < shape.size(); i++)
        innerType = LLVM::LLVMArrayType::get(innerType, shape[i]);
      rhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(innerType), rhs_v);
    }
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      assert(rhs_v.getType() == lhs_v.getType());
      return ValueCategory(builder.create<SubFOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    } else if (auto pt =
                   dyn_cast<mlir::LLVM::LLVMPointerType>(lhs_v.getType())) {
      if (auto IT = dyn_cast<mlir::IntegerType>(rhs_v.getType())) {
        mlir::Value vals[1] = {builder.create<SubIOp>(
            loc, builder.create<ConstantIntOp>(loc, 0, IT.getWidth()), rhs_v)};
        return ValueCategory(
            builder.create<LLVM::GEPOp>(loc, lhs_v.getType(), lhs_v,
                                        ArrayRef<mlir::Value>(vals)),
            false);
      }
      mlir::Value val =
          builder.create<SubIOp>(loc,
                                 builder.create<LLVM::PtrToIntOp>(
                                     loc, getMLIRType(BO->getType()), lhs_v),
                                 builder.create<LLVM::PtrToIntOp>(
                                     loc, getMLIRType(BO->getType()), rhs_v));
      val = builder.create<DivSIOp>(
          loc, val,
          builder.create<IndexCastOp>(
              loc, val.getType(),
              builder.create<polygeist::TypeSizeOp>(
                  loc, builder.getIndexType(),
                  mlir::TypeAttr::get(pt.getElementType()))));
      return ValueCategory(val, /*isReference*/ false);
    } else {
      return ValueCategory(
          markNoSignedWrap(Glob.CGM, builder.create<SubIOp>(loc, lhs_v, rhs_v),
                           BO->getType(), operandTypeAsWritten(BO->getLHS()),
                           operandTypeAsWritten(BO->getRHS())),
          /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Assign: {
    assert(lhs.isReference);
    // C struct assignment `a[i] = b[j]` of a struct whose members share one
    // type (CombinedStructABI: `struct { double x, y, z; }` is a row of
    // memref<?x3xf64>): both sides are references to such a row, and the
    // assignment copies the members one by one. Reading the row as one
    // value (below) asserts on a rank-2 reference (W8b repros
    // r_struct_assign*.c: `out[i] = p`, `out[1][i] = pos[2][i]`). A
    // `_Complex` value is such a row too (memref<?x2xf32>): the implicit
    // copy assignment of std::complex<float> (`_M_value = __z._M_value`)
    // asserted the same way (CIRCT OPEN_ISSUES O70 (a), hp_fft_hls
    // `d[i] = std::complex<float>(a[i], b[i])`).
    // A floating `_Complex` assignment copies the real and the imaginary
    // part, whatever form the right side has (a row of an array, a
    // temporary from createComplexFloat, a struct value): both parts are
    // read before either is written.
    if (const auto *cty = BO->getType()->getAs<clang::ComplexType>())
      if (cty->getElementType()->isRealFloatingType()) {
        mlir::Value re = getComplexPart(loc, rhs.val, 0);
        mlir::Value im = getComplexPart(loc, rhs.val, 1);
        getComplexPartRef(loc, lhs.val, 0).store(loc, builder, re);
        getComplexPartRef(loc, lhs.val, 1).store(loc, builder, im);
        return lhs;
      }
    if (rhs.isReference &&
        (BO->getType()->isRecordType() || BO->getType()->isAnyComplexType()))
      if (auto rmt = dyn_cast<MemRefType>(rhs.val.getType()))
        if (auto lmt = dyn_cast<MemRefType>(lhs.val.getType()))
          if (rmt.getRank() == 2 && lmt.getRank() == 2 &&
              rmt.getElementType() == lmt.getElementType() &&
              !rmt.isDynamicDim(1) && rmt.getDimSize(1) == lmt.getDimSize(1)) {
            lhs.store(loc, builder, rhs, /*isArray*/ true);
            return lhs;
          }
    mlir::Value tostore = rhs.getValue(loc, builder);
    mlir::Type subType;
    if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(lhs.val.getType()))
      subType = PT.getElementType();
    else
      subType = lhs.val.getType().cast<MemRefType>().getElementType();
    if (tostore.getType() != subType) {
      if (auto prevTy = dyn_cast<mlir::IntegerType>(tostore.getType())) {
        if (auto postTy = dyn_cast<mlir::IntegerType>(subType)) {
          bool signedType = isSignedIntegerQualType(BO->getType());

          if (prevTy.getWidth() < postTy.getWidth()) {
            if (signedType) {
              tostore = builder.create<arith::ExtSIOp>(loc, postTy, tostore);
            } else {
              tostore = builder.create<arith::ExtUIOp>(loc, postTy, tostore);
            }
          } else if (prevTy.getWidth() > postTy.getWidth()) {
            tostore = builder.create<arith::TruncIOp>(loc, postTy, tostore);
          }
        }
      }
    }
    notifyHLSAssign(BO, tostore);
    lhs.store(loc, builder, tostore);
    return lhs;
  }

  case clang::BinaryOperator::Opcode::BO_Comma: {
    return rhs;
  }

  case clang::BinaryOperator::Opcode::BO_AddAssign: {
    assert(lhs.isReference);

    mlir::Value result;
    if (isa<clang::ComplexType>(BO->getType())) {
      mlir::Value prev = lhs.val;
      mlir::Value rhsV = rhs.val;
      mlir::Value real = builder.create<AddFOp>(
          loc, getComplexPart(loc, prev, 0), getComplexPart(loc, rhsV, 0));
      mlir::Value imag = builder.create<AddFOp>(
          loc, getComplexPart(loc, prev, 1), getComplexPart(loc, rhsV, 1));
      result = createComplexFloat(loc, real, imag, BO->getType())
                   .getValue(loc, builder);
      notifyHLSAssign(BO, result);
    lhs.store(loc, builder, result);
      return lhs;
    }
    auto prev = lhs.getValue(loc, builder);
    // An integer or floating LHS (bool, char, _BitInt, float, ...) goes
    // through the computation type like every other compound assignment:
    // `int += 1.5` is sitofp / addf / fptosi, `bool += 1` stores
    // (b + 1) != 0, `float += double` adds in double and rounds the sum.
    if (prev.getType().isa<mlir::IntegerType, mlir::FloatType>())
      return emitScalarCompound(
          [&](mlir::Value l, mlir::Value r, bool) -> mlir::Value {
            return builder.create<AddIOp>(loc, l, r);
          },
          [&](mlir::Value l, mlir::Value r) -> mlir::Value {
            return builder.create<AddFOp>(loc, l, r);
          });
    if (auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(prev.getType())) {
      result = builder.create<LLVM::GEPOp>(
          loc, pt, prev,
          std::vector<mlir::Value>({rhs.getValue(loc, builder)}));
    } else if (auto postTy = dyn_cast<mlir::MemRefType>(prev.getType())) {
      mlir::Value rhsV = rhs.getValue(loc, builder);
      auto shape = std::vector<int64_t>(postTy.getShape());
      shape[0] = ShapedType::kDynamic;
      postTy = mlir::MemRefType::get(shape, postTy.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     postTy.getMemorySpace());
      auto ptradd = rhsV;
      ptradd = castToIndex(loc, ptradd,
                           !isSignedIntegerQualType(BO->getRHS()->getType()));
      result = builder.create<polygeist::SubIndexOp>(loc, postTy, prev, ptradd);
    } else {
      assert(false && "Unsupported add assign type");
    }
    notifyHLSAssign(BO, result);
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_SubAssign: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex sub");
    }
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);
    if (prev.getType().isa<mlir::IntegerType, mlir::FloatType>())
      return emitScalarCompound(
          [&](mlir::Value l, mlir::Value r, bool) -> mlir::Value {
            return builder.create<SubIOp>(loc, l, r);
          },
          [&](mlir::Value l, mlir::Value r) -> mlir::Value {
            return builder.create<SubFOp>(loc, l, r);
          });

    mlir::Value result =
        builder.create<SubIOp>(loc, prev, rhs.getValue(loc, builder));
    notifyHLSAssign(BO, result);
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_MulAssign: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex mult");
    }
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);
    if (prev.getType().isa<mlir::IntegerType, mlir::FloatType>())
      return emitScalarCompound(
          [&](mlir::Value l, mlir::Value r, bool) -> mlir::Value {
            return builder.create<MulIOp>(loc, l, r);
          },
          [&](mlir::Value l, mlir::Value r) -> mlir::Value {
            return builder.create<MulFOp>(loc, l, r);
          });

    mlir::Value result =
        builder.create<MulIOp>(loc, prev, rhs.getValue(loc, builder));
    notifyHLSAssign(BO, result);
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_DivAssign: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex div");
    }
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);
    if (prev.getType().isa<mlir::IntegerType, mlir::FloatType>())
      return emitScalarCompound(
          [&](mlir::Value l, mlir::Value r, bool isSigned) -> mlir::Value {
            if (isSigned)
              return builder.create<arith::DivSIOp>(loc, l, r);
            return builder.create<arith::DivUIOp>(loc, l, r);
          },
          [&](mlir::Value l, mlir::Value r) -> mlir::Value {
            return builder.create<arith::DivFOp>(loc, l, r);
          });

    mlir::Value result;
    if (signedType)
      result =
          builder.create<arith::DivSIOp>(loc, prev, rhs.getValue(loc, builder));
    else
      result =
          builder.create<arith::DivUIOp>(loc, prev, rhs.getValue(loc, builder));
    notifyHLSAssign(BO, result);
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_ShrAssign: {
    return emitScalarCompound(
        [&](mlir::Value l, mlir::Value r, bool isSigned) -> mlir::Value {
          if (isSigned)
            return builder.create<ShRSIOp>(loc, l, r);
          return builder.create<ShRUIOp>(loc, l, r);
        },
        nullptr);
  }
  case clang::BinaryOperator::Opcode::BO_ShlAssign: {
    return emitScalarCompound(
        [&](mlir::Value l, mlir::Value r, bool) -> mlir::Value {
          return builder.create<ShLIOp>(loc, l, r);
        },
        nullptr);
  }
  case clang::BinaryOperator::Opcode::BO_RemAssign: {
    return emitScalarCompound(
        [&](mlir::Value l, mlir::Value r, bool isSigned) -> mlir::Value {
          if (isSigned)
            return builder.create<RemSIOp>(loc, l, r);
          return builder.create<RemUIOp>(loc, l, r);
        },
        [&](mlir::Value l, mlir::Value r) -> mlir::Value {
          return builder.create<RemFOp>(loc, l, r);
        });
  }
  case clang::BinaryOperator::Opcode::BO_AndAssign: {
    return emitScalarCompound(
        [&](mlir::Value l, mlir::Value r, bool) -> mlir::Value {
          return builder.create<AndIOp>(loc, l, r);
        },
        nullptr);
  }
  case clang::BinaryOperator::Opcode::BO_OrAssign: {
    return emitScalarCompound(
        [&](mlir::Value l, mlir::Value r, bool) -> mlir::Value {
          return builder.create<OrIOp>(loc, l, r);
        },
        nullptr);
  }
  case clang::BinaryOperator::Opcode::BO_XorAssign: {
    return emitScalarCompound(
        [&](mlir::Value l, mlir::Value r, bool) -> mlir::Value {
          return builder.create<XOrIOp>(loc, l, r);
        },
        nullptr);
  }

  default: {
    BO->dump();
    llvm_unreachable("unhandled opcode");
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

ValueCategory MLIRScanner::CommonFieldLookup(mlir::Location loc,
                                             clang::QualType CT,
                                             const FieldDecl *FD,
                                             mlir::Value val, bool isLValue) {
  assert(FD && "Attempting to lookup field of nullptr");
  auto rd = FD->getParent();

  // The member of a scalar wrapper is the object itself.
  if (Glob.getScalarWrapperField(rd) == FD) {
    if (isLValue)
      val = ValueCategory(val, /*isReference*/ true).getValue(loc, builder);
    return ValueCategory(val, /*isReference*/ true);
  }

  auto ST = cast<llvm::StructType>(getLLVMType(CT));

  size_t fnum = 0;

  auto CXRD = dyn_cast<CXXRecordDecl>(rd);

  if (isLLVMStructABI(rd, ST)) {
    auto &layout = Glob.CGM.getTypes().getCGRecordLayout(rd);
    fnum = layout.getLLVMFieldNo(FD);
  } else {
    fnum = 0;
    if (CXRD)
      fnum += CXRD->getDefinition()->getNumBases();
    for (auto field : rd->fields()) {
      if (field == FD) {
        break;
      }
      fnum++;
    }
  }

  if (auto mt = dyn_cast<MemRefType>(val.getType())) {
    auto shape = std::vector<int64_t>(mt.getShape());
    if (shape.size() > 1) {
      shape.erase(shape.begin());
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      shape[0] = ShapedType::kDynamic;
      auto mt1 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      mlir::Value sub0 = builder.create<polygeist::SubIndexOp>(
          loc, mt0, val, getConstantIndex(0));
      mlir::Value sub1 = builder.create<polygeist::SubIndexOp>(
          loc, mt1, sub0, getConstantIndex(fnum));
      if (isLValue)
        sub1 = ValueCategory(sub1, /*isReference*/ true).getValue(loc, builder);
      return ValueCategory(sub1, /*isReference*/ true);
    }
    auto ET = mt.getElementType();
    if (shape.size() == 1 &&
        !ET.isa<mlir::LLVM::LLVMStructType, mlir::LLVM::LLVMArrayType>()) {
      // A struct whose members share one type is a memref of that type with
      // the members as its (only) dimension (CombinedStructABI): a reference
      // to one `struct { unsigned long a, b; }` is memref<2xi64>. Member
      // `fnum` is the view starting at element fnum, as the rank > 1 case
      // above builds after selecting the row (W8b repro r_bfs.c, machsuite
      // bfs_bulk: `nodes[n].edge_begin` with `node_t nodes[N]`).
      shape[0] = ShapedType::kDynamic;
      auto mt1 = mlir::MemRefType::get(shape, ET, MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      mlir::Value sub = builder.create<polygeist::SubIndexOp>(
          loc, mt1, val, getConstantIndex(fnum));
      if (isLValue)
        sub = ValueCategory(sub, /*isReference*/ true).getValue(loc, builder);
      return ValueCategory(sub, /*isReference*/ true);
    }
    assert(ET.isa<mlir::LLVM::LLVMStructType>() ||
           ET.isa<mlir::LLVM::LLVMArrayType>());
    val = builder.create<polygeist::Memref2PointerOp>(
        loc, LLVM::LLVMPointerType::get(ET, mt.getMemorySpaceAsInt()), val);
  }

  auto PT = val.getType().cast<mlir::LLVM::LLVMPointerType>();
  mlir::Value vec[] = {builder.create<ConstantIntOp>(loc, 0, 32),
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
  if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(PT.getElementType())) {
    ET = ST.getBody()[fnum];
  } else {
    ET = PT.getElementType().cast<mlir::LLVM::LLVMArrayType>().getElementType();
  }
  mlir::Value commonGEP = builder.create<mlir::LLVM::GEPOp>(
      loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()), val,
      vec);

  bool isArray = false;
  auto subType = Glob.getMLIRType(
      Glob.CGM.getContext().getPointerType(FD->getType()), &isArray);
  assert(!isArray);
  if (rd->isUnion()) {
    if (auto PT2 = dyn_cast<LLVM::LLVMPointerType>(subType))
      commonGEP = builder.create<mlir::LLVM::BitcastOp>(
          loc,
          mlir::LLVM::LLVMPointerType::get(PT2.getElementType(),
                                           PT.getAddressSpace()),
          commonGEP);
    else {
      auto mt = subType.cast<mlir::MemRefType>();
      commonGEP = builder.create<polygeist::Pointer2MemrefOp>(
          loc,
          mlir::MemRefType::get(
              mt.getShape(), mt.getElementType(), MemRefLayoutAttrInterface(),
              wrapIntegerMemorySpace(PT.getAddressSpace(), mt.getContext())),
          commonGEP);
    }
  } else {
    if (auto mt = dyn_cast<mlir::MemRefType>(subType)) {
      commonGEP = builder.create<polygeist::Pointer2MemrefOp>(
          loc,
          mlir::MemRefType::get(
              mt.getShape(), mt.getElementType(), MemRefLayoutAttrInterface(),
              wrapIntegerMemorySpace(PT.getAddressSpace(), mt.getContext())),
          commonGEP);
    }
  }
  if (isLValue)
    commonGEP =
        ValueCategory(commonGEP, /*isReference*/ true).getValue(loc, builder);
  return ValueCategory(commonGEP, /*isReference*/ true);
}

ValueCategory MLIRScanner::VisitDeclRefExpr(DeclRefExpr *E) {
  auto loc = getMLIRLocation(E->getLocation());
  auto name = E->getDecl()->getName().str();

  if (auto tocall = dyn_cast<FunctionDecl>(E->getDecl())) {
    auto f = Glob.GetOrCreateMLIRFunction(tocall);
    auto FT = f.getFunctionType();
    mlir::Type RT = LLVM::LLVMVoidType::get(f.getContext());
    if (FT.getNumResults() != 0)
      RT = FT.getResult(0);
    LLVM::LLVMFunctionType LFT = LLVM::LLVMFunctionType::get(
        RT, FT.getInputs(), /*unsupported presentlyFT.isVariadic()*/ false);

    return ValueCategory(builder.create<polygeist::GetFuncOp>(
                             loc, LLVM::LLVMPointerType::get(LFT), f.getName()),
                         /*isReference*/ true);
  }

  if (auto VD = dyn_cast<VarDecl>(E->getDecl())) {
    if (Captures.find(VD) != Captures.end()) {
      FieldDecl *field = Captures[VD];
      auto res = CommonFieldLookup(
          loc, cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(),
          field, ThisVal.val,
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
        // A qualified name (`ns::table[i]`, the Vitis math headers' tables)
        // is an ordinary global with --xlx-headers; otherwise it keeps the
        // LLVM-global path, which cannot index a memref-typed global.
        (E->hasQualifier() && !cgeist::hls::xlxHeaders())) {
      return ValueCategory(builder.create<mlir::LLVM::AddressOfOp>(
                               loc, Glob.GetOrCreateLLVMGlobal(VD)),
                           /*isReference*/ true);
    }

    auto gv = Glob.GetOrCreateGlobal(VD, /*prefix=*/"");

    auto mt = gv.first.getType();
    auto gv2 = builder.create<memref::GetGlobalOp>(loc, mt, gv.first.getName());
    auto shape = std::vector<int64_t>(mt.getShape());
    shape[0] = ShapedType::kDynamic;
    auto val = builder.create<memref::CastOp>(
        loc,
        MemRefType::get(shape, mt.getElementType(), MemRefLayoutAttrInterface(),
                        mt.getMemorySpace()),
        gv2);
    bool isArray = gv.second;
    // TODO check reference
    if (isArray)
      return ValueCategory(val, /*isReference*/ true);
    else
      return ValueCategory(val, /*isReference*/ true);
    // return gv2;
  }
  E->dump();
  E->getDecl()->dump();
  llvm::errs() << "couldn't find " << name << "\n";
  llvm_unreachable("couldnt find value");
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
  llvm_unreachable("unhandled typeid");
}

ValueCategory
MLIRScanner::VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  assert(ThisVal.val);
  auto toset = Visit(expr->getExpr());
  assert(!ThisVal.isReference);
  assert(toset.val);

  bool isArray = false;
  Glob.getMLIRType(expr->getExpr()->getType(), &isArray);

  auto cfl = CommonFieldLookup(
      loc, cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(),
      expr->getField(), ThisVal.val, /*isLValue*/ false);
  assert(cfl.val);
  cfl.store(loc, builder, toset, isArray);
  return cfl;
}

ValueCategory MLIRScanner::VisitCXXNoexceptExpr(CXXNoexceptExpr *expr) {
  auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                    expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitMemberExpr(MemberExpr *ME) {
  auto loc = getMLIRLocation(ME->getExprLoc());
  // A static data member named through an object (`obj.N`, Vitis_Libraries
  // solver svd.hpp `traits.OFF_DIAG_II`): the member is the variable itself;
  // the object expression is not evaluated for a value (it has no side
  // effects in the cases met). It used to assert in cast<FieldDecl>.
  if (auto *VD = dyn_cast<VarDecl>(ME->getMemberDecl())) {
    auto *ref = DeclRefExpr::Create(
        Glob.CGM.getContext(), NestedNameSpecifierLoc(), SourceLocation(), VD,
        /*RefersToEnclosingVariableOrCapture=*/false, ME->getExprLoc(),
        VD->getType(), VK_LValue);
    return Visit(ref);
  }
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
    base = base.dereference(loc, builder);
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
      loc, OT, field, base.val,
      isa<clang::ReferenceType>(
          field->getType()->getUnqualifiedDesugaredType()));
}

mlir::Value MLIRScanner::GetAddressOfDerivedClass(
    mlir::Location loc, mlir::Value value, const CXXRecordDecl *DerivedClass,
    CastExpr::path_const_iterator Start, CastExpr::path_const_iterator End) {
  const ASTContext &Context = Glob.CGM.getContext();

  // Within a scalar-wrapper chain every class is the same scalar.
  if (Glob.getScalarWrapperField(DerivedClass))
    return castScalarWrapperRef(
        builder, loc, value,
        getMLIRType(Context.getLValueReferenceType(
            QualType(DerivedClass->getTypeForDecl(), 0))));

  SmallVector<const CXXRecordDecl *> ToBase = {DerivedClass};
  SmallVector<const CXXBaseSpecifier *> Bases;
  for (auto I = Start; I != End; I++) {
    const CXXBaseSpecifier *Base = *I;
    const auto *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());
    ToBase.push_back(BaseDecl);
    Bases.push_back(Base);
  }

  for (int i = ToBase.size() - 1; i > 0; i--) {
    const CXXBaseSpecifier *Base = Bases[i - 1];

    const auto *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());
    const auto *RD = ToBase[i - 1];
    // A step from a scalar-wrapper base to a wrapper derived class (the
    // wrapper part of a path that ends in a class that is not one) is the
    // identity: both are the same scalar.
    if (Glob.getScalarWrapperField(RD)) {
      value = castScalarWrapperRef(
          builder, loc, value,
          getMLIRType(Context.getLValueReferenceType(
              QualType(RD->getTypeForDecl(), 0))));
      continue;
    }
    // Get the layout.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    assert(!Base->isVirtual() && "Should not see virtual bases here!");

    // Add the offset.

    mlir::Type nt = getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
        QualType(RD->getTypeForDecl(), 0)));

    mlir::Value Offset = nullptr;
    if (isLLVMStructABI(RD, /*ST*/ nullptr)) {
      Offset = builder.create<arith::ConstantIntOp>(
          loc, -(ssize_t)Layout.getBaseClassOffset(BaseDecl).getQuantity(), 32);
    } else {
      Offset = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      bool found = false;
      for (auto f : RD->bases()) {
        if (f.getType().getTypePtr()->getUnqualifiedDesugaredType() ==
            Base->getType()->getUnqualifiedDesugaredType()) {
          found = true;
          break;
        }
        bool subType = false;
        mlir::Type nt = Glob.getMLIRType(f.getType(), &subType, false);
        Offset = builder.create<arith::SubIOp>(
            loc, Offset,
            builder.create<IndexCastOp>(
                loc, Offset.getType(),
                builder.create<polygeist::TypeSizeOp>(
                    loc, builder.getIndexType(), mlir::TypeAttr::get(nt))));
      }
      assert(found);
    }

    mlir::Value ptr = value;
    if (auto PT = dyn_cast<LLVM::LLVMPointerType>(ptr.getType())) {
      ptr = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(builder.getI8Type(), PT.getAddressSpace()),
          ptr);
    } else {
      unsigned memorySpace =
          ptr.getType().cast<mlir::MemRefType>().getMemorySpaceAsInt();
      ptr = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(builder.getI8Type(), memorySpace),
          ptr);
    }

    mlir::Value idx[] = {Offset};
    ptr = builder.create<LLVM::GEPOp>(loc, ptr.getType(), ptr, idx);

    if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(nt))
      value = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              PT.getElementType(),
              ptr.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          ptr);
    else
      value = builder.create<polygeist::Pointer2MemrefOp>(loc, nt, ptr);
  }

  return value;
}

mlir::Value MLIRScanner::GetAddressOfBaseClass(
    mlir::Location loc, mlir::Value value, const CXXRecordDecl *DerivedClass,
    ArrayRef<const clang::Type *> BaseTypes, ArrayRef<bool> BaseVirtuals) {
  const CXXRecordDecl *RD = DerivedClass;

  for (auto tup : llvm::zip(BaseTypes, BaseVirtuals)) {

    // Within a scalar-wrapper chain every class is the same scalar: from the
    // first wrapper on the path (the derived class itself, or a wrapper base
    // of a class that is not one) the rest of the path is the identity.
    if (Glob.getScalarWrapperField(RD))
      return castScalarWrapperRef(
          builder, loc, value,
          getMLIRType(Glob.CGM.getContext().getPointerType(
              QualType(BaseTypes.back(), 0))));

    auto BaseType = std::get<0>(tup);

    const auto *BaseDecl =
        cast<CXXRecordDecl>(BaseType->castAs<RecordType>()->getDecl());
    // Add the offset.

    mlir::Type nt = getMLIRType(
        Glob.CGM.getContext().getPointerType(QualType(BaseType, 0)));

    size_t fnum;
    bool subIndex = true;

    if (isLLVMStructABI(RD, /*ST*/ nullptr)) {
      auto &layout = Glob.CGM.getTypes().getCGRecordLayout(RD);
      if (std::get<1>(tup))
        fnum = layout.getVirtualBaseIndex(BaseDecl);
      else {
        if (!layout.hasNonVirtualBaseLLVMField(BaseDecl)) {
          subIndex = false;
        } else {
          fnum = layout.getNonVirtualBaseLLVMFieldNo(BaseDecl);
        }
      }
    } else {
      assert(!std::get<1>(tup) && "Should not see virtual bases here!");
      fnum = 0;
      bool found = false;
      for (auto f : RD->bases()) {
        if (f.getType().getTypePtr()->getUnqualifiedDesugaredType() ==
            BaseType->getUnqualifiedDesugaredType()) {
          found = true;
          break;
        }
        fnum++;
      }
      assert(found);
    }

    if (subIndex) {
      bool done = false;
      if (auto mt = dyn_cast<MemRefType>(value.getType())) {
        auto shape = std::vector<int64_t>(mt.getShape());
        if (shape.size() > 1) {
          shape.erase(shape.begin());
          auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                           MemRefLayoutAttrInterface(),
                                           mt.getMemorySpace());
          value = builder.create<polygeist::SubIndexOp>(loc, mt0, value,
                                                        getConstantIndex(fnum));
          done = true;
        } else {
          auto ET = mt.getElementType();
          assert(ET.isa<mlir::LLVM::LLVMStructType>() ||
                 ET.isa<mlir::LLVM::LLVMArrayType>());
          value = builder.create<polygeist::Memref2PointerOp>(
              loc, LLVM::LLVMPointerType::get(ET, mt.getMemorySpaceAsInt()),
              value);
        }
      }
      if (!done) {
        mlir::Value idx[] = {
            builder.create<arith::ConstantIntOp>(loc, 0, 32),
            builder.create<arith::ConstantIntOp>(loc, fnum, 32)};
        auto PT = value.getType().cast<LLVM::LLVMPointerType>();
        mlir::Type ET;
        if (auto ST =
                dyn_cast<mlir::LLVM::LLVMStructType>(PT.getElementType())) {
          ET = ST.getBody()[fnum];
        } else {
          ET = PT.getElementType()
                   .cast<mlir::LLVM::LLVMArrayType>()
                   .getElementType();
        }

        value = builder.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()), value,
            idx);
      }
    }

    auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(nt);
    if (auto opt = dyn_cast<mlir::LLVM::LLVMPointerType>(value.getType())) {
      if (!pt) {
        value = builder.create<polygeist::Pointer2MemrefOp>(loc, nt, value);
      } else {
        if (value.getType() != nt)
          value = builder.create<mlir::LLVM::BitcastOp>(loc, pt, value);
      }
    } else {
      assert(value.getType().isa<MemRefType>());
      if (pt) {
        value = builder.create<polygeist::Memref2PointerOp>(loc, pt, value);
      } else {
        if (value.getType() != nt) {
          auto anyPT = LLVM::LLVMPointerType::get(builder.getI8Type());
          value =
              builder.create<polygeist::Memref2PointerOp>(loc, anyPT, value);
          value = builder.create<polygeist::Pointer2MemrefOp>(loc, nt, value);
        }
      }
    }

    RD = BaseDecl;
  }

  return value;
}

ValueCategory MLIRScanner::VisitCastExpr(CastExpr *E) {
  auto loc = getMLIRLocation(E->getExprLoc());
  switch (E->getCastKind()) {

  case clang::CastKind::CK_NullToPointer: {
    auto llvmType = getMLIRType(E->getType());
    if (llvmType.isa<LLVM::LLVMPointerType>())
      return ValueCategory(builder.create<mlir::LLVM::ZeroOp>(loc, llvmType),
                           /*isReference*/ false);
    else
      return ValueCategory(
          builder.create<polygeist::Pointer2MemrefOp>(
              loc, llvmType,
              builder.create<mlir::LLVM::ZeroOp>(
                  loc, LLVM::LLVMPointerType::get(builder.getI8Type()))),
          false);
  }
  case clang::CastKind::CK_UserDefinedConversion: {
    return Visit(E->getSubExpr());
  }
  case clang::CastKind::CK_Dynamic: {
    E->dump();
    llvm_unreachable("dynamic cast not handled yet\n");
  }
  case clang::CastKind::CK_UncheckedDerivedToBase:
  case clang::CastKind::CK_DerivedToBase: {
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    assert(se.val);
    auto Derived =
        (E->isLValue() || E->isXValue())
            ? cast<CXXRecordDecl>(
                  E->getSubExpr()->getType()->castAs<RecordType>()->getDecl())
            : E->getSubExpr()->getType()->getPointeeCXXRecordDecl();
    SmallVector<const clang::Type *> BaseTypes;
    SmallVector<bool> BaseVirtual;
    for (auto B : E->path()) {
      BaseTypes.push_back(B->getType().getTypePtr());
      BaseVirtual.push_back(B->isVirtual());
    }

    mlir::Value val =
        GetAddressOfBaseClass(loc, se.val, Derived, BaseTypes, BaseVirtual);
    if (E->getCastKind() != clang::CastKind::CK_UncheckedDerivedToBase &&
        !isa<CXXThisExpr>(E->IgnoreParens()) &&
        !Glob.getScalarWrapperField(Derived)) {
      mlir::Value ptr = val;
      if (auto MT = dyn_cast<MemRefType>(ptr.getType()))
        ptr = builder.create<polygeist::Memref2PointerOp>(
            loc, LLVM::LLVMPointerType::get(MT.getElementType()), ptr);
      mlir::Value nullptr_llvm =
          builder.create<mlir::LLVM::ZeroOp>(loc, ptr.getType());
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, ptr, nullptr_llvm);
      if (auto MT = dyn_cast<MemRefType>(ptr.getType()))
        nullptr_llvm =
            builder.create<polygeist::Pointer2MemrefOp>(loc, MT, nullptr_llvm);
      val = builder.create<arith::SelectOp>(loc, ne, ptr, nullptr_llvm);
    }
    return ValueCategory(val, se.isReference);
  }
  case clang::CastKind::CK_BaseToDerived: {
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    assert(se.val);
    auto Derived =
        (E->isLValue() || E->isXValue())
            ? cast<CXXRecordDecl>(E->getType()->castAs<RecordType>()->getDecl())
            : E->getType()->getPointeeCXXRecordDecl();
    mlir::Value val = GetAddressOfDerivedClass(loc, se.val, Derived,
                                               E->path_begin(), E->path_end());
    /*
    if (ShouldNullCheckClassCastValue(E)) {
        mlir::Value ptr = val;
        if (auto MT = dyn_cast<MemRefType>(ptr.getType()))
            ptr = builder.create<polygeist::Memref2PointerOp>(loc,
    LLVM::LLVMPointerType::get(MT.getElementType()), ptr); auto nullptr_llvm =
    builder.create<mlir::LLVM::ZeroOp>(loc, ptr.getType()); auto ne =
    builder.create<mlir::LLVM::ICmpOp>( loc, mlir::LLVM::ICmpPredicate::ne, ptr,
    nullptr_llvm); if (auto MT = dyn_cast<MemRefType>(ptr.getType()))
           nullptr_llvm = builder.create<polygeist::Pointer2MemrefOp>(loc, MT,
    nullptr_llvm); val = builder.create<arith::SelectOp>(loc, ne, val,
    nullptr_llvm);
    }
    */
    return ValueCategory(val, se.isReference);
  }
  case clang::CastKind::CK_LValueBitCast: {
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    auto scalar = se.val;
    assert(se.isReference);
    if (auto spt = dyn_cast<mlir::LLVM::LLVMPointerType>(scalar.getType())) {
      auto nt = getMLIRType(E->getType());
      LLVM::LLVMPointerType pt = dyn_cast<LLVM::LLVMPointerType>(nt);
      if (!pt) {
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(loc, nt, scalar),
            false);
      }
      pt = LLVM::LLVMPointerType::get(pt.getElementType(),
                                      spt.getAddressSpace());
      auto nval = builder.create<mlir::LLVM::BitcastOp>(loc, pt, scalar);
      return ValueCategory(nval, /*isReference*/ true);
    }
    if (!scalar.getType().isa<mlir::MemRefType>()) {
      E->dump();
      E->getType()->dump();
      llvm::errs() << "scalar: " << scalar << "\n";
    }
    auto ut = scalar.getType().cast<mlir::MemRefType>();
    auto mlirty =
        getMLIRType(Glob.CGM.getContext().getLValueReferenceType(E->getType()));

    if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(mlirty)) {
      return ValueCategory(
          builder.create<mlir::polygeist::Memref2PointerOp>(loc, PT, scalar),
          /*isReference*/ true);
    } else if (auto mt = dyn_cast<mlir::MemRefType>(mlirty)) {
      auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      ut.getMemorySpace());
      if (ut.getShape().size() == mt.getShape().size() + 1 &&
          ut.getElementType() == mt.getElementType()) {
        return ValueCategory(builder.create<mlir::polygeist::SubIndexOp>(
                                 loc, ty, scalar, getConstantIndex(0)),
                             /*isReference*/ true);
      }
      bool badShape = ut.getShape().size() != mt.getShape().size();
      if (!badShape)
        for (size_t i = 1; i < ut.getShape().size(); i++) {
          if (ut.getShape()[i] != mt.getShape()[i]) {
            badShape = true;
            break;
          }
        }
      if (badShape || ut.getElementType() != mt.getElementType()) {
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(
                loc, ty,
                builder.create<polygeist::Memref2PointerOp>(
                    loc, LLVM::LLVMPointerType::get(builder.getI8Type()),
                    scalar)),
            /*isReference*/ true);
      }
      return ValueCategory(builder.create<memref::CastOp>(loc, ty, scalar),
                           /*isReference*/ true);
    } else {
      E->dump();
      E->getType()->dump();
      llvm::errs() << " scalar: " << scalar << " mlirty: " << mlirty << "\n";
      llvm_unreachable("illegal type for cast");
      llvm_unreachable("illegal type for cast");
    }
  }
  case clang::CastKind::CK_BitCast: {

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getIdentifier() &&
              sr->getDecl()->getName() == "polybench_alloc_data") {
            if (auto mt =
                    dyn_cast<mlir::MemRefType>(getMLIRType(E->getType()))) {
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
                    dyn_cast<mlir::MemRefType>(getMLIRType(E->getType()))) {
              auto shape = std::vector<int64_t>(mt.getShape());

              auto elemSize = getTypeSize(
                  loc, cast<clang::PointerType>(
                           E->getType()->getUnqualifiedDesugaredType())
                           ->getPointeeType());
              mlir::Value allocSize = builder.create<IndexCastOp>(
                  loc, mlir::IndexType::get(builder.getContext()),
                  Visit(CI->getArg(0)).getValue(loc, builder));
              if (sr->getDecl()->getName() == "calloc") {
                allocSize = builder.create<MulIOp>(
                    loc, allocSize,
                    builder.create<arith::IndexCastOp>(
                        loc, mlir::IndexType::get(builder.getContext()),
                        Visit(CI->getArg(1)).getValue(loc, builder)));
              }
              mlir::Value args[1] = {
                  builder.create<DivUIOp>(loc, allocSize, elemSize)};
              auto alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
              if (sr->getDecl()->getName() == "calloc") {
                mlir::Value val = alloc;
                if (val.getType().isa<MemRefType>()) {
                  val = builder.create<polygeist::Memref2PointerOp>(
                      loc, LLVM::LLVMPointerType::get(builder.getI8Type()),
                      val);
                } else {
                  val = builder.create<LLVM::BitcastOp>(
                      loc,
                      LLVM::LLVMPointerType::get(
                          builder.getI8Type(),
                          val.getType()
                              .cast<LLVM::LLVMPointerType>()
                              .getAddressSpace()),
                      val);
                }
                auto i8_0 = builder.create<arith::ConstantIntOp>(loc, 0, 8);
                auto sizev = builder.create<arith::IndexCastOp>(
                    loc, builder.getI64Type(), allocSize);
                auto falsev =
                    builder.create<arith::ConstantIntOp>(loc, false, 1);
                builder.create<LLVM::MemsetOp>(loc, val, i8_0, sizev, falsev);
              }
              return ValueCategory(alloc, /*isReference*/ false);
            }
        }
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    auto scalar = se.getValue(loc, builder);
    if (auto spt = dyn_cast<mlir::LLVM::LLVMPointerType>(scalar.getType())) {
      auto nt = getMLIRType(E->getType());
      LLVM::LLVMPointerType pt = dyn_cast<LLVM::LLVMPointerType>(nt);
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

    if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(mlirty)) {
      return ValueCategory(
          builder.create<mlir::polygeist::Memref2PointerOp>(loc, PT, scalar),
          /*isReference*/ false);
    } else if (auto mt = dyn_cast<mlir::MemRefType>(mlirty)) {
      auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      ut.getMemorySpace());
      if (ut.getShape().size() == mt.getShape().size() + 1 &&
          ut.getElementType() == mt.getElementType()) {
        return ValueCategory(builder.create<mlir::polygeist::SubIndexOp>(
                                 loc, ty, scalar, getConstantIndex(0)),
                             /*isReference*/ false);
      }
      bool badShape = ut.getShape().size() != mt.getShape().size();
      if (!badShape)
        for (size_t i = 1; i < ut.getShape().size(); i++) {
          if (ut.getShape()[i] != mt.getShape()[i]) {
            badShape = true;
            break;
          }
        }
      if (badShape || ut.getElementType() != mt.getElementType()) {
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(
                loc, ty,
                builder.create<polygeist::Memref2PointerOp>(
                    loc, LLVM::LLVMPointerType::get(builder.getI8Type()),
                    scalar)),
            /*isReference*/ false);
      }
      return ValueCategory(builder.create<memref::CastOp>(loc, ty, scalar),
                           /*isReference*/ false);
    } else {
      E->dump();
      E->getType()->dump();
      llvm::errs() << " scalar: " << scalar << " mlirty: " << mlirty << "\n";
      llvm_unreachable("illegal type for cast");
      llvm_unreachable("illegal type for cast");
    }
  }
  case clang::CastKind::CK_LValueToRValue: {
    if (auto dr = dyn_cast<DeclRefExpr>(E->getSubExpr())) {
      if (auto VD = dyn_cast<VarDecl>(dr->getDecl()->getCanonicalDecl())) {
        // A const variable with a constant initialiser reads as that
        // constant (clang's own constant evaluation): static const members
        // such as `static const int F = W - I`, const globals, constexpr
        // scalar wrappers.
        if (mlir::Value c = tryEmitConstVarValue(loc, VD, E->getType()))
          return ValueCategory(c, /*isReference*/ false);
        if (NOUR_Constant == dr->isNonOdrUse()) {
          auto VarD = cast<VarDecl>(VD);
          if (!VarD->getInit()) {
            E->dump();
            VarD->dump();
          }
          assert(VarD->getInit());
          return Visit(VarD->getInit());
        }
      }
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

    auto lres = prev.getValue(loc, builder);
    if (!prev.isReference) {
      E->dump();
      lres.dump();
    }
    assert(prev.isReference);
    return ValueCategory(lres, /*isReference*/ false);
  }
  case clang::CastKind::CK_LValueToRValueBitCast: {
    // __builtin_bit_cast(To, lvalue): the object representation reread as
    // `To`. Between scalars of one width that is an arith.bitcast.
    // (--scalar-wrapper-abi only; unhandled as before when the option is off.)
    if (!ScalarWrapperABI) {
      if (EmittingFunctionDecl)
        EmittingFunctionDecl->dump();
      E->dump();
      llvm_unreachable("unhandled cast");
    }
    auto se = Visit(E->getSubExpr());
    assert(se.val && "bit_cast operand");
    mlir::Value v = se.getValue(loc, builder);
    mlir::Type to = getMLIRType(E->getType());
    if (v.getType() == to)
      return ValueCategory(v, /*isReference*/ false);
    if (v.getType().isIntOrFloat() && to.isIntOrFloat() &&
        v.getType().getIntOrFloatBitWidth() == to.getIntOrFloatBitWidth())
      return ValueCategory(builder.create<arith::BitcastOp>(loc, to, v),
                           /*isReference*/ false);
    E->dump();
    llvm::errs() << "from: " << v.getType() << " to: " << to << "\n";
    llvm_unreachable("unsupported __builtin_bit_cast");
  }
  case clang::CastKind::CK_IntegralToFloating: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    auto ty = getMLIRType(E->getType()).cast<mlir::FloatType>();
    bool signedType = isSignedIntegerQualType(E->getSubExpr()->getType());
    if (signedType)
      return ValueCategory(
          builder.create<mlir::arith::SIToFPOp>(loc, ty, scalar),
          /*isReference*/ false);
    else
      return ValueCategory(
          builder.create<mlir::arith::UIToFPOp>(loc, ty, scalar),
          /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingToIntegral: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    auto ty = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = isSignedIntegerQualType(E->getType());
    if (signedType)
      return ValueCategory(
          builder.create<mlir::arith::FPToSIOp>(loc, ty, scalar),
          /*isReference*/ false);
    else
      return ValueCategory(
          builder.create<mlir::arith::FPToUIOp>(loc, ty, scalar),
          /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    assert(scalar);
    auto postTy = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    if (scalar.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return ValueCategory(
          builder.create<mlir::LLVM::PtrToIntOp>(loc, postTy, scalar),
          /*isReference*/ false);
    }
    if (scalar.getType().isa<mlir::IndexType>() ||
        postTy.isa<mlir::IndexType>()) {
      return ValueCategory(builder.create<IndexCastOp>(loc, postTy, scalar),
                           false);
    }
    if (!scalar.getType().isa<mlir::IntegerType>()) {
      E->dump();
      llvm::errs() << " scalar: " << scalar << "\n";
    }
    auto prevTy = scalar.getType().cast<mlir::IntegerType>();
    bool signedType = isSignedIntegerQualType(E->getSubExpr()->getType());

    if (prevTy == postTy)
      return ValueCategory(scalar, /*isReference*/ false);
    if (prevTy.getWidth() < postTy.getWidth()) {
      if (signedType) {
        if (auto CI = scalar.getDefiningOp<arith::ConstantIntOp>()) {
          return ValueCategory(
              builder.create<arith::ConstantOp>(
                  loc, postTy,
                  mlir::IntegerAttr::get(
                      postTy, CI.getValue().cast<IntegerAttr>().getValue().sext(
                                  postTy.getWidth()))),
              /*isReference*/ false);
        }
        return ValueCategory(
            builder.create<arith::ExtSIOp>(loc, postTy, scalar),
            /*isReference*/ false);
      } else {
        if (auto CI = scalar.getDefiningOp<arith::ConstantIntOp>()) {
          return ValueCategory(
              builder.create<arith::ConstantOp>(
                  loc, postTy,
                  mlir::IntegerAttr::get(
                      postTy, CI.getValue().cast<IntegerAttr>().getValue().zext(
                                  postTy.getWidth()))),
              /*isReference*/ false);
        }
        return ValueCategory(
            builder.create<arith::ExtUIOp>(loc, postTy, scalar),
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
      return ValueCategory(builder.create<arith::TruncIOp>(loc, postTy, scalar),
                           /*isReference*/ false);
    }
  }
  case clang::CastKind::CK_FloatingCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
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
    if (prevTy.getWidth() == postTy.getWidth()) {
      // f16 <-> bf16: neither extf nor truncf applies; go through f32.
      scalar = builder.create<arith::ExtFOp>(loc, builder.getF32Type(), scalar);
      return ValueCategory(builder.create<arith::TruncFOp>(loc, postTy, scalar),
                           /*isReference*/ false);
    }
    if (prevTy.getWidth() < postTy.getWidth()) {
      return ValueCategory(builder.create<arith::ExtFOp>(loc, postTy, scalar),
                           /*isReference*/ false);
    } else {
      return ValueCategory(builder.create<arith::TruncFOp>(loc, postTy, scalar),
                           /*isReference*/ false);
    }
  }
  case clang::CastKind::CK_FloatingComplexCast: {
    auto sub = Visit(E->getSubExpr());
    auto complex = sub.val;
    mlir::FloatType prevScalarTy;
    mlir::FloatType postScalarTy;
    auto prevTy = complex.getType();
    auto postTy = getMLIRType(E->getType());
    if (auto mt = dyn_cast<MemRefType>(postTy)) {
      postScalarTy = mt.getElementType().cast<mlir::FloatType>();
    } else if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(postTy)) {
      postScalarTy = ST.getBody()[0].cast<mlir::FloatType>();
    } else {
      llvm_unreachable("unexpected complex type\n");
    }
    if (auto mt = dyn_cast<MemRefType>(prevTy)) {
      prevScalarTy = mt.getElementType().cast<mlir::FloatType>();
    } else if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(prevTy)) {
      prevScalarTy = ST.getBody()[0].cast<mlir::FloatType>();
    } else {
      llvm_unreachable("unexpected complex type\n");
    }
    auto castScalar = [&](mlir::Value complex, int fnum) -> mlir::Value {
      mlir::Value scalar = getComplexPart(loc, complex, fnum);
      assert(scalar);
      if (prevScalarTy.getWidth() < postScalarTy.getWidth()) {
        return builder.create<arith::ExtFOp>(loc, postScalarTy, scalar);
      } else {
        return builder.create<arith::TruncFOp>(loc, postScalarTy, scalar);
      }
    };
    auto real = castScalar(complex, 0);
    auto imag = castScalar(complex, 1);
    return createComplexFloat(loc, real, imag, E->getType());
  }
  case clang::CastKind::CK_ArrayToPointerDecay: {
    ValueCategory array = Visit(E->getSubExpr());
    // An array lvalue reached through a pointer to an array (`*__begin` of a
    // range-based for over `int (&)[4][8]`, bound to `auto &row`) is that
    // pointer, a memref one rank above the array (memref<?x8xi32> for an
    // int[8]); the array is the pointer's row 0. Decaying the pointer itself
    // kept the extra rank, and a later `*p` asserted in getValue (CIRCT W9
    // round 8 review defect 8, reproducer rf2.cpp: a nested range-for over
    // the rows of a 2-D array reference).
    if (auto mt = array.val ? dyn_cast<MemRefType>(array.val.getType())
                            : MemRefType()) {
      // The array's own extents; the memref must be exactly one leading
      // (pointer) dimension above them.
      // The pointer's dimension is dynamic, and the array's elements are
      // scalars of the memref's element type: an array of structs
      // (PE pes[3][3] as memref<3x3x3xi32>, the struct's fields the last
      // dimension) is not a pointer to an array, whatever its sizes.
      llvm::SmallVector<int64_t, 4> extents;
      bool constant = true;
      clang::QualType elem = E->getSubExpr()->getType();
      while (const clang::ArrayType *AT = elem->getAsArrayTypeUnsafe()) {
        auto *CAT = dyn_cast<clang::ConstantArrayType>(AT);
        if (!CAT) {
          constant = false;
          break;
        }
        extents.push_back(CAT->getSize().getSExtValue());
        elem = AT->getElementType();
      }
      bool scalarElems = constant && (elem->isBuiltinType() ||
                                      elem->isEnumeralType()) &&
                         getMLIRType(elem) == mt.getElementType();
      if (array.isReference && scalarElems && !extents.empty() &&
          mt.getRank() == static_cast<int64_t>(extents.size()) + 1 &&
          ShapedType::isDynamic(mt.getShape()[0]) &&
          mt.getShape().drop_front() == llvm::ArrayRef<int64_t>(extents) &&
          mt.getLayout().isIdentity()) {
        auto row = mlir::MemRefType::get(mt.getShape().drop_front(),
                                         mt.getElementType(),
                                         MemRefLayoutAttrInterface(),
                                         mt.getMemorySpace());
        array = ValueCategory(builder.create<polygeist::SubIndexOp>(
                                  loc, row, array.val, getConstantIndex(0)),
                              /*isReference*/ true);
      }
    }
    return CommonArrayToPointer(loc, array);

#if 0
    auto mt = scalar.val.getType().cast<mlir::MemRefType>();
    auto shape2 = std::vector<int64_t>(mt.getShape());
    if (shape2.size() == 0) {
      E->dump();
      //nex.dump();
      assert(0);
    }
    shape2[0] = ShapedType::kDynamic;
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
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(scalar.getType())) {
      scalar = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getElementType()), scalar);
    }
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(scalar.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, scalar, nullptr_llvm);
      return ValueCategory(ne, /*isReference*/ false);
    }
    function.dump();
    llvm::errs() << "scalar: " << scalar << "\n";
    E->dump();
    llvm_unreachable("unhandled ptrtobool cast");
  }
  case clang::CastKind::CK_PointerToIntegral: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(scalar.getType())) {
      scalar = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getElementType()), scalar);
    }
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(scalar.getType())) {
      auto mlirType = getMLIRType(E->getType());
      auto val = builder.create<mlir::LLVM::PtrToIntOp>(loc, mlirType, scalar);
      return ValueCategory(val, /*isReference*/ false);
    }
    function.dump();
    llvm::errs() << "scalar: " << scalar << "\n";
    E->dump();
    llvm_unreachable("unhandled ptrtoint cast");
  }
  case clang::CastKind::CK_IntegralToBoolean: {
    auto res = Visit(E->getSubExpr()).getValue(loc, builder);
    auto prevTy = res.getType().cast<mlir::IntegerType>();
    res = builder.create<arith::CmpIOp>(
        loc, CmpIPredicate::ne, res,
        builder.create<ConstantIntOp>(loc, 0, prevTy));
    auto postTy = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = isSignedIntegerQualType(E->getType());
    if (postTy.getWidth() > 1) {
      if (signedType) {
        res = builder.create<arith::ExtSIOp>(loc, postTy, res);
      } else {
        res = builder.create<arith::ExtUIOp>(loc, postTy, res);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingToBoolean: {
    auto res = Visit(E->getSubExpr()).getValue(loc, builder);
    auto prevTy = res.getType().cast<mlir::FloatType>();
    auto postTy = getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = isSignedIntegerQualType(E->getType());
    auto Zero = builder.create<ConstantFloatOp>(
        loc, APFloat::getZero(prevTy.getFloatSemantics()), prevTy);
    res = builder.create<arith::CmpFOp>(loc, CmpFPredicate::UNE, res, Zero);
    if (1 < postTy.getWidth()) {
      if (signedType) {
        res = builder.create<arith::ExtSIOp>(loc, postTy, res);
      } else {
        res = builder.create<arith::ExtUIOp>(loc, postTy, res);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralToPointer: {
    auto vc = Visit(E->getSubExpr());
    if (!vc.val) {
      E->dump();
    }
    assert(vc.val);
    auto res = vc.getValue(loc, builder);
    auto postTy = getMLIRType(E->getType());
    if (postTy.isa<LLVM::LLVMPointerType>())
      res = builder.create<LLVM::IntToPtrOp>(loc, postTy, res);
    else {
      assert(postTy.isa<MemRefType>());
      res = builder.create<LLVM::IntToPtrOp>(
          loc, LLVM::LLVMPointerType::get(builder.getI8Type()), res);
      res = builder.create<polygeist::Pointer2MemrefOp>(loc, postTy, res);
    }
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingRealToComplex: {
    auto sub = Visit(E->getSubExpr());
    auto complex = sub.val;
    auto convertedType = getMLIRType(E->getType());
    auto real = sub.getValue(loc, builder);
    mlir::FloatType fty;
    if (auto mt = dyn_cast<MemRefType>(convertedType)) {
      auto elty = mt.getElementType();
      if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(elty)) {
        fty = ST.getBody()[0].cast<FloatType>();
      } else {
        fty = elty.cast<FloatType>();
      }
    } else if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(convertedType)) {
      fty = ST.getBody()[0].cast<FloatType>();
    } else {
      llvm_unreachable("unexpected complex type");
    }
    auto zero = builder.create<ConstantFloatOp>(
        loc, APFloat(fty.getFloatSemantics(), "0"), fty);
    return createComplexFloat(loc, real, zero, E->getType());
  }

  default:
    if (EmittingFunctionDecl)
      EmittingFunctionDecl->dump();
    E->dump();
    llvm_unreachable("unhandled cast");
  }
}

ValueCategory
MLIRScanner::VisitConditionalOperator(clang::ConditionalOperator *E) {
  auto loc = getMLIRLocation(E->getExprLoc());
  auto cond = Visit(E->getCond()).getValue(loc, builder);
  assert(cond != nullptr);
  if (auto mt = dyn_cast<mlir::MemRefType>(cond.getType())) {
    cond = builder.create<polygeist::Memref2PointerOp>(
        loc,
        LLVM::LLVMPointerType::get(mt.getElementType(),
                                   mt.getMemorySpaceAsInt()),
        cond);
  }
  if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
    auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
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

  builder.setInsertionPointToStart(&ifOp.getElseRegion().back());

  auto falseExpr = Visit(E->getFalseExpr());

  bool isReference = E->isLValue() || E->isXValue() ||
                     (trueExpr.isReference && falseExpr.isReference);

  builder.setInsertionPointToEnd(&ifOp.getThenRegion().back());

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
        if (auto mt = dyn_cast<MemRefType>(trueExpr.val.getType()))
          if (mt.getShape().size() != 1) {
            E->dump();
            E->getTrueExpr()->dump();
            llvm::errs() << " trueExpr: " << trueExpr.val << "\n";
            assert(0);
          }
      truev = trueExpr.getValue(loc, builder);
    }
    assert(truev != nullptr);
    truearray.push_back(truev);
    builder.create<mlir::scf::YieldOp>(loc, truearray);
  }

  builder.setInsertionPointToEnd(&ifOp.getElseRegion().back());

  std::vector<mlir::Value> falsearray;
  if (!E->getType()->isVoidType()) {
    mlir::Value falsev;
    if (isReference) {
      assert(falseExpr.isReference);
      falsev = falseExpr.val;
    } else
      falsev = falseExpr.getValue(loc, builder);
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
  if (types.size() == 0)
    return ValueCategory();
  return ValueCategory(newIfOp.getResult(0), /*isReference*/ isReference);
}

ValueCategory MLIRScanner::VisitSizeOfPackExpr(SizeOfPackExpr *expr) {
  const auto loc = getMLIRLocation(expr->getExprLoc());
  const auto val = expr->getPackLength();
  const auto ty = getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(builder.create<arith::ConstantIntOp>(loc, val, ty),
                       /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitStmtExpr(clang::StmtExpr *stmt) {
  ValueCategory off = nullptr;
  for (auto a : stmt->getSubStmt()->children()) {
    off = Visit(a);
  }
  return off;
}

mlir::Value MLIRASTConsumer::CallMalloc(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value arg) {
  mlir::OpBuilder fbuilder(module->getContext());
  fbuilder.setInsertionPointToStart(module->getBody());
  std::string name = "malloc";
  auto ctx = module->getContext();
  mlir::Type types[] = {mlir::IntegerType::get(ctx, 64)};
  if (CStyleMemRef) {
    if (functions.find(name) == functions.end()) {
      auto funcType = fbuilder.getFunctionType(
          types,
          mlir::MemRefType::get({ShapedType::kDynamic}, builder.getI8Type()));
      functions[name] =
          fbuilder.create<mlir::func::FuncOp>(module->getLoc(), name, funcType);
    }
    std::vector args = {arg};
    auto callOp =
        builder.create<mlir::func::CallOp>(loc, functions[name], args);

    return callOp->getResult(0);
  } else {

    if (llvmFunctions.find(name) == llvmFunctions.end()) {
      auto llvmFnType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMPointerType::get(builder.getContext()), types, false);
      LLVM::Linkage lnk = LLVM::Linkage::External;
      llvmFunctions[name] = fbuilder.create<LLVM::LLVMFuncOp>(
          module->getLoc(), name, llvmFnType, lnk);
    }
    auto i64 = mlir::IntegerType::get(ctx, 64);
    arg = builder.create<IndexCastOp>(loc, i64, arg);
    std::vector args = {arg};
    auto callOp =
        builder.create<mlir::LLVM::CallOp>(loc, llvmFunctions[name], args);
    return callOp->getResult(0);
  }
}
mlir::LLVM::LLVMFuncOp MLIRASTConsumer::GetOrCreateFreeFunction() {
  std::string name = "free";
  if (llvmFunctions.find(name) != llvmFunctions.end()) {
    return llvmFunctions[name];
  }
  auto ctx = module->getContext();
  mlir::Type types[] = {LLVM::LLVMPointerType::get(ctx)};
  auto llvmFnType =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), types, false);

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

  if (name != "malloc" && name != "free")
    name = (PrefixABI + name);

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
MLIRASTConsumer::GetOrCreateLLVMGlobal(const ValueDecl *FD,
                                       std::string prefix) {
  std::string name = prefix + CGM.getMangledName(FD).str();

  name = (PrefixABI + name);

  if (llvmGlobals.find(name) != llvmGlobals.end()) {
    return llvmGlobals[name];
  }

  LLVM::Linkage lnk;
  auto VD = dyn_cast<VarDecl>(FD);
  if (!VD)
    FD->dump();
  VD = VD->getCanonicalDecl();

  auto linkage = CGM.getLLVMLinkageVarDefinition(VD);
  switch (linkage) {
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

  auto rt = getMLIRType(FD->getType());

  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());

  auto glob = builder.create<LLVM::GlobalOp>(
      module->getLoc(), rt, /*constant*/ false, lnk, name, mlir::Attribute());

  if (VD->getInit() ||
      VD->isThisDeclarationADefinition() == VarDecl::Definition ||
      VD->isThisDeclarationADefinition() == VarDecl::TentativeDefinition) {
    Block *blk = new Block();
    builder.setInsertionPointToStart(blk);
    mlir::Value res;
    if (auto init = VD->getInit()) {
      MLIRScanner ms(*this, module, LTInfo);
      ms.setEntryAndAllocBlock(blk);
      res = ms.Visit(const_cast<Expr *>(init))
                .getValue(getMLIRLocation(init->getBeginLoc()), builder);
    } else {
      res = builder.create<polygeist::UndefOp>(module->getLoc(), rt);
    }
    bool legal = true;
    for (Operation &op : *blk) {
      auto iface = dyn_cast<MemoryEffectOpInterface>(op);
      if (!iface || !iface.hasNoEffect()) {
        legal = false;
        break;
      }
    }
    if (legal) {
      builder.create<LLVM::ReturnOp>(module->getLoc(),
                                     std::vector<mlir::Value>({res}));
      glob.getInitializerRegion().push_back(blk);
    } else {
      Block *blk2 = new Block();
      builder.setInsertionPointToEnd(blk2);
      mlir::Value nres =
          builder.create<polygeist::UndefOp>(module->getLoc(), rt);
      builder.create<LLVM::ReturnOp>(module->getLoc(),
                                     std::vector<mlir::Value>({nres}));
      glob.getInitializerRegion().push_back(blk2);

      builder.setInsertionPointToStart(module->getBody());
      auto funcName = name + "@init";
      LLVM::GlobalCtorsOp ctors = nullptr;
      for (auto &op : *module->getBody()) {
        if (auto c = dyn_cast<LLVM::GlobalCtorsOp>(&op)) {
          ctors = c;
        }
      }
      SmallVector<mlir::Attribute> funcs;
      funcs.push_back(FlatSymbolRefAttr::get(module->getContext(), funcName));
      SmallVector<mlir::Attribute> idxs;
      idxs.push_back(builder.getI32IntegerAttr(0));
      if (ctors) {
        for (auto f : ctors.getCtors())
          funcs.push_back(f);
        for (auto v : ctors.getPriorities())
          idxs.push_back(v);
        ctors->erase();
      }

      builder.create<LLVM::GlobalCtorsOp>(module->getLoc(),
                                          builder.getArrayAttr(funcs),
                                          builder.getArrayAttr(idxs));

      auto llvmFnType = LLVM::LLVMFunctionType::get(
          mlir::LLVM::LLVMVoidType::get(module->getContext()),
          ArrayRef<mlir::Type>(), false);

      auto func = builder.create<LLVM::LLVMFuncOp>(
          module->getLoc(), funcName, llvmFnType, LLVM::Linkage::Private);
      func.getRegion().push_back(blk);
      builder.setInsertionPointToEnd(blk);
      builder.create<LLVM::StoreOp>(
          module->getLoc(), res,
          builder.create<LLVM::AddressOfOp>(module->getLoc(), glob));
      builder.create<LLVM::ReturnOp>(module->getLoc(), ArrayRef<mlir::Value>());
    }
  }
  if (lnk == LLVM::Linkage::Private || lnk == LLVM::Linkage::Internal) {
    SymbolTable::setSymbolVisibility(glob,
                                     mlir::SymbolTable::Visibility::Private);
  }
  cgeist::hls::onGlobal(VD, glob);
  return llvmGlobals[name] = glob;
}

std::pair<mlir::memref::GlobalOp, bool>
MLIRASTConsumer::GetOrCreateGlobal(const ValueDecl *FD, std::string prefix,
                                   bool tryInit) {
  std::string name = prefix + CGM.getMangledName(FD).str();

  name = (PrefixABI + name);

  // A global an earlier translation unit created: this unit's declaration
  // must reach the HLS pragmas that name it (an ARRAY_PARTITION in a unit
  // that only declares the global `extern`, CIRCT OPEN_ISSUES O75, flowgnn
  // gin), and when the earlier unit only declared it and this one defines
  // it, the global gets this unit's definition (and initializer).
  mlir::memref::GlobalOp pendingDefinition;
  if (globals.find(name) != globals.end()) {
    auto entry = globals[name];
    const auto *VD0 = dyn_cast<VarDecl>(FD);
    if (!VD0)
      return entry;
    cgeist::hls::onGlobal(VD0->getCanonicalDecl(), entry.first);
    const VarDecl *def = VD0->getDefinition();
    if (!def || entry.first.getInitialValue())
      return entry;
    pendingDefinition = entry.first;
  }

  bool isArray = false;
  auto rt = getMLIRType(FD->getType(), &isArray);
  unsigned memspace = 0;

  mlir::MemRefType mr =
      getMLIRType(CGM.getContext().getLValueReferenceType(FD->getType()))
          .cast<MemRefType>();
  std::vector<int64_t> shape(mr.getShape());
  if (shape[0] == ShapedType::kDynamic)
    shape[0] = 1;
  mr = mlir::MemRefType::get(shape, mr.getElementType(),
                             MemRefLayoutAttrInterface(),
                             wrapIntegerMemorySpace(memspace, mr.getContext()));
  // A definition whose type differs from the earlier declaration's (an
  // `extern T a[];` of unknown size): the uses already have that type.
  if (pendingDefinition && pendingDefinition.getType() != mr)
    return globals[name];

  mlir::SymbolTable::Visibility lnk;
  mlir::Attribute initial_value;

  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());

  auto VD = dyn_cast<VarDecl>(FD);
  if (!VD)
    FD->dump();
  VD = VD->getCanonicalDecl();
  // The declaration that defines the global in this unit, when there is one:
  // the canonical (first) declaration is an `extern` one when a header
  // declares the global before the unit defines it, and its definition and
  // initializer must not be lost (`extern int W[4]; int W[4] = {...};` gave a
  // bare declaration; W9r7 review defect 2).
  const VarDecl *DefVD = VD->getDefinition();
  if (!DefVD)
    DefVD = VD->getActingDefinition();
  if (!DefVD)
    DefVD = VD;

  if (DefVD->isThisDeclarationADefinition() == VarDecl::Definition) {
    initial_value = builder.getUnitAttr();
  } else if (DefVD->isThisDeclarationADefinition() ==
             VarDecl::TentativeDefinition) {
    initial_value = builder.getUnitAttr();
  }

  switch (CGM.getLLVMLinkageVarDefinition(DefVD)) {
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

  mlir::memref::GlobalOp globalOp = pendingDefinition;
  if (globalOp) {
    if (initial_value)
      globalOp.setInitialValueAttr(initial_value);
  } else {
    globalOp = builder.create<mlir::memref::GlobalOp>(
        module->getLoc(), builder.getStringAttr(name),
        /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(mr),
        initial_value, mlir::UnitAttr(), /*alignment*/ nullptr);
    SymbolTable::setSymbolVisibility(globalOp, lnk);
  }

  // TODO we should probably set the mem spaces here? But I am not sure that
  // works well with our one-module representation because the host-side "stubs"
  // have the default 0 memespace and it is only the device side globals that
  // should have the gpu specific memspace
  if (VD->hasAttr<CUDAConstantAttr>()) {
    globalOp->setAttr("polygeist.cuda_constant", builder.getUnitAttr());
  } else if (VD->hasAttr<CUDADeviceAttr>()) {
    globalOp->setAttr("polygeist.cuda_device", builder.getUnitAttr());
  }

  globals[name] = std::make_pair(globalOp, isArray);
  cgeist::hls::onGlobal(VD, globalOp);

  if (tryInit)
    if (auto init = initSkippingCleanups(DefVD)) {
      MLIRScanner ms(*this, module, LTInfo);
      mlir::Block *B = new Block();
      ms.setEntryAndAllocBlock(B);
      OpBuilder builder(module->getContext());
      builder.setInsertionPointToEnd(B);
      auto op = builder.create<memref::AllocaOp>(module->getLoc(), mr);

      bool initialized = false;
      // clang's constant evaluation first (--scalar-wrapper-abi): it folds
      // `static const int F = W - I`, scalar-wrapper globals with constexpr
      // constructors and their arrays (`const ap_fixed tbl[4] = {...}`
      // becomes a dense initial value), which the expression visitor below
      // cannot.
      if (auto dense = evaluateConstantInitializer(VD, mr)) {
        initial_value = dense;
        initialized = true;
      } else if (auto dense =
                     evaluateXlxStringInitializer(init, VD->getType(), mr)) {
        // --xlx-headers: a table of string constructions.
        initial_value = dense;
        initialized = true;
      }
      if (initialized) {
      } else if (isa<InitListExpr>(init)) {
        if (auto A = ms.InitializeValueByInitListExpr(
                op, const_cast<clang::Expr *>(init))) {
          initialized = true;
          initial_value = A;
        }
      } else {
        auto VC = ms.Visit(const_cast<clang::Expr *>(init));
        if (!VC.isReference) {
          if (auto cop = VC.val.getDefiningOp<arith::ConstantOp>()) {
            initial_value = cop.getValue();
            initial_value = SplatElementsAttr::get(
                RankedTensorType::get(mr.getShape(), mr.getElementType()),
                initial_value);
            initialized = true;
          }
        }
      }

      // An array of structs (or a struct): memref.global cannot hold a
      // value of llvm.struct elements, so the constant goes into
      // `hls.struct_initial_value`, one attribute per scalar in memory order
      // (elements, then members depth-first), which hls-est's
      // --split-struct-memrefs turns into the initial values of the
      // per-member memories.
      mlir::ArrayAttr structInit;
      if (!initialized &&
          (structInit = evaluateStructInitializer(VD, init, mr))) {
        globalOp->setAttr("hls.struct_initial_value", structInit);
      } else if (!initialized) {
        if (!(HLSDynamicGlobalInit &&
              emitDynamicInitializer(DefVD, init, globalOp, isArray))) {
          FD->dump();
          init->dump();
          llvm::errs() << " warning not initializing global: " << name
                       << "\n";
        }
      } else {
        globalOp.setInitialValueAttr(initial_value);
      }
      delete B;
    }

  return globals[name];
}

bool MLIRASTConsumer::emitDynamicInitializer(const VarDecl *VD,
                                            const Expr *init,
                                            mlir::memref::GlobalOp global,
                                            bool isArray) {
  using namespace mlir;
  // An initializer list of non-constant elements and a reference are left
  // to the warning: only one value is computed and stored here.
  if (isArray || isa<InitListExpr>(init) || VD->getType()->isReferenceType())
    return false;
  MemRefType mt = global.getType();
  if (mt.getRank() < 1)
    return false;
  std::string fname = ("__hls_global_init." + global.getSymName()).str();
  if (module->lookupSymbol(fname))
    return true;
  Location loc = getMLIRLocation(VD->getLocation());
  OpBuilder mb(module->getContext());
  mb.setInsertionPointToEnd(module->getBody());
  auto fn = mb.create<func::FuncOp>(loc, fname, mb.getFunctionType({}, {}));
  // The declaration's position orders the initializers as C++ does within a
  // translation unit (check_semantics runs them in that order).
  fn->setAttr("hls.global_init",
              mb.getI64IntegerAttr(static_cast<int64_t>(
                  VD->getLocation().getRawEncoding())));
  Block *entry = fn.addEntryBlock();
  MLIRScanner ms(*this, module, LTInfo);
  ms.setEntryAndAllocBlock(entry);
  OpBuilder &b = ms.getBuilder();
  ValueCategory vc = ms.Visit(const_cast<Expr *>(init));
  // A reference is loaded only when it names one element (a class object
  // of an array type, or an array, is not one value).
  auto refTy = vc.val && vc.isReference ? dyn_cast<MemRefType>(vc.val.getType())
                                        : MemRefType();
  bool oneElement = refTy && refTy.getRank() == 1 &&
                    (refTy.getShape()[0] == 1 || refTy.isDynamicDim(0));
  mlir::Value v = !vc.val                         ? mlir::Value()
                  : !vc.isReference               ? vc.val
                  : oneElement && refTy.getElementType() == mt.getElementType()
                      ? vc.getValue(loc, b)
                      : mlir::Value();
  if (!v || v.getType() != mt.getElementType()) {
    fn.erase();
    return false;
  }
  auto gv = b.create<memref::GetGlobalOp>(loc, mt, global.getName());
  mlir::Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  llvm::SmallVector<mlir::Value> idx(mt.getRank(), zero);
  b.create<memref::StoreOp>(loc, v, gv, idx);
  b.create<func::ReturnOp>(loc);
  return true;
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

mlir::func::FuncOp
MLIRASTConsumer::GetOrCreateMLIRFunction(
    const FunctionDecl *FD, bool getDeviceStub,
    const clang::CXXRecordDecl *xlxStreamObject) {
  assert(FD->getTemplatedKind() !=
         FunctionDecl::TemplatedKind::TK_FunctionTemplate);
  assert(
      FD->getTemplatedKind() !=
      FunctionDecl::TemplatedKind::TK_DependentFunctionTemplateSpecialization);
  std::string name;
  if (getDeviceStub)
    name =
        CGM.getMangledName(GlobalDecl(FD, KernelReferenceKind::Kernel)).str();
  else if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
  else if (auto CC = dyn_cast<CXXDestructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
  else
    name = CGM.getMangledName(FD).str();

  // (--xlx-headers) A member of hls::stream is never emitted: every call is
  // an external call, as with the stand-in headers. A member inherited from
  // hls::stream<T, 0> and called on an hls::stream<T, D> object is named as
  // a member of the object's class, so that the name carries D (the
  // stand-in's hls::stream<T, D> declares every member itself).
  const auto *streamMember = dyn_cast<CXXMethodDecl>(FD);
  bool xlxStream = streamMember && isXlxStreamClass(streamMember->getParent());
  if (xlxStream && xlxStreamObject &&
      xlxStreamObject != streamMember->getParent() &&
      isXlxStreamClass(xlxStreamObject)) {
    // The class's part of the mangled name: the RTTI name of the class is
    // `_ZTI` + its nested name (`N3hls6streamIiLi0EEE`), which the member's
    // name (`_ZN3hls6streamIiLi0EE4readEv`, `_ZNK...`) contains without the
    // closing `E`. Substitution indices are the same for both classes (the
    // depth literal is no substitution candidate).
    auto classPart = [&](const clang::CXXRecordDecl *rd) {
      std::string rtti;
      llvm::raw_string_ostream os(rtti);
      std::unique_ptr<clang::MangleContext> mc(
          CGM.getContext().createMangleContext());
      mc->mangleCXXRTTI(CGM.getContext().getRecordType(rd), os);
      os.flush();
      StringRef r(rtti);
      if (!r.consume_front("_ZTIN") || !r.consume_back("E"))
        return std::string();
      return r.str();
    };
    std::string from = classPart(streamMember->getParent());
    std::string to = classPart(xlxStreamObject);
    StringRef n(name);
    for (StringRef head : {"_ZN", "_ZNK"})
      if (!from.empty() && !to.empty() && n.starts_with((head + from).str())) {
        name = (head + to + n.drop_front(head.size() + from.size())).str();
        break;
      }
  }

  name = (PrefixABI + name);

  assert(name != "free");

  

  llvm::GlobalValue::LinkageTypes LV;
  if (!FD->hasBody() || xlxStream)
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
  if (xlxStream || !FD->isDefined(Def, /*checkforfriend*/ true))
    Def = FD;

  if (functions.find(name) != functions.end()) {
    auto function = functions[name];

    if (!xlxStream && Def->isThisDeclarationADefinition()) {
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
      if (auto mt = dyn_cast<MemRefType>(t)) {
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
    if (llvmType && !CStyleMemRef) {
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

  

  mlir::func::FuncOp function = mlir::func::FuncOp(mlir::func::FuncOp::create(
      getMLIRLocation(FD->getLocation()), name, funcType));
  
  if (HLSAnnotate)
    function->setAttr("hls.orig_name",
                      StringAttr::get(builder.getContext(), FD->getNameAsString()));
  // (--xlx-headers) A member of hls::stream<T, D>: the bits Vitis packs T
  // into, which the IR type cannot tell (`bool` is an i8 there; Vitis packs
  // a struct of a float and a bool into 33 bits, VITIS_NOTES N58, kernels
  // w9r7_sb_*). hls-est's FIFO model reads it.
  if (HLSAnnotate && xlxStream) {
    const auto *spec = dyn_cast<clang::ClassTemplateSpecializationDecl>(
        streamMember->getParent());
    if (spec && spec->getTemplateArgs().size() >= 1 &&
        spec->getTemplateArgs()[0].getKind() == clang::TemplateArgument::Type)
      if (std::optional<int64_t> bits =
              packedStreamBits(spec->getTemplateArgs()[0].getAsType()))
        if (*bits > 0)
          function->setAttr("hls.stream_element_bits",
                            builder.getI64IntegerAttr(*bits));
  }
  
  if (LV == llvm::GlobalValue::InternalLinkage ||
      LV == llvm::GlobalValue::PrivateLinkage || !FD->isDefined() ||
      xlxStream || FD->hasAttr<CUDAGlobalAttr>() ||
      FD->hasAttr<CUDADeviceAttr>()) {
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

  if (!xlxStream && Def->isThisDeclarationADefinition()) {
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

namespace {
/// Finds an expression clang's error recovery left in a function body (a
/// RecoveryExpr, or any expression built on one: a use of an undeclared
/// name, a call that matched no overload).
class ErrorExprFinder : public clang::RecursiveASTVisitor<ErrorExprFinder> {
public:
  bool found = false;
  bool VisitExpr(clang::Expr *e) {
    if (e->containsErrors()) {
      found = true;
      return false;
    }
    return true;
  }
};
} // namespace

/// A function clang's semantic analysis rejected: an invalid declaration (an
/// unknown parameter type) or a body with error-recovery expressions. Its
/// errors are already reported and cgeist exits with a failure; emitting it
/// asserted in the emitter (ValueCategory::getValue). Clang's own CodeGen
/// emits nothing for a translation unit with errors; cgeist still emits the
/// valid functions, whose HLS pragma diagnostics are reported during
/// emission.
static bool hasSemanticErrors(const FunctionDecl *fd) {
  if (fd->isInvalidDecl())
    return true;
  const FunctionDecl *def = nullptr;
  if (!fd->hasBody(def) || !def->getBody())
    return false;
  ErrorExprFinder finder;
  finder.TraverseStmt(def->getBody());
  return finder.found;
}

void MLIRASTConsumer::run() {
  while (functionsToEmit.size()) {
    const FunctionDecl *FD = functionsToEmit.front();
    functionsToEmit.pop_front();
    if (astContext.getDiagnostics().hasErrorOccurred() &&
        hasSemanticErrors(FD))
      continue;
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


/// Whether `--function=` names `fd` by its source name (`top`, or the
/// qualified `ns::top`) rather than its mangled one: the HLS top of a C++
/// design is given by its source name, as Vitis' `set_top` takes it. Every
/// overload of that name is emitted.
static bool isRequestedBySourceName(const FunctionDecl *fd,
                                    const std::set<std::string> &requested) {
  if (!fd->getIdentifier())
    return false;
  return requested.count(fd->getName().str()) ||
         requested.count(fd->getQualifiedNameAsString());
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
    // (--xlx-headers) A member of hls::stream is never emitted, also when an
    // earlier translation unit's external call put its name in emitIfFound
    // and this unit holds its definition (CIRCT OPEN_ISSUES O193: flowgnn/dgn,
    // `operator<<` / `write` of hls::stream<std::array<ap_fixed, 2>> got
    // their synthesis bodies, __fpga_fifo_push, in a three-unit compile).
    if (isXlxStreamMember(fd))
      continue;

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
        emitIfFound.count(name) || isRequestedBySourceName(fd, emitIfFound)) {
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
    // (--xlx-headers) A member of hls::stream is never emitted, also when an
    // earlier translation unit's external call put its name in emitIfFound
    // and this unit holds its definition (CIRCT OPEN_ISSUES O193: flowgnn/dgn,
    // `operator<<` / `write` of hls::stream<std::array<ap_fixed, 2>> got
    // their synthesis bodies, __fpga_fifo_push, in a three-unit compile).
    if (isXlxStreamMember(fd))
      continue;

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
        emitIfFound.count(name) || isRequestedBySourceName(fd, emitIfFound)) {
      functionsToEmit.push_back(fd);
    } else {
    }
  }

  return true;
}

// Wait until Sema has instantiated all the relevant code
// before running codegen on the selected functions.
// Set by parseMLIR: the module is built from more than one translation unit.
static bool gSeveralTranslationUnits = false;
// The mark of a global emitUnusedDefinitions created; eraseUnusedDefinitions
// removes it, and the global when nothing uses it.
static constexpr llvm::StringLiteral kUnusedDefinitionAttr =
    "cgeist.unused_definition";

void MLIRASTConsumer::HandleTranslationUnit(ASTContext &C) {
  cgeist::hls::checkTranslationUnit(C);
  if (cgeist::hls::checkOnly())
    cgeist::hls::checkAllFunctions(C); // no code generation
  else {
    run();
    if (gSeveralTranslationUnits)
      emitUnusedDefinitions(C.getTranslationUnitDecl());
  }
  cgeist::hls::endTranslationUnit();
}

void MLIRASTConsumer::emitUnusedDefinitions(clang::DeclContext *DC) {
  for (clang::Decl *D : DC->decls()) {
    if (isa<clang::NamespaceDecl, clang::LinkageSpecDecl,
            clang::ExternCContextDecl>(D)) {
      emitUnusedDefinitions(cast<clang::DeclContext>(D));
      continue;
    }
    auto *V = dyn_cast<clang::VarDecl>(D);
    // Only a definition another unit can name: external linkage, at file or
    // namespace scope, of a concrete type.
    if (!V || V->isThisDeclarationADefinition() == VarDecl::DeclarationOnly ||
        !V->isFileVarDecl() || !V->hasExternalFormalLinkage() ||
        V->isTemplated() || V->getType()->isDependentType() ||
        V->getType()->isReferenceType() || V->isInvalidDecl())
      continue;
    std::string name = PrefixABI + CGM.getMangledName(V).str();
    auto found = globals.find(name);
    if (found != globals.end()) {
      // An earlier unit's declaration: this definition completes it.
      if (!found->second.first.getInitialValue())
        GetOrCreateGlobal(V, /*prefix=*/"");
      continue;
    }
    // A new global: only memories of integers and floats with a constant (or
    // no) initializer, the tables and weights a design keeps in another
    // file; objects with constructors (streams, std::complex) stay out.
    const clang::Type *base = V->getType()->getBaseElementTypeUnsafe();
    if (const auto *RT = base->getAs<clang::RecordType>())
      if (isXlxStreamClass(RT->getDecl()))
        continue;
    if (const Expr *init = V->getInit())
      if (!init->isConstantInitializer(CGM.getContext(), false))
        continue;
    auto mr = getMLIRType(CGM.getContext().getLValueReferenceType(V->getType()))
                  .dyn_cast<MemRefType>();
    if (!mr || !mr.getElementType().isIntOrFloat())
      continue;
    auto created = GetOrCreateGlobal(V, /*prefix=*/"");
    if (created.first)
      created.first->setAttr(kUnusedDefinitionAttr,
                             mlir::UnitAttr::get(module->getContext()));
  }
}

// After the last translation unit: the definitions emitUnusedDefinitions
// created that no unit used go away again, so that only the globals a
// function names are in the module, as with one unit.
static void eraseUnusedDefinitions(mlir::ModuleOp module) {
  for (auto g : llvm::make_early_inc_range(
           module.getBody()->getOps<mlir::memref::GlobalOp>())) {
    if (!g->hasAttr(kUnusedDefinitionAttr))
      continue;
    g->removeAttr(kUnusedDefinitionAttr);
    if (mlir::SymbolTable::symbolKnownUseEmpty(g, module))
      g.erase();
  }
}

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
    // Through type sugar: the deduced `auto &&__range` of a range-based for
    // over an `int (&)[8]` parameter is an array type behind an AutoType
    // (before, cast<ConstantArrayType> asserted; OPEN_ISSUES O91,
    // reproducer c_rangefor.cpp).
    auto cstArrTy = cast<clang::ConstantArrayType>(curTy->getAsArrayTypeUnsafe());
    shape.push_back(cstArrTy->getSize().getSExtValue());
    curTy = cstArrTy->getElementType();
  }

  elemTy = curTy;
}

// TODO memoize the results?
static bool
isRecursiveStructImpl(const clang::Type *t,
                      SmallPtrSetImpl<const clang::RecordType *> &seen) {
  if (auto PT = dyn_cast<clang::PointerType>(t)) {
    return isRecursiveStructImpl(
        PT->getPointeeType()->getUnqualifiedDesugaredType(), seen);
  } else if (auto RT = dyn_cast<clang::ReferenceType>(t)) {
    return isRecursiveStructImpl(
        RT->getPointeeType()->getUnqualifiedDesugaredType(), seen);
  } else if (auto RT = dyn_cast<clang::RecordType>(t)) {
    if (seen.count(RT))
      return true;
    seen.insert(RT);

    auto CXRD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (CXRD) {
      for (auto f : CXRD->bases()) {
        auto baseTy = f.getType()->getUnqualifiedDesugaredType();
        if (isRecursiveStructImpl(baseTy, seen))
          return true;
      }
    }

    for (auto f : RT->getDecl()->fields()) {
      auto fieldTy = f->getType()->getUnqualifiedDesugaredType();
      if (isRecursiveStructImpl(fieldTy, seen))
        return true;
    }

    return false;
  } else {
    return false;
  }
}

static bool isRecursiveStruct(const clang::RecordType *RT) {
  SmallPtrSet<const clang::RecordType *, 4> seen;
  return isRecursiveStructImpl(RT, seen);
}

mlir::Type MLIRASTConsumer::getMLIRType(clang::QualType qt, bool *implicitRef,
                                        bool allowMerge) {
  if (auto ET = dyn_cast<clang::ElaboratedType>(qt)) {
    return getMLIRType(ET->getNamedType(), implicitRef, allowMerge);
  }
  if (auto ET = dyn_cast<clang::UsingType>(qt)) {
    return getMLIRType(ET->getUnderlyingType(), implicitRef, allowMerge);
  }
  if (auto ET = dyn_cast<clang::ParenType>(qt)) {
    return getMLIRType(ET->getInnerType(), implicitRef, allowMerge);
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
    if (memRefABI && assumeRef) {
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

          // A struct whose members share one type is itself a memref
          // (CombinedStructABI: `struct { unsigned long a, b; }` is
          // memref<2xi64>); the array of them adds its dimensions in front,
          // as the non-full-rank path below does (memref<?x2xi64>), instead
          // of a memref of memrefs (W8b repro r_bfs.c, machsuite bfs_bulk /
          // bfs_queue: `node_t nodes[N]` became memref<Nxmemref<2xi64>>).
          mlir::Type elemMLIR = getMLIRType(elemTy);
          if (auto inner = dyn_cast<mlir::MemRefType>(elemMLIR)) {
            if (inner.hasStaticShape() && inner.getLayout().isIdentity()) {
              shape.append(inner.getShape().begin(), inner.getShape().end());
              return mlir::MemRefType::get(shape, inner.getElementType());
            }
          }
          return mlir::MemRefType::get(shape, elemMLIR);
        }
      }

      // If -memref-fullrank is unset or it cannot be fulfilled.
      auto mt = dyn_cast<MemRefType>(mlirty);
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2[0] = ShapedType::kDynamic;
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
    if (CombinedStructABI && memRefABI && allowMerge) {
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
    // (--xlx-headers) An hls::stream is an i32 handle, the stand-in
    // headers' `int` member.
    if (isXlxStreamClass(RT->getDecl()))
      return mlir::IntegerType::get(module->getContext(), 32);
    // A scalar wrapper is its member (ap_fixed<16,5> -> i16). implicitRef
    // stays false: the value is a scalar, not an aggregate.
    if (const clang::FieldDecl *WF = getScalarWrapperField(RT->getDecl()))
      return getMLIRType(WF->getType(), /*implicitRef*/ nullptr, allowMerge);
    if (typeCache.find(RT) != typeCache.end())
      return typeCache[RT];
    llvm::Type *LT = CGM.getTypes().ConvertType(qt);
    if (!isa<llvm::StructType>(LT)) {
      qt->dump();
      llvm::errs() << "LT: " << *LT << "\n";
    }
    llvm::StructType *ST = cast<llvm::StructType>(LT);

    SmallPtrSet<llvm::Type *, 4> Seen;
    bool notAllSame = false;
    for (size_t i = 0; i < ST->getNumElements(); i++) {
      if (ST->getTypeAtIndex(i) != ST->getTypeAtIndex(0U)) {
        notAllSame = true;
      }
    }

    auto CXRD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (isLLVMStructABI(RT->getDecl(), ST)) {
      auto retTy = typeTranslator.translateType(anonymize(ST));
      return retTy;
    }

    /* TODO
    if (ST->getNumElements() == 1 && !recursive &&
        !RT->getDecl()->fields().empty() && ++RT->getDecl()->field_begin() ==
    RT->getDecl()->field_end()) { auto subT =
    getMLIRType((*RT->getDecl()->field_begin())->getType(), implicitRef,
    allowMerge); return subT;
    }
    */
    bool recursive = isRecursiveStruct(RT);
    if (recursive)
      typeCache[RT] = LLVM::LLVMStructType::getIdentified(
          module->getContext(), ("polygeist@mlir@" + ST->getName()).str());

    SmallVector<mlir::Type, 4> types;

    bool innerLLVM = false;
    if (CXRD) {
      for (auto f : CXRD->bases()) {
        bool subRef = false;
        auto ty = getMLIRType(f.getType(), &subRef, /*allowMerge*/ false);
        assert(!subRef);
        innerLLVM |= ty.isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                            LLVM::LLVMArrayType>();
        types.push_back(ty);
      }
    }

    for (auto f : RT->getDecl()->fields()) {
      bool subRef = false;
      auto ty = getMLIRType(f->getType(), &subRef, /*allowMerge*/ false);
      assert(!subRef);
      innerLLVM |= ty.isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                          LLVM::LLVMArrayType>();
      types.push_back(ty);
    }

    if (types.empty())
      if (ST->getNumElements() == 1 && ST->getElementType(0U)->isIntegerTy(8))
        return typeTranslator.translateType(anonymize(ST));

    if (recursive) {
      auto LR = typeCache[RT].setBody(types, /*isPacked*/ false);
      assert(LR.succeeded());
      return typeCache[RT];
    }

    // With opaque LLVM pointers every pointer or reference field has the same
    // LLVM type, so `notAllSame` cannot tell `A &a; B &b;` apart (the Vitis
    // headers' ap_concat_ref): compare the MLIR field types as well. Only
    // with --xlx-headers, which leaves other inputs unchanged.
    if (cgeist::hls::xlxHeaders())
      for (mlir::Type t : types)
        notAllSame |= t != types[0];
    if (!memRefABI || notAllSame || !allowMerge || innerLLVM) {
      auto retTy =
          mlir::LLVM::LLVMStructType::getLiteral(module->getContext(), types);
      return retTy;
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
    if (!CStyleMemRef && PTT->isCharType()) {
      llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
      return typeTranslator.translateType(T);
    }
    bool subRef = false;
    auto ET = getMLIRType(AT->getElementType(), &subRef, allowMerge);
    int64_t size = ShapedType::kDynamic;
    if (auto CAT = dyn_cast<clang::ConstantArrayType>(AT))
      size = CAT->getSize().getZExtValue();
    if (memRefABI && subRef) {
      auto mt = ET.cast<MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2.insert(shape2.begin(), size);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    }
    if (!memRefABI || !allowMerge ||
        (!CStyleMemRef &&
         ET.isa<LLVM::LLVMPointerType, LLVM::LLVMArrayType,
                LLVM::LLVMFunctionType, LLVM::LLVMStructType>()))
      return LLVM::LLVMArrayType::get(
          ET, (size == ShapedType::kDynamic) ? 0 : size);
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get({size}, ET);
  }

  if (auto AT = dyn_cast<clang::VectorType>(t)) {
    bool subRef = false;
    auto ET = getMLIRType(AT->getElementType(), &subRef, allowMerge);
    int64_t size = AT->getNumElements();
    if (CombinedStructABI && subRef) {
      auto mt = ET.cast<MemRefType>();
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2.insert(shape2.begin(), size);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    }
    if (!memRefABI || !allowMerge || !CombinedStructABI ||
        (!CStyleMemRef &&
         ET.isa<LLVM::LLVMPointerType, LLVM::LLVMArrayType,
                LLVM::LLVMFunctionType, LLVM::LLVMStructType>())) {
      if (mlir::LLVM::LLVMFixedVectorType::isValidElementType(ET)) {
        return mlir::LLVM::LLVMFixedVectorType::get(ET, size);
      }
      return mlir::LLVM::LLVMArrayType::get(ET, size);
    }
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
  if (auto FT = dyn_cast<clang::FunctionNoProtoType>(t)) {
    auto RT = getMLIRType(FT->getReturnType());
    if (RT.isa<mlir::NoneType>())
      RT = LLVM::LLVMVoidType::get(RT.getContext());
    SmallVector<mlir::Type> Args;
    return LLVM::LLVMFunctionType::get(RT, Args, /*isVariadic*/ true);
  }

  if (isa<clang::PointerType, clang::ReferenceType>(t)) {
    int64_t outer = ShapedType::kDynamic;
    auto PTT = isa<clang::PointerType>(t) ? cast<clang::PointerType>(t)
                                                ->getPointeeType()
                                                ->getUnqualifiedDesugaredType()
                                          : cast<clang::ReferenceType>(t)
                                                ->getPointeeType()
                                                ->getUnqualifiedDesugaredType();

    if (!CStyleMemRef && PTT->isCharType()) {
      llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
      return typeTranslator.translateType(T);
    }
    if (PTT->isVoidType()) {
      llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
      auto MT = typeTranslator.translateType(T);
      if (!CStyleMemRef)
        return MT;
      else
        return getVoidMemRefTy();
    }
    bool subRef = false;
    auto subType =
        getMLIRType(isa<clang::PointerType>(t)
                        ? cast<clang::PointerType>(t)->getPointeeType()
                        : cast<clang::ReferenceType>(t)->getPointeeType(),
                    &subRef, /*allowMerge*/ true);

    if (!memRefABI)
      return LLVM::LLVMPointerType::get(subType);

    if (!CStyleMemRef &&
        subType.isa<LLVM::LLVMArrayType, LLVM::LLVMStructType,
                    LLVM::LLVMPointerType, LLVM::LLVMFunctionType>())
      return LLVM::LLVMPointerType::get(subType);

    if (isa<clang::ArrayType>(PTT)) {
      if (subType.isa<MemRefType>()) {
        assert(subRef);
        return subType;
      } else {
        if (!CStyleMemRef)
          return LLVM::LLVMPointerType::get(subType);
      }
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
      } else {
        if (!CStyleMemRef)
          return LLVM::LLVMPointerType::get(subType);
      }
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

  if (auto BIT = dyn_cast<clang::BitIntType>(t)) {
    // _BitInt(N) / unsigned _BitInt(N): a signless iN like the other ints.
    mlir::OpBuilder builder(module->getContext());
    return builder.getIntegerType(BIT->getNumBits());
  }

  // std::nullptr_t (a parameter or temporary of type decltype(nullptr)): a
  // null pointer, as clang's CodeGen gives it (an i8* in LLVM IR).
  if (t->isNullPtrType())
    return LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(module->getContext(), 8));
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
    if (T->isHalfTy()) // _Float16 (and __fp16 storage)
      return builder.getF16Type();
    if (T->isBFloatTy()) // __bf16
      return builder.getBF16Type();
    if (T->isX86_FP80Ty())
      return builder.getF80Type();
    if (T->isFP128Ty())
      return builder.getF128Type();

    if (auto IT = dyn_cast<llvm::IntegerType>(T)) {
      return builder.getIntegerType(IT->getBitWidth());
    }
  }
  qt->dump();
  llvm_unreachable("unhandled type");
}

llvm::Type *MLIRASTConsumer::getLLVMType(clang::QualType t) {
  if (t->isVoidType()) {
    return llvm::Type::getVoidTy(llvmMod.getContext());
  }
  llvm::Type *T = CGM.getTypes().ConvertType(t);
  return T;
}

#include "llvm/TargetParser/Host.h"

#include "HLSAutoInline.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/MultiplexConsumer.h"
class MLIRAction : public clang::ASTFrontendAction {
public:
  std::set<std::string> emitIfFound;
  std::set<std::string> done;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> globals;
  std::map<std::string, mlir::func::FuncOp> functions;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> llvmFunctions;
  MLIRAction(std::string fn, mlir::OwningOpRef<mlir::ModuleOp> &module)
      : module(module) {
    emitIfFound.insert(fn);
  }
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    std::unique_ptr<clang::ASTConsumer> consumer(new MLIRASTConsumer(
        emitIfFound, done, llvmStringGlobals, globals, functions, llvmGlobals,
        llvmFunctions, CI.getPreprocessor(), CI.getASTContext(), module,
        CI.getSourceManager(), CI.getCodeGenOpts()));
    if (!cgeist::hls::autoInlineEnabled())
      return consumer;
    // Vitis's automatic inlining needs clang's own LLVM IR for the sizes
    // (HLSAutoInline.h); cgeist's consumer first, so that its pragma sema
    // has recorded the unroll pragmas when the sizes are measured.
    std::vector<std::unique_ptr<clang::ASTConsumer>> consumers;
    consumers.push_back(std::move(consumer));
    consumers.push_back(cgeist::hls::createInlineSizeConsumer(CI, InFile));
    return std::make_unique<clang::MultiplexConsumer>(std::move(consumers));
  }
};

mlir::func::FuncOp MLIRScanner::EmitDirectCallee(const FunctionDecl *FD) {
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

mlir::Value MLIRScanner::getTypeSize(mlir::Location loc, clang::QualType t) {
  // llvm::Type *T = Glob.CGM.getTypes().ConvertType(t);
  // return (Glob.llvmMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
  bool isArray = false;
  auto innerTy = Glob.getMLIRType(t, &isArray);
  if (isArray) {
    auto MT = innerTy.cast<MemRefType>();
    size_t num = 1;
    for (auto n : MT.getShape()) {
      assert(n > 0);
      num *= n;
    }
    return builder.create<arith::MulIOp>(
        loc,
        builder.create<polygeist::TypeSizeOp>(
            loc, builder.getIndexType(),
            mlir::TypeAttr::get(MT.getElementType())),
        builder.create<arith::ConstantIndexOp>(loc, num));
  }
  assert(!isArray);
  return builder.create<polygeist::TypeSizeOp>(
      loc, builder.getIndexType(),
      mlir::TypeAttr::get(innerTy)); // DLI.getTypeSize(innerTy);
}

mlir::Value MLIRScanner::getTypeAlign(mlir::Location loc, clang::QualType t) {
  // llvm::Type *T = Glob.CGM.getTypes().ConvertType(t);
  // return (Glob.llvmMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
  bool isArray = false;
  auto innerTy = Glob.getMLIRType(t, &isArray, /*allowMerge=*/false);
  assert(!isArray);
  return builder.create<polygeist::TypeAlignOp>(
      loc, builder.getIndexType(),
      mlir::TypeAttr::get(innerTy)); // DLI.getTypeSize(innerTy);
}

#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "XlxPrelude.inc"

static bool parseMLIR(const char *Argv0, std::vector<std::string> filenames,
                      std::string fn, std::vector<std::string> includeDirs,
                      std::vector<std::string> defines,
                      mlir::OwningOpRef<mlir::ModuleOp> &module,
                      llvm::Triple &triple, llvm::DataLayout &DL,
                      llvm::Triple &gpuTriple, llvm::DataLayout &gpuDL) {

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
  mlirclang::ArgumentList Argv;
  Argv.push_back(binary);
  // TODO we should probably preserve the order of these args - they matter in
  // some cases as is with this one - it has to be before the input file
  if (Lang != "") {
    Argv.push_back("-x");
    Argv.emplace_back(Lang);
  }
  for (const auto &filename : filenames) {
    Argv.emplace_back(filename);
  }
  if (FOpenMP)
    Argv.push_back("-fopenmp");
  if (TargetTripleOpt != "") {
    Argv.push_back("-target");
    Argv.emplace_back(TargetTripleOpt);
  }
  if (McpuOpt != "") {
    Argv.emplace_back("-mcpu=", McpuOpt);
  }
  if (Standard != "") {
    Argv.emplace_back("-std=", Standard);
  }
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    Argv.emplace_back(ResourceDir);
  }
  if (SysRoot != "") {
    Argv.push_back("--sysroot");
    Argv.emplace_back(SysRoot);
  }
  if (Verbose) {
    Argv.push_back("-v");
  }
  if (NoCUDAInc) {
    Argv.push_back("-nocudainc");
  }
  if (NoCUDALib) {
    Argv.push_back("-nocudalib");
  }
  if (CUDAGPUArch != "") {
    Argv.emplace_back("--cuda-gpu-arch=", CUDAGPUArch);
  }
  if (CUDAPath != "") {
    Argv.emplace_back("--cuda-path=", CUDAPath);
  }
  if (MArch != "") {
    Argv.emplace_back("-march=", MArch);
  }
  for (const auto &dir : includeDirs) {
    Argv.push_back("-I");
    Argv.emplace_back(dir);
  }
  for (const auto &define : defines) {
    Argv.emplace_back("-D", define);
  }
  for (const auto &Include : Includes) {
    Argv.push_back("-include");
    Argv.emplace_back(Include);
  }
  for (const auto &Arg : XclangArgs) {
    Argv.push_back("-Xclang");
    Argv.emplace_back(Arg);
  }

  Argv.push_back("-emit-ast");

  const unique_ptr<Compilation> compilation(
      driver->BuildCompilation(Argv.getArguments()));
  JobList &Jobs = compilation->getJobs();
  if (Jobs.size() < 1)
    return false;

  MLIRAction Act(fn, module);
  gSeveralTranslationUnits = Jobs.size() > 1;

  for (auto &job : Jobs) {
    std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

    Command *cmd = cast<Command>(&job);
    if (strcmp(cmd->getCreator().getName(), "clang"))
      return false;

    const ArgStringList *args = &cmd->getArguments();

    Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(), *args,
                                                 Diags);
    Clang->getInvocation().getFrontendOpts().DisableFree = false;
    if (XlxPrelude) {
      // A virtual file (absolute, so that the include lookup finds it),
      // included after every -include file.
      static const char *const kXlxPreludeName =
          "/cgeist-builtin-include/xlx/xlx_prelude.h";
      PreprocessorOptions &PPOpts = Clang->getPreprocessorOpts();
      PPOpts.Includes.push_back(kXlxPreludeName);
      PPOpts.addRemappedFile(
          kXlxPreludeName,
          llvm::MemoryBuffer::getMemBuffer(kXlxPreludeText, kXlxPreludeName)
              .release());
    }

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
    if ((Clang->getLangOpts().CUDA ||
         Clang->getLangOpts().OpenMPIsTargetDevice) &&
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

    llvm::Triple jobTriple = Clang->getTarget().getTriple();
    if (triple.str() == "" || !jobTriple.isNVPTX()) {
      triple = jobTriple;
      module.get()->setAttr(
          LLVM::LLVMDialect::getTargetTripleAttrName(),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getTriple().getTriple()));
      DL = llvm::DataLayout(Clang->getTarget().getDataLayoutString());
      module.get()->setAttr(
          LLVM::LLVMDialect::getDataLayoutAttrName(),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getDataLayoutString()));

      module.get()->setAttr(("dlti." + DataLayoutSpecAttr::kAttrKeyword).str(),
                            translateDataLayout(DL, module->getContext()));

      // Add target-cpu and target-features attributes to functions. If
      // we have a decl for the function and it has a target attribute then
      // parse that and add it to the feature set.
      StringRef TargetCPU = Clang->getTarget().getTargetOpts().CPU;
      StringRef TuneCPU = Clang->getTarget().getTargetOpts().TuneCPU;
      std::vector<std::string> Features =
          Clang->getTarget().getTargetOpts().Features;

      if (!TargetCPU.empty()) {
        module.get()->setAttr("polygeist.target-cpu",
                              StringAttr::get(module->getContext(), TargetCPU));
      }
      if (!TuneCPU.empty()) {
        module.get()->setAttr("polygeist.tune-cpu",
                              StringAttr::get(module->getContext(), TuneCPU));
      }
      if (!Features.empty()) {
        llvm::sort(Features);
        module.get()->setAttr(
            "polygeist.target-features",
            StringAttr::get(module->getContext(), llvm::join(Features, ",")));
      }
    }

    // TODO investigate what AMDGCN and AMDGPU are, do we need them both?
    if (jobTriple.isNVPTX() || jobTriple.isAMDGCN() || jobTriple.isAMDGPU()) {
      gpuTriple = jobTriple;
      module.get()->setAttr(
          StringRef("polygeist.gpu_module." +
                    LLVM::LLVMDialect::getTargetTripleAttrName().str()),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getTriple().getTriple()));
      gpuDL = llvm::DataLayout(Clang->getTarget().getDataLayoutString());
      module.get()->setAttr(
          StringRef("polygeist.gpu_module." +
                    LLVM::LLVMDialect::getDataLayoutAttrName().str()),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getDataLayoutString()));
    }

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

    if (Clang->getDiagnostics().hasErrorOccurred()) {
      return false;
    }
  }
  eraseUnusedDefinitions(module.get());
  return true;
}

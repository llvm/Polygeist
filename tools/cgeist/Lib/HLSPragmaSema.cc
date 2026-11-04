//===- HLSPragmaSema.cc - Association and validation of #pragma HLS -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binds each pragma inside a function body to its target by source location,
// resolves variable names and integer constants in the scope of the pragma,
// validates Vitis semantics, detects conflicts, and records the attributes
// to emit. See docs/hls-pragmas.md for the rules implemented here.
//
//===----------------------------------------------------------------------===//

#include <functional>
#include "HLSAutoInline.h"
#include "HLSPragmaInternal.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"

#include <map>

#include <cmath>

using namespace clang;
using namespace cgeist::hls;
using llvm::StringRef;

static llvm::cl::opt<bool> HLSRejectPartitionConflicts(
    "hls-reject-partition-conflicts", llvm::cl::init(false),
    llvm::cl::desc("Reject a second '#pragma HLS array_partition' on a "
                   "dimension an earlier one partitions, instead of merging "
                   "them the Vitis way (the larger factor; the behaviour "
                   "before W10; for comparisons)"));

static llvm::cl::opt<bool> HLSTripOnePipelineUnroll(
    "hls-trip-one-pipeline-unroll", llvm::cl::init(false),
    llvm::cl::desc("A pipeline pragma on a loop whose trip count is 1 "
                   "completely unrolls the loops inside it, and the loop is "
                   "removed with its pragmas, as Vitis does (VITIS_NOTES "
                   "N40); off: the pragma is lost with the folded loop. Off "
                   "by default until hls-est schedules the stream loops of "
                   "the only corpus designs with this shape (OPEN_ISSUES "
                   "O57)"));

/// The bind_storage dictionary a legacy RESOURCE pragma's memory core
/// names: `RAM_2P_BRAM` -> {type = "ram_2p", impl = "bram"}, `RAM_1P` ->
/// {type = "ram_1p"}, `FIFO_SRL` -> {type = "fifo", impl = "srl"}; the
/// `variable` and `latency` entries are kept. nullopt for an operator core
/// (`Mul_LUT`, `AddSub_DSP`), which stays an `hls.resource` attribute.
static std::optional<mlir::DictionaryAttr>
lowerResourceCore(mlir::DictionaryAttr entry, mlir::Builder &b) {
  auto core = entry.getAs<mlir::StringAttr>("core");
  if (!core)
    return std::nullopt;
  llvm::SmallVector<llvm::StringRef, 4> parts;
  llvm::StringRef(core.getValue()).split(parts, '_');
  std::string kind = parts.front().lower();
  std::string type, impl;
  size_t next = 1;
  if ((kind == "ram" || kind == "rom") && parts.size() >= 2) {
    type = kind + "_" + parts[1].lower();
    next = 2;
  } else if (kind == "fifo") {
    type = "fifo";
  } else {
    return std::nullopt;
  }
  if (next < parts.size()) {
    impl = llvm::join(llvm::ArrayRef(parts).drop_front(next), "_");
    impl = llvm::StringRef(impl).lower();
  }
  llvm::SmallVector<mlir::NamedAttribute, 4> fields;
  if (!impl.empty())
    fields.push_back(b.getNamedAttr("impl", b.getStringAttr(impl)));
  if (auto latency = entry.get("latency"))
    fields.push_back(b.getNamedAttr("latency", latency));
  fields.push_back(b.getNamedAttr("type", b.getStringAttr(type)));
  if (auto variable = entry.get("variable"))
    fields.push_back(b.getNamedAttr("variable", variable));
  return b.getDictionaryAttr(fields);
}

namespace {

//===----------------------------------------------------------------------===//
// Source ranges
//===----------------------------------------------------------------------===//

/// A half-open region (begin, end] of source positions.
struct Region {
  SourceLocation begin; ///< exclusive
  SourceLocation end;   ///< inclusive
};

//===----------------------------------------------------------------------===//
// Targets
//===----------------------------------------------------------------------===//

enum class TargetKind { Function, Param, Loop, Local, Global, Operation };

struct TargetKey {
  TargetKind kind;
  const void *ptr;
  unsigned index;
  bool operator<(const TargetKey &o) const {
    return std::tie(kind, ptr, index) < std::tie(o.kind, o.ptr, o.index);
  }
};

/// An entry already placed on a target, for conflict detection.
struct Placed {
  const PragmaRecord *pragma;
  std::string key;
  std::string attrName;
  mlir::Attribute value;
  std::string variable;
  int64_t dim = -1; ///< array_partition/array_reshape only
};

struct TargetBucket {
  TargetKey key;
  std::string what;
  llvm::MapVector<std::string, llvm::SmallVector<mlir::Attribute, 2>,
                  std::map<std::string, unsigned>>
      arrays;
  llvm::SmallVector<std::string, 1> units;
  std::vector<Placed> placed;
  // TargetKind::Operation (bind_op)
  llvm::SmallVector<std::string, 2> opNames;
  unsigned floatWidth = 0;
  const VarDecl *var = nullptr;
  const PragmaRecord *firstPragma = nullptr;
  llvm::SmallVector<int, 2> reports;
};

/// Result of resolving a `variable=` / `port=` name.
struct Resolved {
  enum Kind { None, Local, Param, Global, Return } kind = None;
  const VarDecl *vd = nullptr;
  unsigned paramIndex = 0;
  QualType type; ///< type of the full member path
  std::string path;
  /// For a path of members only (`r.data`, `s.in.buf`): the position of
  /// each member in cgeist's struct type of its record (bases first, then
  /// fields), outermost first; empty for a plain variable or a path with a
  /// subscript.
  llvm::SmallVector<int64_t, 2> memberPositions;
  bool hasSubscript = false;
  /// The name is a data member of the object an instance member function
  /// runs on (`m_Val` in a constructor = `this->m_Val`): the path starts at
  /// the function's `this` argument.
  bool viaThis = false;
};

/// The position path of field `F` in record `RD` as cgeist lays the record
/// out (bases first, then fields), searching the bases depth first; empty
/// when `F` is not a member of `RD`.
static llvm::SmallVector<int64_t, 2> fieldPositionPath(const RecordDecl *RD,
                                                       const FieldDecl *F,
                                                       unsigned depth = 0) {
  if (!RD || depth > 8)
    return {};
  RD = RD->getDefinition();
  if (!RD)
    return {};
  unsigned bases = 0;
  if (const auto *CXRD = dyn_cast<CXXRecordDecl>(RD))
    bases = CXRD->getNumBases();
  if (F->getParent()->getCanonicalDecl() == RD->getCanonicalDecl())
    return {static_cast<int64_t>(bases) + F->getFieldIndex()};
  if (const auto *CXRD = dyn_cast<CXXRecordDecl>(RD)) {
    int64_t k = 0;
    for (const CXXBaseSpecifier &B : CXRD->bases()) {
      if (const RecordDecl *BRD = B.getType()->getAsRecordDecl()) {
        llvm::SmallVector<int64_t, 2> rest =
            fieldPositionPath(BRD, F, depth + 1);
        if (!rest.empty()) {
          rest.insert(rest.begin(), k);
          return rest;
        }
      }
      ++k;
    }
  }
  return {};
}

/// Dimensions of an array-like type: extents (-1 when unknown).
llvm::SmallVector<int64_t, 4> arrayShape(QualType t) {
  llvm::SmallVector<int64_t, 4> shape;
  while (true) {
    t = t.getNonReferenceType().getCanonicalType();
    if (const auto *CAT = dyn_cast<ConstantArrayType>(t.getTypePtr())) {
      shape.push_back(CAT->getSize().getSExtValue());
      t = CAT->getElementType();
    } else if (const auto *AT = dyn_cast<ArrayType>(t.getTypePtr())) {
      shape.push_back(-1);
      t = AT->getElementType();
    } else if (const auto *PT = dyn_cast<PointerType>(t.getTypePtr())) {
      shape.push_back(-1);
      t = PT->getPointeeType();
    } else {
      return shape;
    }
  }
}

/// The array a class holds as its only data member, through its only base
/// and nested such classes (hls::vector<T, N> holds a std::array<T, N>, which
/// holds a T[N]): its dimensions; empty when the class is not such a wrapper.
llvm::SmallVector<int64_t, 4> wrappedArrayShape(QualType t) {
  for (int depth = 0; depth < 8; ++depth) {
    t = t.getNonReferenceType().getCanonicalType();
    const auto *rd = t->getAsCXXRecordDecl();
    if (!rd || !rd->hasDefinition())
      return {};
    rd = rd->getDefinition();
    const FieldDecl *only = nullptr;
    unsigned fields = 0;
    for (const FieldDecl *f : rd->fields()) {
      only = f;
      ++fields;
    }
    if (fields == 0 && rd->getNumBases() == 1) {
      t = rd->bases_begin()->getType();
      continue;
    }
    if (fields != 1 || rd->getNumBases() != 0)
      return {};
    QualType ft = only->getType();
    if (isa<ConstantArrayType>(ft.getCanonicalType().getTypePtr()))
      return arrayShape(ft);
    t = ft;
  }
  return {};
}

/// The dimensions an array_partition / array_reshape pragma on a variable of
/// type `t` addresses, as the variable's memref has them: the array's own;
/// for a class wrapping one array (an hls::vector, a std::array) the
/// variable's one-element dimension, then the wrapped array's (Vitis
/// partitions the vector's lanes: class_ece8893 nikhil_p lab3; CIRCT
/// OPEN_ISSUES O76).
llvm::SmallVector<int64_t, 4> partitionShape(QualType t) {
  llvm::SmallVector<int64_t, 4> shape = arrayShape(t);
  if (!shape.empty())
    return shape;
  llvm::SmallVector<int64_t, 4> wrapped = wrappedArrayShape(t);
  if (wrapped.empty())
    return {};
  shape.push_back(1);
  shape.append(wrapped.begin(), wrapped.end());
  return shape;
}

/// Identifier of a declaration, or "" for unnamed / non-identifier names
/// (operators, structured bindings), where getName() would assert.
StringRef identName(const NamedDecl *ND) {
  if (const IdentifierInfo *II = ND->getIdentifier())
    return II->getName();
  return "";
}

/// Maps a bind_op `op` value to MLIR op names and a float width.
void bindOpTarget(StringRef op, llvm::SmallVectorImpl<std::string> &names,
                  unsigned &width) {
  width = 0;
  StringRef base = op;
  if (op.size() > 1 && (op[0] == 'f' || op[0] == 'd' || op[0] == 'h')) {
    width = op[0] == 'f' ? 32 : op[0] == 'd' ? 64 : 16;
    base = op.drop_front();
  }
  if (width == 0) {
    if (base == "add")
      names.push_back("arith.addi");
    else if (base == "sub")
      names.push_back("arith.subi");
    else if (base == "mul")
      names.push_back("arith.muli");
    return;
  }
  if (base == "add")
    names.push_back("arith.addf");
  else if (base == "sub")
    names.push_back("arith.subf");
  else if (base == "mul")
    names.push_back("arith.mulf");
  else if (base == "div" || base == "recip")
    names.push_back("arith.divf");
  else if (base == "sqrt")
    names.push_back("math.sqrt");
  else if (base == "rsqrt")
    names.push_back("math.rsqrt");
  else if (base == "exp")
    names.push_back("math.exp");
  else if (base == "log")
    names.push_back("math.log");
}

//===----------------------------------------------------------------------===//
// FunctionSema
//===----------------------------------------------------------------------===//

class FunctionSema {
public:
  FunctionSema(const FunctionDecl *FD, mlir::func::FuncOp func,
               unsigned argOffset)
      : FD(FD), func(func), argOffset(argOffset), AC(FD->getASTContext()),
        SM(AC.getSourceManager()), S(State::get()),
        b(func->getContext()) {}

  void run();

private:
  const FunctionDecl *FD;
  mlir::func::FuncOp func;
  unsigned argOffset;
  ASTContext &AC;
  SourceManager &SM;
  State &S;
  mlir::Builder b;

  struct LoopInfo {
    const Stmt *stmt;
    Region region;
    unsigned depth;
  };
  struct DeclInfo {
    const VarDecl *vd;
    SourceRange scope;
  };
  struct BlockInfo {
    SourceRange range;
    std::vector<std::pair<SourceLocation, bool>> children; // begin, isLoop
  };
  std::vector<LoopInfo> loops;
  std::vector<DeclInfo> decls;
  std::vector<BlockInfo> blocks;
  std::vector<SourceRange> excluded;
  std::vector<const UsingDirectiveDecl *> usingDirectives;
  std::map<const Stmt *, std::string> loopLabels;
  llvm::DenseSet<int> placedReports;
  std::map<TargetKey, TargetBucket> buckets;
  std::vector<TargetKey> bucketOrder;

  // Source helpers.
  bool before(SourceLocation a, SourceLocation b) const {
    return SM.isBeforeInTranslationUnit(a, b);
  }
  bool inRegion(const Region &r, SourceLocation p) const {
    return before(r.begin, p) && !before(r.end, p);
  }
  bool inRange(SourceRange r, SourceLocation p) const {
    return !before(p, r.getBegin()) && !before(r.getEnd(), p);
  }
  bool rangeInside(SourceRange inner, SourceRange outer) const {
    return !before(inner.getBegin(), outer.getBegin()) &&
           !before(outer.getEnd(), inner.getEnd());
  }

  void index(const Stmt *s, SourceRange scope, unsigned depth);
  bool belongsHere(SourceLocation p) const;
  const LoopInfo *innermostLoop(SourceLocation p) const;
  bool loopFollows(SourceLocation p) const;
  const BlockInfo *innermostBlock(SourceLocation p) const;
  bool inFunctionOrForBody(SourceLocation p, const LoopInfo *loop) const;

  Resolved resolveVariable(const PragmaRecord &P, const ParsedOption &O,
                           bool allowReturn);
  const VarDecl *lookupLocal(StringRef name, SourceLocation p) const;
  const NamedDecl *lookupNonLocal(StringRef name) const;
  std::optional<int64_t> lookupConstant(StringRef name, SourceLocation p);
  std::optional<int64_t> callConstant(StringRef name,
                                      llvm::ArrayRef<int64_t> args);
  std::optional<int64_t> evalInt(const PragmaRecord &P, const ParsedOption &O);

  void process(PragmaRecord &P);
  std::optional<mlir::DictionaryAttr>
  buildEntry(const PragmaRecord &P, const Resolved &var, bool isOff,
             int64_t &dimOut);
  bool validate(const PragmaRecord &P, const Resolved &var,
                llvm::SmallVectorImpl<mlir::NamedAttribute> &fields);
  void place(const PragmaRecord &P, TargetKey key, std::string what,
             std::string attrName, mlir::Attribute value, std::string conflict,
             std::string variable, int64_t dim);
  void placePartition(const PragmaRecord &P, TargetKey key, std::string what,
                      std::string attrName, mlir::DictionaryAttr value,
                      std::string conflict, std::string variable, int64_t dim,
                      int64_t rank);
  mlir::DictionaryAttr withDim(mlir::DictionaryAttr entry, int64_t dim);
  TargetBucket &bucket(TargetKey key, std::string what);
  void applyTripOnePipelines();
  void flush();

  mlir::NamedAttribute named(StringRef k, mlir::Attribute v) {
    return b.getNamedAttr(k, v);
  }
};

void FunctionSema::index(const Stmt *s, SourceRange scope, unsigned depth) {
  if (!s)
    return;
  if (const auto *L = dyn_cast<LambdaExpr>(s)) {
    excluded.push_back(L->getBody()->getSourceRange());
    return;
  }
  if (isa<BlockExpr>(s)) {
    excluded.push_back(s->getSourceRange());
    return;
  }
  if (const auto *DS = dyn_cast<DeclStmt>(s)) {
    for (const Decl *D : DS->decls()) {
      if (const auto *VD = dyn_cast<VarDecl>(D)) {
        decls.push_back({VD, scope});
        if (VD->getInit())
          index(VD->getInit(), scope, depth);
      } else if (const auto *UD = dyn_cast<UsingDirectiveDecl>(D)) {
        usingDirectives.push_back(UD);
      } else if (const auto *RD = dyn_cast<CXXRecordDecl>(D)) {
        for (const CXXMethodDecl *M : RD->methods())
          if (M->hasBody())
            excluded.push_back(M->getBody()->getSourceRange());
      }
    }
    return;
  }

  if (const auto *LS = dyn_cast<LabelStmt>(s))
    loopLabels[LS->getSubStmt()] = LS->getName();

  // A branch of an if whose condition is a constant (a template argument:
  // Vitis_Libraries vision `if (USE_URAM) { bind_storage ... impl=URAM }
  // else { bind_storage ... impl=BRAM }`) is not emitted, so the pragmas in
  // it are not either (clang's CodeGenFunction::EmitIfStmt skips a constant
  // dead arm; Vitis applies only the live arm's pragmas). Its pragmas do not
  // belong to this function.
  if (const auto *IS = dyn_cast<IfStmt>(s)) {
    const Expr *cond = IS->getCond();
    bool value = false;
    if (cond && !IS->getConditionVariable() && !IS->getInit() &&
        !cond->isValueDependent() &&
        cond->EvaluateAsBooleanCondition(value, AC)) {
      if (const Stmt *dead = value ? IS->getElse() : IS->getThen())
        excluded.push_back(dead->getSourceRange());
    }
  }

  const Stmt *body = nullptr;
  SourceLocation regionBegin, regionEnd;
  if (const auto *F = dyn_cast<ForStmt>(s)) {
    body = F->getBody();
    regionBegin = F->getRParenLoc();
  } else if (const auto *W = dyn_cast<WhileStmt>(s)) {
    body = W->getBody();
    regionBegin = W->getRParenLoc();
  } else if (const auto *R = dyn_cast<CXXForRangeStmt>(s)) {
    body = R->getBody();
    regionBegin = R->getRParenLoc();
  } else if (const auto *D = dyn_cast<DoStmt>(s)) {
    body = D->getBody();
    regionBegin = D->getDoLoc();
    regionEnd = D->getWhileLoc();
  }
  if (body) {
    if (regionEnd.isInvalid())
      regionEnd = body->getEndLoc();
    loops.push_back({s, {regionBegin, regionEnd}, depth + 1});
    ++depth;
  }

  SourceRange childScope = scope;
  if (isa<CompoundStmt, ForStmt, CXXForRangeStmt, IfStmt, SwitchStmt,
          WhileStmt>(s))
    childScope = s->getSourceRange();

  if (const auto *CS = dyn_cast<CompoundStmt>(s)) {
    BlockInfo info;
    info.range = CS->getSourceRange();
    for (const Stmt *c : CS->body()) {
      const Stmt *inner = c;
      while (const auto *LS = dyn_cast<LabelStmt>(inner))
        inner = LS->getSubStmt();
      info.children.push_back(
          {c->getBeginLoc(),
           isa<ForStmt, WhileStmt, DoStmt, CXXForRangeStmt>(inner)});
    }
    blocks.push_back(std::move(info));
  }

  for (const Stmt *c : s->children())
    index(c, childScope, depth);
}

bool FunctionSema::belongsHere(SourceLocation p) const {
  const Stmt *body = FD->getBody();
  if (!body || !before(body->getBeginLoc(), p) ||
      !before(p, body->getEndLoc()))
    return false;
  for (SourceRange r : excluded)
    if (inRange(r, p))
      return false;
  return true;
}

const FunctionSema::LoopInfo *
FunctionSema::innermostLoop(SourceLocation p) const {
  const LoopInfo *best = nullptr;
  for (const LoopInfo &L : loops)
    if (inRegion(L.region, p) && (!best || L.depth > best->depth))
      best = &L;
  return best;
}

const FunctionSema::BlockInfo *
FunctionSema::innermostBlock(SourceLocation p) const {
  const BlockInfo *inner = nullptr;
  for (const BlockInfo &B : blocks)
    if (inRange(B.range, p) && (!inner || rangeInside(B.range, inner->range)))
      inner = &B;
  return inner;
}

bool FunctionSema::inFunctionOrForBody(SourceLocation p,
                                       const LoopInfo *loop) const {
  const BlockInfo *inner = innermostBlock(p);
  if (!inner)
    return false;
  if (inner->range == FD->getBody()->getSourceRange())
    return true;
  if (!loop)
    return false;
  const Stmt *body = nullptr;
  if (const auto *F = dyn_cast<ForStmt>(loop->stmt))
    body = F->getBody();
  else if (const auto *R = dyn_cast<CXXForRangeStmt>(loop->stmt))
    body = R->getBody();
  return body && isa<CompoundStmt>(body) &&
         inner->range == body->getSourceRange();
}

bool FunctionSema::loopFollows(SourceLocation p) const {
  const BlockInfo *inner = innermostBlock(p);
  if (!inner)
    return false;
  for (auto &[loc, isLoop] : inner->children)
    if (before(p, loc))
      return isLoop;
  return false;
}

//===----------------------------------------------------------------------===//
// Name and constant resolution
//===----------------------------------------------------------------------===//

const VarDecl *FunctionSema::lookupLocal(StringRef name,
                                         SourceLocation p) const {
  const DeclInfo *visible = nullptr, *later = nullptr;
  for (const DeclInfo &D : decls) {
    if (identName(D.vd) != name || !inRange(D.scope, p))
      continue;
    const DeclInfo *&slot = before(D.vd->getLocation(), p) ? visible : later;
    if (!slot || rangeInside(D.scope, slot->scope))
      slot = &D;
  }
  // C scoping first; Vitis also accepts a pragma placed before the
  // declaration it names, in the same or an enclosing scope.
  if (visible)
    return visible->vd;
  return later ? later->vd : nullptr;
}

const NamedDecl *FunctionSema::lookupNonLocal(StringRef qualified) const {
  llvm::SmallVector<StringRef, 4> parts;
  qualified.split(parts, "::");
  std::function<const NamedDecl *(const DeclContext *, StringRef)> lookupIn =
      [&](const DeclContext *DC, StringRef name) -> const NamedDecl * {
    if (name.empty())
      return nullptr;
    DeclarationName DN(&AC.Idents.get(name));
    for (NamedDecl *ND : DC->lookup(DN)) {
      // `using Base<C>::B;` in a class: the declaration it brings in.
      if (auto *shadow = dyn_cast<UsingShadowDecl>(ND))
        return shadow->getTargetDecl();
      if (auto *ud = dyn_cast<UsingDecl>(ND)) {
        for (UsingShadowDecl *shadow : ud->shadows())
          return shadow->getTargetDecl();
        continue;
      }
      return ND;
    }
    // A member of a class is also found in its bases (a static constexpr
    // member of a base class template: Vitis_Libraries vision
    // `COLS_ADDRBOUND`), as C++ name lookup does.
    if (const auto *RD = dyn_cast<CXXRecordDecl>(DC))
      if (RD->hasDefinition())
        for (const CXXBaseSpecifier &B : RD->getDefinition()->bases())
          if (const auto *BRD = B.getType()->getAsCXXRecordDecl())
            if (const NamedDecl *ND = lookupIn(BRD, name))
              return ND;
    return nullptr;
  };

  // First component: enclosing contexts, then using-directives (those in the
  // function body and those of each enclosing context).
  const NamedDecl *found = nullptr;
  size_t next = 1;
  if (parts.size() > 1 && parts.front().empty()) { // ::name
    found = lookupIn(AC.getTranslationUnitDecl(), parts[1]);
    next = 2;
  } else {
    for (const DeclContext *DC = FD->getDeclContext(); DC && !found;
         DC = DC->getParent()) {
      found = lookupIn(DC, parts.front());
      for (const UsingDirectiveDecl *UD : DC->using_directives())
        if (!found)
          found = lookupIn(UD->getNominatedNamespace(), parts.front());
    }
    for (const UsingDirectiveDecl *UD : usingDirectives)
      if (!found)
        found = lookupIn(UD->getNominatedNamespace(), parts.front());
  }
  // A type template parameter of this instantiation (or of an enclosing
  // class template specialization) names its argument's class:
  // `#pragma HLS PIPELINE II = CholeskyTraits::INNER_II` with CholeskyTraits
  // a template parameter (Vitis_Libraries solver cholesky.hpp, qrf.hpp).
  auto templateTypeArgument = [&](StringRef name) -> const NamedDecl * {
    auto fromList = [&](const TemplateParameterList *params,
                        const TemplateArgumentList &args) -> const NamedDecl * {
      for (unsigned i = 0; i < params->size() && i < args.size(); ++i)
        if (identName(params->getParam(i)) == name &&
            args[i].getKind() == TemplateArgument::Type)
          if (const auto *RD = args[i].getAsType()->getAsCXXRecordDecl())
            return RD;
      return nullptr;
    };
    if (const FunctionTemplateDecl *T = FD->getPrimaryTemplate())
      if (const TemplateArgumentList *args = FD->getTemplateSpecializationArgs())
        if (const NamedDecl *ND = fromList(T->getTemplateParameters(), *args))
          return ND;
    for (const DeclContext *DC = FD->getDeclContext(); DC; DC = DC->getParent())
      if (const auto *CTS = dyn_cast<ClassTemplateSpecializationDecl>(DC))
        if (const NamedDecl *ND = fromList(
                CTS->getSpecializedTemplate()->getTemplateParameters(),
                CTS->getTemplateArgs()))
          return ND;
    return nullptr;
  };
  if (!found && parts.size() > 1 && !parts.front().empty())
    found = templateTypeArgument(parts.front());
  // Remaining components: members of namespaces, classes and enums; a
  // typedef of a class names the class.
  for (; found && next < parts.size(); ++next) {
    if (const auto *TD = dyn_cast<TypedefNameDecl>(found))
      if (const auto *RD = TD->getUnderlyingType()->getAsCXXRecordDecl())
        found = RD;
    const auto *DC = dyn_cast<DeclContext>(found);
    if (!DC)
      return nullptr;
    found = lookupIn(DC, parts[next]);
  }
  return found;
}

std::optional<int64_t> FunctionSema::lookupConstant(StringRef name,
                                                    SourceLocation p) {
  if (name == "true")
    return 1;
  if (name == "false")
    return 0;

  // Template arguments of this instantiation (function or enclosing class).
  auto fromTemplate = [&](const TemplateParameterList *params,
                          const TemplateArgumentList &args)
      -> std::optional<int64_t> {
    for (unsigned i = 0; i < params->size() && i < args.size(); ++i)
      if (identName(params->getParam(i)) == name &&
          args[i].getKind() == TemplateArgument::Integral)
        return args[i].getAsIntegral().getSExtValue();
    return std::nullopt;
  };
  if (const FunctionTemplateDecl *T = FD->getPrimaryTemplate())
    if (const TemplateArgumentList *args = FD->getTemplateSpecializationArgs())
      if (auto v = fromTemplate(T->getTemplateParameters(), *args))
        return v;
  for (const DeclContext *DC = FD->getDeclContext(); DC; DC = DC->getParent())
    if (const auto *CTS = dyn_cast<ClassTemplateSpecializationDecl>(DC))
      if (auto v = fromTemplate(
              CTS->getSpecializedTemplate()->getTemplateParameters(),
              CTS->getTemplateArgs()))
        return v;

  auto evalVar = [&](const VarDecl *VD) -> std::optional<int64_t> {
    if (!VD->getType()->isIntegralOrEnumerationType())
      return std::nullopt;
    if (!VD->isConstexpr() && !VD->getType().isConstQualified())
      return std::nullopt;
    // A static data member of a class template specialization is
    // instantiated with its initializer (in-class initializers of static
    // constexpr members are instantiated with the class).
    const Expr *init = VD->getAnyInitializer();
    if (!init)
      return std::nullopt;
    Expr::EvalResult result;
    if (init->EvaluateAsInt(result, AC))
      return result.Val.getInt().getSExtValue();
    return std::nullopt;
  };
  if (!name.contains("::"))
    if (const VarDecl *VD = lookupLocal(name, p))
      return evalVar(VD);
  if (const NamedDecl *ND = lookupNonLocal(name)) {
    if (const auto *EC = dyn_cast<EnumConstantDecl>(ND))
      return EC->getInitVal().getSExtValue();
    if (const auto *VD = dyn_cast<VarDecl>(ND))
      return evalVar(VD);
  }
  // An enumerator of an unscoped enum declared in the function body:
  // `enum { fifo_depth = burst_len * 2 }; #pragma HLS STREAM ... depth =
  // fifo_depth` (Vitis_Libraries database scan_col_2.hpp). Looked up last,
  // so a name the lookups above resolve keeps its meaning.
  if (!name.contains("::"))
    for (const Decl *D : FD->decls())
      if (const auto *ED = dyn_cast<EnumDecl>(D))
        if (!ED->isScoped())
          for (const EnumConstantDecl *EC : ED->enumerators())
            if (identName(EC) == name)
              return EC->getInitVal().getSExtValue();
  return std::nullopt;
}

/// Evaluates `name(args)` for a constexpr function (or an already
/// instantiated specialization of a constexpr function template) with
/// integral parameters, using clang's constant evaluator on a synthesized
/// call expression.
std::optional<int64_t>
FunctionSema::callConstant(StringRef name, llvm::ArrayRef<int64_t> args) {
  llvm::SmallVector<const FunctionDecl *, 4> candidates;
  auto consider = [&](const NamedDecl *ND) {
    if (const auto *FD = dyn_cast<FunctionDecl>(ND))
      candidates.push_back(FD);
    else if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(ND))
      for (const FunctionDecl *Spec : FTD->specializations())
        candidates.push_back(Spec);
  };
  llvm::SmallVector<StringRef, 4> parts;
  name.split(parts, "::");
  if (const NamedDecl *first = lookupNonLocal(name)) {
    // All overloads in the context that declares the first match.
    const DeclContext *DC = first->getDeclContext()->getRedeclContext();
    DeclarationName DN(&AC.Idents.get(parts.back()));
    for (const NamedDecl *ND : DC->lookup(DN))
      consider(ND);
    if (candidates.empty())
      consider(first);
  }
  for (const FunctionDecl *FD : candidates) {
    FD = FD->getDefinition() ? FD->getDefinition() : FD;
    if (!FD->isConstexpr() || FD->getNumParams() != args.size())
      continue;
    llvm::SmallVector<Expr *, 4> argExprs;
    bool ok = true;
    for (unsigned i = 0; i < args.size() && ok; ++i) {
      QualType T = FD->getParamDecl(i)->getType().getCanonicalType();
      if (!T->isIntegerType() || T->isBooleanType()) {
        ok = false;
        break;
      }
      unsigned width = AC.getIntWidth(T);
      argExprs.push_back(IntegerLiteral::Create(
          AC, llvm::APInt(width, (uint64_t)args[i], T->isSignedIntegerType()),
          T.getUnqualifiedType(), SourceLocation()));
    }
    if (!ok)
      continue;
    auto *fn = const_cast<FunctionDecl *>(FD);
    auto *ref = DeclRefExpr::Create(AC, NestedNameSpecifierLoc(),
                                    SourceLocation(), fn, false,
                                    SourceLocation(), fn->getType(), VK_LValue);
    auto *callee = ImplicitCastExpr::Create(
        AC, AC.getPointerType(fn->getType()), CK_FunctionToPointerDecay, ref,
        nullptr, VK_PRValue, FPOptionsOverride());
    auto *callExpr = CallExpr::Create(
        AC, callee, argExprs, fn->getReturnType().getNonReferenceType(),
        VK_PRValue, SourceLocation(), FPOptionsOverride());
    Expr::EvalResult result;
    if (callExpr->EvaluateAsInt(result, AC))
      return result.Val.getInt().getSExtValue();
  }
  return std::nullopt;
}

/// Parses an integer literal as spelled (after macro expansion).
std::optional<int64_t> parseIntLiteral(StringRef text) {
  text = text.rtrim("uUlLzZ");
  int64_t v;
  if (text.size() > 1 && text[0] == '0' && llvm::isDigit(text[1])) {
    if (!text.drop_front().getAsInteger(8, v))
      return v;
    return std::nullopt;
  }
  if (!text.getAsInteger(0, v))
    return v;
  return std::nullopt;
}

/// Recursive-descent evaluator for integer constant expressions in options.
class IntExprEvaluator {
public:
  using Lookup = llvm::function_ref<std::optional<int64_t>(StringRef)>;
  using Call = llvm::function_ref<std::optional<int64_t>(
      StringRef, llvm::ArrayRef<int64_t>)>;
  IntExprEvaluator(const std::vector<ValueToken> &toks, Lookup lookup,
                   Call call)
      : toks(toks), lookup(lookup), call(call) {}

  std::optional<int64_t> evaluate() {
    auto v = parseBinary(0);
    if (!v || pos != toks.size())
      return std::nullopt;
    return v;
  }
  std::string unresolved;

private:
  const std::vector<ValueToken> &toks;
  Lookup lookup;
  Call call;
  size_t pos = 0;

  static int precedence(StringRef op) {
    return llvm::StringSwitch<int>(op)
        .Cases("*", "/", "%", 5)
        .Cases("+", "-", 4)
        .Cases("<<", ">>", 3)
        .Case("&", 2)
        .Case("^", 1)
        .Case("|", 0)
        .Default(-1);
  }

  std::optional<int64_t> parsePrimary() {
    if (pos >= toks.size())
      return std::nullopt;
    const ValueToken &T = toks[pos++];
    if (T.text == "(") {
      auto v = parseBinary(0);
      if (!v || pos >= toks.size() || toks[pos].text != ")")
        return std::nullopt;
      ++pos;
      return v;
    }
    if (T.text == "-" || T.text == "+" || T.text == "~") {
      auto v = parsePrimary();
      if (!v)
        return std::nullopt;
      return T.text == "-" ? -*v : T.text == "~" ? ~*v : *v;
    }
    if (T.kind == ValueToken::Number)
      return parseIntLiteral(T.text);
    if (T.kind == ValueToken::Identifier || T.text == "::") {
      // Qualified name: [::] ident (:: ident)*
      std::string name = T.text;
      while (pos + 1 < toks.size() && toks[pos].text == "::" &&
             toks[pos + 1].kind == ValueToken::Identifier) {
        name += "::" + toks[pos + 1].text;
        pos += 2;
      }
      if (T.text == "::" && pos < toks.size() &&
          toks[pos].kind == ValueToken::Identifier)
        name += toks[pos++].text;
      // A call to a constexpr function: name(args...)
      if (pos < toks.size() && toks[pos].text == "(") {
        ++pos;
        llvm::SmallVector<int64_t, 4> args;
        while (pos < toks.size() && toks[pos].text != ")") {
          auto a = parseBinary(0);
          if (!a)
            return std::nullopt;
          args.push_back(*a);
          if (pos < toks.size() && toks[pos].text == ",")
            ++pos;
        }
        if (pos >= toks.size())
          return std::nullopt;
        ++pos; // ')'
        if (auto v = call(name, args))
          return v;
        if (unresolved.empty())
          unresolved = name + "(...)";
        return std::nullopt;
      }
      if (auto v = lookup(name))
        return v;
      if (unresolved.empty())
        unresolved = name;
      return std::nullopt;
    }
    return std::nullopt;
  }

  std::optional<int64_t> parseBinary(int minPrec) {
    auto lhs = parsePrimary();
    while (lhs && pos < toks.size()) {
      int prec = precedence(toks[pos].text);
      if (prec < minPrec || prec < 0)
        break;
      StringRef op = toks[pos++].text;
      auto rhs = parseBinary(prec + 1);
      if (!rhs)
        return std::nullopt;
      if ((op == "/" || op == "%") && *rhs == 0)
        return std::nullopt;
      int64_t l = *lhs, r = *rhs;
      lhs = llvm::StringSwitch<int64_t>(op)
                .Case("*", l * r)
                .Case("/", r ? l / r : 0)
                .Case("%", r ? l % r : 0)
                .Case("+", l + r)
                .Case("-", l - r)
                .Case("<<", l << r)
                .Case(">>", l >> r)
                .Case("&", l & r)
                .Case("^", l ^ r)
                .Case("|", l | r)
                .Default(0);
    }
    return lhs;
  }
};

std::optional<int64_t> FunctionSema::evalInt(const PragmaRecord &P,
                                             const ParsedOption &O) {
  // Named: the evaluator only keeps function_refs to them.
  auto lookup = [&](StringRef name) { return lookupConstant(name, P.loc); };
  auto call = [&](StringRef name, llvm::ArrayRef<int64_t> args) {
    return callConstant(name, args);
  };
  IntExprEvaluator eval(O.value, lookup, call);
  std::optional<int64_t> v = eval.evaluate();
  if (v)
    return v;
  if (!eval.unresolved.empty())
    S.error(O.value.front().loc,
            "cannot resolve '%0' to an integer constant in '%1'")
        << eval.unresolved << pragmaDisplayName(P);
  else
    S.error(O.value.front().loc,
            "invalid integer expression '%0' for option '%1' in '%2'")
        << O.valueText() << O.spec->displayName() << pragmaDisplayName(P);
  return std::nullopt;
}

Resolved FunctionSema::resolveVariable(const PragmaRecord &P,
                                       const ParsedOption &O,
                                       bool allowReturn) {
  Resolved R;
  std::string path = O.valueText();
  R.path = path;
  std::string display = pragmaDisplayName(P);
  bool xlx = cgeist::hls::xlxHeaders();
  // A qualified name `ns::var` (Vitis etc/hls_hotbm_apfixed.h:91) is looked
  // up as a non-local name; the member path starts after it.
  std::string qualified = O.value.front().text;
  size_t pathStart = 1;
  while (xlx && pathStart + 1 < O.value.size() &&
         O.value[pathStart].text == "::" &&
         O.value[pathStart + 1].kind == ValueToken::Identifier) {
    qualified += "::" + O.value[pathStart + 1].text;
    pathStart += 2;
  }
  bool isQualified = pathStart > 1;
  StringRef base = qualified;
  bool isPlainName = O.value.size() == 1;

  if (allowReturn && base == "return" && isPlainName) {
    R.kind = Resolved::Return;
    return R;
  }

  // `variable=this` inside an instance member function names the object the
  // function is called on, which is the function's first argument. Vitis's
  // own hls_vector.h relies on it (`SYN_PRAGMA(HLS AGGREGATE variable=this)`
  // in the member functions), so an hls::vector variable is aggregated once
  // such a member is applied to it; only the aggregate pragmas take it.
  // Under --xlx-headers `*this` is the same object.
  bool namesThis = (base == "this" && isPlainName) ||
                   (xlx && isThisReference(O.value));
  if (namesThis && P.spec &&
      (P.spec->name == "aggregate" || P.spec->name == "disaggregate")) {
    const auto *MD = dyn_cast<CXXMethodDecl>(FD);
    if (MD && MD->isInstance() && argOffset == 1) {
      R.kind = Resolved::Param;
      R.paramIndex = 0;
      R.type = MD->getThisType();
      return R;
    }
  }
  // Any other `this` / `*this` (Vitis header pragmas, e.g. hls_stream_39.h:162)
  // is not a variable of this function; the pragma is reported and dropped.
  // Only under --xlx-headers (the parser rejects `*this` without it).
  if (xlx && isThisReference(O.value)) {
    S.warning(O.value.front().loc,
              "'%0' in '%1' names the enclosing object; ignored")
        << path << display;
    return {};
  }

  if (isQualified) {
    // Only the non-local lookup below can find a qualified name.
  } else if (const VarDecl *VD = lookupLocal(base, P.loc)) {
    R.vd = VD;
    R.kind = VD->hasGlobalStorage() ? Resolved::Global : Resolved::Local;
    R.type = VD->getType();
  } else {
    unsigned i = 0;
    for (const ParmVarDecl *PV : FD->parameters()) {
      if (identName(PV) == base) {
        R.kind = Resolved::Param;
        R.vd = PV;
        R.paramIndex = argOffset + i;
        R.type = PV->getOriginalType();
        break;
      }
      ++i;
    }
  }
  if (R.kind == Resolved::None) {
    if (const NamedDecl *ND = lookupNonLocal(base)) {
      if (const auto *VD = dyn_cast<VarDecl>(ND);
          VD && VD->hasGlobalStorage()) {
        R.kind = Resolved::Global;
        R.vd = VD->getCanonicalDecl();
        R.type = VD->getType();
      } else if (const auto *F = dyn_cast<FieldDecl>(ND)) {
        // A data member named inside an instance member function (a
        // constructor partitioning its member array: Vitis_Libraries blas
        // `WindowRm::m_Val`, security aes `key_list`) is `this->member`:
        // Vitis parses the variable as an expression of the function
        // (hls-llvm-project Parser::ParseHLSVariableExpression), so the
        // pragma applies to that member of the object the function runs on.
        // An array_partition / array_reshape keeps the member path
        // (member_path on the `this` argument, carried to the object at
        // inline time); any other pragma on a member is reported and dropped
        // (not modelled).
        const auto *MD = dyn_cast<CXXMethodDecl>(FD);
        llvm::SmallVector<int64_t, 2> positions;
        if (MD && MD->isInstance() && argOffset == 1)
          positions = fieldPositionPath(MD->getParent(), F);
        bool partition = P.spec && (P.spec->name == "array_partition" ||
                                    P.spec->name == "array_reshape");
        if (!positions.empty() && partition) {
          R.kind = Resolved::Param;
          R.paramIndex = 0;
          R.type = F->getType();
          R.memberPositions = positions;
          R.viaThis = true;
        } else if (!positions.empty()) {
          S.warning(O.value.front().loc,
                    "'%0' in '%1' names a data member of the object; the "
                    "pragma is ignored (only array_partition and "
                    "array_reshape on members are modelled)")
              << base << display;
          return {};
        } else {
          S.error(O.value.front().loc,
                  "'%0' in '%1' names a class member; only local variables, "
                  "parameters and globals can be annotated")
              << base << display;
          return {};
        }
      }
    }
  }
  if (R.kind == Resolved::None && !isQualified) {
    // Declared only in a nested scope that does not contain the pragma
    // (e.g. a buffer inside a loop, pragma at the top of the function).
    // Vitis resolves such names within the function; accept a unique match.
    llvm::SmallVector<const VarDecl *, 2> nested;
    for (const DeclInfo &D : decls)
      if (identName(D.vd) == base)
        nested.push_back(D.vd);
    if (nested.size() == 1) {
      S.warning(O.value.front().loc,
                "'%0' is declared in a nested scope that does not contain "
                "'%1'; using that declaration")
          << base << display;
      R.vd = nested.front();
      R.kind = R.vd->hasGlobalStorage() ? Resolved::Global : Resolved::Local;
      R.type = R.vd->getType();
    } else if (nested.size() > 1) {
      S.error(O.value.front().loc,
              "'%0' in '%1' is ambiguous: it is declared in several nested "
              "scopes that do not contain the pragma")
          << base << display;
      return {};
    }
  }
  if (R.kind == Resolved::None) {
    S.error(O.value.front().loc, "use of undeclared variable '%0' in '%1'")
        << base << display;
    return {};
  }

  // Member and element path: s.a[2]->b
  const std::vector<ValueToken> &v = O.value;
  for (size_t i = pathStart; i < v.size();) {
    QualType t = R.type.getNonReferenceType();
    if (v[i].text == "[") {
      size_t close = i + 1;
      while (close < v.size() && v[close].text != "]")
        ++close;
      std::vector<ValueToken> index(v.begin() + i + 1, v.begin() + close);
      auto lookup = [&](StringRef n) { return lookupConstant(n, P.loc); };
      auto call = [&](StringRef n, llvm::ArrayRef<int64_t> a) {
        return callConstant(n, a);
      };
      std::optional<int64_t> idx =
          IntExprEvaluator(index, lookup, call).evaluate();
      if (const auto *CAT = AC.getAsConstantArrayType(t)) {
        int64_t size = CAT->getSize().getSExtValue();
        if (idx && (*idx < 0 || *idx >= size)) {
          S.error(v[i].loc, "index %0 is out of range for '%1' of size %2 "
                            "(in '%3')")
              << *idx << t.getAsString() << size << display;
          return {};
        }
        R.type = CAT->getElementType();
      } else if (const auto *AT = AC.getAsArrayType(t)) {
        R.type = AT->getElementType();
      } else if (const auto *PT = t->getAs<PointerType>()) {
        R.type = PT->getPointeeType();
      } else {
        S.error(v[i].loc, "subscripted value of type '%0' is not an array "
                          "(in '%1')")
            << t.getAsString() << display;
        return {};
      }
      i = close + 1;
      R.hasSubscript = true;
      continue;
    }
    // '.' or '->' member
    StringRef member = v[i + 1].text;
    if (const auto *PT = t->getAs<PointerType>())
      t = PT->getPointeeType();
    const RecordDecl *RD = t->getAsRecordDecl();
    const FieldDecl *found = nullptr;
    if (RD)
      for (const FieldDecl *F : RD->fields())
        if (identName(F) == member)
          found = F;
    if (!found) {
      S.error(v[i + 1].loc, "no member named '%0' in '%1' (in '%2')")
          << member << t.getAsString() << display;
      return {};
    }
    R.type = found->getType();
    unsigned bases = 0;
    if (const auto *CXRD = dyn_cast<CXXRecordDecl>(RD))
      bases = CXRD->getNumBases();
    R.memberPositions.push_back(static_cast<int64_t>(bases) +
                                found->getFieldIndex());
    i += 2;
  }
  if (R.hasSubscript)
    R.memberPositions.clear();
  return R;
}

//===----------------------------------------------------------------------===//
// Entries
//===----------------------------------------------------------------------===//

std::optional<mlir::DictionaryAttr>
FunctionSema::buildEntry(const PragmaRecord &P, const Resolved &var,
                         bool isOff, int64_t &dimOut) {
  llvm::SmallVector<mlir::NamedAttribute, 8> fields;
  bool ok = true;
  const PragmaSpec &spec = *P.spec;

  for (const OptionSpec &os : spec.options) {
    if (os.name == "off")
      continue;
    const ParsedOption *O = P.get(os.name);
    // With `off`, only the naming options (and dim) are kept.
    if (isOff && os.type != OptType::Variable && os.name != "dim")
      continue;
    mlir::Attribute value;
    if (O) {
      switch (os.type) {
      case OptType::Flag: {
        std::string v = O->hasValue ? StringRef(O->valueText()).lower()
                                    : std::string("true");
        value = b.getBoolAttr(v == "true" || v == "1" || v == "on" ||
                              v == "yes");
        break;
      }
      case OptType::Int: {
        std::optional<int64_t> v = evalInt(P, *O);
        if (!v) {
          ok = false;
          continue;
        }
        // An unroll factor below 1, written as a literal (`factor=0`,
        // `factor=-1`) or as an expression that folds below 1 (hp_fft_hls
        // `unroll factor=UF>>(STAGE-1)` with UF 1, STAGE 2): Vitis 2025.1
        // ignores the pragma with a warning ([HLS 207-5577] "Unroll pragma
        // is ignored, because 'factor' is not positive integer", kernels
        // w9r6_uf0 and w9r6rev uflit0, CIRCT OPEN_ISSUES O70 (b)) and
        // decides on the loop automatically; so does cgeist. Other options
        // below their minimum stay errors (an array_partition factor that
        // folds to 0 fails in Vitis too, [HLS 200-1715], kernel w9r6rev apz).
        if (os.min && *v < *os.min && spec.name == "unroll" &&
            os.name == "factor") {
          S.warning(O->value.front().loc,
                    "option '%0' in '%1' is %2, below %3; the pragma "
                    "is ignored, as Vitis ignores it")
              << os.displayName() << pragmaDisplayName(P) << *v << *os.min;
          return std::nullopt;
        }
        if (os.min && *v < *os.min) {
          S.error(O->value.front().loc,
                  "option '%0' in '%1' must be >= %2, got %3")
              << os.displayName() << pragmaDisplayName(P) << *os.min << *v;
          ok = false;
          continue;
        }
        if (*v > INT32_MAX || *v < INT32_MIN) {
          S.error(O->value.front().loc,
                  "option '%0' in '%1' is out of range: %2")
              << os.displayName() << pragmaDisplayName(P) << *v;
          ok = false;
          continue;
        }
        value = b.getI32IntegerAttr(*v);
        break;
      }
      case OptType::Number: {
        StringRef text = O->valueText();
        bool isFloat = O->value.size() == 1 &&
                       O->value[0].kind == ValueToken::Number &&
                       !text.starts_with("0x") &&
                       text.find_first_of(".eE") != StringRef::npos;
        if (isFloat) {
          double d = std::strtod(text.rtrim("fFlL").str().c_str(), nullptr);
          if (!(d > 0)) {
            S.error(O->value.front().loc,
                    "option '%0' in '%1' must be positive, got %2")
                << os.displayName() << pragmaDisplayName(P) << text;
            ok = false;
            continue;
          }
          value = b.getF64FloatAttr(d);
        } else {
          std::optional<int64_t> v = evalInt(P, *O);
          if (!v) {
            ok = false;
            continue;
          }
          if (*v <= 0 || *v > INT32_MAX) {
            S.error(O->value.front().loc,
                    "option '%0' in '%1' must be positive, got %2")
                << os.displayName() << pragmaDisplayName(P) << *v;
            ok = false;
            continue;
          }
          value = b.getI32IntegerAttr(*v);
        }
        break;
      }
      case OptType::Enum:
      case OptType::IntOrEnum: {
        std::string lower = StringRef(O->valueText()).lower();
        if (llvm::is_contained(os.values, StringRef(lower))) {
          value = b.getStringAttr(os.upperCase ? StringRef(lower).upper()
                                               : lower);
        } else if (O->value.size() == 1 &&
                   O->value[0].kind == ValueToken::Identifier &&
                   !lookupConstant(O->value[0].text, P.loc)) {
          // Neither an enumerator of the option nor a named constant.
          std::string list;
          for (StringRef v : os.values)
            list += (list.empty() ? "" : ", ") + v.str();
          S.error(O->value.front().loc,
                  "invalid value '%0' for option '%1' in '%2': expected an "
                  "integer or one of: %3")
              << O->valueText() << os.displayName() << pragmaDisplayName(P)
              << list;
          ok = false;
          continue;
        } else {
          std::optional<int64_t> v = evalInt(P, *O);
          if (!v) {
            ok = false;
            continue;
          }
          value = b.getI32IntegerAttr(*v);
        }
        break;
      }
      case OptType::Variable:
        // A member of `this` reads as the member path `this.m_Val`, which
        // hls-est's struct splitting places on the member's memory.
        value = b.getStringAttr(var.viaThis ? "this." + O->valueText()
                                            : O->valueText());
        break;
      case OptType::Name:
        value = b.getStringAttr(O->valueText());
        break;
      case OptType::NameList:
      case OptType::IntList: {
        llvm::SmallVector<mlir::Attribute> items;
        ParsedOption item = *O;
        auto flushItem = [&]() {
          if (os.type == OptType::NameList) {
            items.push_back(b.getStringAttr(item.valueText()));
          } else if (auto v = evalInt(P, item)) {
            items.push_back(b.getI32IntegerAttr(*v));
          } else {
            ok = false;
          }
          item.value.clear();
        };
        item.value.clear();
        for (const ValueToken &T : O->value) {
          if (T.text == ",")
            flushItem();
          else
            item.value.push_back(T);
        }
        flushItem();
        value = b.getArrayAttr(items);
        break;
      }
      }
    } else {
      switch (os.defaultKind) {
      case OptionSpec::DefaultKind::None:
        continue;
      case OptionSpec::DefaultKind::Bool:
        value = b.getBoolAttr(os.defaultBool);
        break;
      case OptionSpec::DefaultKind::Int:
        value = b.getI32IntegerAttr(os.defaultInt);
        break;
      case OptionSpec::DefaultKind::String:
        value = b.getStringAttr(os.defaultString);
        break;
      }
      // A stream pragma without a depth: the consumer needs to know that
      // the depth is the default, because Vitis's default depends on the
      // variable: 2 for an hls::stream, the array's size for an array
      // (CIRCT W9 round 8 review defect 4, kernel w9r8rev s1_nodepth.cpp:
      // `#pragma HLS stream variable=buf` on int buf[1024] is
      // fifo_w32_d1024). The depth key keeps the default 2.
      if (spec.name == "stream" && os.name == "depth")
        fields.push_back(named("depth_defaulted", b.getUnitAttr()));
    }
    fields.push_back(named(os.key(), value));
  }
  if (!ok)
    return std::nullopt;
  // An array_partition on a member path: which member, by its position in
  // the struct type, so that hls-est's struct splitting finds the member's
  // memory in a struct with several array members (CIRCT OPEN_ISSUES O78;
  // W9r7 review defect 10, kernel w9r7rev_mp_two).
  if ((spec.name == "array_partition" || spec.name == "array_reshape") &&
      !var.memberPositions.empty()) {
    llvm::SmallVector<mlir::Attribute, 2> positions;
    for (int64_t p : var.memberPositions)
      positions.push_back(b.getI32IntegerAttr(static_cast<int32_t>(p)));
    fields.push_back(named("member_path", b.getArrayAttr(positions)));
  }
  if (!isOff && !validate(P, var, fields))
    return std::nullopt;
  if (isOff) {
    for (const ParsedOption &O : P.options)
      if (O.spec && O.spec->name != "off" &&
          O.spec->type != OptType::Variable && O.spec->name != "dim") {
        S.warning(O.loc, "option '%0' is ignored because '%1' has 'off'")
            << O.spec->displayName() << pragmaDisplayName(P);
        break;
      }
  }

  mlir::DictionaryAttr dict = b.getDictionaryAttr(fields);
  if (auto d = dict.getAs<mlir::IntegerAttr>("dim"))
    dimOut = d.getInt();
  return dict;
}

/// Pragma-specific semantic checks. May adjust `fields`.
bool FunctionSema::validate(const PragmaRecord &P, const Resolved &var,
                            llvm::SmallVectorImpl<mlir::NamedAttribute> &fields) {
  StringRef name = P.spec->name;
  std::string display = pragmaDisplayName(P);
  auto get = [&](StringRef key) -> mlir::Attribute {
    for (auto &f : fields)
      if (f.getName() == key)
        return f.getValue();
    return {};
  };
  auto getInt = [&](StringRef key) -> std::optional<int64_t> {
    if (auto a = llvm::dyn_cast_or_null<mlir::IntegerAttr>(get(key)))
      return a.getInt();
    return std::nullopt;
  };
  auto erase = [&](StringRef key) {
    llvm::erase_if(fields, [&](auto &f) { return f.getName() == key; });
  };
  auto locOf = [&](StringRef opt) {
    const ParsedOption *O = P.get(opt);
    return O ? O->loc : P.loc;
  };

  if (name == "array_partition" || name == "array_reshape") {
    llvm::SmallVector<int64_t, 4> shape = arrayShape(var.type);
    if (shape.empty() && !partitionShape(var.type).empty()) {
      // A class wrapping an array (a single hls::vector, not an array of
      // them): Vitis ignores the pragma ([HLS 214-253] "Ignored unsupported
      // array_partition pragma on variable"; kernel w9r7fix_vp_local and
      // class_ece8893 nikhil_p lab3 `col_ptr_B_local`; W9r7 review defect
      // 8a), so it is dropped with a warning instead of rejected.
      S.warning(locOf("variable"),
                "'%0' on '%1' of class type '%2' is ignored (Vitis: "
                "unsupported array_partition pragma)")
          << display << var.path << var.type.getAsString();
      return false;
    }
    if (shape.empty()) {
      // A scalar of class type (an ap_uint<64>): Vitis 2025.1 accepts the
      // pragma without a message and nothing is partitioned (kernel
      // w9r22i_c7_array_typedef; Vitis_Libraries blas transpMatB2.hpp
      // `l_valIn`); a scalar of a builtin type stays an error.
      if (var.type.getNonReferenceType()->isRecordType()) {
        S.warning(locOf("variable"),
                  "'%0' on the scalar '%1' of type '%2' is ignored, as Vitis "
                  "ignores it")
            << display << var.path << var.type.getAsString();
        return false;
      }
      S.error(locOf("variable"),
              "'%0' requires an array or pointer, but '%1' has type '%2'")
          << display << var.path << var.type.getAsString();
      return false;
    }
    StringRef kind = llvm::cast<mlir::StringAttr>(get("kind")).getValue();
    std::optional<int64_t> factor = getInt("factor");
    if (kind == "complete" && factor) {
      S.warning(locOf("factor"),
                "'factor' is ignored for complete partitioning in '%0'")
          << display;
      erase("factor");
      factor.reset();
    }
    if (kind != "complete" && !factor) {
      S.error(P.loc, "'%0' with type=%1 requires option 'factor'")
          << display << kind;
      return false;
    }
    int64_t dim = *getInt("dim");
    if (dim > static_cast<int64_t>(shape.size())) {
      // Vitis 2025.1 ignores an array_partition whose dim is deeper than the
      // array, with a warning ([HLS 214-253], kernel w8b_pragma_k_dim_range:
      // `dim=3` on `int A_local[8][8]` keeps one bank); the pragma is
      // dropped here too. Unconfirmed for array_reshape, which stays an
      // error.
      if (name == "array_partition") {
        S.warning(locOf("dim"),
                  "dim=%0 is out of range in '%1': '%2' has %3 dimension(s); "
                  "the pragma is ignored, as Vitis ignores it")
            << dim << display << var.path
            << static_cast<int64_t>(shape.size());
        return false;
      }
      S.error(locOf("dim"),
              "dim=%0 is out of range in '%1': '%2' has %3 dimension(s)")
          << dim << display << var.path << static_cast<int64_t>(shape.size());
      return false;
    }
    if (factor && dim > 0 && shape[dim - 1] > 0 && *factor > shape[dim - 1])
      S.warning(locOf("factor"),
                "factor=%0 exceeds the size %1 of dimension %2 of '%3'")
          << *factor << shape[dim - 1] << dim << var.path;
    return true;
  }
  if (name == "loop_tripcount") {
    // Without `max` Vitis 2025.1 accepts the pragma and ignores it (kernel
    // w9r22i_c5_tripcount_nomax: no message, trip count '?' in the loop
    // table; Vitis_Libraries data_compression huffman_treegen.hpp): it is
    // dropped with a warning.
    if (!getInt("max")) {
      S.warning(P.loc, "'%0' without 'max' is ignored, as Vitis ignores it")
          << display;
      return false;
    }
    int64_t mn = *getInt("min"), mx = *getInt("max");
    if (mn > mx) {
      S.error(locOf("min"), "min=%0 exceeds max=%1 in '%2'")
          << mn << mx << display;
      return false;
    }
    if (auto avg = getInt("avg")) {
      if (*avg < mn || *avg > mx) {
        S.error(locOf("avg"), "avg=%0 is not within [min, max] = [%1, %2] "
                              "in '%3'")
            << *avg << mn << mx << display;
        return false;
      }
    } else {
      fields.push_back(named("avg", b.getI32IntegerAttr((mn + mx) / 2)));
    }
    return true;
  }
  if (name == "latency") {
    auto mn = getInt("min"), mx = getInt("max");
    if (mn && mx && *mn > *mx) {
      S.error(locOf("min"), "min=%0 exceeds max=%1 in '%2'")
          << *mn << *mx << display;
      return false;
    }
    return true;
  }
  if (name == "performance") {
    if (!get("target_ti") && !get("target_tl")) {
      S.error(P.loc, "'%0' requires 'target_ti' or 'target_tl'") << display;
      return false;
    }
    return true;
  }
  if (name == "dependence") {
    auto dep = llvm::cast<mlir::BoolAttr>(get("dependent"));
    if (get("distance") && !dep.getValue())
      S.warning(locOf("distance"),
                "'distance' has no effect with dependent=false in '%0'")
          << display;
    return true;
  }
  if (name == "alias") {
    auto ports = llvm::cast<mlir::ArrayAttr>(get("ports"));
    if (ports.size() < 2) {
      S.error(locOf("ports"), "'%0' requires at least two ports") << display;
      return false;
    }
    for (mlir::Attribute a : ports) {
      StringRef port = llvm::cast<mlir::StringAttr>(a).getValue();
      if (llvm::none_of(FD->parameters(), [&](const ParmVarDecl *PV) {
            return identName(PV) == port;
          })) {
        S.error(locOf("ports"), "'%0' in '%1' is not a function parameter")
            << port << display;
        return false;
      }
    }
    bool hasDist = get("distance") != nullptr, hasOff = get("offset") != nullptr;
    if (hasDist == hasOff) {
      S.error(P.loc, "'%0' requires exactly one of 'distance' or 'offset'")
          << display;
      return false;
    }
    if (hasOff &&
        llvm::cast<mlir::ArrayAttr>(get("offset")).size() != ports.size()) {
      S.error(locOf("offset"),
              "'%0' needs one offset per port (%1 ports, %2 offsets)")
          << display << static_cast<int64_t>(ports.size())
          << static_cast<int64_t>(
                 llvm::cast<mlir::ArrayAttr>(get("offset")).size());
      return false;
    }
    return true;
  }
  if (name == "interface") {
    StringRef mode = llvm::cast<mlir::StringAttr>(get("mode")).getValue();
    bool isCtrl = mode.starts_with("ap_ctrl_");
    if (isCtrl && var.kind != Resolved::Return) {
      S.error(locOf("port"), "mode=%0 in '%1' requires port=return")
          << mode << display;
      return false;
    }
    if (var.kind == Resolved::Return && !isCtrl && mode != "s_axilite") {
      S.error(locOf("port"),
              "port=return in '%0' requires an ap_ctrl_* or s_axilite mode")
          << display;
      return false;
    }
    return true;
  }
  return true;
}

TargetBucket &FunctionSema::bucket(TargetKey key, std::string what) {
  auto [it, inserted] = buckets.try_emplace(key);
  if (inserted) {
    it->second.key = key;
    it->second.what = std::move(what);
    bucketOrder.push_back(key);
  }
  return it->second;
}

void FunctionSema::place(const PragmaRecord &P, TargetKey key,
                         std::string what, std::string attrName,
                         mlir::Attribute value, std::string conflict,
                         std::string variable, int64_t dim) {
  TargetBucket &B = bucket(key, what);
  for (const Placed &prev : B.placed) {
    bool overlaps = prev.key == conflict;
    // Partitioning the same array: dim=0 (all) overlaps every dimension.
    if (!overlaps && dim >= 0 && prev.dim >= 0 && prev.variable == variable &&
        prev.key.substr(0, prev.key.find('|')) ==
            conflict.substr(0, conflict.find('|')))
      overlaps = dim == 0 || prev.dim == 0 || dim == prev.dim;
    if (!overlaps)
      continue;
    if (prev.attrName == attrName && prev.value == value) {
      S.warning(P.loc, "duplicate '%0' on the same %1 is ignored")
          << pragmaDisplayName(P) << B.what;
      S.note(prev.pragma->loc, "previous pragma is here");
      S.addOutcome(P.report, "ignored: duplicate of the pragma at " +
                                 prev.pragma->locString);
    } else {
      S.error(P.loc, "'%0' conflicts with an earlier pragma on the same %1")
          << pragmaDisplayName(P) << B.what;
      S.note(prev.pragma->loc, "previous pragma is here");
      S.addOutcome(P.report, "rejected: conflicts with the pragma at " +
                                 prev.pragma->locString);
    }
    return;
  }
  B.placed.push_back({&P, conflict, attrName, value, variable, dim});
  if (!llvm::is_contained(B.reports, P.report)) // dim=0 split per dimension
    B.reports.push_back(P.report);
  placedReports.insert(P.report);
  if (!B.firstPragma)
    B.firstPragma = &P;
  if (llvm::isa<mlir::UnitAttr>(value)) {
    if (!llvm::is_contained(B.units, attrName))
      B.units.push_back(attrName);
  } else {
    B.arrays[attrName].push_back(value);
  }
}

mlir::DictionaryAttr FunctionSema::withDim(mlir::DictionaryAttr entry,
                                           int64_t dim) {
  llvm::SmallVector<mlir::NamedAttribute> fields(entry.begin(), entry.end());
  for (mlir::NamedAttribute &f : fields)
    if (f.getName() == "dim")
      f = named("dim", b.getI32IntegerAttr(dim));
  return b.getDictionaryAttr(fields);
}

namespace {
/// How Vitis combines two array_partition pragmas on the same array and
/// dimension (VITIS_NOTES N17 (c) and the W10 kernels
/// ~/hls-est-work/exp-src/w10conf/w10k_*.cpp, td=off, Vitis 2025.1): the same
/// type keeps the larger factor whatever the order (cyclic 2 / 4 and 4 / 2,
/// block 2 / 4: factor 4); `complete` wins over cyclic or block in either
/// order; cyclic against block is an error ([HLS 214-290] "Multiple
/// conflicting array_partition on dimension N"), even with equal factors.
enum class PartitionMerge { KeepEarlier, TakeLater, Conflict };

PartitionMerge mergePartition(mlir::DictionaryAttr earlier,
                              mlir::DictionaryAttr later) {
  auto kindOf = [](mlir::DictionaryAttr e) {
    auto k = e.getAs<mlir::StringAttr>("kind");
    return k ? k.getValue() : StringRef();
  };
  auto factorOf = [](mlir::DictionaryAttr e) -> int64_t {
    auto f = e.getAs<mlir::IntegerAttr>("factor");
    return f ? f.getInt() : 0;
  };
  StringRef k0 = kindOf(earlier), k1 = kindOf(later);
  if (k0 == k1) {
    if (k0 == "complete")
      return PartitionMerge::KeepEarlier;
    return factorOf(later) > factorOf(earlier) ? PartitionMerge::TakeLater
                                               : PartitionMerge::KeepEarlier;
  }
  if (k0 == "complete")
    return PartitionMerge::KeepEarlier;
  if (k1 == "complete")
    return PartitionMerge::TakeLater;
  return PartitionMerge::Conflict;
}

/// "cyclic factor=4" / "complete", for diagnostics.
std::string describePartition(mlir::DictionaryAttr e) {
  std::string s;
  if (auto k = e.getAs<mlir::StringAttr>("kind"))
    s = k.getValue().str();
  if (auto f = e.getAs<mlir::IntegerAttr>("factor"))
    s += " factor=" + std::to_string(f.getInt());
  return s;
}
} // namespace

/// array_partition (not `off`): a pragma on a dimension that an earlier
/// pragma on the same array already partitions is merged the Vitis way
/// (mergePartition) instead of being rejected; the pragma that loses gets a
/// remark. `dim=0` stands for every dimension: when it meets a pragma on one
/// dimension it is split into one entry per dimension first (Vitis kernel
/// w10k_d0c2_d1c4: `cyclic factor=2 dim=0` then `cyclic factor=4 dim=1` on
/// int[64][32] gives factor 4 on dimension 1 and factor 2 on dimension 2).
/// Everything else (a duplicate, cyclic against block, `off`) goes through
/// place() as before.
void FunctionSema::placePartition(const PragmaRecord &P, TargetKey key,
                                  std::string what, std::string attrName,
                                  mlir::DictionaryAttr value,
                                  std::string conflict, std::string variable,
                                  int64_t dim, int64_t rank) {
  TargetBucket &B = bucket(key, what);
  std::string keyBase = conflict.substr(0, conflict.rfind("|dim="));
  auto keyFor = [&](int64_t d) { return keyBase + "|dim=" + std::to_string(d); };
  auto sameArray = [&](const Placed &p) {
    return p.attrName == attrName && p.variable == variable && p.dim >= 0 &&
           p.key.substr(0, p.key.rfind("|dim=")) == keyBase;
  };
  auto attrsOf = [&]() -> llvm::SmallVector<mlir::Attribute, 2> & {
    return B.arrays[attrName];
  };

  if (rank > 1) {
    // A dim=0 pragma against per-dimension entries: one entry per dimension.
    if (dim == 0 && llvm::any_of(B.placed, [&](const Placed &p) {
          return sameArray(p) && p.dim > 0;
        })) {
      for (int64_t d = 1; d <= rank; ++d)
        placePartition(P, key, what, attrName, withDim(value, d), keyFor(d),
                       variable, d, rank);
      return;
    }
    // A per-dimension pragma against an earlier dim=0 entry: split that one.
    if (dim > 0) {
      for (size_t i = 0; i < B.placed.size(); ++i) {
        Placed prev = B.placed[i];
        if (!sameArray(prev) || prev.dim != 0)
          continue;
        auto prevValue = llvm::cast<mlir::DictionaryAttr>(prev.value);
        llvm::SmallVector<Placed> split;
        llvm::SmallVector<mlir::Attribute> splitValues;
        for (int64_t d = 1; d <= rank; ++d) {
          Placed p = prev;
          p.value = withDim(prevValue, d);
          p.key = keyFor(d);
          p.dim = d;
          split.push_back(p);
          splitValues.push_back(p.value);
        }
        B.placed.erase(B.placed.begin() + i);
        B.placed.insert(B.placed.begin() + i, split.begin(), split.end());
        auto &vals = attrsOf();
        auto it = llvm::find(vals, prev.value);
        if (it != vals.end()) {
          size_t at = it - vals.begin();
          vals.erase(it);
          vals.insert(vals.begin() + at, splitValues.begin(),
                      splitValues.end());
        }
        break;
      }
    }
  }

  // A one-dimensional array: dim=0 and dim=1 name the same dimension.
  auto sameDim = [&](int64_t a, int64_t c) {
    return a == c || (rank == 1 && (a == 0 || a == 1) && (c == 0 || c == 1));
  };
  for (Placed &prev : B.placed) {
    if (!sameArray(prev) || !sameDim(prev.dim, dim) || prev.value == value)
      continue;
    auto prevValue = llvm::cast<mlir::DictionaryAttr>(prev.value);
    PartitionMerge m = mergePartition(prevValue, value);
    if (m == PartitionMerge::Conflict)
      break; // place() reports it
    std::string dimText =
        dim == 0 ? "every dimension" : "dimension " + std::to_string(dim);
    if (m == PartitionMerge::KeepEarlier) {
      S.remark(P.loc, "'%0' (%1) on %2 of %3 is dropped: the earlier pragma "
                      "(%4) partitions it at least as finely, which Vitis "
                      "keeps")
          << pragmaDisplayName(P) << describePartition(value) << dimText
          << B.what << describePartition(prevValue);
      S.note(prev.pragma->loc, "earlier pragma is here");
      S.addOutcome(P.report, "merged: dropped in favour of the pragma at " +
                                 prev.pragma->locString + " (" + dimText +
                                 ")");
      placedReports.insert(P.report);
      return;
    }
    const PragmaRecord *dropped = prev.pragma;
    S.remark(dropped->loc, "'%0' (%1) on %2 of %3 is dropped: the later pragma "
                           "(%4) partitions it more finely, which Vitis keeps")
        << pragmaDisplayName(*dropped) << describePartition(prevValue)
        << dimText << B.what << describePartition(value);
    S.note(P.loc, "later pragma is here");
    S.addOutcome(dropped->report, "merged: dropped in favour of the pragma at " +
                                      P.locString + " (" + dimText + ")");
    auto &vals = attrsOf();
    auto it = llvm::find(vals, prev.value);
    if (it != vals.end())
      *it = value;
    prev.pragma = &P;
    prev.value = value;
    if (!llvm::is_contained(B.reports, P.report))
      B.reports.push_back(P.report);
    placedReports.insert(P.report);
    // The dropped pragma's report keeps its "applied" outcome only if it
    // still provides another dimension.
    if (llvm::none_of(B.placed,
                      [&](const Placed &p) { return p.pragma == dropped; }))
      llvm::erase_value(B.reports, dropped->report);
    return;
  }
  place(P, key, what, attrName, value, conflict, variable, dim);
}

void FunctionSema::process(PragmaRecord &P) {
  std::string display = pragmaDisplayName(P);
  const LoopInfo *loop = innermostLoop(P.loc);
  TargetKey fnKey{TargetKind::Function, FD, 0};
  auto loopKey = [&] { return TargetKey{TargetKind::Loop, loop->stmt, 0}; };
  auto loopWhat = [&]() -> std::string {
    auto it = loop ? loopLabels.find(loop->stmt) : loopLabels.end();
    return it == loopLabels.end() ? "loop" : "loop '" + it->second + "'";
  };

  if (P.isUnknown()) {
    mlir::Attribute text = b.getStringAttr(P.text);
    std::string key = "unknown|" + P.text; // only exact repeats collide
    if (loop)
      place(P, loopKey(), loopWhat(), "hls.unknown_pragmas", text, key, "",
            -1);
    else
      place(P, fnKey, "function", "hls.unknown_pragmas", text, key, "", -1);
    return;
  }

  const PragmaSpec &spec = *P.spec;
  bool isOff = P.isOff();
  std::string baseName = "hls." + spec.name.str();
  std::string attrName = isOff ? baseName + "_off" : baseName;

  // Target.
  Resolved var;
  TargetKey key = fnKey;
  std::string what = "function";
  switch (spec.placement) {
  case Placement::Loop:
    if (!loop) {
      // Vitis 2025.1 ignores an `unroll` outside a loop body with a warning
      // ([HLS 207-5589] "can only be applied inside loop body", kernel
      // w8b_pragma_k_unroll_outside: the loop that follows is pipelined, not
      // unrolled); the pragma is dropped here too. Unconfirmed for the other
      // loop pragmas, which stay errors.
      if (spec.name == "unroll") {
        S.warning(P.loc, "'%0' can only be applied inside a loop body; it is "
                         "ignored, as Vitis ignores it")
            << display;
        return;
      }
      S.error(P.loc, "'%0' must be placed inside a loop body") << display;
      if (loopFollows(P.loc))
        S.note(P.loc, "move it into the body of the loop that follows");
      return;
    }
    key = loopKey();
    what = loopWhat();
    break;
  case Placement::LoopOrFunction:
    // Vitis ignores a dataflow pragma anywhere else, e.g. in a nested `{ }`
    // block or a `while` body ([HLS 207-5576] "Only 'for' loop or function
    // body support the dataflow pragma"; w10rev rv_df_nested3,
    // w10f_df_forblock, w10f_df_while), and inlines the calls by the size
    // rule.
    if (spec.name == "dataflow" && !inFunctionOrForBody(P.loc, loop)) {
      S.warning(P.loc, "'%0' is ignored: only a 'for' loop body or a function "
                       "body supports the dataflow pragma (as Vitis, HLS "
                       "207-5576)")
          << display;
      S.addOutcome(P.report, "ignored: not directly in a 'for' loop body or "
                             "a function body");
      return;
    }
    if (loop) {
      key = loopKey();
      what = loopWhat();
    } else if (spec.name == "pipeline" && loopFollows(P.loc)) {
      // Pipelining a function fully unrolls its loops: when the pragma sits
      // right before a loop that is rarely what was meant.
      S.warning(P.loc,
                "'%0' is outside the loop that follows it and applies to the "
                "enclosing function; place it inside the loop body to apply "
                "it to the loop")
          << display;
    }
    break;
  case Placement::Function:
    break;
  case Placement::Variable:
  case Placement::Port: {
    const ParsedOption *O =
        P.get(spec.placement == Placement::Port ? "port" : "variable");
    var = resolveVariable(P, *O, spec.placement == Placement::Port);
    if (var.kind == Resolved::None)
      return;
    what = var.kind == Resolved::Return ? "function"
                                        : "variable '" + var.path + "'";
    if (spec.name == "bind_op") {
      if (var.kind == Resolved::Global) {
        S.error(O->loc, "'%0' requires a local variable or parameter")
            << display;
        return;
      }
      key = {TargetKind::Operation, var.vd,
             static_cast<unsigned>(std::hash<std::string>()(
                 P.get("op")->valueText()))};
      break;
    }
    if (spec.placement == Placement::Port && var.kind != Resolved::Return &&
        var.kind != Resolved::Param) {
      // An interface on a local variable: Vitis 2025.1 reports [HLS 214-238]
      // ("Unable to apply INTERFACE pragma options ...") and ignores it, the
      // array stays a local memory (kernel w9r22i_c8_iface_nonparam;
      // Vitis_Libraries vision xf_otsuthreshold.hpp `hist`). A global stays
      // an error.
      if (spec.name == "interface" && var.kind == Resolved::Local) {
        S.warning(O->value.front().loc,
                  "'%0' in '%1' is not a function parameter; the pragma is "
                  "ignored, as Vitis ignores it")
            << var.path << display;
        return;
      }
      S.error(O->value.front().loc, "'%0' in '%1' is not a function parameter")
          << var.path << display;
      return;
    }
    if ((spec.name == "function_instantiate") && var.kind != Resolved::Param) {
      S.error(O->value.front().loc, "'%0' in '%1' is not a function parameter")
          << var.path << display;
      return;
    }
    if (spec.name == "reset" && var.kind != Resolved::Global) {
      S.error(O->value.front().loc,
              "'%0' requires a static or global variable, but '%1' is not")
          << display << var.path;
      return;
    }
    switch (var.kind) {
    case Resolved::Local:
      key = {TargetKind::Local, var.vd, 0};
      break;
    case Resolved::Param:
      key = {TargetKind::Param, FD, var.paramIndex};
      break;
    case Resolved::Global:
      key = {TargetKind::Global, var.vd->getCanonicalDecl(), 0};
      break;
    case Resolved::Return:
      key = fnKey;
      break;
    case Resolved::None:
      return;
    }
    break;
  }
  }

  // Loop pragmas may name a variable (dependence): it must resolve.
  if (spec.placement != Placement::Variable &&
      spec.placement != Placement::Port) {
    if (const ParsedOption *O = P.get("variable"))
      if (resolveVariable(P, *O, false).kind == Resolved::None)
        return;
  }

  // `off` pragmas skip validate(); options that contradict `off` outright
  // are errors rather than "ignored" warnings.
  if (isOff && spec.name == "inline" && P.get("recursive")) {
    S.error(P.loc, "'recursive' conflicts with 'off' in '%0'") << display;
    return;
  }

  int64_t dim = -1;
  std::optional<mlir::DictionaryAttr> entry = buildEntry(P, var, isOff, dim);
  if (!entry)
    return;

  mlir::Attribute value = *entry;
  // The legacy RESOURCE pragma on an array: its memory core is the
  // bind_storage it names (Vitis 2025.1: `RESOURCE core=RAM_2P_BRAM` and
  // `bind_storage type=ram_2p impl=bram` both give RAM_2P_BRAM_1R1W, kernels
  // w9r13p_rs_res / rs_bind, CIRCT VITIS_NOTES N98).
  std::string specName = spec.name.str();
  if (spec.name == "resource")
    if (std::optional<mlir::DictionaryAttr> bind =
            lowerResourceCore(*entry, b)) {
      value = *bind;
      attrName = "hls.bind_storage";
      specName = "bind_storage";
    }
  bool variableLevel = spec.placement == Placement::Variable ||
                       spec.placement == Placement::Port;
  if (isOff && !variableLevel)
    value = b.getUnitAttr();
  if (!(spec.name == "array_partition" || spec.name == "array_reshape"))
    dim = -1;

  // Conflict key.
  std::string conflict = specName;
  std::string variable = var.path;
  switch (spec.multiplicity) {
  case Multiplicity::Single:
    break;
  case Multiplicity::PerVariable:
    if (spec.name == "aggregate" || spec.name == "disaggregate")
      conflict = "aggregate";
    conflict += "|" + variable;
    if (spec.name == "bind_op")
      conflict += "|" + P.get("op")->valueText();
    // A port may carry several interfaces of different modes (m_axi data
    // plus an s_axilite offset register); only the same mode twice clashes.
    if (spec.name == "interface")
      conflict += "|" + StringRef(P.get("mode")->valueText()).lower();
    if (dim >= 0)
      conflict += "|dim=" + std::to_string(dim);
    break;
  case Multiplicity::Many: {
    std::string s;
    llvm::raw_string_ostream os(s);
    value.print(os);
    conflict += "|" + os.str();
    break;
  }
  }

  // Two interface pragmas of one mode on one port: the later one decides,
  // as in Vitis 2025.1 (kernels w8b_iface_k_two_maxi / k_two_maxi_bundle:
  // both synthesize without a message, and the port uses the later
  // pragma's bundle). The earlier entry is withdrawn before the later one
  // is placed.
  if (spec.name == "interface" && !isOff) {
    TargetBucket &B = bucket(key, what);
    for (auto it = B.placed.begin(); it != B.placed.end(); ++it) {
      if (it->key != conflict || it->value == value)
        continue;
      S.warning(P.loc, "'%0' replaces the earlier pragma on the same %1; the "
                       "later one decides, as in Vitis")
          << pragmaDisplayName(P) << B.what;
      S.note(it->pragma->loc, "earlier pragma is here");
      S.addOutcome(it->pragma->report,
                   "replaced by the pragma at " + P.locString);
      auto &vals = B.arrays[it->attrName];
      if (auto v = llvm::find(vals, it->value); v != vals.end())
        vals.erase(v);
      B.placed.erase(it);
      break;
    }
  }
  if (spec.name == "array_partition" && !isOff && dim >= 0 &&
      !HLSRejectPartitionConflicts)
    placePartition(P, key, what, attrName,
                   llvm::cast<mlir::DictionaryAttr>(value), conflict, variable,
                   dim, static_cast<int64_t>(partitionShape(var.type).size()));
  else
    place(P, key, what, attrName, value, conflict, variable, dim);
  // Vitis sizes functions for automatic inlining after the user's unroll
  // pragmas and the complete unroll a pipeline pragma implies (VITIS_NOTES
  // N18); HLSAutoInline.cc unrolls these loops first.
  if (placedReports.count(P.report)) {
    if (spec.name == "unroll" && key.kind == TargetKind::Loop) {
      int64_t factor = isOff ? -1 : 0;
      if (auto d = llvm::dyn_cast<mlir::DictionaryAttr>(value))
        if (auto f = d.getAs<mlir::IntegerAttr>("factor"))
          factor = f.getInt();
      recordUnrollLoop(SM, static_cast<const Stmt *>(key.ptr)->getBeginLoc(),
                       factor);
    }
    if (spec.name == "pipeline" && !isOff) {
      if (key.kind == TargetKind::Loop)
        recordPipelineLoop(SM,
                           static_cast<const Stmt *>(key.ptr)->getBeginLoc());
      else if (key.kind == TargetKind::Function)
        recordPipelineFunction(func.getName());
    }
  }
  if (key.kind == TargetKind::Operation) {
    TargetBucket &B = buckets[key];
    B.var = var.vd;
    B.opNames.clear();
    bindOpTarget(StringRef(P.get("op")->valueText()).lower(), B.opNames,
                 B.floatWidth);
  }
}

/// The trip count of `for (i = a; i < b; i++)`-shaped loops with constant
/// bounds (`<`, `<=`, `!=` with `++` / `+= 1`; `>`, `>=`, `!=` with `--` /
/// `-= 1`); nullopt for any other loop.
static std::optional<int64_t> constantTripCount(const Stmt *s,
                                                ASTContext &AC) {
  const auto *F = dyn_cast_or_null<ForStmt>(s);
  if (!F || !F->getInit() || !F->getCond() || !F->getInc())
    return std::nullopt;
  auto constant = [&](const Expr *e) -> std::optional<int64_t> {
    Expr::EvalResult r;
    if (!e || e->isValueDependent() || !e->EvaluateAsInt(r, AC))
      return std::nullopt;
    return r.Val.getInt().getSExtValue();
  };
  auto varOf = [](const Expr *e) -> const VarDecl * {
    if (const auto *d = dyn_cast_or_null<DeclRefExpr>(e ? e->IgnoreParenImpCasts()
                                                        : nullptr))
      return dyn_cast<VarDecl>(d->getDecl());
    return nullptr;
  };
  // Init: `int i = a` or `i = a`.
  const VarDecl *iv = nullptr;
  std::optional<int64_t> start;
  if (const auto *DS = dyn_cast<DeclStmt>(F->getInit())) {
    if (!DS->isSingleDecl())
      return std::nullopt;
    iv = dyn_cast<VarDecl>(DS->getSingleDecl());
    start = iv ? constant(iv->getInit()) : std::nullopt;
  } else if (const auto *BO = dyn_cast<BinaryOperator>(F->getInit())) {
    if (BO->getOpcode() != BO_Assign)
      return std::nullopt;
    iv = varOf(BO->getLHS());
    start = constant(BO->getRHS());
  }
  if (!iv || !start || !iv->getType()->isIntegerType())
    return std::nullopt;
  // Step: +1 or -1.
  int step = 0;
  const Expr *inc = F->getInc()->IgnoreParenImpCasts();
  if (const auto *U = dyn_cast<UnaryOperator>(inc)) {
    if (varOf(U->getSubExpr()) != iv)
      return std::nullopt;
    step = U->isIncrementOp() ? 1 : U->isDecrementOp() ? -1 : 0;
  } else if (const auto *C = dyn_cast<CompoundAssignOperator>(inc)) {
    std::optional<int64_t> by = constant(C->getRHS());
    if (varOf(C->getLHS()) != iv || !by || *by != 1)
      return std::nullopt;
    step = C->getOpcode() == BO_AddAssign   ? 1
           : C->getOpcode() == BO_SubAssign ? -1
                                            : 0;
  }
  if (step == 0)
    return std::nullopt;
  // Condition: `i op b`.
  const auto *cmp = dyn_cast<BinaryOperator>(F->getCond()->IgnoreParenImpCasts());
  if (!cmp || varOf(cmp->getLHS()) != iv)
    return std::nullopt;
  std::optional<int64_t> end = constant(cmp->getRHS());
  if (!end)
    return std::nullopt;
  int64_t a = *start, b = *end;
  switch (cmp->getOpcode()) {
  case BO_LT:
    return step > 0 ? std::max<int64_t>(0, b - a) : std::optional<int64_t>();
  case BO_LE:
    return step > 0 ? std::max<int64_t>(0, b - a + 1)
                    : std::optional<int64_t>();
  case BO_GT:
    return step < 0 ? std::max<int64_t>(0, a - b) : std::optional<int64_t>();
  case BO_GE:
    return step < 0 ? std::max<int64_t>(0, a - b + 1)
                    : std::optional<int64_t>();
  case BO_NE:
    if ((step > 0 && b >= a) || (step < 0 && a >= b))
      return step > 0 ? b - a : a - b;
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

/// A `pipeline` pragma on a loop whose trip count is 1 (VITIS_NOTES N40,
/// Vitis 2025.1 kernels w9p5_t1_flat / w9p5_t1_noflat, corpus stream_hls
/// MultiHeadSelfAttention1 opt5 node13): Vitis completely unrolls the loops
/// inside it, as any pipeline pragma does (`[HLS 214-291]`), then removes
/// the one-iteration loop together with its pipeline (and loop_flatten)
/// pragma, and its automatic pipelining decides on the enclosing nest
/// (`[HLS 214-376]`). The MLIR canonicalizer folds the one-iteration loop
/// before the pragma could be materialized, so the pragma's effect is moved
/// here: every loop nested in it gets a complete `unroll` (a partial unroll
/// pragma of its own is overridden to complete, `[HLS 214-291]`; an `unroll
/// off` is left alone), and the pipeline pragma is withdrawn.
void FunctionSema::applyTripOnePipelines() {
  if (!HLSTripOnePipelineUnroll)
    return;
  static constexpr llvm::StringLiteral kPipeline = "hls.pipeline";
  static constexpr llvm::StringLiteral kUnroll = "hls.unroll";
  for (const LoopInfo &L : loops) {
    auto it = buckets.find(TargetKey{TargetKind::Loop, L.stmt, 0});
    if (it == buckets.end() || !it->second.arrays.count(kPipeline.str()))
      continue;
    std::optional<int64_t> trips = constantTripCount(L.stmt, AC);
    if (!trips || *trips != 1)
      continue;
    TargetBucket &B = it->second;
    const PragmaRecord *P = nullptr;
    for (const Placed &p : B.placed)
      if (p.attrName == kPipeline)
        P = p.pragma;
    if (!P)
      continue;
    B.arrays.erase(kPipeline.str());
    llvm::erase_if(B.placed,
                   [&](const Placed &p) { return p.attrName == kPipeline; });
    if (llvm::none_of(B.placed,
                      [&](const Placed &p) { return p.pragma == P; }))
      llvm::erase_value(B.reports, P->report);
    if (B.firstPragma == P) // the loop's other pragmas (loop_flatten) remain
      B.firstPragma = B.placed.empty() ? nullptr : B.placed.front().pragma;
    S.warning(P->loc, "'%0' is on a loop with a trip count of 1: the loops "
                      "inside it are completely unrolled and the loop is "
                      "removed with its pragmas, as Vitis does")
        << pragmaDisplayName(*P);
    S.addOutcome(P->report, "applied as a complete unroll of the loops "
                            "inside the trip-count-1 loop, which is removed "
                            "(as Vitis)");
    mlir::Attribute complete = b.getDictionaryAttr(
        {named("factor", b.getI32IntegerAttr(0)),
         named("skip_exit_check", b.getBoolAttr(false))});
    for (const LoopInfo &I : loops) {
      if (I.stmt == L.stmt || !inRegion(L.region, I.stmt->getBeginLoc()))
        continue;
      auto label = loopLabels.find(I.stmt);
      TargetBucket &IB =
          bucket(TargetKey{TargetKind::Loop, I.stmt, 0},
                 label == loopLabels.end() ? "loop"
                                           : "loop '" + label->second + "'");
      if (llvm::is_contained(IB.units, kUnroll.str() + "_off"))
        continue;
      // A partial `unroll factor=N` of its own is overridden: Vitis logs
      // `[HLS 214-291] ... marked as complete unroll implied by the pipeline
      // pragma` and unrolls it completely (kernel w9rv_t1_part, as for any
      // pipeline pragma, VITIS_NOTES N43).
      auto own = IB.arrays.find(kUnroll.str());
      if (own != IB.arrays.end()) {
        own->second.clear();
        own->second.push_back(complete);
        continue;
      }
      IB.arrays[kUnroll.str()].push_back(complete);
      if (!IB.firstPragma)
        IB.firstPragma = P;
    }
  }
}

void FunctionSema::flush() {
  std::string funcName = func.getName().str();
  for (const TargetKey &key : bucketOrder) {
    TargetBucket &B = buckets[key];
    llvm::SmallVector<std::pair<std::string, mlir::Attribute>, 2> attrs;
    for (auto &[name, entries] : B.arrays)
      attrs.push_back({name, b.getArrayAttr(entries)});
    for (const std::string &u : B.units)
      attrs.push_back({u, b.getUnitAttr()});
    if (attrs.empty())
      continue;

    auto merge = [&](mlir::Attribute existing, mlir::Attribute add) {
      auto oldArr = llvm::dyn_cast_or_null<mlir::ArrayAttr>(existing);
      auto newArr = llvm::dyn_cast<mlir::ArrayAttr>(add);
      if (!oldArr || !newArr)
        return add;
      llvm::SmallVector<mlir::Attribute> all(oldArr.begin(), oldArr.end());
      all.append(newArr.begin(), newArr.end());
      return static_cast<mlir::Attribute>(b.getArrayAttr(all));
    };

    const PragmaRecord &first = *B.firstPragma;
    switch (key.kind) {
    case TargetKind::Function:
      for (auto &[name, value] : attrs)
        func->setAttr(name, merge(func->getAttr(name), value));
      for (int r : B.reports)
        S.addOutcome(r, "func.func @" + funcName);
      break;
    case TargetKind::Param:
      for (auto &[name, value] : attrs)
        func.setArgAttr(key.index, name,
                        merge(func.getArgAttr(key.index, name), value));
      for (int r : B.reports)
        S.addOutcome(r, "argument #" + std::to_string(key.index) + " (" +
                            B.what + ") of @" + funcName);
      break;
    case TargetKind::Global: {
      GlobalTarget &G =
          S.globals[static_cast<const VarDecl *>(key.ptr)];
      G.name = identName(static_cast<const VarDecl *>(key.ptr)).str();
      G.pragmaName = pragmaDisplayName(first);
      G.loc = first.loc;
      G.reports.append(B.reports.begin(), B.reports.end());
      for (auto &attr : attrs) {
        bool found = false;
        for (auto &existing : G.attrs) {
          if (existing.first == attr.first) {
            existing.second = merge(existing.second, attr.second);
            found = true;
          }
        }
        if (!found)
          G.attrs.push_back(attr);
      }
      break;
    }
    case TargetKind::Loop:
    case TargetKind::Local:
    case TargetKind::Operation: {
      TagClass cls = key.kind == TargetKind::Loop    ? TagClass::Loop
                     : key.kind == TargetKind::Local ? TagClass::Storage
                                                     : TagClass::Operation;
      TagRecord &T = S.newTag(cls);
      T.attrs = attrs;
      if (S.checkOnlyMode) {
        // No code: report the AST-level target now.
        std::string what =
            key.kind == TargetKind::Operation
                ? "operation assigned to variable '" +
                      identName(B.var).str() + "'"
                : B.what;
        for (int r : B.reports)
          S.addOutcome(r, what + " in " + funcName + " (check-only)");
        T.tagged = T.materialized = true;
        continue;
      }
      T.pragmaName = pragmaDisplayName(first);
      T.what = B.what;
      T.locString = first.locString;
      T.funcName = funcName;
      T.reports = B.reports;
      if (key.kind == TargetKind::Loop) {
        S.loopTags[static_cast<const Stmt *>(key.ptr)] = T.id;
      } else if (key.kind == TargetKind::Local) {
        S.storageTags[static_cast<const VarDecl *>(key.ptr)] = T.id;
      } else {
        T.opNames.assign(B.opNames.begin(), B.opNames.end());
        T.floatWidth = B.floatWidth;
        T.what = "variable '" + identName(B.var).str() + "'";
        S.assignTags[B.var].push_back(T.id);
      }
      break;
    }
    }
  }
}

void FunctionSema::run() {
  const Stmt *body = FD->getBody();
  if (!body)
    return;
  index(body, body->getSourceRange(), 0);
  for (PragmaRecord &P : S.pragmas) {
    if (!P.valid || !belongsHere(P.loc))
      continue;
    P.associated = true;
    process(P);
    if (!placedReports.count(P.report) && !S.hasOutcome(P.report))
      S.addOutcome(P.report, "rejected: see the diagnostic above");
  }
  applyTripOnePipelines();
  flush();
}

//===----------------------------------------------------------------------===//
// Translation-unit checks
//===----------------------------------------------------------------------===//

class BodyCollector : public RecursiveASTVisitor<BodyCollector> {
public:
  std::vector<SourceRange> bodies;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool VisitFunctionDecl(FunctionDecl *FD) {
    if (FD->doesThisDeclarationHaveABody() && FD->getBody())
      bodies.push_back(FD->getBody()->getSourceRange());
    return true;
  }
};

} // namespace

void cgeist::hls::enterFunction(const FunctionDecl *FD,
                                mlir::func::FuncOp func, unsigned argOffset) {
  State &S = State::get();
  if (!isEnabled() || !S.PP || S.analyzedFunctions.count(FD))
    return;
  S.analyzedFunctions[FD] = true;
  FunctionSema(FD, func, argOffset).run();
}

namespace {
class FunctionCollector : public RecursiveASTVisitor<FunctionCollector> {
public:
  std::vector<const FunctionDecl *> functions;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }
  bool VisitFunctionDecl(FunctionDecl *FD) {
    if (FD->doesThisDeclarationHaveABody() && FD->getBody() &&
        !FD->isDependentContext())
      functions.push_back(FD);
    return true;
  }
};
} // namespace

void cgeist::hls::checkAllFunctions(ASTContext &astCtx) {
  State &S = State::get();
  if (!isEnabled() || !S.PP || !S.mlirCtx)
    return;
  S.checkOnlyMode = true;
  FunctionCollector C;
  C.TraverseDecl(astCtx.getTranslationUnitDecl());
  mlir::OpBuilder b(S.mlirCtx);
  for (const FunctionDecl *FD : C.functions) {
    const auto *MD = dyn_cast<CXXMethodDecl>(FD);
    unsigned offset = MD && MD->isInstance() ? 1 : 0;
    llvm::SmallVector<mlir::Type> args(FD->getNumParams() + offset,
                                       b.getI32Type());
    // A detached stand-in: it only receives the function-level attributes.
    // Named as in diagnostics (qualified, with template arguments).
    std::string name;
    llvm::raw_string_ostream os(name);
    FD->getNameForDiagnostic(os, astCtx.getPrintingPolicy(),
                             /*Qualified=*/true);
    auto func = mlir::func::FuncOp::create(b.getUnknownLoc(), os.str(),
                                           b.getFunctionType(args, {}));
    enterFunction(FD, func, offset);
    func->destroy();
  }
}

void cgeist::hls::checkTranslationUnit(ASTContext &astCtx) {
  State &S = State::get();
  if (!isEnabled() || S.checkedTU)
    return;
  S.checkedTU = true;
  BodyCollector C;
  C.TraverseDecl(astCtx.getTranslationUnitDecl());
  SourceManager &SM = astCtx.getSourceManager();
  for (PragmaRecord &P : S.pragmas) {
    if (!P.valid)
      continue;
    bool inside = llvm::any_of(C.bodies, [&](SourceRange r) {
      return SM.isBeforeInTranslationUnit(r.getBegin(), P.loc) &&
             SM.isBeforeInTranslationUnit(P.loc, r.getEnd());
    });
    if (!inside) {
      S.error(P.loc, "'%0' must be placed inside a function body")
          << pragmaDisplayName(P);
      S.addOutcome(P.report, "rejected: outside a function body");
    }
  }
}

//===- HLSPragma.cc - Shared state and codegen hooks for #pragma HLS ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HLSAutoInline.h"
#include "polygeist/Dialect.h"
#include "HLSPragmaInternal.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"

#include <optional>

using namespace clang;
using namespace cgeist::hls;

extern llvm::cl::opt<bool> HLSAnnotate;

static llvm::cl::opt<bool> HLSPragmaCheckOnly(
    "hls-pragma-check-only", llvm::cl::init(false),
    llvm::cl::desc("With --hls-annotate: parse, place and validate the HLS "
                   "pragmas of every function body without generating code, "
                   "and report their targets (implies --hls-pragma-report)"));

static llvm::cl::opt<bool> HLSPragmaReport(
    "hls-pragma-report", llvm::cl::init(false),
    llvm::cl::desc("With --hls-annotate: print, for every HLS pragma, the op(s) "
                   "it was attached to or why it was not applied"));

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

State &State::get() {
  static State state;
  return state;
}

void State::resetTranslationUnit() {
  PP = nullptr;
  pragmas.clear();
  analyzedFunctions.clear();
  loopTags.clear();
  storageTags.clear();
  assignTags.clear();
  globals.clear();
  globalOps.clear();
  checkedTU = false;
}

TagRecord &State::newTag(TagClass cls) {
  TagRecord &T = tags.emplace_back();
  T.id = tags.size() - 1;
  T.cls = cls;
  return T;
}

mlir::Location State::tag(mlir::Location base, unsigned tagId) const {
  mlir::MLIRContext *ctx = base.getContext();
  mlir::Builder b(ctx);
  auto meta = b.getDictionaryAttr(
      b.getNamedAttr(kTagKey, b.getI64IntegerAttr(tagId)));
  return mlir::FusedLoc::get({base}, meta, ctx);
}

// Custom diagnostics take their format string as the ID, so every distinct
// message text gets its own ID; arguments are streamed by the caller.
DiagnosticBuilder State::error(SourceLocation loc, llvm::StringRef fmt) {
  DiagnosticsEngine &D = PP->getDiagnostics();
  return D.Report(loc, D.getDiagnosticIDs()->getCustomDiagID(
                           DiagnosticIDs::Error, fmt));
}
DiagnosticBuilder State::warning(SourceLocation loc, llvm::StringRef fmt) {
  DiagnosticsEngine &D = PP->getDiagnostics();
  return D.Report(loc, D.getDiagnosticIDs()->getCustomDiagID(
                           DiagnosticIDs::Warning, fmt));
}
DiagnosticBuilder State::note(SourceLocation loc, llvm::StringRef fmt) {
  DiagnosticsEngine &D = PP->getDiagnostics();
  return D.Report(loc, D.getDiagnosticIDs()->getCustomDiagID(
                           DiagnosticIDs::Note, fmt));
}
DiagnosticBuilder State::remark(SourceLocation loc, llvm::StringRef fmt) {
  DiagnosticsEngine &D = PP->getDiagnostics();
  return D.Report(loc, D.getDiagnosticIDs()->getCustomDiagID(
                           DiagnosticIDs::Remark, fmt));
}

std::string State::locString(SourceLocation loc) const {
  if (!PP || loc.isInvalid())
    return "<unknown>";
  PresumedLoc PL = PP->getSourceManager().getPresumedLoc(
      PP->getSourceManager().getExpansionLoc(loc));
  if (PL.isInvalid())
    return "<unknown>";
  return (llvm::Twine(PL.getFilename()) + ":" + llvm::Twine(PL.getLine()) +
          ":" + llvm::Twine(PL.getColumn()))
      .str();
}

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

bool cgeist::hls::isEnabled() { return HLSAnnotate; }
bool cgeist::hls::checkOnly() { return HLSAnnotate && HLSPragmaCheckOnly; }
bool cgeist::hls::reportEnabled() {
  return HLSAnnotate && (HLSPragmaReport || HLSPragmaCheckOnly);
}

void cgeist::hls::beginTranslationUnit(Preprocessor &PP,
                                       mlir::MLIRContext *ctx) {
  State &S = State::get();
  S.resetTranslationUnit();
  S.PP = &PP;
  S.mlirCtx = ctx;
  registerPragmaHandler(PP);
}

mlir::Location cgeist::hls::loopLocation(const Stmt *loop,
                                         mlir::Location base) {
  State &S = State::get();
  auto it = S.loopTags.find(loop);
  if (it == S.loopTags.end())
    return base;
  S.tags[it->second].tagged = true;
  return S.tag(base, it->second);
}

void cgeist::hls::onLocalStorage(const VarDecl *VD, mlir::Operation *storage) {
  State &S = State::get();
  if (!VD || !storage)
    return;
  auto it = S.storageTags.find(VD);
  if (it == S.storageTags.end())
    return;
  S.tags[it->second].tagged = true;
  storage->setLoc(S.tag(storage->getLoc(), it->second));
}

/// The depth D of an `hls::stream<T, D>` object type (arrays of it
/// included), when D > 0.
static std::optional<int64_t> streamTemplateDepth(clang::QualType t) {
  const clang::Type *base = t->getBaseElementTypeUnsafe();
  const auto *rd = base ? base->getAsCXXRecordDecl() : nullptr;
  const auto *spec =
      llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(rd);
  if (!spec ||
      spec->getSpecializedTemplate()->getQualifiedNameAsString() !=
          "hls::stream")
    return std::nullopt;
  const clang::TemplateArgumentList &args = spec->getTemplateArgs();
  if (args.size() < 2 || args[1].getKind() != clang::TemplateArgument::Integral)
    return std::nullopt;
  int64_t d = args[1].getAsIntegral().getSExtValue();
  if (d <= 0)
    return std::nullopt;
  return d;
}

void cgeist::hls::onGlobal(const VarDecl *VD, mlir::Operation *global) {
  if (!isEnabled() || !VD || !global)
    return;
  State::get().globalOps[VD->getCanonicalDecl()] = global;
  // A namespace-scope `hls::stream<T, D>` is constructed by the
  // translation unit's initialiser, which cgeist does not emit, so no call
  // in the MLIR names its type; the depth is kept on the global
  // (`hls.stream_depth`), which hls-est's FIFO model reads when no
  // constructor call reaches the handle (CIRCT W9 round 6 review defect 5,
  // kernel w9r6rev n49b: `static hls::stream<W, 1024> gs;` used only
  // through `hls::stream<W>&` parameters is a 1024-deep FIFO, 2 BRAM).
  // (A static local's constructor runs in its function, where hls-est
  // sees it.)
  if (std::optional<int64_t> depth = VD->isStaticLocal()
                                         ? std::nullopt
                                         : streamTemplateDepth(VD->getType()))
    global->setAttr("hls.stream_depth",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(global->getContext(), 64),
                        *depth));
}

void cgeist::hls::onAssignedValue(const VarDecl *VD, mlir::Value value) {
  State &S = State::get();
  if (!VD || !value)
    return;
  auto it = S.assignTags.find(VD);
  if (it == S.assignTags.end())
    return;
  // Search the expression computed for this assignment: `x = a * b + c`
  // binds the multiplication of `op=mul`. The walk only follows pure
  // arithmetic, so it stops at loads of other variables and never reaches
  // operations of earlier statements.
  auto isArith = [](mlir::Operation *op) {
    llvm::StringRef d = op->getName().getDialectNamespace();
    return (d == "arith" || d == "math") && op->getNumResults() == 1;
  };
  for (unsigned id : it->second) {
    TagRecord &T = S.tags[id];
    auto matches = [&](mlir::Operation *op) {
      if (!llvm::is_contained(T.opNames, op->getName().getStringRef().str()))
        return false;
      if (!T.floatWidth)
        return true;
      auto ft = llvm::dyn_cast<mlir::FloatType>(
          mlir::getElementTypeOrSelf(op->getResult(0).getType()));
      return ft && ft.getWidth() == T.floatWidth;
    };
    llvm::SmallVector<mlir::Operation *, 8> work;
    llvm::SmallPtrSet<mlir::Operation *, 8> seen;
    if (mlir::Operation *root = value.getDefiningOp())
      work.push_back(root);
    while (!work.empty()) {
      mlir::Operation *op = work.pop_back_val();
      if (!seen.insert(op).second || !isArith(op) || seen.size() > 64)
        continue;
      if (matches(op)) {
        op->setLoc(S.tag(op->getLoc(), id));
        T.tagged = true;
        continue; // do not look inside a bound operation
      }
      for (mlir::Value operand : op->getOperands())
        if (mlir::Operation *def = operand.getDefiningOp())
          work.push_back(def);
    }
  }
}

void cgeist::hls::endTranslationUnit() {
  State &S = State::get();
  if (!isEnabled() || !S.PP)
    return;
  for (auto &[VD, G] : S.globals) {
    auto it = S.globalOps.find(VD);
    if (S.checkOnlyMode) {
      for (int r : G.reports)
        S.addOutcome(r, "global '" + G.name + "' (check-only)");
      continue;
    }
    if (it == S.globalOps.end()) {
      S.warning(G.loc, "global '%0' is not used by any emitted function; "
                       "'%1' has no effect")
          << G.name << G.pragmaName;
      for (int r : G.reports)
        S.addOutcome(r, "not applied: global '" + G.name +
                            "' is not used by any emitted function");
      continue;
    }
    mlir::Operation *op = it->second;
    for (int r : G.reports) {
      auto sym = op->getAttrOfType<mlir::StringAttr>("sym_name");
      S.addOutcome(r, op->getName().getStringRef().str() + " @" +
                          (sym ? sym.getValue().str() : G.name));
    }
    for (auto &[name, value] : G.attrs) {
      mlir::Attribute merged = value;
      auto oldArr = op->getAttrOfType<mlir::ArrayAttr>(name);
      auto newArr = llvm::dyn_cast<mlir::ArrayAttr>(value);
      if (oldArr && newArr) {
        llvm::SmallVector<mlir::Attribute> all(oldArr.begin(), oldArr.end());
        all.append(newArr.begin(), newArr.end());
        merged = mlir::ArrayAttr::get(op->getContext(), all);
      }
      op->setAttr(name, merged);
    }
  }
  // bind_op pragmas whose operation never showed up during codegen.
  for (auto &[VD, ids] : S.assignTags) {
    for (unsigned id : ids) {
      TagRecord &T = S.tags[id];
      if (T.tagged)
        continue;
      T.materialized = true; // already reported
      llvm::errs() << T.locString << ": warning: '" << T.pragmaName
                   << "': no matching operation is assigned to " << T.what
                   << "; the pragma has no effect\n";
      for (int r : T.reports)
        S.addOutcome(r, "not applied: no matching operation is assigned to " +
                            T.what);
    }
  }
  // Valid pragmas inside functions that were never emitted.
  for (PragmaRecord &P : S.pragmas)
    if (P.valid && !P.associated && !S.hasOutcome(P.report))
      S.addOutcome(P.report,
                   "not applied: the enclosing function is not emitted");
  S.PP = nullptr;
}

//===----------------------------------------------------------------------===//
// Inlining
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool> HLSIgnoreInlineOff(
    "hls-ignore-inline-off", llvm::cl::init(false),
    llvm::cl::desc("Let cgeist's inliner inline functions marked '#pragma HLS "
                   "inline off' too (the behaviour before the pragma was "
                   "honoured; for comparisons)"));

static llvm::cl::opt<bool> HLSKeepMisfitInlineOff(
    "hls-keep-misfit-inline-off", llvm::cl::init(false),
    llvm::cl::desc("Keep a call of a function marked '#pragma HLS inline off' "
                   "also when an array argument's static shape differs from "
                   "the parameter's (the argument is reinterpreted to the "
                   "parameter type), as Vitis keeps the module whatever the "
                   "argument's size (VITIS_NOTES N123 / OPEN_ISSUES O156 in "
                   "the hls-est tree; off: such a call is inlined)"));

bool cgeist::hls::keepMisfitInlineOff() { return HLSKeepMisfitInlineOff; }

static llvm::cl::opt<bool> HLSInlineCarryPartition(
    "hls-inline-carry-partition", llvm::cl::init(true),
    llvm::cl::desc("When a call is inlined, carry the `array_partition` "
                   "pragmas of the callee's array parameters to the array the "
                   "argument names (an alloca, a global, or the caller's own "
                   "parameter), as Vitis partitions the root array whether "
                   "or not it inlines the callee; when the parameter's first "
                   "dimension is smaller than the root's, applied one "
                   "dimension deeper in the root and dropped past its rank "
                   "(Vitis HLS 214-248 / 214-253)"));

/// The array a call argument names through casts: memref.cast and the
/// memref.reinterpret_cast that castCallerMemRefArg builds for an argument
/// whose static shape differs from the parameter's.
static mlir::Value partitionRootOf(mlir::Value v) {
  for (unsigned guard = 0; guard < 16; ++guard) {
    if (auto c = v.getDefiningOp<mlir::memref::CastOp>()) {
      v = c.getSource();
      continue;
    }
    if (auto r = v.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
      v = r.getSource();
      continue;
    }
    break;
  }
  return v;
}

/// The callee parameter `paramTy`'s partition entries `entries` reach the
/// array `arg` names: appended to its alloca, its memref.global, or (the
/// caller's own parameter) its argument attributes with the first-dimension
/// extent of the parameter that carried the pragma (`rows`), compared with
/// the root array once one is reached (VITIS_NOTES N123 in the hls-est tree:
/// kernels w9r16p_pm_ctl / pm_mis / pm_big / pm_two / pm_two2; DGNN_Booster
/// convLSTM / EvolveGCN). An entry whose `rows` is smaller than the root's
/// first dimension applies one dimension deeper in the root; what falls past
/// the root's rank is dropped with Vitis' warning 214-253.
static void carryPartition(mlir::Operation *call, mlir::Value arg,
                           mlir::MemRefType paramTy, mlir::ArrayAttr entries,
                           mlir::StringRef calleeName);

/// The entries of `entries` that name a member of the object (`member_path`,
/// set for a pragma on a data member inside a member function or for a
/// member path `s.buf`), appended unchanged to the object `arg` names: the
/// member's array is the same whatever the object parameter's shape (a
/// constructor's `this` is memref<?x struct>, the object memref<1 x
/// struct>). Returns the other entries.
static mlir::ArrayAttr carryMemberPartitions(mlir::Value arg,
                                             mlir::ArrayAttr entries) {
  mlir::MLIRContext *ctx = arg.getContext();
  llvm::SmallVector<mlir::Attribute> members, others;
  for (mlir::Attribute a : entries) {
    auto d = llvm::dyn_cast<mlir::DictionaryAttr>(a);
    if (d && d.get("member_path"))
      members.push_back(a);
    else
      others.push_back(a);
  }
  if (members.empty())
    return entries;
  auto append = [&](mlir::ArrayAttr old) {
    llvm::SmallVector<mlir::Attribute> all;
    if (old)
      all.append(old.begin(), old.end());
    for (mlir::Attribute a : members)
      if (!llvm::is_contained(all, a))
        all.push_back(a);
    return mlir::ArrayAttr::get(ctx, all);
  };
  mlir::Value root = partitionRootOf(arg);
  if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(root)) {
    auto fn = llvm::dyn_cast<mlir::FunctionOpInterface>(
        ba.getOwner()->getParentOp());
    if (fn && ba.getOwner() == &fn.getFunctionBody().front())
      fn.setArgAttr(ba.getArgNumber(), "hls.array_partition",
                    append(fn.getArgAttrOfType<mlir::ArrayAttr>(
                        ba.getArgNumber(), "hls.array_partition")));
  } else {
    mlir::Operation *holder = nullptr;
    if (auto a = root.getDefiningOp<mlir::memref::AllocaOp>())
      holder = a;
    else if (auto a = root.getDefiningOp<mlir::memref::AllocOp>())
      holder = a;
    else if (auto gg = root.getDefiningOp<mlir::memref::GetGlobalOp>())
      holder =
          mlir::SymbolTable::lookupNearestSymbolFrom(gg, gg.getNameAttr());
    if (holder)
      holder->setAttr("hls.array_partition",
                      append(holder->getAttrOfType<mlir::ArrayAttr>(
                          "hls.array_partition")));
  }
  return mlir::ArrayAttr::get(ctx, others);
}

static void carryPartition(mlir::Operation *call, mlir::Value arg,
                           mlir::MemRefType paramTy, mlir::ArrayAttr entries,
                           mlir::StringRef calleeName) {
  mlir::MLIRContext *ctx = call->getContext();
  entries = carryMemberPartitions(arg, entries);
  if (entries.empty())
    return;
  mlir::Value root = partitionRootOf(arg);
  auto rootTy = llvm::dyn_cast<mlir::MemRefType>(root.getType());
  if (!rootTy || !rootTy.hasStaticShape() || !paramTy.hasStaticShape() ||
      rootTy.getRank() != paramTy.getRank() || rootTy.getRank() == 0)
    return;
  // The entries with the rows of the parameter that first carried them.
  llvm::SmallVector<mlir::Attribute> withRows, kept;
  for (mlir::Attribute a : entries) {
    auto d = llvm::dyn_cast<mlir::DictionaryAttr>(a);
    if (!d)
      continue;
    int64_t rows = paramTy.getDimSize(0);
    if (auto r = d.getAs<mlir::IntegerAttr>("rows"))
      rows = r.getInt();
    llvm::SmallVector<mlir::NamedAttribute> fields;
    for (mlir::NamedAttribute f : d)
      if (f.getName() != "rows")
        fields.push_back(f);
    mlir::DictionaryAttr plain = mlir::DictionaryAttr::get(ctx, fields);
    fields.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(ctx, "rows"),
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), rows)));
    withRows.push_back(mlir::DictionaryAttr::get(ctx, fields));
    // A parameter in between does not decide (kernels pm_two / pm_two2):
    // the rows travel on to the root.
    if (llvm::isa<mlir::BlockArgument>(root))
      continue;
    if (rows < rootTy.getDimSize(0)) {
      // A parameter with fewer rows than the root: Vitis applies the pragma
      // one dimension deeper in the root (dimension d of the parameter to
      // dimension d + 1; dim=0 to dimensions 2..rank) and ignores what falls
      // past the root's rank (214-253). Kernels w9r16p_pm_mis (dim=2
      // complete, ignored), w9r16rv_pm_dim1 (dim=1 cyclic 2: "Cyclic
      // partitioning with factor 2 on dimension 2"), w9r16fx_pm_d2cyc (dim=2
      // cyclic 2, ignored), w9r16fx_pm_d0 (dim=0 complete: "Complete
      // partitioning on dimension 2" and 214-253), each with a [50][8]
      // parameter on g[100][8].
      int64_t dim = 1; // the pragma default
      if (auto di = d.getAs<mlir::IntegerAttr>("dim"))
        dim = di.getInt();
      int64_t rank = rootTy.getRank();
      llvm::SmallVector<int64_t> dims;
      bool dropped = false;
      if (dim == 0) {
        for (int64_t q = 2; q <= rank; ++q)
          dims.push_back(q);
        dropped = true; // the parameter's last dimension falls past the rank
      } else if (dim + 1 <= rank) {
        dims.push_back(dim + 1);
      } else {
        dropped = true;
      }
      auto var = d.getAs<mlir::StringAttr>("variable");
      if (dropped)
        llvm::errs() << call->getLoc()
                     << ": warning: array_partition on parameter '"
                     << (var ? var.getValue() : "?") << "' of '" << calleeName
                     << "' ignored past dimension " << rank
                     << ": its first dimension (" << rows
                     << ") is smaller than that of the array it reaches ("
                     << rootTy.getDimSize(0)
                     << "), mismatched array size w.r.t. its call site "
                        "(Vitis HLS 214-253)\n";
      for (int64_t q : dims) {
        llvm::SmallVector<mlir::NamedAttribute> shifted;
        for (mlir::NamedAttribute f : plain)
          if (f.getName() != "dim")
            shifted.push_back(f);
        shifted.push_back(mlir::NamedAttribute(
            mlir::StringAttr::get(ctx, "dim"),
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), q)));
        kept.push_back(mlir::DictionaryAttr::get(ctx, shifted));
      }
      continue;
    }
    kept.push_back(plain);
  }
  auto append = [&](mlir::ArrayAttr old,
                    llvm::ArrayRef<mlir::Attribute> add) -> mlir::ArrayAttr {
    llvm::SmallVector<mlir::Attribute> all;
    if (old)
      all.append(old.begin(), old.end());
    for (mlir::Attribute a : add)
      if (!llvm::is_contained(all, a))
        all.push_back(a);
    return mlir::ArrayAttr::get(ctx, all);
  };
  if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(root)) {
    auto fn = llvm::dyn_cast<mlir::FunctionOpInterface>(
        ba.getOwner()->getParentOp());
    if (!fn || ba.getOwner() != &fn.getFunctionBody().front())
      return;
    unsigned i = ba.getArgNumber();
    fn.setArgAttr(i, "hls.array_partition",
                  append(fn.getArgAttrOfType<mlir::ArrayAttr>(
                             i, "hls.array_partition"),
                         withRows));
    return;
  }
  if (kept.empty())
    return;
  mlir::Operation *holder = nullptr;
  if (auto a = root.getDefiningOp<mlir::memref::AllocaOp>())
    holder = a;
  else if (auto a = root.getDefiningOp<mlir::memref::AllocOp>())
    holder = a;
  else if (auto gg = root.getDefiningOp<mlir::memref::GetGlobalOp>())
    holder = mlir::SymbolTable::lookupNearestSymbolFrom(gg, gg.getNameAttr());
  if (!holder)
    return;
  holder->setAttr("hls.array_partition",
                  append(holder->getAttrOfType<mlir::ArrayAttr>(
                             "hls.array_partition"),
                         kept));
}

/// Whether an array partition is involved in the misfit argument `arg` of
/// `call` to `callable`: a partitioned parameter of the callee, a partition
/// on the array the argument names, or a partitioned parameter of another
/// callee that array is passed to (the partition is carried to it when that
/// call is inlined). hls-est cannot yet bank a partitioned array through the
/// reinterpreted view a kept call receives (OPEN_ISSUES O167 (a) in the
/// hls-est tree), so such a call is still inlined.
static bool misfitTouchesPartition(mlir::Operation *call,
                                   mlir::Operation *callable,
                                   mlir::Value arg) {
  auto hasPartitionedParam = [](mlir::Operation *f) {
    auto fn = llvm::dyn_cast_or_null<mlir::FunctionOpInterface>(f);
    if (!fn)
      return false;
    for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i)
      if (fn.getArgAttr(i, "hls.array_partition"))
        return true;
    return false;
  };
  if (hasPartitionedParam(callable))
    return true;
  mlir::Value root = partitionRootOf(arg);
  if (auto ba = llvm::dyn_cast<mlir::BlockArgument>(root)) {
    auto fn = llvm::dyn_cast<mlir::FunctionOpInterface>(
        ba.getOwner()->getParentOp());
    if (fn && ba.getOwner() == &fn.getFunctionBody().front() &&
        fn.getArgAttr(ba.getArgNumber(), "hls.array_partition"))
      return true;
  } else if (mlir::Operation *def = root.getDefiningOp()) {
    mlir::Operation *holder = def;
    if (auto gg = llvm::dyn_cast<mlir::memref::GetGlobalOp>(def))
      holder = mlir::SymbolTable::lookupNearestSymbolFrom(gg, gg.getNameAttr());
    if (holder && holder->hasAttr("hls.array_partition"))
      return true;
  }
  // The array (or a view of it), passed down through any chain of calls,
  // reaching a partitioned parameter: the partition is carried to it when
  // that call is inlined (mult_op_module_mmv: BRAM_A2 reaches mmv_ij's
  // partitioned parameter through gemm_ijk_with_mmv).
  llvm::SmallVector<mlir::Value> work;
  llvm::SmallPtrSet<void *, 16> seen;
  auto push = [&](mlir::Value v) {
    if (seen.insert(v.getAsOpaquePointer()).second)
      work.push_back(v);
  };
  push(root);
  if (auto gg = root.getDefiningOp<mlir::memref::GetGlobalOp>())
    if (auto module = call->getParentOfType<mlir::ModuleOp>())
      module.walk([&](mlir::memref::GetGlobalOp other) {
        if (other.getNameAttr() == gg.getNameAttr())
          push(other.getResult());
      });
  while (!work.empty()) {
    mlir::Value v = work.pop_back_val();
    for (mlir::OpOperand &use : v.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (llvm::isa<mlir::memref::CastOp, mlir::memref::ReinterpretCastOp,
                    mlir::memref::SubViewOp>(user) ||
          user->getName().getStringRef() == "polygeist.subindex") {
        if (user->getNumResults() == 1)
          push(user->getResult(0));
        continue;
      }
      auto other = llvm::dyn_cast<mlir::CallOpInterface>(user);
      if (!other)
        continue;
      auto fn = llvm::dyn_cast_or_null<mlir::FunctionOpInterface>(
          other.resolveCallable());
      if (!fn || fn.isExternal())
        continue;
      mlir::ValueRange args = other.getArgOperands();
      for (unsigned i = 0, e = std::min<unsigned>(args.size(),
                                                  fn.getNumArguments());
           i < e; ++i) {
        if (args[i] != v)
          continue;
        if (fn.getArgAttr(i, "hls.array_partition"))
          return true;
        push(fn.getArgument(i));
      }
    }
  }
  return false;
}

namespace {
/// The func dialect's inliner interface: FuncInlinerInterface of
/// mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp (LLVM 18, same
/// licence as this file), except that a call to a function marked
/// `#pragma HLS inline off` is not legal to inline.
struct HLSFuncInlinerInterface : public mlir::DialectInlinerInterface {
  explicit HLSFuncInlinerInterface(mlir::Dialect *dialect)
      : DialectInlinerInterface(dialect) {
    latest = this;
  }

  /// The most recently created instance (cgeist creates one context).
  static const HLSFuncInlinerInterface *latest;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// Vitis keeps a function marked `#pragma HLS inline off` as its own module
  /// (one instance per concurrent caller, VITIS_NOTES N14 in the hls-est
  /// tree), so its calls stay calls. Every other call can be inlined.
  ///
  /// Exception: a call that is not valid IR on its own. cgeist casts an array
  /// argument to the shape the parameter declares (--memref-fullrank), and C
  /// ignores the first dimension of an array parameter, so `f(a)` with
  /// `int a[32][32]` and `void f(int p[8][32])` gives a memref.cast from
  /// memref<32x32xi32> to memref<8x32xi32>, which MLIR rejects. Inlining
  /// folds such a cast away (the callee's accesses then use `a` directly);
  /// a kept call would keep it. Such a call is inlined as before, with a
  /// warning.
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    if (!callable)
      return true;
    // An hls:: math library function kept as a core (XlxMathCore.h), except
    // at a call whose arguments are all constants: Vitis evaluates such a
    // call at compile time and builds no core (VITIS_NOTES N1a: hls::sqrt of
    // a constant folds into the caller; N16: a constant call costs 0), so its
    // body is inlined and folded here as well.
    if (callable->hasAttr("hls.math_core"))
      return llvm::all_of(call->getOperands(), [](mlir::Value v) {
        return mlir::matchPattern(v, mlir::m_Constant());
      });
    // `inline off`, or a call Vitis would not inline automatically
    // (hls.inline_auto_off, set by the pass before the inliner,
    // HLSAutoInline.h).
    bool pragmaOff =
        !HLSIgnoreInlineOff && callable->hasAttr("hls.inline_off");
    bool autoOff = call->hasAttr(cgeist::hls::kInlineAutoOffAttr);
    if (!pragmaOff && !autoOff)
      return true;
    for (auto [i, operand] : llvm::enumerate(call->getOperands())) {
      // castCallerMemRefArg reinterprets an argument whose static shape
      // differs from the parameter's (a valid op, since a chain of casts
      // would fold back into an invalid one).
      if (auto view =
              operand.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
        if (pragmaOff && HLSKeepMisfitInlineOff &&
            !misfitTouchesPartition(call, callable, operand))
          return false;
        auto name = callable->getAttrOfType<mlir::StringAttr>("hls.orig_name");
        llvm::errs() << call->getLoc() << ": warning: '#pragma HLS inline off' "
                     << "of '"
                     << (name ? name.getValue()
                              : mlir::SymbolTable::getSymbolName(callable)
                                    .getValue())
                     << "' not honoured at this call: argument " << i << " ("
                     << view.getSource().getType()
                     << ") does not fit the parameter type " << view.getType()
                     << "; the call is inlined\n";
        return true;
      }
      for (auto cast = operand.getDefiningOp<mlir::memref::CastOp>(); cast;
           cast = cast.getSource().getDefiningOp<mlir::memref::CastOp>()) {
        if (mlir::memref::CastOp::areCastCompatible(cast.getSource().getType(),
                                                     cast.getType()))
          continue;
        if (!pragmaOff)
          return true; // an automatic decision: inline it without a warning
        auto name = callable->getAttrOfType<mlir::StringAttr>("hls.orig_name");
        llvm::errs() << call->getLoc() << ": warning: '#pragma HLS inline off' "
                     << "of '"
                     << (name ? name.getValue()
                              : mlir::SymbolTable::getSymbolName(callable)
                                    .getValue())
                     << "' not honoured at this call: argument " << i << " ("
                     << cast.getSource().getType()
                     << ") does not fit the parameter type " << cast.getType()
                     << "; the call is inlined\n";
        return true;
      }
    }
    return false;
  }

  /// All operations can be inlined.
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  /// All functions can be inlined.
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// A callee marked polygeist.volatile_access (Passes.h) passes the mark to
  /// the function it is inlined into: its volatile accesses are now in that
  /// function's body, and CanonicalizeFor reads the mark per function.
  ///
  /// A call marked hls.ap_compound_assign (an ap_fixed `/=`, `*=`, `+=`,
  /// `-=`, CGCall.cc) passes the mark to every store and every call of the
  /// inlined body, so that the stores the compound assignment leaves carry
  /// it whatever is inlined first.
  void processInlinedCallBlocks(
      mlir::Operation *call,
      llvm::iterator_range<mlir::Region::iterator> blocks) const final {
    // The stores are marked through their location (a tag materialized
    // into the attribute at the end, HLSPragmaMaterialize.cc): the
    // canonicalizations that fold the operator's views rebuild the stores
    // and keep only the location.
    if (call->hasAttr("hls.ap_compound_assign")) {
      auto mark = mlir::UnitAttr::get(call->getContext());
      State &S = State::get();
      if (!S.compoundAssignTag) {
        TagRecord &T = S.newTag(TagClass::Operation);
        T.attrs.push_back({"hls.ap_compound_assign", mark});
        T.opNames = {"affine.store", "memref.store", "llvm.store"};
        T.pragmaName = "ap_fixed compound assignment";
        T.what = "store";
        S.compoundAssignTag = T.id;
      }
      unsigned id = *S.compoundAssignTag;
      for (mlir::Block &b : blocks)
        b.walk([&](mlir::Operation *op) {
          if (llvm::isa<mlir::CallOpInterface>(op))
            op->setAttr("hls.ap_compound_assign", mark);
          else if (llvm::isa<mlir::memref::StoreOp,
                             mlir::affine::AffineStoreOp,
                             mlir::LLVM::StoreOp>(op))
            op->setLoc(S.tag(op->getLoc(), id));
        });
    }
    auto callOp = llvm::dyn_cast<mlir::CallOpInterface>(call);
    if (!callOp)
      return;
    mlir::Operation *callee = callOp.resolveCallable();
    // The callee's parameter partitions reach the caller's arrays (Vitis
    // applies them whether or not it inlines the callee; N24 (f), N123).
    if (HLSInlineCarryPartition && callee)
      if (auto fn = llvm::dyn_cast<mlir::FunctionOpInterface>(callee)) {
        auto name = callee->getAttrOfType<mlir::StringAttr>("hls.orig_name");
        llvm::StringRef calleeName =
            name ? name.getValue()
                 : mlir::SymbolTable::getSymbolName(callee).getValue();
        mlir::ValueRange args = callOp.getArgOperands();
        for (unsigned i = 0, e = std::min<unsigned>(args.size(),
                                                    fn.getNumArguments());
             i < e; ++i) {
          auto entries =
              fn.getArgAttrOfType<mlir::ArrayAttr>(i, "hls.array_partition");
          auto paramTy =
              llvm::dyn_cast<mlir::MemRefType>(fn.getArgumentTypes()[i]);
          if (entries && paramTy)
            carryPartition(call, args[i], paramTy, entries, calleeName);
        }
      }
    if (!callee || !callee->hasAttr(mlir::polygeist::kVolatileAccessAttrName))
      return;
    if (auto caller = call->getParentOfType<mlir::FunctionOpInterface>())
      caller->setAttr(mlir::polygeist::kVolatileAccessAttrName,
                      mlir::UnitAttr::get(call->getContext()));
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final {
    // Only return needs to be handled here.
    auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    mlir::OpBuilder builder(op);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest,
                                       returnOp.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op,
                        llvm::ArrayRef<mlir::Value> valuesToRepl) const final {
    // Only return needs to be handled here.
    auto returnOp = llvm::cast<mlir::func::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
} // namespace

const HLSFuncInlinerInterface *HLSFuncInlinerInterface::latest = nullptr;

// The Vitis HLS headers' proxy objects (ap_range_ref, ap_bit_ref,
// ap_concat_ref, af_range_ref, af_bit_ref: a reference member plus indices,
// etc/ap_int_ref.h, ap_fixed_ref.h) are reached through
// polygeist.memref2pointer in their always_inline member functions. Without an
// inliner interface on the polygeist dialect those functions stay calls and
// the proxy stays a memref of an LLVM struct holding a memref, which is not
// valid MLIR once printed. With --xlx-headers the polygeist ops are legal to
// inline, so that the proxies can be scalarised in their callers
// (Lib/XlxProxyScalarize.cc, run by the driver under the same option).
// --xlx-headers also covers the other code-generation changes the real
// headers need (cgeist::hls::xlxHeaders() callers): struct field types
// compared as MLIR types, and namespace-qualified globals as memref globals.
static llvm::cl::opt<bool> HLSInlinePolygeistOps(
    "xlx-headers", llvm::cl::init(false),
    llvm::cl::desc("Code generation for the real Vitis HLS headers: inline "
                   "and scalarise their proxy objects (ap_range_ref, "
                   "ap_bit_ref, ap_concat_ref, ...), keep struct fields of "
                   "different MLIR types apart, and address "
                   "namespace-qualified globals as memref globals"));

bool cgeist::hls::xlxHeaders() { return HLSInlinePolygeistOps; }

bool cgeist::hls::isApFixedClass(const clang::CXXRecordDecl *rd) {
  for (int depth = 0; rd && depth < 8; ++depth) {
    rd = rd->getDefinition();
    if (!rd)
      return false;
    llvm::StringRef name = rd->getName();
    if (name == "ap_fixed" || name == "ap_ufixed" || name == "ap_fixed_base")
      return true;
    if (rd->getNumBases() != 1)
      return false;
    rd = rd->bases_begin()->getType()->getAsCXXRecordDecl();
  }
  return false;
}

namespace {
struct PolygeistOpsInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return HLSInlinePolygeistOps;
  }
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return HLSInlinePolygeistOps;
  }
};
} // namespace

void cgeist::hls::registerInlinerExtension(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::func::FuncDialect *dialect) {
        dialect->addInterfaces<HLSFuncInlinerInterface>();
        // The inliner extension relies on the ControlFlow dialect.
        ctx->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
      });
  registry.addExtension(
      +[](mlir::MLIRContext *, mlir::polygeist::PolygeistDialect *dialect) {
        dialect->addInterfaces<PolygeistOpsInlinerInterface>();
      });
}

bool cgeist::hls::inlinerExtensionActive(mlir::MLIRContext *ctx) {
  auto *dialect = ctx->getOrLoadDialect<mlir::func::FuncDialect>();
  const auto *iface =
      dialect->getRegisteredInterface<mlir::DialectInlinerInterface>();
  return iface && iface == HLSFuncInlinerInterface::latest;
}

//===----------------------------------------------------------------------===//
// AGGREGATE through callee parameters
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool> HLSPropagateAggregate(
    "hls-propagate-aggregate", llvm::cl::init(true),
    llvm::cl::desc("With --hls-annotate: an AGGREGATE pragma on a function "
                   "parameter (or on `this` of a member function) also marks "
                   "the variable every call passes to it, as Vitis aggregates "
                   "the actual argument"));

namespace {
constexpr llvm::StringLiteral kAggregateAttr = "hls.aggregate";

/// The value `view` is a view of whole elements of: memref.cast and
/// polygeist.subindex steps that keep the element type are followed (an
/// element `a[i]` of a struct array, the array decayed to a pointer).
/// A view of a member (a different element type) has no root: the pragma
/// then reaches only that member, which is not modelled.
mlir::Value elementViewRoot(mlir::Value view) {
  auto elementType = [](mlir::Value v) -> mlir::Type {
    auto t = llvm::dyn_cast<mlir::MemRefType>(v.getType());
    return t ? t.getElementType() : mlir::Type();
  };
  mlir::Type elem = elementType(view);
  if (!elem)
    return {};
  while (mlir::Operation *def = view.getDefiningOp()) {
    mlir::Value source;
    if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(def))
      source = cast.getSource();
    else if (auto sub = llvm::dyn_cast<mlir::polygeist::SubIndexOp>(def))
      source = sub.getSource();
    else
      break;
    if (elementType(source) != elem)
      return {};
    view = source;
  }
  return view;
}

/// The callee's entry, recorded on the actual with the name of the callee
/// it was passed to (`from`; the nearest one along a chain of calls).
mlir::Attribute propagatedEntry(mlir::Attribute entry,
                                mlir::func::FuncOp callee) {
  auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(entry);
  if (!dict)
    return entry;
  mlir::NamedAttrList fields(dict);
  fields.set("from",
             mlir::StringAttr::get(dict.getContext(), callee.getSymName()));
  return fields.getDictionary(dict.getContext());
}
} // namespace

void cgeist::hls::propagateAggregate(mlir::ModuleOp module) {
  if (!isEnabled() || !HLSPropagateAggregate)
    return;
  llvm::StringMap<llvm::SmallVector<mlir::func::CallOp>> callsTo;
  module.walk([&](mlir::func::CallOp call) {
    callsTo[call.getCallee()].push_back(call);
  });
  mlir::SymbolTable symbols(module);

  llvm::SmallVector<std::pair<mlir::func::FuncOp, unsigned>> work;
  module.walk([&](mlir::func::FuncOp fn) {
    for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i)
      if (fn.getArgAttr(i, kAggregateAttr))
        work.push_back({fn, i});
  });

  while (!work.empty()) {
    auto [callee, index] = work.pop_back_val();
    auto entries = callee.getArgAttrOfType<mlir::ArrayAttr>(index,
                                                            kAggregateAttr);
    if (!entries || entries.empty())
      continue;
    mlir::Attribute entry = propagatedEntry(entries[0], callee);
    auto mark = mlir::ArrayAttr::get(module.getContext(), {entry});
    for (mlir::func::CallOp call : callsTo.lookup(callee.getSymName())) {
      if (index >= call.getNumOperands())
        continue;
      mlir::Value root = elementViewRoot(call.getOperand(index));
      if (!root)
        continue;
      if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(root)) {
        auto caller = llvm::dyn_cast<mlir::func::FuncOp>(
            arg.getOwner()->getParentOp());
        if (!caller || arg.getOwner() != &caller.getBody().front() ||
            caller.getArgAttr(arg.getArgNumber(), kAggregateAttr))
          continue;
        caller.setArgAttr(arg.getArgNumber(), kAggregateAttr, mark);
        work.push_back({caller, arg.getArgNumber()});
        continue;
      }
      mlir::Operation *storage = root.getDefiningOp();
      if (auto get = llvm::dyn_cast<mlir::memref::GetGlobalOp>(storage))
        storage = symbols.lookup(get.getName());
      if (!storage ||
          !llvm::isa<mlir::memref::AllocaOp, mlir::memref::AllocOp,
                     mlir::memref::GlobalOp>(storage) ||
          storage->hasAttr(kAggregateAttr))
        continue;
      storage->setAttr(kAggregateAttr, mark);
    }
  }
}

namespace {
struct PropagateAggregatePass
    : public mlir::PassWrapper<PropagateAggregatePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropagateAggregatePass)
  llvm::StringRef getArgument() const final {
    return "hls-propagate-aggregate";
  }
  llvm::StringRef getDescription() const final {
    return "Mark the actual arguments of parameters that carry an AGGREGATE "
           "pragma";
  }
  void runOnOperation() final { propagateAggregate(getOperation()); }
};
} // namespace

std::unique_ptr<mlir::Pass> cgeist::hls::createPropagateAggregatePass() {
  return std::make_unique<PropagateAggregatePass>();
}

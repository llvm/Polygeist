//===- HLSPragmaMaterialize.cc - Location tags to hls.* attributes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After the pass pipeline, every op whose location carries an HLS tag gets
// the attributes recorded for that tag, provided the op is of the right kind
// (a loop for loop pragmas, an allocation for variable pragmas, a matching
// arithmetic op for bind_op). Tags are then removed from all locations so
// the printed IR (including debug info) is unaffected.
//
//===----------------------------------------------------------------------===//

#include "HLSPragmaInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace cgeist::hls;

namespace {

std::optional<unsigned> tagOf(mlir::FusedLoc loc) {
  auto dict = llvm::dyn_cast_or_null<mlir::DictionaryAttr>(loc.getMetadata());
  if (!dict)
    return std::nullopt;
  if (auto id = dict.getAs<mlir::IntegerAttr>(kTagKey))
    return id.getInt();
  return std::nullopt;
}

void collectTags(mlir::Location loc, llvm::SetVector<unsigned> &ids) {
  if (auto fused = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
    if (auto id = tagOf(fused))
      ids.insert(*id);
    for (mlir::Location l : fused.getLocations())
      collectTags(l, ids);
  } else if (auto call = llvm::dyn_cast<mlir::CallSiteLoc>(loc)) {
    collectTags(call.getCallee(), ids);
  } else if (auto nameLoc = llvm::dyn_cast<mlir::NameLoc>(loc)) {
    collectTags(nameLoc.getChildLoc(), ids);
  } else if (auto opaque = llvm::dyn_cast<mlir::OpaqueLoc>(loc)) {
    collectTags(opaque.getFallbackLocation(), ids);
  }
}

/// `loc` without HLS tags.
mlir::Location strip(mlir::Location loc) {
  mlir::MLIRContext *ctx = loc.getContext();
  if (auto fused = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
    llvm::SmallVector<mlir::Location> inner;
    for (mlir::Location l : fused.getLocations())
      inner.push_back(strip(l));
    mlir::Attribute meta = tagOf(fused) ? mlir::Attribute() : fused.getMetadata();
    if (inner.empty())
      return mlir::UnknownLoc::get(ctx);
    return mlir::FusedLoc::get(inner, meta, ctx);
  }
  if (auto call = llvm::dyn_cast<mlir::CallSiteLoc>(loc))
    return mlir::CallSiteLoc::get(strip(call.getCallee()),
                                  strip(call.getCaller()));
  if (auto nameLoc = llvm::dyn_cast<mlir::NameLoc>(loc))
    return mlir::NameLoc::get(nameLoc.getName(), strip(nameLoc.getChildLoc()));
  return loc;
}

bool accepts(const TagRecord &T, mlir::Operation *op) {
  switch (T.cls) {
  case TagClass::Loop:
    return llvm::isa<mlir::LoopLikeOpInterface>(op);
  case TagClass::Storage:
    return llvm::isa<mlir::memref::AllocaOp, mlir::memref::AllocOp,
                     mlir::LLVM::AllocaOp>(op);
  case TagClass::Operation: {
    if (!llvm::is_contained(T.opNames, op->getName().getStringRef().str()))
      return false;
    if (!T.floatWidth)
      return true;
    auto ft = op->getNumResults() == 1
                  ? llvm::dyn_cast<mlir::FloatType>(mlir::getElementTypeOrSelf(
                        op->getResult(0).getType()))
                  : mlir::FloatType();
    return ft && ft.getWidth() == T.floatWidth;
  }
  }
  return false;
}

void apply(const TagRecord &T, mlir::Operation *op) {
  for (auto &[name, value] : T.attrs) {
    auto oldArr = op->getAttrOfType<mlir::ArrayAttr>(name);
    auto newArr = llvm::dyn_cast<mlir::ArrayAttr>(value);
    if (oldArr && newArr) {
      llvm::SmallVector<mlir::Attribute> all(oldArr.begin(), oldArr.end());
      for (mlir::Attribute a : newArr)
        if (!llvm::is_contained(all, a))
          all.push_back(a);
      op->setAttr(name, mlir::ArrayAttr::get(op->getContext(), all));
    } else {
      op->setAttr(name, value);
    }
  }
}

} // namespace

void cgeist::hls::materialize(mlir::ModuleOp module, bool reportLost) {
  State &S = State::get();

  module->walk([&](mlir::Operation *op) {
    llvm::SetVector<unsigned> ids;
    collectTags(op->getLoc(), ids);
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument arg : block.getArguments())
          arg.setLoc(strip(arg.getLoc()));
    if (ids.empty())
      return;
    for (unsigned id : ids) {
      if (id >= S.tags.size())
        continue;
      TagRecord &T = S.tags[id];
      if (!accepts(T, op))
        continue;
      apply(T, op);
      T.materialized = true;
      if (reportEnabled()) {
        auto func = op->getParentOfType<mlir::func::FuncOp>();
        std::string where = op->getName().getStringRef().str() + " (" +
                            T.what + ")" +
                            (func ? " in @" + func.getName().str() : "");
        for (int r : T.reports)
          S.addOutcome(r, where);
      }
    }
    op->setLoc(strip(op->getLoc()));
  });

  if (reportLost) {
    for (TagRecord &T : S.tags) {
      if (T.materialized || !T.tagged)
        continue;
      // A function that was not kept (e.g. inlined everywhere and removed)
      // took the target with it; that is not a lost pragma.
      if (!module.lookupSymbol(T.funcName)) {
        for (int r : T.reports)
          S.addOutcome(r, "not applied: @" + T.funcName +
                              " was removed (inlined into its callers or "
                              "unused), and so was the " + T.what);
        continue;
      }
      T.materialized = true; // report once
      for (int r : T.reports)
        S.addOutcome(r, T.cls == TagClass::Operation
                            ? "not applied: the operation assigned to " +
                                  T.what + " was folded away"
                            : "not applied: the " + T.what +
                                  " was removed during optimization");
      llvm::errs() << T.locString << ": warning: '" << T.pragmaName << "': "
                   << (T.cls == TagClass::Loop
                           ? "the loop it applies to was removed during "
                             "optimization (e.g. fully unrolled or proven "
                             "dead)"
                       : T.cls == TagClass::Storage
                           ? T.what + " was promoted to registers or removed "
                                      "during optimization"
                           : "the operation it binds was folded away during "
                             "optimization")
                   << "; the pragma has no effect\n";
    }
  }

  if (reportEnabled()) {
    for (ReportEntry &E : S.report) {
      if (E.outcomes.empty())
        E.outcomes.push_back("not applied: the target was not emitted");
      for (const std::string &o : E.outcomes)
        llvm::errs() << E.loc << ": remark: '" << E.pragma << "' -> " << o
                     << "\n";
    }
    S.report.clear();
  }
}

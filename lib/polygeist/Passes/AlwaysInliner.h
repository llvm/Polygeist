#ifndef _POLYGEIST_PASSES_ALWAYSINLINER_H_
#define _POLYGEIST_PASSES_ALWAYSINLINER_H_

#include "PassDetails.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

struct AlwaysInlinerInterface : public mlir::InlinerInterface {
  using InlinerInterface::InlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final {
    // Only "std.return" needs to be handled here.
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op);
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
                        mlir::ArrayRef<mlir::Value> valuesToRepl) const final {
    // Only "std.return" needs to be handled here.
    auto returnOp = mlir::cast<mlir::func::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};

[[maybe_unused]] static void alwaysInlineCall(mlir::func::CallOp caller) {
  // Build the inliner interface.
  AlwaysInlinerInterface interface(caller.getContext());

  auto callable = caller.getCallableForCallee();
  mlir::CallableOpInterface callableOp;
  if (mlir::SymbolRefAttr symRef =
          mlir::dyn_cast<mlir::SymbolRefAttr>(callable)) {
    auto *symbolOp =
        caller->getParentOfType<mlir::ModuleOp>().lookupSymbol(symRef);
    callableOp = mlir::dyn_cast_or_null<mlir::CallableOpInterface>(symbolOp);
  } else {
    return;
  }
  mlir::Region *targetRegion = callableOp.getCallableRegion();
  if (!targetRegion)
    return;
  if (targetRegion->empty())
    return;
  if (inlineCall(interface, caller, callableOp, targetRegion,
                 /*shouldCloneInlinedRegion=*/true)
          .succeeded()) {
    caller.erase();
  }
};

#endif // _POLYGEIST_PASSES_ALWAYSINLINER_H_

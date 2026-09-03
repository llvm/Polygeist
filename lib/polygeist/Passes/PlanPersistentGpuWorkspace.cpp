//===- PlanPersistentGpuWorkspace.cpp - stable graph scratch -------------===//

#include "PassDetails.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;

namespace {

struct ScratchRoot {
  Operation *op;
  Value result;
  RankedTensorType type;
};

struct MemrefScratchRoot {
  memref::AllocOp op;
  MemRefType type;
};

struct PlanPersistentGpuWorkspacePass
    : public mlir::polygeist::PlanPersistentGpuWorkspaceBase<
          PlanPersistentGpuWorkspacePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (functionName.empty()) {
      module.emitError("--plan-persistent-gpu-workspace requires function=");
      return signalPassFailure();
    }

    auto function = module.lookupSymbol<func::FuncOp>(functionName);
    if (!function) {
      module.emitError() << "workspace function not found: " << functionName;
      return signalPassFailure();
    }
    bool isRecursive = false;
    function.walk([&](func::CallOp call) {
      if (call.getCallee() == functionName)
        isRecursive = true;
    });
    if (isRecursive) {
      function.emitError(
          "persistent global workspace is not safe for recursive functions");
      return signalPassFailure();
    }

    SmallVector<ScratchRoot> roots;
    SmallVector<MemrefScratchRoot> memrefRoots;
    function.walk([&](Operation *op) {
      Value result;
      if (auto empty = dyn_cast<tensor::EmptyOp>(op)) {
        result = empty.getResult();
      } else if (auto alloc = dyn_cast<bufferization::AllocTensorOp>(op)) {
        if (!alloc.getDynamicSizes().empty() || alloc.getCopy()) {
          return;
        }
        result = alloc.getResult();
      } else {
        if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
          auto type = alloc.getType();
          if (!type.hasStaticShape() || !alloc.getDynamicSizes().empty() ||
              !alloc.getSymbolOperands().empty()) {
            return;
          }
          memrefRoots.push_back({alloc, type});
        }
        return;
      }
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type || !type.hasStaticShape()) {
        return;
      }
      roots.push_back({op, result, type});
    });

    if (roots.empty() && memrefRoots.empty()) {
      function->setAttr("polygeist.persistent_workspace",
                        UnitAttr::get(module.getContext()));
      return;
    }

    SymbolTable symbolTable(module);
    unsigned ordinal = 0;
    for (ScratchRoot root : roots) {
      auto memrefType = MemRefType::get(root.type.getShape(),
                                        root.type.getElementType());
      std::string name = ("__polygeist_workspace_" + functionName + "_" +
                          std::to_string(ordinal++));

      OpBuilder globalBuilder(module.getContext());
      globalBuilder.setInsertionPointToStart(module.getBody());
      auto global = globalBuilder.create<memref::GlobalOp>(
          root.op->getLoc(), globalBuilder.getStringAttr(name),
          globalBuilder.getStringAttr("private"), TypeAttr::get(memrefType),
          Attribute{}, UnitAttr{}, globalBuilder.getI64IntegerAttr(4096));
      symbolTable.insert(global);

      OpBuilder builder(root.op);
      auto getGlobal = builder.create<memref::GetGlobalOp>(
          root.op->getLoc(), memrefType, global.getName());
      auto tensor = builder.create<bufferization::ToTensorOp>(
          root.op->getLoc(), root.type, getGlobal.getResult(),
          /*restrict=*/true, /*writable=*/true);
      root.result.replaceAllUsesWith(tensor.getResult());
      root.op->erase();
    }

    for (MemrefScratchRoot root : memrefRoots) {
      std::string name = ("__polygeist_workspace_" + functionName + "_" +
                          std::to_string(ordinal++));
      OpBuilder globalBuilder(module.getContext());
      globalBuilder.setInsertionPointToStart(module.getBody());
      auto global = globalBuilder.create<memref::GlobalOp>(
          root.op.getLoc(), globalBuilder.getStringAttr(name),
          globalBuilder.getStringAttr("private"), TypeAttr::get(root.type),
          Attribute{}, UnitAttr{}, globalBuilder.getI64IntegerAttr(4096));
      symbolTable.insert(global);

      OpBuilder builder(root.op);
      auto getGlobal = builder.create<memref::GetGlobalOp>(
          root.op.getLoc(), root.type, global.getName());
      SmallVector<memref::DeallocOp> deallocs;
      for (Operation *user : root.op.getResult().getUsers())
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          deallocs.push_back(dealloc);
      root.op.getResult().replaceAllUsesWith(getGlobal.getResult());
      for (memref::DeallocOp dealloc : deallocs)
        dealloc.erase();
      root.op.erase();
    }

    function->setAttr("polygeist.persistent_workspace",
                      UnitAttr::get(module.getContext()));
  }
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createPlanPersistentGpuWorkspacePass() {
  return std::make_unique<PlanPersistentGpuWorkspacePass>();
}
} // namespace polygeist
} // namespace mlir

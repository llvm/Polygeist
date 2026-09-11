//===- PlanGpuDataResidency.cpp - automatic GPU buffer ownership ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Passes/Passes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"

#include <optional>

using namespace mlir;

namespace {

static constexpr StringLiteral kPersistentWorkspaceCandidateAttr =
    "polygeist.persistent_gpu_workspace_candidate";

static bool isGpuRuntimeCall(Operation *op) {
  auto call = dyn_cast<func::CallOp>(op);
  if (!call)
    return false;
  StringRef callee = call.getCallee();
  return callee.starts_with("polygeist_cublas_") ||
         callee.starts_with("polygeist_cublaslt_") ||
         callee.starts_with("polygeist_cudnn_") ||
         callee.starts_with("polygeist_cub_") ||
         callee.starts_with("polygeist_cufft_") ||
         callee.starts_with("polygeist_cusolver_") ||
         callee.starts_with("polygeist_cusparse_") ||
         callee.starts_with("polygeist_cutensor_") ||
         callee.starts_with("polygeist_cutensornet_") ||
         callee.starts_with("polygeist_cuda_") ||
         callee.starts_with("polygeist_rmsnorm_");
}

// Return true for operations which only preserve or inspect a memref
// descriptor. Replacing their source with a gpu.alloc result is safe: none of
// them dereference the pointer on the host.
static bool isDescriptorOnlyUse(Operation *op) {
  if (isa<memref::CastOp, memref::SubViewOp, memref::ReinterpretCastOp,
          memref::CollapseShapeOp, memref::ExpandShapeOp, memref::DimOp,
          memref::ExtractStridedMetadataOp,
          memref::ExtractAlignedPointerAsIndexOp, bufferization::ToTensorOp>(
          op))
    return true;

  // Library lowering converts an extracted memref pointer through index and
  // integer arithmetic before llvm.inttoptr. These operations manipulate only
  // the descriptor-derived address; they do not access host memory.
  StringRef name = op->getName().getStringRef();
  return name.starts_with("arith.") || name == "llvm.inttoptr" ||
         name == "llvm.ptrtoint" ||
         name == "builtin.unrealized_conversion_cast";
}

// A promoted value may flow through an arbitrary number of view operations.
// Reject the complete argument when any branch performs host computation. This
// conservative all-or-nothing rule avoids introducing coherence points inside
// a function and leaves such arguments on the mapped-host fallback.
static func::FuncOp getOwnedCallee(func::CallOp call) {
  auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  if (!callee || callee.isDeclaration() ||
      (!callee.isPrivate() &&
       !callee->hasAttr("polygeist.gpu_resident_callee")))
    return {};
  return callee;
}

static bool hasOnlyDeviceOrDescriptorUses(Value root) {
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (isa<gpu::LaunchFuncOp, gpu::HostRegisterOp>(owner))
        continue;
      if (isGpuRuntimeCall(owner))
        continue;
      // A compiler-generated private helper is part of the selected
      // function's ownership region. Follow the corresponding formal
      // argument instead of forcing frontends to inline a large repeated
      // computation. Public/external callees remain conservative because
      // they may also be invoked with host pointers elsewhere.
      if (auto call = dyn_cast<func::CallOp>(owner)) {
        if (func::FuncOp callee = getOwnedCallee(call)) {
          unsigned operand = use.getOperandNumber();
          if (operand < callee.getNumArguments()) {
            worklist.push_back(callee.getArgument(operand));
            continue;
          }
        }
      }
      if (isDescriptorOnlyUse(owner)) {
        worklist.append(owner->getResults().begin(), owner->getResults().end());
        continue;
      }
      // Runtime-library pointer arguments are reached through descriptor-only
      // extraction followed by integer/pointer scalar operations, not through
      // the memref SSA value itself. Direct memref calls are unknown host ABI
      // consumers and therefore intentionally rejected here.
      return false;
    }
  }
  return true;
}

// Collect registrations anywhere below a descriptor/view chain.  A device
// allocation must never be passed to gpu.host_register: registration is only
// meaningful for host storage and is both unnecessary and invalid for the
// replacement produced by this pass.
static void collectHostRegistrations(
    Value root, SmallVectorImpl<gpu::HostRegisterOp> &registrations) {
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (auto registration = dyn_cast<gpu::HostRegisterOp>(user)) {
        registrations.push_back(registration);
        continue;
      }
      if (isDescriptorOnlyUse(user))
        worklist.append(user->getResults().begin(), user->getResults().end());
    }
  }
}

static bool hasSimpleDynamicLaunchUses(Value root) {
  return llvm::all_of(root.getUsers(), [](Operation *user) {
    return isa<gpu::LaunchFuncOp, memref::DeallocOp>(user);
  });
}

static bool hasGpuDispatch(func::FuncOp function) {
  bool found = false;
  llvm::DenseSet<Operation *> visited;
  SmallVector<func::FuncOp> worklist{function};
  while (!worklist.empty() && !found) {
    func::FuncOp current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    current.walk([&](Operation *op) {
      if (isa<gpu::LaunchFuncOp>(op) || isGpuRuntimeCall(op)) {
        found = true;
        return WalkResult::interrupt();
      }
      if (auto call = dyn_cast<func::CallOp>(op))
        if (func::FuncOp callee = getOwnedCallee(call))
          worklist.push_back(callee);
      return WalkResult::advance();
    });
  }
  return found;
}

struct PlanGpuDataResidencyPass
    : public mlir::polygeist::PlanGpuDataResidencyBase<
          PlanGpuDataResidencyPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (functionName.empty()) {
      module.emitError("--plan-gpu-data-residency requires function=");
      return signalPassFailure();
    }
    auto function = module.lookupSymbol<func::FuncOp>(functionName);
    if (!function) {
      module.emitError() << "GPU residency function not found: "
                         << functionName;
      return signalPassFailure();
    }
    if (function.isDeclaration() || !hasGpuDispatch(function))
      return;

    // The persistent-workspace planner initially uses private host globals to
    // give scratch a stable lifetime before GPU outlining.  At this late point
    // the complete use graph is visible.  Replace a marked global with one
    // function-scoped device allocation only when every get_global in this
    // function is device/descriptor-only.  Host-observable workspaces retain
    // the mapped-host fallback.
    llvm::SmallDenseMap<Operation *, SmallVector<memref::GetGlobalOp, 2>, 4>
        workspaceGets;
    function.walk([&](memref::GetGlobalOp getGlobal) {
      auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          getGlobal, getGlobal.getNameAttr());
      if (!global || !global.isPrivate() ||
          !global->hasAttr(kPersistentWorkspaceCandidateAttr))
        return;
      workspaceGets[global.getOperation()].push_back(getGlobal);
    });

    for (auto &[globalOperation, gets] : workspaceGets) {
      auto global = cast<memref::GlobalOp>(globalOperation);
      if (gets.empty() || llvm::any_of(gets, [](memref::GetGlobalOp get) {
            return !hasOnlyDeviceOrDescriptorUses(get.getResult());
          }))
        continue;

      auto type = cast<MemRefType>(global.getType());
      if (!type.hasStaticShape())
        continue;

      OpBuilder entryBuilder = OpBuilder::atBlockBegin(&function.front());
      Type tokenType = gpu::AsyncTokenType::get(module.getContext());
      Value stream =
          entryBuilder
              .create<gpu::WaitOp>(global.getLoc(), tokenType, ValueRange{})
              .getAsyncToken();
      auto deviceAllocation = entryBuilder.create<gpu::AllocOp>(
          global.getLoc(), type, tokenType, ValueRange{stream},
          /*dynamicSizes=*/ValueRange{}, /*symbolOperands=*/ValueRange{});
      entryBuilder.setInsertionPointAfter(deviceAllocation);
      entryBuilder.create<gpu::WaitOp>(global.getLoc(), /*asyncToken=*/Type(),
                                       ValueRange{deviceAllocation.getAsyncToken()});

      SmallVector<gpu::HostRegisterOp> registrations;
      for (memref::GetGlobalOp get : gets) {
        collectHostRegistrations(get.getResult(), registrations);
        get.getResult().replaceAllUsesWith(deviceAllocation.getMemref());
        get.erase();
      }
      for (gpu::HostRegisterOp registration : registrations)
        if (registration && registration->getBlock())
          registration.erase();

      SmallVector<func::ReturnOp> returns;
      function.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
      for (func::ReturnOp ret : returns) {
        OpBuilder returnBuilder(ret);
        returnBuilder.create<gpu::DeallocOp>(
            ret.getLoc(), /*asyncTokenType=*/Type(),
            /*asyncDependencies=*/ValueRange{}, deviceAllocation.getMemref());
      }
    }

    // Bufferization may introduce scratch allocations after the source-level
    // ownership boundary has been formed.  A gpu.launch_func cannot safely
    // dereference the ordinary heap pointer produced by memref.alloc.  Turn
    // scratch buffers whose complete use graph is device/descriptor-only into
    // device allocations as well.  This scans the selected function itself;
    // pipelines with an owned helper run it once more after inlining so the
    // scratch allocation can be hoisted into the outer lifetime.
    SmallVector<memref::AllocOp> localAllocations;
    function.walk([&](memref::AllocOp allocation) {
      // GPU-to-LLVM can expand a dynamic gpu.alloc used directly by outlined
      // launches. It cannot yet reconcile the descriptor when that allocation
      // also flows through a view or library-pointer extraction. Keep mixed
      // dynamic scratch on the mapped-host fallback.
      if ((allocation.getType().hasStaticShape() ||
           hasSimpleDynamicLaunchUses(allocation.getResult())) &&
          hasOnlyDeviceOrDescriptorUses(allocation.getResult()))
        localAllocations.push_back(allocation);
    });

    for (memref::AllocOp allocation : localAllocations) {
      SmallVector<memref::DeallocOp> deallocations;
      SmallVector<gpu::HostRegisterOp> registrations;
      collectHostRegistrations(allocation.getResult(), registrations);
      for (Operation *user : allocation.getResult().getUsers())
        if (auto deallocation = dyn_cast<memref::DeallocOp>(user))
          deallocations.push_back(deallocation);

      func::FuncOp allocationFunction =
          allocation->getParentOfType<func::FuncOp>();
      OpBuilder allocationBuilder(allocation);
      SmallVector<Value> dynamicSizes(allocation.getDynamicSizes());
      bool hoisted = false;
      // Hoist loop-invariant scratch storage to the owning function's entry
      // when its shape is read directly from a function argument.  Besides
      // amortizing cudaMalloc across repeated calls, this avoids leaving a
      // device allocation in a CFG loop after SCF lowering.
      if (allocationFunction) {
        OpBuilder entryBuilder =
            OpBuilder::atBlockBegin(&allocationFunction.front());
        SmallVector<Value> entrySizes;
        bool canHoist = true;
        for (Value size : dynamicSizes) {
          auto dim = size.getDefiningOp<memref::DimOp>();
          std::optional<int64_t> constantDimension =
              dim ? dim.getConstantIndex() : std::nullopt;
          auto sourceArgument =
              dim ? dyn_cast<BlockArgument>(dim.getSource()) : BlockArgument();
          if (!dim || !constantDimension || !sourceArgument ||
              sourceArgument.getOwner() != &allocationFunction.front()) {
            canHoist = false;
            break;
          }
          entrySizes.push_back(entryBuilder.create<memref::DimOp>(
              dim.getLoc(), dim.getSource(), *constantDimension));
        }
        if (canHoist) {
          allocationBuilder = entryBuilder;
          dynamicSizes = std::move(entrySizes);
          hoisted = true;
        }
      }
      Type localTokenType = gpu::AsyncTokenType::get(module.getContext());
      Value allocationStream =
          allocationBuilder
              .create<gpu::WaitOp>(allocation.getLoc(), localTokenType,
                                   ValueRange{})
              .getAsyncToken();
      auto deviceAllocation = allocationBuilder.create<gpu::AllocOp>(
          allocation.getLoc(), allocation.getType(), localTokenType,
          /*asyncDependencies=*/ValueRange{allocationStream},
          dynamicSizes, allocation.getSymbolOperands());
      allocationBuilder.setInsertionPointAfter(deviceAllocation);
      allocationBuilder.create<gpu::WaitOp>(
          allocation.getLoc(), /*asyncToken=*/Type(),
          ValueRange{deviceAllocation.getAsyncToken()});
      allocation.getResult().replaceAllUsesWith(deviceAllocation.getMemref());
      allocation.erase();
      for (gpu::HostRegisterOp registration : registrations)
        if (registration && registration->getBlock())
          registration.erase();
      for (memref::DeallocOp deallocation : deallocations) {
        OpBuilder deallocationBuilder(deallocation);
        deallocationBuilder.create<gpu::DeallocOp>(
            deallocation.getLoc(), /*asyncTokenType=*/Type(),
            /*asyncDependencies=*/ValueRange{},
            deviceAllocation.getMemref());
        deallocation.erase();
      }
      if (deallocations.empty() && hoisted) {
        SmallVector<func::ReturnOp> allocationReturns;
        allocationFunction.walk(
            [&](func::ReturnOp ret) { allocationReturns.push_back(ret); });
        for (func::ReturnOp ret : allocationReturns) {
          OpBuilder returnBuilder(ret);
          returnBuilder.create<gpu::DeallocOp>(
              ret.getLoc(), /*asyncTokenType=*/Type(),
              /*asyncDependencies=*/ValueRange{},
              deviceAllocation.getMemref());
        }
      }
    }

    struct Promotion {
      BlockArgument host;
      Value device;
    };
    SmallVector<Promotion> promotions;
    SmallVector<gpu::HostRegisterOp> obsoleteRegistrations;

    Block &entry = function.front();
    OpBuilder builder = OpBuilder::atBlockBegin(&entry);
    Location loc = function.getLoc();
    Type asyncTokenType = gpu::AsyncTokenType::get(module.getContext());
    Value entryStream;
    unsigned memrefArgumentCount =
        llvm::count_if(entry.getArguments(), [](BlockArgument argument) {
          return isa<MemRefType>(argument.getType());
        });

    for (BlockArgument argument : entry.getArguments()) {
      if (!promoteFunctionArguments)
        break;
      auto type = dyn_cast<MemRefType>(argument.getType());
      if (!type || !type.getLayout().isIdentity() ||
          !hasOnlyDeviceOrDescriptorUses(argument))
        continue;

      // Distinct C pointer arguments may overlap. Independent device shadows
      // would then destroy the program's aliasing semantics, so require the
      // usual C/LLVM noalias proof unless the pipeline explicitly supplies
      // the equivalent whole-function guarantee.
      if (memrefArgumentCount > 1 && !assumeNoAlias &&
          !function.getArgAttr(argument.getArgNumber(), "llvm.noalias"))
        continue;

      SmallVector<Value> dynamicSizes;
      SmallVector<Operation *> hostDimensionOps;
      for (auto [dim, size] : llvm::enumerate(type.getShape()))
        if (ShapedType::isDynamic(size)) {
          auto dimension = builder.create<memref::DimOp>(loc, argument, dim);
          dynamicSizes.push_back(dimension);
          hostDimensionOps.push_back(dimension);
        }

      if (!entryStream)
        entryStream =
            builder.create<gpu::WaitOp>(loc, asyncTokenType, ValueRange{})
                .getAsyncToken();
      auto allocation = builder.create<gpu::AllocOp>(
          loc, type, asyncTokenType, ValueRange{entryStream}, dynamicSizes,
          /*symbolOperands=*/ValueRange{});
      Value device = allocation.getMemref();
      entryStream = allocation.getAsyncToken();

      // Collect registrations before rewiring; a device allocation must never
      // be passed to gpu.host_register.
      SmallVector<Value> worklist{argument};
      llvm::DenseSet<Value> visited;
      while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        if (!visited.insert(value).second)
          continue;
        for (Operation *user : value.getUsers()) {
          if (auto registration = dyn_cast<gpu::HostRegisterOp>(user))
            obsoleteRegistrations.push_back(registration);
          else if (isDescriptorOnlyUse(user))
            worklist.append(user->getResults().begin(),
                            user->getResults().end());
          else if (auto call = dyn_cast<func::CallOp>(user)) {
            if (func::FuncOp callee = getOwnedCallee(call)) {
              for (OpOperand &operand : call->getOpOperands())
                if (operand.get() == value &&
                    operand.getOperandNumber() < callee.getNumArguments())
                  worklist.push_back(
                      callee.getArgument(operand.getOperandNumber()));
            }
          }
        }
      }

      // Rewire pre-existing computation first. The boundary memcpy is created
      // afterwards so its host operand is not accidentally replaced.
      argument.replaceUsesWithIf(device, [&](OpOperand &use) {
        return !llvm::is_contained(hostDimensionOps, use.getOwner());
      });
      auto copy = builder.create<gpu::MemcpyOp>(
          loc, asyncTokenType, ValueRange{entryStream}, device, argument);
      entryStream = copy.getAsyncToken();
      promotions.push_back({argument, device});
    }

    for (gpu::HostRegisterOp registration : obsoleteRegistrations)
      if (registration && registration->getBlock())
        registration.erase();

    if (promotions.empty())
      return;

    builder.create<gpu::WaitOp>(loc, /*asyncToken=*/Type(),
                                ValueRange{entryStream});

    SmallVector<func::ReturnOp> returns;
    function.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
    for (func::ReturnOp ret : returns) {
      OpBuilder exitBuilder(ret);
      Value exitStream =
          exitBuilder
              .create<gpu::WaitOp>(ret.getLoc(), asyncTokenType, ValueRange{})
              .getAsyncToken();
      for (Promotion &promotion : promotions) {
        auto copy = exitBuilder.create<gpu::MemcpyOp>(
            ret.getLoc(), asyncTokenType, ValueRange{exitStream},
            promotion.host, promotion.device);
        exitStream = copy.getAsyncToken();
        auto deallocation = exitBuilder.create<gpu::DeallocOp>(
            ret.getLoc(), asyncTokenType, ValueRange{exitStream},
            promotion.device);
        exitStream = deallocation.getAsyncToken();
      }
      exitBuilder.create<gpu::WaitOp>(ret.getLoc(), /*asyncToken=*/Type(),
                                      ValueRange{exitStream});
    }
    function->setAttr("polygeist.gpu_data_residency",
                      UnitAttr::get(module.getContext()));
  }
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createPlanGpuDataResidencyPass() {
  return std::make_unique<PlanGpuDataResidencyPass>();
}
} // namespace polygeist
} // namespace mlir

//===- InstrumentGpuRegionTiming.cpp - aggregate GPU timing -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Passes/Passes.h"

#include "llvm/ADT/DenseSet.h"

#include <cstdint>

using namespace mlir;

namespace {

enum TimingCategory : int32_t {
  Host = 0,
  Compute = 1,
  Alloc = 2,
  Free = 3,
  HostToDevice = 4,
  DeviceToHost = 5,
  DeviceToDevice = 6,
  HostToHost = 7,
};

static bool isRuntimeDispatch(func::CallOp call) {
  StringRef callee = call.getCallee();
  if (callee.starts_with("polygeist_gpu_region_timing_") ||
      callee.starts_with("polygeist_cublas_pipeline_") ||
      callee.starts_with("polygeist_cuda_graph_") ||
      callee.starts_with("polygeist_cublas_time_"))
    return false;
  if (callee.starts_with("polygeist_cublas_") ||
      callee.starts_with("polygeist_cublaslt_") ||
      callee.starts_with("polygeist_cudnn_") ||
      callee.starts_with("polygeist_cub_") ||
      callee.starts_with("polygeist_cufft_") ||
      callee.starts_with("polygeist_cusolver_") ||
      callee.starts_with("polygeist_cusparse_") ||
      callee.starts_with("polygeist_cutensor_") ||
      callee.starts_with("polygeist_cutensornet_") ||
      callee.starts_with("polygeist_cuda_") ||
      callee.starts_with("polygeist_rmsnorm_") ||
      callee.starts_with("polygeist_whisper_"))
    return true;

  auto calleeOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  return calleeOp && calleeOp->hasAttr("polygeist.gpu_resident_callee");
}

static bool isDispatch(Operation *op) {
  if (isa<gpu::LaunchOp, gpu::LaunchFuncOp>(op))
    return true;
  if (auto call = dyn_cast<func::CallOp>(op))
    return isRuntimeDispatch(call);
  return false;
}

static bool containsDispatch(Operation *root) {
  bool found = false;
  root->walk([&](Operation *op) {
    if (!isDispatch(op))
      return WalkResult::advance();
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

static bool isDevicePreservingView(Operation *op) {
  return isa<memref::CastOp, memref::SubViewOp, memref::ReinterpretCastOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp>(op);
}

static bool isDeviceValue(Value value, llvm::DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (value.getDefiningOp<gpu::AllocOp>())
    return true;
  Operation *owner = value.getDefiningOp();
  if (!owner || !isDevicePreservingView(owner))
    return false;
  for (Value operand : owner->getOperands())
    if (isa<BaseMemRefType>(operand.getType()) &&
        isDeviceValue(operand, visited))
      return true;
  return false;
}

static bool isDeviceValue(Value value) {
  llvm::DenseSet<Value> visited;
  return isDeviceValue(value, visited);
}

static TimingCategory memcpyCategory(gpu::MemcpyOp copy) {
  bool destinationDevice = isDeviceValue(copy.getDst());
  bool sourceDevice = isDeviceValue(copy.getSrc());
  if (destinationDevice && sourceDevice)
    return DeviceToDevice;
  if (destinationDevice)
    return HostToDevice;
  if (sourceDevice)
    return DeviceToHost;
  return HostToHost;
}

static uint64_t stableRegionId(StringRef name) {
  // FNV-1a is deliberately reproduced here rather than using llvm::hash_value,
  // whose result is not a persistent file-format identifier.
  uint64_t hash = UINT64_C(14695981039346656037);
  for (unsigned char byte : name.bytes()) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static func::FuncOp ensureDeclaration(ModuleOp module, StringRef symbol,
                                      FunctionType type, OpBuilder &builder) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(symbol))
    return existing;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto declaration =
      builder.create<func::FuncOp>(module.getLoc(), symbol, type);
  declaration.setPrivate();
  return declaration;
}

struct InstrumentGpuRegionTimingPass
    : public mlir::polygeist::InstrumentGpuRegionTimingBase<
          InstrumentGpuRegionTimingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    SmallVector<func::FuncOp> functions;
    module.walk([&](func::FuncOp function) {
      if (function.isDeclaration() ||
          function->hasAttr("polygeist.gpu_timing_instrumented"))
        return;
      if (!functionName.empty() && function.getName() != functionName)
        return;
      if (functionName.empty() &&
          !function->hasAttr("polygeist.gpu_data_residency"))
        return;
      functions.push_back(function);
    });

    if (!functionName.empty() && functions.empty()) {
      module.emitError() << "GPU timing function not found: " << functionName;
      return signalPassFailure();
    }
    if (functions.empty())
      return;

    auto begin = ensureDeclaration(
        module, "polygeist_gpu_region_timing_begin",
        builder.getFunctionType({builder.getI64Type()}, {}), builder);
    auto enter = ensureDeclaration(
        module, "polygeist_gpu_region_timing_enter",
        builder.getFunctionType({builder.getI32Type()}, {}), builder);
    auto leave = ensureDeclaration(module, "polygeist_gpu_region_timing_leave",
                                   builder.getFunctionType({}, {}), builder);
    auto end = ensureDeclaration(module, "polygeist_gpu_region_timing_end",
                                 builder.getFunctionType({}, {}), builder);

    for (func::FuncOp function : functions)
      instrumentFunction(function, begin, enter, leave, end);
  }

  void instrumentFunction(func::FuncOp function, func::FuncOp begin,
                          func::FuncOp enter, func::FuncOp leave,
                          func::FuncOp end) {
    bool hasDispatch = false;
    function.walk([&](Operation *op) { hasDispatch |= isDispatch(op); });
    if (!hasDispatch)
      return;

    SmallVector<std::pair<Operation *, Operation *>> spans;
    for (Block &block : function.getBody()) {
      Operation *first = nullptr;
      Operation *last = nullptr;
      for (Operation &op : block) {
        if (!containsDispatch(&op))
          continue;
        first = first ? first : &op;
        last = &op;
      }
      if (!first)
        continue;
      spans.push_back({first, last});
    }

    SmallVector<std::pair<Operation *, TimingCategory>> memoryOperations;
    function.walk([&](Operation *op) {
      if (isa<gpu::AllocOp>(op))
        memoryOperations.push_back({op, Alloc});
      else if (isa<gpu::DeallocOp>(op))
        memoryOperations.push_back({op, Free});
      else if (auto copy = dyn_cast<gpu::MemcpyOp>(op))
        memoryOperations.push_back({op, memcpyCategory(copy)});
    });

    Location loc = function.getLoc();
    OpBuilder entryBuilder = OpBuilder::atBlockBegin(&function.front());
    uint64_t regionId = stableRegionId(function.getName());
    Value id = entryBuilder.create<arith::ConstantIntOp>(
        loc, static_cast<int64_t>(regionId), 64);
    entryBuilder.create<func::CallOp>(loc, begin, ValueRange{id});

    for (auto [first, last] : spans) {
      OpBuilder before(first);
      Value category = before.create<arith::ConstantIntOp>(
          first->getLoc(), static_cast<int32_t>(Compute), 32);
      before.create<func::CallOp>(first->getLoc(), enter, ValueRange{category});
      OpBuilder after(last);
      after.setInsertionPointAfter(last);
      after.create<func::CallOp>(last->getLoc(), leave, ValueRange{});
    }

    for (auto [operation, timingCategory] : memoryOperations) {
      OpBuilder before(operation);
      Value category = before.create<arith::ConstantIntOp>(
          operation->getLoc(), static_cast<int32_t>(timingCategory), 32);
      before.create<func::CallOp>(operation->getLoc(), enter,
                                  ValueRange{category});
      OpBuilder after(operation);
      after.setInsertionPointAfter(operation);
      after.create<func::CallOp>(operation->getLoc(), leave, ValueRange{});
    }

    SmallVector<func::ReturnOp> returns;
    function.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
    for (func::ReturnOp ret : returns) {
      OpBuilder returnBuilder(ret);
      returnBuilder.create<func::CallOp>(ret.getLoc(), end, ValueRange{});
    }

    function->setAttr("polygeist.gpu_timing_instrumented",
                      UnitAttr::get(function.getContext()));
    function->setAttr(
        "polygeist.gpu_timing_region_id",
        IntegerAttr::get(IntegerType::get(function.getContext(), 64),
                         APInt(64, regionId)));
  }
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createInstrumentGpuRegionTimingPass() {
  return std::make_unique<InstrumentGpuRegionTimingPass>();
}
} // namespace polygeist
} // namespace mlir

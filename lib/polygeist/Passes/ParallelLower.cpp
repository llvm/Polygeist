//===- ParallelLower.cpp - Lower gpu code to triple nested loops ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//

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
#include "mlir/Transforms/Passes.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <mutex>
#include <regex>

#include "RuntimeWrapperUtils.h"

#define DEBUG_TYPE "parallel-lower-opt"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::func;
using namespace polygeist;

namespace {
// The store to load forwarding relies on three conditions:
//
// 1) they need to have mathematically equivalent affine access functions
// (checked after full composition of load/store operands); this implies that
// they access the same single memref element for all iterations of the common
// surrounding loop,
//
// 2) the store op should dominate the load op,
//
// 3) among all op's that satisfy both (1) and (2), the one that postdominates
// all store op's that have a dependence into the load, is provably the last
// writer to the particular memref location being loaded at the load op, and its
// store value can be forwarded to the load. Note that the only dependences
// that are to be considered are those that are satisfied at the block* of the
// innermost common surrounding loop of the <store, load> being considered.
//
// (* A dependence being satisfied at a block: a dependence that is satisfied by
// virtue of the destination operation appearing textually / lexically after
// the source operation within the body of a 'affine.for' operation; thus, a
// dependence is always either satisfied by a loop or by a block).
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - they are sufficient, but not necessary --- since they
// don't reason about loops that are guaranteed to execute at least once or
// multiple sources to forward from.
//
// TODO: more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO: do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
// TODO do not take wrap argument, instead, always wrap and if we will be
// lowering to cpu, remove them before continuing
struct ParallelLower : public ParallelLowerBase<ParallelLower> {
  ParallelLower(bool wrapParallelOps,
                PolygeistGPUStructureMode gpuKernelStructureMode)
      : wrapParallelOps(wrapParallelOps),
        gpuKernelStructureMode(gpuKernelStructureMode) {}
  void runOnOperation() override;
  bool wrapParallelOps;
  PolygeistGPUStructureMode gpuKernelStructureMode;
};
struct ConvertCudaRTtoCPU : public ConvertCudaRTtoCPUBase<ConvertCudaRTtoCPU> {
  void runOnOperation() override;
};
struct ConvertCudaRTtoGPU : public ConvertCudaRTtoGPUBase<ConvertCudaRTtoGPU> {
  void runOnOperation() override;
};
struct ConvertCudaRTtoHipRT
    : public ConvertCudaRTtoHipRTBase<ConvertCudaRTtoHipRT> {
  void runOnOperation() override;
};
struct FixGPUFunc : public FixGPUFuncBase<FixGPUFunc> {
  void runOnOperation() override;
};

} // end anonymous namespace

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createConvertCudaRTtoGPUPass() {
  return std::make_unique<ConvertCudaRTtoGPU>();
}
std::unique_ptr<Pass> createConvertCudaRTtoHipRTPass() {
  return std::make_unique<ConvertCudaRTtoHipRT>();
}
std::unique_ptr<Pass> createConvertCudaRTtoCPUPass() {
  return std::make_unique<ConvertCudaRTtoCPU>();
}
std::unique_ptr<Pass>
createParallelLowerPass(bool wrapParallelOps,
                        PolygeistGPUStructureMode gpuKernelStructureMode) {
  return std::make_unique<ParallelLower>(wrapParallelOps,
                                         gpuKernelStructureMode);
}
std::unique_ptr<Pass> createFixGPUFuncPass() {
  return std::make_unique<FixGPUFunc>();
}
} // namespace polygeist
} // namespace mlir

#include "AlwaysInliner.h"

// TODO
mlir::Value callMalloc(mlir::OpBuilder &ibuilder, mlir::ModuleOp module,
                       mlir::Location loc, mlir::Value arg) {
  static std::mutex _mutex;
  std::unique_lock<std::mutex> lock(_mutex);

  mlir::OpBuilder builder(module.getContext());
  SymbolTableCollection symbolTable;
  std::vector args = {arg};
  if (auto fn = dyn_cast_or_null<func::FuncOp>(symbolTable.lookupSymbolIn(
          module, builder.getStringAttr("malloc")))) {
    return ibuilder.create<mlir::func::CallOp>(loc, fn, args)->getResult(0);
  }
  if (!dyn_cast_or_null<LLVM::LLVMFuncOp>(symbolTable.lookupSymbolIn(
          module, builder.getStringAttr("malloc")))) {
    auto *ctx = module->getContext();
    mlir::Type types[] = {mlir::IntegerType::get(ctx, 64)};
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(mlir::IntegerType::get(ctx, 8)), types,
        false);

    LLVM::Linkage lnk = LLVM::Linkage::External;
    builder.setInsertionPointToStart(module.getBody());
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "malloc", llvmFnType,
                                     lnk);
  }

  auto fn = cast<LLVM::LLVMFuncOp>(
      symbolTable.lookupSymbolIn(module, builder.getStringAttr("malloc")));
  return ibuilder.create<mlir::LLVM::CallOp>(loc, fn, args)->getResult(0);
}
mlir::LLVM::LLVMFuncOp GetOrCreateFreeFunction(ModuleOp module) {
  static std::mutex _mutex;
  std::unique_lock<std::mutex> lock(_mutex);

  mlir::OpBuilder builder(module.getContext());
  SymbolTableCollection symbolTable;
  if (auto fn = dyn_cast_or_null<LLVM::LLVMFuncOp>(
          symbolTable.lookupSymbolIn(module, builder.getStringAttr("free"))))
    return fn;
  auto *ctx = module->getContext();
  auto llvmFnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(ctx),
      ArrayRef<mlir::Type>(LLVM::LLVMPointerType::get(builder.getContext())),
      false);

  LLVM::Linkage lnk = LLVM::Linkage::External;
  builder.setInsertionPointToStart(module.getBody());
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "free", llvmFnType,
                                          lnk);
}

LogicalResult fixupGetFunc(LLVM::CallOp, OpBuilder &rewriter,
                           SmallVectorImpl<Value> &);

void ParallelLower::runOnOperation() {
  // The inliner should only be run on operations that define a symbol table,
  // as the callgraph will need to resolve references.

  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());

  std::function<void(LLVM::CallOp)> LLVMcallInliner;
  std::function<void(CallOp)> callInliner = [&](CallOp caller) {
    // Build the inliner interface.
    AlwaysInlinerInterface interface(&getContext());

    auto callable = caller.getCallableForCallee();
    CallableOpInterface callableOp;
    if (SymbolRefAttr symRef = dyn_cast<SymbolRefAttr>(callable)) {
      if (!symRef.isa<FlatSymbolRefAttr>())
        return;
      auto *symbolOp =
          symbolTable.lookupNearestSymbolFrom(getOperation(), symRef);
      callableOp = dyn_cast_or_null<CallableOpInterface>(symbolOp);
    } else {
      return;
    }
    Region *targetRegion = callableOp.getCallableRegion();
    if (!targetRegion)
      return;
    if (targetRegion->empty())
      return;
    {
      SmallVector<CallOp> ops;
      callableOp.walk([&](CallOp caller) { ops.push_back(caller); });
      for (auto op : ops)
        callInliner(op);
    }
    {
      SmallVector<LLVM::CallOp> ops;
      callableOp.walk([&](LLVM::CallOp caller) { ops.push_back(caller); });
      for (auto op : ops)
        LLVMcallInliner(op);
    }
    OpBuilder b(caller);
    auto allocScope = b.create<memref::AllocaScopeOp>(caller.getLoc(),
                                                      caller.getResultTypes());
    allocScope.getRegion().push_back(new Block());
    b.setInsertionPointToStart(&allocScope.getRegion().front());
    auto exOp = b.create<scf::ExecuteRegionOp>(caller.getLoc(),
                                               caller.getResultTypes());
    Block *blk = new Block();
    exOp.getRegion().push_back(blk);
    caller->moveBefore(blk, blk->begin());
    caller.replaceAllUsesWith(allocScope.getResults());
    b.setInsertionPointToEnd(blk);
    b.create<scf::YieldOp>(caller.getLoc(), caller.getResults());
    if (inlineCall(interface, caller, callableOp, targetRegion,
                   /*shouldCloneInlinedRegion=*/true)
            .succeeded()) {
      caller.erase();
    }
    b.setInsertionPointToEnd(&allocScope.getRegion().front());
    b.create<memref::AllocaScopeReturnOp>(allocScope.getLoc(),
                                          exOp.getResults());
  };
  LLVMcallInliner = [&](LLVM::CallOp caller) {
    // Build the inliner interface.
    AlwaysInlinerInterface interface(&getContext());

    auto callable = caller.getCallableForCallee();
    CallableOpInterface callableOp;
    if (SymbolRefAttr symRef = dyn_cast<SymbolRefAttr>(callable)) {
      if (!symRef.isa<FlatSymbolRefAttr>())
        return;
      auto *symbolOp =
          symbolTable.lookupNearestSymbolFrom(getOperation(), symRef);
      callableOp = dyn_cast_or_null<CallableOpInterface>(symbolOp);
    } else {
      return;
    }
    Region *targetRegion = callableOp.getCallableRegion();
    if (!targetRegion)
      return;
    if (targetRegion->empty())
      return;
    {
      SmallVector<CallOp> ops;
      callableOp.walk([&](CallOp caller) { ops.push_back(caller); });
      for (auto op : ops)
        callInliner(op);
    }
    {
      SmallVector<LLVM::CallOp> ops;
      callableOp.walk([&](LLVM::CallOp caller) { ops.push_back(caller); });
      for (auto op : ops)
        LLVMcallInliner(op);
    }
    OpBuilder b(caller);
    auto allocScope = b.create<memref::AllocaScopeOp>(caller.getLoc(),
                                                      caller.getResultTypes());
    allocScope.getRegion().push_back(new Block());
    b.setInsertionPointToStart(&allocScope.getRegion().front());
    auto exOp = b.create<scf::ExecuteRegionOp>(caller.getLoc(),
                                               caller.getResultTypes());
    Block *blk = new Block();
    exOp.getRegion().push_back(blk);
    caller->moveBefore(blk, blk->begin());
    caller.replaceAllUsesWith(allocScope.getResults());
    b.setInsertionPointToEnd(blk);
    b.create<scf::YieldOp>(caller.getLoc(), caller.getResults());
    if (inlineCall(interface, caller, callableOp, targetRegion,
                   /*shouldCloneInlinedRegion=*/true)
            .succeeded()) {
      caller.erase();
    }
    b.setInsertionPointToEnd(&allocScope.getRegion().front());
    b.create<memref::AllocaScopeReturnOp>(allocScope.getLoc(),
                                          exOp.getResults());
  };

  {
    SmallVector<CallOp> dimsToInline;
    getOperation()->walk([&](CallOp bidx) {
      if (bidx.getCallee() == "_ZN4dim3C1EOS_" ||
          bidx.getCallee() == "_ZN4dim3C1Ejjj")
        dimsToInline.push_back(bidx);
    });
    for (auto op : dimsToInline)
      callInliner(op);
  }

  {

    SmallVector<Operation *> inlineOps;
    SmallVector<mlir::Value> toFollowOps;
    SetVector<FunctionOpInterface> toinl;

    getOperation().walk(
        [&](mlir::gpu::ThreadIdOp bidx) { inlineOps.push_back(bidx); });
    getOperation().walk(
        [&](mlir::gpu::GridDimOp bidx) { inlineOps.push_back(bidx); });
    getOperation().walk(
        [&](mlir::NVVM::Barrier0Op bidx) { inlineOps.push_back(bidx); });

    SymbolUserMap symbolUserMap(symbolTable, getOperation());
    while (inlineOps.size()) {
      auto op = inlineOps.back();
      inlineOps.pop_back();
      auto lop = op->getParentOfType<gpu::LaunchOp>();
      auto fop = op->getParentOfType<FunctionOpInterface>();
      if (!lop || lop->isAncestor(fop)) {
        toinl.insert(fop);
        for (Operation *m : symbolUserMap.getUsers(fop)) {
          if (isa<LLVM::CallOp, func::CallOp>(m))
            inlineOps.push_back(m);
          else if (isa<polygeist::GetFuncOp>(m)) {
            toFollowOps.push_back(m->getResult(0));
          }
        }
      }
    }
    for (auto F : toinl) {
      SmallVector<LLVM::CallOp> ltoinl;
      SmallVector<func::CallOp> mtoinl;
      SymbolUserMap symbolUserMap(symbolTable, getOperation());
      for (Operation *m : symbolUserMap.getUsers(F)) {
        if (auto l = dyn_cast<LLVM::CallOp>(m))
          ltoinl.push_back(l);
        else if (auto mc = dyn_cast<func::CallOp>(m))
          mtoinl.push_back(mc);
      }
      for (auto l : ltoinl) {
        LLVMcallInliner(l);
      }
      for (auto m : mtoinl) {
        callInliner(m);
      }
    }
    while (toFollowOps.size()) {
      auto op = toFollowOps.back();
      toFollowOps.pop_back();
      SmallVector<LLVM::CallOp> ltoinl;
      SmallVector<func::CallOp> mtoinl;
      bool inlined = false;
      for (auto u : op.getUsers()) {
        if (auto cop = dyn_cast<LLVM::CallOp>(u)) {
          if (!cop.getCallee() && cop->getOperand(0) == op) {
            OpBuilder builder(cop);
            SmallVector<Value> vals;
            if (fixupGetFunc(cop, builder, vals).succeeded()) {
              if (vals.size())
                cop.getResult().replaceAllUsesWith(vals[0]);
              cop.erase();
              inlined = true;
              break;
            }
          } else if (cop.getCallee())
            ltoinl.push_back(cop);
        } else if (auto cop = dyn_cast<func::CallOp>(u)) {
          mtoinl.push_back(cop);
        } else {
          for (auto r : u->getResults())
            toFollowOps.push_back(r);
        }
      }
      for (auto l : ltoinl) {
        LLVMcallInliner(l);
        inlined = true;
      }
      for (auto m : mtoinl) {
        callInliner(m);
        inlined = true;
      }
      if (inlined)
        toFollowOps.push_back(op);
    }
  }

  // Only supports single block functions at the moment.

  SmallVector<gpu::LaunchOp> toHandle;
  getOperation().walk(
      [&](gpu::LaunchOp launchOp) { toHandle.push_back(launchOp); });
  for (gpu::LaunchOp launchOp : toHandle) {
    {
      SmallVector<CallOp> ops;
      launchOp.walk([&](CallOp caller) { ops.push_back(caller); });
      for (auto op : ops)
        callInliner(op);
    }
    {
      SmallVector<LLVM::CallOp> lops;
      launchOp.walk([&](LLVM::CallOp caller) { lops.push_back(caller); });
      for (auto op : lops)
        LLVMcallInliner(op);
    }

    mlir::IRRewriter builder(launchOp.getContext());
    auto loc = launchOp.getLoc();

    builder.setInsertionPoint(launchOp->getBlock(), launchOp->getIterator());
    auto zindex = builder.create<ConstantIndexOp>(loc, 0);

    auto oneindex = builder.create<ConstantIndexOp>(loc, 1);

    async::ExecuteOp asyncOp = nullptr;
    if (!launchOp.getAsyncDependencies().empty()) {
      SmallVector<Value> dependencies;
      for (auto v : launchOp.getAsyncDependencies()) {
        auto tok = v.getDefiningOp<polygeist::StreamToTokenOp>();
        dependencies.push_back(builder.create<polygeist::StreamToTokenOp>(
            tok.getLoc(), builder.getType<async::TokenType>(),
            tok.getSource()));
      }
      asyncOp = builder.create<mlir::async::ExecuteOp>(
          loc, /*results*/ TypeRange(), /*dependencies*/ dependencies,
          /*operands*/ ValueRange());
      Block *blockB = asyncOp.getBody();
      builder.setInsertionPointToStart(blockB);
    }

    if (wrapParallelOps) {
      auto pw = builder.create<polygeist::GPUWrapperOp>(
          loc,
          ValueRange({launchOp.getGridSizeX(), launchOp.getGridSizeY(),
                      launchOp.getGridSizeZ(), launchOp.getBlockSizeX(),
                      launchOp.getBlockSizeY(), launchOp.getBlockSizeZ()}));
      builder.setInsertionPointToStart(pw.getBody());
    }

    auto block = builder.create<mlir::scf::ParallelOp>(
        loc, std::vector<Value>({zindex, zindex, zindex}),
        std::vector<Value>({launchOp.getGridSizeX(), launchOp.getGridSizeY(),
                            launchOp.getGridSizeZ()}),
        std::vector<Value>({oneindex, oneindex, oneindex}));
    Block *blockB = &block.getRegion().front();
    builder.setInsertionPointToStart(blockB);

    if (gpuKernelStructureMode == PGSM_BlockThreadWrappers) {
      auto gpuBlock = builder.create<polygeist::GPUBlockOp>(
          loc, blockB->getArguments()[0], blockB->getArguments()[1],
          blockB->getArguments()[2]);
      builder.setInsertionPointToStart(&gpuBlock.getRegion().front());
    } else if (gpuKernelStructureMode == PGSM_BlockThreadNoops) {
      auto noop = builder.create<polygeist::NoopOp>(
          loc, ValueRange({blockB->getArguments()[0], blockB->getArguments()[1],
                           blockB->getArguments()[2]}));
      noop->setAttr("polygeist.noop_type",
                    StringAttr::get(noop->getContext(), "gpu_kernel.block"));
    }

    auto threadr = builder.create<mlir::scf::ParallelOp>(
        loc, std::vector<Value>({zindex, zindex, zindex}),
        std::vector<Value>({launchOp.getBlockSizeX(), launchOp.getBlockSizeY(),
                            launchOp.getBlockSizeZ()}),
        std::vector<Value>({oneindex, oneindex, oneindex}));
    Block *threadB = &threadr.getRegion().front();
    builder.setInsertionPointToStart(threadB);
    Operation *mergeLoc = threadB->getTerminator();

    if (gpuKernelStructureMode == PGSM_BlockThreadWrappers) {
      auto gpuThread = builder.create<polygeist::GPUThreadOp>(
          loc, threadB->getArguments()[0], threadB->getArguments()[1],
          threadB->getArguments()[2]);
      builder.setInsertionPointToStart(&gpuThread.getRegion().front());
      mergeLoc = gpuThread.getRegion().front().getTerminator();
    } else if (gpuKernelStructureMode == PGSM_BlockThreadNoops ||
               gpuKernelStructureMode == PGSM_ThreadNoop) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(mergeLoc);
      auto noop = builder.create<polygeist::NoopOp>(
          loc,
          ValueRange({threadB->getArguments()[0], threadB->getArguments()[1],
                      threadB->getArguments()[2]}));
      if (gpuKernelStructureMode == PGSM_BlockThreadNoops) {
        noop->setAttr("polygeist.noop_type",
                      StringAttr::get(noop->getContext(), "gpu_kernel.thread"));
      } else if (gpuKernelStructureMode == PGSM_ThreadNoop) {
        noop->setAttr(
            "polygeist.noop_type",
            StringAttr::get(noop->getContext(), "gpu_kernel.thread_only"));
      }
    }
    launchOp.getRegion().front().getTerminator()->erase();

    SmallVector<Value> launchArgs;
    llvm::append_range(launchArgs, blockB->getArguments());
    llvm::append_range(launchArgs, threadB->getArguments());
    launchArgs.push_back(launchOp.getGridSizeX());
    launchArgs.push_back(launchOp.getGridSizeY());
    launchArgs.push_back(launchOp.getGridSizeZ());
    launchArgs.push_back(launchOp.getBlockSizeX());
    launchArgs.push_back(launchOp.getBlockSizeY());
    launchArgs.push_back(launchOp.getBlockSizeZ());
    builder.inlineBlockBefore(&launchOp.getRegion().front(), mergeLoc,
                              launchArgs);

    auto container = threadr;

    container.walk([&](mlir::gpu::BlockIdOp bidx) {
      int idx = -1;
      if (bidx.getDimension() == gpu::Dimension::x)
        idx = 0;
      else if (bidx.getDimension() == gpu::Dimension::y)
        idx = 1;
      else if (bidx.getDimension() == gpu::Dimension::z)
        idx = 2;
      else
        llvm_unreachable("illegal dimension");
      builder.replaceOp(bidx,
                        ValueRange((mlir::Value)blockB->getArgument(idx)));
    });

    container.walk([&](mlir::memref::AllocaOp alop) {
      if (auto ia =
              dyn_cast_or_null<IntegerAttr>(alop.getType().getMemorySpace()))
        if (ia.getValue() == 5) {
          builder.setInsertionPointToStart(blockB);
          auto newAlloca = builder.create<memref::AllocaOp>(
              alop.getLoc(),
              MemRefType::get(alop.getType().getShape(),
                              alop.getType().getElementType(),
                              alop.getType().getLayout(), Attribute()));
          builder.replaceOpWithNewOp<memref::CastOp>(alop, alop.getType(),
                                                     newAlloca);
        }
    });

    container.walk([&](mlir::LLVM::AllocaOp alop) {
      auto PT = alop.getType().cast<LLVM::LLVMPointerType>();
      if (PT.getAddressSpace() == 5) {
        builder.setInsertionPointToStart(blockB);
        auto newAlloca = builder.create<LLVM::AllocaOp>(
            alop.getLoc(), LLVM::LLVMPointerType::get(PT.getElementType(), 0),
            alop.getArraySize());
        builder.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(alop, PT, newAlloca);
      }
    });

    // If we are compiling for GPU
    if (gpuKernelStructureMode != PGSM_Discard) {
      // Tag device side get globals with an attribute so that CSE does not
      // decide to reuse the host side get global for the device
      std::vector<mlir::memref::GetGlobalOp> ggops;
      container.walk([&](mlir::memref::GetGlobalOp getGlobalOp) {
        ggops.push_back(getGlobalOp);
      });
      for (auto ggo : ggops) {
        builder.setInsertionPoint(ggo);
        builder.replaceOp(
            ggo, builder
                     .create<polygeist::GetDeviceGlobalOp>(
                         ggo->getLoc(), ggo.getType(), ggo.getNameAttr())
                     ->getResults());
      }
    }

    container.walk([&](mlir::gpu::ThreadIdOp bidx) {
      int idx = -1;
      if (bidx.getDimension() == gpu::Dimension::x)
        idx = 0;
      else if (bidx.getDimension() == gpu::Dimension::y)
        idx = 1;
      else if (bidx.getDimension() == gpu::Dimension::z)
        idx = 2;
      else
        llvm_unreachable("illegal dimension");
      builder.replaceOp(bidx, ValueRange(threadB->getArgument(idx)));
    });

    container.walk([&](mlir::NVVM::Barrier0Op op) {
      builder.setInsertionPoint(op);
      builder.replaceOpWithNewOp<mlir::polygeist::BarrierOp>(
          op, threadB->getArguments());
    });

    container.walk([&](gpu::GridDimOp bidx) {
      Value val = nullptr;
      if (bidx.getDimension() == gpu::Dimension::x)
        val = launchOp.getGridSizeX();
      else if (bidx.getDimension() == gpu::Dimension::y)
        val = launchOp.getGridSizeY();
      else if (bidx.getDimension() == gpu::Dimension::z)
        val = launchOp.getGridSizeZ();
      else
        llvm_unreachable("illegal dimension");
      builder.replaceOp(bidx, val);
    });

    container.walk([&](gpu::BlockDimOp bidx) {
      Value val = nullptr;
      if (bidx.getDimension() == gpu::Dimension::x)
        val = launchOp.getBlockSizeX();
      else if (bidx.getDimension() == gpu::Dimension::y)
        val = launchOp.getBlockSizeY();
      else if (bidx.getDimension() == gpu::Dimension::z)
        val = launchOp.getBlockSizeZ();
      else
        llvm_unreachable("illegal dimension");
      builder.replaceOp(bidx, val);
    });

    container.walk([&](affine::AffineStoreOp storeOp) {
      builder.setInsertionPoint(storeOp);
      auto map = storeOp.getAffineMap();
      std::vector<Value> indices;
      for (size_t i = 0; i < map.getNumResults(); i++) {
        auto apply = builder.create<affine::AffineApplyOp>(
            storeOp.getLoc(), map.getSliceMap(i, 1), storeOp.getMapOperands());
        indices.push_back(apply->getResult(0));
      }
      builder.replaceOpWithNewOp<memref::StoreOp>(storeOp, storeOp.getValue(),
                                                  storeOp.getMemref(), indices);
    });

    container.walk([&](affine::AffineLoadOp storeOp) {
      builder.setInsertionPoint(storeOp);
      auto map = storeOp.getAffineMap();
      std::vector<Value> indices;
      for (size_t i = 0; i < map.getNumResults(); i++) {
        auto apply = builder.create<affine::AffineApplyOp>(
            storeOp.getLoc(), map.getSliceMap(i, 1), storeOp.getMapOperands());
        indices.push_back(apply->getResult(0));
      }
      builder.replaceOpWithNewOp<memref::LoadOp>(storeOp, storeOp.getMemref(),
                                                 indices);
    });
    builder.eraseOp(launchOp);
  }

  // Fold the copy memtype cast
  {
    mlir::RewritePatternSet rpl(getOperation()->getContext());
    GreedyRewriteConfig config;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
  }
}

void FixGPUFunc::runOnOperation() {

  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());

  std::function<void(CallOp)> callInliner = [&](CallOp caller) {
    // Build the inliner interface.
    AlwaysInlinerInterface interface(&getContext());

    auto callable = caller.getCallableForCallee();
    CallableOpInterface callableOp;
    if (SymbolRefAttr symRef = dyn_cast<SymbolRefAttr>(callable)) {
      auto *symbolOp =
          symbolTable.lookupNearestSymbolFrom(getOperation(), symRef);
      callableOp = dyn_cast_or_null<CallableOpInterface>(symbolOp);
    } else {
      return;
    }
    Region *targetRegion = callableOp.getCallableRegion();
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
  gpu::GPUModuleOp gpum = getOperation();
  auto getDirectlyNestedCallOp = [&](Operation *func) -> func::CallOp {
    if (func->getNumRegions() != 1)
      return nullptr;
    auto &reg = func->getRegion(0);
    auto &blocks = reg.getBlocks();
    if (blocks.size() != 1)
      return nullptr;
    auto block = &blocks.front();
    if (auto callOp = dyn_cast<func::CallOp>(block->front())) {
      if (!callOp->getNextNode()->hasTrait<OpTrait::IsTerminator>())
        return nullptr;
      return callOp;
    } else {
      return nullptr;
    }
  };
  gpum->walk([&](gpu::GPUFuncOp gpuFuncOp) {
    auto callOp = getDirectlyNestedCallOp(gpuFuncOp);
    if (!callOp)
      return;
    Operation *funcOp;
    if (SymbolRefAttr symRef =
            dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee())) {
      auto *symbolOp =
          symbolTable.lookupNearestSymbolFrom(getOperation(), symRef);
      funcOp = dyn_cast_or_null<CallableOpInterface>(symbolOp);
    } else {
      return;
    }
    auto callOp2 = getDirectlyNestedCallOp(funcOp);

    if (callOp2)
      callInliner(callOp2);

    callInliner(callOp);
  });
}

static void replaceCallWithSuccess(Operation *call, OpBuilder &bz) {
  call->replaceAllUsesWith(bz.create<ConstantIntOp>(
      call->getLoc(), 0, call->getResult(0).getType()));
  call->erase();
}

void ConvertCudaRTtoCPU::runOnOperation() {
  // The inliner should only be run on operations that define a symbol table,
  // as the callgraph will need to resolve references.

  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());

  std::function<void(Operation * call, StringRef callee)> replace =
      [&](Operation *call, StringRef callee) {
        if (callee == "cudaMemcpy" || callee == "cudaMemcpyAsync") {
          OpBuilder bz(call);
          auto falsev = bz.create<ConstantIntOp>(call->getLoc(), false, 1);
          auto dst = call->getOperand(0);
          if (auto mt = dyn_cast<MemRefType>(dst.getType())) {
            dst = bz.create<polygeist::Memref2PointerOp>(
                call->getLoc(),
                LLVM::LLVMPointerType::get(mt.getElementType(),
                                           mt.getMemorySpaceAsInt()),
                dst);
          }
          auto src = call->getOperand(1);
          if (auto mt = dyn_cast<MemRefType>(src.getType())) {
            src = bz.create<polygeist::Memref2PointerOp>(
                call->getLoc(),
                LLVM::LLVMPointerType::get(mt.getElementType(),
                                           mt.getMemorySpaceAsInt()),
                src);
          }
          bz.create<LLVM::MemcpyOp>(call->getLoc(), dst, src,
                                    call->getOperand(2),
                                    /*isVolatile*/ falsev);
          call->replaceAllUsesWith(bz.create<ConstantIntOp>(
              call->getLoc(), 0, call->getResult(0).getType()));
          call->erase();
        } else if (callee == "cudaMemcpyToSymbol") {
          OpBuilder bz(call);
          auto falsev = bz.create<ConstantIntOp>(call->getLoc(), false, 1);
          auto dst = call->getOperand(0);
          if (auto mt = dyn_cast<MemRefType>(dst.getType())) {
            dst = bz.create<polygeist::Memref2PointerOp>(
                call->getLoc(),
                LLVM::LLVMPointerType::get(mt.getElementType(),
                                           mt.getMemorySpaceAsInt()),
                dst);
          }
          auto src = call->getOperand(1);
          if (auto mt = dyn_cast<MemRefType>(src.getType())) {
            src = bz.create<polygeist::Memref2PointerOp>(
                call->getLoc(),
                LLVM::LLVMPointerType::get(mt.getElementType(),
                                           mt.getMemorySpaceAsInt()),
                src);
          }
          bz.create<LLVM::MemcpyOp>(
              call->getLoc(),
              bz.create<LLVM::GEPOp>(call->getLoc(), dst.getType(), dst,
                                     std::vector<Value>({call->getOperand(3)})),
              src, call->getOperand(2),
              /*isVolatile*/ falsev);
          call->replaceAllUsesWith(bz.create<ConstantIntOp>(
              call->getLoc(), 0, call->getResult(0).getType()));
          call->erase();
        } else if (callee == "cudaMemset") {
          OpBuilder bz(call);
          auto falsev = bz.create<ConstantIntOp>(call->getLoc(), false, 1);
          auto dst = call->getOperand(0);
          if (auto mt = dyn_cast<MemRefType>(dst.getType())) {
            dst = bz.create<polygeist::Memref2PointerOp>(
                call->getLoc(),
                LLVM::LLVMPointerType::get(mt.getElementType(),
                                           mt.getMemorySpaceAsInt()),
                dst);
          }
          bz.create<LLVM::MemsetOp>(call->getLoc(), dst,
                                    bz.create<TruncIOp>(call->getLoc(),
                                                        bz.getI8Type(),
                                                        call->getOperand(1)),
                                    call->getOperand(2),
                                    /*isVolatile*/ falsev);
          call->replaceAllUsesWith(bz.create<ConstantIntOp>(
              call->getLoc(), 0, call->getResult(0).getType()));
          call->erase();
        } else if (callee == "cudaMalloc" || callee == "cudaMallocHost") {
          OpBuilder bz(call);
          Value arg = call->getOperand(1);
          if (arg.getType().cast<IntegerType>().getWidth() < 64)
            arg =
                bz.create<arith::ExtUIOp>(call->getLoc(), bz.getI64Type(), arg);
          mlir::Value alloc =
              callMalloc(bz, getOperation(), call->getLoc(), arg);
          bz.create<LLVM::StoreOp>(call->getLoc(), alloc, call->getOperand(0));
          {
            auto retv = bz.create<ConstantIntOp>(
                call->getLoc(), 0,
                call->getResult(0).getType().cast<IntegerType>().getWidth());
            Value vals[] = {retv};
            call->replaceAllUsesWith(ArrayRef<Value>(vals));
            call->erase();
          }
        } else if (callee == "cudaFree" || callee == "cudaFreeHost") {
          auto mf = GetOrCreateFreeFunction(getOperation());
          OpBuilder bz(call);
          Value args[] = {call->getOperand(0)};
          bz.create<mlir::LLVM::CallOp>(call->getLoc(), mf, args);
          {
            auto retv = bz.create<ConstantIntOp>(
                call->getLoc(), 0,
                call->getResult(0).getType().cast<IntegerType>().getWidth());
            Value vals[] = {retv};
            call->replaceAllUsesWith(ArrayRef<Value>(vals));
            call->erase();
          }
        } else if (callee == "cudaDeviceSynchronize" ||
                   callee == "cudaThreadSynchronize") {
          // TODO if we have async kernels we must preserve this and lower it to
          // a CPU equivalent
          OpBuilder bz(call);
          auto retv = bz.create<ConstantIntOp>(
              call->getLoc(), 0,
              call->getResult(0).getType().cast<IntegerType>().getWidth());
          Value vals[] = {retv};
          call->replaceAllUsesWith(ArrayRef<Value>(vals));
          call->erase();
        } else if (callee == "cudaGetLastError" ||
                   callee == "cudaPeekAtLastError") {
          OpBuilder bz(call);
          auto retv = bz.create<ConstantIntOp>(
              call->getLoc(), 0,
              call->getResult(0).getType().cast<IntegerType>().getWidth());
          Value vals[] = {retv};
          call->replaceAllUsesWith(ArrayRef<Value>(vals));
          call->erase();
        }
      };

  getOperation().walk([&](LLVM::CallOp call) {
    if (!call.getCallee())
      return;
    replace(call, *call.getCallee());
  });

  getOperation().walk([&](CallOp call) { replace(call, call.getCallee()); });

  // Fold the copy memtype cast
  {
    mlir::RewritePatternSet rpl(getOperation()->getContext());
    GreedyRewriteConfig config;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
  }
}

// Returns a list of all symbols provided by cudart (obtained from
// libcudart_static.a)
static std::vector<llvm::StringRef> getCudartSymbols();
static std::map<llvm::StringRef, llvm::StringRef>
getCudartEquivalentSymbolsMap();

namespace {

bool isCudartCall(StringRef name) {
  static std::vector<llvm::StringRef> sortedCudartSymbols = []() {
    auto tmp = getCudartSymbols();
    std::sort(tmp.begin(), tmp.end());
    return tmp;
  }();
  return std::binary_search(sortedCudartSymbols.begin(),
                            sortedCudartSymbols.end(), name);
}

llvm::StringRef getHipName(StringRef name) {
  return getCudartEquivalentSymbolsMap()[name];
}

bool isHipCallEquivalent(StringRef name) {
  return getCudartEquivalentSymbolsMap().count(name) > 0;
}

} // namespace

static void setCallee(func::CallOp call, StringRef symName) {
  call.setCallee(symName);
}
static void setCallee(LLVM::CallOp call, StringRef symName) {
  call.setCallee(symName);
}
template <typename CallOpTy, typename FuncOpTy>
void replaceCallOp(ModuleOp m, CallOpTy call, llvm::StringRef callee) {
  auto loc = call->getLoc();
  OpBuilder moduleBuilder = OpBuilder::atBlockEnd(m.getBody());
  OpBuilder callBuilder(call);
  auto funcOp = m.lookupSymbol<FuncOpTy>(callee);
  if (isHipCallEquivalent(callee)) {
    assert(funcOp);
    auto hipName = getHipName(callee);
    if (!m.lookupSymbol<FuncOpTy>(hipName)) {
      auto hipFuncOp =
          cast<FuncOpTy>(moduleBuilder.clone(*funcOp.getOperation()));
      hipFuncOp.setSymName(hipName);
    }
    setCallee(call, hipName);
  } else {
    llvm::errs() << "warning: Unsupported CUDART call " << callee
                 << " for conversion to HIP, will be removed instead\n";
    replaceCallWithSuccess(call, callBuilder);
  }
}

void ConvertCudaRTtoHipRT::runOnOperation() {
  getOperation().walk([&](LLVM::CallOp call) {
    if (!call.getCallee())
      return;
    auto name = *call.getCallee();
    if (!isCudartCall(name))
      return;
    replaceCallOp<LLVM::CallOp, LLVM::LLVMFuncOp>(getOperation(), call, name);
  });

  getOperation().walk([&](CallOp call) {
    auto name = call.getCallee();
    if (!isCudartCall(name))
      return;
    replaceCallOp<CallOp, func::FuncOp>(getOperation(), call, name);
  });

  OpBuilder builder(&getContext());
  getOperation().walk([&](mlir::NVVM::Barrier0Op op) {
    builder.setInsertionPoint(op);
    builder.create<mlir::ROCDL::BarrierOp>(op->getLoc());
    op->erase();
  });
}

void ConvertCudaRTtoGPU::runOnOperation() {
  std::function<void(Operation * call, llvm::StringRef callee)> replaceWithOp =
      [&](Operation *call, llvm::StringRef callee) {
        auto loc = call->getLoc();
        OpBuilder bz(call);

        // TODO Functions that would be nice to replace with MLIR ops
        if (callee == "cudaMemcpy") {
        } else if (callee == "cudaMemcpyAsync") {
        } else if (callee == "cudaMemcpyToSymbol") {
        } else if (callee == "cudaMemset") {
        } else if (callee == "cudaMallocHost") {
        } else if (callee == "cudaMalloc") {
        } else if (callee == "cudaFreeHost") {
        } else if (callee == "cudaFree") {
        } else if (callee == "cudaDeviceSynchronize" ||
                   callee == "cudaThreadSynchronize") {
        }
      };
  getOperation().walk([&](LLVM::CallOp call) {
    if (!call.getCallee())
      return;
    auto name = *call.getCallee();
    if (!isCudartCall(name))
      return;
    replaceWithOp(call, name);
  });
  getOperation().walk([&](CallOp call) {
    auto name = call.getCallee();
    if (!isCudartCall(name))
      return;
    replaceWithOp(call, name);
  });
}

// clang-format off
std::vector<llvm::StringRef> cudartSymbols = {
"cudaGetDevice",
"cudaWaitExternalSemaphoresAsync_ptsz",
"cudaStreamAddCallback",
"cudaMemcpyArrayToArray",
"cudaDeviceReset",
"cudaGraphAddEventRecordNode",
"cudaGetSurfaceObjectResourceDesc",
"cudaGraphicsSubResourceGetMappedArray",
"cudaMemRangeGetAttributes",
"cudaGraphAddKernelNode",
"cudaGraphDestroy",
"cudaGraphAddExternalSemaphoresSignalNode",
"cudaGraphExecChildGraphNodeSetParams",
"cudaEGLStreamConsumerReleaseFrame",
"__cudaRegisterManagedVar",
"cudaMemcpy2DFromArray",
"cudaEventRecord_ptsz",
"cudaSetDoubleForHost",
"cudaGraphExternalSemaphoresWaitNodeSetParams",
"cudaMemPoolSetAttribute",
"cudaDeviceFlushGPUDirectRDMAWrites",
"cudaDestroyExternalMemory",
"cudaDeviceGetGraphMemAttribute",
"cudaEGLStreamConsumerConnect",
"cudaGraphUpload",
"cudaDestroyTextureObject",
"cudaHostGetFlags",
"cudaStreamQuery_ptsz",
"cudaHostGetDevicePointer",
"cudaPointerGetAttributes",
"cudaWaitExternalSemaphoresAsync_v2",
"cudaFuncSetAttribute",
"cudaDeviceGetSharedMemConfig",
"cudaGetDeviceFlags",
"cudaGraphGetNodes",
"cudaGraphMemAllocNodeGetParams",
"cudaMemcpy3D",
"cudaMemcpy2DArrayToArray",
"cudaBindTextureToArray",
"cudaDeviceDisablePeerAccess",
"cudaGraphMemsetNodeGetParams",
"cudaGraphExecExternalSemaphoresWaitNodeSetParams",
"cudaGraphNodeGetDependentNodes",
"cudaEventDestroy",
"cudaDeviceCanAccessPeer",
"cudaArrayGetInfo",
"cudaMemcpyAsync",
"cudaStreamEndCapture_ptsz",
"cudaGraphMemFreeNodeGetParams",
"cudaGraphExecMemcpyNodeSetParams1D",
"cudaOccupancyMaxActiveBlocksPerMultiprocessor",
"cudaGraphAddChildGraphNode",
"cudaGraphicsGLRegisterImage",
"cudaGraphExecMemcpyNodeSetParamsToSymbol",
"cudaProfilerInitialize",
"cudaWaitExternalSemaphoresAsync",
"cudaMalloc3DArray",
"cudaGraphKernelNodeSetParams",
"cudaProfilerStart",
"cudaGraphChildGraphNodeGetGraph",
"cudaGetErrorString",
"cudaMemset",
"cudaGraphMemcpyNodeSetParamsFromSymbol",
"cudaMemset3D",
"cudaGraphExecMemcpyNodeSetParamsFromSymbol",
"cudaMemcpyArrayToArray_ptds",
"cudaMemcpy2D",
"cudaGraphDestroyNode",
"cudaStreamWaitEvent",
"cudaMemcpy2DToArrayAsync_ptsz",
"cudaGraphEventRecordNodeGetEvent",
"cudaSetDoubleForDevice",
"cudaLaunchCooperativeKernel_ptsz",
"cudaLaunchKernel",
"cudaFuncSetSharedMemConfig",
"cudaPeekAtLastError",
"cudaMemcpy3DAsync_ptsz",
"cudaEventCreate",
"cudaMemPrefetchAsync_ptsz",
"cudaMalloc",
"cudaMemPoolSetAccess",
"cudaBindTexture2D",
"cudaMemPoolTrimTo",
"cudaThreadGetLimit",
"cudaGraphMemsetNodeSetParams",
"cudaGLRegisterBufferObject",
"cudaGraphicsVDPAURegisterOutputSurface",
"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
"cudaEventCreateFromEGLSync",
"cudaGraphExternalSemaphoresSignalNodeGetParams",
"cudaMemPoolExportPointer",
"cudaGraphNodeFindInClone",
"cudaGetTextureAlignmentOffset",
"cudaSignalExternalSemaphoresAsync_v2_ptsz",
"cudaGraphKernelNodeGetAttribute",
"cudaHostUnregister",
"cudaStreamSetAttribute",
"cudaLaunchHostFunc",
"__cudaRegisterFatBinaryEnd",
"cudaGetTextureObjectResourceDesc",
"cudaGraphExternalSemaphoresSignalNodeSetParams",
"cudaMemPoolImportFromShareableHandle",
"cudaStreamDestroy",
"cudaMalloc3D",
"cudaGLSetGLDevice",
"cudaGraphRetainUserObject",
"cudaGraphExecExternalSemaphoresSignalNodeSetParams",
"cudaMemAdvise",
"cudaEventRecordWithFlags",
"cudaMemcpy3DPeerAsync_ptsz",
"cudaGraphExecMemcpyNodeSetParams",
"cudaProfilerStop",
"cudaFreeMipmappedArray",
"cudaStreamCopyAttributes_ptsz",
"cudaMemcpyFromArray",
"cudaMemcpy3DPeer",
"cudaMemPoolImportPointer",
"cudaMemPoolCreate",
"cudaCreateTextureObject",
"cudaGraphExecDestroy",
"cudaMemGetInfo",
"cudaStreamGetFlags",
"cudaGetMipmappedArrayLevel",
"cudaMemset2DAsync_ptsz",
"cudaMemcpyAsync_ptsz",
"cudaCreateSurfaceObject",
"cudaMemRangeGetAttribute",
"cudaStreamCopyAttributes",
"cudaMemcpyToSymbol",
"cudaMemcpy3D_ptds",
"cudaGLUnregisterBufferObject",
"cudaGraphInstantiate",
"cudaStreamBeginCapture",
"cudaDestroySurfaceObject",
"cudaMemcpy3DAsync",
"cudaFuncGetAttributes",
"cudaStreamIsCapturing_ptsz",
"cudaChooseDevice",
"cudaGraphExecMemsetNodeSetParams",
"cudaArrayGetPlane",
"__cudaPopCallConfiguration",
"cudaThreadSetCacheConfig",
"cudaStreamAttachMemAsync_ptsz",
"cudaGLMapBufferObjectAsync",
"cudaMemcpyFromArrayAsync_ptsz",
"cudaMemcpy2DFromArrayAsync_ptsz",
"cudaMemcpyToArrayAsync_ptsz",
"cudaArrayGetSparseProperties",
"cudaExternalMemoryGetMappedMipmappedArray",
"cudaGraphClone",
"cudaStreamGetPriority_ptsz",
"cudaRuntimeGetVersion",
"cudaMemPoolDestroy",
"cudaGraphMemcpyNodeSetParamsToSymbol",
"cudaGraphExecUpdate",
"cudaEGLStreamConsumerDisconnect",
"cudaGetSymbolAddress",
"__cudaRegisterVar",
"cudaStreamGetCaptureInfo",
"cudaMemcpy3DPeerAsync",
"cudaMemcpyPeer",
"cudaDeviceGetByPCIBusId",
"cudaEGLStreamProducerDisconnect",
"cudaEGLStreamConsumerAcquireFrame",
"__cudaRegisterTexture",
"cudaGraphicsVDPAURegisterVideoSurface",
"cudaDeviceSetCacheConfig",
"cudaMemcpyFromArrayAsync",
"cudaGraphEventRecordNodeSetEvent",
"cudaGraphAddMemcpyNode",
"cudaDeviceGetDefaultMemPool",
"cudaStreamSynchronize_ptsz",
"cudaBindSurfaceToArray",
"cudaMallocAsync",
"cudaGraphGetEdges",
"cudaGetDriverEntryPoint_ptsz",
"cudaGraphMemcpyNodeSetParams1D",
"cudaGraphKernelNodeCopyAttributes",
"cudaVDPAUSetVDPAUDevice",
"cudaDeviceGraphMemTrim",
"cudaGraphicsResourceGetMappedMipmappedArray",
"cudaThreadSynchronize",
"cudaDeviceGetTexture1DLinearMaxWidth",
"cudaDeviceSynchronize",
"cudaMemcpyFromSymbolAsync",
"cudaSetValidDevices",
"cudaOccupancyAvailableDynamicSMemPerBlock",
"cudaStreamSetAttribute_ptsz",
"cudaMemcpyFromSymbol",
"cudaStreamEndCapture",
"cudaImportExternalMemory",
"__cudaRegisterSurface",
"cudaThreadSetLimit",
"cudaGLMapBufferObject",
"cudaBindTextureToMipmappedArray",
"cudaGraphUpload_ptsz",
"cudaGLGetDevices",
"cudaGraphAddMemAllocNode",
"cudaMemsetAsync",
"cudaGLUnmapBufferObjectAsync",
"cudaUserObjectRetain",
"cudaGraphNodeGetDependencies",
"cudaStreamCreateWithPriority",
"cudaStreamGetCaptureInfo_ptsz",
"cudaStreamGetAttribute",
"cudaStreamAttachMemAsync",
"cudaGetDeviceCount",
"cudaMemset3D_ptds",
"cudaFreeAsync",
"cudaUserObjectRelease",
"cudaCreateChannelDesc",
"cudaGetSurfaceReference",
"cudaGetChannelDesc",
"cudaGraphDebugDotPrint",
"cudaEGLStreamProducerPresentFrame",
"cudaEventQuery",
"cudaStreamBeginCapture_ptsz",
"cudaMallocMipmappedArray",
"cudaThreadExchangeStreamCaptureMode",
"cudaStreamGetFlags_ptsz",
"cudaStreamUpdateCaptureDependencies_ptsz",
"cudaGraphicsGLRegisterBuffer",
"cudaDeviceGetNvSciSyncAttributes",
"cudaEGLStreamProducerReturnFrame",
"cudaIpcOpenEventHandle",
"cudaMemPoolGetAccess",
"cudaGraphicsResourceGetMappedPointer",
"cudaMallocFromPoolAsync",
"cudaCtxResetPersistingL2Cache",
"cudaMemcpyFromSymbol_ptds",
"cudaDeviceEnablePeerAccess",
"cudaEGLStreamConsumerConnectWithFlags",
"cudaGraphInstantiateWithFlags",
"__cudaRegisterHostVar",
"cudaGetLastError",
"cudaMemcpy3DPeer_ptds",
"cudaGraphAddMemsetNode",
"cudaEGLStreamProducerConnect",
"cudaExternalMemoryGetMappedBuffer",
"cudaGetExportTable",
"cudaMallocManaged",
"cudaThreadExit",
"cudaDeviceGetMemPool",
"cudaGraphicsMapResources",
"cudaGraphEventWaitNodeGetEvent",
"cudaDeviceGetCacheConfig",
"cudaStreamQuery",
"cudaGraphGetRootNodes",
"cudaGraphMemcpyNodeSetParams",
"cudaDeviceSetGraphMemAttribute",
"cudaHostAlloc",
"cudaMemcpy2DAsync",
"cudaFreeHost",
"cudaGLUnmapBufferObject",
"cudaGraphAddEmptyNode",
"cudaMemcpyToArray",
"cudaMemcpy2DFromArrayAsync",
"cudaMemset_ptds",
"cudaDeviceSetSharedMemConfig",
"cudaGraphicsResourceSetMapFlags",
"cudaIpcGetEventHandle",
"cudaGraphAddEventWaitNode",
"cudaGraphKernelNodeSetAttribute",
"cudaEventRecordWithFlags_ptsz",
"cudaGraphicsUnregisterResource",
"cudaGraphHostNodeSetParams",
"cudaGetSymbolSize",
"cudaMemcpyToArray_ptds",
"cudaMemcpyToArrayAsync",
"cudaGraphicsUnmapResources",
"cudaSetDevice",
"cudaMemcpyFromSymbolAsync_ptsz",
"cudaMemcpyToSymbol_ptds",
"cudaGraphKernelNodeGetParams",
"cudaIpcGetMemHandle",
"cudaMipmappedArrayGetSparseProperties",
"cudaMemcpy",
"cudaFreeArray",
"cudaLaunchKernel_ptsz",
"cudaStreamWaitEvent_ptsz",
"cudaGraphCreate",
"cudaDeviceGetStreamPriorityRange",
"__cudaUnregisterFatBinary",
"cudaGraphEventWaitNodeSetEvent",
"cudaDeviceGetPCIBusId",
"cudaMemPoolExportToShareableHandle",
"cudaDeviceGetAttribute",
"cudaStreamAddCallback_ptsz",
"cudaGraphicsEGLRegisterImage",
"cudaMemset3DAsync_ptsz",
"cudaMemsetAsync_ptsz",
"cudaGLSetBufferObjectMapFlags",
"cudaMemcpy2DToArrayAsync",
"cudaMemcpy2DToArray",
"cudaVDPAUGetDevice",
"cudaUnbindTexture",
"cudaGetFuncBySymbol",
"cudaGraphAddHostNode",
"cudaSignalExternalSemaphoresAsync_ptsz",
"cudaStreamCreateWithFlags",
"__cudaInitModule",
"cudaGraphExecEventRecordNodeSetEvent",
"cudaMemPrefetchAsync",
"cudaFuncSetCacheConfig",
"cudaStreamGetAttribute_ptsz",
"cudaDeviceSetLimit",
"cudaDriverGetVersion",
"cudaGraphExternalSemaphoresWaitNodeGetParams",
"cudaGraphMemcpyNodeGetParams",
"cudaGetTextureReference",
"cudaDeviceSetMemPool",
"cudaSignalExternalSemaphoresAsync",
"cudaSetDeviceFlags",
"cudaMemcpy2D_ptds",
"cudaGraphLaunch_ptsz",
"cudaMemset3DAsync",
"cudaEventCreateWithFlags",
"cudaStreamCreate",
"cudaMallocAsync_ptsz",
"cudaEventElapsedTime",
"cudaGraphLaunch",
"cudaGetTextureObjectTextureDesc",
"cudaStreamGetCaptureInfo_v2",
"__cudaRegisterFunction",
"cudaGraphAddDependencies",
"cudaMemset2D",
"cudaGraphExecKernelNodeSetParams",
"cudaDeviceGetP2PAttribute",
"cudaDestroyExternalSemaphore",
"cudaFreeAsync_ptsz",
"__cudaRegisterFatBinary",
"cudaGraphAddMemcpyNodeToSymbol",
"cudaStreamUpdateCaptureDependencies",
"cudaGraphAddMemFreeNode",
"cudaDeviceGetLimit",
"cudaStreamGetCaptureInfo_v2_ptsz",
"__cudaPushCallConfiguration",
"cudaMemcpy2DFromArray_ptds",
"cudaGetTextureObjectResourceViewDesc",
"cudaGraphNodeGetType",
"cudaMemcpyToSymbolAsync",
"cudaSignalExternalSemaphoresAsync_v2",
"cudaMallocFromPoolAsync_ptsz",
"cudaLaunchCooperativeKernel",
"cudaStreamIsCapturing",
"cudaHostRegister",
"cudaGraphAddExternalSemaphoresWaitNode",
"cudaGraphExecEventWaitNodeSetEvent",
"cudaIpcOpenMemHandle",
"cudaLaunchCooperativeKernelMultiDevice",
"cudaMemcpy_ptds",
"cudaMemcpy2DAsync_ptsz",
"cudaGetDeviceProperties",
"cudaImportExternalSemaphore",
"cudaMemcpyToSymbolAsync_ptsz",
"cudaBindTexture",
"cudaGraphicsResourceGetMappedEglFrame",
"cudaIpcCloseMemHandle",
"cudaWaitExternalSemaphoresAsync_v2_ptsz",
"cudaGraphHostNodeGetParams",
"cudaStreamSynchronize",
"cudaEventSynchronize",
"cudaUserObjectCreate",
"cudaGetErrorName",
"cudaThreadGetCacheConfig",
"cudaGraphRemoveDependencies",
"cudaStreamGetPriority",
"cudaMemset2DAsync",
"cudaMemcpy2DArrayToArray_ptds",
"cudaGraphReleaseUserObject",
"cudaFree",
"cudaGetDriverEntryPoint",
"cudaMemcpy2DToArray_ptds",
"cudaGraphAddMemcpyNodeFromSymbol",
"cudaMemPoolGetAttribute",
"cudaMemset2D_ptds",
"cudaGraphAddMemcpyNode1D",
"cudaMallocHost",
"cudaGraphExecHostNodeSetParams",
"cudaMallocArray",
"cudaLaunchHostFunc_ptsz",
"cudaMemcpyFromArray_ptds",
"cudaEventRecord",
"cudaMemcpyPeerAsync",
"cudaMallocPitch",
};
// clang-format on
static std::vector<llvm::StringRef> getCudartSymbols() { return cudartSymbols; }

// clang-format off
static std::map<llvm::StringRef, llvm::StringRef> cudartEquivalentSymbolsMap = {
{"cudaGetDevice", "hipGetDevice"},
{"cudaWaitExternalSemaphoresAsync_ptsz", "hipWaitExternalSemaphoresAsync_ptsz"},
{"cudaStreamAddCallback", "hipStreamAddCallback"},
{"cudaMemcpyArrayToArray", "hipMemcpyArrayToArray"},
{"cudaDeviceReset", "hipDeviceReset"},
{"cudaGraphAddEventRecordNode", "hipGraphAddEventRecordNode"},
{"cudaGetSurfaceObjectResourceDesc", "hipGetSurfaceObjectResourceDesc"},
{"cudaGraphicsSubResourceGetMappedArray", "hipGraphicsSubResourceGetMappedArray"},
{"cudaMemRangeGetAttributes", "hipMemRangeGetAttributes"},
{"cudaGraphAddKernelNode", "hipGraphAddKernelNode"},
{"cudaGraphDestroy", "hipGraphDestroy"},
{"cudaGraphAddExternalSemaphoresSignalNode", "hipGraphAddExternalSemaphoresSignalNode"},
{"cudaGraphExecChildGraphNodeSetParams", "hipGraphExecChildGraphNodeSetParams"},
{"cudaEGLStreamConsumerReleaseFrame", "hipEGLStreamConsumerReleaseFrame"},
{"__cudaRegisterManagedVar", "__hipRegisterManagedVar"},
{"cudaMemcpy2DFromArray", "hipMemcpy2DFromArray"},
{"cudaEventRecord_ptsz", "hipEventRecord_ptsz"},
{"cudaSetDoubleForHost", "hipSetDoubleForHost"},
{"cudaGraphExternalSemaphoresWaitNodeSetParams", "hipGraphExternalSemaphoresWaitNodeSetParams"},
{"cudaMemPoolSetAttribute", "hipMemPoolSetAttribute"},
{"cudaDeviceFlushGPUDirectRDMAWrites", "hipDeviceFlushGPUDirectRDMAWrites"},
{"cudaDestroyExternalMemory", "hipDestroyExternalMemory"},
{"cudaDeviceGetGraphMemAttribute", "hipDeviceGetGraphMemAttribute"},
{"cudaEGLStreamConsumerConnect", "hipEGLStreamConsumerConnect"},
{"cudaGraphUpload", "hipGraphUpload"},
{"cudaDestroyTextureObject", "hipDestroyTextureObject"},
{"cudaHostGetFlags", "hipHostGetFlags"},
{"cudaStreamQuery_ptsz", "hipStreamQuery_ptsz"},
{"cudaHostGetDevicePointer", "hipHostGetDevicePointer"},
{"cudaPointerGetAttributes", "hipPointerGetAttributes"},
{"cudaWaitExternalSemaphoresAsync_v2", "hipWaitExternalSemaphoresAsync_v2"},
{"cudaFuncSetAttribute", "hipFuncSetAttribute"},
{"cudaDeviceGetSharedMemConfig", "hipDeviceGetSharedMemConfig"},
{"cudaGetDeviceFlags", "hipGetDeviceFlags"},
{"cudaGraphGetNodes", "hipGraphGetNodes"},
{"cudaGraphMemAllocNodeGetParams", "hipGraphMemAllocNodeGetParams"},
{"cudaMemcpy3D", "hipMemcpy3D"},
{"cudaMemcpy2DArrayToArray", "hipMemcpy2DArrayToArray"},
{"cudaBindTextureToArray", "hipBindTextureToArray"},
{"cudaDeviceDisablePeerAccess", "hipDeviceDisablePeerAccess"},
{"cudaGraphMemsetNodeGetParams", "hipGraphMemsetNodeGetParams"},
{"cudaGraphExecExternalSemaphoresWaitNodeSetParams", "hipGraphExecExternalSemaphoresWaitNodeSetParams"},
{"cudaGraphNodeGetDependentNodes", "hipGraphNodeGetDependentNodes"},
{"cudaEventDestroy", "hipEventDestroy"},
{"cudaDeviceCanAccessPeer", "hipDeviceCanAccessPeer"},
{"cudaArrayGetInfo", "hipArrayGetInfo"},
{"cudaMemcpyAsync", "hipMemcpyAsync"},
{"cudaStreamEndCapture_ptsz", "hipStreamEndCapture_ptsz"},
{"cudaGraphMemFreeNodeGetParams", "hipGraphMemFreeNodeGetParams"},
{"cudaGraphExecMemcpyNodeSetParams1D", "hipGraphExecMemcpyNodeSetParams1D"},
{"cudaOccupancyMaxActiveBlocksPerMultiprocessor", "hipOccupancyMaxActiveBlocksPerMultiprocessor"},
{"cudaGraphAddChildGraphNode", "hipGraphAddChildGraphNode"},
{"cudaGraphicsGLRegisterImage", "hipGraphicsGLRegisterImage"},
{"cudaGraphExecMemcpyNodeSetParamsToSymbol", "hipGraphExecMemcpyNodeSetParamsToSymbol"},
{"cudaProfilerInitialize", "hipProfilerInitialize"},
{"cudaWaitExternalSemaphoresAsync", "hipWaitExternalSemaphoresAsync"},
{"cudaMalloc3DArray", "hipMalloc3DArray"},
{"cudaGraphKernelNodeSetParams", "hipGraphKernelNodeSetParams"},
{"cudaProfilerStart", "hipProfilerStart"},
{"cudaGraphChildGraphNodeGetGraph", "hipGraphChildGraphNodeGetGraph"},
{"cudaGetErrorString", "hipGetErrorString"},
{"cudaMemset", "hipMemset"},
{"cudaGraphMemcpyNodeSetParamsFromSymbol", "hipGraphMemcpyNodeSetParamsFromSymbol"},
{"cudaMemset3D", "hipMemset3D"},
{"cudaGraphExecMemcpyNodeSetParamsFromSymbol", "hipGraphExecMemcpyNodeSetParamsFromSymbol"},
{"cudaMemcpyArrayToArray_ptds", "hipMemcpyArrayToArray_ptds"},
{"cudaMemcpy2D", "hipMemcpy2D"},
{"cudaGraphDestroyNode", "hipGraphDestroyNode"},
{"cudaStreamWaitEvent", "hipStreamWaitEvent"},
{"cudaMemcpy2DToArrayAsync_ptsz", "hipMemcpy2DToArrayAsync_ptsz"},
{"cudaGraphEventRecordNodeGetEvent", "hipGraphEventRecordNodeGetEvent"},
{"cudaSetDoubleForDevice", "hipSetDoubleForDevice"},
{"cudaLaunchCooperativeKernel_ptsz", "hipLaunchCooperativeKernel_ptsz"},
{"cudaLaunchKernel", "hipLaunchKernel"},
{"cudaFuncSetSharedMemConfig", "hipFuncSetSharedMemConfig"},
{"cudaPeekAtLastError", "hipPeekAtLastError"},
{"cudaMemcpy3DAsync_ptsz", "hipMemcpy3DAsync_ptsz"},
{"cudaEventCreate", "hipEventCreate"},
{"cudaMemPrefetchAsync_ptsz", "hipMemPrefetchAsync_ptsz"},
{"cudaMalloc", "hipMalloc"},
{"cudaMemPoolSetAccess", "hipMemPoolSetAccess"},
{"cudaBindTexture2D", "hipBindTexture2D"},
{"cudaMemPoolTrimTo", "hipMemPoolTrimTo"},
{"cudaThreadGetLimit", "hipThreadGetLimit"},
{"cudaGraphMemsetNodeSetParams", "hipGraphMemsetNodeSetParams"},
{"cudaGLRegisterBufferObject", "hipGLRegisterBufferObject"},
{"cudaGraphicsVDPAURegisterOutputSurface", "hipGraphicsVDPAURegisterOutputSurface"},
{"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"},
{"cudaEventCreateFromEGLSync", "hipEventCreateFromEGLSync"},
{"cudaGraphExternalSemaphoresSignalNodeGetParams", "hipGraphExternalSemaphoresSignalNodeGetParams"},
{"cudaMemPoolExportPointer", "hipMemPoolExportPointer"},
{"cudaGraphNodeFindInClone", "hipGraphNodeFindInClone"},
{"cudaGetTextureAlignmentOffset", "hipGetTextureAlignmentOffset"},
{"cudaSignalExternalSemaphoresAsync_v2_ptsz", "hipSignalExternalSemaphoresAsync_v2_ptsz"},
{"cudaGraphKernelNodeGetAttribute", "hipGraphKernelNodeGetAttribute"},
{"cudaHostUnregister", "hipHostUnregister"},
{"cudaStreamSetAttribute", "hipStreamSetAttribute"},
{"cudaLaunchHostFunc", "hipLaunchHostFunc"},
{"__cudaRegisterFatBinaryEnd", "__hipRegisterFatBinaryEnd"},
{"cudaGetTextureObjectResourceDesc", "hipGetTextureObjectResourceDesc"},
{"cudaGraphExternalSemaphoresSignalNodeSetParams", "hipGraphExternalSemaphoresSignalNodeSetParams"},
{"cudaMemPoolImportFromShareableHandle", "hipMemPoolImportFromShareableHandle"},
{"cudaStreamDestroy", "hipStreamDestroy"},
{"cudaMalloc3D", "hipMalloc3D"},
{"cudaGLSetGLDevice", "hipGLSetGLDevice"},
{"cudaGraphRetainUserObject", "hipGraphRetainUserObject"},
{"cudaGraphExecExternalSemaphoresSignalNodeSetParams", "hipGraphExecExternalSemaphoresSignalNodeSetParams"},
{"cudaMemAdvise", "hipMemAdvise"},
{"cudaEventRecordWithFlags", "hipEventRecordWithFlags"},
{"cudaMemcpy3DPeerAsync_ptsz", "hipMemcpy3DPeerAsync_ptsz"},
{"cudaGraphExecMemcpyNodeSetParams", "hipGraphExecMemcpyNodeSetParams"},
{"cudaProfilerStop", "hipProfilerStop"},
{"cudaFreeMipmappedArray", "hipFreeMipmappedArray"},
{"cudaStreamCopyAttributes_ptsz", "hipStreamCopyAttributes_ptsz"},
{"cudaMemcpyFromArray", "hipMemcpyFromArray"},
{"cudaMemcpy3DPeer", "hipMemcpy3DPeer"},
{"cudaMemPoolImportPointer", "hipMemPoolImportPointer"},
{"cudaMemPoolCreate", "hipMemPoolCreate"},
{"cudaCreateTextureObject", "hipCreateTextureObject"},
{"cudaGraphExecDestroy", "hipGraphExecDestroy"},
{"cudaMemGetInfo", "hipMemGetInfo"},
{"cudaStreamGetFlags", "hipStreamGetFlags"},
{"cudaGetMipmappedArrayLevel", "hipGetMipmappedArrayLevel"},
{"cudaMemset2DAsync_ptsz", "hipMemset2DAsync_ptsz"},
{"cudaMemcpyAsync_ptsz", "hipMemcpyAsync_ptsz"},
{"cudaCreateSurfaceObject", "hipCreateSurfaceObject"},
{"cudaMemRangeGetAttribute", "hipMemRangeGetAttribute"},
{"cudaStreamCopyAttributes", "hipStreamCopyAttributes"},
{"cudaMemcpyToSymbol", "hipMemcpyToSymbol"},
{"cudaMemcpy3D_ptds", "hipMemcpy3D_ptds"},
{"cudaGLUnregisterBufferObject", "hipGLUnregisterBufferObject"},
{"cudaGraphInstantiate", "hipGraphInstantiate"},
{"cudaStreamBeginCapture", "hipStreamBeginCapture"},
{"cudaDestroySurfaceObject", "hipDestroySurfaceObject"},
{"cudaMemcpy3DAsync", "hipMemcpy3DAsync"},
{"cudaFuncGetAttributes", "hipFuncGetAttributes"},
{"cudaStreamIsCapturing_ptsz", "hipStreamIsCapturing_ptsz"},
{"cudaChooseDevice", "hipChooseDevice"},
{"cudaGraphExecMemsetNodeSetParams", "hipGraphExecMemsetNodeSetParams"},
{"cudaArrayGetPlane", "hipArrayGetPlane"},
{"__cudaPopCallConfiguration", "__hipPopCallConfiguration"},
{"cudaThreadSetCacheConfig", "hipThreadSetCacheConfig"},
{"cudaStreamAttachMemAsync_ptsz", "hipStreamAttachMemAsync_ptsz"},
{"cudaGLMapBufferObjectAsync", "hipGLMapBufferObjectAsync"},
{"cudaMemcpyFromArrayAsync_ptsz", "hipMemcpyFromArrayAsync_ptsz"},
{"cudaMemcpy2DFromArrayAsync_ptsz", "hipMemcpy2DFromArrayAsync_ptsz"},
{"cudaMemcpyToArrayAsync_ptsz", "hipMemcpyToArrayAsync_ptsz"},
{"cudaArrayGetSparseProperties", "hipArrayGetSparseProperties"},
{"cudaExternalMemoryGetMappedMipmappedArray", "hipExternalMemoryGetMappedMipmappedArray"},
{"cudaGraphClone", "hipGraphClone"},
{"cudaStreamGetPriority_ptsz", "hipStreamGetPriority_ptsz"},
{"cudaRuntimeGetVersion", "hipRuntimeGetVersion"},
{"cudaMemPoolDestroy", "hipMemPoolDestroy"},
{"cudaGraphMemcpyNodeSetParamsToSymbol", "hipGraphMemcpyNodeSetParamsToSymbol"},
{"cudaGraphExecUpdate", "hipGraphExecUpdate"},
{"cudaEGLStreamConsumerDisconnect", "hipEGLStreamConsumerDisconnect"},
{"cudaGetSymbolAddress", "hipGetSymbolAddress"},
{"__cudaRegisterVar", "__hipRegisterVar"},
{"cudaStreamGetCaptureInfo", "hipStreamGetCaptureInfo"},
{"cudaMemcpy3DPeerAsync", "hipMemcpy3DPeerAsync"},
{"cudaMemcpyPeer", "hipMemcpyPeer"},
{"cudaDeviceGetByPCIBusId", "hipDeviceGetByPCIBusId"},
{"cudaEGLStreamProducerDisconnect", "hipEGLStreamProducerDisconnect"},
{"cudaEGLStreamConsumerAcquireFrame", "hipEGLStreamConsumerAcquireFrame"},
{"__cudaRegisterTexture", "__hipRegisterTexture"},
{"cudaGraphicsVDPAURegisterVideoSurface", "hipGraphicsVDPAURegisterVideoSurface"},
{"cudaDeviceSetCacheConfig", "hipDeviceSetCacheConfig"},
{"cudaMemcpyFromArrayAsync", "hipMemcpyFromArrayAsync"},
{"cudaGraphEventRecordNodeSetEvent", "hipGraphEventRecordNodeSetEvent"},
{"cudaGraphAddMemcpyNode", "hipGraphAddMemcpyNode"},
{"cudaDeviceGetDefaultMemPool", "hipDeviceGetDefaultMemPool"},
{"cudaStreamSynchronize_ptsz", "hipStreamSynchronize_ptsz"},
{"cudaBindSurfaceToArray", "hipBindSurfaceToArray"},
{"cudaMallocAsync", "hipMallocAsync"},
{"cudaGraphGetEdges", "hipGraphGetEdges"},
{"cudaGetDriverEntryPoint_ptsz", "hipGetDriverEntryPoint_ptsz"},
{"cudaGraphMemcpyNodeSetParams1D", "hipGraphMemcpyNodeSetParams1D"},
{"cudaGraphKernelNodeCopyAttributes", "hipGraphKernelNodeCopyAttributes"},
{"cudaVDPAUSetVDPAUDevice", "hipVDPAUSetVDPAUDevice"},
{"cudaDeviceGraphMemTrim", "hipDeviceGraphMemTrim"},
{"cudaGraphicsResourceGetMappedMipmappedArray", "hipGraphicsResourceGetMappedMipmappedArray"},
{"cudaThreadSynchronize", "hipDeviceSynchronize"},
{"cudaDeviceGetTexture1DLinearMaxWidth", "hipDeviceGetTexture1DLinearMaxWidth"},
{"cudaDeviceSynchronize", "hipDeviceSynchronize"},
{"cudaMemcpyFromSymbolAsync", "hipMemcpyFromSymbolAsync"},
{"cudaSetValidDevices", "hipSetValidDevices"},
{"cudaOccupancyAvailableDynamicSMemPerBlock", "hipOccupancyAvailableDynamicSMemPerBlock"},
{"cudaStreamSetAttribute_ptsz", "hipStreamSetAttribute_ptsz"},
{"cudaMemcpyFromSymbol", "hipMemcpyFromSymbol"},
{"cudaStreamEndCapture", "hipStreamEndCapture"},
{"cudaImportExternalMemory", "hipImportExternalMemory"},
{"__cudaRegisterSurface", "__hipRegisterSurface"},
{"cudaThreadSetLimit", "hipThreadSetLimit"},
{"cudaGLMapBufferObject", "hipGLMapBufferObject"},
{"cudaBindTextureToMipmappedArray", "hipBindTextureToMipmappedArray"},
{"cudaGraphUpload_ptsz", "hipGraphUpload_ptsz"},
{"cudaGLGetDevices", "hipGLGetDevices"},
{"cudaGraphAddMemAllocNode", "hipGraphAddMemAllocNode"},
{"cudaMemsetAsync", "hipMemsetAsync"},
{"cudaGLUnmapBufferObjectAsync", "hipGLUnmapBufferObjectAsync"},
{"cudaUserObjectRetain", "hipUserObjectRetain"},
{"cudaGraphNodeGetDependencies", "hipGraphNodeGetDependencies"},
{"cudaStreamCreateWithPriority", "hipStreamCreateWithPriority"},
{"cudaStreamGetCaptureInfo_ptsz", "hipStreamGetCaptureInfo_ptsz"},
{"cudaStreamGetAttribute", "hipStreamGetAttribute"},
{"cudaStreamAttachMemAsync", "hipStreamAttachMemAsync"},
{"cudaGetDeviceCount", "hipGetDeviceCount"},
{"cudaMemset3D_ptds", "hipMemset3D_ptds"},
{"cudaFreeAsync", "hipFreeAsync"},
{"cudaUserObjectRelease", "hipUserObjectRelease"},
{"cudaCreateChannelDesc", "hipCreateChannelDesc"},
{"cudaGetSurfaceReference", "hipGetSurfaceReference"},
{"cudaGetChannelDesc", "hipGetChannelDesc"},
{"cudaGraphDebugDotPrint", "hipGraphDebugDotPrint"},
{"cudaEGLStreamProducerPresentFrame", "hipEGLStreamProducerPresentFrame"},
{"cudaEventQuery", "hipEventQuery"},
{"cudaStreamBeginCapture_ptsz", "hipStreamBeginCapture_ptsz"},
{"cudaMallocMipmappedArray", "hipMallocMipmappedArray"},
{"cudaThreadExchangeStreamCaptureMode", "hipThreadExchangeStreamCaptureMode"},
{"cudaStreamGetFlags_ptsz", "hipStreamGetFlags_ptsz"},
{"cudaStreamUpdateCaptureDependencies_ptsz", "hipStreamUpdateCaptureDependencies_ptsz"},
{"cudaGraphicsGLRegisterBuffer", "hipGraphicsGLRegisterBuffer"},
{"cudaDeviceGetNvSciSyncAttributes", "hipDeviceGetNvSciSyncAttributes"},
{"cudaEGLStreamProducerReturnFrame", "hipEGLStreamProducerReturnFrame"},
{"cudaIpcOpenEventHandle", "hipIpcOpenEventHandle"},
{"cudaMemPoolGetAccess", "hipMemPoolGetAccess"},
{"cudaGraphicsResourceGetMappedPointer", "hipGraphicsResourceGetMappedPointer"},
{"cudaMallocFromPoolAsync", "hipMallocFromPoolAsync"},
{"cudaCtxResetPersistingL2Cache", "hipCtxResetPersistingL2Cache"},
{"cudaMemcpyFromSymbol_ptds", "hipMemcpyFromSymbol_ptds"},
{"cudaDeviceEnablePeerAccess", "hipDeviceEnablePeerAccess"},
{"cudaEGLStreamConsumerConnectWithFlags", "hipEGLStreamConsumerConnectWithFlags"},
{"cudaGraphInstantiateWithFlags", "hipGraphInstantiateWithFlags"},
{"__cudaRegisterHostVar", "__hipRegisterHostVar"},
{"cudaGetLastError", "hipGetLastError"},
{"cudaMemcpy3DPeer_ptds", "hipMemcpy3DPeer_ptds"},
{"cudaGraphAddMemsetNode", "hipGraphAddMemsetNode"},
{"cudaEGLStreamProducerConnect", "hipEGLStreamProducerConnect"},
{"cudaExternalMemoryGetMappedBuffer", "hipExternalMemoryGetMappedBuffer"},
{"cudaGetExportTable", "hipGetExportTable"},
{"cudaMallocManaged", "hipMallocManaged"},
{"cudaThreadExit", "hipThreadExit"},
{"cudaDeviceGetMemPool", "hipDeviceGetMemPool"},
{"cudaGraphicsMapResources", "hipGraphicsMapResources"},
{"cudaGraphEventWaitNodeGetEvent", "hipGraphEventWaitNodeGetEvent"},
{"cudaDeviceGetCacheConfig", "hipDeviceGetCacheConfig"},
{"cudaStreamQuery", "hipStreamQuery"},
{"cudaGraphGetRootNodes", "hipGraphGetRootNodes"},
{"cudaGraphMemcpyNodeSetParams", "hipGraphMemcpyNodeSetParams"},
{"cudaDeviceSetGraphMemAttribute", "hipDeviceSetGraphMemAttribute"},
{"cudaHostAlloc", "hipHostAlloc"},
{"cudaMemcpy2DAsync", "hipMemcpy2DAsync"},
{"cudaFreeHost", "hipFreeHost"},
{"cudaGLUnmapBufferObject", "hipGLUnmapBufferObject"},
{"cudaGraphAddEmptyNode", "hipGraphAddEmptyNode"},
{"cudaMemcpyToArray", "hipMemcpyToArray"},
{"cudaMemcpy2DFromArrayAsync", "hipMemcpy2DFromArrayAsync"},
{"cudaMemset_ptds", "hipMemset_ptds"},
{"cudaDeviceSetSharedMemConfig", "hipDeviceSetSharedMemConfig"},
{"cudaGraphicsResourceSetMapFlags", "hipGraphicsResourceSetMapFlags"},
{"cudaIpcGetEventHandle", "hipIpcGetEventHandle"},
{"cudaGraphAddEventWaitNode", "hipGraphAddEventWaitNode"},
{"cudaGraphKernelNodeSetAttribute", "hipGraphKernelNodeSetAttribute"},
{"cudaEventRecordWithFlags_ptsz", "hipEventRecordWithFlags_ptsz"},
{"cudaGraphicsUnregisterResource", "hipGraphicsUnregisterResource"},
{"cudaGraphHostNodeSetParams", "hipGraphHostNodeSetParams"},
{"cudaGetSymbolSize", "hipGetSymbolSize"},
{"cudaMemcpyToArray_ptds", "hipMemcpyToArray_ptds"},
{"cudaMemcpyToArrayAsync", "hipMemcpyToArrayAsync"},
{"cudaGraphicsUnmapResources", "hipGraphicsUnmapResources"},
{"cudaSetDevice", "hipSetDevice"},
{"cudaMemcpyFromSymbolAsync_ptsz", "hipMemcpyFromSymbolAsync_ptsz"},
{"cudaMemcpyToSymbol_ptds", "hipMemcpyToSymbol_ptds"},
{"cudaGraphKernelNodeGetParams", "hipGraphKernelNodeGetParams"},
{"cudaIpcGetMemHandle", "hipIpcGetMemHandle"},
{"cudaMipmappedArrayGetSparseProperties", "hipMipmappedArrayGetSparseProperties"},
{"cudaMemcpy", "hipMemcpy"},
{"cudaFreeArray", "hipFreeArray"},
{"cudaLaunchKernel_ptsz", "hipLaunchKernel_ptsz"},
{"cudaStreamWaitEvent_ptsz", "hipStreamWaitEvent_ptsz"},
{"cudaGraphCreate", "hipGraphCreate"},
{"cudaDeviceGetStreamPriorityRange", "hipDeviceGetStreamPriorityRange"},
{"__cudaUnregisterFatBinary", "__hipUnregisterFatBinary"},
{"cudaGraphEventWaitNodeSetEvent", "hipGraphEventWaitNodeSetEvent"},
{"cudaDeviceGetPCIBusId", "hipDeviceGetPCIBusId"},
{"cudaMemPoolExportToShareableHandle", "hipMemPoolExportToShareableHandle"},
{"cudaDeviceGetAttribute", "hipDeviceGetAttribute"},
{"cudaStreamAddCallback_ptsz", "hipStreamAddCallback_ptsz"},
{"cudaGraphicsEGLRegisterImage", "hipGraphicsEGLRegisterImage"},
{"cudaMemset3DAsync_ptsz", "hipMemset3DAsync_ptsz"},
{"cudaMemsetAsync_ptsz", "hipMemsetAsync_ptsz"},
{"cudaGLSetBufferObjectMapFlags", "hipGLSetBufferObjectMapFlags"},
{"cudaMemcpy2DToArrayAsync", "hipMemcpy2DToArrayAsync"},
{"cudaMemcpy2DToArray", "hipMemcpy2DToArray"},
{"cudaVDPAUGetDevice", "hipVDPAUGetDevice"},
{"cudaUnbindTexture", "hipUnbindTexture"},
{"cudaGetFuncBySymbol", "hipGetFuncBySymbol"},
{"cudaGraphAddHostNode", "hipGraphAddHostNode"},
{"cudaSignalExternalSemaphoresAsync_ptsz", "hipSignalExternalSemaphoresAsync_ptsz"},
{"cudaStreamCreateWithFlags", "hipStreamCreateWithFlags"},
{"__cudaInitModule", "__hipInitModule"},
{"cudaGraphExecEventRecordNodeSetEvent", "hipGraphExecEventRecordNodeSetEvent"},
{"cudaMemPrefetchAsync", "hipMemPrefetchAsync"},
{"cudaFuncSetCacheConfig", "hipFuncSetCacheConfig"},
{"cudaStreamGetAttribute_ptsz", "hipStreamGetAttribute_ptsz"},
{"cudaDeviceSetLimit", "hipDeviceSetLimit"},
{"cudaDriverGetVersion", "hipDriverGetVersion"},
{"cudaGraphExternalSemaphoresWaitNodeGetParams", "hipGraphExternalSemaphoresWaitNodeGetParams"},
{"cudaGraphMemcpyNodeGetParams", "hipGraphMemcpyNodeGetParams"},
{"cudaGetTextureReference", "hipGetTextureReference"},
{"cudaDeviceSetMemPool", "hipDeviceSetMemPool"},
{"cudaSignalExternalSemaphoresAsync", "hipSignalExternalSemaphoresAsync"},
{"cudaSetDeviceFlags", "hipSetDeviceFlags"},
{"cudaMemcpy2D_ptds", "hipMemcpy2D_ptds"},
{"cudaGraphLaunch_ptsz", "hipGraphLaunch_ptsz"},
{"cudaMemset3DAsync", "hipMemset3DAsync"},
{"cudaEventCreateWithFlags", "hipEventCreateWithFlags"},
{"cudaStreamCreate", "hipStreamCreate"},
{"cudaMallocAsync_ptsz", "hipMallocAsync_ptsz"},
{"cudaEventElapsedTime", "hipEventElapsedTime"},
{"cudaGraphLaunch", "hipGraphLaunch"},
{"cudaGetTextureObjectTextureDesc", "hipGetTextureObjectTextureDesc"},
{"cudaStreamGetCaptureInfo_v2", "hipStreamGetCaptureInfo_v2"},
{"__cudaRegisterFunction", "__hipRegisterFunction"},
{"cudaGraphAddDependencies", "hipGraphAddDependencies"},
{"cudaMemset2D", "hipMemset2D"},
{"cudaGraphExecKernelNodeSetParams", "hipGraphExecKernelNodeSetParams"},
{"cudaDeviceGetP2PAttribute", "hipDeviceGetP2PAttribute"},
{"cudaDestroyExternalSemaphore", "hipDestroyExternalSemaphore"},
{"cudaFreeAsync_ptsz", "hipFreeAsync_ptsz"},
{"__cudaRegisterFatBinary", "__hipRegisterFatBinary"},
{"cudaGraphAddMemcpyNodeToSymbol", "hipGraphAddMemcpyNodeToSymbol"},
{"cudaStreamUpdateCaptureDependencies", "hipStreamUpdateCaptureDependencies"},
{"cudaGraphAddMemFreeNode", "hipGraphAddMemFreeNode"},
{"cudaDeviceGetLimit", "hipDeviceGetLimit"},
{"cudaStreamGetCaptureInfo_v2_ptsz", "hipStreamGetCaptureInfo_v2_ptsz"},
{"__cudaPushCallConfiguration", "__hipPushCallConfiguration"},
{"cudaMemcpy2DFromArray_ptds", "hipMemcpy2DFromArray_ptds"},
{"cudaGetTextureObjectResourceViewDesc", "hipGetTextureObjectResourceViewDesc"},
{"cudaGraphNodeGetType", "hipGraphNodeGetType"},
{"cudaMemcpyToSymbolAsync", "hipMemcpyToSymbolAsync"},
{"cudaSignalExternalSemaphoresAsync_v2", "hipSignalExternalSemaphoresAsync_v2"},
{"cudaMallocFromPoolAsync_ptsz", "hipMallocFromPoolAsync_ptsz"},
{"cudaLaunchCooperativeKernel", "hipLaunchCooperativeKernel"},
{"cudaStreamIsCapturing", "hipStreamIsCapturing"},
{"cudaHostRegister", "hipHostRegister"},
{"cudaGraphAddExternalSemaphoresWaitNode", "hipGraphAddExternalSemaphoresWaitNode"},
{"cudaGraphExecEventWaitNodeSetEvent", "hipGraphExecEventWaitNodeSetEvent"},
{"cudaIpcOpenMemHandle", "hipIpcOpenMemHandle"},
{"cudaLaunchCooperativeKernelMultiDevice", "hipLaunchCooperativeKernelMultiDevice"},
{"cudaMemcpy_ptds", "hipMemcpy_ptds"},
{"cudaMemcpy2DAsync_ptsz", "hipMemcpy2DAsync_ptsz"},
{"cudaGetDeviceProperties", "mgpurtCudaGetDeviceProperties"},
{"cudaImportExternalSemaphore", "hipImportExternalSemaphore"},
{"cudaMemcpyToSymbolAsync_ptsz", "hipMemcpyToSymbolAsync_ptsz"},
{"cudaBindTexture", "hipBindTexture"},
{"cudaGraphicsResourceGetMappedEglFrame", "hipGraphicsResourceGetMappedEglFrame"},
{"cudaIpcCloseMemHandle", "hipIpcCloseMemHandle"},
{"cudaWaitExternalSemaphoresAsync_v2_ptsz", "hipWaitExternalSemaphoresAsync_v2_ptsz"},
{"cudaGraphHostNodeGetParams", "hipGraphHostNodeGetParams"},
{"cudaStreamSynchronize", "hipStreamSynchronize"},
{"cudaEventSynchronize", "hipEventSynchronize"},
{"cudaUserObjectCreate", "hipUserObjectCreate"},
{"cudaGetErrorName", "hipGetErrorName"},
{"cudaThreadGetCacheConfig", "hipThreadGetCacheConfig"},
{"cudaGraphRemoveDependencies", "hipGraphRemoveDependencies"},
{"cudaStreamGetPriority", "hipStreamGetPriority"},
{"cudaMemset2DAsync", "hipMemset2DAsync"},
{"cudaMemcpy2DArrayToArray_ptds", "hipMemcpy2DArrayToArray_ptds"},
{"cudaGraphReleaseUserObject", "hipGraphReleaseUserObject"},
{"cudaFree", "hipFree"},
{"cudaGetDriverEntryPoint", "hipGetDriverEntryPoint"},
{"cudaMemcpy2DToArray_ptds", "hipMemcpy2DToArray_ptds"},
{"cudaGraphAddMemcpyNodeFromSymbol", "hipGraphAddMemcpyNodeFromSymbol"},
{"cudaMemPoolGetAttribute", "hipMemPoolGetAttribute"},
{"cudaMemset2D_ptds", "hipMemset2D_ptds"},
{"cudaGraphAddMemcpyNode1D", "hipGraphAddMemcpyNode1D"},
{"cudaMallocHost", "hipMallocHost"},
{"cudaGraphExecHostNodeSetParams", "hipGraphExecHostNodeSetParams"},
{"cudaMallocArray", "hipMallocArray"},
{"cudaLaunchHostFunc_ptsz", "hipLaunchHostFunc_ptsz"},
{"cudaMemcpyFromArray_ptds", "hipMemcpyFromArray_ptds"},
{"cudaEventRecord", "hipEventRecord"},
{"cudaMemcpyPeerAsync", "hipMemcpyPeerAsync"},
{"cudaMallocPitch", "hipMallocPitch"},
};
// clang-format on

static std::map<llvm::StringRef, llvm::StringRef>
getCudartEquivalentSymbolsMap() {
  return cudartEquivalentSymbolsMap;
}

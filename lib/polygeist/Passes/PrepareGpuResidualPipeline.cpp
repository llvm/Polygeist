//===- PrepareGpuResidualPipeline.cpp - GPU residual bridge --------------===//

#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;

namespace {

static Value stripMemrefCasts(Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    auto result = dyn_cast<OpResult>(value);
    auto loop = result ? dyn_cast<scf::ForOp>(result.getOwner()) : scf::ForOp();
    if (loop && result.getResultNumber() < loop.getInitArgs().size()) {
      value = loop.getInitArgs()[result.getResultNumber()];
      continue;
    }
    break;
  }
  return value;
}

struct PrepareGpuResidualPipelinePass
    : public mlir::polygeist::PrepareGpuResidualPipelineBase<
          PrepareGpuResidualPipelinePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (functionName.empty()) {
      module.emitError("--prepare-gpu-residual-pipeline requires function=");
      return signalPassFailure();
    }
    auto function = module.lookupSymbol<func::FuncOp>(functionName);
    if (!function) {
      module.emitError() << "GPU residual function not found: " << functionName;
      return signalPassFailure();
    }

    // One-shot bufferization can leave a terminal compatibility round-trip:
    //   alloc; copy(cast %dst, alloc); copy(alloc, %dst)
    // when the tensor result has already been written back to %dst.  This is
    // a pure no-op if the two external endpoints alias after stripping
    // descriptor casts. Remove it before copies are converted to GPU kernels;
    // otherwise it creates two launches and a dynamic allocation inside the
    // maximal CUDA Graph sequence.
    SmallVector<memref::AllocOp> allocations;
    function.walk([&](memref::AllocOp alloc) { allocations.push_back(alloc); });
    for (memref::AllocOp alloc : allocations) {
      memref::CopyOp copyIn;
      memref::CopyOp copyOut;
      bool otherUser = false;
      for (Operation *user : alloc.getResult().getUsers()) {
        auto copy = dyn_cast<memref::CopyOp>(user);
        if (!copy) {
          otherUser = true;
          break;
        }
        if (copy.getTarget() == alloc.getResult() && !copyIn)
          copyIn = copy;
        else if (copy.getSource() == alloc.getResult() && !copyOut)
          copyOut = copy;
        else
          otherUser = true;
      }
      if (otherUser || !copyIn || !copyOut ||
          stripMemrefCasts(copyIn.getSource()) !=
              stripMemrefCasts(copyOut.getTarget()))
        continue;
      copyIn.erase();
      copyOut.erase();
      alloc.erase();
    }

    // LowerSubmapInverse marks affine write-backs only after proving their
    // destination map injective on the complete static iteration domain.
    // Bufferization turns their tensor iter_args into a single loop-carried
    // memref. Recover that proof here and expose the perfect loop nest as one
    // parallel iteration space for the standard GPU mapper.
    SmallVector<scf::ForOp> writebacks;
    function.walk([&](scf::ForOp loop) {
      if (loop->hasAttr("polygeist.injective_writeback"))
        writebacks.push_back(loop);
    });
    for (scf::ForOp outer : writebacks) {
      unsigned numRegionArgs = outer.getNumRegionIterArgs();
      if (numRegionArgs > 1 || outer.getNumResults() != numRegionArgs)
        continue;
      SmallVector<scf::ForOp> nest;
      scf::ForOp current = outer;
      while (current) {
        if (current.getNumRegionIterArgs() != numRegionArgs ||
            current.getNumResults() != numRegionArgs)
          break;
        nest.push_back(current);
        scf::ForOp child;
        for (Operation &op : current.getBody()->without_terminator()) {
          if (auto candidate = dyn_cast<scf::ForOp>(op)) {
            if (child) {
              child = nullptr;
              break;
            }
            child = candidate;
          } else {
            child = nullptr;
            break;
          }
        }
        current = child;
      }
      if (nest.empty())
        continue;
      scf::ForOp inner = nest.back();
      bool hasNestedLoop = false;
      for (Operation &op : inner.getBody()->without_terminator())
        hasNestedLoop |= isa<scf::ForOp>(op);
      if (hasNestedLoop)
        continue;

      SmallVector<Value> lower, upper, step;
      for (scf::ForOp loop : nest) {
        lower.push_back(loop.getLowerBound());
        upper.push_back(loop.getUpperBound());
        step.push_back(loop.getStep());
      }
      OpBuilder builder(outer);
      auto parallel =
          builder.create<scf::ParallelOp>(outer.getLoc(), lower, upper, step);
      IRMapping mapping;
      Value destination;
      Value allocation;
      memref::CopyOp snapshotIn;
      memref::CopyOp snapshotOut;
      if (numRegionArgs == 1) {
        destination = outer.getInitArgs().front();
        allocation = destination;
      } else {
        // Current one-shot bufferization removes the tensor iter_arg and
        // writes directly to the allocated result buffer. Recover that
        // destination from the unique store target in the innermost loop.
        for (Operation &op : inner.getBody()->without_terminator()) {
          auto store = dyn_cast<memref::StoreOp>(op);
          if (!store)
            continue;
          if (allocation && allocation != store.getMemref()) {
            allocation = nullptr;
            break;
          }
          allocation = store.getMemref();
        }
        destination = allocation;
      }
      if (auto alloc = allocation.getDefiningOp<memref::AllocOp>()) {
        for (Operation *user : allocation.getUsers()) {
          if (auto copy = dyn_cast<memref::CopyOp>(user)) {
            if (copy.getTarget() == allocation)
              snapshotIn = copy;
            else if (copy.getSource() == allocation)
              snapshotOut = copy;
          }
        }
        if (numRegionArgs == 1)
          for (Operation *user : outer.getResult(0).getUsers())
            if (auto copy = dyn_cast<memref::CopyOp>(user))
              if (copy.getSource() == outer.getResult(0))
                snapshotOut = copy;
        // This is the bufferized form of
        //   result = tensor.insert(..., original)
        // followed by writing result back to original. Because the submap
        // lowering already proved every written destination unique, direct
        // stores preserve both the updated subset and every untouched element.
        if (snapshotIn && snapshotOut &&
            snapshotIn.getSource() == snapshotOut.getTarget())
          destination = snapshotIn.getSource();
        else
          alloc = nullptr;
      }
      if (!destination)
        continue;
      for (auto [loop, iv] : llvm::zip(nest, parallel.getInductionVars())) {
        mapping.map(loop.getInductionVar(), iv);
        if (numRegionArgs == 1)
          mapping.map(loop.getRegionIterArgs().front(), destination);
      }
      if (numRegionArgs == 0 && allocation != destination)
        mapping.map(allocation, destination);
      builder.setInsertionPointToStart(parallel.getBody());
      for (Operation &op : inner.getBody()->without_terminator())
        builder.clone(op, mapping);
      if (numRegionArgs == 1)
        outer.getResult(0).replaceAllUsesWith(destination);
      outer.erase();
      if (snapshotIn && snapshotOut &&
          snapshotIn.getSource() == snapshotOut.getTarget()) {
        Value allocation = snapshotIn.getTarget();
        snapshotIn.erase();
        snapshotOut.erase();
        if (auto alloc = allocation.getDefiningOp<memref::AllocOp>();
            alloc && allocation.use_empty())
          alloc.erase();
      }
    }

    // memref.copy is otherwise lowered to a host loop by the existing ABI
    // pipeline. linalg.copy follows the same generic parallel-loop/GPU
    // outlining route as residual linalg.generic operations.
    SmallVector<memref::CopyOp> copies;
    function.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
    for (memref::CopyOp copy : copies) {
      OpBuilder builder(copy);
      builder.create<linalg::CopyOp>(copy.getLoc(), copy.getSource(),
                                     copy.getTarget());
      copy.erase();
    }

    SmallVector<gpu::LaunchFuncOp> launches;
    function.walk([&](gpu::LaunchFuncOp launch) {
      launch->setAttr("polygeist.cuda_graph_safe",
                      UnitAttr::get(module.getContext()));
      launches.push_back(launch);
    });
    if (launches.empty())
      return;

    // Host registration is deliberately emitted only after outlining (the
    // second invocation of this pass). Registering bases, rather than every
    // subview, keeps graph capture independent of view construction and lets
    // the runtime's page-granular cache coalesce aliases.
    Block &entry = function.front();
    OpBuilder builder = OpBuilder::atBlockBegin(&entry);
    Location loc = function.getLoc();
    for (BlockArgument argument : entry.getArguments()) {
      auto type = dyn_cast<MemRefType>(argument.getType());
      if (!type)
        continue;
      auto unranked =
          UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
      Value cast = builder.create<memref::CastOp>(loc, unranked, argument);
      builder.create<gpu::HostRegisterOp>(loc, cast);
    }

    SmallVector<memref::GlobalOp> globals;
    module.walk([&](memref::GlobalOp global) { globals.push_back(global); });
    for (memref::GlobalOp global : globals) {
      auto type = global.getType();
      if (!type || !type.hasStaticShape())
        continue;
      Value value =
          builder.create<memref::GetGlobalOp>(loc, type, global.getSymName());
      auto unranked =
          UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
      Value cast = builder.create<memref::CastOp>(loc, unranked, value);
      builder.create<gpu::HostRegisterOp>(loc, cast);
    }
    function->setAttr("polygeist.gpu_residual_pipeline",
                      UnitAttr::get(module.getContext()));
  }
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createPrepareGpuResidualPipelinePass() {
  return std::make_unique<PrepareGpuResidualPipelinePass>();
}
} // namespace polygeist
} // namespace mlir

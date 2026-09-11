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

#include "llvm/ADT/DenseSet.h"

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

// Return true when a local host allocation (possibly through any number of
// memref-producing view/descriptor operations) is passed to an outlined GPU
// kernel.  Library calls already map their pointer operands in the runtime;
// generated kernels instead consume the memref pointer directly and therefore
// require an explicit gpu.host_register when the allocation was not promoted
// to gpu.alloc.
static bool reachesGpuLaunch(Value root) {
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (isa<gpu::LaunchFuncOp>(user))
        return true;
      for (Value result : user->getResults())
        if (isa<BaseMemRefType>(result.getType()))
          worklist.push_back(result);
    }
  }
  return false;
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

    // Bufferization of tensor.insert_slice/submap write-backs commonly makes
    // a chain of full snapshots around a small update:
    //
    //   copy %external -> %a; update(subview %a)
    //   copy %a -> %b; copy %b -> %external
    //
    // The chain only exists to express tensor value semantics. Once the
    // destination is a memref, redirecting the snapshots to the original
    // buffer preserves the ordered update and avoids copying the complete KV
    // cache (or activation vector) before and after every slice insertion.
    // Require an exact type and a closed copy chain back to the same value;
    // this keeps the rewrite deliberately narrower than general copy
    // forwarding.
    SmallVector<memref::AllocOp> snapshotRoots;
    function.walk([&](memref::AllocOp alloc) {
      memref::CopyOp incoming;
      for (Operation *user : alloc.getResult().getUsers())
        if (auto copy = dyn_cast<memref::CopyOp>(user))
          if (copy.getTarget() == alloc.getResult()) {
            if (incoming) {
              incoming = nullptr;
              break;
            }
            incoming = copy;
          }
      if (incoming && !incoming.getSource().getDefiningOp<memref::AllocOp>())
        snapshotRoots.push_back(alloc);
    });
    for (memref::AllocOp root : snapshotRoots) {
      if (!root || root->getBlock() == nullptr)
        continue;
      memref::CopyOp rootIncoming;
      for (Operation *user : root.getResult().getUsers())
        if (auto copy = dyn_cast<memref::CopyOp>(user))
          if (copy.getTarget() == root.getResult())
            rootIncoming = copy;
      if (!rootIncoming)
        continue;
      Value external = rootIncoming.getSource();
      if (external.getType() != root.getType())
        continue;

      SmallVector<memref::AllocOp> chain;
      SmallVector<memref::CopyOp> chainCopies;
      chain.push_back(root);
      chainCopies.push_back(rootIncoming);
      Value current = root.getResult();
      bool closed = false;
      while (true) {
        memref::CopyOp outgoing;
        for (Operation *user : current.getUsers()) {
          auto copy = dyn_cast<memref::CopyOp>(user);
          if (!copy || copy.getSource() != current)
            continue;
          if (outgoing) {
            outgoing = nullptr;
            break;
          }
          outgoing = copy;
        }
        if (!outgoing)
          break;
        chainCopies.push_back(outgoing);
        Value next = outgoing.getTarget();
        if (next == external) {
          closed = true;
          break;
        }
        auto nextAlloc = next.getDefiningOp<memref::AllocOp>();
        if (!nextAlloc || next.getType() != external.getType())
          break;
        memref::CopyOp uniqueIncoming;
        for (Operation *user : next.getUsers())
          if (auto copy = dyn_cast<memref::CopyOp>(user))
            if (copy.getTarget() == next) {
              if (uniqueIncoming) {
                uniqueIncoming = nullptr;
                break;
              }
              uniqueIncoming = copy;
            }
        if (uniqueIncoming != outgoing)
          break;
        chain.push_back(nextAlloc);
        current = next;
      }
      if (!closed)
        continue;
      for (memref::CopyOp copy : chainCopies)
        copy.erase();
      for (memref::AllocOp alloc : chain)
        alloc.getResult().replaceAllUsesWith(external);
      for (memref::AllocOp alloc : llvm::reverse(chain))
        if (alloc.getResult().use_empty())
          alloc.erase();
    }

    // LowerSubmap can also expose plain perfect copy/fill nests that carry no
    // tensor iter_arg. Promote only the trivially injective form: one store in
    // the innermost body, with every induction variable used directly as a
    // distinct destination index. This covers embedding/cache view copies
    // without speculating about modulo, clamped, or reduction-like writes.
    SmallVector<scf::ForOp> injectiveCopies;
    function.walk([&](scf::ForOp loop) {
      if (!isa_and_nonnull<scf::ForOp>(loop->getParentOp()))
        injectiveCopies.push_back(loop);
    });
    for (scf::ForOp outer : injectiveCopies) {
      if (outer.getNumRegionIterArgs() != 0 || outer.getNumResults() != 0)
        continue;
      SmallVector<scf::ForOp> nest;
      scf::ForOp current = outer;
      bool perfect = true;
      while (current) {
        if (current.getNumRegionIterArgs() != 0 || current.getNumResults() != 0) {
          perfect = false;
          break;
        }
        nest.push_back(current);
        scf::ForOp child;
        bool sawNonLoop = false;
        for (Operation &op : current.getBody()->without_terminator()) {
          auto candidate = dyn_cast<scf::ForOp>(op);
          if (!candidate) {
            sawNonLoop = true;
            continue;
          }
          if (child || sawNonLoop) {
            child = nullptr;
            perfect = false;
            break;
          }
          child = candidate;
        }
        if (!perfect)
          break;
        // The innermost loop contains the scalar load/store body. Every
        // enclosing level of a perfect nest must contain only its child loop.
        if (!child)
          break;
        if (sawNonLoop) {
          perfect = false;
          break;
        }
        current = child;
      }
      if (!perfect || nest.empty())
        continue;
      scf::ForOp inner = nest.back();
      memref::StoreOp store;
      for (Operation &op : inner.getBody()->without_terminator()) {
        if (auto candidate = dyn_cast<memref::StoreOp>(op)) {
          if (store) {
            store = nullptr;
            break;
          }
          store = candidate;
        } else if (op.getNumRegions() != 0 || op.hasTrait<OpTrait::IsTerminator>()) {
          store = nullptr;
          break;
        }
      }
      if (!store)
        continue;
      llvm::SmallDenseSet<unsigned> usedStoreDims;
      bool injective = true;
      for (scf::ForOp loop : nest) {
        bool found = false;
        for (auto [index, storeIndex] : llvm::enumerate(store.getIndices())) {
          if (storeIndex == loop.getInductionVar() &&
              usedStoreDims.insert(index).second) {
            found = true;
            break;
          }
        }
        if (!found) {
          injective = false;
          break;
        }
      }
      if (!injective)
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
      for (auto [loop, iv] : llvm::zip(nest, parallel.getInductionVars()))
        mapping.map(loop.getInductionVar(), iv);
      builder.setInsertionPointToStart(parallel.getBody());
      for (Operation &op : inner.getBody()->without_terminator())
        builder.clone(op, mapping);
      outer.erase();
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

    // Dynamic scratch that also participates in host/library descriptor
    // operations cannot always be represented as gpu.alloc by the current
    // GPU-to-LLVM conversion.  Keep such allocations target-neutral in the
    // residency planner, then map them here for generated kernels.  Emitting
    // registration immediately after the allocation keeps it outside later
    // CUDA Graph scopes and makes the rule apply to arbitrary raised C code,
    // not to a particular benchmark harness.
    SmallVector<memref::AllocOp> localAllocations;
    function.walk([&](memref::AllocOp alloc) {
      if (reachesGpuLaunch(alloc.getResult()))
        localAllocations.push_back(alloc);
    });
    for (memref::AllocOp alloc : localAllocations) {
      auto type = alloc.getType();
      auto unranked =
          UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
      OpBuilder afterAlloc(alloc);
      afterAlloc.setInsertionPointAfter(alloc);
      Value cast =
          afterAlloc.create<memref::CastOp>(alloc.getLoc(), unranked, alloc);
      afterAlloc.create<gpu::HostRegisterOp>(alloc.getLoc(), cast);

      // Preserve the dialect-level lifetime contract for allocations that do
      // have an explicit deallocation.  (Application scratch produced by
      // one-shot bufferization is frequently function-lifetime and has none.)
      SmallVector<memref::DeallocOp> deallocations;
      for (Operation *user : alloc.getResult().getUsers())
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          deallocations.push_back(dealloc);
      for (memref::DeallocOp dealloc : deallocations) {
        OpBuilder beforeDealloc(dealloc);
        beforeDealloc.create<gpu::HostUnregisterOp>(dealloc.getLoc(), cast);
      }
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

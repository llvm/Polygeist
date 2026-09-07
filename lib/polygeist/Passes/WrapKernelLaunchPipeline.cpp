//===- WrapKernelLaunchPipeline.cpp - runtime pipeline scopes -------------===//
//
// Inserts begin/end calls around functions that contain matched kernel
// dispatches. The runtime can use this explicit scope to keep CUDA mappings,
// temporary allocations, descriptors, streams, and future device-resident
// values alive across a sequence of lowered library calls.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "KernelLaunchLoweringUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Passes/Passes.h"

#include <limits>

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace {

static bool isRuntimePipelineCall(func::CallOp call, StringRef beginSymbol,
                                  StringRef endSymbol) {
  StringRef callee = call.getCallee();
  return callee == beginSymbol || callee == endSymbol;
}

static bool isCudaShimCall(func::CallOp call) {
  StringRef callee = call.getCallee();
  if (!callee.startswith("polygeist_"))
    return false;
  if (callee.startswith("polygeist_cublas_pipeline_"))
    return false;
  if (callee.startswith("polygeist_cuda_graph_"))
    return false;
  return callee.startswith("polygeist_cublas_") ||
         callee.startswith("polygeist_cudnn_") ||
         callee.startswith("polygeist_cutensor_") ||
         callee.startswith("polygeist_cutensornet_") ||
         callee.startswith("polygeist_cuda_") ||
         callee.startswith("polygeist_rmsnorm_") ||
         callee.startswith("polygeist_whisper_");
}

// A generated GPU wrapper is not required to use a polygeist_* symbol.  The
// attribute is an explicit ABI promise that the wrapper only enqueues
// asynchronous device work on polygeist_cuda_graph_stream(), has no host
// result, and performs no synchronization. Accept the promise on either the
// call or its symbol declaration so outlining/code generation can mark the
// generated function once rather than rewriting every call site.
static bool hasCudaGraphSafeAttr(func::CallOp call) {
  if (call->hasAttr("polygeist.cuda_graph_safe"))
    return true;
  auto module = call->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  if (auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee()))
    return callee->hasAttr("polygeist.cuda_graph_safe");
  return false;
}

static bool isCudaDispatchCall(func::CallOp call) {
  return isCudaShimCall(call) || hasCudaGraphSafeAttr(call);
}

static bool isGeneratedCudaLaunch(Operation *op) {
  auto launch = dyn_cast<gpu::LaunchFuncOp>(op);
  return launch && launch.getNumResults() == 0 &&
         launch->hasAttr("polygeist.cuda_graph_safe");
}

// Host operations admitted between two device dispatches by the maximal
// graph mode. They may prepare scalar descriptors or views, but cannot read or
// write application tensor elements. They execute during warmup and capture;
// replay uses the graph nodes instantiated from that prepared metadata.
static bool isCudaGraphMetadataOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name.startswith("arith.") || name.startswith("shape.") ||
      name == "affine.apply")
    return true;
  if (name == "memref.alloca" || name == "memref.cast" ||
      name == "memref.subview" || name == "memref.reinterpret_cast" ||
      name == "memref.dim" || name == "memref.get_global" ||
      name == "memref.extract_strided_metadata" ||
      name == "memref.extract_aligned_pointer_as_index" ||
      name == "memref.store" || name == "gpu.host_register")
    return true;
  if (name == "llvm.inttoptr" || name == "llvm.ptrtoint" ||
      name == "builtin.unrealized_conversion_cast")
    return true;
  return false;
}

static bool isCudaDispatchOperation(Operation *op) {
  if (auto call = dyn_cast<func::CallOp>(op))
    return isCudaDispatchCall(call);
  return isGeneratedCudaLaunch(op);
}

static bool isCudaGraphSafeCall(func::CallOp call,
                                bool captureHostMappedCutensornet,
                                bool captureHostMappedLibraries);

static bool isCudaGraphSafeOperation(Operation *op,
                                     bool captureHostMappedCutensornet,
                                     bool captureHostMappedLibraries) {
  if (auto call = dyn_cast<func::CallOp>(op))
    return isCudaGraphSafeCall(call, captureHostMappedCutensornet,
                               captureHostMappedLibraries);
  return isGeneratedCudaLaunch(op);
}

static bool isCudaGraphSafeCall(func::CallOp call,
                                bool captureHostMappedCutensornet,
                                bool captureHostMappedLibraries) {
  if (call.getNumResults() != 0)
    return false;
  if (hasCudaGraphSafeAttr(call))
    return true;
  if (!isCudaShimCall(call))
    return false;
  StringRef callee = call.getCallee();
  if (captureHostMappedCutensornet &&
      (callee == "polygeist_cutensornet_contraction2_f64" ||
       callee == "polygeist_cutensornet_network_f32" ||
       callee == "polygeist_cutensornet_network_f64"))
    return true;
  if (!captureHostMappedLibraries)
    return false;
  // Deliberately exact: each entry has been audited to enqueue only work on
  // the shared stream after its first (warmup) invocation has populated
  // descriptors, plans, mappings, and internal buffers.
  return callee == "polygeist_cublas_sgemm" ||
         callee == "polygeist_cublas_sgemv" ||
         callee == "polygeist_cublas_sgemv_T" ||
         callee == "polygeist_cublas_memset_zero_1d_f32" ||
         callee == "polygeist_cublas_memset_zero_2d_f32" ||
         callee == "polygeist_cudnn_pointwise_graph_f32" ||
         callee == "polygeist_cudnn_softmax_forward_out_f32" ||
         callee == "polygeist_cutensor_permute_f32" ||
         callee == "polygeist_rmsnorm_f32";
}

static bool alreadyGraphWrapped(func::FuncOp func) {
  bool found = false;
  func.walk([&](scf::IfOp ifOp) {
    if (ifOp->hasAttr("polygeist.cuda_graph_scope")) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static func::FuncOp ensureGraphBeginDecl(ModuleOp module, StringRef symbol,
                                         OpBuilder &builder) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(symbol))
    return existing;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto type =
      builder.getFunctionType({builder.getI64Type()}, {builder.getI32Type()});
  auto function = builder.create<func::FuncOp>(module.getLoc(), symbol, type);
  function.setPrivate();
  return function;
}

static void wrapCudaGraphRuns(func::FuncOp func, func::FuncOp graphBegin,
                              func::FuncOp graphEnd, int64_t &nextGraphId,
                              bool captureHostMappedCutensornet,
                              bool captureHostMappedLibraries,
                              bool maximalDeviceSequence) {
  if (alreadyGraphWrapped(func))
    return;

  SmallVector<Block *> blocks;
  func.walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        blocks.push_back(&block);
  });

  for (Block *block : blocks) {
    SmallVector<SmallVector<Operation *>> runs;
    if (maximalDeviceSequence) {
      Operation *first = nullptr;
      Operation *last = nullptr;
      for (Operation &op : *block) {
        if (isCudaGraphSafeOperation(&op, captureHostMappedCutensornet,
                                     captureHostMappedLibraries)) {
          first = first ? first : &op;
          last = &op;
        }
      }
      bool invalid = false;
      if (first && last)
        for (Operation *op = first; op != last; op = op->getNextNode())
          if (op != first &&
              !isCudaGraphSafeOperation(op, captureHostMappedCutensornet,
                                        captureHostMappedLibraries) &&
              !isCudaGraphMetadataOperation(op))
            invalid = true;
      if (first && last && !invalid) {
        SmallVector<Operation *> run;
        for (Operation *op = first;; op = op->getNextNode()) {
          run.push_back(op);
          if (op == last)
            break;
        }
        runs.push_back(std::move(run));
      }
    }
    if (maximalDeviceSequence && !runs.empty()) {
      // The common case is one function-level block. Nested control flow is
      // conservatively handled by the original consecutive-run logic below.
    } else {
      SmallVector<Operation *> current;
      for (Operation &op : *block) {
        if (isCudaGraphSafeOperation(&op, captureHostMappedCutensornet,
                                     captureHostMappedLibraries)) {
          current.push_back(&op);
          continue;
        }
        if (!current.empty()) {
          runs.push_back(std::move(current));
          current.clear();
        }
      }
      if (!current.empty())
        runs.push_back(std::move(current));
    }

    for (SmallVector<Operation *> &run : runs) {
      Operation *first = run.front();
      Location loc = first->getLoc();
      OpBuilder builder(first);
      Value id = builder.create<arith::ConstantIntOp>(loc, nextGraphId++, 64);
      auto begin =
          builder.create<func::CallOp>(loc, graphBegin, ValueRange{id});
      Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      Value execute = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, begin.getResult(0), zero);
      auto ifOp = builder.create<scf::IfOp>(loc, execute,
                                            /*withElseRegion=*/false);
      ifOp->setAttr("polygeist.cuda_graph_scope", builder.getUnitAttr());

      Operation *yield = ifOp.thenBlock()->getTerminator();
      for (Operation *op : run)
        op->moveBefore(yield);
      OpBuilder endBuilder(yield);
      endBuilder.create<func::CallOp>(loc, graphEnd, ValueRange{id});
    }
  }
}

// Operations that only construct scalar metadata or tensor/memref views may
// remain between asynchronous library calls.  Anything capable of executing
// host-side tensor computation is a boundary: the stream must be synchronized
// before that operation can consume a preceding GPU result.
static bool isPipelineTransparent(Operation *op, StringRef beginSymbol,
                                  StringRef endSymbol) {
  if (isGeneratedCudaLaunch(op))
    return true;
  if (auto call = dyn_cast<func::CallOp>(op))
    return isCudaDispatchCall(call) ||
           isRuntimePipelineCall(call, beginSymbol, endSymbol);

  StringRef name = op->getName().getStringRef();
  if (name.startswith("arith.") || name.startswith("shape."))
    return true;
  if (name == "tensor.empty" || name == "tensor.cast" || name == "tensor.dim" ||
      name == "tensor.extract_slice" || name == "tensor.collapse_shape" ||
      name == "tensor.expand_shape")
    return true;
  if (name == "bufferization.to_tensor" || name == "bufferization.to_memref")
    return true;
  if (name == "memref.cast" || name == "memref.subview" ||
      name == "memref.reinterpret_cast" || name == "memref.dim")
    return true;
  if (name == "polygeist.submap")
    return true;
  if (name == "builtin.unrealized_conversion_cast")
    return true;
  return false;
}

static bool containsRawKernelLaunch(func::FuncOp func) {
  bool found = false;
  func.walk([&](LaunchOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

static bool containsCudaDispatchCall(func::FuncOp func, StringRef beginSymbol,
                                     StringRef endSymbol) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (isRuntimePipelineCall(call, beginSymbol, endSymbol))
        return WalkResult::advance();
    }
    if (isCudaDispatchOperation(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool containsGeneratedCudaLaunch(func::FuncOp func) {
  bool found = false;
  func.walk([&](gpu::LaunchFuncOp launch) {
    if (!isGeneratedCudaLaunch(launch))
      return WalkResult::advance();
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

static bool alreadyWrapped(func::FuncOp func, StringRef beginSymbol,
                           StringRef endSymbol) {
  bool sawBegin = false;
  bool sawEnd = false;
  func.walk([&](func::CallOp call) {
    StringRef callee = call.getCallee();
    sawBegin |= callee == beginSymbol;
    sawEnd |= callee == endSymbol;
  });
  return sawBegin || sawEnd;
}

// Return true only for operations that can execute while device work remains
// outstanding. In particular, memory reads/writes and unknown calls are not
// metadata operations: they could consume a preceding device result.
static bool isControlFlowCoalescingSafe(Operation *op, StringRef beginSymbol,
                                        StringRef endSymbol,
                                        StringRef graphBeginSymbol,
                                        StringRef graphEndSymbol) {
  if (auto call = dyn_cast<func::CallOp>(op)) {
    StringRef callee = call.getCallee();
    return isRuntimePipelineCall(call, beginSymbol, endSymbol) ||
           callee == graphBeginSymbol || callee == graphEndSymbol ||
           isCudaDispatchCall(call);
  }
  if (isGeneratedCudaLaunch(op))
    return true;

  StringRef name = op->getName().getStringRef();
  if (name.startswith("arith.") || name.startswith("shape.") ||
      name == "affine.apply")
    return true;
  if (name == "tensor.empty" || name == "tensor.cast" ||
      name == "tensor.dim" || name == "tensor.extract_slice" ||
      name == "tensor.collapse_shape" || name == "tensor.expand_shape" ||
      name == "tensor.insert_slice")
    return true;
  if (name == "bufferization.to_tensor" ||
      name == "bufferization.to_memref")
    return true;
  if (name == "memref.cast" || name == "memref.subview" ||
      name == "memref.reinterpret_cast" || name == "memref.dim" ||
      name == "memref.extract_strided_metadata" ||
      name == "memref.extract_aligned_pointer_as_index")
    return true;
  if (name == "llvm.inttoptr" || name == "llvm.ptrtoint" ||
      name == "builtin.unrealized_conversion_cast" ||
      name == "polygeist.submap")
    return true;
  if (name == "affine.yield" || name == "scf.yield")
    return true;

  // Loops and conditionals themselves are safe only when every operation in
  // every region is safe. This deliberately excludes memory-effecting ops
  // that happen to be nested below otherwise innocuous control flow.
  if (name == "affine.for" || name == "scf.for" || name == "scf.if") {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          if (!isControlFlowCoalescingSafe(
                  &nested, beginSymbol, endSymbol, graphBeginSymbol,
                  graphEndSymbol))
            return false;
    return true;
  }
  return false;
}

static Operation *getEntryBlockAncestor(Operation *op, func::FuncOp func) {
  while (op->getParentOp() && op->getParentOp() != func.getOperation())
    op = op->getParentOp();
  return op->getParentOp() == func.getOperation() ? op : nullptr;
}

// Existing ABI lowering may have placed a pipeline scope inside a loop around
// each individual dispatch. When the complete entry-block span from the first
// to the last dispatch contains only device operations and metadata/control
// flow, lift those scopes outside the span. This converts O(iterations)
// synchronizations into one without relying on benchmark names or shapes.
static bool coalescePipelineScopes(func::FuncOp func, StringRef beginSymbol,
                                   StringRef endSymbol,
                                   StringRef graphBeginSymbol,
                                   StringRef graphEndSymbol) {
  if (func.isDeclaration() || func.empty())
    return false;

  SmallVector<func::CallOp> scopeCalls;
  SmallVector<Operation *> relevantOps;
  unsigned begins = 0;
  unsigned ends = 0;
  func.walk([&](Operation *op) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (isRuntimePipelineCall(call, beginSymbol, endSymbol)) {
        scopeCalls.push_back(call);
        begins += call.getCallee() == beginSymbol;
        ends += call.getCallee() == endSymbol;
        return;
      }
    }
    if (isCudaDispatchOperation(op))
      relevantOps.push_back(op);
  });
  if (begins == 0 || begins != ends || relevantOps.empty())
    return false;

  Block &entry = func.getBody().front();
  DenseMap<Operation *, unsigned> positions;
  unsigned position = 0;
  for (Operation &op : entry)
    positions[&op] = position++;

  Operation *first = nullptr;
  Operation *last = nullptr;
  unsigned firstPosition = std::numeric_limits<unsigned>::max();
  unsigned lastPosition = 0;
  for (Operation *op : relevantOps) {
    Operation *ancestor = getEntryBlockAncestor(op, func);
    if (!ancestor || ancestor->getBlock() != &entry)
      return false;
    unsigned current = positions.lookup(ancestor);
    if (!first || current < firstPosition) {
      first = ancestor;
      firstPosition = current;
    }
    if (!last || current > lastPosition) {
      last = ancestor;
      lastPosition = current;
    }
  }

  for (Operation *op = first;; op = op->getNextNode()) {
    if (!isControlFlowCoalescingSafe(op, beginSymbol, endSymbol,
                                     graphBeginSymbol, graphEndSymbol))
      return false;
    if (op == last)
      break;
  }

  Location beginLoc = first->getLoc();
  Location endLoc = last->getLoc();
  for (func::CallOp call : scopeCalls)
    call.erase();
  OpBuilder beginBuilder(first);
  beginBuilder.create<func::CallOp>(beginLoc, beginSymbol, TypeRange{},
                                    ValueRange{});
  OpBuilder endBuilder(last);
  endBuilder.setInsertionPointAfter(last);
  endBuilder.create<func::CallOp>(endLoc, endSymbol, TypeRange{}, ValueRange{});
  func->setAttr("polygeist.pipeline_scope_coalesced",
                UnitAttr::get(func.getContext()));
  return true;
}

struct WrapKernelLaunchPipelinePass
    : public mlir::polygeist::WrapKernelLaunchPipelineBase<
          WrapKernelLaunchPipelinePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    OpBuilder moduleBuilder(ctx);

    SmallVector<func::FuncOp> funcs;
    module.walk([&](func::FuncOp func) { funcs.push_back(func); });

    if (useCudaGraphs) {
      func::FuncOp graphBegin =
          ensureGraphBeginDecl(module, graphBeginSymbol, moduleBuilder);
      func::FuncOp graphEnd =
          ensureShimDecl(module, graphEndSymbol,
                         TypeRange{moduleBuilder.getI64Type()}, moduleBuilder);
      int64_t nextGraphId = 0;
      for (func::FuncOp func : funcs)
        if (!func.isDeclaration())
          wrapCudaGraphRuns(func, graphBegin, graphEnd, nextGraphId,
                            captureHostMappedCutensornet,
                            captureHostMappedLibraries,
                            maximalDeviceSequence);
    }

    bool needsDeclarations = false;
    for (func::FuncOp func : funcs) {
      if (func.isDeclaration())
        continue;
      if (alreadyWrapped(func, beginSymbol, endSymbol))
        continue;
      if (containsCudaDispatchCall(func, beginSymbol, endSymbol) ||
          containsRawKernelLaunch(func) || containsGeneratedCudaLaunch(func)) {
        needsDeclarations = true;
        break;
      }
    }

    if (needsDeclarations) {
      ensureShimDecl(module, beginSymbol, TypeRange{}, moduleBuilder);
      ensureShimDecl(module, endSymbol, TypeRange{}, moduleBuilder);
    }

    for (func::FuncOp func : funcs) {
      if (func.isDeclaration())
        continue;
      if (alreadyWrapped(func, beginSymbol, endSymbol))
        continue;
      if (!containsCudaDispatchCall(func, beginSymbol, endSymbol) &&
          !containsRawKernelLaunch(func) && !containsGeneratedCudaLaunch(func))
        continue;

      // Form maximal GPU-only regions independently in every block. This is
      // conservative across control-flow edges but remains correct: a region
      // always ends before host computation or a block terminator.
      SmallVector<Block *> blocks;
      func.walk([&](Operation *op) {
        for (Region &region : op->getRegions())
          for (Block &block : region)
            blocks.push_back(&block);
      });
      for (Block *block : blocks) {
        SmallVector<Operation *> operations;
        for (Operation &op : *block)
          operations.push_back(&op);

        bool active = false;
        for (Operation *op : operations) {
          bool cudaCall = isCudaDispatchOperation(op);
          if (cudaCall && !active) {
            OpBuilder beginBuilder(op);
            beginBuilder.create<func::CallOp>(op->getLoc(), beginSymbol,
                                              TypeRange{}, ValueRange{});
            active = true;
          }
          if (active && !cudaCall &&
              !isPipelineTransparent(op, beginSymbol, endSymbol)) {
            OpBuilder endBuilder(op);
            endBuilder.create<func::CallOp>(op->getLoc(), endSymbol,
                                            TypeRange{}, ValueRange{});
            active = false;
          }
        }
        if (active) {
          Operation *terminator = block->getTerminator();
          OpBuilder endBuilder(terminator);
          endBuilder.create<func::CallOp>(terminator->getLoc(), endSymbol,
                                          TypeRange{}, ValueRange{});
        }
      }
    }

    if (coalesceControlFlow) {
      for (func::FuncOp func : funcs)
        coalescePipelineScopes(func, beginSymbol, endSymbol, graphBeginSymbol,
                               graphEndSymbol);
    }
  }
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createWrapKernelLaunchPipelinePass() {
  return std::make_unique<WrapKernelLaunchPipelinePass>();
}
} // namespace polygeist
} // namespace mlir

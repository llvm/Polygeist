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
      name == "memref.store")
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
                                bool captureHostMappedCutensornet);

static bool isCudaGraphSafeOperation(Operation *op,
                                     bool captureHostMappedCutensornet) {
  if (auto call = dyn_cast<func::CallOp>(op))
    return isCudaGraphSafeCall(call, captureHostMappedCutensornet);
  return isGeneratedCudaLaunch(op);
}

static bool isCudaGraphSafeCall(func::CallOp call,
                                bool captureHostMappedCutensornet) {
  if (call.getNumResults() != 0)
    return false;
  if (hasCudaGraphSafeAttr(call))
    return true;
  if (!isCudaShimCall(call))
    return false;
  return captureHostMappedCutensornet &&
         (call.getCallee() == "polygeist_cutensornet_contraction2_f64" ||
          call.getCallee() == "polygeist_cutensornet_network_f32" ||
          call.getCallee() == "polygeist_cutensornet_network_f64");
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
        if (isCudaGraphSafeOperation(&op, captureHostMappedCutensornet)) {
          first = first ? first : &op;
          last = &op;
        }
      }
      bool invalid = false;
      if (first && last)
        for (Operation *op = first; op != last; op = op->getNextNode())
          if (op != first &&
              !isCudaGraphSafeOperation(op, captureHostMappedCutensornet) &&
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
        if (isCudaGraphSafeOperation(&op, captureHostMappedCutensornet)) {
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

    if (!needsDeclarations)
      return;

    ensureShimDecl(module, beginSymbol, TypeRange{}, moduleBuilder);
    ensureShimDecl(module, endSymbol, TypeRange{}, moduleBuilder);

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

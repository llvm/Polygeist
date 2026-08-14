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

static bool isCudaGraphSafeCall(func::CallOp call,
                                bool captureHostMappedCutensornet) {
  if (!isCudaShimCall(call) || call.getNumResults() != 0)
    return false;
  if (call->hasAttr("polygeist.cuda_graph_safe"))
    return true;
  return captureHostMappedCutensornet &&
         call.getCallee() == "polygeist_cutensornet_contraction2_f64";
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
  auto type = builder.getFunctionType({builder.getI64Type()},
                                      {builder.getI32Type()});
  auto function =
      builder.create<func::FuncOp>(module.getLoc(), symbol, type);
  function.setPrivate();
  return function;
}

static void wrapCudaGraphRuns(func::FuncOp func, func::FuncOp graphBegin,
                              func::FuncOp graphEnd, int64_t &nextGraphId,
                              bool captureHostMappedCutensornet) {
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
    SmallVector<Operation *> current;
    for (Operation &op : *block) {
      auto call = dyn_cast<func::CallOp>(&op);
      if (call &&
          isCudaGraphSafeCall(call, captureHostMappedCutensornet)) {
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

    for (SmallVector<Operation *> &run : runs) {
      Operation *first = run.front();
      Location loc = first->getLoc();
      OpBuilder builder(first);
      Value id = builder.create<arith::ConstantIntOp>(loc, nextGraphId++, 64);
      auto begin = builder.create<func::CallOp>(loc, graphBegin, ValueRange{id});
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
  if (auto call = dyn_cast<func::CallOp>(op))
    return isCudaShimCall(call) ||
           isRuntimePipelineCall(call, beginSymbol, endSymbol);

  StringRef name = op->getName().getStringRef();
  if (name.startswith("arith.") || name.startswith("shape."))
    return true;
  if (name == "tensor.empty" || name == "tensor.cast" ||
      name == "tensor.dim" || name == "tensor.extract_slice" ||
      name == "tensor.collapse_shape" || name == "tensor.expand_shape")
    return true;
  if (name == "bufferization.to_tensor" ||
      name == "bufferization.to_memref")
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

static bool containsCudaShimCall(func::FuncOp func, StringRef beginSymbol,
                                 StringRef endSymbol) {
  bool found = false;
  func.walk([&](func::CallOp call) {
    if (isRuntimePipelineCall(call, beginSymbol, endSymbol))
      return WalkResult::advance();
    if (isCudaShimCall(call)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
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
      func::FuncOp graphEnd = ensureShimDecl(
          module, graphEndSymbol, TypeRange{moduleBuilder.getI64Type()},
          moduleBuilder);
      int64_t nextGraphId = 0;
      for (func::FuncOp func : funcs)
        if (!func.isDeclaration())
          wrapCudaGraphRuns(func, graphBegin, graphEnd, nextGraphId,
                            captureHostMappedCutensornet);
    }

    bool needsDeclarations = false;
    for (func::FuncOp func : funcs) {
      if (func.isDeclaration())
        continue;
      if (alreadyWrapped(func, beginSymbol, endSymbol))
        continue;
      if (containsCudaShimCall(func, beginSymbol, endSymbol) ||
          containsRawKernelLaunch(func)) {
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
      if (!containsCudaShimCall(func, beginSymbol, endSymbol) &&
          !containsRawKernelLaunch(func))
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
          auto call = dyn_cast<func::CallOp>(op);
          bool cudaCall = call && isCudaShimCall(call);
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

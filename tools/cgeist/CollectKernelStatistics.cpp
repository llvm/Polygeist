#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Program.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include "polygeist/Dialect.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"

namespace {
using namespace mlir;

// Per cuda block trip count
static double estimateTripCount(Block *block, unsigned threadNum) {
  auto op = block->getParentOp();
  if (isa<gpu::GPUFuncOp>(op)) {
    return threadNum;
  }
  double curBlockTripCount = [&]() -> double {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto lbCstOp =
          forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
      auto ubCstOp =
          forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
      auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
      if (lbCstOp && ubCstOp && stepCstOp)
        return mlir::ceilDiv(ubCstOp.value() - lbCstOp.value(),
                             stepCstOp.value());
      else
        return 1.0;
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // We assume both then and else of a non-root ifOps have a tripCount of 1
      if (!isa<gpu::LaunchFuncOp>(ifOp->getParentOp()))
        return 1.0;
      auto condOp = ifOp.getCondition().getDefiningOp();
      if (auto cmpOp = dyn_cast<arith::CmpIOp>(condOp)) {
        bool isThen = block->getParent() == &ifOp.getThenRegion();
        if (cmpOp.getPredicate() == arith::CmpIPredicate::eq) {
          if (isThen) {
            // Assume it is an if that executes once per block
            return 1.0 / threadNum;
          } else {
            // Assume it always executes
            return 1.0;
          }
        } else if (cmpOp.getPredicate() == arith::CmpIPredicate::ne) {
          if (!isThen) {
            // Assume it is an if that executes once per block
            return 1.0 / threadNum;
          } else {
            // Assume it always executes
            return 1.0;
          }
        } else {
          // TODO the programemr may have written something like "tid < 1" or
          // "tid <= 1", check for that

          // Assume it always executes
          return 1.0;
        }
      } else {
        // Assume it always executes
        return 1.0;
      }
    } else {
      // What else?
      return 1.0;
    }
  }();

  return curBlockTripCount * estimateTripCount(op->getBlock(), threadNum);
  // TODO use memoization
}

int estimateStride(mlir::OperandRange indices) {
  const int UNKNOWN = -1;

  auto isTidX = [&](mlir::Value v) -> bool {
    if (auto op = v.getDefiningOp())
      if (auto threadIdx = dyn_cast<gpu::ThreadIdOp>(op))
        if (threadIdx.getDimension() == gpu::Dimension::x)
          return true;
    return false;
  };
  std::function<int64_t(mlir::Value)> getValue = [&](mlir::Value v) -> int64_t {
    if (auto op = v.getDefiningOp()) {
      if (auto constIndexOp = dyn_cast<arith::ConstantIndexOp>(op)) {
        return constIndexOp.value();
      } else if (auto constIntOp = dyn_cast<arith::ConstantIntOp>(op)) {
        return constIntOp.value();
      } else {
        return UNKNOWN;
      }
    } else {
      return UNKNOWN;
    }
  };
  std::function<int64_t(mlir::Value)> getTidXMultiplier =
      [&](mlir::Value v) -> int64_t {
    if (isTidX(v))
      return 1;
    if (auto op = v.getDefiningOp()) {
      if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
        return getTidXMultiplier(addOp.getLhs()) +
               getTidXMultiplier(addOp.getLhs());
      } else if (auto subOp = dyn_cast<arith::SubIOp>(op)) {
        return getTidXMultiplier(subOp.getLhs()) +
               getTidXMultiplier(subOp.getLhs());
      } else if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
        if (isTidX(mulOp.getLhs()))
          return getValue(mulOp.getRhs());
        if (isTidX(mulOp.getRhs()))
          return getValue(mulOp.getLhs());
      } else if (auto constIndexOp = dyn_cast<arith::ConstantIndexOp>(op)) {
        return 0;
      } else if (auto constIntOp = dyn_cast<arith::ConstantIntOp>(op)) {
        return 0;
      } else {
        return 0;
      }
    }
    return 0;
  };

  // Let us consider only the last index for now (not a problem unless the last
  // dimension of the memref is less than 32
  auto index = indices.back();
  int stride = getTidXMultiplier(index);
  if (stride < 0)
    stride = -1;
  return stride;
}

static void generateAlternativeKernelDescs(mlir::ModuleOp m) {
  // Generate alternative kernel annotations
  m->walk([&](polygeist::AlternativesOp aop) {
    if (aop->getAttrOfType<StringAttr>("alternatives.type").getValue() !=
        "gpu_kernel")
      return;

    auto oldDescs = aop->getAttrOfType<ArrayAttr>("alternatives.descs");
    std::vector<mlir::Attribute> descs;

    unsigned regionId = 0;
    for (auto &region : aop->getRegions()) {
      gpu::LaunchFuncOp launchOp = nullptr;
      region.walk([&](gpu::LaunchFuncOp l) {
        launchOp = l;
        return WalkResult::interrupt();
      });
      assert(launchOp);
      auto gpuFunc = launchOp->getParentOfType<ModuleOp>().lookupSymbol(
          launchOp.getKernel());

      // Assume 1024 threads per block by default
      unsigned threadNum = 1024;
      if (auto bound = gpuFunc->getAttrOfType<IntegerAttr>("nvvm.maxntidx")) {
        threadNum = bound.getInt();
      } else if (auto bound = gpuFunc->getAttrOfType<IntegerAttr>(
                     "rocdl.max_flat_work_group_size")) {
        threadNum = bound.getInt();
      }

      mlir::DataLayout DLI(aop->getParentOfType<ModuleOp>());

      unsigned f16Ops = 0;
      unsigned f32Ops = 0;
      unsigned f64Ops = 0;
      typedef std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned>
          MemOpMap;
      MemOpMap loads, stores;
      auto addTo = [&](MemOpMap &m, unsigned bytes, unsigned stride,
                       unsigned memSpace, unsigned num) {
        auto index = std::make_tuple(bytes, stride, memSpace);
        if (m.count(index))
          m[index] += num;
        else
          m[index] = num;
      };
      gpuFunc->walk([&](Block *block) {
        auto blockTrips = std::lround(estimateTripCount(block, threadNum));
        for (auto &op : *block) {
          if (isa<arith::MulFOp>(&op) || isa<arith::DivFOp>(&op) ||
              isa<arith::SubFOp>(&op) || isa<arith::AddFOp>(&op) ||
              isa<arith::RemFOp>(&op) || false) {
            int width =
                op.getOperand(0).getType().dyn_cast<FloatType>().getWidth();
            // TODO these are pretty random atm
            if (width == 16) {
              f16Ops += blockTrips;
            } else if (width == 32) {
              f32Ops += blockTrips;
            } else if (width == 64) {
              f64Ops += blockTrips;
            }
          } else if (auto load = dyn_cast<memref::LoadOp>(&op)) {
            int bytes = DLI.getTypeSize(load.getResult().getType());
            auto stride = estimateStride(load.getIndices());
            addTo(loads, bytes, stride,
                  load.getMemRefType().getMemorySpaceAsInt(), blockTrips);
          } else if (auto store = dyn_cast<memref::StoreOp>(&op)) {
            int bytes = DLI.getTypeSize(store.getValue().getType());
            auto stride = estimateStride(store.getIndices());
            addTo(loads, bytes, stride,
                  store.getMemRefType().getMemorySpaceAsInt(), blockTrips);
          }
        }
      });

      auto toString = [&](MemOpMap m) {
        std::string s = "";
        for (auto &[k, v] : m) {
          s += std::to_string(std::get<0>(k));
          s += ":";
          s += std::to_string(std::get<1>(k));
          s += ":";
          s += std::to_string(std::get<2>(k));
          s += "=";
          s += std::to_string(v);
          s += ";";
        }
        return s;
      };

      std::string newDesc = oldDescs[regionId].cast<StringAttr>().str() +
                            "f16=" + std::to_string(f16Ops) + "," +
                            "f32=" + std::to_string(f32Ops) + "," +
                            "f64=" + std::to_string(f64Ops) + "," +
                            "loads=" + toString(loads) + "," +
                            "stores=" + toString(stores) + ",";
      descs.push_back(StringAttr::get(m->getContext(), newDesc));

      regionId++;
    }
    aop->setAttr("alternatives.descs", ArrayAttr::get(m->getContext(), descs));
  });
}
} // namespace

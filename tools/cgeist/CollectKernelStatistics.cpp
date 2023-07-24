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
#include "mlir/IR/Value.h"
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
#include <limits>

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

typedef llvm::Optional<int64_t> StrideTy;
StrideTy estimateStride(mlir::OperandRange indices) {
  const StrideTy UNKNOWN = {};

  auto sub = [](StrideTy a, StrideTy b) -> StrideTy {
    if (a && b)
      return a.value() - b.value();
    else
      return {};
  };
  auto mul = [](StrideTy a, StrideTy b) -> StrideTy {
    if (a && b)
      return a.value() * b.value();
    else
      return {};
  };
  auto add = [](StrideTy a, StrideTy b) -> StrideTy {
    if (a && b)
      return a.value() + b.value();
    else
      return {};
  };

  auto isTidX = [&](mlir::Value v) -> bool {
    if (auto op = v.getDefiningOp())
      if (auto threadIdx = dyn_cast<gpu::ThreadIdOp>(op))
        if (threadIdx.getDimension() == gpu::Dimension::x)
          return true;
    return false;
  };
  std::function<StrideTy(mlir::Value)> getValue =
      [&](mlir::Value v) -> StrideTy {
    if (auto op = v.getDefiningOp()) {
      if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
        // m0(f(x) + g(x)) = m0(f(x)) + m0(g(x))
        return add(getValue(addOp.getLhs()), getValue(addOp.getLhs()));
      } else if (auto subOp = dyn_cast<arith::SubIOp>(op)) {
        // m0(f(x) - g(x)) = m0(f(x)) - m0(g(x))
        return sub(getValue(subOp.getLhs()), getValue(subOp.getLhs()));
      } else if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
        // m0(f(x) * g(x)) = m0(f(x)) * m0(g(x))
        return mul(getValue(subOp.getLhs()), getValue(subOp.getLhs()));
      } else if (auto constIndexOp = dyn_cast<arith::ConstantIndexOp>(op)) {
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
  std::function<StrideTy(mlir::Value)> getTidXCoef =
      [&](mlir::Value v) -> StrideTy {
    if (isTidX(v)) {
      return 1;
    } else if (auto op = v.getDefiningOp()) {
      if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
        // m1(f(x) + g(x)) = m1(f(x)) + m1(g(x))
        return add(getTidXCoef(addOp.getLhs()), getTidXCoef(addOp.getLhs()));
      } else if (auto subOp = dyn_cast<arith::SubIOp>(op)) {
        // m1(f(x) - g(x)) = m1(f(x)) - m1(g(x))
        return sub(getTidXCoef(subOp.getLhs()), getTidXCoef(subOp.getLhs()));
      } else if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
        // m1(f(x) * g(x)) = m1(f(x)) * m0(g(x)) + m1(g(x)) * m0(f(x))
        return add(mul(getTidXCoef(mulOp.getLhs()), getValue(mulOp.getRhs())),
                   mul(getTidXCoef(mulOp.getRhs()), getValue(mulOp.getLhs())));
      } else if (auto constIndexOp = dyn_cast<arith::ConstantIndexOp>(op)) {
        return 0;
      } else if (auto constIntOp = dyn_cast<arith::ConstantIntOp>(op)) {
        return 0;
      } else {
        return UNKNOWN;
      }
    } else if (auto ba = v.dyn_cast<BlockArgument>()) {
      return 0;
      if (isa<gpu::GPUFuncOp>(ba.getOwner()->getParentOp())) {
        return 0;
      } else if (auto forOp =
                     dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp())) {
        return getTidXCoef(forOp.getOpOperandForRegionIterArg(ba).get());
      } else {
        return UNKNOWN;
      }
    } else {
      return UNKNOWN;
    }
  };

  // Let us consider only the last index for now (not a problem unless the last
  // dimension of the memref is less than 32
  auto index = indices.back();

  // Get the coefficient in front of the first degree of threadIdx.x
  return getTidXCoef(index);
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

      typedef std::map<unsigned, unsigned> ArithOpMap;
      ArithOpMap floatOps, intOps;
      typedef std::tuple<unsigned, StrideTy, unsigned> MemOpType;
      typedef std::map<MemOpType, unsigned> MemOpMap;
      MemOpMap loads, stores;
      auto addTo = [&](auto &m, auto index, unsigned num) {
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
            addTo(floatOps, width, blockTrips);
          } else if (isa<arith::MulIOp>(&op) || isa<arith::DivUIOp>(&op) ||
                     isa<arith::DivSIOp>(&op) || isa<arith::SubIOp>(&op) ||
                     isa<arith::AddIOp>(&op) || isa<arith::RemUIOp>(&op) ||
                     isa<arith::RemSIOp>(&op)) {
            int width = DLI.getTypeSize(op.getOperand(0).getType());
            addTo(intOps, width, blockTrips);
          } else if (auto load = dyn_cast<memref::LoadOp>(&op)) {
            int bytes = DLI.getTypeSize(load.getResult().getType());
            auto stride = estimateStride(load.getIndices());
            auto memSpace = load.getMemRefType().getMemorySpaceAsInt();
            addTo(loads, std::make_tuple(bytes, stride, memSpace), blockTrips);
          } else if (auto store = dyn_cast<memref::StoreOp>(&op)) {
            int bytes = DLI.getTypeSize(store.getValue().getType());
            auto stride = estimateStride(store.getIndices());
            auto memSpace = store.getMemRefType().getMemorySpaceAsInt();
            addTo(stores, std::make_tuple(bytes, stride, memSpace), blockTrips);
          }
        }
      });

      auto toStringA = [&](ArithOpMap m) {
        std::string s = "";
        for (auto &[k, v] : m) {
          s += std::to_string(k);
          s += ":";
          s += std::to_string(v);
          s += ";";
        }
        return s;
      };
      auto toStringM = [&](MemOpMap m) {
        std::string s = "";
        for (auto &[k, v] : m) {
          s += std::to_string(std::get<0>(k));
          s += "/";
          auto stride = std::get<1>(k);
          if (stride)
            s += std::to_string(stride.value());
          else
            s += "unk";
          s += "/";
          s += std::to_string(std::get<2>(k));
          s += ":";
          s += std::to_string(v);
          s += ";";
        }
        return s;
      };

      std::string newDesc = oldDescs[regionId].cast<StringAttr>().str() +
                            "floatOps=" + toStringA(floatOps) + "," +
                            "intOps=" + toStringA(intOps) + "," +
                            "loads=" + toStringM(loads) + "," +
                            "stores=" + toStringM(stores) + ",";
      descs.push_back(StringAttr::get(m->getContext(), newDesc));

      regionId++;
    }
    aop->setAttr("alternatives.descs", ArrayAttr::get(m->getContext(), descs));
  });
}
} // namespace

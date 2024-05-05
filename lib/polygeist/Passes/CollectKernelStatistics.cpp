#include "PassDetails.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "polygeist/Passes/Passes.h"

using namespace mlir;
using namespace polygeist;

extern llvm::cl::opt<PolygeistAlternativesMode> PolygeistAlternativesMode;

namespace {

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

typedef std::optional<int64_t> StrideTy;
std::array<StrideTy, 3> estimateStride(mlir::OperandRange indices,
                                       mlir::MemRefType mt,
                                       ArrayRef<int64_t> dims) {
  if (indices.size() == 0)
    return {0, 0, 0};

  const StrideTy UNKNOWN = {};

  auto sub = [](StrideTy a, StrideTy b) -> StrideTy {
    if (a && b)
      return a.value() - b.value();
    else
      return {};
  };
  auto mul = [](StrideTy a, StrideTy b) -> StrideTy {
    if ((a && a.value() == 0) || (b && b.value() == 0)) {
      return 0;
    } else if (a && b) {
      return a.value() * b.value();
    } else {
      return {};
    }
  };
  auto add = [](StrideTy a, StrideTy b) -> StrideTy {
    if (a && b)
      return a.value() + b.value();
    else
      return {};
  };

  auto isGdim = [&](mlir::Value v) -> bool {
    if (auto op = v.getDefiningOp())
      if (auto threadIdx = dyn_cast<gpu::GridDimOp>(op))
        return true;
    return false;
  };
  auto isBdim = [&](mlir::Value v) -> bool {
    if (auto op = v.getDefiningOp())
      if (auto threadIdx = dyn_cast<gpu::BlockDimOp>(op))
        return true;
    return false;
  };
  auto isBid = [&](mlir::Value v) -> bool {
    if (auto op = v.getDefiningOp())
      if (auto threadIdx = dyn_cast<gpu::BlockIdOp>(op))
        return true;
    return false;
  };
  auto isAnyTid = [&](mlir::Value v) -> bool {
    if (auto op = v.getDefiningOp())
      if (auto threadIdx = dyn_cast<gpu::ThreadIdOp>(op))
        return true;
    return false;
  };
  auto isTidDim = [&](mlir::Value v, auto dim) -> bool {
    if (auto op = v.getDefiningOp())
      if (auto threadIdx = dyn_cast<gpu::ThreadIdOp>(op))
        if (threadIdx.getDimension() == dim)
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
        return mul(getValue(mulOp.getLhs()), getValue(mulOp.getLhs()));
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
  std::function<StrideTy(mlir::Value, std::function<bool(mlir::Value)>)>
      getTidXCoef = [&](mlir::Value v,
                        std::function<bool(mlir::Value)> isTidI) -> StrideTy {
    if (isTidI(v)) {
      return 1;
    } else if (isAnyTid(v) || isBid(v) || isBdim(v) || isGdim(v)) {
      return 0;
    } else if (auto op = v.getDefiningOp()) {
      if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
        // m1(f(x) + g(x)) = m1(f(x)) + m1(g(x))
        return add(getTidXCoef(addOp.getLhs(), isTidI),
                   getTidXCoef(addOp.getLhs(), isTidI));
      } else if (auto subOp = dyn_cast<arith::SubIOp>(op)) {
        // m1(f(x) - g(x)) = m1(f(x)) - m1(g(x))
        return sub(getTidXCoef(subOp.getLhs(), isTidI),
                   getTidXCoef(subOp.getLhs(), isTidI));
      } else if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
        // m1(f(x) * g(x)) = m1(f(x)) * m0(g(x)) + m1(g(x)) * m0(f(x))
        return add(
            mul(getTidXCoef(mulOp.getLhs(), isTidI), getValue(mulOp.getRhs())),
            mul(getTidXCoef(mulOp.getRhs(), isTidI), getValue(mulOp.getLhs())));
      } else if (auto constIndexOp = dyn_cast<arith::ConstantIndexOp>(op)) {
        return 0;
      } else if (auto constIntOp = dyn_cast<arith::ConstantIntOp>(op)) {
        return 0;
      } else {
        return UNKNOWN;
      }
    } else if (auto ba = dyn_cast<BlockArgument>(v)) {
      return 0;
      if (isa<gpu::GPUFuncOp>(ba.getOwner()->getParentOp())) {
        return 0;
      } else if (auto forOp =
                     dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp())) {
        return getTidXCoef(forOp.getOpOperandForRegionIterArg(ba).get(),
                           isTidI);
      } else {
        return UNKNOWN;
      }
    } else {
      return UNKNOWN;
    }
  };

  std::array<StrideTy, 3> dimStrides;
  int i = 0;
  for (auto dim : {
           gpu::Dimension::x,
           gpu::Dimension::y,
           gpu::Dimension::z,
       }) {

    std::vector<StrideTy> strides;

    for (auto index : indices) {
      auto stride =
          getTidXCoef(index, [&](mlir::Value v) { return isTidDim(v, dim); });

      strides.push_back(stride);
    }

    StrideTy stride = strides.back();
    for (int i = strides.size() - 2; i >= 0; i--) {
      stride = add(stride, mul(strides[i], mt.getDimSize(i + 1)));
    }

    dimStrides[i++] = stride;
  }

  return dimStrides;
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

      bool isBlockDimKnown = false;
      auto blockDims = [&]() -> std::array<int64_t, 3> {
        gpu::KernelDim3 blockDims = launchOp.getBlockSizeOperandValues();
        auto x = getConstantIntValue(blockDims.x);
        auto y = getConstantIntValue(blockDims.y);
        auto z = getConstantIntValue(blockDims.z);
        if (x && y && z) {
          isBlockDimKnown = true;
          return {x.value(), y.value(), z.value()};
        } else {
          isBlockDimKnown = false;
          return {1024, 1, 1};
        }
      }();

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
      typedef std::tuple<unsigned, std::array<StrideTy, 3>, unsigned> MemOpType;
      typedef std::map<MemOpType, unsigned> MemOpMap;
      MemOpMap loads, stores;
      auto addTo = [&](auto &m, auto index, unsigned num) {
        if (m.count(index))
          m[index] += num;
        else
          m[index] = num;
      };
      auto isCudaDeviceGlobal = [&](mlir::Value mr) {
        if (auto getGlobalOp =
                dyn_cast_or_null<memref::GetGlobalOp>(mr.getDefiningOp())) {
          auto *symbolTableOp =
              getGlobalOp->getParentWithTrait<OpTrait::SymbolTable>();
          if (!symbolTableOp)
            return false;
          auto global =
              dyn_cast_or_null<memref::GlobalOp>(SymbolTable::lookupSymbolIn(
                  symbolTableOp, getGlobalOp.getNameAttr()));
          if (!global)
            return false;
          return global->hasAttr("polygeist.cuda_device");
        }
        return false;
      };
      auto isCudaConstantGlobal = [&](mlir::Value mr) {
        if (auto getGlobalOp =
                dyn_cast_or_null<memref::GetGlobalOp>(mr.getDefiningOp())) {
          auto *symbolTableOp =
              getGlobalOp->getParentWithTrait<OpTrait::SymbolTable>();
          if (!symbolTableOp)
            return false;
          auto global =
              dyn_cast_or_null<memref::GlobalOp>(SymbolTable::lookupSymbolIn(
                  symbolTableOp, getGlobalOp.getNameAttr()));
          if (!global)
            return false;
          return global->hasAttr("polygeist.cuda_constant");
        }
        return false;
      };
      gpuFunc->walk([&](Block *block) {
        auto blockTrips = std::lround(estimateTripCount(block, threadNum));
        for (auto &op : *block) {
          if (isa<arith::MulFOp>(&op) || isa<arith::DivFOp>(&op) ||
              isa<arith::SubFOp>(&op) || isa<arith::AddFOp>(&op) ||
              isa<arith::RemFOp>(&op) || false) {
            int width =
                dyn_cast<FloatType>(op.getOperand(0).getType()).getWidth();
            addTo(floatOps, width, blockTrips);
          } else if (isa<arith::MulIOp>(&op) || isa<arith::DivUIOp>(&op) ||
                     isa<arith::DivSIOp>(&op) || isa<arith::SubIOp>(&op) ||
                     isa<arith::AddIOp>(&op) || isa<arith::RemUIOp>(&op) ||
                     isa<arith::RemSIOp>(&op)) {
            int width = DLI.getTypeSize(op.getOperand(0).getType());
            addTo(intOps, width, blockTrips);
          } else if (auto load = dyn_cast<memref::LoadOp>(&op)) {
            int bytes = DLI.getTypeSize(load.getResult().getType());
            auto stride = estimateStride(load.getIndices(),
                                         load.getMemRefType(), blockDims);
            auto memSpace = load.getMemRefType().getMemorySpaceAsInt();
            if (isCudaConstantGlobal(load.getMemRef()))
              memSpace = 4;
            if (isCudaDeviceGlobal(load.getMemRef()))
              memSpace = 1;
            addTo(loads, std::make_tuple(bytes, stride, memSpace), blockTrips);
          } else if (auto store = dyn_cast<memref::StoreOp>(&op)) {
            int bytes = DLI.getTypeSize(store.getValue().getType());
            auto stride = estimateStride(
                store.getIndices(),
                store.getMemRef().getType().cast<MemRefType>(), blockDims);
            auto memSpace = store.getMemRefType().getMemorySpaceAsInt();
            if (isCudaConstantGlobal(store.getMemRef()))
              memSpace = 4;
            if (isCudaDeviceGlobal(store.getMemRef()))
              memSpace = 1;
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
          auto strides = std::get<1>(k);
          auto appendStride = [&](std::string dimStr, int dim) {
            auto stride = strides[dim];
            s += dimStr + ":";
            if (stride)
              s += std::to_string(strides[dim].value());
            else
              s += "unk";
            s += "|";
          };
          appendStride("x", 0);
          appendStride("y", 1);
          appendStride("z", 2);
          s += "/";
          s += std::to_string(std::get<2>(k));
          s += ":";
          s += std::to_string(v);
          s += ";";
        }
        return s;
      };

      std::string newDesc =
          oldDescs[regionId].cast<StringAttr>().str() + "blockDims=" +
          (isBlockDimKnown ? "x:" + std::to_string(blockDims[0]) +
                                 ";"
                                 "y:" +
                                 std::to_string(blockDims[1]) +
                                 ";"
                                 "z:" +
                                 std::to_string(blockDims[2]) + ";"
                           : "unk") +
          "," + "floatOps=" + toStringA(floatOps) + "," +
          "intOps=" + toStringA(intOps) + "," + "loads=" + toStringM(loads) +
          "," + "stores=" + toStringM(stores) + ",";
      descs.push_back(StringAttr::get(m->getContext(), newDesc));

      regionId++;
    }
    aop->setAttr("alternatives.descs", ArrayAttr::get(m->getContext(), descs));
  });
}
} // namespace

struct CollectKernelStatisticsPass
    : public CollectKernelStatisticsBase<CollectKernelStatisticsPass> {
  void runOnOperation() override {
    generateAlternativeKernelDescs(getOperation());
  }
};

std::unique_ptr<Pass> mlir::polygeist::createCollectKernelStatisticsPass() {
  return std::make_unique<CollectKernelStatisticsPass>();
}

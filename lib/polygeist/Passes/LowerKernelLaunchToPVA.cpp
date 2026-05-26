//===- LowerKernelLaunchToPVA.cpp - kernel.launch → PVA ABI --------------===//
//
// Lowers `kernel.launch @cudnnConvolution2D_9tap_i{8,16}` ops to
// `func.call @polygeist_pva_conv2d_3x3_i{8,16}`, the runtime-shim ABI for
// NVIDIA PVA Solutions' single-channel integer Conv2d operator
// (libpva_operator on Orin's Programmable Vision Accelerator).
//
// Why a separate pass: PVA is a distinct backend from cuBLAS/cuDNN —
// different vendor library (`libpva_operator` / `libcupva_host`), different
// host-side staging (PVA-allocated memory accessed via
// `CupvaMemGetHostPointer`, not cudaMemcpy), and different hardware
// semantics (Q-format quantized filter with REPLICATE border, not a raw
// integer multiply-accumulate). Wedging this into the cuBLAS pass would
// muddy the cuBLAS pass's symbol map; routing it through its own pass
// keeps each backend self-contained.
//
// cuDNN deliberately fails on standalone INT8/INT16 forward conv on Orin
// (CUDNN_STATUS_BAD_PARAM), and there's no host fallback either — PVA is
// the only Orin path for those dtypes today.
//
// This pass and `--lower-kernel-launch-to-cublas` handle disjoint launch
// symbol sets, so the relative order doesn't matter; both should run
// before LLVM lowering. The conv-lowering body is shared via
// `KernelLaunchLoweringUtils.h` since it's purely a memref/scalar layout
// transformation that's the same for any conv backend.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "KernelLaunchLoweringUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace {

// Map a matcher-emitted kernel symbol to its PVA runtime-shim symbol.
// Empty StringRef means "not a PVA target — leave for another pass."
static StringRef pvaShimSymbolFor(StringRef libSym) {
  if (libSym == "cudnnConvolution2D_9tap_i16")
    return "polygeist_pva_conv2d_3x3_i16";
  if (libSym == "cudnnConvolution2D_9tap_i8")
    return "polygeist_pva_conv2d_3x3_i8";
  if (libSym == "pvaBoxFilter_3x3_i8")
    return "polygeist_pva_boxfilter_3x3_i8";
  if (libSym == "pvaBoxFilter_3x3_i16")
    return "polygeist_pva_boxfilter_3x3_i16";
  if (libSym == "pvaGaussianFilter_3x3_i8")
    return "polygeist_pva_gaussian_3x3_i8";
  if (libSym == "pvaGaussianFilter_3x3_i16")
    return "polygeist_pva_gaussian_3x3_i16";
  if (libSym == "pvaBilateralFilter_3x3_i8")
    return "polygeist_pva_bilateral_3x3_i8";
  if (libSym == "pvaBilateralFilter_3x3_i16")
    return "polygeist_pva_bilateral_3x3_i16";
  if (libSym == "pvaHistogramEqualization_i8")
    return "polygeist_pva_histeq_i8";
  return StringRef();
}

// Classify the launch shape so the right lowering helper is invoked.
enum class PvaLaunchKind { Conv9tap, ImageFilter2op };
static PvaLaunchKind pvaLaunchKindFor(StringRef libSym) {
  if (libSym.starts_with("cudnnConvolution2D_9tap_"))
    return PvaLaunchKind::Conv9tap;
  // pvaBoxFilter_*, future pvaGaussianFilter_*, pvaMedianFilter_*, etc.
  return PvaLaunchKind::ImageFilter2op;
}

struct LowerKernelLaunchToPVAPass
    : public mlir::polygeist::LowerKernelLaunchToPVABase<
          LowerKernelLaunchToPVAPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<LaunchOp> launches;
    module.walk([&](LaunchOp op) { launches.push_back(op); });

    for (LaunchOp launch : launches) {
      auto sym = launch->getAttrOfType<SymbolRefAttr>("kernel");
      if (!sym) continue;
      StringRef libSym = sym.getLeafReference().getValue();
      StringRef shim = pvaShimSymbolFor(libSym);
      if (shim.empty()) continue;  // not ours; another pass will handle it

      LogicalResult r = failure();
      switch (pvaLaunchKindFor(libSym)) {
      case PvaLaunchKind::Conv9tap:
        r = lowerCudnnConv2D9tap(launch, module, shim);
        break;
      case PvaLaunchKind::ImageFilter2op:
        r = lowerImageFilter2Operand(launch, module, shim);
        break;
      }
      if (failed(r))
        return signalPassFailure();
    }

    // Drop any kernel.defn that has no remaining uses. The matcher injects
    // stub defns to satisfy the verifier; after lowering, the ones we
    // claimed have no callers. (We don't filter by which symbols we
    // claimed: scripts often inject stubs for every symbol the matcher
    // could emit, only some of which the input actually used.)
    SmallVector<DefnOp> deadDefns;
    module.walk([&](DefnOp d) {
      if (SymbolTable::symbolKnownUseEmpty(d, module))
        deadDefns.push_back(d);
    });
    for (DefnOp d : deadDefns)
      d.erase();
  }
};

} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createLowerKernelLaunchToPVAPass() {
  return std::make_unique<LowerKernelLaunchToPVAPass>();
}
} // namespace polygeist
} // namespace mlir

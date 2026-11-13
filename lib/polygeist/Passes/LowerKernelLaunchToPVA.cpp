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
  if (libSym == "pvaBoxFilter_3x3_u8")
    return "polygeist_pva_boxfilter_3x3_u8";
  if (libSym == "pvaBoxFilter_3x3_s8")
    return "polygeist_pva_boxfilter_3x3_s8";
  if (libSym == "pvaBoxFilter_3x3_u16")
    return "polygeist_pva_boxfilter_3x3_u16";
  if (libSym == "pvaBoxFilter_3x3_s16")
    return "polygeist_pva_boxfilter_3x3_s16";
  if (libSym == "pvaGaussianFilter_3x3_u8")
    return "polygeist_pva_gaussian_3x3_u8";
  if (libSym == "pvaGaussianFilter_3x3_s8")
    return "polygeist_pva_gaussian_3x3_s8";
  if (libSym == "pvaGaussianFilter_3x3_u16")
    return "polygeist_pva_gaussian_3x3_u16";
  if (libSym == "pvaGaussianFilter_3x3_s16")
    return "polygeist_pva_gaussian_3x3_s16";
  if (libSym == "pvaMorphologyDilate_3x3_u8")
    return "polygeist_pva_morphology_dilate_3x3_u8";
  if (libSym == "pvaMorphologyDilate_3x3_s8")
    return "polygeist_pva_morphology_dilate_3x3_s8";
  if (libSym == "pvaMorphologyDilate_3x3_u16")
    return "polygeist_pva_morphology_dilate_3x3_u16";
  if (libSym == "pvaMorphologyDilate_3x3_s16")
    return "polygeist_pva_morphology_dilate_3x3_s16";
  if (libSym == "pvaBilateralFilter_3x3_u8")
    return "polygeist_pva_bilateral_3x3_u8";
  if (libSym == "pvaImageHistogram_256_u8_u32")
    return "polygeist_pva_histogram_256_u8_u32";
  if (libSym == "pvaImageHistogram_256_u8_s32")
    return "polygeist_pva_histogram_256_u8_s32";
  if (libSym == "pvaImageHistogram_256_u16_u32")
    return "polygeist_pva_histogram_256_u16_u32";
  if (libSym == "pvaImageHistogram_256_u16_s32")
    return "polygeist_pva_histogram_256_u16_s32";
  if (libSym == "pvaHistogramEqualization_u8")
    return "polygeist_pva_histeq_u8";
  return StringRef();
}

// These pre-typed symbols came from early feasibility scripts.  Their ABIs
// discarded semantic information needed to prove an equivalent vendor call:
// Gaussian and bilateral sigmas, source signedness, and exact box/histogram
// equalization rounding.  Do not leave them for a later pass: reject them
// explicitly so an old artifact cannot silently regain the unsafe lowering.
static bool isRetiredLegacyPvaSymbol(StringRef libSym) {
  return libSym == "pvaBoxFilter_3x3_i8" ||
         libSym == "pvaBoxFilter_3x3_i16" ||
         libSym == "pvaGaussianFilter_3x3_i8" ||
         libSym == "pvaGaussianFilter_3x3_i16" ||
         libSym == "pvaBilateralFilter_3x3_i8" ||
         libSym == "pvaBilateralFilter_3x3_i16" ||
         libSym == "pvaHistogramEqualization_i8";
}

static bool requiresNumericalContract(StringRef libSym) {
  return libSym.starts_with("pvaBoxFilter_") ||
         libSym.starts_with("pvaGaussianFilter_") ||
         libSym.starts_with("pvaMorphologyDilate_") ||
         libSym.starts_with("pvaBilateralFilter_") ||
         libSym.starts_with("pvaImageHistogram_") ||
         libSym.starts_with("pvaHistogramEqualization_");
}

// Classify the launch shape so the right lowering helper is invoked.
enum class PvaLaunchKind { Conv9tap, ImageFilter2op, FlatTyped };
static PvaLaunchKind pvaLaunchKindFor(StringRef libSym) {
  if (libSym.starts_with("cudnnConvolution2D_9tap_"))
    return PvaLaunchKind::Conv9tap;
  if (libSym.ends_with("_u8") || libSym.ends_with("_s8") ||
      libSym.ends_with("_u16") || libSym.ends_with("_s16") ||
      libSym.contains("Histogram_256_"))
    return PvaLaunchKind::FlatTyped;
  // pvaBoxFilter_*, future pvaGaussianFilter_*, pvaMedianFilter_*, etc.
  return PvaLaunchKind::ImageFilter2op;
}

// Typed semantic matches use a deliberately simple ABI-preserving launch:
// scalar parameters remain scalar operands and flat source memrefs become raw
// data pointers.  The first two operands are always image height/width.  This
// avoids manufacturing rank/layout information in the matcher; the runtime
// constructs the required single-channel HWC NVCV tensors explicitly.
static LogicalResult lowerFlatTypedLaunch(LaunchOp launch, ModuleOp module,
                                          StringRef shim) {
  if (launch.getNumResults() != 0)
    return launch.emitError("typed PVA launch must be memref-form void");
  if (launch.getNumOperands() < 4)
    return launch.emitError("typed PVA launch requires h, w and two buffers");
  if (!launch.getOperand(0).getType().isInteger(32) ||
      !launch.getOperand(1).getType().isInteger(32))
    return launch.emitError("typed PVA launch h/w operands must be i32");

  OpBuilder builder(launch);
  SmallVector<Value> operands;
  SmallVector<Type> types;
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  for (Value operand : launch.getOperands()) {
    if (auto memref = dyn_cast<MemRefType>(operand.getType())) {
      if (memref.getRank() != 1)
        return launch.emitError("typed PVA buffer operands must be flat memrefs");
      operands.push_back(memrefBasePtr(builder, launch.getLoc(), operand));
      types.push_back(ptrType);
    } else {
      if (!(operand.getType().isInteger(32) || operand.getType().isF32()))
        return launch.emitError("typed PVA scalar operands must be i32/f32");
      operands.push_back(operand);
      types.push_back(operand.getType());
    }
  }
  func::FuncOp declaration = ensureShimDecl(module, shim, types, builder);
  auto call = builder.create<func::CallOp>(launch.getLoc(), declaration,
                                           operands);
  if (auto contract =
          launch->getAttrOfType<StringAttr>("polygeist.numerical_contract"))
    call->setAttr("polygeist.numerical_contract", contract);
  if (auto budget = launch->getAttrOfType<IntegerAttr>(
          "polygeist.max_abs_error_budget"))
    call->setAttr("polygeist.max_abs_error_budget", budget);
  launch.erase();
  return success();
}

struct LowerKernelLaunchToPVAPass
    : public mlir::polygeist::LowerKernelLaunchToPVABase<
          LowerKernelLaunchToPVAPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<LaunchOp> launches;
    module.walk([&](LaunchOp op) { launches.push_back(op); });

    // Preflight before mutating any launch so failure is atomic and reports
    // every unsafe symbol/contract in the module, rather than only the first.
    bool foundInvalidPvaLaunch = false;
    for (LaunchOp launch : launches) {
      auto sym = launch->getAttrOfType<SymbolRefAttr>("kernel");
      if (!sym)
        continue;
      StringRef libSym = sym.getLeafReference().getValue();
      if (isRetiredLegacyPvaSymbol(libSym)) {
        launch.emitError()
            << "retired unsafe PVA ABI @" << libSym
            << ": semantic parameters/signedness/rounding were not preserved; "
               "rerun structural matching to produce a typed, parameterized "
               "PVA launch";
        foundInvalidPvaLaunch = true;
        continue;
      }
      if (!requiresNumericalContract(libSym))
        continue;
      auto contract = launch->getAttrOfType<StringAttr>(
          "polygeist.numerical_contract");
      if (!contract || (contract.getValue() != "exact" &&
                        contract.getValue() != "approximate")) {
        launch.emitError()
            << "typed PVA image launch @" << libSym
            << " requires polygeist.numerical_contract = exact|approximate";
        foundInvalidPvaLaunch = true;
        continue;
      }
      if (contract.getValue() == "approximate") {
        auto budget = launch->getAttrOfType<IntegerAttr>(
            "polygeist.max_abs_error_budget");
        if (!budget || budget.getInt() < 0) {
          launch.emitError()
              << "approximate PVA image launch @" << libSym
              << " requires a nonnegative i64 "
                 "polygeist.max_abs_error_budget";
          foundInvalidPvaLaunch = true;
        }
      }
    }
    if (foundInvalidPvaLaunch)
      return signalPassFailure();

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
      case PvaLaunchKind::FlatTyped:
        r = lowerFlatTypedLaunch(launch, module, shim);
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

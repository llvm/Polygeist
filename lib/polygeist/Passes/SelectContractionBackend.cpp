//===- SelectContractionBackend.cpp - contraction backend policy ---------===//

#include "PassDetails.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace {

struct ContractionShape {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t batch = 1;
};

static std::optional<int64_t> getPositiveI64(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr || attr.getInt() <= 0)
    return std::nullopt;
  return attr.getInt();
}

static bool checkedMul(int64_t a, int64_t b, int64_t &result) {
  if (a <= 0 || b <= 0 || a > std::numeric_limits<int64_t>::max() / b)
    return false;
  result = a * b;
  return true;
}

static int64_t roundUp(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

static std::optional<ContractionShape> getExplicitShape(LaunchOp launch) {
  auto m = getPositiveI64(launch, "polygeist.contraction.m");
  auto n = getPositiveI64(launch, "polygeist.contraction.n");
  auto k = getPositiveI64(launch, "polygeist.contraction.k");
  if (!m || !n || !k)
    return std::nullopt;
  auto batch = getPositiveI64(launch, "polygeist.contraction.batch");
  return ContractionShape{*m, *n, *k, batch.value_or(1)};
}

static std::optional<ShapedType> getShapedType(Value value) {
  if (auto type = dyn_cast<ShapedType>(value.getType()))
    return type;
  return std::nullopt;
}

static bool hasStaticPositiveShape(ShapedType type) {
  if (!type.hasRank() || !type.hasStaticShape())
    return false;
  return llvm::all_of(type.getShape(), [](int64_t d) { return d > 0; });
}

// Compatibility inference for the existing semantic SGEMM ISA symbols. The
// suffix records the physical row-major transpose state, not a source name.
static std::optional<ContractionShape> inferSgemmShape(LaunchOp launch,
                                                       StringRef symbol) {
  if (!symbol.starts_with("cublasSgemm_") || launch.getNumOperands() < 3)
    return std::nullopt;
  StringRef suffix = symbol.drop_front(StringRef("cublasSgemm_").size());
  if (suffix.size() < 2 || (suffix[0] != 'n' && suffix[0] != 't') ||
      (suffix[1] != 'n' && suffix[1] != 't'))
    return std::nullopt;

  auto a = getShapedType(launch.getOperand(0));
  auto b = getShapedType(launch.getOperand(1));
  auto c = getShapedType(launch.getOperand(2));
  if (!a || !b || !c || !hasStaticPositiveShape(*a) ||
      !hasStaticPositiveShape(*b) || !hasStaticPositiveShape(*c) ||
      a->getRank() < 2 || b->getRank() < 2 || c->getRank() < 2)
    return std::nullopt;

  ArrayRef<int64_t> as = a->getShape();
  ArrayRef<int64_t> bs = b->getShape();
  ArrayRef<int64_t> cs = c->getShape();
  const int64_t m = cs[cs.size() - 2];
  const int64_t n = cs.back();
  const int64_t ak = suffix[0] == 'n' ? as.back() : as[as.size() - 2];
  const int64_t bk = suffix[1] == 'n' ? bs[bs.size() - 2] : bs.back();
  if (ak != bk)
    return std::nullopt;
  int64_t batch = 1;
  for (int64_t d : cs.drop_back(2)) {
    if (!checkedMul(batch, d, batch))
      return std::nullopt;
  }
  return ContractionShape{m, n, ak, batch};
}

static bool isF32Contraction(LaunchOp launch) {
  bool sawShapedOperand = false;
  for (Value operand : launch.getOperands()) {
    auto shaped = getShapedType(operand);
    if (!shaped)
      continue;
    sawShapedOperand = true;
    if (!shaped->getElementType().isF32())
      return false;
  }
  return sawShapedOperand;
}

static bool isSemanticContraction(LaunchOp launch, StringRef symbol) {
  if (auto kind =
          launch->getAttrOfType<StringAttr>("polygeist.contraction.kind"))
    return kind.getValue() == "gemm" || kind.getValue() == "contraction" ||
           kind.getValue() == "syrk" || kind.getValue() == "syr2k";
  return symbol.starts_with("cublasSgemm_") ||
         symbol.starts_with("cutensornetContraction") ||
         symbol.starts_with("cutensornetNetwork") ||
         symbol.starts_with("cublasDsyrk") ||
         symbol.starts_with("cublasDsyr2k") ||
         symbol.starts_with("cublasSsyrk") ||
         symbol.starts_with("cublasSsyr2k");
}

static bool isSymmetricRankK(LaunchOp launch, StringRef symbol) {
  if (auto kind =
          launch->getAttrOfType<StringAttr>("polygeist.contraction.kind"))
    if (kind.getValue() == "syrk" || kind.getValue() == "syr2k")
      return true;
  return symbol.starts_with("cublasDsyrk") ||
         symbol.starts_with("cublasDsyr2k") ||
         symbol.starts_with("cublasSsyrk") ||
         symbol.starts_with("cublasSsyr2k");
}

struct SelectContractionBackendPass
    : public mlir::polygeist::SelectContractionBackendBase<
          SelectContractionBackendPass> {
  void runOnOperation() override {
    getOperation().walk([&](LaunchOp launch) {
      auto kernel = launch->getAttrOfType<SymbolRefAttr>("kernel");
      if (!kernel)
        return;
      StringRef symbol = kernel.getLeafReference().getValue();
      if (!isSemanticContraction(launch, symbol))
        return;

      auto shape = getExplicitShape(launch);
      if (!shape)
        shape = inferSgemmShape(launch, symbol);

      const bool tf32Eligible =
          allowTf32 && isF32Contraction(launch) && targetArch == "sm_87";

      auto setDecision = [&](StringRef backend, StringRef reason) {
        MLIRContext *ctx = launch.getContext();
        launch->setAttr("polygeist.contraction.backend",
                        StringAttr::get(ctx, backend));
        launch->setAttr("polygeist.contraction.selection_reason",
                        StringAttr::get(ctx, reason));
        launch->setAttr("polygeist.contraction.target",
                        StringAttr::get(ctx, targetArch));
      };

      // On sm_87, cuBLAS's FP32 SYRK/SYR2K route showed no repeatable benefit
      // from TF32 math mode in controlled silicon measurements.  Keep these
      // semantic operations on their strict vendor-library implementations;
      // a future backend with an independently validated rank-k MMA lowering
      // can revise this policy without relying on source or symbol names.
      if (tf32Eligible && isSymmetricRankK(launch, symbol)) {
        setDecision("vendor-library", "no-verified-rankk-tf32-benefit");
        return;
      }

      if (!shape) {
        // cuBLAS accepts runtime dimensions and performs its own legal
        // algorithm selection.  Generated MMA and cuBLASDx require stronger
        // static shape/layout evidence, but lack of a compile-time shape must
        // not prevent an explicitly authorized TF32 cuBLAS route.
        setDecision(tf32Eligible ? "cublas-tf32" : "vendor-library",
                    tf32Eligible ? "dynamic-shape-cublas-dispatch"
                                 : "dynamic-or-unavailable-shape");
        return;
      }
      launch->setAttr(
          "polygeist.contraction.m",
          IntegerAttr::get(IndexType::get(launch.getContext()), shape->m));
      launch->setAttr(
          "polygeist.contraction.n",
          IntegerAttr::get(IndexType::get(launch.getContext()), shape->n));
      launch->setAttr(
          "polygeist.contraction.k",
          IntegerAttr::get(IndexType::get(launch.getContext()), shape->k));
      launch->setAttr(
          "polygeist.contraction.batch",
          IntegerAttr::get(IndexType::get(launch.getContext()), shape->batch));

      if (!tf32Eligible) {
        setDecision("vendor-library", "tf32-disabled-or-unsupported");
        return;
      }

      int64_t outputs = 0, operations = 0;
      if (!checkedMul(shape->m, shape->n, outputs) ||
          !checkedMul(outputs, shape->k, operations)) {
        setDecision("vendor-library", "shape-product-overflow");
        return;
      }
      if (operations < minimumMmaOperations || outputs < minimumMmaOutputs) {
        setDecision("cuda-fma", "insufficient-mma-work");
        return;
      }
      // cuBLASDx is a block-level API.  A work threshold alone does not prove
      // that one static problem fits a block or that a useful independent
      // batch exists.  Require an earlier structural/planning pass to record
      // that legality fact; otherwise use global cuBLAS for large GEMMs.
      if (operations >= cublasDxOperations &&
          launch->hasAttr("polygeist.contraction.cublasdx_legal")) {
        setDecision("cublasdx", "large-gemm-work");
        return;
      }
      if (operations >= cublasDxOperations) {
        setDecision("cublas-tf32", "large-global-gemm");
        return;
      }

      int64_t directMN = 0, directWork = 0, wmmaMN = 0, wmmaWork = 0;
      bool validPadding =
          checkedMul(roundUp(shape->m, 16), roundUp(shape->n, 8), directMN) &&
          checkedMul(directMN, roundUp(shape->k, 4), directWork) &&
          checkedMul(roundUp(shape->m, 16), roundUp(shape->n, 16), wmmaMN) &&
          checkedMul(wmmaMN, roundUp(shape->k, 8), wmmaWork);
      if (!validPadding) {
        setDecision("vendor-library", "padded-shape-overflow");
        return;
      }
      const double paddingRatio =
          static_cast<double>(wmmaWork) / static_cast<double>(directWork);
      if (operations >= minimumWmmaOperations &&
          paddingRatio <= maxWmmaPaddingRatio)
        setDecision("wmma", "padding-friendly-medium-gemm");
      else
        setDecision("direct-mma", "irregular-small-gemm");
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::polygeist::createSelectContractionBackendPass() {
  return std::make_unique<SelectContractionBackendPass>();
}

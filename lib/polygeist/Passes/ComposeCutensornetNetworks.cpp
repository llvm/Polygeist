//===- ComposeCutensornetNetworks.cpp ------------------------------------===//
//
// Compose already-proven binary contraction labels and multiplicative
// linalg.generic stages into one variable-arity Einstein network. The pass
// deliberately reasons from SSA dataflow, affine maps, and scalar combiner
// semantics; function names, MFEM ranks, and fixed element sizes are absent.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SetVector.h"

#include <optional>

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace {

struct NetworkLeaf {
  Value value;
  SmallVector<unsigned, 6> modes;
};

static bool isCutensornetBinary(LaunchOp launch) {
  auto symbol = launch->getAttrOfType<FlatSymbolRefAttr>("kernel");
  if (!symbol || launch.getNumOperands() != 3 ||
      launch.getNumResults() != 1)
    return false;
  StringRef name = symbol.getValue();
  return name == "cutensornetContraction2_f64" ||
         name == "cutensornetContraction2_f64_r4r5r4" ||
         name == "cutensornetContraction2_f64_r5r4r4" ||
         name == "cutensornetContraction2_f64_r5r5r4";
}

static std::optional<unsigned> dimPosition(AffineExpr expr) {
  if (auto dim = expr.dyn_cast<AffineDimExpr>())
    return dim.getPosition();
  return std::nullopt;
}

static bool getProjectedModes(AffineMap map,
                              SmallVectorImpl<unsigned> &modes) {
  modes.clear();
  for (AffineExpr expr : map.getResults()) {
    auto position = dimPosition(expr);
    if (!position)
      return false;
    modes.push_back(*position);
  }
  return true;
}

static bool isAllParallel(linalg::GenericOp generic) {
  return llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType type) {
    return type == utils::IteratorType::parallel;
  });
}

static bool isPointwiseProduct(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      generic.getNumResults() != 1 || !isAllParallel(generic) ||
      !generic.getRegion().hasOneBlock())
    return false;
  Block &body = generic.getRegion().front();
  if (body.getNumArguments() != 2)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto multiply = yield.getOperand(0).getDefiningOp<arith::MulFOp>();
  if (!multiply)
    return false;
  Value input = body.getArgument(0), output = body.getArgument(1);
  return (multiply.getLhs() == input && multiply.getRhs() == output) ||
         (multiply.getLhs() == output && multiply.getRhs() == input);
}

static bool isAdditiveContraction(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1 ||
      generic.getNumResults() != 1 || !generic.getRegion().hasOneBlock())
    return false;
  bool hasReduction = llvm::any_of(
      generic.getIteratorTypesArray(), [](utils::IteratorType type) {
        return type == utils::IteratorType::reduction;
      });
  if (!hasReduction)
    return false;
  Block &body = generic.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto add = yield.getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;
  Value a = body.getArgument(0), b = body.getArgument(1);
  Value out = body.getArgument(2);
  Value productValue = add.getLhs() == out ? add.getRhs()
                                           : add.getRhs() == out
                                                 ? add.getLhs() : Value();
  auto multiply = productValue.getDefiningOp<arith::MulFOp>();
  return multiply &&
         ((multiply.getLhs() == a && multiply.getRhs() == b) ||
          (multiply.getLhs() == b && multiply.getRhs() == a));
}

static bool isFullIdentitySlice(tensor::ExtractSliceOp slice) {
  auto sourceType = dyn_cast<RankedTensorType>(slice.getSource().getType());
  auto resultType = dyn_cast<RankedTensorType>(slice.getType());
  if (!sourceType || !resultType ||
      sourceType.getRank() != resultType.getRank())
    return false;
  for (OpFoldResult offset : slice.getMixedOffsets()) {
    auto value = getConstantIntValue(offset);
    if (!value || *value != 0)
      return false;
  }
  for (OpFoldResult stride : slice.getMixedStrides()) {
    auto value = getConstantIntValue(stride);
    if (!value || *value != 1)
      return false;
  }
  for (auto [dim, size] : llvm::enumerate(slice.getMixedSizes())) {
    if (sourceType.isDynamicDim(dim))
      return false;
    auto value = getConstantIntValue(size);
    if (!value || *value != sourceType.getDimSize(dim))
      return false;
  }
  return true;
}

// The compatibility cuTensorNet ABI can safely mutate an output tensor whose
// storage comes directly from a public memref argument. A tensor computed by
// earlier residual Linalg or opaque host-ABI launches is not a stable network
// accumulator: lowering it back to a pointer can alias a stale pre-stage
// buffer after one-shot bufferization. Keep such graphs as their correct
// pairwise calls until the entire connected region uses a bufferizable,
// device-resident library operation.
static bool hasDirectAbiDestination(Value value) {
  for (int hops = 0; hops < 16; ++hops) {
    if (auto cast = value.getDefiningOp<tensor::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
      value = slice.getSource();
      continue;
    }
    if (auto submap = value.getDefiningOp<polygeist::SubmapOp>()) {
      value = submap.getBase();
      continue;
    }
    if (auto toTensor =
            value.getDefiningOp<bufferization::ToTensorOp>())
      return isa<BlockArgument>(toTensor.getMemref());
    return false;
  }
  return false;
}

static bool hasInjectiveDestinationView(Value value) {
  for (int hops = 0; hops < 16; ++hops) {
    if (auto cast = value.getDefiningOp<tensor::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
      value = slice.getSource();
      continue;
    }
    if (auto submap = value.getDefiningOp<polygeist::SubmapOp>()) {
      // If a logical view dimension is absent from every physical address
      // expression, multiple logical output elements alias one destination.
      // The CPU reference can reduce through such a zero-stride dimension,
      // but the current cuTensorNet network ABI cannot preserve its exact
      // write-back semantics.
      AffineMap map = submap.getMap();
      for (unsigned dim = 0; dim < submap.getSizes().size(); ++dim) {
        bool present = llvm::any_of(
            map.getResults(),
            [dim](AffineExpr expr) { return expr.isFunctionOfDim(dim); });
        if (!present)
          return false;
      }
      value = submap.getBase();
      continue;
    }
    return true;
  }
  return false;
}

struct NetworkTrace {
  MLIRContext *context;
  unsigned nextMode = 0;
  unsigned contractionCount = 0;
  SmallVector<NetworkLeaf, 8> leaves;
  llvm::SetVector<Operation *> consumed;

  unsigned freshMode() { return nextMode++; }

  FailureOr<SmallVector<unsigned, 6>> translateMap(
      AffineMap map, DenseMap<unsigned, unsigned> &localToGlobal) {
    SmallVector<unsigned, 6> result;
    for (AffineExpr expr : map.getResults()) {
      auto local = dimPosition(expr);
      if (!local)
        return failure();
      auto existing = localToGlobal.find(*local);
      if (existing != localToGlobal.end()) {
        result.push_back(existing->second);
      } else {
        unsigned global = freshMode();
        localToGlobal[*local] = global;
        result.push_back(global);
      }
    }
    return result;
  }

  LogicalResult trace(Value value, ArrayRef<unsigned> requestedModes) {
    if (auto cast = value.getDefiningOp<tensor::CastOp>()) {
      // Old iterator-count-independent matcher artifacts can use an unranked
      // tensor.cast around a generic contraction ABI.  Such a value does not
      // carry enough mode information to join a ranked tensor network, but it
      // must be rejected cleanly rather than querying ShapedType::getRank and
      // crashing the whole module pass.
      auto sourceType = dyn_cast<RankedTensorType>(cast.getSource().getType());
      auto resultType = dyn_cast<RankedTensorType>(cast.getType());
      if (!sourceType || !resultType ||
          sourceType.getRank() != resultType.getRank())
        return failure();
      consumed.insert(cast);
      return trace(cast.getSource(), requestedModes);
    }
    if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
      if (!isFullIdentitySlice(slice))
        return failure();
      consumed.insert(slice);
      return trace(slice.getSource(), requestedModes);
    }
    if (auto launch = value.getDefiningOp<LaunchOp>()) {
      if (!isCutensornetBinary(launch))
        return addLeaf(value, requestedModes);
      auto maps = launch->getAttrOfType<ArrayAttr>("contraction_maps");
      if (!maps || maps.size() != 3)
        return failure();
      auto outputMap = dyn_cast<AffineMapAttr>(maps[2]);
      if (!outputMap ||
          outputMap.getValue().getNumResults() != requestedModes.size())
        return failure();
      DenseMap<unsigned, unsigned> localToGlobal;
      for (auto [expr, global] :
           llvm::zip(outputMap.getValue().getResults(), requestedModes)) {
        auto local = dimPosition(expr);
        if (!local)
          return failure();
        auto insertion = localToGlobal.try_emplace(*local, global);
        if (!insertion.second && insertion.first->second != global)
          return failure();
      }
      auto lhsMap = dyn_cast<AffineMapAttr>(maps[0]);
      auto rhsMap = dyn_cast<AffineMapAttr>(maps[1]);
      if (!lhsMap || !rhsMap)
        return failure();
      auto lhsModes = translateMap(lhsMap.getValue(), localToGlobal);
      auto rhsModes = translateMap(rhsMap.getValue(), localToGlobal);
      if (failed(lhsModes) || failed(rhsModes))
        return failure();
      consumed.insert(launch);
      contractionCount++;
      if (failed(trace(launch.getOperand(0), *lhsModes)) ||
          failed(trace(launch.getOperand(1), *rhsModes)))
        return failure();
      return success();
    }
    if (auto generic = value.getDefiningOp<linalg::GenericOp>()) {
      if (!isPointwiseProduct(generic))
        return addLeaf(value, requestedModes);
      auto maps = generic.getIndexingMapsArray();
      if (maps.size() != 2 ||
          maps[1].getNumResults() != requestedModes.size())
        return failure();
      DenseMap<unsigned, unsigned> localToGlobal;
      for (auto [expr, global] :
           llvm::zip(maps[1].getResults(), requestedModes)) {
        auto local = dimPosition(expr);
        if (!local)
          return failure();
        localToGlobal[*local] = global;
      }
      auto inputModes = translateMap(maps[0], localToGlobal);
      auto outputModes = translateMap(maps[1], localToGlobal);
      if (failed(inputModes) || failed(outputModes))
        return failure();
      consumed.insert(generic);
      if (failed(trace(generic.getDpsInputOperand(0)->get(), *inputModes)) ||
          failed(trace(generic.getDpsInitOperand(0)->get(), *outputModes)))
        return failure();
      return success();
    }
    return addLeaf(value, requestedModes);
  }

  LogicalResult addLeaf(Value value, ArrayRef<unsigned> modes) {
    auto shaped = dyn_cast<RankedTensorType>(value.getType());
    if (!shaped || shaped.getRank() != (int64_t)modes.size() ||
        !(shaped.getElementType().isF32() ||
          shaped.getElementType().isF64()))
      return failure();
    leaves.push_back({value, SmallVector<unsigned, 6>(modes)});
    return success();
  }
};

static bool intermediatesDoNotEscape(const NetworkTrace &trace,
                                     Operation *sink) {
  for (Operation *operation : trace.consumed)
    for (Value result : operation->getResults())
      for (Operation *user : result.getUsers())
        if (user != sink && !trace.consumed.contains(user))
          return false;
  return true;
}

static DefnOp createNetworkDefinition(ModuleOp module, StringRef name,
                                      TypeRange inputs, Type resultType,
                                      unsigned outputOperand) {
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto definition = builder.create<DefnOp>(
      module.getLoc(), name, builder.getFunctionType(inputs, resultType),
      builder.getStringAttr("private"), ArrayAttr(), ArrayAttr());
  SmallVector<Location> locations(inputs.size(), module.getLoc());
  Block *block = builder.createBlock(&definition.getBody(), {}, inputs,
                                     locations);
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(block);
  bodyBuilder.create<YieldOp>(module.getLoc(),
                              block->getArgument(outputOperand));
  return definition;
}

static LogicalResult composeSink(linalg::GenericOp sink,
                                 unsigned &definitionCounter) {
  if (!isAdditiveContraction(sink))
    return failure();
  auto maps = sink.getIndexingMapsArray();
  if (maps.size() != 3)
    return failure();

  NetworkTrace trace{sink.getContext()};
  trace.nextMode = sink.getNumLoops();
  SmallVector<unsigned, 6> lhsModes, rhsModes, outputModes;
  if (!getProjectedModes(maps[0], lhsModes) ||
      !getProjectedModes(maps[1], rhsModes) ||
      !getProjectedModes(maps[2], outputModes))
    return failure();
  if (failed(trace.trace(sink.getDpsInputOperand(0)->get(), lhsModes)) ||
      failed(trace.trace(sink.getDpsInputOperand(1)->get(), rhsModes)) ||
      trace.contractionCount < 2 || trace.leaves.size() < 3 ||
      trace.nextMode > 64 ||
      !intermediatesDoNotEscape(trace, sink))
    return failure();

  Value output = sink.getDpsInitOperand(0)->get();
  if (!hasDirectAbiDestination(output) ||
      !hasInjectiveDestinationView(output))
    return failure();
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  if (!outputType || outputType.getRank() != (int64_t)outputModes.size())
    return failure();
  for (const NetworkLeaf &leaf : trace.leaves) {
    auto type = cast<RankedTensorType>(leaf.value.getType());
    if (type.getElementType() != outputType.getElementType())
      return failure();
  }

  SmallVector<Value, 8> operands;
  SmallVector<Type, 8> operandTypes;
  SmallVector<Attribute, 8> networkMaps;
  auto makeMap = [&](ArrayRef<unsigned> modeList) {
    SmallVector<AffineExpr, 6> expressions;
    for (unsigned mode : modeList)
      expressions.push_back(getAffineDimExpr(mode, sink.getContext()));
    return AffineMapAttr::get(AffineMap::get(
        trace.nextMode, 0, expressions, sink.getContext()));
  };
  for (const NetworkLeaf &leaf : trace.leaves) {
    operands.push_back(leaf.value);
    operandTypes.push_back(leaf.value.getType());
    networkMaps.push_back(makeMap(leaf.modes));
  }
  unsigned outputOperand = operands.size();
  operands.push_back(output);
  operandTypes.push_back(output.getType());
  networkMaps.push_back(makeMap(outputModes));

  ModuleOp module = sink->getParentOfType<ModuleOp>();
  std::string prefix = outputType.getElementType().isF64()
                           ? "cutensornetNetwork_f64_n"
                           : "cutensornetNetwork_f32_n";
  std::string symbol =
      prefix + std::to_string(trace.leaves.size()) + "_" +
      std::to_string(definitionCounter++);
  createNetworkDefinition(module, symbol, operandTypes, sink.getResult(0).getType(),
                          outputOperand);

  OpBuilder builder(sink);
  auto launch = builder.create<LaunchOp>(
      sink.getLoc(), sink.getResultTypes(), symbol, operands);
  launch->setAttr("network_maps", builder.getArrayAttr(networkMaps));
  launch->setAttr("network_accumulate", builder.getUnitAttr());
  launch->setAttr("polygeist.tensor_network_inputs",
                  builder.getI64IntegerAttr(trace.leaves.size()));
  sink.getResult(0).replaceAllUsesWith(launch.getResult(0));
  sink.erase();

  // These operations have been semantically subsumed. Erase only after the
  // no-escape proof above; transparent tensor views are included in the same
  // set and disappear in reverse dataflow order.
  SmallVector<Operation *> pending(trace.consumed.begin(),
                                   trace.consumed.end());
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *&operation : pending) {
      if (!operation ||
          !llvm::all_of(operation->getResults(),
                        [](Value result) { return result.use_empty(); }))
        continue;
      operation->erase();
      operation = nullptr;
      changed = true;
    }
  }
  return success();
}

struct ComposeCutensornetNetworksPass
    : public ComposeCutensornetNetworksBase<ComposeCutensornetNetworksPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<linalg::GenericOp> candidates;
    module.walk([&](linalg::GenericOp generic) {
      if (!generic->getParentOfType<DefnOp>() &&
          isAdditiveContraction(generic))
        candidates.push_back(generic);
    });
    unsigned counter = 0;
    for (linalg::GenericOp candidate : llvm::reverse(candidates))
      (void)composeSink(candidate, counter);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::polygeist::createComposeCutensornetNetworksPass() {
  return std::make_unique<ComposeCutensornetNetworksPass>();
}

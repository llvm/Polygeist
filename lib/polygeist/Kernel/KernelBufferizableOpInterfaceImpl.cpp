//===- KernelBufferizableOpInterfaceImpl.cpp -----------------------------===//

#include "polygeist/Kernel/KernelBufferizableOpInterfaceImpl.h"
#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::polygeist::kernel;

namespace {

static FailureOr<SmallVector<int64_t>> getResultDestinations(LaunchOp launch) {
  auto attr = launch->getAttrOfType<DenseI64ArrayAttr>(
      "polygeist.result_destinations");
  if (attr) {
    if (attr.size() != launch.getNumResults())
      return failure();
    SmallVector<int64_t> result;
    for (auto [index, value] : llvm::enumerate(attr.asArrayRef())) {
      if (isa<TensorType>(launch.getResult(index).getType())) {
        if (value < 0 || value >= (int64_t)launch.getNumOperands() ||
            !isa<TensorType>(launch.getOperand(value).getType()))
          return failure();
      } else if (value != -1) {
        return failure();
      }
      result.push_back(value);
    }
    return result;
  }

  auto kernelAttr = launch->getAttrOfType<FlatSymbolRefAttr>("kernel");
  if (!kernelAttr)
    return failure();
  auto defn = SymbolTable::lookupNearestSymbolFrom<DefnOp>(launch, kernelAttr);
  if (!defn || !defn.getBody().hasOneBlock())
    return failure();
  auto yield = dyn_cast<YieldOp>(defn.getBody().front().getTerminator());
  if (!yield || yield.getNumOperands() != launch.getNumResults())
    return failure();

  SmallVector<int64_t> result;
  for (Value yielded : yield.getOperands()) {
    if (!isa<TensorType>(yielded.getType())) {
      result.push_back(-1);
      continue;
    }
    auto arg = dyn_cast<BlockArgument>(yielded);
    if (!arg || arg.getOwner() != &defn.getBody().front() ||
        arg.getArgNumber() >= launch.getNumOperands())
      return failure();
    result.push_back(arg.getArgNumber());
  }
  return result;
}

struct LaunchOpInterface
    : public BufferizableOpInterface::ExternalModel<LaunchOpInterface,
                                                    LaunchOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &) const {
    // Conservative until matcher-emitted destination-read metadata is added.
    // This can introduce a copy, but cannot lose a destination value.
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &operand,
                               const AnalysisState &) const {
    auto destinations = getResultDestinations(cast<LaunchOp>(op));
    if (failed(destinations))
      return false;
    return llvm::is_contained(*destinations, operand.getOperandNumber());
  }

  bool isNotConflicting(Operation *op, OpOperand *read, OpOperand *write,
                        const AnalysisState &) const {
    // Typed ABI preparation grants this capability only after validating an
    // elementwise reference and an exact-alias-safe backend. The two accesses
    // must belong to this launch and use the same tensor value. Reads by other
    // operations, shifted views, and unannotated kernels remain conservative.
    return op->hasAttr("polygeist.elementwise_exact_alias") &&
           read->getOwner() == op && write->getOwner() == op &&
           read->get() == write->get();
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &operand,
                                      const AnalysisState &) const {
    auto launch = cast<LaunchOp>(op);
    auto destinations = getResultDestinations(launch);
    if (failed(destinations))
      return {};
    AliasingValueList aliases;
    for (auto [resultNumber, operandNumber] : llvm::enumerate(*destinations))
      if (operandNumber == operand.getOperandNumber())
        aliases.addAlias({launch.getResult(resultNumber),
                          BufferRelation::Equivalent});
    return aliases;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto launch = cast<LaunchOp>(op);
    auto destinations = getResultDestinations(launch);
    if (failed(destinations))
      return launch.emitError(
          "cannot determine destination operand for every tensor result");

    SmallVector<Value> operands;
    operands.reserve(launch.getNumOperands());
    for (Value operand : launch.getOperands()) {
      if (!isa<TensorType>(operand.getType())) {
        operands.push_back(operand);
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, operand, options);
      if (failed(buffer))
        return failure();
      operands.push_back(*buffer);
    }

    SmallVector<Type> scalarTypes;
    for (Type type : launch.getResultTypes())
      if (!isa<TensorType>(type))
        scalarTypes.push_back(type);

    OperationState state(launch.getLoc(), LaunchOp::getOperationName());
    state.addOperands(operands);
    state.addTypes(scalarTypes);
    state.addAttributes(launch->getAttrs());
    state.addAttribute("polygeist.bufferized", rewriter.getUnitAttr());
    state.attributes.set("polygeist.result_destinations",
                         rewriter.getDenseI64ArrayAttr(*destinations));
    Operation *bufferized = rewriter.create(state);
    SmallVector<Value> resultBuffers;
    unsigned scalarResult = 0;
    for (int64_t operandNumber : *destinations)
      resultBuffers.push_back(operandNumber < 0
          ? bufferized->getResult(scalarResult++) : operands[operandNumber]);
    replaceOpWithBufferizedValues(rewriter, op, resultBuffers);
    return success();
  }
};

} // namespace

void mlir::polygeist::kernel::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, KernelDialect *) {
    LaunchOp::attachInterface<LaunchOpInterface>(*ctx);
  });
}

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

static FailureOr<SmallVector<unsigned>> getResultDestinations(LaunchOp launch) {
  auto attr = launch->getAttrOfType<DenseI64ArrayAttr>(
      "polygeist.result_destinations");
  if (attr) {
    if (attr.size() != launch.getNumResults())
      return failure();
    SmallVector<unsigned> result;
    for (int64_t value : attr.asArrayRef()) {
      if (value < 0 || value >= (int64_t)launch.getNumOperands())
        return failure();
      result.push_back((unsigned)value);
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

  SmallVector<unsigned> result;
  for (Value yielded : yield.getOperands()) {
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

    SmallVector<Value> resultBuffers;
    resultBuffers.reserve(destinations->size());
    for (unsigned operandNumber : *destinations)
      resultBuffers.push_back(operands[operandNumber]);

    OperationState state(launch.getLoc(), LaunchOp::getOperationName());
    state.addOperands(operands);
    state.addAttributes(launch->getAttrs());
    state.addAttribute("polygeist.bufferized", rewriter.getUnitAttr());
    state.addAttribute("polygeist.result_destinations",
                       rewriter.getDenseI64ArrayAttr(llvm::to_vector(
                           llvm::map_range(*destinations, [](unsigned value) {
                             return (int64_t)value;
                           }))));
    rewriter.create(state);
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

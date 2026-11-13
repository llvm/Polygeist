//===- KernelLibraryConstantPropagation.cpp -----------------------------===//

#include "PassDetails.h"

#include "KernelLibraryConstantPropagation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace {

struct UniformConstantFact {
  TypedAttr value;
  SmallVector<LaunchOp> producers;
};

static std::optional<TypedAttr> getScalarConstant(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;
  if (auto typed = dyn_cast<TypedAttr>(attr)) {
    if (!isa<ShapedType>(typed.getType()))
      return typed;
    if (auto elements = dyn_cast<ElementsAttr>(typed); elements && elements.isSplat()) {
      auto splat = dyn_cast<TypedAttr>(elements.getSplatValue<Attribute>());
      if (splat)
        return splat;
    }
  }
  return std::nullopt;
}

static std::optional<TypedAttr> inferDefnResultConstant(DefnOp defn,
                                                        unsigned resultNumber) {
  if (!defn.getRegion().hasOneBlock())
    return std::nullopt;
  auto yield = dyn_cast<YieldOp>(defn.getRegion().front().getTerminator());
  if (!yield || resultNumber >= yield.getNumOperands())
    return std::nullopt;

  Value yielded = yield.getOperand(resultNumber);
  while (auto cast = yielded.getDefiningOp<tensor::CastOp>())
    yielded = cast.getSource();
  if (auto fill = yielded.getDefiningOp<linalg::FillOp>())
    return getScalarConstant(fill.getInputs().front());
  auto generic = yielded.getDefiningOp<linalg::GenericOp>();
  if (!generic || generic.getNumResults() != 1 ||
      generic.getNumDpsInits() != 1 || !generic.getRegion().hasOneBlock() ||
      !llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType iterator) {
                      return iterator == utils::IteratorType::parallel;
                    }))
    return std::nullopt;

  auto resultType = dyn_cast<ShapedType>(generic.getResult(0).getType());
  if (!resultType)
    return std::nullopt;
  ArrayRef<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != generic->getNumOperands())
    return std::nullopt;
  AffineMap outputMap = maps.back();
  if (!outputMap.isIdentity() ||
      outputMap.getNumResults() != static_cast<unsigned>(resultType.getRank()))
    return std::nullopt;

  auto bodyYield = dyn_cast<linalg::YieldOp>(
      generic.getRegion().front().getTerminator());
  if (!bodyYield || bodyYield.getNumOperands() != 1)
    return std::nullopt;
  return getScalarConstant(bodyYield.getOperand(0));
}

static std::optional<UniformConstantFact>
inferUniformConstant(Value value, ModuleOp module,
                     llvm::SmallPtrSetImpl<Operation *> &visiting) {
  if (auto scalar = getScalarConstant(value))
    return UniformConstantFact{*scalar, {}};

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto forOp = dyn_cast_or_null<scf::ForOp>(
        argument.getOwner()->getParentOp());
    if (!forOp || argument.getArgNumber() == 0)
      return std::nullopt;
    unsigned iterNumber = argument.getArgNumber() - 1;
    if (iterNumber >= forOp.getInitArgs().size())
      return std::nullopt;
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (yield.getOperand(iterNumber) != argument)
      return std::nullopt;
    return inferUniformConstant(forOp.getInitArgs()[iterNumber], module,
                                visiting);
  }

  Operation *def = value.getDefiningOp();
  if (!def || !visiting.insert(def).second)
    return std::nullopt;
  auto finish = [&](std::optional<UniformConstantFact> fact) {
    visiting.erase(def);
    return fact;
  };

  if (auto launch = dyn_cast<LaunchOp>(def)) {
    auto symbol = launch->getAttrOfType<SymbolRefAttr>("kernel");
    if (!symbol)
      return finish(std::nullopt);
    auto defn = module.lookupSymbol<DefnOp>(symbol.getRootReference());
    if (!defn)
      return finish(std::nullopt);
    auto resultNumber = cast<OpResult>(value).getResultNumber();
    auto constant = inferDefnResultConstant(defn, resultNumber);
    if (!constant)
      return finish(std::nullopt);
    return finish(UniformConstantFact{*constant, {launch}});
  }
  if (auto cast = dyn_cast<tensor::CastOp>(def))
    return finish(inferUniformConstant(cast.getSource(), module, visiting));
  if (auto slice = dyn_cast<tensor::ExtractSliceOp>(def))
    return finish(inferUniformConstant(slice.getSource(), module, visiting));
  if (auto submap = dyn_cast<polygeist::SubmapOp>(def))
    return finish(inferUniformConstant(submap.getBase(), module, visiting));
  if (auto inverse = dyn_cast<polygeist::SubmapInverseOp>(def))
    return finish(
        inferUniformConstant(inverse.getOperand(1), module, visiting));
  if (auto insert = dyn_cast<tensor::InsertSliceOp>(def)) {
    auto source = inferUniformConstant(insert.getSource(), module, visiting);
    auto dest = source
                    ? inferUniformConstant(insert.getDest(), module, visiting)
                    : std::nullopt;
    if (!source || !dest || source->value != dest->value)
      return finish(std::nullopt);
    source->producers.append(dest->producers);
    return finish(std::move(source));
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    unsigned resultNumber = cast<OpResult>(value).getResultNumber();
    auto thenYield = cast<scf::YieldOp>(ifOp.thenYield());
    auto elseYield = cast<scf::YieldOp>(ifOp.elseYield());
    auto thenFact = inferUniformConstant(thenYield.getOperand(resultNumber),
                                         module, visiting);
    auto elseFact = thenFact
                        ? inferUniformConstant(elseYield.getOperand(resultNumber),
                                               module, visiting)
                        : std::nullopt;
    if (!thenFact || !elseFact || thenFact->value != elseFact->value)
      return finish(std::nullopt);
    thenFact->producers.append(elseFact->producers);
    return finish(std::move(thenFact));
  }
  if (auto forOp = dyn_cast<scf::ForOp>(def)) {
    unsigned resultNumber = cast<OpResult>(value).getResultNumber();
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value iterArgument = forOp.getRegionIterArgs()[resultNumber];
    if (yield.getOperand(resultNumber) == iterArgument)
      return finish(inferUniformConstant(forOp.getInitArgs()[resultNumber],
                                         module, visiting));
    auto initial = inferUniformConstant(forOp.getInitArgs()[resultNumber],
                                        module, visiting);
    auto yielded = initial
                       ? inferUniformConstant(yield.getOperand(resultNumber),
                                              module, visiting)
                       : std::nullopt;
    if (!initial || !yielded || initial->value != yielded->value)
      return finish(std::nullopt);
    initial->producers.append(yielded->producers);
    return finish(std::move(initial));
  }

  return finish(std::nullopt);
}

struct PropagateKernelLibraryConstantsPass
    : public mlir::polygeist::PropagateKernelLibraryConstantsBase<
          PropagateKernelLibraryConstantsPass> {
  void runOnOperation() override {
    propagateKernelLibraryConstants(getOperation());
  }
};

} // namespace

void mlir::polygeist::propagateKernelLibraryConstants(ModuleOp module) {
  module.walk([&](LaunchOp launch) {
    NamedAttrList constants;
    for (auto indexedOperand : llvm::enumerate(launch.getOperands())) {
      llvm::SmallPtrSet<Operation *, 8> visiting;
      auto fact = inferUniformConstant(indexedOperand.value(), module, visiting);
      if (!fact)
        continue;
      constants.set(("operand_" + Twine(indexedOperand.index())).str(),
                    fact->value);
    }
    if (constants.empty())
      launch->removeAttr(kUniformOperandConstantsAttr);
    else
      launch->setAttr(kUniformOperandConstantsAttr,
                      constants.getDictionary(module.getContext()));
  });
}

std::optional<TypedAttr> mlir::polygeist::getUniformOperandConstant(
    LaunchOp launch, unsigned operandNumber) {
  auto constants =
      launch->getAttrOfType<DictionaryAttr>(kUniformOperandConstantsAttr);
  if (!constants)
    return std::nullopt;
  auto constant =
      constants.getAs<TypedAttr>(("operand_" + Twine(operandNumber)).str());
  return constant ? std::optional<TypedAttr>(constant) : std::nullopt;
}

std::unique_ptr<Pass>
mlir::polygeist::createPropagateKernelLibraryConstantsPass() {
  return std::make_unique<PropagateKernelLibraryConstantsPass>();
}

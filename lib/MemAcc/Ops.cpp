#include "MemAcc/Ops.h"
#include "mlir/IR/TypeUtilities.h"
using namespace mlir;
// using namespace MemAcc;
#include "MemAcc/Dialect.h"

#define GET_OP_CLASSES
#include "MemAcc/MemAccOps.cpp.inc"

#define DEBUG_TYPE "memacc"

template <typename... Types> using type_list = std::tuple<Types...> *;

/// Returns a non-null type only if the provided type is one of the allowed
/// types or one of the allowed shaped types of the allowed types. Returns the
/// element type if a valid shaped type is provided.
template <typename... ShapedTypes, typename... ElementTypes>
static Type getUnderlyingType(Type type, type_list<ShapedTypes...>,
                              type_list<ElementTypes...>) {
  if (llvm::isa<ShapedType>(type) && !llvm::isa<ShapedTypes...>(type))
    return {};

  auto underlyingType = getElementTypeOrSelf(type);
  if (!llvm::isa<ElementTypes...>(underlyingType))
    return {};

  return underlyingType;
}

/// Get allowed underlying types for vectors, tensors, and memrefs.
template <typename... ElementTypes>
static Type getTypeIfLikeOrMemRef(Type type) {
  return getUnderlyingType(type,
                           type_list<VectorType, TensorType, MemRefType>(),
                           type_list<ElementTypes...>());
}

static bool areValidCastInputsAndOutputs(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  return succeeded(verifyCompatibleShapes(inputs.front(), outputs.front()));
}

static bool areIndexCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(inputs.front());
  auto dstType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(outputs.front());
  if (!srcType || !dstType)
    return false;

  return (srcType.isIndex() && dstType.isSignlessInteger()) ||
         (srcType.isSignlessInteger() && dstType.isIndex());
}

bool MemAcc::IndexCastOp::areCastCompatible(TypeRange inputs,
                                            TypeRange outputs) {
  return areIndexCastCompatible(inputs, outputs);
}

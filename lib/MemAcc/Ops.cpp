#include "MemAcc/Ops.h"

#define DEBUG_TYPE  "memacc"

using namespace mlir;
using namespace MemAcc;

MLIR_DEFINE_EXPLICIT_TYPE_ID(MemAcc::GenericAccessOp)
void mlir::MemAcc::GenericAccessOp::print(OpAsmPrinter &p) {
    // Example: Printing the operation name with no operands or attributes
    // and assuming the region is printed automatically.
    p.printGenericOp(getOperation());
}

ParseResult mlir::MemAcc::GenericAccessOp::parse(OpAsmParser &parser, OperationState &result) {
    // Example: Assuming the operation takes no operands and no attributes
    // and resides within a single region that is parsed automatically.
    return parser.parseRegion(*result.addRegion());
}

LogicalResult mlir::MemAcc::GenericAccessOp::verifyInvariantsImpl() {
    // Example: No custom verification logic, return success
    return success();
}
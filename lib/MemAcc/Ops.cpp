#include "MemAcc/Ops.h"

// using namespace mlir;
// using namespace MemAcc;
#include "MemAcc/Dialect.h"

#define GET_OP_CLASSES
#include "MemAcc/MemAccOps.cpp.inc"

#define DEBUG_TYPE  "memacc"

// void mlir::MemAcc::GenericStoreOp::print(OpAsmPrinter &p) {
//     // Example: Printing the operation name with no operands or attributes
//     // and assuming the region is printed automatically.
//     p.printGenericOp(getOperation());
// }

// ParseResult mlir::MemAcc::GenericStoreOp::parse(OpAsmParser &parser, OperationState &result) {
//     // Example: Assuming the operation takes no operands and no attributes
//     // and resides within a single region that is parsed automatically.
//     return parser.parseRegion(*result.addRegion());
// }

// LogicalResult mlir::MemAcc::GenericStoreOp::verifyInvariantsImpl() {
//     // Example: No custom verification logic, return success
//     return success();
// }

// void mlir::MemAcc::GenericStoreOp::build(OpBuilder &builder, OperationState &state /*, add parameters as needed */) {
//     // Example: Setup the operation with a region if your operation requires one
//     Region *bodyRegion = state.addRegion();
    
//     // If your operation does not require additional attributes or operands,
//     // you may not need to do anything else here. However, if it does,
//     // you should add them to the state before it's used to create the operation.
//     // For example, to add an operand:
//     // state.addOperands(inputOperand);

//     // If your operation has attributes, add them as well:
//     // state.addAttribute("myAttr", builder.getSomeAttributeType(value));

//     // To initialize the body region with a block, uncomment the following lines:
//     // auto *block = new Block();
//     // bodyRegion->push_back(block);
//     // You may also want to add operations to the block here or setup its arguments.
// }

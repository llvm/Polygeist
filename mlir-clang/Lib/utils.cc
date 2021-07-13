#include "utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

using namespace mlir;
using namespace llvm;

Operation *mlirclang::buildLinalgOp(const AbstractOperation *op,
                                    ArrayRef<Value> operands,
                                    ArrayRef<Type> opResultTypes,
                                    OpBuilder &b) {
  StringRef name = op->name;
  if (name.compare("linalg.copy") == 0) {
    return b.create<linalg::CopyOp>(b.getUnknownLoc(), operands[1],
                                    operands[0]);
  } else {
    llvm::report_fatal_error(llvm::Twine("builder not supported for: ") + name);
    return nullptr;
  }
}

Operation *mlirclang::replaceFuncByOperation(FuncOp f, StringRef opName,
                                             ArrayRef<Value> operands,
                                             OpBuilder &b) {
  MLIRContext *ctx = f->getContext();
  assert(ctx->isOperationRegistered(opName) &&
         "Provided lower_to opName should be registered.");

  const AbstractOperation *op = AbstractOperation::lookup(opName, ctx);

  if (opName.startswith("linalg"))
    return buildLinalgOp(op, operands, f.getCallableResults(), b);

  // NOTE: The attributes of the provided FuncOp is ignored.
  OperationState opState(b.getUnknownLoc(), op->name, ValueRange(operands),
                         f.getCallableResults(), {});
  return b.createOperation(opState);
}

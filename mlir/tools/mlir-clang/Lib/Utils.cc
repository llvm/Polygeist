#include "Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace llvm;

Operation *mlirclang::replaceFuncByOperation(FuncOp f, StringRef opName,
                                             ArrayRef<Value> operands,
                                             OpBuilder &b) {
  MLIRContext *ctx = f->getContext();
  assert(ctx->isOperationRegistered(opName) &&
         "Provided lower_to opName should be registered.");

  const AbstractOperation *op = AbstractOperation::lookup(opName, ctx);

  // NOTE: The attributes of the provided FuncOp is ignored.
  OperationState opState(b.getUnknownLoc(), op->name, ValueRange(operands),
                         f.getCallableResults(), {});
  return b.createOperation(opState);
}

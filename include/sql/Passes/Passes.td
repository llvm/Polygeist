#ifndef SQL_PASSES
#define SQL_PASSES

include "mlir/Pass/PassBase.td"


def ParallelLower : Pass<"sql-lower", "mlir::ModuleOp"> {
  let summary = "Lower sql op to mlir";
  let dependentDialects =
      ["arith::AirthDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
  let constructor = "mlir::sql::createSQLLowerPass()";
}

#endif // SQL_PASSES

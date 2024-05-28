#ifndef SQL_PASSES
#define SQL_PASSES

include "mlir/Pass/PassBase.td"


def SQLLower : Pass<"sql-lower", "mlir::ModuleOp"> {
  let summary = "Lower sql op to mlir";
  let dependentDialects =
      ["arith::ArithDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
  let constructor = "mlir::sql::createSQLLowerPass()";
}


def SQLRaising : Pass<"sql-raising", "mlir::ModuleOp"> {
  let summary = "Raise sql op to mlir";
  let dependentDialects =
      ["arith::ArithDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
  let constructor = "mlir::sql::createSQLRaisingPass()";
}

#endif // SQL_PASSES

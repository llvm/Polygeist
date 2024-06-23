//===- ConvertFromOpenScop.h ------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/IslScop.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Support/Utils.h"
#include "polymer/Target/ISL.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "isl/schedule.h"

using namespace polymer;
using namespace mlir;
using namespace mlir::func;

typedef llvm::StringMap<mlir::Operation *> StmtOpMap;
typedef llvm::StringMap<mlir::Value> NameValueMap;
typedef llvm::StringMap<std::string> IterScatNameMap;
typedef llvm::StringMap<mlir::Value> SymbolTable;

namespace polymer {
mlir::func::FuncOp createFuncOpFromIsl(std::unique_ptr<IslScop> scop,
                                       mlir::func::FuncOp f,
                                       isl_schedule *newSchedule) {
  OpBuilder b(f);
  IRMapping mapping;
  mlir::func::FuncOp g = cast<func::FuncOp>(b.clone(*f, mapping));
  if (scop->applySchedule(newSchedule, g, mapping).failed()) {
    g->erase();
    g = nullptr;
  }
  isl_schedule_free(newSchedule);
  return g;
}
} // namespace polymer

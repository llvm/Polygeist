//===- ExtractScopStmt.h - Extract scop stmt to func ------------------C++-===//
//
// This file declares the transformation that extracts scop statements into MLIR
// functions.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_TRANSFORMS_EXTRACTSCOPSTMT_H
#define POLYMER_TRANSFORMS_EXTRACTSCOPSTMT_H

#include "mlir/Pass/Pass.h"

/// TODO: place this macro at the right position.
#define SCOP_STMT_ATTR_NAME "scop.stmt"

namespace polymer {
void registerExtractScopStmtPass();
std::unique_ptr<mlir::Pass> createExtractScopStmtPass();
} // namespace polymer

#endif

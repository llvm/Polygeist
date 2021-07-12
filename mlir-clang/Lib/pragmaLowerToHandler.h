#ifndef MLIR_TOOLS_MLIRCLANG_LIB_PRAGMALOWERTOHANDLER_H
#define MLIR_TOOLS_MLIRCLANG_LIB_PRAGMALOWERTOHANDLER_H

#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"

/// POD holds information processed from the lower_to pragma.
struct LowerToInfo {
  llvm::StringMap<std::string> SymbolTable;
};

void addPragmaLowerToHandlers(clang::Preprocessor &PP, LowerToInfo &LTInfo);

#endif

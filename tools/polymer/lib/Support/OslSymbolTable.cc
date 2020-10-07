//===- OslSymbolTable.cc ----------------------------------------*- C++ -*-===//
//
// This file implements the OslSymbolTable class that stores the mapping between
// symbols and MLIR values.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslSymbolTable.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace llvm;

namespace polymer {

Value OslSymbolTable::getValue(StringRef key) {
  // Key is a loop IV.
  if (nameToLoopIV.find(key) != nameToLoopIV.end())
    return nameToLoopIV.lookup(key);
  // Key is a memref value.
  if (nameToMemref.find(key) != nameToMemref.end())
    return nameToMemref.lookup(key);
  // If nothing is found, return NULL.
  return nullptr;
}

Operation *OslSymbolTable::getOperation(StringRef key) {
  // If key corresponds to an Op of a statement.
  if (nameToStmtOp.find(key) != nameToStmtOp.end())
    return nameToStmtOp.lookup(key);

  return nullptr;
}

void OslSymbolTable::setValue(StringRef key, Value val, SymbolType type) {
  switch (type) {
  case LoopIV:
    nameToLoopIV[key] = val;
    break;
  case Memref:
    nameToMemref[key] = val;
    break;
  default:
    assert(false && "Symbole type for Value not recognized.");
  }
}

void OslSymbolTable::setOperation(StringRef key, Operation *val,
                                  SymbolType type) {
  switch (type) {
  case StmtOp:
    nameToStmtOp[key] = val;
    break;
  default:
    assert(false && "Symbole type for Operation not recognized.");
  }
}

unsigned OslSymbolTable::getNumValues(SymbolType type) {
  switch (type) {
  case LoopIV:
    return nameToLoopIV.size();
  case Memref:
    return nameToMemref.size();
  default:
    assert(false && "Symbole type for Value not recognized.");
  }
}

unsigned OslSymbolTable::getNumOperations(SymbolType type) {
  switch (type) {
  case StmtOp:
    return nameToStmtOp.size();
  default:
    assert(false && "Symbole type for Operation not recognized.");
  }
}

void OslSymbolTable::getValueSymbols(SmallVectorImpl<StringRef> &symbols) {
  symbols.reserve(getNumValues(Memref) + getNumValues(LoopIV));

  for (auto &it : nameToLoopIV)
    symbols.push_back(it.first());
  for (auto &it : nameToMemref)
    symbols.push_back(it.first());
}
void OslSymbolTable::getOperationSymbols(SmallVectorImpl<StringRef> &symbols) {
  symbols.reserve(getNumOperations(StmtOp));

  for (auto &it : nameToStmtOp)
    symbols.push_back(it.first());
}
} // namespace polymer

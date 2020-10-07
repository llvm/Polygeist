//===- OslSymbolTable.h -----------------------------------------*- C++ -*-===//
//
// This file declares the OslSymbolTable class that stores the mapping between
// symbols and MLIR values.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

using namespace llvm;
using namespace mlir;

namespace mlir {
class Operation;
class Value;
} // namespace mlir

namespace polymer {

class OslSymbolTable {
public:
  enum SymbolType { LoopIV, Memref, StmtOp };

  Value getValue(StringRef key);

  Operation *getOperation(StringRef key);

  void setValue(StringRef key, Value val, SymbolType type);

  void setOperation(StringRef key, Operation *val, SymbolType type);

  unsigned getNumValues(SymbolType type);

  unsigned getNumOperations(SymbolType type);

  void getValueSymbols(SmallVectorImpl<StringRef> &symbols);

  void getOperationSymbols(SmallVectorImpl<StringRef> &symbols);

private:
  StringMap<Operation *> nameToStmtOp;
  StringMap<Value> nameToLoopIV;
  StringMap<Value> nameToMemref;
};

} // namespace polymer

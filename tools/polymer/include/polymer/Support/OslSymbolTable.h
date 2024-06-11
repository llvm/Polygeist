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

class ScopStmtOpSet;

class PolymerSymbolTable {
public:
  using OpSet = ScopStmtOpSet;
  using OpSetPtr = std::unique_ptr<OpSet>;

  enum SymbolType { LoopIV, Memref, StmtOpSet };

  Value getValue(StringRef key);

  OpSet getOpSet(StringRef key);

  void setValue(StringRef key, Value val, SymbolType type);

  void setOpSet(StringRef key, OpSet val, SymbolType type);

  unsigned getNumValues(SymbolType type);

  unsigned getNumOpSets(SymbolType type);

  void getValueSymbols(SmallVectorImpl<StringRef> &symbols);

  void getOpSetSymbols(SmallVectorImpl<StringRef> &symbols);

private:
  StringMap<OpSet> nameToStmtOpSet;
  StringMap<Value> nameToLoopIV;
  StringMap<Value> nameToMemref;
};

} // namespace polymer

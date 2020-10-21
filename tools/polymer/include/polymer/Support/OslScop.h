//===- OslScop.h ------------------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//
#ifndef POLYMER_SUPPORT_OSLSCOP_H
#define POLYMER_SUPPORT_OSLSCOP_H

#include "mlir/Support/LLVM.h"

#include <cassert>
#include <cstdint>
#include <vector>

struct osl_scop;
struct osl_statement;
struct osl_generic;

namespace mlir {
class LogicalResult;
}

namespace polymer {

/// A wrapper for the osl_scop struct in the openscop library.
class OslScop {
public:
  OslScop();
  OslScop(osl_scop *scop) : scop(scop) {}

  ~OslScop();

  /// Get the raw scop pointer.
  osl_scop *get() { return scop; }

  /// Print the content of the Scop to the stdout.
  void print();

  /// Validate whether the scop is well-formed.
  bool validate();

  /// Simply create a new statement in the linked list scop->statement.
  void createStatement();

  /// Create a new relation and initialize its contents. The new relation will
  /// be created under the scop member.
  /// The target here is an index:
  /// 1) if it's 0, then it means the context;
  /// 2) otherwise, if it is a positive number, it corresponds to a statement of
  /// id=(target-1).
  void addRelation(int target, int type, int numRows, int numCols,
                   int numOutputDims, int numInputDims, int numLocalDims,
                   int numParams, llvm::ArrayRef<int64_t> eqs,
                   llvm::ArrayRef<int64_t> inEqs);

  /// Add a new generic field to a statement. `target` gives the statement ID.
  /// `content` specifies the data field in the generic.
  void addGeneric(int target, llvm::StringRef tag, llvm::StringRef content);

  /// Check whether the name refers to a symbol.
  bool isSymbol(llvm::StringRef name);

  /// Get statement by index.
  mlir::LogicalResult getStatement(unsigned index, osl_statement **stmt);

  /// Get extension by interface name
  osl_generic *getExtension(llvm::StringRef interface) const;

private:
  osl_scop *scop;
};

} // namespace polymer

#endif

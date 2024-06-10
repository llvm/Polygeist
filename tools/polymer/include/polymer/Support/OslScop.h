//===- OslScop.h ------------------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//
#ifndef POLYMER_SUPPORT_OSLSCOP_H
#define POLYMER_SUPPORT_OSLSCOP_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct osl_scop;
struct osl_statement;
struct osl_generic;

namespace mlir {
namespace affine {
class AffineValueMap;
class FlatAffineValueConstraints;
} // namespace affine
struct LogicalResult;
class Operation;
class Value;
namespace func {
class FuncOp;
}
} // namespace mlir

namespace polymer {

class ScopStmt;
class ScatTreeNode;

/// A wrapper for the osl_scop struct in the openscop library.
class OslScop {
public:
  using SymbolTable = llvm::StringMap<mlir::Value>;
  using ValueTable = llvm::DenseMap<mlir::Value, std::string>;
  using MemRefToId = llvm::DenseMap<mlir::Value, unsigned>;
  using ScopStmtMap = std::map<std::string, ScopStmt>;
  using ScopStmtNames = std::vector<std::string>;

  OslScop();
  OslScop(osl_scop *scop);

  ~OslScop();

  /// Get the raw scop pointer.
  osl_scop *get() { return scop; }

  /// Print the content of the Scop to the stdout.
  void print();

  /// Validate whether the scop is well-formed.
  bool validate();

  /// Simply create a new statement in the linked list scop->statement.
  void createStatement();
  /// Get statement by index.
  mlir::LogicalResult getStatement(unsigned index, osl_statement **stmt) const;
  /// Get the total number of statements
  unsigned getNumStatements() const;

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

  /// Add the relation defined by cst to the context of the current scop.
  void addContextRelation(mlir::affine::FlatAffineValueConstraints cst);
  /// Add the domain relation.
  void addDomainRelation(int stmtId,
                         mlir::affine::FlatAffineValueConstraints &cst);
  /// Add the scattering relation.
  void addScatteringRelation(int stmtId,
                             mlir::affine::FlatAffineValueConstraints &cst,
                             llvm::ArrayRef<mlir::Operation *> ops);
  /// Add the access relation.
  mlir::LogicalResult
  addAccessRelation(int stmtId, bool isRead, mlir::Value memref,
                    mlir::affine::AffineValueMap &vMap,
                    mlir::affine::FlatAffineValueConstraints &cst);

  /// Add a new generic field to a statement. `target` gives the statement ID.
  /// `content` specifies the data field in the generic.
  void addGeneric(int target, llvm::StringRef tag, llvm::StringRef content);
  void addExtensionGeneric(llvm::StringRef tag, llvm::StringRef content);
  void addParametersGeneric(llvm::StringRef tag, llvm::StringRef content);
  void addStatementGeneric(int stmtId, llvm::StringRef tag,
                           llvm::StringRef content);
  void addBodyExtension(int stmtId, const ScopStmt &stmt);

  /// Check whether the name refers to a symbol.
  bool isSymbol(llvm::StringRef name);

  /// Get extension by interface name
  osl_generic *getExtension(llvm::StringRef interface) const;

  /// Initialize the symbol table.
  void initializeSymbolTable(mlir::func::FuncOp f,
                             mlir::affine::FlatAffineValueConstraints *cst);

  bool isParameterSymbol(llvm::StringRef name) const;
  bool isDimSymbol(llvm::StringRef name) const;
  bool isArraySymbol(llvm::StringRef name) const;
  bool isConstantSymbol(llvm::StringRef name) const;

  /// Get the symbol table object.
  /// TODO: maybe not expose the symbol table to the external world like this.
  SymbolTable *getSymbolTable();
  ValueTable *getValueTable();

  /// Get the mapping from memref Value to its id.
  MemRefToId *getMemRefIdMap();

  /// Get the ScopStmtMap.
  ScopStmtMap *getScopStmtMap();

  /// Get the list of stmt names followed by their insertion order
  ScopStmtNames *getScopStmtNames();

private:
  /// Create a 1-d array that carries all the constraints in a relation,
  /// arranged in the row-major order.
  void createConstraintRows(mlir::affine::FlatAffineValueConstraints &cst,
                            llvm::SmallVectorImpl<int64_t> &eqs,
                            bool isEq = true);

  /// Create access relation constraints.
  mlir::LogicalResult createAccessRelationConstraints(
      mlir::affine::AffineValueMap &vMap,
      mlir::affine::FlatAffineValueConstraints &cst,
      mlir::affine::FlatAffineValueConstraints &domain);

  void addArraysExtension();
  void addScatnamesExtension();
  void addParameterNames();

  /// The internal storage of the Scop.
  osl_scop *scop;

  /// The scattering tree maintained.
  std::unique_ptr<ScatTreeNode> scatTreeRoot;
  /// Number of memrefs recorded.
  MemRefToId memRefIdMap;
  /// Symbol table for MLIR values.
  SymbolTable symbolTable;
  ValueTable valueTable;
  ///
  ScopStmtMap scopStmtMap;

  ScopStmtNames scopStmtNames;
};

/// A wrapper for the osl_scop struct in the openscop library.
class IslScop {
public:
  using SymbolTable = llvm::StringMap<mlir::Value>;
  using ValueTable = llvm::DenseMap<mlir::Value, std::string>;
  using MemRefToId = llvm::DenseMap<mlir::Value, unsigned>;
  using ScopStmtMap = std::map<std::string, ScopStmt>;
  using ScopStmtNames = std::vector<std::string>;

  IslScop();
  IslScop(osl_scop *scop);

  ~IslScop();

  /// Get the raw scop pointer.
  osl_scop *get() { return scop; }

  /// Print the content of the Scop to the stdout.
  void print();

  /// Validate whether the scop is well-formed.
  bool validate();

  /// Simply create a new statement in the linked list scop->statement.
  void createStatement();
  /// Get statement by index.
  mlir::LogicalResult getStatement(unsigned index, osl_statement **stmt) const;
  /// Get the total number of statements
  unsigned getNumStatements() const;

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

  /// Add the relation defined by cst to the context of the current scop.
  void addContextRelation(mlir::affine::FlatAffineValueConstraints cst);
  /// Add the domain relation.
  void addDomainRelation(int stmtId,
                         mlir::affine::FlatAffineValueConstraints &cst);
  /// Add the scattering relation.
  void addScatteringRelation(int stmtId,
                             mlir::affine::FlatAffineValueConstraints &cst,
                             llvm::ArrayRef<mlir::Operation *> ops);
  /// Add the access relation.
  mlir::LogicalResult
  addAccessRelation(int stmtId, bool isRead, mlir::Value memref,
                    mlir::affine::AffineValueMap &vMap,
                    mlir::affine::FlatAffineValueConstraints &cst);

  /// Add a new generic field to a statement. `target` gives the statement ID.
  /// `content` specifies the data field in the generic.
  void addGeneric(int target, llvm::StringRef tag, llvm::StringRef content);
  void addExtensionGeneric(llvm::StringRef tag, llvm::StringRef content);
  void addParametersGeneric(llvm::StringRef tag, llvm::StringRef content);
  void addStatementGeneric(int stmtId, llvm::StringRef tag,
                           llvm::StringRef content);
  void addBodyExtension(int stmtId, const ScopStmt &stmt);

  /// Check whether the name refers to a symbol.
  bool isSymbol(llvm::StringRef name);

  /// Get extension by interface name
  osl_generic *getExtension(llvm::StringRef interface) const;

  /// Initialize the symbol table.
  void initializeSymbolTable(mlir::func::FuncOp f,
                             mlir::affine::FlatAffineValueConstraints *cst);

  bool isParameterSymbol(llvm::StringRef name) const;
  bool isDimSymbol(llvm::StringRef name) const;
  bool isArraySymbol(llvm::StringRef name) const;
  bool isConstantSymbol(llvm::StringRef name) const;

  /// Get the symbol table object.
  /// TODO: maybe not expose the symbol table to the external world like this.
  SymbolTable *getSymbolTable();
  ValueTable *getValueTable();

  /// Get the mapping from memref Value to its id.
  MemRefToId *getMemRefIdMap();

  /// Get the ScopStmtMap.
  ScopStmtMap *getScopStmtMap();

  /// Get the list of stmt names followed by their insertion order
  ScopStmtNames *getScopStmtNames();

private:
  /// Create a 1-d array that carries all the constraints in a relation,
  /// arranged in the row-major order.
  void createConstraintRows(mlir::affine::FlatAffineValueConstraints &cst,
                            llvm::SmallVectorImpl<int64_t> &eqs,
                            bool isEq = true);

  /// Create access relation constraints.
  mlir::LogicalResult createAccessRelationConstraints(
      mlir::affine::AffineValueMap &vMap,
      mlir::affine::FlatAffineValueConstraints &cst,
      mlir::affine::FlatAffineValueConstraints &domain);

  void addArraysExtension();
  void addScatnamesExtension();
  void addParameterNames();

  /// The internal storage of the Scop.
  osl_scop *scop;

  /// The scattering tree maintained.
  std::unique_ptr<ScatTreeNode> scatTreeRoot;
  /// Number of memrefs recorded.
  MemRefToId memRefIdMap;
  /// Symbol table for MLIR values.
  SymbolTable symbolTable;
  ValueTable valueTable;
  ///
  ScopStmtMap scopStmtMap;

  ScopStmtNames scopStmtNames;
};

} // namespace polymer

#endif

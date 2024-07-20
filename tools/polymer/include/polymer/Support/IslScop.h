//===- IslScop.h ------------------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//
#ifndef POLYMER_SUPPORT_OSLSCOP_H
#define POLYMER_SUPPORT_OSLSCOP_H

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "polymer/Support/ScatteringUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct isl_schedule;
struct isl_union_set;
struct isl_mat;
struct isl_ctx;
struct isl_set;
struct isl_space;
struct isl_basic_set;
struct isl_basic_map;

#define __isl_keep
#define __isl_give
#define __isl_take

namespace mlir {
namespace affine {
class AffineValueMap;
class AffineForOp;
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

class IslMLIRBuilder;
class ScopStmt;

/// A wrapper for the osl_scop struct in the openscop library.
class IslScop {
public:
  using SymbolTable = llvm::StringMap<mlir::Value>;
  using ValueTable = llvm::DenseMap<mlir::Value, std::string>;
  using MemRefToId = llvm::DenseMap<mlir::Value, std::string>;
  using ScopStmtMap = std::map<std::string, ScopStmt>;
  using ScopStmtNames = std::vector<std::string>;

  IslScop();
  ~IslScop();

  /// Simply create a new statement in the linked list scop->statement.
  void createStatement();

  /// Add the relation defined by cst to the context of the current scop.
  void addContextRelation(mlir::affine::FlatAffineValueConstraints cst);
  /// Add the domain relation.
  void addDomainRelation(int stmtId,
                         mlir::affine::FlatAffineValueConstraints &cst);
  /// Add the access relation.
  mlir::LogicalResult
  addAccessRelation(int stmtId, bool isRead, mlir::Value memref,
                    mlir::affine::AffineValueMap &vMap,
                    mlir::affine::FlatAffineValueConstraints &cst);

  /// Initialize the symbol table.
  void initializeSymbolTable(mlir::func::FuncOp f,
                             mlir::affine::FlatAffineValueConstraints *cst);

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

  void dumpSchedule(llvm::raw_ostream &os);
  void dumpAccesses(llvm::raw_ostream &os);

  void buildSchedule(llvm::SmallVector<mlir::Operation *> ops) {
    loopId = 0;
    schedule = buildSequenceSchedule(ops);
  }

  static llvm::SmallVector<mlir::Operation *>
  getSequenceScheduleOpList(mlir::Operation *begin, mlir::Operation *end);
  static llvm::SmallVector<mlir::Operation *>
  getSequenceScheduleOpList(mlir::Block *block);

  isl_schedule *getSchedule() { return schedule; }

  mlir::func::FuncOp applySchedule(__isl_take isl_schedule *newSchedule,
                                   mlir::func::FuncOp f);

private:
  struct IslStmt {
    isl_basic_set *domain;
    std::vector<isl_basic_map *> readRelations;
    std::vector<isl_basic_map *> writeRelations;
  };
  std::vector<IslStmt> islStmts;
  isl_schedule *schedule = nullptr;
  unsigned loopId = 0;

  template <typename T>
  __isl_give isl_schedule *buildLoopSchedule(T loopOp, unsigned depth);
  __isl_give isl_schedule *
  buildParallelSchedule(mlir::affine::AffineParallelOp parallelOp,
                        unsigned depth);
  __isl_give isl_schedule *buildForSchedule(mlir::affine::AffineForOp forOp,
                                            unsigned depth);
  __isl_give isl_schedule *buildLeafSchedule(mlir::func::CallOp callOp);
  __isl_give isl_schedule *
  buildSequenceSchedule(llvm::SmallVector<mlir::Operation *> ops,
                        unsigned depth = 0);

  IslStmt &getIslStmt(std::string name);

  __isl_give isl_space *
  setupSpace(__isl_take isl_space *space,
             mlir::affine::FlatAffineValueConstraints &cst, std::string name);

  __isl_give isl_mat *
  createConstraintRows(mlir::affine::FlatAffineValueConstraints &cst,
                       bool isEq);

  /// Create access relation constraints.
  mlir::LogicalResult createAccessRelationConstraints(
      mlir::affine::AffineValueMap &vMap,
      mlir::affine::FlatAffineValueConstraints &cst,
      mlir::affine::FlatAffineValueConstraints &domain);

  /// The internal storage of the Scop.
  // osl_scop *scop;
  isl_ctx *ctx;

  /// Number of memrefs recorded.
  MemRefToId memRefIdMap;
  /// Symbol table for MLIR values.
  SymbolTable symbolTable;
  ValueTable valueTable;
  ///
  ScopStmtMap scopStmtMap;

  ScopStmtNames scopStmtNames;

  friend class IslMLIRBuilder;
};

} // namespace polymer

#endif

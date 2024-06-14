//===- IslScop.cc -----------------------------------------------*- C++ -*-===//

#include "polymer/Support/OslScop.h"
#include "polymer/Support/ScatteringUtils.h"
#include "polymer/Support/ScopStmt.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "isl/space_type.h"
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/mat.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>

using namespace polymer;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "islscop"

IslScop::IslScop() {
  scatTreeRoot = std::make_unique<ScatTreeNode>();
  ctx = isl_ctx_alloc();
  // TODO ISL
}

// IslScop::IslScop(osl_scop *scop)
//     : scop(scop), scatTreeRoot{std::make_unique<ScatTreeNode>()} {}

IslScop::~IslScop() {
  for (IslStmt &stmt : islStmts) {
    for (auto &rel : stmt.readRelations)
      rel = isl_basic_map_free(rel);
    for (auto &rel : stmt.writeRelations)
      rel = isl_basic_map_free(rel);
  }
  isl_ctx_free(ctx);
}

void IslScop::print() {
  // TODO ISL
}

bool IslScop::validate() {
  // TODO ISL
  return true;
}

void IslScop::createStatement() { islStmts.push_back({}); }

void IslScop::addRelation(int target, int type, int numRows, int numCols,
                          int numOutputDims, int numInputDims, int numLocalDims,
                          int numParams, llvm::ArrayRef<int64_t> eqs,
                          llvm::ArrayRef<int64_t> inEqs) {
  // TODO ISL
}

void IslScop::addContextRelation(affine::FlatAffineValueConstraints cst) {
  // Project out the dim IDs in the context with only the symbol IDs left.
  SmallVector<mlir::Value, 8> dimValues;
  cst.getValues(0, cst.getNumDimVars(), &dimValues);
  for (mlir::Value dimValue : dimValues)
    cst.projectOut(dimValue);
  if (cst.getNumDimAndSymbolVars() > 0)
    cst.removeIndependentConstraints(0, cst.getNumDimAndSymbolVars());

  paramSpace =
      isl_space_set_alloc(ctx, cst.getNumSymbolVars(), cst.getNumDimVars());
  LLVM_DEBUG(llvm::errs() << "context relation space: ");
  LLVM_DEBUG(isl_space_dump(paramSpace));
}

namespace {
struct IslStr {
  char *s;
  char *str() { return s; }
  IslStr(char *s) : s(s) {}
  ~IslStr() { free(s); }
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, IslStr s) {
  return os << s.str();
}
} // namespace

void IslScop::dumpTadashi(llvm::raw_ostream &os) {
  LLVM_DEBUG(llvm::errs() << "Dumping tadashi\n");
  auto o = [&os](unsigned n) -> llvm::raw_ostream & {
    return os << std::string(n, ' ');
  };

  isl_union_set *domain = isl_union_set_empty(paramSpace);
  for (IslStmt &stmt : islStmts) {
    isl_union_set *set = isl_union_set_from_basic_set(stmt.domain);
    domain = isl_union_set_union(set, domain);
  }

  o(0) << "domain: " << '"' << IslStr(isl_union_set_to_str(domain)) << '"'
       << "\n";

  os << "accesses:\n";
  for (unsigned stmtId = 0; stmtId < islStmts.size(); stmtId++) {
    auto &stmt = islStmts[stmtId];
    o(2) << "- " << scopStmtNames[stmtId] << ":"
         << "\n";
    o(6) << "reads:"
         << "\n";
    for (auto rel : stmt.readRelations)
      o(8) << "- " << '"' << IslStr(isl_basic_map_to_str(rel)) << '"' << "\n";
    o(6) << "writes:"
         << "\n";
    for (auto rel : stmt.writeRelations)
      o(8) << "- " << '"' << IslStr(isl_basic_map_to_str(rel)) << '"' << "\n";
  }
  domain = isl_union_set_free(domain);
}

void IslScop::addDomainRelation(int stmtId,
                                affine::FlatAffineValueConstraints &cst) {
  SmallVector<int64_t, 8> eqs, inEqs;
  isl_mat *eqMat = createConstraintRows(cst, /*isEq=*/true);
  isl_mat *ineqMat = createConstraintRows(cst, /*isEq=*/false);
  LLVM_DEBUG({
    llvm::errs() << "Adding domain relation\n";
    llvm::errs() << " ISL eq mat:\n";
    isl_mat_dump(eqMat);
    llvm::errs() << " ISL ineq mat:\n";
    isl_mat_dump(ineqMat);
    llvm::errs() << "\n";
  });

  isl_space *space =
      isl_space_set_alloc(ctx, cst.getNumSymbolVars(), cst.getNumDimVars());
  space = isl_space_set_tuple_name(space, isl_dim_set,
                                   scopStmtNames[stmtId].c_str());
  LLVM_DEBUG(llvm::errs() << "space: ");
  LLVM_DEBUG(isl_space_dump(space));

  islStmts[stmtId].domain = isl_basic_set_from_constraint_matrices(
      space, eqMat, ineqMat, isl_dim_div, isl_dim_set, isl_dim_param,
      isl_dim_cst);
  LLVM_DEBUG(llvm::errs() << "bset: ");
  LLVM_DEBUG(isl_basic_set_dump(islStmts[stmtId].domain));
}

void IslScop::addScatteringRelation(
    int stmtId, mlir::affine::FlatAffineValueConstraints &cst,
    llvm::ArrayRef<mlir::Operation *> ops) {
  // First insert the enclosing ops into the scat tree.
  SmallVector<unsigned, 8> scats;
  scatTreeRoot->insertScopStmt(ops, scats);

  // Elements (N of them) in `scattering` are constants, and there are IVs
  // interleaved them. Therefore, we have 2N - 1 number of scattering
  // equalities.
  unsigned numScatEqs = scats.size() * 2 - 1;
  // Columns include new scattering dimensions and those from the domain.
  unsigned numScatCols = numScatEqs + cst.getNumCols() + 1;

  // Initialize contents for equalities.
  isl_mat *eqMat = isl_mat_alloc(ctx, numScatEqs, numScatCols);
  for (unsigned j = 0; j < numScatEqs; j++) {

    // Initializing scattering dimensions by setting the diagonal to -1.
    for (unsigned k = 0; k < numScatEqs; k++)
      eqMat = isl_mat_set_element_si(eqMat, j, k, (k == j));

    // Relating the loop IVs to the scattering dimensions. If it's the odd
    // equality, set its scattering dimension to the loop IV; otherwise, it's
    // scattering dimension will be set in the following constant section.
    for (unsigned k = 0; k < cst.getNumDimVars(); k++)
      eqMat = isl_mat_set_element_si(eqMat, j, k + numScatEqs,
                                     (j % 2) ? (k == (j / 2)) : 0);

    // TODO: consider the parameters that may appear in the scattering
    // dimension.
    for (unsigned k = 0; k < cst.getNumLocalVars() + cst.getNumSymbolVars();
         k++)
      eqMat = isl_mat_set_element_si(eqMat, j,
                                     k + numScatEqs + cst.getNumDimVars(), 0);

    // Relating the constants (the last column) to the scattering dimensions.
    eqMat = isl_mat_set_element_si(eqMat, j, numScatCols - 2,
                                   (j % 2) ? 0 : scats[j / 2]);
  }
  LLVM_DEBUG({
    llvm::errs() << "Adding scattering relation\n";
    llvm::errs() << " ISL eq mat:\n";
    isl_mat_dump(eqMat);
    llvm::errs() << "\n";
  });
  isl_mat_free(eqMat);

  // Then put them into the scop as a SCATTERING relation.
  // addRelation(stmtId + 1, OSL_TYPE_SCATTERING, numScatEqs, numScatCols,
  //             numScatEqs, cst.getNumDimVars(), cst.getNumLocalVars(),
  //             cst.getNumSymbolVars(), eqs, inEqs);
}

LogicalResult
IslScop::addAccessRelation(int stmtId, bool isRead, mlir::Value memref,
                           affine::AffineValueMap &vMap,
                           affine::FlatAffineValueConstraints &domain) {
  affine::FlatAffineValueConstraints cst;
  // Insert the address dims and put constraints in it.
  if (createAccessRelationConstraints(vMap, cst, domain).failed()) {
    LLVM_DEBUG(llvm::dbgs() << "createAccessRelationConstraints failed\n");
    return failure();
  }

  // Create a new dim of memref and set its value to its corresponding ID.
  memRefIdMap.try_emplace(memref, "A" + std::to_string(memRefIdMap.size() + 1));

  isl_mat *eqMat = createConstraintRows(cst, /*isEq=*/true);
  isl_mat *ineqMat = createConstraintRows(cst, /*isEq=*/false);

  unsigned rows = isl_mat_rows(eqMat);
  unsigned cols = 2 * domain.getNumDimVars();
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      isl_val *val = isl_mat_get_element_val(eqMat, i, j);
      val = isl_val_neg(val);
      eqMat = isl_mat_set_element_val(eqMat, i, j, val);
    }
  }

  LLVM_DEBUG({
    llvm::errs() << "Adding access relation\n";
    dbgs() << "Resolved MLIR access constraints:\n";
    cst.dump();
    llvm::errs() << " ISL eq mat:\n";
    isl_mat_dump(eqMat);
    llvm::errs() << " ISL ineq mat:\n";
    isl_mat_dump(ineqMat);
    llvm::errs() << "\n";
  });

  assert(cst.getNumInequalities() == 0);
  isl_space *space =
      isl_space_alloc(ctx, domain.getNumSymbolVars(), domain.getNumDimVars(),
                      cst.getNumConstraints());
  space = isl_space_set_tuple_name(space, isl_dim_in,
                                   scopStmtNames[stmtId].c_str());
  space =
      isl_space_set_tuple_name(space, isl_dim_out, memRefIdMap[memref].c_str());

  isl_basic_map *bmap = isl_basic_map_from_constraint_matrices(
      space, eqMat, ineqMat, isl_dim_in, isl_dim_div, isl_dim_out,
      isl_dim_param, isl_dim_cst);
  if (isRead)
    islStmts[stmtId].readRelations.push_back(bmap);
  else
    islStmts[stmtId].writeRelations.push_back(bmap);

  return success();
}

void IslScop::addGeneric(int target, llvm::StringRef tag,
                         llvm::StringRef content) {
  // TODO ISL
}

void IslScop::addExtensionGeneric(llvm::StringRef tag,
                                  llvm::StringRef content) {
  addGeneric(0, tag, content);
}

void IslScop::addParametersGeneric(llvm::StringRef tag,
                                   llvm::StringRef content) {
  addGeneric(-1, tag, content);
}

void IslScop::addStatementGeneric(int stmtId, llvm::StringRef tag,
                                  llvm::StringRef content) {
  addGeneric(stmtId + 1, tag, content);
}

/// We determine whether the name refers to a symbol by looking up the parameter
/// list of the scop.
bool IslScop::isSymbol(llvm::StringRef name) {
  // TODO ISL
}

// LogicalResult IslScop::getStatement(unsigned index,
//                                     osl_statement **stmt) const {
//   // TODO ISL
// }

// unsigned IslScop::getNumStatements() const {
//   return osl_statement_number(scop->statement);
// }

// osl_generic_p IslScop::getExtension(llvm::StringRef tag) const {
// }

void IslScop::addParameterNames() {
  std::string body;
  llvm::raw_string_ostream ss(body);

  SmallVector<std::string, 8> names;

  for (const auto &it : symbolTable)
    if (isParameterSymbol(it.first()))
      names.push_back(std::string(it.first()));

  std::sort(names.begin(), names.end());
  for (const auto &s : names)
    ss << s << " ";

  addParametersGeneric("strings", body);
}

void IslScop::addScatnamesExtension() {
  std::string body;
  llvm::raw_string_ostream ss(body);

  unsigned numScatnames = scatTreeRoot->getDepth();
  numScatnames = (numScatnames - 2) * 2 + 1;
  for (unsigned i = 0; i < numScatnames; i++)
    ss << formatv("c{0}", i + 1) << " ";

  addExtensionGeneric("scatnames", body);
}

void IslScop::addArraysExtension() {
  std::string body;
  llvm::raw_string_ostream ss(body);

  unsigned numArraySymbols = 0;
  for (const auto &it : symbolTable)
    if (isArraySymbol(it.first())) {
      ss << it.first().drop_front() << " " << it.first() << " ";
      numArraySymbols++;
    }

  addExtensionGeneric("arrays",
                      std::string(formatv("{0} {1}", numArraySymbols, body)));
}

void IslScop::addBodyExtension(int stmtId, const ScopStmt &stmt) {
  std::string body;
  llvm::raw_string_ostream ss(body);

  SmallVector<mlir::Operation *, 8> forOps;
  stmt.getEnclosingOps(forOps, /*forOnly=*/true);

  unsigned numIVs = forOps.size();
  ss << numIVs << " ";

  llvm::DenseMap<mlir::Value, unsigned> ivToId;
  for (unsigned i = 0; i < numIVs; i++) {
    mlir::affine::AffineForOp forOp =
        cast<mlir::affine::AffineForOp>(forOps[i]);
    // forOp.dump();
    ivToId[forOp.getInductionVar()] = i;
  }

  for (unsigned i = 0; i < numIVs; i++)
    ss << "i" << i << " ";

  mlir::func::CallOp caller = stmt.getCaller();
  mlir::func::FuncOp callee = stmt.getCallee();
  ss << "\n" << callee.getName() << "(";

  SmallVector<std::string, 8> ivs;
  llvm::SetVector<unsigned> visited;
  for (unsigned i = 0; i < caller.getNumOperands(); i++) {
    mlir::Value operand = caller.getOperand(i);
    if (ivToId.find(operand) != ivToId.end()) {
      ivs.push_back(std::string(formatv("i{0}", ivToId[operand])));
      visited.insert(ivToId[operand]);
    }
  }

  for (unsigned i = 0; i < numIVs; i++)
    if (!visited.contains(i)) {
      visited.insert(i);
      ivs.push_back(std::string(formatv("i{0}", i)));
    }

  for (unsigned i = 0; i < ivs.size(); i++) {
    ss << ivs[i];
    if (i != ivs.size() - 1)
      ss << ", ";
  }

  // for (unsigned i = 0; i < numIVs; i++) {
  //   ss << "i" << i;
  //   if (i != numIVs - 1)
  //     ss << ", ";
  // }

  ss << ")";

  addGeneric(stmtId + 1, "body", body);
}

void IslScop::initializeSymbolTable(mlir::func::FuncOp f,
                                    affine::FlatAffineValueConstraints *cst) {
  symbolTable.clear();

  unsigned numDimIds = cst->getNumDimVars();
  unsigned numSymbolIds = cst->getNumDimAndSymbolVars() - numDimIds;

  SmallVector<mlir::Value, 8> dimValues, symbolValues;
  cst->getValues(0, numDimIds, &dimValues);
  cst->getValues(numDimIds, cst->getNumDimAndSymbolVars(), &symbolValues);

  // Setup the symbol table.
  for (unsigned i = 0; i < numDimIds; i++) {
    std::string sym(formatv("i{0}", i));
    symbolTable.insert(std::make_pair(sym, dimValues[i]));
    valueTable.insert(std::make_pair(dimValues[i], sym));
  }
  for (unsigned i = 0; i < numSymbolIds; i++) {
    std::string sym(formatv("P{0}", i));
    symbolTable.insert(std::make_pair(sym, symbolValues[i]));
    valueTable.insert(std::make_pair(symbolValues[i], sym));
  }
  for (const auto &it : memRefIdMap) {
    std::string sym(formatv("A{0}", it.second));
    symbolTable.insert(std::make_pair(sym, it.first));
    valueTable.insert(std::make_pair(it.first, sym));
  }
  // constants
  unsigned numConstants = 0;
  for (mlir::Value arg : f.getBody().begin()->getArguments()) {
    if (valueTable.find(arg) == valueTable.end()) {
      std::string sym(formatv("C{0}", numConstants++));
      symbolTable.insert(std::make_pair(sym, arg));
      valueTable.insert(std::make_pair(arg, sym));
    }
  }

  // Setup relative fields in the OpenScop representation.
  // Parameter names
  addParameterNames();
  // Scat names
  addScatnamesExtension();
  // Array names
  addArraysExtension();
}

bool IslScop::isParameterSymbol(llvm::StringRef name) const {
  return name.startswith("P");
}

bool IslScop::isDimSymbol(llvm::StringRef name) const {
  return name.startswith("i");
}

bool IslScop::isArraySymbol(llvm::StringRef name) const {
  return name.startswith("A");
}

bool IslScop::isConstantSymbol(llvm::StringRef name) const {
  return name.startswith("C");
}

isl_mat *IslScop::createConstraintRows(affine::FlatAffineValueConstraints &cst,
                                       bool isEq) {
  unsigned numRows = isEq ? cst.getNumEqualities() : cst.getNumInequalities();
  unsigned numDimIds = cst.getNumDimVars();
  unsigned numLocalIds = cst.getNumLocalVars();
  unsigned numSymbolIds = cst.getNumSymbolVars();

  LLVM_DEBUG(llvm::errs() << "createConstraintRows " << numRows << " "
                          << numDimIds << " " << numLocalIds << " "
                          << numSymbolIds << "\n");

  unsigned numCols = cst.getNumCols();
  isl_mat *mat = isl_mat_alloc(ctx, numRows, numCols);

  for (unsigned i = 0; i < numRows; i++) {
    // Get the row based on isEq.
    auto row = isEq ? cst.getEquality(i) : cst.getInequality(i);

    assert(row.size() == numCols);

    // Dims stay at the same positions.
    for (unsigned j = 0; j < numDimIds; j++)
      mat = isl_mat_set_element_si(mat, i, j, (int64_t)row[j]);
    // Output local ids before symbols.
    for (unsigned j = 0; j < numLocalIds; j++)
      mat = isl_mat_set_element_si(mat, i, j + numDimIds,
                                   (int64_t)row[j + numDimIds + numSymbolIds]);
    // Output symbols in the end.
    for (unsigned j = 0; j < numSymbolIds; j++)
      mat = isl_mat_set_element_si(mat, i, j + numDimIds + numLocalIds,
                                   (int64_t)row[j + numDimIds]);
    // Finally outputs the constant.
    mat =
        isl_mat_set_element_si(mat, i, numCols - 1, (int64_t)row[numCols - 1]);
  }
  return mat;
}

LogicalResult IslScop::createAccessRelationConstraints(
    mlir::affine::AffineValueMap &vMap,
    mlir::affine::FlatAffineValueConstraints &cst,
    mlir::affine::FlatAffineValueConstraints &domain) {
  cst = mlir::affine::FlatAffineValueConstraints();
  cst.mergeAndAlignVarsWithOther(0, &domain);

  LLVM_DEBUG({
    dbgs() << "Building access relation.\n"
           << " + Domain:\n";
    domain.dump();
  });

  SmallVector<mlir::Value, 8> idValues;
  domain.getValues(0, domain.getNumDimAndSymbolVars(), &idValues);
  llvm::SetVector<mlir::Value> idValueSet;
  for (auto val : idValues)
    idValueSet.insert(val);

  for (auto operand : vMap.getOperands())
    if (!idValueSet.contains(operand)) {
      llvm::errs() << "Operand missing: ";
      operand.dump();
    }

  // The results of the affine value map, which are the access addresses, will
  // be placed to the leftmost of all columns.
  return cst.composeMap(&vMap);
}

IslScop::SymbolTable *IslScop::getSymbolTable() { return &symbolTable; }

IslScop::ValueTable *IslScop::getValueTable() { return &valueTable; }

IslScop::MemRefToId *IslScop::getMemRefIdMap() { return &memRefIdMap; }

IslScop::ScopStmtMap *IslScop::getScopStmtMap() { return &scopStmtMap; }

IslScop::ScopStmtNames *IslScop::getScopStmtNames() { return &scopStmtNames; }

//===- IslScop.cc -----------------------------------------------*- C++ -*-===//

#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Support/LLVM.h"
#include "pluto/internal/pluto.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "isl/aff_type.h"
#include "isl/id.h"
#include "isl/space_type.h"
#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/mat.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>

using namespace polymer;
using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "islscop"

IslScop::IslScop() { ctx = isl_ctx_alloc(); }

IslScop::~IslScop() {
  for (auto &stmt : islStmts) {
    for (auto &rel : stmt.readRelations)
      rel = isl_basic_map_free(rel);
    for (auto &rel : stmt.writeRelations)
      rel = isl_basic_map_free(rel);
  }
  domain = isl_union_set_free(domain);
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

  paramSpace = getSpace(cst, "paramspace");
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

void IslScop::computeDomainFromStatementDomains() {
  assert(!domain);
  domain = isl_union_set_empty(paramSpace);
  for (IslStmt &stmt : islStmts) {
    isl_union_set *set =
        isl_union_set_from_basic_set(isl_basic_set_copy(stmt.domain));
    domain = isl_union_set_union(set, domain);
  }
  assert(domain);
}

inline void islAssert(const isl_size &size) { assert(size != isl_size_error); }
inline unsigned unsignedFromIslSize(const isl_size &size) {
  islAssert(size);
  return static_cast<unsigned>(size);
}

#define ISL_DEBUG(S, X)                                                        \
  LLVM_DEBUG({                                                                 \
    llvm::dbgs() << S;                                                         \
    X;                                                                         \
    llvm::dbgs() << "\n";                                                      \
  })

static isl_multi_union_pw_aff *mapToDimension(isl_union_set *uset, unsigned N) {
  assert(!isl_union_set_is_empty(uset));
  // assert(!isl_union_set_is_null(uset));
  N += 1;

  ISL_DEBUG("MTD USET: ", isl_union_set_dump(uset));
  auto res = isl_union_pw_multi_aff_empty(isl_union_set_get_space(uset));
  ISL_DEBUG("MTD RES: ", isl_union_pw_multi_aff_dump(res));

  isl_set_list *bsetlist = isl_union_set_get_set_list(uset);
  for (unsigned i = 0; i < unsignedFromIslSize(isl_set_list_size(bsetlist));
       i++) {
    isl_set *set = isl_set_list_get_at(bsetlist, i);
    ISL_DEBUG("MTD SET: ", isl_set_dump(set));
    unsigned Dim = unsignedFromIslSize(isl_set_dim(set, isl_dim_set));
    assert(Dim >= N);
    auto pma = isl_pw_multi_aff_project_out_map(isl_set_get_space(set),
                                                isl_dim_set, N, Dim - N);
    ISL_DEBUG("MTD PMA: ", isl_pw_multi_aff_dump(pma));
    if (N > 1)
      pma = isl_pw_multi_aff_drop_dims(pma, isl_dim_out, 0, N - 1);
    ISL_DEBUG("MTD PMA: ", isl_pw_multi_aff_dump(pma));

    res = isl_union_pw_multi_aff_add_pw_multi_aff(res, pma);
    ISL_DEBUG("MTD RES: ", isl_union_pw_multi_aff_dump(res));
  }

  return isl_multi_union_pw_aff_from_union_pw_multi_aff(res);
}

isl_schedule *
IslScop::buildParallelSchedule(affine::AffineParallelOp parallelOp,
                               unsigned depth) {
  isl_schedule *schedule = buildLoopSchedule(parallelOp, depth);
  isl_schedule_node *node = isl_schedule_get_root(schedule);
  node = isl_schedule_node_first_child(node);
  node = isl_schedule_node_band_set_permutable(node, 1);
  schedule = isl_schedule_node_get_schedule(node);
  return schedule;
}

template <typename T>
isl_schedule *IslScop::buildLoopSchedule(T loopOp, unsigned depth) {
  SmallVector<Operation *> body;
  for (auto it = loopOp.getBody()->begin();
       it != std::next(loopOp.getBody()->end(), -1); it++)
    body.push_back(&*it);

  isl_schedule *child = buildSequenceSchedule(body, depth + 1);
  ISL_DEBUG("CHILD:\n", isl_schedule_dump(child));
  isl_union_set *domain = isl_schedule_get_domain(child);
  isl_schedule *schedule = isl_schedule_from_domain(domain);
  ISL_DEBUG("MUPA dom: ", isl_union_set_dump(domain));
  isl_multi_union_pw_aff *mupa = mapToDimension(domain, depth);
  mupa = isl_multi_union_pw_aff_set_tuple_name(
      mupa, isl_dim_set, ("L" + std::to_string(loopId++)).c_str());
  ISL_DEBUG("MUPA: ", isl_multi_union_pw_aff_dump(mupa));
  schedule = isl_schedule_insert_partial_schedule(child, mupa);

  ISL_DEBUG("Created for schedule:\n", isl_schedule_dump(schedule));

  return schedule;
}

isl_schedule *IslScop::buildForSchedule(affine::AffineForOp forOp,
                                        unsigned depth) {
  isl_schedule *schedule = buildLoopSchedule(forOp, depth);
  return schedule;
}

isl_schedule *IslScop::buildLeafSchedule(func::CallOp callOp) {
  // TODO check that we are really calling  a statement
  auto &stmt = getIslStmt(callOp.getCallee().str());
  auto *schedule = isl_schedule_from_domain(
      isl_union_set_from_basic_set(isl_basic_set_copy(stmt.domain)));
  LLVM_DEBUG({
    llvm::errs() << "Created leaf schedule:\n";
    isl_schedule_dump(schedule);
    llvm::errs() << "\n";
  });
  return schedule;
}

isl_schedule *IslScop::buildSequenceSchedule(SmallVector<Operation *> ops,
                                             unsigned depth) {
  auto len = ops.size();
  if (len == 1) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(ops[0])) {
      return buildForSchedule(forOp, depth);
    } else if (auto parallelOp = dyn_cast<affine::AffineParallelOp>(ops[0])) {
      return buildParallelSchedule(parallelOp, depth);
    } else if (auto callOp = dyn_cast<func::CallOp>(ops[0])) {
      return buildLeafSchedule(callOp);
    } else {
      llvm_unreachable("only for ops for now");
    }
  }

  isl_schedule *schedule = nullptr;
  for (auto curOp : ops) {
    isl_schedule *child;
    if (auto forOp = dyn_cast<affine::AffineForOp>(curOp)) {
      child = buildForSchedule(forOp, depth);
    } else if (auto parallelOp = dyn_cast<affine::AffineParallelOp>(curOp)) {
      child = buildParallelSchedule(parallelOp, depth);
    } else if (auto callOp = dyn_cast<func::CallOp>(curOp)) {
      child = buildLeafSchedule(callOp);
    } else {
      llvm_unreachable("only for ops for now");
    }

    if (!schedule)
      schedule = child;
    else
      schedule = isl_schedule_sequence(schedule, child);
  }

  LLVM_DEBUG({
    llvm::errs() << "Created sequence schedule:\n";
    isl_schedule_dump(schedule);
    llvm::errs() << "\n";
  });

  return schedule;
}

IslScop::IslStmt &IslScop::getIslStmt(std::string name) {
  auto found = std::find(scopStmtNames.begin(), scopStmtNames.end(), name);
  assert(found != scopStmtNames.end());
  auto id = std::distance(scopStmtNames.begin(), found);
  return islStmts[id];
}

void IslScop::dumpTadashi(llvm::raw_ostream &os) {
  LLVM_DEBUG(llvm::errs() << "Dumping tadashi\n");
  LLVM_DEBUG(llvm::errs() << "Schedule:\n\n");
  LLVM_DEBUG(isl_schedule_dump(schedule));
  LLVM_DEBUG(llvm::errs() << "\n");

  auto o = [&os](unsigned n) -> llvm::raw_ostream & {
    return os << std::string(n, ' ');
  };

  o(0) << "schedule: " << IslStr(isl_schedule_to_str(schedule)) << "\n";

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

isl_space *IslScop::getSpace(affine::FlatAffineValueConstraints &cst,
                             std::string name) {
  isl_space *space =
      isl_space_set_alloc(ctx, cst.getNumSymbolVars(), cst.getNumDimVars());
  for (unsigned i = 0; i < cst.getNumSymbolVars(); i++) {
    Value val =
        cst.getValue(cst.getVarKindOffset(presburger::VarKind::Symbol) + i);
    std::string sym = valueTable[val];
    isl_id *id = isl_id_alloc(ctx, sym.c_str(), nullptr);
    space = isl_space_set_dim_id(space, isl_dim_param, i, id);
  }
  space = isl_space_set_tuple_name(space, isl_dim_set, name.c_str());
  return space;
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

  isl_space *space = getSpace(cst, scopStmtNames[stmtId]);
  LLVM_DEBUG(llvm::errs() << "space: ");
  LLVM_DEBUG(isl_space_dump(space));
  islStmts[stmtId].domain = isl_basic_set_from_constraint_matrices(
      space, eqMat, ineqMat, isl_dim_div, isl_dim_set, isl_dim_param,
      isl_dim_cst);
  LLVM_DEBUG(llvm::errs() << "bset: ");
  LLVM_DEBUG(isl_basic_set_dump(islStmts[stmtId].domain));
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
  isl_space *space = getSpace(cst, memRefIdMap[memref]);

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
  return true;
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

void IslScop::addScatnamesExtension() {}

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

  // Setup the symbol table.
  for (unsigned i = 0; i < cst->getNumVars(); i++) {
    Value val = cst->getValue(i);
    std::string sym;
    switch (cst->getVarKindAt(i)) {
    case presburger::VarKind::Domain:
      sym = "I";
      break;
    case presburger::VarKind::Local:
      sym = "O";
      break;
    case presburger::VarKind::Symbol:
      sym = "P";
      break;
    case presburger::VarKind::Range:
      sym = "R";
      break;
    }
    sym += std::to_string(i - cst->getVarKindOffset(cst->getVarKindAt(i)));
    // symbolTable.insert(std::make_pair(sym, val));
    // valueTable.insert(std::make_pair(val, sym));
    valueTable[val] = sym;
    symbolTable[sym] = val;
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

  // // Setup relative fields in the OpenScop representation.
  // // Parameter names
  // addParameterNames();
  // // Scat names
  // addScatnamesExtension();
  // // Array names
  // addArraysExtension();
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

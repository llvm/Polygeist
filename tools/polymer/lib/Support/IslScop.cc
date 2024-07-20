//===- IslScop.cc -----------------------------------------------*- C++ -*-===//

#include "polymer/Support/IslScop.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
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

#include "polly/CodeGen/IslNodeBuilder.h"
#include "polly/Support/GICHelper.h"

#include "isl/aff_type.h"
#include "isl/ast.h"
#include "isl/id_to_id.h"
#include "isl/printer.h"
#include "isl/space_type.h"
#include <isl/aff.h>
#include <isl/ast_build.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/map.h>
#include <isl/mat.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>
#include <memory>

using namespace polymer;
using namespace mlir;

using llvm::dbgs;
using llvm::errs;
using llvm::formatv;

#define DEBUG_TYPE "islscop"

IslScop::IslScop() { ctx = isl_ctx_alloc(); }

IslScop::~IslScop() {
  for (auto &stmt : islStmts) {
    for (auto &rel : stmt.readRelations)
      rel = isl_basic_map_free(rel);
    for (auto &rel : stmt.writeRelations)
      rel = isl_basic_map_free(rel);
    stmt.domain = isl_basic_set_free(stmt.domain);
  }
  isl_schedule_free(schedule);
  isl_ctx_free(ctx);
}

void IslScop::createStatement() { islStmts.push_back({}); }

void IslScop::addContextRelation(affine::FlatAffineValueConstraints cst) {
  // Project out the dim IDs in the context with only the symbol IDs left.
  SmallVector<mlir::Value, 8> dimValues;
  cst.getValues(0, cst.getNumDimVars(), &dimValues);
  for (mlir::Value dimValue : dimValues)
    cst.projectOut(dimValue);
  if (cst.getNumDimAndSymbolVars() > 0)
    cst.removeIndependentConstraints(0, cst.getNumDimAndSymbolVars());
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

static __isl_give isl_multi_union_pw_aff *
mapToDimension(__isl_take isl_union_set *uset, unsigned N) {
  assert(!isl_union_set_is_empty(uset));
  N += 1;

  isl_union_pw_multi_aff *res =
      isl_union_pw_multi_aff_empty(isl_union_set_get_space(uset));
  isl_set_list *bsetlist = isl_union_set_get_set_list(uset);
  for (unsigned i = 0; i < unsignedFromIslSize(isl_set_list_size(bsetlist));
       i++) {
    isl_set *set = isl_set_list_get_at(bsetlist, i);
    unsigned Dim = unsignedFromIslSize(isl_set_dim(set, isl_dim_set));
    assert(Dim >= N);
    auto pma = isl_pw_multi_aff_project_out_map(isl_set_get_space(set),
                                                isl_dim_set, N, Dim - N);
    isl_set_free(set);

    if (N > 1)
      pma = isl_pw_multi_aff_drop_dims(pma, isl_dim_out, 0, N - 1);
    res = isl_union_pw_multi_aff_add_pw_multi_aff(res, pma);
  }

  isl_set_list_free(bsetlist);
  isl_union_set_free(uset);

  return isl_multi_union_pw_aff_from_union_pw_multi_aff(res);
}

static constexpr char parallelLoopMark[] = "parallel";
static isl_id *getParallelLoopMark(isl_ctx *ctx) {
  isl_id *loopMark = isl_id_alloc(ctx, parallelLoopMark, nullptr);
  return loopMark;
}
static bool isParallelLoopMark(isl_id *id) {
  return std::string(parallelLoopMark) == isl_id_get_name(id);
}

isl_schedule *
IslScop::buildParallelSchedule(affine::AffineParallelOp parallelOp,
                               unsigned depth) {
  isl_schedule *schedule = buildLoopSchedule(parallelOp, depth);
  isl_schedule_node *node = isl_schedule_get_root(schedule);
  schedule = isl_schedule_free(schedule);
  node = isl_schedule_node_first_child(node);
  node = isl_schedule_node_band_set_permutable(node, 1);
  node = isl_schedule_node_insert_mark(node, getParallelLoopMark(ctx));
  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);
  return schedule;
}

template <typename T>
isl_schedule *IslScop::buildLoopSchedule(T loopOp, unsigned depth) {
  SmallVector<Operation *> body = getSequenceScheduleOpList(loopOp.getBody());

  isl_schedule *child = buildSequenceSchedule(body, depth + 1);
  ISL_DEBUG("CHILD:\n", isl_schedule_dump(child));
  isl_union_set *domain = isl_schedule_get_domain(child);
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
  // TODO check that we are really calling a statement
  auto &stmt = getIslStmt(callOp.getCallee().str());
  isl_schedule *schedule = isl_schedule_from_domain(
      isl_union_set_from_basic_set(isl_basic_set_copy(stmt.domain)));
  LLVM_DEBUG({
    llvm::errs() << "Created leaf schedule:\n";
    isl_schedule_dump(schedule);
    llvm::errs() << "\n";
  });
  return schedule;
}

SmallVector<Operation *> IslScop::getSequenceScheduleOpList(Operation *begin,
                                                            Operation *end) {
  SmallVector<Operation *> ops;
  for (auto op = begin; op != end; op = op->getNextNode()) {
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
      auto thenOps = getSequenceScheduleOpList(ifOp.getThenBlock());
      if (ifOp.hasElse()) {
        auto elseOps = getSequenceScheduleOpList(ifOp.getElseBlock());
        ops.insert(ops.end(), elseOps.begin(), elseOps.end());
      }
      ops.insert(ops.end(), thenOps.begin(), thenOps.end());
    } else {
      ops.push_back(op);
    }
  }
  return ops;
}

SmallVector<Operation *> IslScop::getSequenceScheduleOpList(Block *block) {
  // We cannot be yielding anything
  assert(block->back().getNumOperands() == 0);
  return getSequenceScheduleOpList(&block->front(), &block->back());
}

isl_schedule *IslScop::buildSequenceSchedule(SmallVector<Operation *> ops,
                                             unsigned depth) {
  auto buildOpSchedule = [&](Operation *op) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
      return buildForSchedule(forOp, depth);
    } else if (auto parallelOp = dyn_cast<affine::AffineParallelOp>(op)) {
      return buildParallelSchedule(parallelOp, depth);
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      return buildLeafSchedule(callOp);
    } else if (auto alloca = dyn_cast<memref::AllocaOp>(op)) {
      return (isl_schedule *)nullptr;
    } else {
      llvm_unreachable("unhandled op");
    }
  };

  auto len = ops.size();
  if (len == 1)
    return buildOpSchedule(ops[0]);

  isl_schedule *schedule = nullptr;
  for (auto curOp : ops) {
    isl_schedule *child = buildOpSchedule(curOp);
    if (!child)
      continue;
    if (!schedule)
      schedule = child;
    else
      schedule = isl_schedule_sequence(schedule, child);
  }
  assert(schedule);

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

void IslScop::dumpAccesses(llvm::raw_ostream &os) {
  auto o = [&os](unsigned n) -> llvm::raw_ostream & {
    return os << std::string(n, ' ');
  };
  isl_union_set *domain = isl_schedule_get_domain(schedule);
  o(0) << "domain: \"" << IslStr(isl_union_set_to_str(domain)) << "\"\n";
  domain = isl_union_set_free(domain);
  o(0) << "accesses:\n";
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
}
void IslScop::dumpSchedule(llvm::raw_ostream &os) {
  LLVM_DEBUG(llvm::errs() << "Dumping islexternal\n");
  LLVM_DEBUG(llvm::errs() << "Schedule:\n\n");
  LLVM_DEBUG(isl_schedule_dump(schedule));
  LLVM_DEBUG(llvm::errs() << "\n");

  os << IslStr(isl_schedule_to_str(schedule)) << "\n";
}

isl_space *IslScop::setupSpace(isl_space *space,
                               affine::FlatAffineValueConstraints &cst,
                               std::string name) {
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

  isl_space *space =
      isl_space_set_alloc(ctx, cst.getNumSymbolVars(), cst.getNumDimVars());
  space = setupSpace(space, cst, scopStmtNames[stmtId]);
  LLVM_DEBUG(llvm::errs() << "space: ");
  LLVM_DEBUG(isl_space_dump(space));
  islStmts[stmtId].domain = isl_basic_set_from_constraint_matrices(
      space, eqMat, ineqMat, isl_dim_set, isl_dim_div, isl_dim_param,
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
  isl_space *space = isl_space_alloc(ctx, cst.getNumSymbolVars(),
                                     cst.getNumDimVars() - vMap.getNumResults(),
                                     vMap.getNumResults());
  space = setupSpace(space, cst, memRefIdMap[memref]);

  isl_basic_map *bmap = isl_basic_map_from_constraint_matrices(
      space, eqMat, ineqMat, isl_dim_out, isl_dim_in, isl_dim_div,
      isl_dim_param, isl_dim_cst);
  if (isRead)
    islStmts[stmtId].readRelations.push_back(bmap);
  else
    islStmts[stmtId].writeRelations.push_back(bmap);

  ISL_DEBUG("Created relation: ", isl_basic_map_dump(bmap));

  return success();
}

void IslScop::initializeSymbolTable(mlir::func::FuncOp f,
                                    affine::FlatAffineValueConstraints *cst) {
  symbolTable.clear();

  // Setup the symbol table.
  for (unsigned i = 0; i < cst->getNumDimAndSymbolVars(); i++) {
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

namespace polymer {
class IslMLIRBuilder {
public:
  OpBuilder &b;
  IRMapping funcMapping;
  IslScop &scop;
  Location loc = b.getUnknownLoc();
  typedef llvm::MapVector<isl_id *, Value> IDToValueTy;
  IDToValueTy IDToValue{};

  Value createOp(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "Expression not of type isl_ast_expr_op");
    switch (isl_ast_expr_get_op_type(Expr)) {
    case isl_ast_op_error:
    case isl_ast_op_cond:
    case isl_ast_op_call:
    case isl_ast_op_member:
      llvm_unreachable("Unsupported isl ast expression");
    case isl_ast_op_access:
      return createOpAccess(Expr);
    case isl_ast_op_max:
    case isl_ast_op_min:
      return createOpNAry(Expr);
    case isl_ast_op_add:
    case isl_ast_op_sub:
    case isl_ast_op_mul:
    case isl_ast_op_div:
    case isl_ast_op_fdiv_q: // Round towards -infty
    case isl_ast_op_pdiv_q: // Dividend is non-negative
    case isl_ast_op_pdiv_r: // Dividend is non-negative
    case isl_ast_op_zdiv_r: // Result only compared against zero
      return createOpBin(Expr);
    case isl_ast_op_minus:
      return createOpUnary(Expr);
    case isl_ast_op_select:
      return createOpSelect(Expr);
    case isl_ast_op_and:
    case isl_ast_op_or:
      return createOpBoolean(Expr);
    case isl_ast_op_and_then:
    case isl_ast_op_or_else:
      return createOpBooleanConditional(Expr);
    case isl_ast_op_eq:
    case isl_ast_op_le:
    case isl_ast_op_lt:
    case isl_ast_op_ge:
    case isl_ast_op_gt:
      return createOpICmp(Expr);
    case isl_ast_op_address_of:
      return createOpAddressOf(Expr);
    }

    llvm_unreachable("Unsupported isl_ast_expr_op kind.");
  }

  Value createOpAddressOf(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }
  Value createOpUnary(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }
  Value createOpAccess(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }

  Value createMul(Value LHS, Value RHS, std::string Name = "") {
    return b.create<arith::MulIOp>(loc, LHS, RHS);
  }
  Value createSub(Value LHS, Value RHS, std::string Name = "") {
    return b.create<arith::SubIOp>(loc, LHS, RHS);
  }
  Value createAdd(Value LHS, Value RHS, std::string Name = "") {
    return b.create<arith::AddIOp>(loc, LHS, RHS);
  }

  Value createOpBin(__isl_take isl_ast_expr *Expr) {
    Value LHS, RHS, Res;
    Type MaxType;
    isl_ast_op_type OpType;

    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "isl ast expression not of type isl_ast_op");
    assert(isl_ast_expr_get_op_n_arg(Expr) == 2 &&
           "not a binary isl ast expression");

    OpType = isl_ast_expr_get_op_type(Expr);

    LHS = create(isl_ast_expr_get_op_arg(Expr, 0));
    RHS = create(isl_ast_expr_get_op_arg(Expr, 1));

    MaxType = convertToMaxWidth(LHS, RHS);

    switch (OpType) {
    default:
      llvm_unreachable("This is no binary isl ast expression");
    case isl_ast_op_add:
      Res = createAdd(LHS, RHS);
      break;
    case isl_ast_op_sub:
      Res = createSub(LHS, RHS);
      break;
    case isl_ast_op_mul:
      Res = createMul(LHS, RHS);
      break;
    case isl_ast_op_div:
      Res = b.create<arith::DivSIOp>(loc, LHS, RHS);
      break;
    case isl_ast_op_pdiv_q: // Dividend is non-negative
      Res = b.create<arith::DivUIOp>(loc, LHS, RHS);
      break;
    case isl_ast_op_fdiv_q: { // Round towards -infty
      // if (auto Const = dyn_cast<arith::ConstantIntOp>(RHS)) {
      //   auto &Val = Const.getValue();
      //   if (Val.isPowerOf2() && Val.isNonNegative()) {
      //     Res = b.create<arith::ShRSIOp>(loc, LHS, Val.ceilLogBase2());
      //     break;
      //   }
      // }

      // TODO: Review code and check that this calculation does not yield
      //       incorrect overflow in some edge cases.
      //
      // floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
      Value One = b.create<arith::ConstantIntOp>(loc, 1, MaxType);
      Value Zero = b.create<arith::ConstantIntOp>(loc, 0, MaxType);
      Value Sum1 = createSub(LHS, RHS, "pexp.fdiv_q.0");
      Value Sum2 = createAdd(Sum1, One, "pexp.fdiv_q.1");
      Value isNegative =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, LHS, Zero);
      Value Dividend = b.create<arith::SelectOp>(loc, isNegative, Sum2, LHS);
      Res = b.create<arith::DivSIOp>(loc, Dividend, RHS);
      break;
    }
    case isl_ast_op_pdiv_r: // Dividend is non-negative
      Res = b.create<arith::RemUIOp>(loc, LHS, RHS);
      break;
    case isl_ast_op_zdiv_r: // Result only compared against zero
      Res = b.create<arith::RemSIOp>(loc, LHS, RHS);
      break;
    }
    isl_ast_expr_free(Expr);
    return Res;
  }

  Value createOpNAry(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "isl ast expression not of type isl_ast_op");
    assert(isl_ast_expr_get_op_n_arg(Expr) >= 2 &&
           "We need at least two operands in an n-ary operation");

    std::function<Value(Value, Value)> Aggregate;
    switch (isl_ast_expr_get_op_type(Expr)) {
    default:
      llvm_unreachable("This is not a an n-ary isl ast expression");
    case isl_ast_op_max:
      Aggregate = [&](Value x, Value y) {
        return b.create<arith::MaxSIOp>(loc, x, y);
      };
      break;
    case isl_ast_op_min:
      Aggregate = [&](Value x, Value y) {
        return b.create<arith::MinSIOp>(loc, x, y);
      };
      break;
    }

    Value V = create(isl_ast_expr_get_op_arg(Expr, 0));

    for (int i = 1; i < isl_ast_expr_get_op_n_arg(Expr); ++i) {
      Value OpV = create(isl_ast_expr_get_op_arg(Expr, i));
      convertToMaxWidth(V, OpV);
      V = Aggregate(OpV, V);
    }

    isl_ast_expr_free(Expr);
    return V;
  }
  Value createOpSelect(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }
  Value createOpICmp(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "Expected an isl_ast_expr_op expression");

    Value LHS, RHS, Res;

    auto *Op0 = isl_ast_expr_get_op_arg(Expr, 0);
    auto *Op1 = isl_ast_expr_get_op_arg(Expr, 1);
    bool HasNonAddressOfOperand =
        isl_ast_expr_get_type(Op0) != isl_ast_expr_op ||
        isl_ast_expr_get_type(Op1) != isl_ast_expr_op ||
        isl_ast_expr_get_op_type(Op0) != isl_ast_op_address_of ||
        isl_ast_expr_get_op_type(Op1) != isl_ast_op_address_of;

    // TODO not sure if we would ever get pointers here
    bool UseUnsignedCmp = !HasNonAddressOfOperand;

    LHS = create(Op0);
    RHS = create(Op1);

    if (LHS.getType() != RHS.getType()) {
      convertToMaxWidth(LHS, RHS);
    }

    isl_ast_op_type OpType = isl_ast_expr_get_op_type(Expr);
    assert(OpType >= isl_ast_op_eq && OpType <= isl_ast_op_gt &&
           "Unsupported ICmp isl ast expression");
    static_assert(isl_ast_op_eq + 4 == isl_ast_op_gt,
                  "Isl ast op type interface changed");

    arith::CmpIPredicate Predicates[5][2] = {
        {arith::CmpIPredicate::eq, arith::CmpIPredicate::eq},
        {arith::CmpIPredicate::sle, arith::CmpIPredicate::ule},
        {arith::CmpIPredicate::slt, arith::CmpIPredicate::ult},
        {arith::CmpIPredicate::sge, arith::CmpIPredicate::uge},
        {arith::CmpIPredicate::sgt, arith::CmpIPredicate::ugt},
    };

    Res = b.create<arith::CmpIOp>(
        loc, Predicates[OpType - isl_ast_op_eq][UseUnsignedCmp], LHS, RHS);

    isl_ast_expr_free(Expr);
    return Res;
  }

  Value createOpBoolean(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }
  Value createOpBooleanConditional(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }
  Value createId(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_id &&
           "Expression not of type isl_ast_expr_ident");

    isl_id *Id;
    Value V;

    Id = isl_ast_expr_get_id(Expr);

    assert(IDToValue.count(Id) && "Identifier not found");

    V = IDToValue[Id];
    assert(V && "Unknown parameter id found");

    isl_id_free(Id);
    isl_ast_expr_free(Expr);

    return V;
  }
  IntegerType getType(__isl_keep isl_ast_expr *Expr) {
    // XXX: We assume i64 is large enough. This is often true, but in general
    //      incorrect. Also, on 32bit architectures, it would be beneficial to
    //      use a smaller type. We can and should directly derive this
    //      information during code generation.
    return IntegerType::get(b.getContext(), 64);
  }
  Value createInt(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_int &&
           "Expression not of type isl_ast_expr_int");
    isl_val *Val;
    Value V;
    APInt APValue;
    IntegerType T;

    Val = isl_ast_expr_get_val(Expr);
    APValue = polly::APIntFromVal(Val);

    auto BitWidth = APValue.getBitWidth();
    if (BitWidth <= 64)
      T = getType(Expr);
    else
      T = b.getIntegerType(BitWidth);

    APValue = APValue.sext(T.getWidth());
    V = b.create<arith::ConstantIntOp>(loc, APValue.getSExtValue(), T);

    isl_ast_expr_free(Expr);
    return V;
  }
  Value create(__isl_take isl_ast_expr *Expr) {
    switch (isl_ast_expr_get_type(Expr)) {
    case isl_ast_expr_error:
      llvm_unreachable("Code generation error");
    case isl_ast_expr_op:
      return createOp(Expr);
    case isl_ast_expr_id:
      return createId(Expr);
    case isl_ast_expr_int:
      return createInt(Expr);
    }
    llvm_unreachable("Unexpected enum value");
  }

  void createUser(__isl_keep isl_ast_node *User) {
    ISL_DEBUG("Building User:\n", isl_ast_node_dump(User));

    isl_ast_expr *Expr = isl_ast_node_user_get_expr(User);
    if (isl_ast_expr_get_op_type(Expr) != isl_ast_op_call) {
      llvm_unreachable("unexpected op type");
    }
    isl_ast_expr *CalleeExpr = isl_ast_expr_get_op_arg(Expr, 0);
    isl_id *Id = isl_ast_expr_get_id(CalleeExpr);
    const char *CalleeName = isl_id_get_name(Id);

    SmallVector<Value> ivs;
    for (int i = 0; i < isl_ast_expr_get_op_n_arg(Expr) - 1; ++i) {
      isl_ast_expr *SubExpr = isl_ast_expr_get_op_arg(Expr, i + 1);
      Value V = create(SubExpr);
      ivs.push_back(V);
    }

    ScopStmt &stmt = scop.scopStmtMap.at(std::string(CalleeName));
    func::CallOp origCaller = stmt.getCaller();
    SmallVector<Value> args;
    for (Value origArg : origCaller.getArgOperands()) {
      auto ba = origArg.dyn_cast<BlockArgument>();
      if (ba) {
        Operation *owner = ba.getOwner()->getParentOp();
        if (isa<func::FuncOp>(owner)) {
          args.push_back(funcMapping.lookup(ba));
        } else if (isa<affine::AffineForOp, affine::AffineParallelOp>(owner)) {
          SmallVector<Operation *> enclosing;
          stmt.getEnclosingOps(enclosing);
          unsigned ivId = 0;
          for (auto *op : enclosing) {
            if (isa<affine::AffineIfOp>(op)) {
              continue;
            } else if (isa<affine::AffineForOp, affine::AffineParallelOp>(op)) {
              if (owner == op)
                break;
              ivId++;
            } else {
              llvm_unreachable("non-affine enclosing op");
            }
          }
          Value arg = ivs[ivId];
          if (arg.getType() != origArg.getType()) {
            // This can only happen to index types as we may have replaced them
            // with the target system width
            assert(origArg.getType().isa<IndexType>());
            arg = b.create<arith::IndexCastOp>(loc, origArg.getType(), arg);
          }
          args.push_back(arg);
        } else {
          llvm_unreachable("unexpected");
        }
      } else {
        Operation *op = origArg.getDefiningOp();
        assert(op);
        if (auto alloca = dyn_cast<memref::AllocaOp>(op)) {
          assert(alloca->getAttr("scop.scratchpad"));
          auto newAlloca = funcMapping.lookup(op)->getResult(0);
          args.push_back(newAlloca);
        } else {
          assert("unexpected");
        }
      }
    }

    b.create<func::CallOp>(loc, StringRef(CalleeName), TypeRange(), args);

    isl_ast_expr_free(Expr);
    isl_ast_node_free(User);
    isl_ast_expr_free(CalleeExpr);
    isl_id_free(Id);
  }

  void createMark(__isl_take isl_ast_node *Node) {
    ISL_DEBUG("Building Mark:\n", isl_ast_node_dump(Node));

    auto *Id = isl_ast_node_mark_get_id(Node);
    auto Child = isl_ast_node_mark_get_node(Node);
    isl_ast_node_free(Node);

    if (isParallelLoopMark(Id)) {
      assert(isl_ast_node_get_type(Child) == isl_ast_node_for);
      createFor<scf::ParallelOp>(Child);
    } else {
      llvm_unreachable("Unknown mark");
    }

    isl_id_free(Id);
  }

  void createIf(__isl_take isl_ast_node *If) {
    ISL_DEBUG("Building If:\n", isl_ast_node_dump(If));
    isl_ast_expr *Cond = isl_ast_node_if_get_cond(If);

    Value Predicate = create(Cond);

    bool hasElse = isl_ast_node_if_has_else(If);
    auto ifOp = b.create<scf::IfOp>(loc, TypeRange(), Predicate,
                                    /*addThenBlock=*/true,
                                    /*addElseBlock=*/hasElse);

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    b.create<scf::YieldOp>(loc);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    create(isl_ast_node_if_get_then(If));

    if (hasElse) {
      b.setInsertionPointToStart(&ifOp.getElseRegion().front());
      b.create<scf::YieldOp>(loc);
      b.setInsertionPointToStart(&ifOp.getElseRegion().front());
      create(isl_ast_node_if_get_else(If));
    }

    isl_ast_node_free(If);
  }

  void createBlock(__isl_keep isl_ast_node *Block) {
    ISL_DEBUG("Building Block:\n", isl_ast_node_dump(Block));
    isl_ast_node_list *List = isl_ast_node_block_get_children(Block);

    for (int i = 0; i < isl_ast_node_list_n_ast_node(List); ++i)
      create(isl_ast_node_list_get_ast_node(List, i));

    isl_ast_node_free(Block);
    isl_ast_node_list_free(List);
  }

  isl::ast_expr getUpperBound(isl::ast_node_for For,
                              arith::CmpIPredicate &Predicate) {
    isl::ast_expr Cond = For.cond();
    isl::ast_expr Iterator = For.iterator();
    // The isl code generation can generate arbitrary expressions to check if
    // the upper bound of a loop is reached, but it provides an option to
    // enforce 'atomic' upper bounds. An 'atomic upper bound is always of the
    // form iv <= expr, where expr is an (arbitrary) expression not containing
    // iv.
    //
    // We currently only support atomic upper bounds for ease of codegen
    //
    // This is needed for parallel loops but maybe we can weaken the requirement
    // for sequential loops if needed
    assert(isl_ast_expr_get_type(Cond.get()) == isl_ast_expr_op &&
           "conditional expression is not an atomic upper bound");

    isl_ast_op_type OpType = isl_ast_expr_get_op_type(Cond.get());

    switch (OpType) {
    case isl_ast_op_le:
      Predicate = arith::CmpIPredicate::sle;
      break;
    case isl_ast_op_lt:
      Predicate = arith::CmpIPredicate::slt;
      break;
    default:
      llvm_unreachable("Unexpected comparison type in loop condition");
    }

    isl::ast_expr Arg0 = Cond.get_op_arg(0);

    assert(isl_ast_expr_get_type(Arg0.get()) == isl_ast_expr_id &&
           "conditional expression is not an atomic upper bound");

    isl::id UBID = Arg0.get_id();

    assert(isl_ast_expr_get_type(Iterator.get()) == isl_ast_expr_id &&
           "Could not get the iterator");

    isl::id IteratorID = Iterator.get_id();

    assert(UBID.get() == IteratorID.get() &&
           "conditional expression is not an atomic upper bound");

    return Cond.get_op_arg(1);
  }

  template <class... Ts> void convertToIndex(Ts &&...args) {
    SmallVector<Value *> Args({&args...});
    for (unsigned I = 0; I < Args.size(); I++) {
      Type Ty = Args[I]->getType();
      if (!Ty.isa<IndexType>()) {
        *Args[I] =
            b.create<arith::IndexCastOp>(loc, b.getIndexType(), *Args[I]);
      }
    }
  }

  template <class... Ts> Type convertToMaxWidth(Ts &&...args) {
    SmallVector<Value *> Args({&args...});
    if (llvm::all_of(Args,
                     [&](Value *V) { return V->getType().isa<IndexType>(); }))
      return Args[0]->getType();
    Type MaxTypeI = Args[0]->getType();
    IntegerType MaxType;
    if (MaxTypeI.isa<IndexType>())
      // TODO This is temporary and we should get the target system index here
      MaxType = b.getI64Type();
    else
      MaxType = MaxTypeI.cast<IntegerType>();
    unsigned MaxWidth = MaxType.getWidth();
    for (unsigned I = 0; I < Args.size(); I++) {
      Type Ty = Args[I]->getType();
      if (Ty.isa<IndexType>())
        // TODO This is temporary and we should get the target system index here
        Ty = b.getI64Type();
      if (Ty.cast<IntegerType>().getWidth() > MaxWidth) {
        MaxType = Ty.cast<IntegerType>();
        MaxWidth = MaxType.getWidth();
      }
    }
    for (unsigned I = 0; I < Args.size(); I++) {
      Type Ty = Args[I]->getType();
      if (Ty.isa<IndexType>()) {
        *Args[I] = b.create<arith::IndexCastOp>(loc, MaxType, *Args[I]);
      } else if (Ty != MaxType) {
        *Args[I] = b.create<arith::ExtSIOp>(loc, MaxType, *Args[I]);
      }
    }
    return MaxType;
  }

  template <typename ForOpTy = scf::ForOp>
  void createFor(__isl_take isl_ast_node *For) {
    ISL_DEBUG("Building For:\n", isl_ast_node_dump(For));
    isl_ast_node *Body = isl_ast_node_for_get_body(For);
    isl_ast_expr *Init = isl_ast_node_for_get_init(For);
    isl_ast_expr *Inc = isl_ast_node_for_get_inc(For);
    isl_ast_expr *Iterator = isl_ast_node_for_get_iterator(For);
    isl_id *IteratorID = isl_ast_expr_get_id(Iterator);
    arith::CmpIPredicate Predicate;
    isl_ast_expr *UB =
        getUpperBound(isl::manage_copy(For).as<isl::ast_node_for>(), Predicate)
            .release();

    Value ValueLB = create(Init);
    Value ValueUB = create(UB);
    Value ValueInc = create(Inc);
    convertToMaxWidth(ValueLB, ValueUB, ValueInc);

    if (Predicate == arith::CmpIPredicate::sle)
      ValueUB = b.create<arith::AddIOp>(
          loc, ValueUB,
          b.create<arith::ConstantIntOp>(loc, 1, ValueUB.getType()));

    // scf::ParallelOp only supports index as bounds
    if constexpr (std::is_same<ForOpTy, scf::ParallelOp>::value) {
      convertToIndex(ValueLB, ValueUB, ValueInc);
    }

    auto forOp = b.create<ForOpTy>(loc, ValueLB, ValueUB, ValueInc);

    if constexpr (std::is_same<ForOpTy, scf::ForOp>::value) {
      IDToValue[IteratorID] = forOp.getInductionVar();
    } else if constexpr (std::is_same<ForOpTy, scf::ParallelOp>::value) {
      IDToValue[IteratorID] = forOp.getInductionVars()[0];
    } else {
      // static_assert(0);
      llvm_unreachable("?");
    }

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(forOp.getBody());
    create(Body);

    isl_ast_expr_free(Iterator);
    isl_id_free(IteratorID);
    isl_ast_node_free(For);
  }

  void create(__isl_take isl_ast_node *Node) {
    switch (isl_ast_node_get_type(Node)) {
    case isl_ast_node_error:
      llvm_unreachable("code generation error");
    case isl_ast_node_mark:
      createMark(Node);
      return;
    case isl_ast_node_for:
      createFor(Node);
      return;
    case isl_ast_node_if:
      createIf(Node);
      return;
    case isl_ast_node_user:
      createUser(Node);
      return;
    case isl_ast_node_block:
      createBlock(Node);
      return;
    }

    llvm_unreachable("Unknown isl_ast_node type");
  }

  void mapParams(__isl_take isl_union_set *domain) {
    isl_space *space = isl_union_set_get_space(domain);

    int nparams = isl_space_dim(space, isl_dim_param);
    for (int i = 0; i < nparams; i++) {
      isl_id *Id = isl_space_get_dim_id(space, isl_dim_param, i);
      const char *paramName = isl_id_get_name(Id);
      Value V = scop.symbolTable[paramName];
      IDToValue[Id] = funcMapping.lookup(V);
      isl_id_free(Id);
    }
    isl_space_free(space);
    isl_union_set_free(domain);
  }
};
} // namespace polymer

func::FuncOp IslScop::applySchedule(isl_schedule *newSchedule,
                                    func::FuncOp originalFunc) {
  IRMapping oldToNewMapping;
  OpBuilder moduleBuilder(originalFunc);
  func::FuncOp f =
      cast<func::FuncOp>(moduleBuilder.clone(*originalFunc, oldToNewMapping));

  assert(f.getFunctionBody().getBlocks().size() == 1);

  // Cleanup body. Leave only scratchpad allocations and tarminator.
  // TODO is there anything else we need to keep?
  Operation *op = &f.getFunctionBody().front().front();
  while (true) {
    if (auto alloca = dyn_cast<memref::AllocaOp>(op)) {
      assert(alloca->getAttr("scop.scratchpad"));
      op = op->getNextNode();
      continue;
    }
    auto next = op->getNextNode();
    if (!next)
      break;
    // TODO should check it is a stmt call and not some random one
    assert(isa<affine::AffineDialect>(op->getDialect()) ||
           isa<func::CallOp>(op));
    op->erase();
    op = next;
  }

  // TODO we also need to allocate new arrays which may have been introduced,
  // see polly::NodeBuilder::allocateNewArrays, buildAliasScopes

  OpBuilder b(f.getFunctionBody().front().getTerminator());

  LLVM_DEBUG({
    llvm::dbgs() << "Applying new schedule to scop:\n";
    isl_schedule_dump(newSchedule);
  });
  isl_union_set *domain = isl_schedule_get_domain(newSchedule);
  isl_ast_build *build = isl_ast_build_alloc(ctx);
  IslMLIRBuilder bc = {b, oldToNewMapping, *this};
  isl_ast_node *node =
      isl_ast_build_node_from_schedule(build, isl_schedule_copy(newSchedule));

  bc.mapParams(domain);
  bc.create(node);
  LLVM_DEBUG(llvm::dbgs() << f << "\n");

  isl_ast_build_free(build);
  isl_schedule_free(newSchedule);

  return f;
}

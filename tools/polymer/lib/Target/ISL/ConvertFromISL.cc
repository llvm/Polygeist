//===- ConvertFromOpenScop.h ------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#include "cloog/cloog.h"
#include "osl/osl.h"
#include "pluto/internal/pluto.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"
extern "C" {
#include "pluto/internal/ast_transform.h"
}

#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Support/Utils.h"
#include "polymer/Target/ISL.h"
#include "polymer/Target/OpenScop.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

using namespace polymer;
using namespace mlir;
using namespace mlir::func;

typedef llvm::StringMap<mlir::Operation *> StmtOpMap;
typedef llvm::StringMap<mlir::Value> NameValueMap;
typedef llvm::StringMap<std::string> IterScatNameMap;

namespace {
typedef llvm::StringMap<mlir::Value> SymbolTable;

/// Build AffineExpr from a clast_expr.
/// TODO: manage the priviledge.
class AffineExprBuilder {
public:
  AffineExprBuilder(MLIRContext *context, PolymerSymbolTable *symTable,
                    SymbolTable *symbolTable, IslScop *scop,
                    CloogOptions *options)
      : b(context), context(context), scop(scop), symTable(symTable),
        symbolTable(symbolTable), options(options) {
    reset();
  }

  LogicalResult process(clast_expr *expr,
                        llvm::SmallVectorImpl<AffineExpr> &affExprs);

  void reset();

  LogicalResult process(clast_name *expr,
                        llvm::SmallVectorImpl<AffineExpr> &affExprs);
  LogicalResult process(clast_term *expr,
                        llvm::SmallVectorImpl<AffineExpr> &affExprs);
  LogicalResult process(clast_binary *expr,
                        llvm::SmallVectorImpl<AffineExpr> &affExprs);
  LogicalResult process(clast_reduction *expr,
                        llvm::SmallVectorImpl<AffineExpr> &affExprs);

  LogicalResult
  processSumReduction(clast_reduction *expr,
                      llvm::SmallVectorImpl<AffineExpr> &affExprs);
  LogicalResult
  processMinOrMaxReduction(clast_reduction *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs);

  /// OpBuilder used to create AffineExpr.
  OpBuilder b;
  /// The MLIR context
  MLIRContext *context;
  /// The IslScop of the whole program.
  IslScop *scop;
  /// TODO: keep only one of them
  PolymerSymbolTable *symTable;
  SymbolTable *symbolTable;
  ///
  CloogOptions *options;

  llvm::StringMap<unsigned> symbolNames;
  llvm::StringMap<unsigned> dimNames;

  llvm::DenseMap<Value, std::string> valueMap;
};
} // namespace

void AffineExprBuilder::reset() {
  symbolNames.clear();
  dimNames.clear();
}

/// Get the int64_t representation of a cloog_int_t.
static LogicalResult getI64(cloog_int_t num, int64_t *res) {
  // TODO: is there a better way to work around this file-based interface?
  // First, we read the cloog integer into a char buffer.
  char buf[100]; // Should be sufficient for int64_t in string.
  FILE *bufFile = fmemopen(reinterpret_cast<void *>(buf), 32, "w");
  cloog_int_print(bufFile, num);
  fclose(bufFile); // Should close the file or the buf won't be flushed.

  // Then we parse the string as int64_t.
  *res = strtoll(buf, NULL, 10);

  // TODO: error handling.
  return success();
}

LogicalResult
AffineExprBuilder::process(clast_expr *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs) {

  switch (expr->type) {
  case clast_expr_name:
    if (failed(process(reinterpret_cast<clast_name *>(expr), affExprs)))
      return failure();
    break;
  case clast_expr_term:
    if (failed(process(reinterpret_cast<clast_term *>(expr), affExprs)))
      return failure();
    break;
  case clast_expr_bin:
    if (failed(process(reinterpret_cast<clast_binary *>(expr), affExprs)))
      return failure();
    break;
  case clast_expr_red:
    if (failed(process(reinterpret_cast<clast_reduction *>(expr), affExprs)))
      return failure();
    break;
  }
  return success();
}

/// Find the name in the scop to determine the type (dim or symbol). The
/// position is decided by the size of dimNames/symbolNames.
/// TODO: handle the dim case.
LogicalResult
AffineExprBuilder::process(clast_name *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  if (scop->isSymbol(expr->name)) {
    if (symbolNames.find(expr->name) != symbolNames.end())
      affExprs.push_back(b.getAffineSymbolExpr(symbolNames[expr->name]));
    else {
      affExprs.push_back(b.getAffineSymbolExpr(symbolNames.size()));
      size_t numSymbols = symbolNames.size();
      symbolNames[expr->name] = numSymbols;

      Value v = symbolTable->lookup(expr->name);
      valueMap[v] = expr->name;
    }
  } else if (mlir::Value iv = symbolTable->lookup(expr->name)) {
    if (dimNames.find(expr->name) != dimNames.end())
      affExprs.push_back(b.getAffineDimExpr(dimNames[expr->name]));
    else {
      affExprs.push_back(b.getAffineDimExpr(dimNames.size()));
      size_t numDims = dimNames.size();
      dimNames[expr->name] = numDims;
      valueMap[iv] = expr->name;
    }
  } else {
    return failure();
  }

  return success();
}

LogicalResult
AffineExprBuilder::process(clast_term *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  // First get the I64 representation of a cloog int.
  int64_t constant;
  if (failed(getI64(expr->val, &constant)))
    return failure();

  // Next create a constant AffineExpr.
  AffineExpr affExpr = b.getAffineConstantExpr(constant);

  // If var is not NULL, it means this term is var * val. We should create the
  // expr that denotes var and multiplies it with the AffineExpr for val.
  if (expr->var) {
    SmallVector<AffineExpr, 1> varAffExprs;
    if (failed(process(expr->var, varAffExprs)))
      return failure();
    assert(varAffExprs.size() == 1 &&
           "There should be a single expression that stands for the var expr.");

    affExpr = affExpr * varAffExprs[0];
  }

  affExprs.push_back(affExpr);

  return success();
}

LogicalResult
AffineExprBuilder::process(clast_binary *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  // Handle the LHS expression.
  SmallVector<AffineExpr, 1> lhsAffExprs;
  if (failed(process(expr->LHS, lhsAffExprs)))
    return failure();
  assert(lhsAffExprs.size() == 1 &&
         "There should be a single LHS affine expr.");

  // Handle the RHS expression, which is an integer constant.
  int64_t rhs;
  if (failed(getI64(expr->RHS, &rhs)))
    return failure();
  AffineExpr rhsAffExpr = b.getAffineConstantExpr(rhs);

  AffineExpr affExpr;

  switch (expr->type) {
  case clast_bin_fdiv:
    affExpr = lhsAffExprs[0].floorDiv(rhsAffExpr);
    break;
  case clast_bin_cdiv:
  case clast_bin_div: // TODO: check out this
    affExpr = lhsAffExprs[0].ceilDiv(rhsAffExpr);
    break;
  case clast_bin_mod:
    affExpr = lhsAffExprs[0] % rhsAffExpr;
    break;
  }

  affExprs.push_back(affExpr);

  return success();
}

LogicalResult
AffineExprBuilder::process(clast_reduction *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  if (expr->n == 1) {
    if (failed(process(expr->elts[0], affExprs)))
      return failure();
    return success();
  }

  switch (expr->type) {
  case clast_red_sum:
    if (failed(processSumReduction(expr, affExprs)))
      return failure();
    break;
  case clast_red_min:
  case clast_red_max:
    if (failed(processMinOrMaxReduction(expr, affExprs)))
      return failure();
    break;
  }

  return success();
}

LogicalResult AffineExprBuilder::processSumReduction(
    clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  assert(expr->n >= 1 && "Number of reduction elements should be non-zero.");
  assert(expr->elts[0]->type == clast_expr_term &&
         "The first element should be a term.");

  // Build the reduction expression.
  unsigned numAffExprs = affExprs.size();
  if (failed(process(expr->elts[0], affExprs)))
    return failure();
  assert(numAffExprs + 1 == affExprs.size() &&
         "A single affine expr should be appended after processing an expr in "
         "reduction.");

  for (int i = 1; i < expr->n; ++i) {
    assert(expr->elts[i]->type == clast_expr_term &&
           "Each element in the reduction list should be a term.");

    clast_term *term = reinterpret_cast<clast_term *>(expr->elts[i]);
    SmallVector<AffineExpr, 1> currExprs;
    if (failed(process(term, currExprs)))
      return failure();
    assert(currExprs.size() == 1 &&
           "There should be one affine expr corresponds to a single term.");

    // TODO: deal with negative terms.
    // numAffExprs is the index for the current affExpr, i.e., the newly
    // appended one from processing expr->elts[0].
    affExprs[numAffExprs] = affExprs[numAffExprs] + currExprs[0];
  }

  return success();
}

LogicalResult AffineExprBuilder::processMinOrMaxReduction(
    clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  if (failed(process(expr->elts[0], affExprs)))
    return failure();

  for (int i = 1; i < expr->n; i++) {
    if (failed(process(expr->elts[i], affExprs)))
      return failure();
  }

  return success();
}
/// Builds the mapping from the iterator names in a statement to their
/// corresponding names in <scatnames>, based on the matrix provided by the
/// scattering relation.
static void buildIterToScatNameMap(IterScatNameMap &iterToScatName,
                                   osl_statement_p stmt,
                                   osl_generic_p scatnames) {
  // Get the body from the statement.
  osl_body_p body = osl_statement_get_body(stmt);
  assert(body != nullptr && "The body of the statement should not be NULL.");
  assert(body->expression != nullptr &&
         "The body expression should not be NULL.");
  assert(body->iterators != nullptr &&
         "The body iterators should not be NULL.");

  // Get iterator names.
  unsigned numIterNames = osl_strings_size(body->iterators);
  llvm::SmallVector<std::string, 8> iterNames(numIterNames);
  for (unsigned i = 0; i < numIterNames; i++)
    iterNames[i] = body->iterators->string[i];

  // Split the scatnames into a list of strings.
  osl_strings_p scatNamesData =
      reinterpret_cast<osl_scatnames_p>(scatnames->data)->names;
  unsigned numScatNames = osl_strings_size(scatNamesData);

  llvm::SmallVector<std::string, 8> scatNames(numScatNames);
  for (unsigned i = 0; i < numScatNames; i++)
    scatNames[i] = scatNamesData->string[i];

  // Get the scattering relation.
  osl_relation_p scats = stmt->scattering;
  assert(scats != nullptr && "scattering in the statement should not be NULL.");
  assert(scats->nb_input_dims == static_cast<int>(iterNames.size()) &&
         "# input dims should equal to # iter names.");
  assert(scats->nb_output_dims <= static_cast<int>(scatNames.size()) &&
         "# output dims should be less than or equal to # scat names.");

  // Build the mapping.
  for (int i = 0; i < scats->nb_output_dims; i++)
    for (int j = 0; j < scats->nb_input_dims; j++)
      if (scats->m[i][j + scats->nb_output_dims + 1].dp)
        iterToScatName[iterNames[j]] = scatNames[i];
}

namespace {

/// Build mapping between the iter names in the original code to the scatname in
/// the OpenScop.
class IterScatNameMapper {
public:
  IterScatNameMapper(IslScop *scop) : scop(scop) {}

  void visitStmtList(clast_stmt *s);

  IterScatNameMap getIterScatNameMap() { return iterScatNameMap; };

private:
  void visit(clast_for *forStmt);
  void visit(clast_guard *guardStmt);
  void visit(clast_user_stmt *userStmt);

  IslScop *scop;

  IterScatNameMap iterScatNameMap;
};

} // namespace

void IterScatNameMapper::visitStmtList(clast_stmt *s) {
  for (; s; s = s->next) {
    if (CLAST_STMT_IS_A(s, stmt_user)) {
      visit(reinterpret_cast<clast_user_stmt *>(s));
    } else if (CLAST_STMT_IS_A(s, stmt_for)) {
      visit(reinterpret_cast<clast_for *>(s));
    } else if (CLAST_STMT_IS_A(s, stmt_guard)) {
      visit(reinterpret_cast<clast_guard *>(s));
    }
  }
}

void IterScatNameMapper::visit(clast_for *forStmt) {
  visitStmtList(forStmt->body);
}
void IterScatNameMapper::visit(clast_guard *guardStmt) {
  visitStmtList(guardStmt->then);
}

void IterScatNameMapper::visit(clast_user_stmt *userStmt) {
  // TODO ISL
}

namespace {
/// Import MLIR code from the clast AST.
class Importer {
public:
  Importer(MLIRContext *context, ModuleOp module, PolymerSymbolTable *symTable,
           IslScop *scop, CloogOptions *options);

  LogicalResult processStmtList(clast_stmt *s);

  mlir::Operation *getFunc() { return func; }

private:
  void initializeSymbolTable();
  void initializeFuncOpInterface();
  void initializeSymbol(mlir::Value val);

  LogicalResult processStmt(clast_root *rootStmt);
  LogicalResult processStmt(clast_for *forStmt);
  LogicalResult processStmt(clast_guard *guardStmt);
  LogicalResult processStmt(clast_user_stmt *userStmt);
  LogicalResult processStmt(clast_assignment *ass);

  std::string getSourceFuncName() const;
  mlir::func::FuncOp getSourceFuncOp();

  LogicalResult getAffineLoopBound(clast_expr *expr,
                                   llvm::SmallVectorImpl<mlir::Value> &operands,
                                   AffineMap &affMap, bool isUpper = false);
  void
  getAffineExprForLoopIterator(clast_stmt *subst,
                               llvm::SmallVectorImpl<mlir::Value> &operands,
                               AffineMap &affMap);
  void getInductionVars(clast_user_stmt *userStmt, osl_body_p body,
                        SmallVectorImpl<mlir::Value> &inductionVars);

  LogicalResult parseUserStmtBody(llvm::StringRef body, std::string &calleeName,
                                  llvm::SmallVectorImpl<std::string> &args);

  bool isMemrefArg(llvm::StringRef argName);

  /// Functions are always inserted before the module terminator.
  Block::iterator getFuncInsertPt() {
    return std::prev(module.getBody()->end());
  }
  /// A helper to create a callee.
  void createCalleeAndCallerArgs(llvm::StringRef calleeName,
                                 llvm::ArrayRef<std::string> args,
                                 mlir::func::FuncOp &callee,
                                 SmallVectorImpl<mlir::Value> &callerArgs);

  /// Number of internal functions created.
  int64_t numInternalFunctions = 0;

  /// The current builder, pointing at where the next Instruction should be
  /// generated.
  OpBuilder b;
  /// The current context.
  MLIRContext *context;
  /// The current module being created.
  ModuleOp module;
  /// The main function.
  FuncOp func;
  /// The OpenScop object pointer.
  IslScop *scop;
  /// The symbol table for labels in the OpenScop input (to be deprecated).
  PolymerSymbolTable *symTable;
  /// The symbol table that will be built on the fly.
  SymbolTable symbolTable;

  /// Map from symbol names to block arguments.
  llvm::DenseMap<llvm::StringRef, BlockArgument> symNameToArg;
  /// Map from callee names to callee operation.
  llvm::StringMap<Operation *> calleeMap;

  // Map from an not yet initialized symbol to the Values that depend on it.
  llvm::StringMap<llvm::SetVector<mlir::Value>> symbolToDeps;
  // Map from a value to all the symbols it depends on.
  llvm::DenseMap<mlir::Value, llvm::SetVector<llvm::StringRef>>
      valueToDepSymbols;

  IterScatNameMap iterScatNameMap;

  llvm::StringMap<clast_stmt *> lhsToAss;

  CloogOptions *options;
};
} // namespace

Importer::Importer(MLIRContext *context, ModuleOp module,
                   PolymerSymbolTable *symTable, IslScop *scop,
                   CloogOptions *options)
    : b(context), context(context), module(module), scop(scop),
      symTable(symTable), options(options) {
  b.setInsertionPointToStart(module.getBody());
}

mlir::func::FuncOp Importer::getSourceFuncOp() {
  std::string sourceFuncName = getSourceFuncName();
  mlir::Operation *sourceFuncOp = module.lookupSymbol(sourceFuncName);

  assert(sourceFuncOp != nullptr &&
         "sourceFuncName cannot be found in the module");
  assert(isa<mlir::func::FuncOp>(sourceFuncOp) &&
         "Found sourceFuncOp should be of type mlir::func::FuncOp.");
  return cast<mlir::func::FuncOp>(sourceFuncOp);
}

/// If there is anything in the comment, we will use it as a function name.
/// Otherwise, we return an empty string.
std::string Importer::getSourceFuncName() const {
  // TODO ISL
}

bool Importer::isMemrefArg(llvm::StringRef argName) {
  // TODO: should find a better way to do this, e.g., using the old symbol
  // table.
  return argName.size() >= 2 && argName[0] == 'A';
}

LogicalResult Importer::processStmtList(clast_stmt *s) {
  for (; s; s = s->next) {
    if (CLAST_STMT_IS_A(s, stmt_root)) {
      if (failed(processStmt(reinterpret_cast<clast_root *>(s))))
        return failure();
    } else if (CLAST_STMT_IS_A(s, stmt_ass)) {
      if (failed(processStmt(reinterpret_cast<clast_assignment *>(s))))
        return failure();
    } else if (CLAST_STMT_IS_A(s, stmt_user)) {
      if (failed(processStmt(reinterpret_cast<clast_user_stmt *>(s))))
        return failure();
    } else if (CLAST_STMT_IS_A(s, stmt_for)) {
      if (failed(processStmt(reinterpret_cast<clast_for *>(s))))
        return failure();
    } else if (CLAST_STMT_IS_A(s, stmt_guard)) {
      if (failed(processStmt(reinterpret_cast<clast_guard *>(s))))
        return failure();
    } else {
      assert(false && "clast_stmt type not supported");
    }
  }

  return success();
}

void Importer::initializeFuncOpInterface() {
  IslScop::ValueTable *oslValueTable = scop->getValueTable();

  /// First collect the source FuncOp in the original MLIR code.
  mlir::func::FuncOp sourceFuncOp = getSourceFuncOp();

  // OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(module.getBody(), getFuncInsertPt());

  // The default function name is main.
  std::string funcName("main");
  // If the comment is provided, we will use it as the function name.
  // TODO: make sure it is safe.
  std::string sourceFuncName = getSourceFuncName();
  if (!sourceFuncName.empty()) {
    funcName = std::string(formatv("{0}_opt", sourceFuncName));
  }
  // Create the function interface.
  func = b.create<FuncOp>(sourceFuncOp.getLoc(), funcName,
                          sourceFuncOp.getFunctionType());

  // Initialize the symbol table for these entryBlock arguments
  auto &entryBlock = *func.addEntryBlock();
  b.setInsertionPointToStart(&entryBlock);
  b.create<mlir::func::ReturnOp>(UnknownLoc::get(context));

  b.setInsertionPointToStart(&entryBlock);
  for (unsigned i = 0; i < entryBlock.getNumArguments(); i++) {
    std::string argSymbol = oslValueTable->lookup(sourceFuncOp.getArgument(i));

    mlir::Value arg = entryBlock.getArgument(i);
    // If the source type is not index, cast it to index then.
    if (scop->isParameterSymbol(argSymbol) &&
        arg.getType() != b.getIndexType()) {
      mlir::Operation *op = b.create<mlir::arith::IndexCastOp>(
          sourceFuncOp.getLoc(), b.getIndexType(), arg);
      symbolTable[argSymbol] = op->getResult(0);
    } else {
      symbolTable[argSymbol] = arg;
    }
  }
}

/// Translate the root statement as a function. The name of the function is by
/// default "main".
LogicalResult Importer::processStmt(clast_root *rootStmt) {
  // Create the function.
  initializeFuncOpInterface();
  // Initialize several values before start.
  initializeSymbolTable();

  return success();
}

/// Initialize the value in the symbol table.
void Importer::initializeSymbol(mlir::Value val) {
  assert(val != nullptr);
  IslScop::ValueTable *oslValueTable = scop->getValueTable();

  auto &entryBlock = *func.getBody().begin();

  OpBuilder::InsertionGuard guard(b);

  std::string symbol = oslValueTable->lookup(val);
  assert(!symbol.empty() && "val to initialize should have a corresponding "
                            "symbol in the original code.");

  /// Symbols that are the block arguments won't be taken care of at this stage.
  /// initializeFuncOpInterface() should already have done that.
  if (mlir::BlockArgument arg = val.dyn_cast<mlir::BlockArgument>())
    return;

  // This defOp should be cloned to the target function, while its operands
  // may be symbols that are not yet initialized (e.g., IVs in loops not
  // constructed). We should place them into the symbolToDeps map.
  mlir::Operation *defOp = val.getDefiningOp();
  if (isa<memref::AllocaOp>(defOp) && defOp->getNumOperands() == 0) {
    b.setInsertionPointToStart(&entryBlock);
    symbolTable[symbol] = b.clone(*defOp)->getResult(0);
    return;
  }

  // This indicates whether we have set an insertion point.
  bool hasInsertionPoint = false;

  // First we examine the AST structure.
  mlir::Operation *parentOp = defOp->getParentOp();
  if (mlir::affine::AffineForOp forOp =
          dyn_cast<mlir::affine::AffineForOp>(parentOp)) {
    mlir::Value srcIV = forOp.getInductionVar();
    std::string ivName = oslValueTable->lookup(srcIV);
    mlir::Value dstIV = symbolTable[ivName];
    if (dstIV == nullptr) {
      symbolToDeps[ivName].insert(val);
      valueToDepSymbols[val].insert(ivName);
    } else {
      // Now the loop IV is there, we just find its owner for loop and clone
      // the op.
      mlir::Block *blockToInsert = dstIV.cast<mlir::BlockArgument>().getOwner();
      hasInsertionPoint = true;
      b.setInsertionPointToStart(blockToInsert);
    }
  } else if (mlir::func::FuncOp funOp =
                 dyn_cast<mlir::func::FuncOp>(parentOp)) {
    // Insert at the beginning of this function.
    hasInsertionPoint = true;
    b.setInsertionPointToStart(&entryBlock);
  } else {
    assert(false);
  }

  SmallVector<mlir::Value, 8> newOperands;
  // Next, we check whether all operands are in the symbol table.
  for (mlir::Value operand : defOp->getOperands()) {
    std::string operandSymbol = oslValueTable->lookup(operand);
    if (operandSymbol.empty()) {
      mlir::Operation *operandDefOp = operand.getDefiningOp();
      if (operandDefOp && isa<mlir::arith::ConstantOp>(operandDefOp)) {
        newOperands.push_back(b.clone(*operandDefOp)->getResult(0));
        continue;
      }
    }

    assert(!operandSymbol.empty() &&
           "operand should be in the original symbol table.");
    mlir::Value newOperand = symbolTable[operandSymbol];
    // If the symbol is not yet initialized, we update the two dependence
    // tables. Note that here we implicitly assume that the operand symbol
    // should exist.
    assert(newOperand != nullptr);
    newOperands.push_back(newOperand);
  }

  // The operands are not sufficient, should wait.
  if (newOperands.size() < defOp->getNumOperands())
    return;

  // Finally do the initialization.
  if (!hasInsertionPoint)
    return;

  IRMapping vMap;
  for (unsigned i = 0; i < newOperands.size(); i++)
    vMap.map(defOp->getOperand(i), newOperands[i]);

  mlir::Operation *newOp = b.clone(*defOp, vMap);
  assert(newOp != nullptr);
  assert(newOp->getNumResults() == 1 && "Should only have one result.");

  symbolTable[symbol] = newOp->getResult(0);
}

void Importer::initializeSymbolTable() {
  IslScop::SymbolTable *oslSymbolTable = scop->getSymbolTable();

  OpBuilder::InsertionGuard guard(b);

  auto &entryBlock = *func.getBody().begin();
  b.setInsertionPointToStart(&entryBlock);

  /// Constants
  symbolTable["zero"] =
      b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), b.getIndexType(),
                                        b.getIntegerAttr(b.getIndexType(), 0));

  for (const auto &it : *oslSymbolTable)
    initializeSymbol(it.second);
}

LogicalResult
Importer::parseUserStmtBody(llvm::StringRef body, std::string &calleeName,
                            llvm::SmallVectorImpl<std::string> &args) {
  unsigned bodyLen = body.size();
  unsigned pos = 0;

  // Read until the left bracket for the function name.
  for (; pos < bodyLen && body[pos] != '('; pos++)
    calleeName.push_back(body[pos]);
  pos++; // Consume the left bracket.

  // Read argument names.
  while (pos < bodyLen) {
    std::string arg;
    for (; pos < bodyLen && body[pos] != ',' && body[pos] != ')'; pos++) {
      if (body[pos] != ' ') // Ignore whitespaces
        arg.push_back(body[pos]);
    }

    if (!arg.empty())
      args.push_back(arg);
    // Consume either ',' or ')'.
    pos++;
  }

  return success();
}

void Importer::createCalleeAndCallerArgs(
    llvm::StringRef calleeName, llvm::ArrayRef<std::string> args,
    mlir::func::FuncOp &callee, SmallVectorImpl<mlir::Value> &callerArgs) {
  // TODO: avoid duplicated callee creation
  // Cache the current insertion point before changing it for the new callee
  // function.
  auto currBlock = b.getBlock();
  auto currPt = b.getInsertionPoint();

  // Create the callee.
  // First, we create the callee function type.
  unsigned numArgs = args.size();
  llvm::SmallVector<mlir::Type, 8> calleeArgTypes;

  for (unsigned i = 0; i < numArgs; i++) {
    if (isMemrefArg(args[i])) {
      // Memref. A memref name and its number of dimensions.
      auto memName = args[i];
      auto memShape = std::vector<int64_t>(std::stoi(args[i + 1]), -1);
      MemRefType memType = MemRefType::get(memShape, b.getF32Type());
      calleeArgTypes.push_back(memType);
      i++;
    } else {
      // Loop IV.
      calleeArgTypes.push_back(b.getIndexType());
    }
  }

  auto calleeType = b.getFunctionType(calleeArgTypes, {});
  // TODO: should we set insertion point for the callee before the main
  // function?
  b.setInsertionPoint(module.getBody(), getFuncInsertPt());
  callee = b.create<FuncOp>(UnknownLoc::get(context), calleeName, calleeType);
  calleeMap[calleeName] = callee;

  // Create the caller.
  b.setInsertionPoint(currBlock, currPt);

  // Initialise all the caller arguments. The first argument should be the
  // memory object, which is set to be a BlockArgument.
  auto &entryBlock = *func.getBlocks().begin();

  for (unsigned i = 0; i < numArgs; i++) {
    if (isMemrefArg(args[i])) {
      // TODO: refactorize this.
      auto memShape = std::vector<int64_t>(std::stoi(args[i + 1]), -1);
      MemRefType memType = MemRefType::get(memShape, b.getF32Type());

      // TODO: refactorize these two lines into a single API.
      Value memref = symTable->getValue(args[i]);
      if (!memref) {
        memref = entryBlock.addArgument(memType, b.getUnknownLoc());
        symTable->setValue(args[i], memref, PolymerSymbolTable::Memref);
      }
      callerArgs.push_back(memref);
      i++;
    } else if (auto val = symTable->getValue(args[i])) {
      // The rest of the arguments are access indices. They could be the loop
      // IVs or the parameters. Loop IV
      callerArgs.push_back(val);
      // Symbol.
      // TODO: manage sym name by the symTable.
    } else if (symNameToArg.find(args[i]) != symNameToArg.end()) {
      callerArgs.push_back(symNameToArg.lookup(args[i]));
      // TODO: what if an index is a constant?
    } else if (iterScatNameMap.find(args[i]) != iterScatNameMap.end()) {
      auto newArgName = iterScatNameMap[args[i]];
      if (auto iv = symTable->getValue(newArgName)) {
        callerArgs.push_back(iv);
        // We should set the symbol table for args[i], otherwise we cannot
        // build a correct mapping from the original symbol table (only
        // args[i] exists in it).
        symTable->setValue(args[i], iv, PolymerSymbolTable::LoopIV);
      } else {
        llvm::errs() << "Cannot find the scatname " << newArgName
                     << " as a valid loop IV.\n";
        return;
      }
    } else { // TODO: error handling
      llvm::errs() << "Cannot find " << args[i]
                   << " as a loop IV name or a symbole name. Please check if "
                      "the statement body uses the same iterator name as the "
                      "one in <scatnames>.\n";
      return;
    }
  }
}

void Importer::getAffineExprForLoopIterator(
    clast_stmt *subst, llvm::SmallVectorImpl<mlir::Value> &operands,
    AffineMap &affMap) {
  assert(CLAST_STMT_IS_A(subst, stmt_ass) &&
         "Should use clast assignment here.");

  clast_assignment *substAss = reinterpret_cast<clast_assignment *>(subst);

  AffineExprBuilder builder(context, symTable, &symbolTable, scop, options);
  SmallVector<AffineExpr, 1> affExprs;
  auto res = succeeded(builder.process(substAss->RHS, affExprs));
  assert(res);

  // Insert dim operands.
  for (llvm::StringRef dimName : builder.dimNames.keys()) {
    mlir::Value iv = symbolTable[dimName];
    assert(iv != nullptr);
    operands.push_back(iv);
  }
  // Symbol operands
  for (llvm::StringRef symName : builder.symbolNames.keys()) {
    mlir::Value operand = symbolTable[symName];
    assert(operand != nullptr);
    operands.push_back(operand);
  }

  // Create the AffineMap for loop bound.
  affMap = AffineMap::get(builder.dimNames.size(), builder.symbolNames.size(),
                          affExprs, context);
}

void Importer::getInductionVars(clast_user_stmt *userStmt, osl_body_p body,
                                SmallVectorImpl<mlir::Value> &inductionVars) {
  char *expr = osl_util_identifier_substitution(body->expression->string[0],
                                                body->iterators->string);
  // dbgs() << "Getting induction vars from: " << (*body->expression->string[0])
  //        << '\n' << (*expr) << '\n';
  char *tmp = expr;
  clast_stmt *subst;

  /* Print the body expression, substituting the @...@ markers. */
  while (*expr) {
    if (*expr == '@') {
      int iterator;
      expr += sscanf(expr, "@%d", &iterator) + 2; /* 2 for the @s */
      subst = userStmt->substitutions;
      for (int i = 0; i < iterator; i++)
        subst = subst->next;

      SmallVector<mlir::Value, 8> substOperands;
      AffineMap substMap;
      getAffineExprForLoopIterator(subst, substOperands, substMap);

      mlir::Operation *op;
      if (substMap.isSingleConstant())
        op = b.create<mlir::arith::ConstantOp>(
            b.getUnknownLoc(), b.getIndexType(),
            b.getIntegerAttr(b.getIndexType(),
                             substMap.getSingleConstantResult()));
      else
        op = b.create<mlir::affine::AffineApplyOp>(b.getUnknownLoc(), substMap,
                                                   substOperands);

      inductionVars.push_back(op->getResult(0));
    } else {
      expr++;
    }
  }
  free(tmp);
}

static mlir::Value findBlockArg(mlir::Value v) {
  mlir::Value r = v;
  while (r != nullptr) {
    if (r.isa<BlockArgument>())
      break;

    mlir::Operation *defOp = r.getDefiningOp();
    if (!defOp || defOp->getNumOperands() != 1)
      return nullptr;
    if (!isa<mlir::arith::IndexCastOp>(defOp))
      return nullptr;

    r = defOp->getOperand(0);
  }

  return r;
}

/// Create a custom call operation for each user statement. A user statement
/// should be in the format of <stmt-id>`(`<ssa-id>`)`, in which a SSA ID can be
/// a memref, a loop IV, or a symbol parameter (defined as a block argument). We
/// will also generate the declaration of the function to be called, which has
/// an empty body, in order to make the compiler happy.
LogicalResult Importer::processStmt(clast_user_stmt *userStmt) {
  // TODO ISL
}

/// Process the if statement.
LogicalResult Importer::processStmt(clast_guard *guardStmt) {
  // TODO ISL
}

/// We treat the provided the clast_expr as a loop bound. If it is a min/max
/// reduction, we will expand that into multiple expressions.
static LogicalResult processClastLoopBound(clast_expr *expr,
                                           AffineExprBuilder &builder,
                                           SmallVectorImpl<AffineExpr> &exprs) {
  // TODO ISL
}

LogicalResult
Importer::getAffineLoopBound(clast_expr *expr,
                             llvm::SmallVectorImpl<mlir::Value> &operands,
                             AffineMap &affMap, bool isUpper) {
  // TODO ISL
}

/// Generate the affine::AffineForOp from a clast_for statement. First we create
/// AffineMaps for the lower and upper bounds. Then we decide the step if
/// there is any. And finally, we create the affine::AffineForOp instance and
/// generate its body.
LogicalResult Importer::processStmt(clast_for *forStmt) {
  // TODO ISL
}

LogicalResult Importer::processStmt(clast_assignment *ass) {
  // TODO ISL
}

static std::unique_ptr<IslScop> readOpenScop(llvm::MemoryBufferRef buf) {
  // TODO ISL
}

static void updateCloogOptionsByPlutoProg(CloogOptions *options,
                                          const PlutoProg *prog) {
  // TODO ISL
}

static void unrollJamClastByPlutoProg(clast_stmt *root, const PlutoProg *prog,
                                      CloogOptions *cloogOptions,
                                      unsigned ufactor) {
  // TODO ISL
}

static void markParallel(clast_stmt *root, const PlutoProg *prog,
                         CloogOptions *cloogOptions) {
  pluto_mark_parallel(root, prog, cloogOptions);
}

static void transformClastByPlutoProg(clast_stmt *root, const PlutoProg *prog,
                                      CloogOptions *cloogOptions,
                                      PlutoOptions *plutoOptions) {
  if (plutoOptions->unrolljam)
    unrollJamClastByPlutoProg(root, prog, cloogOptions, plutoOptions->ufactor);
  if (plutoOptions->parallel)
    markParallel(root, prog, cloogOptions);
}

namespace polymer {
mlir::Operation *createFuncOpFromIsl(std::unique_ptr<IslScop> scop,
                                     ModuleOp module,
                                     PolymerSymbolTable &symTable,
                                     MLIRContext *context, PlutoProg *prog,
                                     const char *dumpClastAfterPluto) {
  // TODO ISL
}
} // namespace polymer

OwningOpRef<ModuleOp>
polymer::translateIslToModule(std::unique_ptr<IslScop> scop,
                              MLIRContext *context) {
  context->loadDialect<affine::AffineDialect>();
  OwningOpRef<ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0)));

  PolymerSymbolTable symTable;
  if (!createFuncOpFromIsl(std::move(scop), module.get(), symTable, context))
    return {};

  return module;
}

static OwningOpRef<ModuleOp> translateIslToModule(llvm::SourceMgr &sourceMgr,
                                                  MLIRContext *context) {
  llvm::SMDiagnostic err;
  std::unique_ptr<IslScop> scop =
      readOpenScop(*sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()));

  return translateIslToModule(std::move(scop), context);
}

namespace polymer {

void registerFromIslTranslation() {
  TranslateToMLIRRegistration fromLLVM(
      "import-isl", "Import ISL",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return ::translateIslToModule(sourceMgr, context);
      });
}

} // namespace polymer

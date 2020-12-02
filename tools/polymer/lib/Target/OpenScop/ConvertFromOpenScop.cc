//===- ConvertFromOpenScop.h ------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#include "cloog/cloog.h"
#include "osl/osl.h"

#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/OpenScop.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "mlir/Translation.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

using namespace polymer;
using namespace mlir;

typedef llvm::StringMap<mlir::Operation *> StmtOpMap;
typedef llvm::StringMap<mlir::Value> NameValueMap;

namespace {

/// Build AffineExpr from a clast_expr.
/// TODO: manage the priviledge.
class AffineExprBuilder {
public:
  AffineExprBuilder(MLIRContext *context, OslSymbolTable *symTable,
                    OslScop *scop, CloogOptions *options)
      : b(context), context(context), symTable(symTable), scop(scop),
        options(options) {
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
  /// The OslScop of the whole program.
  OslScop *scop;
  ///
  OslSymbolTable *symTable;
  ///
  CloogOptions *options;

  llvm::StringMap<unsigned> symbolNames;
  llvm::StringMap<unsigned> dimNames;
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
  default:
    assert(false && "Unrecognized clast_expr_type.\n");
    return failure();
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
      symbolNames[expr->name] = symbolNames.size();
    }
  } else if (auto iv = symTable->getValue(expr->name)) {
    if (dimNames.find(expr->name) != dimNames.end())
      affExprs.push_back(b.getAffineDimExpr(dimNames[expr->name]));
    else {
      affExprs.push_back(b.getAffineDimExpr(dimNames.size()));
      dimNames[expr->name] = dimNames.size();
    }
  } else {
    llvm::errs()
        << expr->name
        << " is not a valid name can be found as a symbol or a loop IV.\n";
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
  default:
    assert(false && "Unrecognized clast_binary type.\n");
    return failure();
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
  default:
    llvm::errs() << "Clast expr type: " << expr->type
                 << " is not yet supported\n";
    return failure();
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

  for (unsigned i = 1; i < expr->n; ++i) {
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

  for (unsigned i = 1; i < expr->n; i++) {
    if (failed(process(expr->elts[i], affExprs)))
      return failure();
  }

  return success();
}
/// Builds the mapping from the iterator names in a statement to their
/// corresponding names in <scatnames>, based on the matrix provided by the
/// scattering relation.
static void
buildIterToScatNameMap(llvm::StringMap<llvm::StringRef> &iterToScatName,
                       osl_statement_p stmt, osl_generic_p scatnames) {
  // Get the body from the statement.
  osl_body_p body = osl_statement_get_body(stmt);
  assert(body != nullptr && "The body of the statement should not be NULL.");
  assert(body->expression != nullptr &&
         "The body expression should not be NULL.");
  assert(body->iterators != nullptr &&
         "The body iterators should not be NULL.");

  // Get iterator names.
  unsigned numIterNames = osl_strings_size(body->iterators);
  llvm::SmallVector<llvm::StringRef, 8> iterNames(numIterNames);
  for (unsigned i = 0; i < numIterNames; i++)
    iterNames[i] = body->iterators->string[i];

  // Split the scatnames into a list of strings.
  osl_strings_p scatNamesData =
      reinterpret_cast<osl_scatnames_p>(scatnames->data)->names;
  unsigned numScatNames = osl_strings_size(scatNamesData);

  llvm::SmallVector<llvm::StringRef, 8> scatNames(numScatNames);
  for (unsigned i = 0; i < numScatNames; i++)
    scatNames[i] = scatNamesData->string[i];

  // Get the scattering relation.
  osl_relation_p scats = stmt->scattering;
  assert(scats != nullptr && "scattering in the statement should not be NULL.");
  assert(scats->nb_input_dims == iterNames.size() &&
         "# input dims should equal to # iter names.");
  assert(scats->nb_output_dims <= scatNames.size() &&
         "# output dims should be less than or equal to # scat names.");

  // Build the mapping.
  for (unsigned i = 0; i < scats->nb_output_dims; i++)
    for (unsigned j = 0; j < scats->nb_input_dims; j++)
      if (scats->m[i][j + scats->nb_output_dims + 1].dp)
        iterToScatName[iterNames[j]] = scatNames[i];
}

namespace {
/// Build mapping between the iter names in the original code to the scatname in
/// the OpenScop.
class IterScatNameMapper {
public:
  IterScatNameMapper(OslScop *scop, CloogOptions *options)
      : scop(scop), options(options) {}

  void visitStmtList(clast_stmt *s);

  llvm::StringMap<llvm::StringRef> getIterScatNameMap() {
    return iterScatNameMap;
  };

private:
  void visit(clast_for *forStmt);
  void visit(clast_guard *guardStmt);
  void visit(clast_user_stmt *userStmt);

  OslScop *scop;
  clast_stmt *root;
  CloogOptions *options;

  llvm::StringMap<llvm::StringRef> iterScatNameMap;
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
  osl_statement_p stmt;
  if (failed(scop->getStatement(userStmt->statement->number - 1, &stmt)))
    return assert(false);

  osl_body_p body = osl_statement_get_body(stmt);
  assert(body != NULL && "The body of the statement should not be NULL.");
  assert(body->expression != NULL && "The body expression should not be NULL.");
  assert(body->iterators != NULL && "The body iterators should not be NULL.");

  // Map iterator names in the current statement to the values in <scatnames>.
  osl_generic_p scatnames = scop->getExtension("scatnames");
  assert(scatnames && "There should be a <scatnames> in the scop.");
  buildIterToScatNameMap(iterScatNameMap, stmt, scatnames);
}

namespace {
/// Import MLIR code from the clast AST.
class Importer {
public:
  using SymbolTable = llvm::StringMap<mlir::Value>;

  Importer(MLIRContext *context, ModuleOp module, OslSymbolTable *symTable,
           OslScop *scop, CloogOptions *options);

  LogicalResult processStmtList(clast_stmt *s);

  mlir::Operation *getFunc() { return func; }

private:
  void initializeSymbolTable();
  void initializeFuncOpInterface();
  void initializeSymbol(mlir::Value val);

  void fulfillSymbolDependence(llvm::StringRef symbol);

  LogicalResult processStmt(clast_root *rootStmt);
  LogicalResult processStmt(clast_for *forStmt);
  LogicalResult processStmt(clast_guard *guardStmt);
  LogicalResult processStmt(clast_user_stmt *userStmt);
  LogicalResult processStmt(clast_assignment *ass);

  std::string getSourceFuncName() const;
  mlir::FuncOp getSourceFuncOp();

  LogicalResult getAffineLoopBound(clast_expr *expr,
                                   llvm::SmallVectorImpl<mlir::Value> &operands,
                                   AffineMap &affMap, bool isUpper = false);
  void
  getAffineExprForLoopIterator(clast_stmt *subst,
                               llvm::SmallVectorImpl<mlir::Value> &operands,
                               AffineMap &affMap);

  LogicalResult parseUserStmtBody(llvm::StringRef body, std::string &calleeName,
                                  llvm::SmallVectorImpl<std::string> &args);

  bool isMemrefArg(llvm::StringRef argName);
  bool isResultArg(llvm::StringRef argName);

  /// Functions are always inserted before the module terminator.
  Block::iterator getFuncInsertPt() {
    return std::prev(module.getBody()->end());
  }
  /// A helper to create a callee.
  void createCalleeAndCallerArgs(
      llvm::StringRef calleeName, llvm::ArrayRef<std::string> args,
      mlir::FuncOp &callee, SmallVectorImpl<mlir::Value> &callerArgs,
      llvm::StringMap<llvm::StringRef> &iterToScatName);

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
  OslScop *scop;
  /// The symbol table for labels in the OpenScop input (to be deprecated).
  OslSymbolTable *symTable;
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

  llvm::StringMap<llvm::StringRef> iterScatNameMap;

  clast_stmt *root;
  CloogOptions *options;
  FILE *output;
};
} // namespace

Importer::Importer(MLIRContext *context, ModuleOp module,
                   OslSymbolTable *symTable, OslScop *scop,
                   CloogOptions *options)
    : b(context), context(context), module(module), scop(scop),
      symTable(symTable), options(options) {
  b.setInsertionPointToStart(module.getBody());
}

mlir::FuncOp Importer::getSourceFuncOp() {
  std::string sourceFuncName = getSourceFuncName();
  mlir::Operation *sourceFuncOp = module.lookupSymbol(sourceFuncName);

  assert(sourceFuncOp != nullptr &&
         "sourceFuncName cannot be found in the module");
  assert(isa<mlir::FuncOp>(sourceFuncOp) &&
         "Found sourceFuncOp should be of type mlir::FuncOp.");
  return cast<mlir::FuncOp>(sourceFuncOp);
}

/// If there is anything in the comment, we will use it as a function name.
/// Otherwise, we return an empty string.
std::string Importer::getSourceFuncName() const {
  osl_generic_p comment = scop->getExtension("comment");
  if (comment) {
    char *commentStr = reinterpret_cast<osl_comment_p>(comment->data)->comment;
    return std::string(commentStr);
  }
  return std::string("");
}

bool Importer::isMemrefArg(llvm::StringRef argName) {
  // TODO: should find a better way to do this, e.g., using the old symbol
  // table.
  return argName.size() >= 2 && argName[0] == 'A';
}

bool Importer::isResultArg(llvm::StringRef argName) {
  return argName.size() >= 2 && argName[0] == 'S';
}

LogicalResult Importer::processStmtList(clast_stmt *s) {
  // These lines can be optimized.
  if (iterScatNameMap.empty()) {
    IterScatNameMapper iterScatNameMapper(scop, options);
    iterScatNameMapper.visitStmtList(s);
    iterScatNameMap = iterScatNameMapper.getIterScatNameMap();
  }

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
      llvm::errs() << "clast_stmt type not supported\n";
      return failure();
    }
  }

  // // Post update the function type.
  // auto &entryBlock = *func.getBlocks().begin();
  // auto funcType = b.getFunctionType(entryBlock.getArgumentTypes(),
  // llvm::None); func.setType(funcType);

  return success();
}

void Importer::initializeFuncOpInterface() {
  OslScop::SymbolTable *oslSymbolTable = scop->getSymbolTable();
  OslScop::ValueTable *oslValueTable = scop->getValueTable();

  /// First collect the source FuncOp in the original MLIR code.
  mlir::FuncOp sourceFuncOp = getSourceFuncOp();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(module.getBody(), getFuncInsertPt());

  // The default function name is main.
  llvm::StringRef funcName("main");
  // If the comment is provided, we will use it as the function name.
  // TODO: make sure it is safe.
  std::string sourceFuncName = getSourceFuncName();
  if (!sourceFuncName.empty())
    funcName = llvm::StringRef(formatv("{0}_new", sourceFuncName));
  // Create the function interface.
  func =
      b.create<FuncOp>(sourceFuncOp.getLoc(), funcName, sourceFuncOp.getType());

  // Initialize the symbol table for these entryBlock arguments
  auto &entryBlock = *func.addEntryBlock();
  for (unsigned i = 0; i < entryBlock.getNumArguments(); i++) {
    llvm::StringRef argSymbol =
        oslValueTable->lookup(sourceFuncOp.getArgument(i));
    symbolTable[argSymbol] = entryBlock.getArgument(i);
  }

  // Generate an entry block and implicitly insert a ReturnOp at its end.
  b.setInsertionPoint(&entryBlock, entryBlock.end());
  b.create<mlir::ReturnOp>(UnknownLoc::get(context));
}

/// Translate the root statement as a function. The name of the function is by
/// default "main".
LogicalResult Importer::processStmt(clast_root *rootStmt) {
  // Create the function.
  initializeFuncOpInterface();
  // For the rest of the body
  mlir::Block &entryBlock = *func.getBody().begin();
  b.setInsertionPointToStart(&entryBlock);
  // Initialize several values before start.
  initializeSymbolTable();

  return success();
}

/// Initialize the value in the symbol table.
void Importer::initializeSymbol(mlir::Value val) {
  assert(val != nullptr);
  OslScop::SymbolTable *oslSymbolTable = scop->getSymbolTable();
  OslScop::ValueTable *oslValueTable = scop->getValueTable();

  auto &entryBlock = *func.getBody().begin();

  OpBuilder::InsertionGuard guard(b);

  StringRef symbol = oslValueTable->lookup(val);
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
  if (isa<mlir::AllocaOp>(defOp)) {
    /// Skip scratchpad for now.
    return;
  }

  // This indicates whether we have set an insertion point.
  bool hasInsertionPoint = false;

  // First we examine the AST structure.
  mlir::Operation *parentOp = defOp->getParentOp();
  if (mlir::AffineForOp forOp = dyn_cast<mlir::AffineForOp>(parentOp)) {
    mlir::Value srcIV = forOp.getInductionVar();
    llvm::StringRef ivName = oslValueTable->lookup(srcIV);
    mlir::Value dstIV = symbolTable[ivName];
    if (dstIV == nullptr) {
      // if (isa<mlir::AllocaOp>(defOp)) {
      //   llvm::errs() << "Located under iv name: " << ivName << "\n";
      // }
      symbolToDeps[ivName].insert(val);
      valueToDepSymbols[val].insert(ivName);
    } else {
      // Now the loop IV is there, we just find its owner for loop and clone
      // the op.
      mlir::Block *blockToInsert = dstIV.cast<mlir::BlockArgument>().getOwner();
      // blockToInsert->dump();
      hasInsertionPoint = true;
      b.setInsertionPointToStart(blockToInsert);
    }
  } else if (mlir::FuncOp funOp = dyn_cast<mlir::FuncOp>(parentOp)) {
    // Insert at the beginning of this function.
    hasInsertionPoint = true;
    b.setInsertionPointToStart(&entryBlock);
  } else {
    assert(false);
  }

  // if (isa<mlir::AllocaOp>(defOp)) {
  //   llvm::errs() << "Has insertion point: " << hasInsertionPoint << "\n";
  // }

  SmallVector<mlir::Value, 8> newOperands;
  // Next, we check whether all operands are in the symbol table.
  for (mlir::Value operand : defOp->getOperands()) {
    llvm::StringRef operandSymbol = oslValueTable->lookup(operand);
    if (operandSymbol.empty()) {
      mlir::Operation *operandDefOp = operand.getDefiningOp();
      if (operandDefOp && isa<mlir::ConstantOp>(operandDefOp)) {
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
    if (newOperand == nullptr) {
      symbolToDeps[operandSymbol].insert(val);
      valueToDepSymbols[val].insert(operandSymbol);
      continue;
    }
    newOperands.push_back(newOperand);
  }

  // The operands are not sufficient, should wait.
  if (newOperands.size() < defOp->getNumOperands())
    return;

  // Finally do the initialization.
  if (!hasInsertionPoint)
    return;

  BlockAndValueMapping vMap;
  for (unsigned i = 0; i < newOperands.size(); i++)
    vMap.map(defOp->getOperand(i), newOperands[i]);

  mlir::Operation *newOp = b.clone(*defOp, vMap);
  assert(newOp->getNumResults() == 1 && "Should only have one result.");

  symbolTable[symbol] = newOp->getResult(0);

  // Update the dependence table.
  fulfillSymbolDependence(symbol);
}

/**
 * symbol now exists in the symbolTable, and we can check whether we can
 * initialize other symbols that depends on them.
 */
void Importer::fulfillSymbolDependence(llvm::StringRef symbol) {
  // llvm::errs() << "Fulfilling symbol: " << symbol << "\n";
  assert(symbolTable[symbol] != nullptr &&
         "Symbol to be fulfilled not in the symbol table.");

  for (mlir::Value val : symbolToDeps[symbol]) {
    assert(valueToDepSymbols[val].contains(symbol));

    valueToDepSymbols[val].remove(symbol);
    if (valueToDepSymbols[val].empty()) {

      initializeSymbol(val);
    }
  }

  symbolToDeps.erase(symbol);
}

void Importer::initializeSymbolTable() {
  OslScop::SymbolTable *oslSymbolTable = scop->getSymbolTable();
  OslScop::ValueTable *oslValueTable = scop->getValueTable();

  OpBuilder::InsertionGuard guard(b);

  auto &entryBlock = *func.getBody().begin();
  b.setInsertionPointToStart(&entryBlock);

  /// Constants
  symbolTable["zero"] =
      b.create<mlir::ConstantOp>(b.getUnknownLoc(), b.getIndexType(),
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
    mlir::FuncOp &callee, SmallVectorImpl<mlir::Value> &callerArgs,
    llvm::StringMap<llvm::StringRef> &iterToScatName) {
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

  auto calleeType = b.getFunctionType(calleeArgTypes, llvm::None);
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
        memref = entryBlock.addArgument(memType);
        symTable->setValue(args[i], memref, OslSymbolTable::Memref);
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
    } else if (iterToScatName.find(args[i]) != iterToScatName.end()) {
      auto newArgName = iterToScatName[args[i]];
      if (auto iv = symTable->getValue(newArgName)) {
        callerArgs.push_back(iv);
        // We should set the symbol table for args[i], otherwise we cannot
        // build a correct mapping from the original symbol table (only
        // args[i] exists in it).
        symTable->setValue(args[i], iv, OslSymbolTable::LoopIV);
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
  // clast_pprint(stderr, &(substAss->stmt), 0, options);

  AffineExprBuilder builder(context, symTable, scop, options);
  SmallVector<AffineExpr, 1> affExprs;
  // llvm::errs() << "RHS: ";
  // clast_pprint_expr(options, stderr, substAss->RHS);
  // llvm::errs() << "\n";
  assert(succeeded(builder.process(substAss->RHS, affExprs)));

  // Insert dim operands.
  for (llvm::StringRef dimName : builder.dimNames.keys()) {
    mlir::Value iv = symbolTable[dimName];
    // if (iv == nullptr) {
    //   affExprs[0].dump();
    //   llvm::errs() << "Missing: " << dimName << "\n";
    // }
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

/// Create a custom call operation for each user statement. A user statement
/// should be in the format of <stmt-id>`(`<ssa-id>`)`, in which a SSA ID can be
/// a memref, a loop IV, or a symbol parameter (defined as a block argument). We
/// will also generate the declaration of the function to be called, which has
/// an empty body, in order to make the compiler happy.
LogicalResult Importer::processStmt(clast_user_stmt *userStmt) {
  OslScop::ScopStmtMap *scopStmtMap = scop->getScopStmtMap();
  OslScop::ValueTable *valueTable = scop->getValueTable();
  OslScop::SymbolTable *symbolTabele = scop->getSymbolTable();

  osl_statement_p stmt;
  if (failed(scop->getStatement(userStmt->statement->number - 1, &stmt)))
    return failure();

  osl_body_p body = osl_statement_get_body(stmt);
  assert(body != NULL && "The body of the statement should not be NULL.");
  assert(body->expression != NULL && "The body expression should not be NULL.");
  assert(body->iterators != NULL && "The body iterators should not be NULL.");

  // Map iterator names in the current statement to the values in <scatnames>.
  osl_generic_p scatnames = scop->getExtension("scatnames");
  assert(scatnames && "There should be a <scatnames> in the scop.");

  // clast_pprint(stderr, (clast_stmt *)userStmt, 0, options);
  SmallVector<mlir::Value, 8> inductionVars;
  unsigned numIters = osl_strings_size(body->iterators);

  char *expr = osl_util_identifier_substitution(body->expression->string[0],
                                                body->iterators->string);
  char *tmp = expr;
  clast_stmt *subst;

  /* Print the body expression, substituting the @...@ markers. */
  while (*expr) {
    // llvm::errs() << expr << "\n";
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
        op = b.create<mlir::ConstantOp>(
            b.getUnknownLoc(), b.getIndexType(),
            b.getIntegerAttr(b.getIndexType(),
                             substMap.getSingleConstantResult()));
      else
        op = b.create<mlir::AffineApplyOp>(b.getUnknownLoc(), substMap,
                                           substOperands);
      inductionVars.push_back(op->getResult(0));
    } else {
      *expr++;
    }
  }
  free(tmp);

  // Parse the statement body.
  llvm::SmallVector<std::string, 8> args;
  std::string calleeName;
  if (failed(parseUserStmtBody(body->expression->string[0], calleeName, args)))
    return failure();

  // Create the callee and the caller args.
  FuncOp callee;
  llvm::SmallVector<mlir::Value, 8> callerArgs;

  Location loc = b.getUnknownLoc();

  // If the calleeName can be found in the scopStmtMap, i.e., we have the
  // definition of the callee already, we will generate the caller based on that
  // interface.
  if (scopStmtMap->find(calleeName) != scopStmtMap->end()) {
    const auto &it = scopStmtMap->find(calleeName);
    callee = it->second.getCallee();
    // Note that caller is in the original function.
    mlir::CallOp origCaller = it->second.getCaller();
    // origCaller.dump();
    loc = origCaller.getLoc();
    unsigned currInductionVar = 0;

    for (mlir::Value arg : origCaller.getOperands()) {
      llvm::StringRef argSymbol = valueTable->lookup(arg);

      // Special handling for the memory allocation case.
      mlir::Operation *defOp = arg.getDefiningOp();
      if (defOp && isa<mlir::AllocaOp>(defOp)) {
        // If this memory has been allocated, need to check its owner.
        if (mlir::Value val = this->symbolTable.lookup(argSymbol)) {
          DominanceInfo dom(func);
          if (dom.dominates(val.getParentBlock(), b.getBlock())) {
            callerArgs.push_back(val);
            continue;
          }
        }

        // Otherwise, create a new alloca op.
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(b.getBlock());
        mlir::Operation *newDefOp = b.clone(*defOp);

        this->symbolTable[argSymbol] = newDefOp->getResult(0);
        // fulfillSymbolDependence(argSymbol);

        callerArgs.push_back(newDefOp->getResult(0));
      } else if (scop->isDimSymbol(argSymbol)) {
        callerArgs.push_back(inductionVars[currInductionVar++]);
      } else if (mlir::Value val = this->symbolTable.lookup(argSymbol)) {
        callerArgs.push_back(val);
      } else {
        llvm::StringRef scatName = iterScatNameMap.lookup(argSymbol);
        // Dealing with loop IV.
        if (!scatName.empty()) {
          argSymbol = scatName;
          if (mlir::Value val = this->symbolTable.lookup(argSymbol))
            callerArgs.push_back(val);
          else // no IV symbol means this statement is called errside the loop.
            callerArgs.push_back(this->symbolTable.lookup("zero"));
        } else {
          llvm::errs() << "Missing symbol: " << argSymbol << "\n";
          assert(false && "Cannot insert the correct number of caller args. "
                          "Some symbols must be missing in the symbol table.");
        }
      }
    }
  } else {
    createCalleeAndCallerArgs(calleeName, args, callee, callerArgs,
                              iterScatNameMap);
  }

  // Finally create the CallOp.
  mlir::CallOp caller = b.create<mlir::CallOp>(loc, callee, callerArgs);

  return success();
}

/// Process the if statement.
LogicalResult Importer::processStmt(clast_guard *guardStmt) {
  // Build the integer set.
  SmallVector<AffineExpr, 4> conds;
  SmallVector<bool, 4> eqFlags;

  AffineExprBuilder builder(context, symTable, scop, options);
  // llvm::errs() << "Guard stmt:"
  //              << "\n";
  // clast_pprint(stderr, (clast_stmt *)guardStmt, 0, options);

  for (unsigned i = 0; i < guardStmt->n; i++) {
    clast_equation eq = guardStmt->eq[i];

    SmallVector<AffineExpr, 4> lhsExprs, rhsExprs;
    // builder.reset();
    if (failed(builder.process(eq.LHS, lhsExprs)))
      return failure();
    // builder.reset();
    if (failed(builder.process(eq.RHS, rhsExprs)))
      return failure();

    assert(lhsExprs.size() == 1);

    for (AffineExpr rhsExpr : rhsExprs) {
      AffineExpr eqExpr;
      if (eq.sign >= 0)
        eqExpr = lhsExprs[0] - rhsExpr;
      else
        eqExpr = rhsExpr - lhsExprs[0];
      conds.push_back(eqExpr);
      eqFlags.push_back(eq.sign == 0);
    }
  }

  IntegerSet iset = IntegerSet::get(builder.dimNames.size(),
                                    builder.symbolNames.size(), conds, eqFlags);
  SmallVector<mlir::Value, 8> operands;
  for (auto dimName : builder.dimNames.keys()) {
    mlir::Value iv = symbolTable[dimName];
    llvm::errs() << dimName << "\n";
    assert(iv != nullptr);
    operands.push_back(iv);
  }
  for (auto symName : builder.symbolNames.keys()) {
    mlir::Value sym = symbolTable[symName];
    assert(sym != nullptr);
    operands.push_back(sym);
  }

  mlir::AffineIfOp ifOp =
      b.create<mlir::AffineIfOp>(b.getUnknownLoc(), iset, operands, false);

  Block *entryBlock = ifOp.getThenBlock();
  b.setInsertionPointToStart(entryBlock);
  processStmtList(guardStmt->then);
  b.setInsertionPointAfter(ifOp);

  return success();
}

LogicalResult
Importer::getAffineLoopBound(clast_expr *expr,
                             llvm::SmallVectorImpl<mlir::Value> &operands,
                             AffineMap &affMap, bool isUpper) {
  AffineExprBuilder builder(context, symTable, scop, options);
  SmallVector<AffineExpr, 4> boundExprs;
  // Build the AffineExpr for the loop bound.
  if (failed(builder.process(expr, boundExprs)))
    return failure();

  // If looking at the upper bound, we should add 1 to all of them.
  if (isUpper)
    for (auto &expr : boundExprs)
      expr = expr + b.getAffineConstantExpr(1);

  // Insert dim operands.
  for (auto dimName : builder.dimNames.keys()) {
    if (auto iv = symbolTable[dimName]) {
      operands.push_back(iv);
    } else {
      llvm::errs() << "Dim " << dimName
                   << " cannot be recognized as a value.\n";
      return failure();
    }
  }

  // Create or get BlockArgument for the symbols. We assume all symbols come
  // from the BlockArgument of the generated function.
  auto &entryBlock = *func.getBlocks().begin();
  for (auto symName : builder.symbolNames.keys()) {
    mlir::Value operand = symbolTable[symName];
    assert(operand != nullptr);
    operands.push_back(operand);
  }

  // Create the AffineMap for loop bound.
  affMap = AffineMap::get(builder.dimNames.size(), builder.symbolNames.size(),
                          boundExprs, context);

  return success();
}

/// Generate the AffineForOp from a clast_for statement. First we create
/// AffineMaps for the lower and upper bounds. Then we decide the step if
/// there is any. And finally, we create the AffineForOp instance and generate
/// its body.
LogicalResult Importer::processStmt(clast_for *forStmt) {
  // Get loop bounds.
  AffineMap lbMap, ubMap;
  llvm::SmallVector<mlir::Value, 8> lbOperands, ubOperands;
  OslScop::ValueTable *oslValueTable = scop->getValueTable();

  assert((forStmt->LB && forStmt->UB) && "Unbounded loops are not allowed.");
  // TODO: simplify these sanity checks.
  assert(!(forStmt->LB->type == clast_expr_red &&
           reinterpret_cast<clast_reduction *>(forStmt->LB)->type ==
               clast_red_min) &&
         "If the lower bound is a reduced result, it should not use min for "
         "reduction.");
  assert(!(forStmt->UB->type == clast_expr_red &&
           reinterpret_cast<clast_reduction *>(forStmt->UB)->type ==
               clast_red_max) &&
         "If the lower bound is a reduced result, it should not use max for "
         "reduction.");

  if (failed(getAffineLoopBound(forStmt->LB, lbOperands, lbMap)) ||
      failed(
          getAffineLoopBound(forStmt->UB, ubOperands, ubMap, /*isUpper=*/true)))
    return failure();

  int64_t stride = 1;
  if (cloog_int_gt_si(forStmt->stride, 1)) {
    if (failed(getI64(forStmt->stride, &stride)))
      return failure();
  }

  // Create the for operation.
  mlir::AffineForOp forOp = b.create<mlir::AffineForOp>(
      UnknownLoc::get(context), lbOperands, lbMap, ubOperands, ubMap, stride);
  // forOp.dump();

  // Update the loop IV mapping.
  auto &entryBlock = *forOp.getLoopBody().getBlocks().begin();
  // TODO: confirm is there a case that forOp has multiple operands.
  assert(entryBlock.getNumArguments() == 1 &&
         "affine.for should only have one block argument (iv).");

  symTable->setValue(forStmt->iterator, entryBlock.getArgument(0),
                     OslSymbolTable::LoopIV);

  // Symbol table is mutable.
  // TODO: is there a better way to improve this? Not very safe.
  mlir::Value symValue = symbolTable[forStmt->iterator];
  symbolTable[forStmt->iterator] = entryBlock.getArgument(0);

  fulfillSymbolDependence(forStmt->iterator);

  for (const auto &it : iterScatNameMap) {
    if (it.second.equals(forStmt->iterator)) {
      // llvm::errs() << it.first() << " -> " << it.second << "\n";
      symbolTable[it.first()] = symbolTable[it.second];
      fulfillSymbolDependence(it.first());
    }
  }

  // Create the loop body
  b.setInsertionPointToStart(&entryBlock);
  // llvm::errs() << "Entry is empty: " << entryBlock.empty() << "\n";
  // entryBlock.dump();
  entryBlock.walk([&](mlir::AffineYieldOp op) { b.setInsertionPoint(op); });
  processStmtList(forStmt->body);
  b.setInsertionPointAfter(forOp);

  // Restore the symbol value.
  // llvm::errs() << forStmt->iterator << "\n";
  // if (symValue != nullptr) {
  //   symValue.dump();
  //   symValue.cast<mlir::BlockArgument>().getOwner()->dump();
  // }
  symbolTable[forStmt->iterator] = symValue;
  return success();
}

LogicalResult Importer::processStmt(clast_assignment *ass) {

  SmallVector<mlir::Value, 8> substOperands;
  AffineMap substMap;
  getAffineExprForLoopIterator((clast_stmt *)ass, substOperands, substMap);

  mlir::Operation *op;
  if (substMap.isSingleConstant()) {
    op = b.create<mlir::ConstantOp>(
        b.getUnknownLoc(), b.getIndexType(),
        b.getIntegerAttr(b.getIndexType(), substMap.getSingleConstantResult()));
  } else if (substMap.getNumResults() == 0) {
    op = b.create<mlir::AffineApplyOp>(b.getUnknownLoc(), substMap,
                                       substOperands);
  } else {
    assert(ass->RHS->type == clast_expr_red);
    clast_reduction *red = reinterpret_cast<clast_reduction *>(ass->RHS);

    assert(red->type != clast_red_sum);
    if (red->type == clast_red_max)
      op = b.create<mlir::AffineMaxOp>(b.getUnknownLoc(), substMap,
                                       substOperands);
    else
      op = b.create<mlir::AffineMinOp>(b.getUnknownLoc(), substMap,
                                       substOperands);
  }

  symbolTable[ass->LHS] = op->getResult(0);

  return success();
}

static std::unique_ptr<OslScop> readOpenScop(llvm::MemoryBufferRef buf) {
  // Read OpenScop by OSL API.
  // TODO: is there a better way to get the FILE pointer from
  // MemoryBufferRef?
  FILE *inputFile = fmemopen(
      reinterpret_cast<void *>(const_cast<char *>(buf.getBufferStart())),
      buf.getBufferSize(), "r");

  auto scop = std::make_unique<OslScop>(osl_scop_read(inputFile));
  fclose(inputFile);

  return scop;
}

mlir::Operation *
polymer::createFuncOpFromOpenScop(std::unique_ptr<OslScop> scop,
                                  ModuleOp module, OslSymbolTable &symTable,
                                  MLIRContext *context) {
  // TODO: turn these C struct into C++ classes.
  CloogState *state = cloog_state_malloc();
  CloogOptions *options = cloog_options_malloc(state);
  options->openscop = 1;

  CloogInput *input = cloog_input_from_osl_scop(options->state, scop->get());
  cloog_options_copy_from_osl_scop(scop->get(), options);

  // Create cloog_program
  CloogProgram *program =
      cloog_program_alloc(input->context, input->ud, options);
  program = cloog_program_generate(program, options);

  // Convert to clast
  clast_stmt *rootStmt = cloog_clast_create(program, options);
  clast_pprint(stderr, rootStmt, 0, options);

  // Process the input.
  Importer deserializer(context, module, &symTable, scop.get(), options);
  if (failed(deserializer.processStmtList(rootStmt)))
    return nullptr;

  // Cannot use cloog_input_free, some pointers don't exist.
  free(input);
  cloog_program_free(program);

  options->scop = NULL; // Prevents freeing the scop object.
  cloog_options_free(options);
  cloog_state_free(state);

  return deserializer.getFunc();
}

OwningModuleRef
polymer::translateOpenScopToModule(std::unique_ptr<OslScop> scop,
                                   MLIRContext *context) {
  context->loadDialect<AffineDialect>();
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get("", /*line=*/0, /*column=*/0, context)));

  OslSymbolTable symTable;
  if (!createFuncOpFromOpenScop(std::move(scop), module.get(), symTable,
                                context))
    return {};

  return module;
}

static OwningModuleRef translateOpenScopToModule(llvm::SourceMgr &sourceMgr,
                                                 MLIRContext *context) {
  llvm::SMDiagnostic err;
  std::unique_ptr<OslScop> scop =
      readOpenScop(*sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()));

  return translateOpenScopToModule(std::move(scop), context);
}

namespace polymer {

void registerFromOpenScopTranslation() {
  TranslateToMLIRRegistration fromLLVM(
      "import-scop", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return ::translateOpenScopToModule(sourceMgr, context);
      });
}

} // namespace polymer

//===- ExtractScopStmt.cc - Extract scop stmt to func -----------------C++-===//
//
// This file implements the transformation that extracts scop statements into
// MLIR functions.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/ExtractScopStmt.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

using CalleeName = SmallString<16>;
using OpToCalleeMap = llvm::DenseMap<Operation *, Operation *>;
using CalleeToCallersMap =
    llvm::DenseMap<Operation *, llvm::SetVector<Operation *>>;

/// Discover the operations that have memory write effects.
/// TODO: support CallOp.
static void discoverMemWriteOps(mlir::FuncOp f,
                                SmallVectorImpl<Operation *> &ops) {
  bool hasAffineScope = false;
  f.getOperation()->walk([&](Operation *op) {
    if (isa<mlir::AffineForOp>(op))
      hasAffineScope = true;
    if (isa<mlir::AffineWriteOpInterface, mlir::StoreOp>(op))
      ops.push_back(op);
  });

  if (!hasAffineScope)
    ops.clear();
}

/// Returns the newly created scratchpad.
static mlir::Value
insertScratchpadForInterprocUses(mlir::Operation *defOp,
                                 mlir::Operation *defInCalleeOp,
                                 CalleeToCallersMap &calleeToCallers,
                                 mlir::FuncOp topLevelFun, OpBuilder &b) {
  assert(defOp->getNumResults() == 1);
  assert(topLevelFun.getBlocks().size() != 0);

  Block &entryBlock = *topLevelFun.getBlocks().begin();
  mlir::OpResult val = defOp->getResult(0);

  // Set the allocation point after where the val is defined.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&entryBlock);

  // The memref shape is 1 and the type is derived from val.
  mlir::Type memrefType = MemRefType::get({1}, val.getType());
  mlir::Operation *allocaOp =
      b.create<mlir::AllocaOp>(defOp->getLoc(), memrefType.cast<MemRefType>());
  mlir::Value memref = allocaOp->getResult(0);

  // Give the callee an additional argument
  mlir::Operation *calleeOp = defInCalleeOp;
  while (calleeOp != nullptr) {
    if (isa<mlir::FuncOp>(calleeOp))
      break;
    calleeOp = calleeOp->getParentOp();
  }

  mlir::FuncOp callee = cast<mlir::FuncOp>(calleeOp);
  mlir::Block &calleeEntryBlock = *callee.getBlocks().begin();
  mlir::BlockArgument scratchpad = calleeEntryBlock.addArgument(memrefType);
  callee.setType(b.getFunctionType(
      TypeRange(calleeEntryBlock.getArgumentTypes()), llvm::None));

  // Store within the callee for the used value.
  b.setInsertionPointAfter(defInCalleeOp);
  b.create<mlir::AffineStoreOp>(allocaOp->getLoc(), defInCalleeOp->getResult(0),
                                scratchpad, b.getConstantAffineMap(0),
                                std::vector<mlir::Value>());

  // llvm::errs() << "Updated callee interface:\n";
  // callee.dump();

  // Setup the operand for the caller as well.
  // llvm::errs() << "Updated callers:\n";
  SetVector<mlir::Operation *> callerOpsToRemove;
  for (mlir::Operation *callerOp : calleeToCallers[calleeOp]) {
    mlir::CallOp caller = cast<mlir::CallOp>(callerOp);
    SmallVector<mlir::Value, 8> newOperands;
    for (mlir::Value operand : caller.getOperands())
      newOperands.push_back(operand);
    newOperands.push_back(memref);

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(callerOp);
    mlir::CallOp newCaller =
        b.create<mlir::CallOp>(callerOp->getLoc(), caller.getCallee(),
                               caller.getResultTypes(), newOperands);
    calleeToCallers[calleeOp].insert(newCaller);
    callerOpsToRemove.insert(callerOp);
    callerOp->erase();
  }

  calleeToCallers[calleeOp].set_subtract(callerOpsToRemove);

  return memref;
}

/// Get all the ops belongs to a statement starting from the given
/// operation. The sequence of the operations in defOps will be reversed,
/// depth-first, starting from op. Note that the initial op will be placed in
/// the resulting ops as well.
static void getScopStmtOps(Operation *writeOp, SetVector<Operation *> &ops,
                           SetVector<mlir::Value> &args,
                           OpToCalleeMap &opToCallee,
                           CalleeToCallersMap &calleeToCallers,
                           mlir::FuncOp topLevelFun, OpBuilder &b) {
  SmallVector<Operation *, 8> worklist;
  worklist.push_back(writeOp);
  ops.insert(writeOp);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // If op is already in another callee.
    if (!isa<mlir::ConstantOp>(op) && opToCallee[op]) {
      OpBuilder::InsertionGuard guard(b);
      mlir::Value scratchpad = insertScratchpadForInterprocUses(
          op, opToCallee[op], calleeToCallers, topLevelFun, b);
      args.insert(scratchpad);

      b.setInsertionPointAfter(op);
      mlir::Operation *loadOp = b.create<mlir::AffineLoadOp>(
          op->getLoc(), scratchpad, b.getConstantAffineMap(0),
          std::vector<mlir::Value>());

      ops.insert(loadOp);
      op->replaceAllUsesWith(loadOp);

      scratchpad.dump();
      loadOp->dump();
      continue;
    }

    // Types of operation that terminates the recusion:
    // Memory allocation ops will be omitted, reaching them means the end of
    // recursion. We will take care of these ops in other passes. The result of
    // these allocation op, i.e., memref, will be treated as input arguments to
    // the new statement function.
    // Also, we should leave the dim SSA value in the original scope. Otherwise,
    // if we consume it in the callee, the AffineValueMap built for the accesses
    // that use this dim cannot relate it with the global context.
    if (isa<mlir::AllocaOp, mlir::AllocOp, mlir::DimOp, mlir::AffineApplyOp,
            mlir::IndexCastOp>(op)) {
      for (mlir::Value result : op->getResults())
        args.insert(result);
      continue;
    }

    // Keep the op in the given set. ops also stores the "visited" information:
    // any op inside ops will be treated as visited and won't be inserted into
    // the worklist again.
    ops.insert(op);

    // Recursively visit other defining ops that are not in ops.
    for (mlir::Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      // We find the defining op and place it in the worklist, if it is not null
      // and has not been visited yet.
      if (defOp) {
        if (!ops.contains(defOp))
          worklist.push_back(defOp);
      }
      // Otherwise, stop the recursion at values that don't have a defining op,
      // i.e., block arguments, which could be loop IVs, external arguments,
      // etc. And insert them into the argument list (args).
      else
        args.insert(operand);
    }
  }

  return;
}

static void getCalleeName(unsigned calleeId, CalleeName &calleeName,
                          char prefix = 'S') {
  calleeName.push_back(prefix);
  calleeName += std::to_string(calleeId);
}

/// Create the function definition that contains all the operations that belong
/// to a Scop statement. The function name will be the given calleeName, its
/// contents will be ops, and its type depends on the given list of args. This
/// callee function has a single block in it, and it has no returned value. The
/// callee will be inserted at the end of the whole module.
static mlir::FuncOp createCallee(StringRef calleeName,
                                 const SetVector<Operation *> &ops,
                                 const SetVector<mlir::Value> &args,
                                 mlir::ModuleOp m, Operation *writeOp,
                                 OpToCalleeMap &opToCallee, OpBuilder &b) {
  assert(ops.contains(writeOp) && "writeOp should be a member in ops.");

  // unsigned numArgs = args.size();
  unsigned numOps = ops.size();

  // Get a list of types of all function arguments, and use it to create the
  // function type.
  TypeRange argTypes = ValueRange(args.getArrayRef()).getTypes();
  mlir::FunctionType calleeType = b.getFunctionType(argTypes, llvm::None);

  // Insert the new callee before the end of the module body.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(m.getBody(), std::prev(m.getBody()->end()));

  // Create the callee. Its loc is determined by the writeOp.
  mlir::FuncOp callee =
      b.create<mlir::FuncOp>(writeOp->getLoc(), calleeName, calleeType);
  mlir::Block *entryBlock = callee.addEntryBlock();
  b.setInsertionPointToStart(entryBlock);
  // Terminator
  b.create<mlir::ReturnOp>(callee.getLoc());
  b.setInsertionPointToStart(entryBlock);

  // Create the mapping from the args to the newly created BlockArguments, to
  // replace the uses of the values in the original function to the newly
  // declared entryBlock's input.
  BlockAndValueMapping mapping;
  mapping.map(args, entryBlock->getArguments());
  // for (auto arg : args) {
  //   arg.dump();
  //   llvm::errs() << "->\n";
  //   mapping.lookupOrDefault(arg).dump();
  // }

  // Clone the operations into the new callee function. In case they are not in
  // the correct order, we sort them topologically beforehand.
  SetVector<Operation *> sortedOps = topologicalSort(ops);
  SmallVector<Operation *, 8> clonedOps;

  // Ensures that the cloned operations have their uses updated to the
  // corresponding values in the current scope.
  for (unsigned i = 0; i < numOps; i++) {
    // Build the value mapping while cloning operations.
    sortedOps[i]->walk([&](mlir::Operation *op) {
      for (mlir::Value operand : op->getOperands()) {
        mlir::Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        if (mlir::ConstantOp constOp = dyn_cast<mlir::ConstantOp>(defOp)) {
          mapping.map(defOp->getResult(0), b.clone(*defOp)->getResult(0));
        }
      }
    });

    clonedOps.push_back(b.clone(*sortedOps[i], mapping));
  }

  for (unsigned i = 0; i < sortedOps.size(); i++) {
    opToCallee[sortedOps[i]] = clonedOps[i];
  }

  // Set the scop_stmt attribute for identification at a later stage.
  // TODO: in the future maybe we could create a customized dialect, e.g., Scop,
  // that contains scop stmt FuncOp, e.g., ScopStmtOp.
  callee.setAttr(SCOP_STMT_ATTR_NAME, b.getUnitAttr());

  // Set the callee to be private for inlining.
  callee.setPrivate();

  return callee;
}

/// Create a caller to the callee right after the writeOp, which will be removed
/// later.
static mlir::CallOp createCaller(mlir::FuncOp callee,
                                 const SetVector<mlir::Value> &args,
                                 Operation *writeOp, OpBuilder &b) {
  // llvm::errs() << "Create caller for: " << callee.getName() << "\n";
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(writeOp);
  // writeOp->dump();

  return b.create<mlir::CallOp>(writeOp->getLoc(), callee,
                                ValueRange(args.getArrayRef()));
}

/// Remove those ops that are already in the callee, and not have uses by other
/// ops. We will first sort these ops topologically, and then remove them in a
/// reversed order.
static void removeExtractedOps(SetVector<Operation *> &opsToRemove) {
  opsToRemove = topologicalSort(opsToRemove);
  unsigned numOpsToRemove = opsToRemove.size();

  for (unsigned i = 0; i < numOpsToRemove; i++) {
    Operation *op = opsToRemove[numOpsToRemove - i - 1];
    // TODO: need to check if this should be allowed to happen.
    if (op->getUses().empty())
      op->erase();
  }
}

/// The main function that extracts scop statements as functions. Returns the
/// number of callees extracted from this function.
static unsigned extractScopStmt(mlir::FuncOp f, unsigned numCallees,
                                OpBuilder &b) {
  // First discover those write ops that will be the "terminator" of each scop
  // statement in the given function.
  SmallVector<Operation *, 8> writeOps;
  discoverMemWriteOps(f, writeOps);

  SetVector<Operation *> opsToRemove;
  // Map from an op in the original funcOp to which callee it would belong to.
  OpToCalleeMap opToCallee;
  CalleeToCallersMap calleeToCallers;

  unsigned numWriteOps = writeOps.size();

  // Use the top-level module to locate places for new functions insertion.
  mlir::ModuleOp m = cast<mlir::ModuleOp>(f.getParentOp());
  // A writeOp will result in a new caller/callee pair.
  for (unsigned i = 0; i < numWriteOps; i++) {
    SetVector<Operation *> ops;
    SetVector<mlir::Value> args;

    // Get all the ops inside a statement that corresponds to the current write
    // operation.
    Operation *writeOp = writeOps[i];
    getScopStmtOps(writeOp, ops, args, opToCallee, calleeToCallers, f, b);

    // Get the name of the callee. Should be in the form of "S<id>".
    CalleeName calleeName;
    getCalleeName(i + numCallees, calleeName);

    // Create the callee.
    mlir::FuncOp callee =
        createCallee(calleeName, ops, args, m, writeOp, opToCallee, b);
    // Create the caller.
    mlir::CallOp caller = createCaller(callee, args, writeOp, b);
    calleeToCallers[callee].insert(caller);
    // llvm::errs() << "Caller inserted:\n";
    // caller.dump();

    // All the ops that have been placed in the callee should be removed.
    opsToRemove.set_union(ops);
  }

  // Remove those extracted ops in the original function.
  removeExtractedOps(opsToRemove);

  return numWriteOps;
}

namespace {

class ExtractScopStmtPass
    : public mlir::PassWrapper<ExtractScopStmtPass,
                               OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    SmallVector<mlir::FuncOp, 4> funcs;
    m.walk([&](mlir::FuncOp f) { funcs.push_back(f); });

    unsigned numCallees = 0;
    for (mlir::FuncOp f : funcs) {
      numCallees += extractScopStmt(f, numCallees, b);
    }
  }
};

} // namespace

void polymer::registerExtractScopStmtPass() {
  PassRegistration<ExtractScopStmtPass>(
      "extract-scop-stmt", "Extract SCoP statements into functions.");
}

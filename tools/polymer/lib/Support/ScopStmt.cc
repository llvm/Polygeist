//===- ScopStmt.cc ----------------------------------------------*- C++ -*-===//
//
// This file declares the class ScopStmt.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/ScopStmt.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace mlir;
using namespace polymer;

namespace polymer {

class ScopStmtImpl {
public:
  using EnclosingOpList = SmallVector<Operation *, 8>;

  ScopStmtImpl(llvm::StringRef name, mlir::CallOp caller, mlir::FuncOp callee)
      : name(name), caller(caller), callee(callee) {}

  static std::unique_ptr<ScopStmtImpl> get(mlir::Operation *callerOp,
                                           mlir::Operation *calleeOp);

  /// A helper function that builds the domain constraints of the
  /// caller, and find and insert all enclosing for/if ops to enclosingOps.
  void initializeDomainAndEnclosingOps();

  void getArgsValueMapping(BlockAndValueMapping &argMap);

  /// Name of the callee, as well as the scop.stmt. It will also be the
  /// symbol in the OpenScop representation.
  llvm::StringRef name;
  /// The caller to the scop.stmt func.
  mlir::CallOp caller;
  /// The scop.stmt callee.
  mlir::FuncOp callee;
  /// The domain of the caller.
  FlatAffineValueConstraints domain;
  /// Enclosing for/if operations for the caller.
  EnclosingOpList enclosingOps;
};

} // namespace polymer

/// Create ScopStmtImpl from only the caller/callee pair.
std::unique_ptr<ScopStmtImpl> ScopStmtImpl::get(mlir::Operation *callerOp,
                                                mlir::Operation *calleeOp) {
  // We assume that the callerOp is of type mlir::CallOp, and the calleeOp is a
  // mlir::FuncOp. If not, these two cast lines will raise error.
  mlir::CallOp caller = cast<mlir::CallOp>(callerOp);
  mlir::FuncOp callee = cast<mlir::FuncOp>(calleeOp);
  llvm::StringRef name = caller.getCallee();

  // Create the stmt instance.
  auto stmt = std::make_unique<ScopStmtImpl>(name, caller, callee);

  // Initialize the domain constraints around the caller. The enclosing ops will
  // be figured out as well in this process.
  stmt->initializeDomainAndEnclosingOps();

  return stmt;
}

static BlockArgument findTopLevelBlockArgument(mlir::Value val) {
  if (val.isa<mlir::BlockArgument>())
    return val.cast<mlir::BlockArgument>();

  mlir::Operation *defOp = val.getDefiningOp();
  assert((defOp && isa<mlir::arith::IndexCastOp>(defOp)) &&
         "Only allow defOp of a parameter to be an IndexCast.");
  return findTopLevelBlockArgument(defOp->getOperand(0));
}

static void
promoteSymbolToTopLevel(mlir::Value val, FlatAffineValueConstraints &domain,
                        llvm::DenseMap<mlir::Value, mlir::Value> &symMap) {
  BlockArgument arg = findTopLevelBlockArgument(val);
  assert(isa<mlir::FuncOp>(arg.getOwner()->getParentOp()) &&
         "Found top-level argument should be a FuncOp argument.");
  // NOTE: This cannot pass since the found argument may not be of index type,
  // i.e., it will be index cast later.
  // assert(isValidSymbol(arg) &&
  //        "Found top-level argument should be a valid symbol.");

  unsigned int pos;
  assert(domain.findId(val, &pos) &&
         "Provided value should be in the given domain");
  domain.setValue(pos, arg);

  symMap[val] = arg;
}

void ScopStmtImpl::initializeDomainAndEnclosingOps() {
  // Extract the affine for/if ops enclosing the caller and insert them into the
  // enclosingOps list.
  getEnclosingAffineForAndIfOps(*caller, &enclosingOps);

  // The domain constraints can then be collected from the enclosing ops.
  assert(succeeded(getIndexSet(enclosingOps, &domain)));

  // Add additional indices that are in the top level block arguments.
  for (Value arg : caller->getOperands()) {
    if (!arg.getType().isIndex())
      continue;
    unsigned pos;
    if (domain.findId(arg, &pos))
      continue;

    domain.appendSymbolId(1);
    domain.dump();
    domain.setValue(domain.getNumDimAndSymbolIds() - 1, arg);
  }

  // Symbol values, which could be a BlockArgument, or the result of DimOp or
  // IndexCastOp, or even an affine.apply. Here we limit the cases to be either
  // BlockArgument or IndexCastOp, and if it is an IndexCastOp, the cast source
  // should be a top-level BlockArgument.
  SmallVector<mlir::Value, 8> symValues;
  llvm::DenseMap<mlir::Value, mlir::Value> symMap;
  domain.getValues(domain.getNumDimIds(), domain.getNumDimAndSymbolIds(),
                   &symValues);
  for (mlir::Value val : symValues)
    promoteSymbolToTopLevel(val, domain, symMap);
}

void ScopStmtImpl::getArgsValueMapping(BlockAndValueMapping &argMap) {
  auto callerArgs = caller.getArgOperands();
  auto calleeArgs = callee.getArguments();
  unsigned numArgs = callerArgs.size();

  argMap.clear();
  for (unsigned i = 0; i < numArgs; i++)
    argMap.map(calleeArgs[i], callerArgs[i]);
}

ScopStmt::ScopStmt(Operation *caller, Operation *callee)
    : impl{ScopStmtImpl::get(caller, callee)} {}

ScopStmt::~ScopStmt() = default;
ScopStmt::ScopStmt(ScopStmt &&) = default;
ScopStmt &ScopStmt::operator=(ScopStmt &&) = default;

FlatAffineValueConstraints *ScopStmt::getDomain() const {
  return &(impl->domain);
}

void ScopStmt::getEnclosingOps(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                               bool forOnly) const {
  for (mlir::Operation *op : impl->enclosingOps)
    if (!forOnly || isa<mlir::AffineForOp>(op))
      ops.push_back(op);
}

mlir::FuncOp ScopStmt::getCallee() const { return impl->callee; }
mlir::CallOp ScopStmt::getCaller() const { return impl->caller; }

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

void ScopStmt::getAccessMapAndMemRef(mlir::Operation *op,
                                     mlir::AffineValueMap *vMap,
                                     mlir::Value *memref) const {
  // Map from callee arguments to caller's. impl holds the callee and caller
  // instances.
  BlockAndValueMapping argMap;
  impl->getArgsValueMapping(argMap);

  // TODO: assert op is in the callee.
  MemRefAccess access(op);

  // Collect the access AffineValueMap that binds to operands in the callee.
  AffineValueMap aMap;
  access.getAccessMap(&aMap);

  // Replace its operands by what the caller uses.
  SmallVector<mlir::Value, 8> operands;
  for (mlir::Value operand : aMap.getOperands()) {
    mlir::Value origArg = findBlockArg(argMap.lookupOrDefault(operand));
    assert(origArg && "The original value cannot be found as a block argument "
                      "of the top function. Try -canonicalize.");
    assert(origArg != operand &&
           "The found original value shouldn't be the same as the operand.");

    operands.push_back(origArg);
  }

  // Set the access AffineValueMap.
  vMap->reset(aMap.getAffineMap(), operands);
  // Set the memref.
  *memref = argMap.lookup(access.memref);
}

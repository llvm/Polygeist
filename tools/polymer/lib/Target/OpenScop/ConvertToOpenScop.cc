//===- EmitOpenScop.cc ------------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for emitting OpenScop representation from
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/OpenScop.h"
#include "polymer/Transforms/ExtractScopStmt.h"

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
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "osl/osl.h"

#include <memory>

using namespace mlir;
using namespace llvm;
using namespace polymer;

#define DEBUG_TYPE "emit-openscop"

namespace {

/// Build OslScop from FuncOp.
class OslScopBuilder {
public:
  OslScopBuilder() {}

  /// Build a scop from a common FuncOp.
  std::unique_ptr<OslScop> build(mlir::FuncOp f);

private:
  /// Find all statements that calls a scop.stmt.
  void buildScopStmtMap(mlir::FuncOp f, OslScop::ScopStmtNames *scopStmtNames,
                        OslScop::ScopStmtMap *scopStmtMap) const;

  /// Build the scop context. The domain of each scop stmt will be updated, by
  /// merging and aligning its IDs with the context as well.
  void buildScopContext(OslScop *scop, OslScop::ScopStmtMap *scopStmtMap,
                        FlatAffineConstraints &ctx) const;
};

} // namespace

/// Build OslScop from a given FuncOp.
std::unique_ptr<OslScop> OslScopBuilder::build(mlir::FuncOp f) {

  /// Context constraints.
  FlatAffineConstraints ctx;

  // Initialize a new Scop per FuncOp. The osl_scop object within it will be
  // created. It doesn't contain any fields, and this may incur some problems,
  // which the validate function won't discover, e.g., no context will cause
  // segfault when printing scop. Please don't just return this object.
  auto scop = std::make_unique<OslScop>();
  // Mapping between scop stmt names and their caller/callee op pairs.
  OslScop::ScopStmtMap *scopStmtMap = scop->getScopStmtMap();
  auto *scopStmtNames = scop->getScopStmtNames();

  // Find all caller/callee pairs in which the callee has the attribute of name
  // SCOP_STMT_ATTR_NAME.
  buildScopStmtMap(f, scopStmtNames, scopStmtMap);
  if (scopStmtMap->empty())
    return nullptr;

  // Build context in it.
  buildScopContext(scop.get(), scopStmtMap, ctx);

  // Counter for the statement inserted.
  unsigned stmtId = 0;
  for (const auto &scopStmtName : *scopStmtNames) {
    llvm::errs() << scopStmtName << "\n";
    const ScopStmt &stmt = scopStmtMap->find(scopStmtName)->second;
    // Collet the domain
    FlatAffineConstraints *domain = stmt.getDomain();
    // Collect the enclosing ops.
    llvm::SmallVector<mlir::Operation *, 8> enclosingOps;
    stmt.getEnclosingOps(enclosingOps);
    // Get the callee.
    mlir::FuncOp callee = stmt.getCallee();

    // Create a statement in OslScop and setup relations in it.
    scop->createStatement();
    scop->addDomainRelation(stmtId, *domain);
    scop->addScatteringRelation(stmtId, *domain, enclosingOps);
    callee.walk([&](mlir::Operation *op) {
      if (isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op)) {
        bool isRead = isa<mlir::AffineReadOpInterface>(op);
        AffineValueMap vMap;
        mlir::Value memref;

        stmt.getAccessMapAndMemRef(op, &vMap, &memref);
        scop->addAccessRelation(stmtId, isRead, memref, vMap, *domain);
      }
    });

    stmtId++;
  }
  // osl_scop_print(stderr, scop->get());

  // Setup the symbol table within the OslScop, which builds the mapping from
  // mlir::Value to their names in the OpenScop representation, and maps them
  // backward.
  scop->initializeSymbolTable(f, &ctx);

  // Insert body extension.
  for (unsigned stmtId = 0; stmtId < scopStmtNames->size(); stmtId++) {
    const ScopStmt &stmt = scopStmtMap->find(scopStmtNames->at(stmtId))->second;
    scop->addBodyExtension(stmtId, stmt);
  }

  // Additionally, setup the name of the function in the comment.
  std::string funcName(f.getName());
  scop->addExtensionGeneric("comment", funcName);

  // osl_scop_print(stderr, scop->get());
  assert(scop->validate() && "The scop object created cannot be validated.");
  return scop;
}

/// Find all statements that calls a scop.stmt.
void OslScopBuilder::buildScopStmtMap(mlir::FuncOp f,
                                      OslScop::ScopStmtNames *scopStmtNames,
                                      OslScop::ScopStmtMap *scopStmtMap) const {
  mlir::ModuleOp m = cast<mlir::ModuleOp>(f.getParentOp());

  f.walk([&](mlir::Operation *op) {
    if (mlir::CallOp caller = dyn_cast<mlir::CallOp>(op)) {
      llvm::StringRef calleeName = caller.getCallee();
      mlir::FuncOp callee = m.lookupSymbol<mlir::FuncOp>(calleeName);

      // If the callee is of scop.stmt, we create a new instance in the map
      if (callee.getAttr(SCOP_STMT_ATTR_NAME)) {
        scopStmtNames->push_back(std::string(calleeName));
        scopStmtMap->insert(
            std::make_pair(calleeName, ScopStmt(caller, callee)));
      }
    }
  });
}

void OslScopBuilder::buildScopContext(OslScop *scop,
                                      OslScop::ScopStmtMap *scopStmtMap,
                                      FlatAffineConstraints &ctx) const {
  ctx.reset();

  // Union with the domains of all Scop statements. We first merge and align the
  // IDs of the context and the domain of the scop statement, and then append
  // the constraints from the domain to the context. Note that we don't want to
  // mess up with the original domain at this point. Trivial redundant
  // constraints will be removed.
  for (const auto &it : *scopStmtMap) {
    FlatAffineConstraints *domain = it.second.getDomain();
    FlatAffineConstraints cst(*domain);

    ctx.mergeAndAlignIdsWithOther(0, &cst);
    ctx.append(cst);
    ctx.removeRedundantConstraints();
  }
  ctx.dump();

  // Then, create the single context relation in scop.
  scop->addContextRelation(ctx);

  // Finally, given that ctx has all the parameters in it, we will make sure
  // that each domain is aligned with them, i.e., every domain has the same
  // parameter columns (Values & order).
  SmallVector<mlir::Value, 8> symValues;
  ctx.getIdValues(ctx.getNumDimIds(), ctx.getNumDimAndSymbolIds(), &symValues);

  for (const auto &it : *scopStmtMap) {
    FlatAffineConstraints *domain = it.second.getDomain();

    // SmallVector<mlir::Value, 8> domSymbols;
    // domain->getIdValues(domain->getNumDimIds(),
    // domain->getNumDimAndSymbolIds(),
    //                     &domSymbols);
    // for (auto sym : domSymbols)
    //   sym.dump();

    for (unsigned i = 0; i < ctx.getNumSymbolIds(); i++) {
      mlir::Value sym = symValues[i];
      unsigned pos;
      if (domain->findId(sym, &pos)) {
        if (pos != i + domain->getNumDimIds())
          domain->swapId(i + domain->getNumDimIds(), pos);
      } else {
        domain->addSymbolId(i, sym);
      }
    }

    // domain->getIdValues(domain->getNumDimIds(),
    // domain->getNumDimAndSymbolIds(),
    //                     &domSymbols);
    // for (auto sym : domSymbols)
    //   sym.dump();
  }
}

std::unique_ptr<OslScop>
polymer::createOpenScopFromFuncOp(mlir::FuncOp f, OslSymbolTable &symTable) {
  return OslScopBuilder().build(f);
}

namespace {

/// This class maintains the state of a working emitter.
class OpenScopEmitterState {
public:
  explicit OpenScopEmitterState(raw_ostream &os) : os(os) {}

  /// The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIdent = 0; // TODO: may not need this.

private:
  OpenScopEmitterState(const OpenScopEmitterState &) = delete;
  void operator=(const OpenScopEmitterState &) = delete;
};

/// Base class for various OpenScop emitters.
class OpenScopEmitterBase {
public:
  explicit OpenScopEmitterBase(OpenScopEmitterState &state)
      : state(state), os(state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitOpError(message);
  }

  /// All of the mutable state we are maintaining.
  OpenScopEmitterState &state;

  /// The stream to emit to.
  raw_ostream &os;

private:
  OpenScopEmitterBase(const OpenScopEmitterBase &) = delete;
  void operator=(const OpenScopEmitterBase &) = delete;
};

/// Emit OpenScop representation from an MLIR module.
class ModuleEmitter : public OpenScopEmitterBase {
public:
  explicit ModuleEmitter(OpenScopEmitterState &state)
      : OpenScopEmitterBase(state) {}

  /// Emit OpenScop definitions for all functions in the given module.
  void emitMLIRModule(ModuleOp module,
                      llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops);

private:
  /// Emit a OpenScop definition for a single function.
  LogicalResult
  emitFuncOp(FuncOp func,
             llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops);
};

LogicalResult ModuleEmitter::emitFuncOp(
    mlir::FuncOp func, llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops) {
  OslSymbolTable symTable;
  auto scop = createOpenScopFromFuncOp(func, symTable);
  if (scop)
    scops.push_back(std::move(scop));
  return success();
}

/// The entry function to the current OpenScop emitter.
void ModuleEmitter::emitMLIRModule(
    ModuleOp module, llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops) {
  // Emit a single OpenScop definition for each function.
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<mlir::FuncOp>(op)) {
      // Will only look at functions that are not attributed as scop.stmt
      if (func.getAttr(SCOP_STMT_ATTR_NAME))
        continue;
      if (failed(emitFuncOp(func, scops))) {
        state.encounteredError = true;
        return;
      }
    }
  }
}
} // namespace

/// TODO: should decouple emitter and openscop builder.
mlir::LogicalResult polymer::translateModuleToOpenScop(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops,
    llvm::raw_ostream &os) {
  OpenScopEmitterState state(os);
  ModuleEmitter(state).emitMLIRModule(module, scops);

  return success();
}

static LogicalResult emitOpenScop(ModuleOp module, llvm::raw_ostream &os) {
  llvm::SmallVector<std::unique_ptr<OslScop>, 8> scops;

  if (failed(translateModuleToOpenScop(module, scops, os)))
    return failure();

  for (auto &scop : scops)
    scop->print();

  return success();
}

void polymer::registerToOpenScopTranslation() {
  static TranslateFromMLIRRegistration toOpenScop("export-scop", emitOpenScop);
}

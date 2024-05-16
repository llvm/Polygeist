//===- EmitOpenScop.cc ------------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for emitting OpenScop representation from
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Visitors.h"
#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/OpenScop.h"
#include "polymer/Transforms/ExtractScopStmt.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "osl/osl.h"

#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::func;
using namespace llvm;
using namespace polymer;

#define DEBUG_TYPE "oslscop"

namespace {

/// Build OslScop from FuncOp.
class OslScopBuilder {
public:
  OslScopBuilder() {}

  /// Build a scop from a common FuncOp.
  std::unique_ptr<OslScop> build(mlir::func::FuncOp f);

private:
  /// Find all statements that calls a scop.stmt.
  void buildScopStmtMap(mlir::func::FuncOp f,
                        OslScop::ScopStmtNames *scopStmtNames,
                        OslScop::ScopStmtMap *scopStmtMap) const;

  /// Build the scop context. The domain of each scop stmt will be updated, by
  /// merging and aligning its IDs with the context as well.
  void buildScopContext(OslScop *scop, OslScop::ScopStmtMap *scopStmtMap,
                        affine::FlatAffineValueConstraints &ctx) const;
};

} // namespace

/// Sometimes the domain generated might be malformed. It is always better to
/// inform this at an early stage.
static void sanityCheckDomain(affine::FlatAffineValueConstraints &dom) {
  if (dom.isEmpty()) {
    llvm::errs() << "A domain is found to be empty!";
    dom.dump();
  }
}

/// Build OslScop from a given FuncOp.
std::unique_ptr<OslScop> OslScopBuilder::build(mlir::func::FuncOp f) {

  /// Context constraints.
  affine::FlatAffineValueConstraints ctx;

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
    const ScopStmt &stmt = scopStmtMap->find(scopStmtName)->second;
    LLVM_DEBUG({
      dbgs() << "Adding relations to statement: \n";
      stmt.getCaller().dump();
    });

    // Collet the domain
    affine::FlatAffineValueConstraints domain = *stmt.getDomain();
    sanityCheckDomain(domain);

    LLVM_DEBUG({
      dbgs() << "Domain:\n";
      domain.dump();
    });

    // Collect the enclosing ops.
    llvm::SmallVector<mlir::Operation *, 8> enclosingOps;
    stmt.getEnclosingOps(enclosingOps);
    // Get the callee.
    mlir::func::FuncOp callee = stmt.getCallee();

    LLVM_DEBUG({
      dbgs() << "Callee:\n";
      callee.dump();
    });

    // Create a statement in OslScop and setup relations in it.
    scop->createStatement();
    scop->addDomainRelation(stmtId, domain);
    scop->addScatteringRelation(stmtId, domain, enclosingOps);
    auto res = callee.walk([&](mlir::Operation *op) {
      if (isa<mlir::affine::AffineReadOpInterface>(op) ||
          isa<mlir::affine::AffineWriteOpInterface>(op)) {
        LLVM_DEBUG(dbgs() << "Creating access relation for: " << *op << '\n');

        bool isRead = isa<mlir::affine::AffineReadOpInterface>(op);
        affine::AffineValueMap vMap;
        mlir::Value memref;

        stmt.getAccessMapAndMemRef(op, &vMap, &memref);
        if (scop->addAccessRelation(stmtId, isRead, memref, vMap, domain)
                .failed())
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted())
      return nullptr;

    stmtId++;
  }

  // Setup the symbol table within the OslScop, which builds the mapping from
  // mlir::Value to their names in the OpenScop representation, and maps them
  // backward.
  scop->initializeSymbolTable(f, &ctx);

  // Insert body extension.
  for (unsigned stmtId = 0; stmtId < scopStmtNames->size(); stmtId++) {
    const ScopStmt &stmt = scopStmtMap->find(scopStmtNames->at(stmtId))->second;
    scop->addBodyExtension(stmtId, stmt);
  }
  auto res = scop->validate();
  assert(res && "The scop object created cannot be validated.");

  // Additionally, setup the name of the function in the comment.
  std::string funcName(f.getName());
  scop->addExtensionGeneric("comment", funcName);

  res = scop->validate();
  assert(res && "The scop object created cannot be validated.");

  return scop;
}

/// Find all statements that calls a scop.stmt.
void OslScopBuilder::buildScopStmtMap(mlir::func::FuncOp f,
                                      OslScop::ScopStmtNames *scopStmtNames,
                                      OslScop::ScopStmtMap *scopStmtMap) const {
  mlir::ModuleOp m = cast<mlir::ModuleOp>(f->getParentOp());

  f.walk([&](mlir::Operation *op) {
    if (mlir::func::CallOp caller = dyn_cast<mlir::func::CallOp>(op)) {
      llvm::StringRef calleeName = caller.getCallee();
      mlir::func::FuncOp callee =
          m.lookupSymbol<mlir::func::FuncOp>(calleeName);

      // If the callee is of scop.stmt, we create a new instance in the map
      if (callee->getAttr(SCOP_STMT_ATTR_NAME)) {
        scopStmtNames->push_back(std::string(calleeName));
        scopStmtMap->insert(
            std::make_pair(calleeName, ScopStmt(caller, callee)));
      }
    }
  });
}

void OslScopBuilder::buildScopContext(
    OslScop *scop, OslScop::ScopStmtMap *scopStmtMap,
    affine::FlatAffineValueConstraints &ctx) const {
  LLVM_DEBUG(dbgs() << "--- Building SCoP context ...\n");

  // First initialize the symbols of the ctx by the order of arg number.
  // This simply aims to make mergeAndAlignVarsWithOthers work.
  SmallVector<Value> symbols;
  for (const auto &it : *scopStmtMap) {
    auto domain = it.second.getDomain();
    SmallVector<Value> syms;
    domain->getValues(domain->getNumDimVars(), domain->getNumDimAndSymbolVars(),
                      &syms);

    for (Value sym : syms) {
      // Find the insertion position.
      auto it = symbols.begin();
      while (it != symbols.end()) {
        auto lhs = it->cast<BlockArgument>();
        auto rhs = sym.cast<BlockArgument>();
        if (lhs.getArgNumber() >= rhs.getArgNumber())
          break;
        ++it;
      }
      if (it == symbols.end() || *it != sym)
        symbols.insert(it, sym);
    }
  }

  ctx = affine::FlatAffineValueConstraints(/*numDims=*/0,
                                           /*numSymbols=*/symbols.size());
  ctx.setValues(0, symbols.size(), symbols);

  // Union with the domains of all Scop statements. We first merge and align the
  // IDs of the context and the domain of the scop statement, and then append
  // the constraints from the domain to the context. Note that we don't want to
  // mess up with the original domain at this point. Trivial redundant
  // constraints will be removed.
  for (const auto &it : *scopStmtMap) {
    affine::FlatAffineValueConstraints *domain = it.second.getDomain();
    affine::FlatAffineValueConstraints cst(*domain);

    LLVM_DEBUG(dbgs() << "Statement:\n");
    LLVM_DEBUG(it.second.getCaller().dump());
    LLVM_DEBUG(it.second.getCallee().dump());
    LLVM_DEBUG(dbgs() << "Target domain: \n");
    LLVM_DEBUG(domain->dump());

    LLVM_DEBUG({
      dbgs() << "Domain values: \n";
      SmallVector<Value> values;
      domain->getValues(0, domain->getNumDimAndSymbolVars(), &values);
      for (Value value : values)
        dbgs() << " * " << value << '\n';
    });

    ctx.mergeAndAlignVarsWithOther(0, &cst);
    ctx.append(cst);
    ctx.removeRedundantConstraints();

    LLVM_DEBUG(dbgs() << "Updated context: \n");
    LLVM_DEBUG(ctx.dump());

    LLVM_DEBUG({
      dbgs() << "Context values: \n";
      SmallVector<Value> values;
      ctx.getValues(0, ctx.getNumDimAndSymbolVars(), &values);
      for (Value value : values)
        dbgs() << " * " << value << '\n';
    });
  }

  // Then, create the single context relation in scop.
  scop->addContextRelation(ctx);

  // Finally, given that ctx has all the parameters in it, we will make sure
  // that each domain is aligned with them, i.e., every domain has the same
  // parameter columns (Values & order).
  SmallVector<mlir::Value, 8> symValues;
  ctx.getValues(ctx.getNumDimVars(), ctx.getNumDimAndSymbolVars(), &symValues);

  // Add and align domain SYMBOL columns.
  for (const auto &it : *scopStmtMap) {
    affine::FlatAffineValueConstraints *domain = it.second.getDomain();
    // For any symbol missing in the domain, add them directly to the end.
    for (unsigned i = 0; i < ctx.getNumSymbolVars(); ++i) {
      unsigned pos;
      if (!domain->findVar(symValues[i], &pos)) // insert to the back
        domain->appendSymbolVar(symValues[i]);
      else
        LLVM_DEBUG(dbgs() << "Found " << symValues[i] << '\n');
    }

    // Then do the aligning.
    LLVM_DEBUG(domain->dump());
    for (unsigned i = 0; i < ctx.getNumSymbolVars(); i++) {
      mlir::Value sym = symValues[i];
      unsigned pos;
      domain->findVar(sym, &pos);

      unsigned posAsCtx = i + domain->getNumDimVars();
      LLVM_DEBUG(dbgs() << "Swapping " << posAsCtx << " " << pos << "\n");
      if (pos != posAsCtx)
        domain->swapVar(posAsCtx, pos);
    }

    // for (unsigned i = 0; i < ctx.getNumSymbolVars(); i++) {
    //   mlir::Value sym = symValues[i];
    //   unsigned pos;
    //   // If the symbol can be found in the domain, we put it in the same
    //   // position as the ctx.
    //   if (domain->findVar(sym, &pos)) {
    //     if (pos != i + domain->getNumDimVars())
    //       domain->swapVar(i + domain->getNumDimVars(), pos);
    //   } else {
    //     domain->insertSymbolId(i, sym);
    //   }
    // }
  }
}

std::unique_ptr<OslScop>
polymer::createOpenScopFromFuncOp(mlir::func::FuncOp f,
                                  OslSymbolTable &symTable) {
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
    mlir::func::FuncOp func,
    llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops) {
  OslSymbolTable symTable;
  auto scop = createOpenScopFromFuncOp(func, symTable);
  if (scop) {
    scops.push_back(std::move(scop));
    return success();
  }
  return failure();
}

/// The entry function to the current OpenScop emitter.
void ModuleEmitter::emitMLIRModule(
    ModuleOp module, llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops) {
  // Emit a single OpenScop definition for each function.
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<mlir::func::FuncOp>(op)) {
      // Will only look at functions that are not attributed as scop.stmt
      if (func->getAttr(SCOP_STMT_ATTR_NAME))
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
  ::ModuleEmitter(state).emitMLIRModule(module, scops);

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
  static TranslateFromMLIRRegistration toOpenScop("export-scop", "Export SCOP",
                                                  emitOpenScop);
}

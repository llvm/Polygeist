//===- OslScop.cc -----------------------------------------------*- C++ -*-===//
//
// This file implements the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScop.h"
#include "polymer/Support/ScatteringUtils.h"
#include "polymer/Support/ScopStmt.h"

#include "osl/osl.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Support/LogicalResult.h"

#include <vector>

using namespace polymer;
using namespace mlir;
using namespace llvm;

/// Create osl_vector from a STL vector. Since the input vector is of type
/// int64_t, we can safely assume the osl_vector we will generate has 64 bits
/// precision. The input vector doesn't contain the e/i indicator.
static void getOslVector(bool isEq, llvm::ArrayRef<int64_t> vec,
                         osl_vector_p *oslVec) {
  *oslVec = osl_vector_pmalloc(64, vec.size() + 1);

  // Set the e/i field.
  osl_int_t val;
  val.dp = isEq ? 0 : 1;
  (*oslVec)->v[0] = val;

  // Set the rest of the vector.
  for (int i = 0, e = vec.size(); i < e; i++) {
    osl_int_t val;
    val.dp = vec[i];
    (*oslVec)->v[i + 1] = val;
  }
}

/// Get the statement given by its index.
static osl_statement_p getOslStatement(osl_scop_p scop, unsigned index) {
  osl_statement_p stmt = scop->statement;
  for (unsigned i = 0; i <= index; i++) {
    // stmt accessed in the linked list before counting to index should not be
    // NULL.
    assert(stmt && "index exceeds the range of statements in scop.");
    if (i == index)
      return stmt;
    stmt = stmt->next;
  }
  return nullptr;
}

OslScop::OslScop() {
  scatTreeRoot = std::make_unique<ScatTreeNode>();

  scop = osl_scop_malloc();

  // Initialize string buffer for language.
  OSL_strdup(scop->language, "C");

  // Use the default interface registry
  osl_interface_p registry = osl_interface_get_default_registry();
  scop->registry = osl_interface_clone(registry);
}

OslScop::OslScop(osl_scop *scop)
    : scop(scop), scatTreeRoot{std::make_unique<ScatTreeNode>()} {}

OslScop::~OslScop() { osl_scop_free(scop); }

void OslScop::print() { osl_scop_print(stdout, scop); }

bool OslScop::validate() {
  // TODO: do we need to check the scoplib compatibility?
  return osl_scop_integrity_check(scop);
}

void OslScop::createStatement() {
  osl_statement_p stmt = osl_statement_malloc();
  osl_statement_add(&(scop->statement), stmt);
}

void OslScop::addRelation(int target, int type, int numRows, int numCols,
                          int numOutputDims, int numInputDims, int numLocalDims,
                          int numParams, llvm::ArrayRef<int64_t> eqs,
                          llvm::ArrayRef<int64_t> inEqs) {
  // Here we preset the precision to 64.
  osl_relation_p rel = osl_relation_pmalloc(64, numRows, numCols);
  rel->type = type;
  rel->nb_output_dims = numOutputDims;
  rel->nb_input_dims = numInputDims;
  rel->nb_local_dims = numLocalDims;
  rel->nb_parameters = numParams;

  // The number of columns in the given equalities and inequalities, which is
  // one less than the number of columns in the OSL representation (missing e/i
  // indicator).
  size_t numColsInEqs = numCols - 1;

  assert(eqs.size() % numColsInEqs == 0 &&
         "Number of elements in the eqs should be an integer multiply of "
         "numColsInEqs");
  size_t numEqs = eqs.size() / numColsInEqs;

  // Replace those allocated vector elements in rel.
  for (int i = 0; i < numRows; i++) {
    osl_vector_p vec;

    if (i >= static_cast<int>(numEqs)) {
      auto inEq = llvm::ArrayRef<int64_t>(&inEqs[(i - numEqs) * numColsInEqs],
                                          numColsInEqs);
      getOslVector(false, inEq, &vec);
    } else {
      auto eq = llvm::ArrayRef<int64_t>(&eqs[i * numColsInEqs], numColsInEqs);
      getOslVector(true, eq, &vec);
    }

    // Replace the vector content of the i-th row by the contents in
    // constraints.
    osl_relation_replace_vector(rel, vec, i);
  }

  // Append the newly created relation to a target linked list, or simply set it
  // to a relation pointer, which is indicated by the target argument.
  if (target == 0) {
    // Simply assign the newly created relation to the context field.
    scop->context = rel;
  } else {
    // Get the pointer to the statement.
    osl_statement_p stmt = getOslStatement(scop, target - 1);

    // Depending on the type of the relation, we decide which field of the
    // statement we should set.
    if (type == OSL_TYPE_DOMAIN) {
      stmt->domain = rel;
    } else if (type == OSL_TYPE_SCATTERING) {
      stmt->scattering = rel;
    } else if (type == OSL_TYPE_ACCESS || type == OSL_TYPE_WRITE ||
               type == OSL_TYPE_READ) {
      osl_relation_list_p relList = osl_relation_list_malloc();
      relList->elt = rel;
      osl_relation_list_add(&(stmt->access), relList);
    }
  }
}

void OslScop::addContextRelation(FlatAffineConstraints cst) {
  // Project out the dim IDs in the context with only the symbol IDs left.
  SmallVector<mlir::Value, 8> dimValues;
  cst.getIdValues(0, cst.getNumDimIds(), &dimValues);
  for (mlir::Value dimValue : dimValues)
    cst.projectOut(dimValue);
  cst.removeIndependentConstraints(0, cst.getNumDimAndSymbolIds());

  SmallVector<int64_t, 8> eqs, inEqs;
  // createConstraintRows(cst, eqs);
  // createConstraintRows(cst, inEqs, /*isEq=*/false);

  unsigned numCols = 2 + cst.getNumSymbolIds();
  unsigned numEntries = inEqs.size() + eqs.size();
  assert(numEntries % (numCols - 1) == 0 &&
         "Total number of entries should be divisible by the number of columns "
         "(excluding e/i)");

  unsigned numRows = (inEqs.size() + eqs.size()) / (numCols - 1);
  // Create the context relation.
  addRelation(0, OSL_TYPE_CONTEXT, numRows, numCols, 0, 0, 0,
              cst.getNumSymbolIds(), eqs, inEqs);
}

void OslScop::addDomainRelation(int stmtId, FlatAffineConstraints &cst) {
  SmallVector<int64_t, 8> eqs, inEqs;
  createConstraintRows(cst, eqs);
  createConstraintRows(cst, inEqs, /*isEq=*/false);

  addRelation(stmtId + 1, OSL_TYPE_DOMAIN, cst.getNumConstraints(),
              cst.getNumCols() + 1, cst.getNumDimIds(), 0, cst.getNumLocalIds(),
              cst.getNumSymbolIds(), eqs, inEqs);
}

void OslScop::addScatteringRelation(int stmtId,
                                    mlir::FlatAffineConstraints &cst,
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

  // Create equalities and inequalities.
  std::vector<int64_t> eqs, inEqs;

  // Initialize contents for equalities.
  eqs.resize(numScatEqs * (numScatCols - 1));
  for (unsigned j = 0; j < numScatEqs; j++) {

    // Initializing scattering dimensions by setting the diagonal to -1.
    for (unsigned k = 0; k < numScatEqs; k++)
      eqs[j * (numScatCols - 1) + k] = -static_cast<int64_t>(k == j);

    // Relating the loop IVs to the scattering dimensions. If it's the odd
    // equality, set its scattering dimension to the loop IV; otherwise, it's
    // scattering dimension will be set in the following constant section.
    for (unsigned k = 0; k < cst.getNumDimIds(); k++)
      eqs[j * (numScatCols - 1) + k + numScatEqs] =
          (j % 2) ? (k == (j / 2)) : 0;

    // TODO: consider the parameters that may appear in the scattering
    // dimension.
    for (unsigned k = 0; k < cst.getNumLocalIds() + cst.getNumSymbolIds(); k++)
      eqs[j * (numScatCols - 1) + k + numScatEqs + cst.getNumDimIds()] = 0;

    // Relating the constants (the last column) to the scattering dimensions.
    eqs[j * (numScatCols - 1) + numScatCols - 2] = (j % 2) ? 0 : scats[j / 2];
  }

  // Then put them into the scop as a SCATTERING relation.
  addRelation(stmtId + 1, OSL_TYPE_SCATTERING, numScatEqs, numScatCols,
              numScatEqs, cst.getNumDimIds(), cst.getNumLocalIds(),
              cst.getNumSymbolIds(), eqs, inEqs);
}

void OslScop::addAccessRelation(int stmtId, bool isRead, mlir::Value memref,
                                AffineValueMap &vMap,
                                FlatAffineConstraints &domain) {
  FlatAffineConstraints cst;
  // Insert the address dims and put constraints in it.
  createAccessRelationConstraints(vMap, cst, domain);

  // Create a new dim of memref and set its value to its corresponding ID.
  memRefIdMap.try_emplace(memref, memRefIdMap.size() + 1);
  cst.addDimId(0, memref);
  cst.setIdToConstant(0, memRefIdMap[memref]);

  SmallVector<int64_t, 8> eqs, inEqs;
  createConstraintRows(cst, eqs);
  // createConstraintRows(cst, inEqs, /*isEq=*/false);
  for (unsigned i = 0; i < eqs.size(); i++)
    eqs[i] = -eqs[i];

  // Then put them into the scop as an ACCESS relation.
  unsigned numOutputDims = cst.getNumConstraints();
  unsigned numInputDims = cst.getNumDimIds() - numOutputDims;
  addRelation(stmtId + 1, isRead ? OSL_TYPE_READ : OSL_TYPE_WRITE,
              cst.getNumConstraints(), cst.getNumCols() + 1, numOutputDims,
              numInputDims, cst.getNumLocalIds(), cst.getNumSymbolIds(), eqs,
              inEqs);
}

void OslScop::addGeneric(int target, llvm::StringRef tag,
                         llvm::StringRef content) {

  osl_generic_p generic = osl_generic_malloc();

  // Add interface.
  osl_interface_p interface = osl_interface_lookup(scop->registry, tag.data());
  generic->interface = osl_interface_nclone(interface, 1);

  // Add content
  char *buf;
  OSL_malloc(buf, char *, (content.size() * sizeof(char) + 10));
  OSL_strdup(buf, content.data());
  generic->data = interface->sread(&buf);

  if (target == 0) {
    // Add to Scop extension.
    osl_generic_add(&(scop->extension), generic);
  } else if (target == -1) {
    // Add to Scop parameters.
    osl_generic_add(&(scop->parameters), generic);
  } else {
    // Add to statement.
    osl_statement_p stmt = getOslStatement(scop, target - 1);
    osl_generic_add(&(stmt->extension), generic);
  }
}

void OslScop::addExtensionGeneric(llvm::StringRef tag,
                                  llvm::StringRef content) {
  addGeneric(0, tag, content);
}

void OslScop::addParametersGeneric(llvm::StringRef tag,
                                   llvm::StringRef content) {
  addGeneric(-1, tag, content);
}

void OslScop::addStatementGeneric(int stmtId, llvm::StringRef tag,
                                  llvm::StringRef content) {
  addGeneric(stmtId + 1, tag, content);
}

/// We determine whether the name refers to a symbol by looking up the parameter
/// list of the scop.
bool OslScop::isSymbol(llvm::StringRef name) {
  osl_generic_p parameters = scop->parameters;
  if (!parameters)
    return false;

  assert(parameters->next == NULL &&
         "Should only exist one parameters generic object.");
  assert(osl_generic_has_URI(parameters, OSL_URI_STRINGS) &&
         "Parameters should be of strings interface.");

  // TODO: cache this result, otherwise we need O(N) each time calling this API.
  osl_strings_p parameterNames =
      reinterpret_cast<osl_strings_p>(parameters->data);
  unsigned numParameters = osl_strings_size(parameterNames);

  for (unsigned i = 0; i < numParameters; i++)
    if (name.equals(parameterNames->string[i]))
      return true;

  return false;
}

LogicalResult OslScop::getStatement(unsigned index,
                                    osl_statement **stmt) const {
  // TODO: cache all the statements.
  osl_statement_p curr = scop->statement;
  if (!curr)
    return failure();

  for (unsigned i = 0; i < index; i++) {
    curr = curr->next;
    if (!curr)
      return failure();
  }

  *stmt = curr;
  return success();
}

unsigned OslScop::getNumStatements() const {
  return osl_statement_number(scop->statement);
}

osl_generic_p OslScop::getExtension(llvm::StringRef tag) const {
  osl_generic_p ext = scop->extension;
  osl_interface_p interface = osl_interface_lookup(scop->registry, tag.data());

  while (ext) {
    if (osl_interface_equal(ext->interface, interface))
      return ext;
    ext = ext->next;
  }

  return nullptr;
}

void OslScop::addParameterNames() {
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

void OslScop::addScatnamesExtension() {
  std::string body;
  llvm::raw_string_ostream ss(body);

  unsigned numScatnames = scatTreeRoot->getDepth();
  numScatnames = (numScatnames - 2) * 2 + 1;
  for (unsigned i = 0; i < numScatnames; i++)
    ss << formatv("c{0}", i + 1) << " ";

  addExtensionGeneric("scatnames", body);
}

void OslScop::addArraysExtension() {
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

void OslScop::addBodyExtension(int stmtId, const ScopStmt &stmt) {
  std::string body;
  llvm::raw_string_ostream ss(body);

  SmallVector<mlir::Operation *, 8> forOps;
  stmt.getEnclosingOps(forOps, /*forOnly=*/true);

  unsigned numIVs = forOps.size();
  ss << numIVs << " ";

  llvm::DenseMap<mlir::Value, unsigned> ivToId;
  for (unsigned i = 0; i < numIVs; i++) {
    mlir::AffineForOp forOp = cast<mlir::AffineForOp>(forOps[i]);
    // forOp.dump();
    ivToId[forOp.getInductionVar()] = i;
  }

  for (unsigned i = 0; i < numIVs; i++)
    ss << "i" << i << " ";

  mlir::CallOp caller = stmt.getCaller();
  mlir::FuncOp callee = stmt.getCallee();
  ss << "\n" << callee.getName() << "(";

  SmallVector<std::string, 8> ivs;
  SetVector<unsigned> visited;
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

void OslScop::initializeSymbolTable(mlir::FuncOp f,
                                    FlatAffineConstraints *cst) {
  symbolTable.clear();

  unsigned numDimIds = cst->getNumDimIds();
  unsigned numSymbolIds = cst->getNumDimAndSymbolIds() - numDimIds;

  SmallVector<mlir::Value, 8> dimValues, symbolValues;
  cst->getIdValues(0, numDimIds, &dimValues);
  cst->getIdValues(numDimIds, cst->getNumDimAndSymbolIds(), &symbolValues);

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

bool OslScop::isParameterSymbol(llvm::StringRef name) const {
  return name.startswith("P");
}

bool OslScop::isDimSymbol(llvm::StringRef name) const {
  return name.startswith("i");
}

bool OslScop::isArraySymbol(llvm::StringRef name) const {
  return name.startswith("A");
}

bool OslScop::isConstantSymbol(llvm::StringRef name) const {
  return name.startswith("C");
}

void OslScop::createConstraintRows(FlatAffineConstraints &cst,
                                   SmallVectorImpl<int64_t> &rows, bool isEq) {
  unsigned numRows = isEq ? cst.getNumEqualities() : cst.getNumInequalities();
  unsigned numDimIds = cst.getNumDimIds();
  unsigned numLocalIds = cst.getNumLocalIds();
  unsigned numSymbolIds = cst.getNumSymbolIds();

  for (unsigned i = 0; i < numRows; i++) {
    // Get the row based on isEq.
    auto row = isEq ? cst.getEquality(i) : cst.getInequality(i);

    unsigned numCols = row.size();
    if (i == 0)
      rows.resize(numRows * numCols);

    // Dims stay at the same positions.
    for (unsigned j = 0; j < numDimIds; j++)
      rows[i * numCols + j] = row[j];
    // Output local ids before symbols.
    for (unsigned j = 0; j < numLocalIds; j++)
      rows[i * numCols + j + numDimIds] = row[j + numDimIds + numSymbolIds];
    // Output symbols in the end.
    for (unsigned j = 0; j < numSymbolIds; j++)
      rows[i * numCols + j + numDimIds + numLocalIds] = row[j + numDimIds];
    // Finally outputs the constant.
    rows[i * numCols + numCols - 1] = row[numCols - 1];
  }
}

void OslScop::createAccessRelationConstraints(
    mlir::AffineValueMap &vMap, mlir::FlatAffineConstraints &cst,
    mlir::FlatAffineConstraints &domain) {
  cst.reset();
  cst.mergeAndAlignIdsWithOther(0, &domain);

  SmallVector<mlir::Value, 8> idValues;
  domain.getAllIdValues(&idValues);
  SetVector<mlir::Value> idValueSet;
  for (auto val : idValues)
    idValueSet.insert(val);

  for (auto operand : vMap.getOperands())
    if (!idValueSet.contains(operand)) {
      llvm::errs() << "Operand missing: " << operand << "\n";
    }

  // The results of the affine value map, which are the access addresses, will
  // be placed to the leftmost of all columns.
  cst.composeMap(&vMap);
}

OslScop::SymbolTable *OslScop::getSymbolTable() { return &symbolTable; }

OslScop::ValueTable *OslScop::getValueTable() { return &valueTable; }

OslScop::MemRefToId *OslScop::getMemRefIdMap() { return &memRefIdMap; }

OslScop::ScopStmtMap *OslScop::getScopStmtMap() { return &scopStmtMap; }

OslScop::ScopStmtNames *OslScop::getScopStmtNames() { return &scopStmtNames; }

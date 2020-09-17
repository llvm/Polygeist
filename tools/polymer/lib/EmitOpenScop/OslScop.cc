//===- OslScop.cc -----------------------------------------------*- C++ -*-===//
//
// This file implements the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//

#include "polymer/OslScop.h"

#include "osl/osl.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

using namespace polymer;

namespace {

/// Create osl_vector from a STL vector. Since the input vector is of type
/// int64_t, we can safely assume the osl_vector we will generate has 64 bits
/// precision. The input vector doesn't contain the e/i indicator.
void getOslVector(bool isEq, llvm::ArrayRef<int64_t> vec,
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
osl_statement_p getOslStatement(osl_scop_p scop, unsigned index) {
  osl_statement_p stmt = scop->statement;
  for (unsigned i = 0; i <= index; i++) {
    // stmt accessed in the linked list before counting to index should not be
    // NULL.
    assert(stmt && "index exceeds the range of statements in scop.");
    if (i == index)
      return stmt;
    stmt = stmt->next;
  }
}

} // namespace

OslScop::OslScop() {
  scop = osl_scop_malloc();

  // Initialize string buffer for language.
  char *language;
  OSL_malloc(language, char *, 2);
  OSL_strdup(language, "C");

  scop->language = language;

  // Use the default interface registry
  osl_interface_p registry = osl_interface_get_default_registry();
  scop->registry = osl_interface_clone(registry);
}

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
         "Number of elements in the eqs should be an integer multiply if "
         "numColsInEqs\n");
  size_t numEqs = eqs.size() / numColsInEqs;

  // Replace those allocated vector elements in rel.
  for (int i = 0; i < numRows; i++) {
    osl_vector_p vec;

    if (i >= numEqs) {
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
      // Initialize a relation list for access if there isn't one.
      if (!stmt->access)
        stmt->access = osl_relation_list_malloc();

      // Append a new list node.
      osl_relation_list_p relList = osl_relation_list_malloc();
      relList->elt = rel;
      osl_relation_list_add(&(stmt->access), relList);
    }
  }
}

void OslScop::addGeneric(int target, llvm::StringRef tag,
                         llvm::StringRef content) {

  osl_generic_p generic = osl_generic_malloc();

  // Add interface.
  osl_interface_p interface = osl_interface_lookup(scop->registry, tag.data());
  generic->interface = osl_interface_nclone(interface, 1);

  // Add content
  char *buf;
  OSL_malloc(buf, char *, content.size() * sizeof(char));
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

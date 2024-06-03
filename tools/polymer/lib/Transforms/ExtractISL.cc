
#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/OpenScop.h"
#include "polymer/Transforms/PlutoTransform.h"

#include "pluto/internal/pluto.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

#define DEBUG_TYPE "pluto-opt"

// TODO is this correct? check out pluto
// #define SCOPLIB_INT_T_IS_LONGLONG // Defined in src/Makefile.am
#define PLUTO_OSL_PRECISION 64

// Below file is mostly from pluto/osl_pluto.c
//
/*
 * This software is available under the MIT license. Please see LICENSE.MIT
 * in the top-level directory for details.
 *
 * This file is part of libpluto.
 */
#include "isl/printer.h"
#include <assert.h>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pluto/osl_pluto.h"

#include "constraints.h"
#include "math_support.h"
#include "pluto/internal/pluto.h"
#include "pluto/matrix.h"
#include "pluto/pluto.h"
#include "pluto_codegen_if.h"
#include "program.h"

#include "candl/candl.h"

#include "osl/body.h"
#include "osl/extensions/arrays.h"
#include "osl/extensions/dependence.h"
#include "osl/extensions/loop.h"
#include "osl/extensions/pluto_unroll.h"
#include "osl/extensions/scatnames.h"
#include "osl/macros.h"
#include "osl/relation_list.h"
#include "osl/scop.h"

#include <isl/aff.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/mat.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>

/*
 * Converts a [A c] pluto transformations to a [eq -I A c] osl scattering
 */
static osl_relation_p pluto_trans_to_osl_scattering(PlutoMatrix *mat,
                                                    int npar) {
  int i, j;
  osl_relation_p smat;

  if (!mat)
    return NULL;

  smat = osl_relation_pmalloc(PLUTO_OSL_PRECISION, mat->nrows,
                              mat->nrows + mat->ncols + 1);
  smat->type = OSL_TYPE_SCATTERING;
  smat->nb_parameters = npar;
  smat->nb_output_dims = mat->nrows;
  smat->nb_input_dims = mat->ncols - npar - 1;
  smat->nb_local_dims = 0;

  for (i = 0; i < smat->nb_rows; i++) {
    for (j = 1; j < smat->nb_columns; j++) {

      /* Only equalities in schedule expected */
      if (j == 0) // eq/neq (first) column
        osl_int_set_si(smat->precision, &smat->m[i][j], 0);

      // fill out the output dims
      else if (j == i + 1)
        osl_int_set_si(smat->precision, &smat->m[i][j], -1);
      else if (j <= smat->nb_output_dims) // non diagonal zeros
        osl_int_set_si(smat->precision, &smat->m[i][j], 0);

      // fill out the intput_dims+params+const
      else
        osl_int_set_si(smat->precision, &smat->m[i][j],
                       mat->val[i][j - smat->nb_output_dims - 1]);
    }
  }

  return smat;
}

static int get_osl_write_access_position(osl_relation_list_p rl,
                                         osl_relation_p access) {
  int num;

  num = -1;

  osl_relation_list_p tmp = rl;
  for (; tmp; tmp = tmp->next) {

    if ((tmp->elt->type == OSL_TYPE_WRITE) ||
        (tmp->elt->type == OSL_TYPE_MAY_WRITE))
      num++;

    if (tmp->elt == access)
      break;
  }
  assert(num >= 0);
  return num;
}

static int get_osl_read_access_position(osl_relation_list_p rl,
                                        osl_relation_p access) {
  int num;

  num = -1;

  osl_relation_list_p tmp = rl;
  for (; tmp; tmp = tmp->next) {

    if (tmp->elt->type == OSL_TYPE_READ)
      num++;

    if (tmp->elt == access)
      break;
  }
  assert(num >= 0);
  return num;
}

/*
 * Returns a list of write or may_write access relations in a list
 */
static osl_relation_list_p
osl_access_list_filter_write(osl_relation_list_p list) {

  osl_relation_list_p copy = osl_relation_list_clone(list);
  osl_relation_list_p filtered = NULL;
  osl_relation_list_p previous = NULL;
  osl_relation_list_p trash;
  int first = 1;

  while (copy != NULL) {
    if ((copy->elt != NULL) && ((copy->elt->type == OSL_TYPE_WRITE) ||
                                (copy->elt->type == OSL_TYPE_MAY_WRITE))) {
      if (first) {
        filtered = copy;
        first = 0;
      }

      previous = copy;
      copy = copy->next;
    } else {
      trash = copy;
      if (!first)
        previous->next = copy->next;
      copy = copy->next;
      trash->next = NULL;
      osl_relation_list_free(trash);
    }
  }

  return filtered;
}

/*
 * Returns a list of read access relations in a list
 */
static osl_relation_list_p
osl_access_list_filter_read(osl_relation_list_p list) {

  osl_relation_list_p copy = osl_relation_list_clone(list);
  osl_relation_list_p filtered = NULL;
  osl_relation_list_p previous = NULL;
  osl_relation_list_p trash;
  int first = 1;

  while (copy != NULL) {
    if ((copy->elt != NULL) && (copy->elt->type == OSL_TYPE_READ)) {
      if (first) {
        filtered = copy;
        first = 0;
      }

      previous = copy;
      copy = copy->next;
    } else {
      trash = copy;
      if (!first)
        previous->next = copy->next;
      copy = copy->next;
      trash->next = NULL;
      osl_relation_list_free(trash);
    }
  }

  return filtered;
}

/*
 * Converts an osl dependence domain to Pluto constraints
 * See osl/extensions/dependence.h for the osl dependence domain matrix format
 */
static PlutoConstraints *
osl_dep_domain_to_pluto_constraints(osl_dependence_p in_dep,
                                    PlutoContext *context) {
  int s_dom_output_dims = in_dep->source_nb_output_dims_domain;
  int t_dom_output_dims = in_dep->target_nb_output_dims_domain;

  int nb_output_dims = in_dep->source_nb_output_dims_domain +
                       in_dep->source_nb_output_dims_access;
  int nb_input_dims = in_dep->target_nb_output_dims_domain +
                      in_dep->target_nb_output_dims_access;

  /* Compute osl domain indexes */
  int osl_ind_source_local_domain = 1 + nb_output_dims + nb_input_dims;
  int osl_ind_source_local_access =
      osl_ind_source_local_domain + in_dep->source_nb_local_dims_domain;
  int osl_ind_target_local_domain =
      osl_ind_source_local_access + in_dep->source_nb_local_dims_access;
  int osl_ind_target_local_access =
      osl_ind_target_local_domain + in_dep->target_nb_local_dims_domain;
  int osl_ind_params =
      osl_ind_target_local_access + in_dep->target_nb_local_dims_access;

  /* Compute pluto constraints domain indexes */
  int pl_ind_target_domain = 1 + in_dep->source_nb_output_dims_domain;
  int pl_ind_params =
      pl_ind_target_domain + in_dep->target_nb_output_dims_domain;

  int rows, cols = 0;

  int nb_pars = in_dep->stmt_source_ptr->domain->nb_parameters;
  int s_dom_rows = in_dep->stmt_source_ptr->domain->nb_rows;
  int t_dom_rows = in_dep->stmt_target_ptr->domain->nb_rows;
  int s_acc_rows = in_dep->ref_source_access_ptr->nb_rows - 1;
  int depth = in_dep->depth;

  //
  rows = s_dom_rows + t_dom_rows +
         (s_acc_rows == 0 ? 1 : s_acc_rows) // special case for 0-dimention
                                            // array(scalar)
         + depth;
  cols = s_dom_output_dims + t_dom_output_dims + nb_pars +
         2; // cols: 2 => eq + const

  PlutoConstraints *cst = pluto_constraints_alloc(rows, cols - 1, context);
  cst->nrows = rows;
  cst->ncols = cols - 1;

  int i = 0;
  int j = 0;
  int osl_constraint = 0;
  int pl_constraint = 0;
  int osl_index = 0;
  int pl_index = 0;

  // copy source domain
  osl_relation_p s_domain = in_dep->stmt_source_ptr->domain;
  for (i = 0; i < s_domain->nb_rows; i++) {

    // copy first column
    if (osl_int_zero(in_dep->domain->precision,
                     in_dep->domain->m[osl_constraint][0])) {
      cst->is_eq[pl_constraint] = 1;
    } else {
      cst->is_eq[pl_constraint] = 0;
    }

    // start of matrix
    osl_index = 1;    // start of src_stmt_domain_output_dims
    pl_index = 1 - 1; // -1 for pluto
    for (j = 0; j < s_dom_output_dims; j++)
      cst->val[pl_constraint][pl_index + j] =
          osl_int_get_si(in_dep->domain->precision,
                         in_dep->domain->m[osl_constraint][osl_index + j]);

    // copy localdims - not supprted by converter
    if (s_domain->nb_local_dims) {
      fprintf(stderr, "local dimensions in domain not supported\n");
      exit(1);
    }

    // copy params + constant
    osl_index = osl_ind_params;
    pl_index = pl_ind_params - 1; // -1 for pluto
    for (j = 0; j < nb_pars + 1; j++)
      cst->val[pl_constraint][pl_index + j] =
          osl_int_get_si(in_dep->domain->precision,
                         in_dep->domain->m[osl_constraint][osl_index + j]);

    osl_constraint++;
    pl_constraint++;
  }

  // copy target domain
  osl_relation_p t_domain = in_dep->stmt_target_ptr->domain;
  for (i = 0; i < t_domain->nb_rows; i++) {

    // copy first column
    if (osl_int_zero(in_dep->domain->precision,
                     in_dep->domain->m[osl_constraint][0])) {
      cst->is_eq[pl_constraint] = 1;
    } else {
      cst->is_eq[pl_constraint] = 0;
    }

    // start of matrix
    osl_index = 1 + nb_output_dims;
    pl_index = pl_ind_target_domain - 1; // -1 for pluto
    for (j = 0; j < t_dom_output_dims; j++)
      cst->val[pl_constraint][pl_index + j] =
          osl_int_get_si(in_dep->domain->precision,
                         in_dep->domain->m[osl_constraint][osl_index + j]);

    // copy local dims - not supported in converter
    if (t_domain->nb_local_dims) {
      fprintf(stderr, "local dimensions in domain not supproted\n");
      exit(1);
    }

    // copy params + constant
    osl_index = osl_ind_params;
    pl_index = pl_ind_params - 1; // -1 for pluto
    for (j = 0; j < nb_pars + 1; j++)
      cst->val[pl_constraint][pl_index + j] =
          osl_int_get_si(in_dep->domain->precision,
                         in_dep->domain->m[osl_constraint][osl_index + j]);

    pl_constraint++;
    osl_constraint++;
  }

  // copy source as well as target access
  int osl_s_index = 0;
  int osl_t_index = 0;
  int pl_s_index = 0;
  int pl_t_index = 0;

  osl_relation_p s_access = in_dep->ref_source_access_ptr;
  osl_relation_p t_access = in_dep->ref_target_access_ptr;

  osl_constraint++; // skip the array_id line

  for (i = 0; i < s_acc_rows; i++) {

    // copy first column
    if (osl_int_zero(in_dep->domain->precision,
                     in_dep->domain->m[osl_constraint][0])) {
      cst->is_eq[pl_constraint] = 1;
    } else {
      cst->is_eq[pl_constraint] = 0;
    }

    osl_s_index = 1;
    osl_t_index = 1 + nb_output_dims;
    pl_s_index = 1 - 1;                    // -1 for pluto
    pl_t_index = pl_ind_target_domain - 1; // -1 for pluto

    for (j = 0; j < s_access->nb_input_dims; j++) {
      cst->val[pl_constraint][pl_s_index + j] =
          osl_int_get_si(in_dep->domain->precision,
                         in_dep->domain->m[osl_constraint][osl_s_index + j]);
    }

    for (j = 0; j < t_access->nb_input_dims; j++) { // t_acc_dims==s_acc_dims
      cst->val[pl_constraint][pl_t_index + j] = osl_int_get_si(
          in_dep->domain->precision,
          in_dep->domain
              ->m[osl_constraint + s_access->nb_rows][osl_t_index + j]);
    }

    // copy local dimensions - not supported by converter
    if (s_access->nb_local_dims || t_access->nb_local_dims) {
      fprintf(stderr, "local dimensions in Access not supproted\n");
      exit(1);
    }

    // copy params + constant
    osl_index = osl_ind_params;
    pl_index = pl_ind_params - 1; // -1 for pluto
    for (j = 0; j < nb_pars + 1; j++) {
      // get src params
      int src_param =
          osl_int_get_si(in_dep->domain->precision,
                         in_dep->domain->m[osl_constraint][osl_index + j]);
      // get tgt params
      int tgt_param = osl_int_get_si(
          in_dep->domain->precision,
          in_dep->domain->m[osl_constraint + s_access->nb_rows][osl_index + j]);

      tgt_param = -tgt_param; // oppose

      cst->val[pl_constraint][pl_index + j] = src_param - tgt_param;
    }

    pl_constraint++;
    osl_constraint++;
  }

  // copy access equalities
  // skip min_depth
  int min_depth = OSL_min(s_access->nb_output_dims, t_access->nb_output_dims);
  osl_constraint += s_access->nb_rows + min_depth;

  // s_acc_rows calculated by subtracting 1 from acc.nb_rows
  // in case of a scalar this results in 0, still add a constraint for pluto
  if (s_acc_rows == 0)
    pl_constraint++;

  // copy depth
  osl_s_index = 1;
  osl_t_index = 1 + nb_output_dims;
  pl_s_index = 1 - 1;                    // -1 for pluto
  pl_t_index = pl_ind_target_domain - 1; // -1 for pluto
  for (i = 0; i < depth; i++) {
    // copy first column
    if (osl_int_zero(in_dep->domain->precision,
                     in_dep->domain->m[osl_constraint][0])) {
      cst->is_eq[pl_constraint] = 1;
    } else {
      cst->is_eq[pl_constraint] = 0;
    }

    // copy subscript equalities
    cst->val[pl_constraint][pl_s_index + i] =
        osl_int_get_si(in_dep->domain->precision,
                       in_dep->domain->m[osl_constraint][osl_s_index + i]);
    cst->val[pl_constraint][pl_t_index + i] =
        osl_int_get_si(in_dep->domain->precision,
                       in_dep->domain->m[osl_constraint][osl_t_index + i]);

    // copy params -> not applicable here

    // copy const == last column
    cst->val[pl_constraint][cst->ncols - 1] = osl_int_get_si(
        in_dep->domain->precision,
        in_dep->domain->m[osl_constraint][in_dep->domain->nb_columns - 1]);

    osl_constraint++;
    pl_constraint++;
  }

  // return new domain
  return cst;
}

/*
 * Converts [A c] PLuto constraints to a [eq A c] domain relation
 */
static osl_relation_p pluto_constraints_to_osl_domain(PlutoConstraints *cst,
                                                      int npar) {
  osl_relation_p rln;

  if (cst == NULL)
    return NULL;

  rln = osl_relation_pmalloc(PLUTO_OSL_PRECISION, cst->nrows, cst->ncols + 1);

  // copy matrix values
  for (int i = 0; i < rln->nb_rows; i++) {
    osl_int_set_si(rln->precision, &rln->m[i][0], cst->is_eq[i] ? 0 : 1);
    for (unsigned j = 0; j < cst->ncols; j++) {
      osl_int_set_si(rln->precision, &rln->m[i][j + 1], cst->val[i][j]);
    }
  }

  rln->type = OSL_TYPE_DOMAIN;
  rln->nb_parameters = npar;
  rln->nb_output_dims = rln->nb_columns - rln->nb_parameters - 2;
  rln->nb_input_dims = 0;
  rln->nb_local_dims = 0;

  return rln;
}

/*
 * In an [eq -I A c] relation, rows can be ordered any way.
 * Returns the index for the row for the nth output dimension.
 */
static int osl_relation_get_row_id_for_nth_dimension(osl_relation_p relation,
                                                     int ndim) {
  int nb_ndims_found = 0;
  int row_id = -1;

  if (relation == NULL)
    return OSL_UNDEFINED;

  if ((relation->nb_rows < ndim) || (0 > ndim)) {
    fprintf(stderr, "error: dimension out of bounds");
    exit(1);
  }

  nb_ndims_found = 0;
  for (int i = 0; i < relation->nb_rows; i++) {
    if (!osl_int_zero(relation->precision, relation->m[i][ndim])) {
      nb_ndims_found++;
      row_id = i;
    }
  }
  if (nb_ndims_found == 0) {
    fprintf(stderr, "error: specified dimension not found");
    exit(1);
  }
  if (nb_ndims_found > 1) {
    fprintf(stderr, "error: specified dimension occurs multiple times");
    exit(1);
  }

  return row_id;
}

/*
 * Converts a [eq -I A c] osl scattering to [A c] pluto transformations
 */
static PlutoMatrix *osl_scattering_to_pluto_trans(osl_relation_p smat,
                                                  PlutoContext *context) {
  int i, j;
  PlutoMatrix *mat;

  if (!smat)
    return NULL;

  if (smat->nb_local_dims) {
    fprintf(stderr, "Cannot handle Local Dimensions in a relation.\n");
    exit(1);
  }

  mat = pluto_matrix_alloc(
      smat->nb_rows, smat->nb_columns - smat->nb_output_dims - 1, context);
  for (i = 0; i < smat->nb_rows; i++) {
    /* Only equalities in schedule expected */
    assert(osl_int_get_si(smat->precision, smat->m[i][0]) == 0);

    int row = osl_relation_get_row_id_for_nth_dimension(smat, i + 1);
    for (j = smat->nb_output_dims + 1; j < smat->nb_columns; j++) {
      mat->val[i][j - smat->nb_output_dims - 1] =
          osl_int_get_si(smat->precision, smat->m[row][j]);
    }
  }

  return mat;
}

/*
 * Converts a [eq -I A c] osl access relation to [A c] pluto matrix
 * Note: a[c] and a, having two and one output dimensions respectively
 * in osl, are converted to a one-dimensional pluto matrix.
 */
static PlutoMatrix *osl_access_relation_to_pluto_matrix(osl_relation_p smat,
                                                        PlutoContext *context) {
  int i, j;

  PlutoMatrix *mat;

  if (smat == NULL)
    return NULL;

  if (smat->nb_local_dims) {
    fprintf(stderr, "Cannot handle Local Dimensions in a relation.\n");
    exit(1);
  }

  int nrows =
      smat->nb_rows == 1 ? smat->nb_rows : smat->nb_rows - 1; // skp id line
  int ncols = smat->nb_columns - smat->nb_output_dims - 1;    //-1: skip 1st col
  mat = pluto_matrix_alloc(nrows, ncols, context);

  // Special case for scalars.
  if (smat->nb_rows == 1) {
    for (j = smat->nb_output_dims + 1; j < smat->nb_columns; j++) {
      mat->val[0][j - (smat->nb_output_dims + 1)] = 0;
    }
  } else {
    // fill in the rest of the information
    for (i = 1; i < smat->nb_rows; i++) {
      int row = osl_relation_get_row_id_for_nth_dimension(smat, i + 1);
      for (j = smat->nb_output_dims + 1; j < smat->nb_columns; j++) {
        mat->val[i - 1][j - (smat->nb_output_dims + 1)] =
            osl_int_get_si(smat->precision, smat->m[row][j]);
      }
    }
  }

  return mat;
}

/*
 * Converts a [eq A c] relation to [A c] Pluto constraints
 */
static PlutoConstraints *
osl_relation_to_pluto_constraints(osl_relation_p rln, PlutoContext *context) {
  if (rln == NULL)
    return NULL;

  if (rln->nb_local_dims) {
    fprintf(stderr, "[osl_relation_to_pluto_constraints] Cannot handle Local "
                    "Dimensions in a relation.\n");
    exit(1);
  }

  PlutoConstraints *cst =
      pluto_constraints_alloc(rln->nb_rows, rln->nb_columns - 1, context);
  cst->nrows = rln->nb_rows;

  // copy matrix values
  for (int i = 0; i < rln->nb_rows; i++) {
    cst->is_eq[i] = osl_int_zero(rln->precision, rln->m[i][0]);
    for (unsigned j = 0; j < cst->ncols; j++) {
      cst->val[i][j] = osl_int_get_si(rln->precision, rln->m[i][j + 1]);
    }
  }

  return cst;
}

/// Read statement info from openscop structures (nvar: max domain dim).
static Stmt **osl_to_pluto_stmts(const osl_scop_p scop, PlutoContext *context) {
  int i, j, k;
  Stmt **stmts;
  int npar, nvar, nstmts, max_sched_rows;
  osl_statement_p scop_stmt;

  npar = scop->context->nb_parameters;
  nstmts = osl_statement_number(scop->statement);

  if (nstmts == 0)
    return NULL;

  /* Max dom dimensionality */
  nvar = -1;
  max_sched_rows = 0;
  scop_stmt = scop->statement;
  for (i = 0; i < nstmts; i++) {
    nvar = PLMAX(nvar, osl_statement_get_nb_iterators(scop_stmt));
    max_sched_rows = PLMAX(max_sched_rows, scop_stmt->scattering->nb_rows);
    scop_stmt = scop_stmt->next;
  }

  stmts = (Stmt **)malloc(nstmts * sizeof(Stmt *));

  scop_stmt = scop->statement;

  for (i = 0; i < nstmts; i++) {
    PlutoConstraints *domain =
        osl_relation_to_pluto_constraints(scop_stmt->domain, context);
    PlutoMatrix *trans =
        osl_scattering_to_pluto_trans(scop_stmt->scattering, context);

    int nb_iter = osl_statement_get_nb_iterators(scop_stmt);

    stmts[i] = pluto_stmt_alloc(nb_iter, domain, trans);

    /* Pad with all zero rows */
    int curr_sched_rows = stmts[i]->trans->nrows;
    for (j = curr_sched_rows; j < max_sched_rows; j++) {
      pluto_stmt_add_hyperplane(stmts[i], H_SCALAR, j);
    }

    pluto_constraints_free(domain);
    pluto_matrix_free(trans);

    Stmt *stmt = stmts[i];

    stmt->id = i;
    stmt->type = ORIG;

    assert(scop_stmt->domain->nb_columns - 1 == (int)stmt->dim + npar + 1);

    for (unsigned j = 0; j < stmt->dim; j++) {
      stmt->is_orig_loop[j] = true;
    }

    /* Tile it if it's tilable unless turned off by .fst/.precut file */
    stmt->tile = 1;

    osl_body_p stmt_body =
        (osl_body_p)osl_generic_lookup(scop_stmt->extension, OSL_URI_BODY);

    for (unsigned j = 0; j < stmt->dim; j++) {
      stmt->iterators[j] = strdup(stmt_body->iterators->string[j]);
    }
    /* Set names for domain dimensions */
    char **names = (char **)malloc((stmt->domain->ncols - 1) * sizeof(char *));
    for (unsigned k = 0; k < stmt->dim; k++) {
      names[k] = stmt->iterators[k];
    }
    osl_strings_p osl_scop_params = NULL;
    if (scop->context->nb_parameters) {
      osl_scop_params = (osl_strings_p)scop->parameters->data;
      for (k = 0; k < npar; k++) {
        names[stmt->dim + k] = osl_scop_params->string[k];
      }
    }
    pluto_constraints_set_names(stmt->domain, names);
    free(names);

    /* Statement text */
    stmt->text = osl_strings_sprint(stmt_body->expression); // appends \n
    stmt->text[strlen(stmt->text) - 1] = '\0'; // remove the \n from end

    /* Read/write accesses */
    osl_relation_list_p wlist = osl_access_list_filter_write(scop_stmt->access);
    osl_relation_list_p rlist = osl_access_list_filter_read(scop_stmt->access);

    osl_relation_list_p rlist_t, wlist_t;
    rlist_t = rlist;
    wlist_t = wlist;

    stmt->nwrites = osl_relation_list_count(wlist);
    if (stmt->nwrites > 0)
      stmt->writes =
          (PlutoAccess **)malloc(stmt->nwrites * sizeof(PlutoAccess *));
    else
      stmt->writes = NULL;

    stmt->nreads = osl_relation_list_count(rlist);
    if (stmt->nreads > 0)
      stmt->reads =
          (PlutoAccess **)malloc(stmt->nreads * sizeof(PlutoAccess *));
    else
      stmt->reads = NULL;

    osl_arrays_p arrays =
        (osl_arrays_p)osl_generic_lookup(scop->extension, OSL_URI_ARRAYS);

    int count = 0;
    while (wlist != NULL) {
      PlutoMatrix *wmat =
          osl_access_relation_to_pluto_matrix(wlist->elt, context);
      stmt->writes[count] = (PlutoAccess *)malloc(sizeof(PlutoAccess));
      stmt->writes[count]->mat = wmat;

      // stmt->writes[count]->symbol = NULL;
      if (arrays) {
        int id = osl_relation_get_array_id(wlist->elt);
        stmt->writes[count]->name = strdup(arrays->names[id - 1]);
      } else {
        stmt->writes[count]->name = NULL;
      }

      count++;
      wlist = wlist->next;
    }

    count = 0;
    while (rlist != NULL) {
      PlutoMatrix *rmat =
          osl_access_relation_to_pluto_matrix(rlist->elt, context);
      stmt->reads[count] = (PlutoAccess *)malloc(sizeof(PlutoAccess));
      stmt->reads[count]->mat = rmat;

      // stmt->reads[count]->symbol = NULL;
      if (arrays) {
        int id = osl_relation_get_array_id(rlist->elt);
        stmt->reads[count]->name = strdup(arrays->names[id - 1]);
      } else {
        stmt->reads[count]->name = NULL;
      }

      count++;
      rlist = rlist->next;
    }

    osl_relation_list_free(wlist_t);
    osl_relation_list_free(rlist_t);

    scop_stmt = scop_stmt->next;
  }

  return stmts;
}

/* Convert an m x (1 + m + n + 1) osl_relation_p [d -I A c]
 * to an m x (m + n + 1) isl_mat [-I A c].
 */
static __isl_give isl_mat *extract_equalities_osl(isl_ctx *ctx,
                                                  osl_relation_p relation) {
  int i, j;
  int n_col, n_row;
  isl_mat *eq;

  n_col = relation->nb_columns;
  n_row = relation->nb_rows;

  eq = isl_mat_alloc(ctx, n_row, n_col - 1);

  for (i = 0; i < n_row; ++i) {
    for (j = 0; j < n_col - 1; ++j) {
      int row = osl_relation_get_row_id_for_nth_dimension(relation, i + 1);
      int t = osl_int_get_si(relation->precision, relation->m[row][1 + j]);
      isl_val *v = isl_val_int_from_si(ctx, t);
      eq = isl_mat_set_element_val(eq, i, j, v);
    }
  }

  return eq;
}

/* Convert a osl_relation_p scattering [0 M A c] to
 * the isl_map { i -> A i + c } in the space prescribed by "dim".
 */
static __isl_give isl_map *
osl_scattering_to_isl_map(osl_relation_p scattering,
                          __isl_take isl_space *dim) {
  int n_col;
  isl_ctx *ctx;
  isl_mat *eq, *ineq;
  isl_basic_map *bmap;

  ctx = isl_space_get_ctx(dim);
  n_col = scattering->nb_columns;

  ineq = isl_mat_alloc(ctx, 0, n_col - 1);
  eq = extract_equalities_osl(ctx, scattering);

  bmap = isl_basic_map_from_constraint_matrices(dim, eq, ineq, isl_dim_out,
                                                isl_dim_in, isl_dim_div,
                                                isl_dim_param, isl_dim_cst);

  return isl_map_from_basic_map(bmap);
}

/* Convert an m x (1 + m + n + 1) osl_relation_p [d -I A c]
 * to an m x (m + n + 1) isl_mat [-I A c].
 */
static __isl_give isl_mat *
extract_equalities_osl_access(isl_ctx *ctx, osl_relation_p relation) {
  int i, j;
  int n_col, n_row;
  isl_mat *eq;

  n_row = relation->nb_rows == 1 ? 1 : relation->nb_rows - 1;
  n_col = relation->nb_columns - (relation->nb_rows == 1 ? 1 : 2);

  eq = isl_mat_alloc(ctx, n_row, n_col);

  if (relation->nb_rows == 1) {
    isl_val *v = isl_val_negone(ctx);
    eq = isl_mat_set_element_val(eq, 0, 0, v);
    for (j = 1; j < n_col; ++j) {
      v = isl_val_zero(ctx);
      eq = isl_mat_set_element_val(eq, 0, j, v);
    }
  } else {
    for (i = 1; i < relation->nb_rows; ++i) {
      for (j = 2; j < relation->nb_columns; ++j) {
        int row = osl_relation_get_row_id_for_nth_dimension(relation, i + 1);
        int t = osl_int_get_si(relation->precision, relation->m[row][j]);
        isl_val *v = isl_val_int_from_si(ctx, t);
        eq = isl_mat_set_element_val(eq, i - 1, j - 2, v);
      }
    }
  }

  return eq;
}

/*
 * Like osl_access_list_to_isl_union_map, but just for a single osl access
 * (read or write)
 */
static __isl_give isl_map *
osl_basic_access_to_isl_union_map(osl_relation_p access,
                                  __isl_take isl_set *dom, char **arrays) {
  int len, n_col;
  isl_ctx *ctx;
  isl_space *dim;
  isl_mat *eq, *ineq;

  ctx = isl_set_get_ctx(dom);

  n_col = access->nb_columns - (access->nb_rows == 1 ? 1 : 2);
  len = access->nb_rows == 1 ? 1 : access->nb_rows - 1;

  isl_basic_map *bmap;
  isl_map *map;
  int arr = osl_relation_get_array_id(access) - 1;

  dim = isl_set_get_space(dom);
  dim = isl_space_from_domain(dim);
  dim = isl_space_add_dims(dim, isl_dim_out, len);
  dim = isl_space_set_tuple_name(dim, isl_dim_out, arrays[arr]);

  ineq = isl_mat_alloc(ctx, 0, n_col);
  eq = extract_equalities_osl_access(ctx, access);

  bmap = isl_basic_map_from_constraint_matrices(dim, eq, ineq, isl_dim_out,
                                                isl_dim_in, isl_dim_div,
                                                isl_dim_param, isl_dim_cst);
  map = isl_map_from_basic_map(bmap);
  map = isl_map_intersect_domain(map, dom);

  return map;
}

/* Convert a osl_relation_list_p describing a series of accesses [eq -I B c]
 * to an isl_union_map with domain "dom" (in space "D").
 * The -I columns identify the output dimensions of the access, the first
 * of them being the identity of the array being accessed.  The remaining
 * output dimensions identiy the array subscripts.
 *
 * Let "A" be array identified by the first entry.
 * The input dimension columns have the form [B c].
 * Each such access is converted to a map { D[i] -> A[B i + c] } * dom.
 *
 */
static __isl_give isl_union_map *
osl_access_list_to_isl_union_map(osl_relation_list_p list,
                                 __isl_take isl_set *dom, char **arrays) {
  int len, n_col;
  isl_ctx *ctx;
  isl_space *space;
  isl_mat *eq, *ineq;
  isl_union_map *res;

  ctx = isl_set_get_ctx(dom);

  space = isl_set_get_space(dom);
  space = isl_space_drop_dims(space, isl_dim_set, 0,
                              isl_space_dim(space, isl_dim_set));
  res = isl_union_map_empty(space);

  for (; list; list = list->next) {

    n_col = list->elt->nb_columns - (list->elt->nb_rows == 1 ? 1 : 2);
    len = list->elt->nb_rows == 1 ? 1 : list->elt->nb_rows - 1;

    isl_basic_map *bmap;
    isl_map *map;
    int arr = osl_relation_get_array_id(list->elt) - 1;

    space = isl_set_get_space(dom);
    space = isl_space_from_domain(space);
    space = isl_space_add_dims(space, isl_dim_out, len);
    space = isl_space_set_tuple_name(space, isl_dim_out, arrays[arr]);

    ineq = isl_mat_alloc(ctx, 0, n_col);
    eq = extract_equalities_osl_access(ctx, list->elt);

    bmap = isl_basic_map_from_constraint_matrices(space, eq, ineq, isl_dim_out,
                                                  isl_dim_in, isl_dim_div,
                                                  isl_dim_param, isl_dim_cst);
    map = isl_map_from_basic_map(bmap);
    map = isl_map_intersect_domain(map, isl_set_copy(dom));
    res = isl_union_map_union(res, isl_union_map_from_map(map));
  }

  isl_set_free(dom);

  return res;
}

/* Set the dimension names of type "type" according to the elements
 * in the array "names".
 */
static __isl_give isl_space *set_names(__isl_take isl_space *space,
                                       enum isl_dim_type type, char **names) {
  int i;

  for (i = 0; i < isl_space_dim(space, type); ++i)
    space = isl_space_set_dim_name(space, type, i, names[i]);

  return space;
}

/* Convert a osl_relation_p containing the constraints of a domain
 * to an isl_set.
 * One shot only; does not take into account the next ptr.
 */
static __isl_give isl_set *osl_relation_to_isl_set(osl_relation_p relation,
                                                   __isl_take isl_space *dim) {
  int i, j;
  int n_eq = 0, n_ineq = 0;
  isl_ctx *ctx;
  isl_mat *eq, *ineq;
  isl_basic_set *bset;

  ctx = isl_space_get_ctx(dim);

  for (i = 0; i < relation->nb_rows; ++i)
    if (osl_int_zero(relation->precision, relation->m[i][0]))
      n_eq++;
    else
      n_ineq++;

  eq = isl_mat_alloc(ctx, n_eq, relation->nb_columns - 1);
  ineq = isl_mat_alloc(ctx, n_ineq, relation->nb_columns - 1);

  n_eq = n_ineq = 0;
  for (i = 0; i < relation->nb_rows; ++i) {
    isl_mat **m;
    int row;

    if (osl_int_zero(relation->precision, relation->m[i][0])) {
      m = &eq;
      row = n_eq++;
    } else {
      m = &ineq;
      row = n_ineq++;
    }

    for (j = 0; j < relation->nb_columns - 1; ++j) {
      int t = osl_int_get_si(relation->precision, relation->m[i][1 + j]);
      *m = isl_mat_set_element_si(*m, row, j, t);
    }
  }

  bset = isl_basic_set_from_constraint_matrices(
      dim, eq, ineq, isl_dim_set, isl_dim_div, isl_dim_param, isl_dim_cst);
  return isl_set_from_basic_set(bset);
}

/* Convert a osl_relation_p describing a union of domains
 * to an isl_set.
 */
static __isl_give isl_set *
osl_relation_list_to_isl_set(osl_relation_p list, __isl_take isl_space *space) {
  isl_set *set;

  set = isl_set_empty(isl_space_copy(space));
  for (; list; list = list->next) {
    isl_set *set_i;
    set_i = osl_relation_to_isl_set(list, isl_space_copy(space));
    set = isl_set_union(set, set_i);
  }

  isl_space_free(space);
  return set;
}

static osl_names_p get_scop_names(osl_scop_p scop) {

  // generate temp names
  osl_names_p names = osl_scop_names(scop);

  // if scop has names substitute them for temp names
  if (scop->context->nb_parameters) {
    osl_strings_free(names->parameters);
    names->parameters =
        osl_strings_clone((osl_strings_p)scop->parameters->data);
  }

  osl_arrays_p arrays =
      (osl_arrays_p)osl_generic_lookup(scop->extension, OSL_URI_ARRAYS);
  if (arrays) {
    osl_strings_free(names->arrays);
    names->arrays = osl_arrays_to_strings(arrays);
  }

  return names;
}

/* Compute dependences based on the iteration domain and access
 * information in the OSL 'scop' and put the result in 'prog'.
 */
static void compute_deps_osl(osl_scop_p scop, PlutoProg *prog,
                             PlutoOptions *options) {
  int i, racc_num, wacc_num;
  int nstmts = osl_statement_number(scop->statement);
  isl_space *space;
  isl_space *param_space;
  isl_set *context;
  isl_union_map *dep_raw, *dep_war, *dep_waw, *dep_rar, *trans_dep_war;
  isl_union_map *trans_dep_waw;
  osl_statement_p stmt;
  osl_strings_p scop_params = NULL;

  if (!options->silent) {
    printf("[pluto] compute_deps (isl%s)\n",
           options->lastwriter ? " with lastwriter" : "");
  }

  isl_ctx *ctx = isl_ctx_alloc();
  assert(ctx);

  osl_names_p names = get_scop_names(scop);

  space = isl_space_set_alloc(ctx, scop->context->nb_parameters, 0);
  if (scop->context->nb_parameters) {
    scop_params = (osl_strings_p)scop->parameters->data;
    space = set_names(space, isl_dim_param, scop_params->string);
  }
  param_space = isl_space_params(isl_space_copy(space));
  context = osl_relation_to_isl_set(scop->context, param_space);

  if (!options->rar)
    dep_rar = isl_union_map_empty(isl_space_copy(space));
  isl_union_map *writes = isl_union_map_empty(isl_space_copy(space));
  isl_union_map *reads = isl_union_map_empty(isl_space_copy(space));
  isl_union_map *schedule = isl_union_map_empty(isl_space_copy(space));
  isl_union_map *empty = isl_union_map_empty(space);

  if (!options->isldepaccesswise) {
    /* Leads to fewer dependences. Each dependence may not have a unique
     * source/target access relating to it, since a union is taken
     * across all reads for a statement (and writes) for a particualr
     * array. Relationship between a dependence and associated dependent
     * data / array elements is lost, and some analyses may not work with
     * such a representation
     */
    for (i = 0, stmt = scop->statement; i < nstmts; ++i, stmt = stmt->next) {
      isl_set *dom;
      isl_map *schedule_i;
      isl_union_map *read_i;
      isl_union_map *write_i;
      char name[20];

      snprintf(name, sizeof(name), "S_%d", i);

      int niter = osl_statement_get_nb_iterators(stmt);
      space = isl_space_set_alloc(ctx, scop->context->nb_parameters, niter);
      if (scop->context->nb_parameters) {
        scop_params = (osl_strings_p)scop->parameters->data;
        space = set_names(space, isl_dim_param, scop_params->string);
      }
      if (niter) {
        osl_body_p stmt_body =
            (osl_body_p)osl_generic_lookup(stmt->extension, OSL_URI_BODY);
        space = set_names(space, isl_dim_set, stmt_body->iterators->string);
      }
      space = isl_space_set_tuple_name(space, isl_dim_set, name);
      dom = osl_relation_list_to_isl_set(stmt->domain, space);
      dom = isl_set_intersect_params(dom, isl_set_copy(context));

      space = isl_space_alloc(ctx, scop->context->nb_parameters, niter,
                              2 * niter + 1);
      if (scop->context->nb_parameters) {
        scop_params = (osl_strings_p)scop->parameters->data;
        space = set_names(space, isl_dim_param, scop_params->string);
      }
      if (niter) {
        osl_body_p stmt_body =
            (osl_body_p)osl_generic_lookup(stmt->extension, OSL_URI_BODY);
        space = set_names(space, isl_dim_in, stmt_body->iterators->string);
      }
      space = isl_space_set_tuple_name(space, isl_dim_in, name);
      schedule_i = osl_scattering_to_isl_map(stmt->scattering, space);

      osl_relation_list_p rlist = osl_access_list_filter_read(stmt->access);
      osl_relation_list_p wlist = osl_access_list_filter_write(stmt->access);

      osl_arrays_p arrays =
          (osl_arrays_p)osl_generic_lookup(scop->extension, OSL_URI_ARRAYS);
      if (arrays) {
        osl_strings_free(names->arrays);
        names->arrays = osl_arrays_to_strings(arrays);
      }

      read_i = osl_access_list_to_isl_union_map(rlist, isl_set_copy(dom),
                                                names->arrays->string);
      write_i = osl_access_list_to_isl_union_map(wlist, isl_set_copy(dom),
                                                 names->arrays->string);

      reads = isl_union_map_union(reads, read_i);
      writes = isl_union_map_union(writes, write_i);
      schedule =
          isl_union_map_union(schedule, isl_union_map_from_map(schedule_i));

      osl_relation_list_free(rlist);
      osl_relation_list_free(wlist);
    }
  } else {
    /* Each dependence is for a particular source and target access. Use
     * <stmt, access> pair while relating to accessed data so each
     * dependence can be associated to a unique source and target access
     */

    for (i = 0, stmt = scop->statement; i < nstmts; ++i, stmt = stmt->next) {
      isl_set *dom;

      racc_num = 0;
      wacc_num = 0;

      osl_relation_list_p access = stmt->access;
      for (; access; access = access->next) {
        isl_map *read_pos;
        isl_map *write_pos;
        isl_map *schedule_i;

        char name[25];

        if (access->elt->type == OSL_TYPE_READ) {
          snprintf(name, sizeof(name), "S_%d_r%d", i, racc_num);
        } else {
          snprintf(name, sizeof(name), "S_%d_w%d", i, wacc_num);
        }

        int niter = osl_statement_get_nb_iterators(stmt);
        space = isl_space_set_alloc(ctx, scop->context->nb_parameters, niter);
        if (scop->context->nb_parameters) {
          scop_params = (osl_strings_p)scop->parameters->data;
          space = set_names(space, isl_dim_param, scop_params->string);

          osl_strings_free(names->parameters);
          names->parameters = osl_strings_clone(scop_params);
        }
        if (niter) {
          osl_body_p stmt_body =
              (osl_body_p)osl_generic_lookup(stmt->extension, OSL_URI_BODY);
          space = set_names(space, isl_dim_set, stmt_body->iterators->string);

          osl_strings_free(names->iterators);
          names->iterators = osl_strings_clone(stmt_body->iterators);
        }
        space = isl_space_set_tuple_name(space, isl_dim_set, name);
        dom = osl_relation_list_to_isl_set(stmt->domain, space);
        dom = isl_set_intersect_params(dom, isl_set_copy(context));

        space = isl_space_alloc(ctx, scop->context->nb_parameters, niter,
                                2 * niter + 1);
        if (scop->context->nb_parameters) {
          scop_params = (osl_strings_p)scop->parameters->data;
          space = set_names(space, isl_dim_param, scop_params->string);
        }
        if (niter) {
          osl_body_p stmt_body =
              (osl_body_p)osl_generic_lookup(stmt->extension, OSL_URI_BODY);
          space = set_names(space, isl_dim_in, stmt_body->iterators->string);
        }
        space = isl_space_set_tuple_name(space, isl_dim_in, name);

        schedule_i = osl_scattering_to_isl_map(stmt->scattering, space);

        osl_arrays_p arrays =
            (osl_arrays_p)osl_generic_lookup(scop->extension, OSL_URI_ARRAYS);
        if (arrays) {
          osl_strings_free(names->arrays);
          names->arrays = osl_arrays_to_strings(arrays);
        }

        if (access->elt->type == OSL_TYPE_READ) {
          read_pos = osl_basic_access_to_isl_union_map(access->elt, dom,
                                                       names->arrays->string);
          reads = isl_union_map_union(reads, isl_union_map_from_map(read_pos));
        } else {
          write_pos = osl_basic_access_to_isl_union_map(access->elt, dom,
                                                        names->arrays->string);
          writes =
              isl_union_map_union(writes, isl_union_map_from_map(write_pos));
        }

        schedule =
            isl_union_map_union(schedule, isl_union_map_from_map(schedule_i));

        if (access->elt->type == OSL_TYPE_READ) {
          racc_num++;
        } else {
          wacc_num++;
        }
      }
    }
  }

  compute_deps_isl(reads, writes, schedule, empty, &dep_raw, &dep_war, &dep_waw,
                   &dep_rar, &trans_dep_war, &trans_dep_waw, options);

  prog->ndeps = 0;
  isl_union_map_foreach_map(dep_raw, &isl_map_count, &prog->ndeps);
  isl_union_map_foreach_map(dep_war, &isl_map_count, &prog->ndeps);
  isl_union_map_foreach_map(dep_waw, &isl_map_count, &prog->ndeps);
  isl_union_map_foreach_map(dep_rar, &isl_map_count, &prog->ndeps);

  prog->deps = (Dep **)malloc(prog->ndeps * sizeof(Dep *));
  for (i = 0; i < prog->ndeps; i++) {
    prog->deps[i] = pluto_dep_alloc();
  }
  prog->ndeps = 0;
  prog->ndeps += extract_deps_from_isl_union_map(dep_raw, prog->deps,
                                                 prog->ndeps, prog->stmts,
                                                 PLUTO_DEP_RAW, prog->context);
  prog->ndeps += extract_deps_from_isl_union_map(dep_war, prog->deps,
                                                 prog->ndeps, prog->stmts,
                                                 PLUTO_DEP_WAR, prog->context);
  prog->ndeps += extract_deps_from_isl_union_map(dep_waw, prog->deps,
                                                 prog->ndeps, prog->stmts,
                                                 PLUTO_DEP_WAW, prog->context);
  prog->ndeps += extract_deps_from_isl_union_map(dep_rar, prog->deps,
                                                 prog->ndeps, prog->stmts,
                                                 PLUTO_DEP_RAR, prog->context);

  if (options->lastwriter) {
    prog->ntransdeps = 0;
    isl_union_map_foreach_map(dep_raw, &isl_map_count, &prog->ntransdeps);
    isl_union_map_foreach_map(trans_dep_war, &isl_map_count, &prog->ntransdeps);
    isl_union_map_foreach_map(trans_dep_waw, &isl_map_count, &prog->ntransdeps);
    isl_union_map_foreach_map(dep_rar, &isl_map_count, &prog->ntransdeps);

    if (prog->ntransdeps > 0) {
      prog->transdeps = (Dep **)malloc(prog->ntransdeps * sizeof(Dep *));
      for (i = 0; i < prog->ntransdeps; i++) {
        prog->transdeps[i] = pluto_dep_alloc();
      }
      int ntransdeps = 0;
      ntransdeps += extract_deps_from_isl_union_map(
          dep_raw, prog->transdeps, ntransdeps, prog->stmts, PLUTO_DEP_RAW,
          prog->context);
      ntransdeps += extract_deps_from_isl_union_map(
          trans_dep_war, prog->transdeps, ntransdeps, prog->stmts,
          PLUTO_DEP_WAR, prog->context);
      ntransdeps += extract_deps_from_isl_union_map(
          trans_dep_waw, prog->transdeps, ntransdeps, prog->stmts,
          PLUTO_DEP_WAW, prog->context);
      ntransdeps += extract_deps_from_isl_union_map(
          dep_rar, prog->transdeps, ntransdeps, prog->stmts, PLUTO_DEP_RAR,
          prog->context);
    }

    isl_union_map_free(trans_dep_war);
    isl_union_map_free(trans_dep_waw);
  }

  isl_union_map_free(dep_raw);
  isl_union_map_free(dep_war);
  isl_union_map_free(dep_waw);
  isl_union_map_free(dep_rar);

  isl_union_map_free(writes);
  isl_union_map_free(reads);
  isl_union_map_free(schedule);
  isl_union_map_free(empty);
  isl_set_free(context);

  if (names)
    osl_names_free(names);

  isl_ctx_free(ctx);
}

/* Read dependences from candl structures. */
static Dep **deps_read(osl_dependence_p candlDeps, PlutoProg *prog) {
  int i, ndeps;
  int spos, tpos;
  Dep **deps;
  int npar = prog->npar;
  Stmt **stmts = prog->stmts;
  PlutoContext *context = prog->context;

  ndeps = osl_nb_dependences(candlDeps);

  deps = (Dep **)malloc(ndeps * sizeof(Dep *));

  for (i = 0; i < ndeps; i++) {
    deps[i] = pluto_dep_alloc();
  }

  osl_dependence_p candl_dep = candlDeps;

  candl_dep = candlDeps;

  IF_DEBUG(candl_dependence_pprint(stdout, candl_dep));

  /* Dependence polyhedra information */
  for (i = 0; i < ndeps; i++) {
    Dep *dep = deps[i];
    dep->id = i;
    switch (candl_dep->type) {
    case OSL_DEPENDENCE_RAW:
      dep->type = PLUTO_DEP_RAW;
    case OSL_DEPENDENCE_WAW:
      dep->type = PLUTO_DEP_WAW;
    case OSL_DEPENDENCE_WAR:
      dep->type = PLUTO_DEP_WAR;
    case OSL_DEPENDENCE_RAR:
      dep->type = PLUTO_DEP_RAR;
    case OSL_UNDEFINED:
    default:
      dep->type = PLUTO_DEP_UNDEFINED;
    }
    dep->src = candl_dep->label_source;
    dep->dest = candl_dep->label_target;

    dep->dpolytope = osl_dep_domain_to_pluto_constraints(candl_dep, context);
    dep->bounding_poly = pluto_constraints_dup(dep->dpolytope);

    pluto_constraints_set_names_range(
        dep->dpolytope, stmts[dep->src]->iterators, 0, 0, stmts[dep->src]->dim);
    // Suffix the destination iterators with a '.
    char **dnames = (char **)malloc(stmts[dep->dest]->dim * sizeof(char *));
    for (unsigned j = 0; j < stmts[dep->dest]->dim; j++) {
      dnames[j] = (char *)malloc(strlen(stmts[dep->dest]->iterators[j]) + 2);
      strcpy(dnames[j], stmts[dep->dest]->iterators[j]);
      strcat(dnames[j], "'");
    }
    pluto_constraints_set_names_range(
        dep->dpolytope, dnames, stmts[dep->src]->dim, 0, stmts[dep->dest]->dim);
    for (unsigned j = 0; j < stmts[dep->dest]->dim; j++) {
      free(dnames[j]);
    }
    free(dnames);

    pluto_constraints_set_names_range(
        dep->dpolytope, prog->params,
        stmts[dep->src]->dim + stmts[dep->dest]->dim, 0, npar);

    fprintf(stdout, "Dep type: %d\n", dep->type);

    switch (dep->type) {
    case PLUTO_DEP_RAW:
      spos = get_osl_write_access_position(candl_dep->stmt_source_ptr->access,
                                           candl_dep->ref_source_access_ptr);
      dep->src_acc = stmts[dep->src]->writes[spos];
      tpos = get_osl_read_access_position(candl_dep->stmt_target_ptr->access,
                                          candl_dep->ref_target_access_ptr);
      dep->dest_acc = stmts[dep->dest]->reads[tpos];

      break;
    case PLUTO_DEP_WAW:
      spos = get_osl_write_access_position(candl_dep->stmt_source_ptr->access,
                                           candl_dep->ref_source_access_ptr);
      dep->src_acc = stmts[dep->src]->writes[spos];
      tpos = get_osl_write_access_position(candl_dep->stmt_target_ptr->access,
                                           candl_dep->ref_target_access_ptr);
      dep->dest_acc = stmts[dep->dest]->writes[tpos];
      break;
    case PLUTO_DEP_WAR:
      spos = get_osl_read_access_position(candl_dep->stmt_source_ptr->access,
                                          candl_dep->ref_source_access_ptr);
      dep->src_acc = stmts[dep->src]->reads[spos];
      tpos = get_osl_write_access_position(candl_dep->stmt_target_ptr->access,
                                           candl_dep->ref_target_access_ptr);
      dep->dest_acc = stmts[dep->dest]->writes[tpos];
      break;
    case PLUTO_DEP_RAR:
      spos = get_osl_read_access_position(candl_dep->stmt_source_ptr->access,
                                          candl_dep->ref_source_access_ptr);
      dep->src_acc = stmts[dep->src]->reads[spos];
      tpos = get_osl_read_access_position(candl_dep->stmt_target_ptr->access,
                                          candl_dep->ref_target_access_ptr);
      dep->dest_acc = stmts[dep->dest]->reads[tpos];
      break;
    default:
      assert(0 && "unexpected dependence type");
    }

    /* Get rid of rows that are all zero */
    unsigned r, c;
    bool *remove = (bool *)malloc(sizeof(bool) * dep->dpolytope->nrows);
    for (r = 0; r < dep->dpolytope->nrows; r++) {
      for (c = 0; c < dep->dpolytope->ncols; c++) {
        if (dep->dpolytope->val[r][c] != 0) {
          break;
        }
      }
      if (c == dep->dpolytope->ncols) {
        remove[r] = true;
      } else {
        remove[r] = false;
      }
    }
    unsigned orig_nrows = dep->dpolytope->nrows;
    int del_count = 0;
    for (r = 0; r < orig_nrows; r++) {
      if (remove[r]) {
        pluto_constraints_remove_row(dep->dpolytope, r - del_count);
        del_count++;
      }
    }
    free(remove);

    int src_dim = stmts[dep->src]->dim;
    int target_dim = stmts[dep->dest]->dim;

    assert(candl_dep->source_nb_output_dims_domain +
               candl_dep->target_nb_output_dims_domain +
               candl_dep->stmt_source_ptr->domain->nb_parameters + 1 ==
           src_dim + target_dim + npar + 1);

    candl_dep = candl_dep->next;
  }

  return deps;
}

/// Extract necessary information from clan_scop to create PlutoProg - a
/// representation of the program sufficient to be used throughout Pluto.
/// PlutoProg also includes dependences; so candl is run here.
PlutoProg *osl_scop_to_pluto_prog(osl_scop_p scop, PlutoContext *context) {
  int i, max_sched_rows, npar;

  PlutoProg *prog = pluto_prog_alloc(context);
  PlutoOptions *options = context->options;

  /* Program parameters */
  npar = scop->context->nb_parameters;

  osl_strings_p osl_scop_params = NULL;
  if (npar >= 1)
    osl_scop_params = (osl_strings_p)scop->parameters->data;

  for (i = 0; i < npar; i++) {
    pluto_prog_add_param(prog, osl_scop_params->string[i], prog->npar);
  }

  pluto_constraints_free(prog->param_context);
  prog->param_context =
      osl_relation_to_pluto_constraints(scop->context, context);

  if (options->codegen_context != -1) {
    for (i = 0; i < prog->npar; i++) {
      pluto_constraints_add_inequality(prog->codegen_context);
      prog->codegen_context->val[i][i] = 1;
      prog->codegen_context->val[i][prog->codegen_context->ncols - 1] =
          -options->codegen_context;
    }
  }
  read_codegen_context_from_file(prog->codegen_context);

  prog->nstmts = osl_statement_number(scop->statement);

  /* Data variables in the program */
  osl_arrays_p arrays =
      (osl_arrays_p)osl_generic_lookup(scop->extension, OSL_URI_ARRAYS);
  if (arrays == NULL) {
    prog->num_data = 0;
    fprintf(stderr, "warning: arrays extension not found\n");
  } else {
    prog->num_data = arrays->nb_names;
    prog->data_names = (char **)malloc(prog->num_data * sizeof(char *));
    for (i = 0; i < prog->num_data; i++) {
      prog->data_names[i] = strdup(arrays->names[i]);
    }
  }

  osl_statement_p scop_stmt = scop->statement;

  prog->nvar = osl_statement_get_nb_iterators(scop_stmt);
  max_sched_rows = 0;
  for (i = 0; i < prog->nstmts; i++) {
    int stmt_num_iter = osl_statement_get_nb_iterators(scop_stmt);
    prog->nvar = PLMAX(prog->nvar, stmt_num_iter);
    max_sched_rows = PLMAX(max_sched_rows, scop_stmt->scattering->nb_rows);
    scop_stmt = scop_stmt->next;
  }

  prog->stmts = osl_to_pluto_stmts(scop, context);

  /* Compute dependences */
  if (options->isldep) {
    compute_deps_osl(scop, prog, options);
  } else {
    /*  Using Candl */
    candl_options_p candlOptions = candl_options_malloc();
    if (options->rar) {
      candlOptions->rar = 1;
    }
    /* No longer supported */
    candlOptions->lastwriter = options->lastwriter;
    candlOptions->scalar_privatization = options->scalpriv;
    // candlOptions->verbose = 1;

    /* Add more infos (depth, label, ...) */
    /* Needed by Candl */
    candl_scop_usr_init(scop);

    osl_dependence_p candl_deps = candl_dependence(scop, candlOptions);
    prog->deps = deps_read(candl_deps, prog);
    prog->ndeps = osl_nb_dependences(candl_deps);
    candl_options_free(candlOptions);
    osl_dependence_free(candl_deps);

    candl_scop_usr_cleanup(scop); // undo candl_scop_user_init

    prog->transdeps = NULL;
    prog->ntransdeps = 0;
  }

  /* Add hyperplanes */
  if (prog->nstmts >= 1) {
    for (i = 0; i < max_sched_rows; i++) {
      pluto_prog_add_hyperplane(prog, prog->num_hyperplanes, H_UNKNOWN);
      prog->hProps[prog->num_hyperplanes - 1].type =
          (i % 2) ? H_LOOP : H_SCALAR;
    }
  }

  /* Hack for linearized accesses */
  FILE *lfp = fopen(".linearized", "r");
  FILE *nlfp = fopen(".nonlinearized", "r");
  char tmpstr[256];
  char linearized[256];
  if (lfp && nlfp) {
    for (i = 0; i < prog->nstmts; i++) {
      rewind(lfp);
      rewind(nlfp);
      while (!feof(lfp) && !feof(nlfp)) {
        fgets(tmpstr, 256, nlfp);
        fgets(linearized, 256, lfp);
        if (strstr(tmpstr, prog->stmts[i]->text)) {
          prog->stmts[i]->text = (char *)realloc(
              prog->stmts[i]->text, sizeof(char) * (strlen(linearized) + 1));
          strcpy(prog->stmts[i]->text, linearized);
        }
      }
    }
    fclose(lfp);
    fclose(nlfp);
  }

  return prog;
}

#define DEBUG_TYPE "pluto-opt"

static void dump_isl(osl_scop_p scop, PlutoOptions *options) {
  int i, racc_num, wacc_num;
  int nstmts = osl_statement_number(scop->statement);
  isl_space *space;
  isl_space *param_space;
  isl_set *context;
  isl_union_map *dep_raw, *dep_war, *dep_waw, *dep_rar, *trans_dep_war;
  isl_union_map *trans_dep_waw;
  osl_statement_p stmt;
  osl_strings_p scop_params = NULL;

  if (!options->silent) {
    printf("[pluto] compute_deps (isl%s)\n",
           options->lastwriter ? " with lastwriter" : "");
  }

  isl_ctx *ctx = isl_ctx_alloc();
  assert(ctx);

  osl_names_p names = get_scop_names(scop);

  space = isl_space_set_alloc(ctx, scop->context->nb_parameters, 0);
  if (scop->context->nb_parameters) {
    scop_params = (osl_strings_p)scop->parameters->data;
    space = set_names(space, isl_dim_param, scop_params->string);
  }
  param_space = isl_space_params(isl_space_copy(space));
  context = osl_relation_to_isl_set(scop->context, param_space);

  if (!options->rar)
    dep_rar = isl_union_map_empty(isl_space_copy(space));
  isl_union_map *writes = isl_union_map_empty(isl_space_copy(space));
  isl_union_map *reads = isl_union_map_empty(isl_space_copy(space));
  isl_union_map *schedule = isl_union_map_empty(isl_space_copy(space));
  isl_union_map *empty = isl_union_map_empty(space);

  if (!options->isldepaccesswise) {
    /* Leads to fewer dependences. Each dependence may not have a unique
     * source/target access relating to it, since a union is taken
     * across all reads for a statement (and writes) for a particualr
     * array. Relationship between a dependence and associated dependent
     * data / array elements is lost, and some analyses may not work with
     * such a representation
     */
    for (i = 0, stmt = scop->statement; i < nstmts; ++i, stmt = stmt->next) {
      isl_set *dom;
      isl_map *schedule_i;
      isl_union_map *read_i;
      isl_union_map *write_i;
      char name[20];

      snprintf(name, sizeof(name), "S_%d", i);

      int niter = osl_statement_get_nb_iterators(stmt);
      space = isl_space_set_alloc(ctx, scop->context->nb_parameters, niter);
      if (scop->context->nb_parameters) {
        scop_params = (osl_strings_p)scop->parameters->data;
        space = set_names(space, isl_dim_param, scop_params->string);
      }
      if (niter) {
        osl_body_p stmt_body =
            (osl_body_p)osl_generic_lookup(stmt->extension, OSL_URI_BODY);
        space = set_names(space, isl_dim_set, stmt_body->iterators->string);
      }
      space = isl_space_set_tuple_name(space, isl_dim_set, name);
      dom = osl_relation_list_to_isl_set(stmt->domain, space);
      dom = isl_set_intersect_params(dom, isl_set_copy(context));

      space = isl_space_alloc(ctx, scop->context->nb_parameters, niter,
                              2 * niter + 1);
      if (scop->context->nb_parameters) {
        scop_params = (osl_strings_p)scop->parameters->data;
        space = set_names(space, isl_dim_param, scop_params->string);
      }
      if (niter) {
        osl_body_p stmt_body =
            (osl_body_p)osl_generic_lookup(stmt->extension, OSL_URI_BODY);
        space = set_names(space, isl_dim_in, stmt_body->iterators->string);
      }
      space = isl_space_set_tuple_name(space, isl_dim_in, name);
      schedule_i = osl_scattering_to_isl_map(stmt->scattering, space);

      osl_relation_list_p rlist = osl_access_list_filter_read(stmt->access);
      osl_relation_list_p wlist = osl_access_list_filter_write(stmt->access);

      osl_arrays_p arrays =
          (osl_arrays_p)osl_generic_lookup(scop->extension, OSL_URI_ARRAYS);
      if (arrays) {
        osl_strings_free(names->arrays);
        names->arrays = osl_arrays_to_strings(arrays);
      }

      read_i = osl_access_list_to_isl_union_map(rlist, isl_set_copy(dom),
                                                names->arrays->string);
      write_i = osl_access_list_to_isl_union_map(wlist, isl_set_copy(dom),
                                                 names->arrays->string);

      reads = isl_union_map_union(reads, read_i);
      writes = isl_union_map_union(writes, write_i);
      schedule =
          isl_union_map_union(schedule, isl_union_map_from_map(schedule_i));

      osl_relation_list_free(rlist);
      osl_relation_list_free(wlist);
    }
  } else {
    /* Each dependence is for a particular source and target access. Use
     * <stmt, access> pair while relating to accessed data so each
     * dependence can be associated to a unique source and target access
     */

    for (i = 0, stmt = scop->statement; i < nstmts; ++i, stmt = stmt->next) {
      isl_set *dom;

      racc_num = 0;
      wacc_num = 0;

      osl_relation_list_p access = stmt->access;
      for (; access; access = access->next) {
        isl_map *read_pos;
        isl_map *write_pos;
        isl_map *schedule_i;

        char name[25];

        if (access->elt->type == OSL_TYPE_READ) {
          snprintf(name, sizeof(name), "S_%d_r%d", i, racc_num);
        } else {
          snprintf(name, sizeof(name), "S_%d_w%d", i, wacc_num);
        }

        int niter = osl_statement_get_nb_iterators(stmt);
        space = isl_space_set_alloc(ctx, scop->context->nb_parameters, niter);
        if (scop->context->nb_parameters) {
          scop_params = (osl_strings_p)scop->parameters->data;
          space = set_names(space, isl_dim_param, scop_params->string);

          osl_strings_free(names->parameters);
          names->parameters = osl_strings_clone(scop_params);
        }
        if (niter) {
          osl_body_p stmt_body =
              (osl_body_p)osl_generic_lookup(stmt->extension, OSL_URI_BODY);
          space = set_names(space, isl_dim_set, stmt_body->iterators->string);

          osl_strings_free(names->iterators);
          names->iterators = osl_strings_clone(stmt_body->iterators);
        }
        space = isl_space_set_tuple_name(space, isl_dim_set, name);
        dom = osl_relation_list_to_isl_set(stmt->domain, space);
        dom = isl_set_intersect_params(dom, isl_set_copy(context));

        space = isl_space_alloc(ctx, scop->context->nb_parameters, niter,
                                2 * niter + 1);
        if (scop->context->nb_parameters) {
          scop_params = (osl_strings_p)scop->parameters->data;
          space = set_names(space, isl_dim_param, scop_params->string);
        }
        if (niter) {
          osl_body_p stmt_body =
              (osl_body_p)osl_generic_lookup(stmt->extension, OSL_URI_BODY);
          space = set_names(space, isl_dim_in, stmt_body->iterators->string);
        }
        space = isl_space_set_tuple_name(space, isl_dim_in, name);

        schedule_i = osl_scattering_to_isl_map(stmt->scattering, space);

        osl_arrays_p arrays =
            (osl_arrays_p)osl_generic_lookup(scop->extension, OSL_URI_ARRAYS);
        if (arrays) {
          osl_strings_free(names->arrays);
          names->arrays = osl_arrays_to_strings(arrays);
        }

        if (access->elt->type == OSL_TYPE_READ) {
          read_pos = osl_basic_access_to_isl_union_map(access->elt, dom,
                                                       names->arrays->string);
          reads = isl_union_map_union(reads, isl_union_map_from_map(read_pos));
        } else {
          write_pos = osl_basic_access_to_isl_union_map(access->elt, dom,
                                                        names->arrays->string);
          writes =
              isl_union_map_union(writes, isl_union_map_from_map(write_pos));
        }

        schedule =
            isl_union_map_union(schedule, isl_union_map_from_map(schedule_i));

        if (access->elt->type == OSL_TYPE_READ) {
          racc_num++;
        } else {
          wacc_num++;
        }
      }
    }
  }

  char *s;
  const char *str;
  // isl_pw_aff *pa;
  isl_printer *p;
  int equal;
  p = isl_printer_to_str(ctx);
  p = isl_printer_set_output_format(p, ISL_FORMAT_ISL);
  p = isl_printer_print_space(p, space);
  s = isl_printer_get_str(p);
  isl_printer_free(p);

  llvm::outs() << s << "\n";

  compute_deps_isl(reads, writes, schedule, empty, &dep_raw, &dep_war, &dep_waw,
                   &dep_rar, &trans_dep_war, &trans_dep_waw, options);

  if (options->lastwriter) {
    isl_union_map_free(trans_dep_war);
    isl_union_map_free(trans_dep_waw);
  }

  isl_union_map_free(dep_raw);
  isl_union_map_free(dep_war);
  isl_union_map_free(dep_waw);
  isl_union_map_free(dep_rar);

  isl_union_map_free(writes);
  isl_union_map_free(reads);
  isl_union_map_free(schedule);
  isl_union_map_free(empty);
  isl_set_free(context);

  if (names)
    osl_names_free(names);

  isl_ctx_free(ctx);
}

namespace polymer {
/// The main function that implements the Pluto based optimization.
/// TODO: transform options?
mlir::func::FuncOp tadashiTransform(mlir::func::FuncOp f, OpBuilder &rewriter) {
  LLVM_DEBUG(dbgs() << "Pluto transforming: \n");
  LLVM_DEBUG(f.dump());

  PlutoContext *context = pluto_context_alloc();
  OslSymbolTable srcTable, dstTable;

  std::unique_ptr<OslScop> scop = createOpenScopFromFuncOp(f, srcTable);
  if (!scop)
    return nullptr;
  if (scop->getNumStatements() == 0)
    return nullptr;

  PlutoOptions options;
  bool debug = true;
  options.silent = !debug;
  options.moredebug = debug;
  options.debug = debug;
  options.isldep = 1;
  options.readscop = 1;

  options.identity = 0;
  // options.parallel = parallelize;
  options.unrolljam = 0;
  options.prevector = 0;
  // options.diamondtile = diamondTiling;

  // if (cloogf != -1)
  //   context->options->cloogf = cloogf;
  // if (cloogl != -1)
  //   context->options->cloogl = cloogl;

  dump_isl(scop->get(), &options);

  return f;
}
} // namespace polymer

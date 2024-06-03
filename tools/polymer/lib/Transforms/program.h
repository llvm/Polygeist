/*
 * Pluto: An automatic parallelier and locality optimizer
 *
 * Copyright (C) 2007 Uday Bondhugula
 *
 * This software is available under the MIT license. Please see LICENSE
 * in the top-level directory for details.
 *
 * This file is part of libpluto.
 *
 */
#ifndef PROGRAM_H
#define PROGRAM_H

// Not replacing this with a forward declaration due to use of enum types
// PlutoStmtType and PlutoHypType.
#include "pluto/internal/pluto.h"
#include "isl/map.h"
#include "isl/union_map.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct plutoOptions PlutoOptions;

// Helper to extract access info from ISL structures.
struct pluto_access_meta_info {
  /* Pointer to an array of accesses */
  PlutoAccess ***accs;
  unsigned index;
  unsigned stmt_dim;
  int npar;
  PlutoContext *context;
};

Stmt *pluto_stmt_alloc(unsigned dim, const PlutoConstraints *domain,
                       const PlutoMatrix *mat);
void pluto_stmt_free(Stmt *stmt);
Stmt *pluto_stmt_dup(const Stmt *src);

void pluto_stmts_print(FILE *fp, Stmt **, int);
void pluto_stmt_print(FILE *fp, const Stmt *stmt);
void pluto_prog_print(FILE *fp, PlutoProg *prog);

Dep *pluto_dep_alloc();
void pluto_dep_print(FILE *fp, const Dep *dep);
void pluto_deps_print(FILE *, PlutoProg *prog);

PlutoProg *pluto_prog_alloc(PlutoContext *context);
void pluto_prog_free(PlutoProg *prog);

int get_coeff_upper_bound(PlutoProg *prog);

void pluto_prog_add_param(PlutoProg *prog, const char *param, int pos);
void pluto_add_stmt(PlutoProg *prog, const PlutoConstraints *domain,
                    const PlutoMatrix *trans, char **iterators,
                    const char *text, PlutoStmtType type);

void pluto_add_stmt_to_end(PlutoProg *prog, const PlutoConstraints *domain,
                           char **iterators, const char *text, int level,
                           PlutoStmtType type);

void pluto_stmt_add_dim(Stmt *stmt, unsigned pos, int time_pos,
                        const char *iter, PlutoHypType type, PlutoProg *prog);
void pluto_stmt_remove_dim(Stmt *stmt, unsigned pos, PlutoProg *prog);
void pluto_prog_add_hyperplane(PlutoProg *prog, int pos, PlutoHypType type);

int get_const_bound_difference(const PlutoConstraints *cst, int depth);
PlutoMatrix *get_alpha(const Stmt *stmt, const PlutoProg *prog);
PlutoMatrix *pluto_stmt_get_remapping(const Stmt *stmt, int **strides);

void get_parametric_extent(const PlutoConstraints *cst, int pos, int npar,
                           const char **params, char **extent, char **p_lbexpr);

void get_parametric_extent_const(const PlutoConstraints *cst, int pos, int npar,
                                 const char **params, char **extent,
                                 char **p_lbexpr);

char *get_parametric_bounding_box(const PlutoConstraints *cst, int start,
                                  int num, int npar, const char **params);

void pluto_separate_stmt(PlutoProg *prog, const Stmt *stmt, int level);
void pluto_separate_stmts(PlutoProg *prog, Stmt **stmts, int num, int level,
                          int offset);

bool pluto_is_hyperplane_scalar(const Stmt *stmt, int level);
bool pluto_is_depth_scalar(Ploop *loop, int depth);
int pluto_stmt_is_member_of(int stmt_id, Stmt **slist, int len);
PlutoAccess **pluto_get_all_waccs(const PlutoProg *prog, int *num);
int pluto_stmt_is_subset_of(Stmt **s1, int n1, Stmt **s2, int n2);
void pluto_stmt_add_hyperplane(Stmt *stmt, PlutoHypType type, unsigned pos);
PlutoMatrix *pluto_get_new_access_func(const PlutoMatrix *acc, const Stmt *stmt,
                                       int **divs);

int extract_deps_from_isl_union_map(__isl_keep isl_union_map *umap, Dep **deps,
                                    int first, Stmt **stmts, PlutoDepType type,
                                    PlutoContext *context);

int pluto_get_max_ind_hyps(const PlutoProg *prog);
int pluto_get_max_ind_hyps_non_scalar(const PlutoProg *prog);
unsigned pluto_stmt_get_num_ind_hyps(const Stmt *stmt);
int pluto_stmt_get_num_ind_hyps_non_scalar(const Stmt *stmt);
int pluto_transformations_full_ranked(PlutoProg *prog);
void pluto_pad_stmt_transformations(PlutoProg *prog);

void pluto_access_print(FILE *fp, const PlutoAccess *acc, const Stmt *stmt);
void pluto_transformations_print(const PlutoProg *prog);
void pluto_transformations_pretty_print(const PlutoProg *prog);
void pluto_print_hyperplane_properties(const PlutoProg *prog);
void pluto_stmt_transformation_print(const Stmt *stmt);
void pluto_stmt_print_hyperplane(FILE *fp, const Stmt *stmt, int level);
void pluto_transformation_print_level(const PlutoProg *prog, int level);

Stmt *pluto_stmt_dup(const Stmt *stmt);
PlutoAccess *pluto_access_dup(const PlutoAccess *acc);
void pluto_dep_free(Dep *dep);
Dep *pluto_dep_dup(Dep *d);
void pluto_remove_stmt(PlutoProg *prog, unsigned stmt_id);

int pluto_prog_get_largest_const_in_domains(const PlutoProg *prog);

void compute_deps_isl(isl_union_map *reads, isl_union_map *writes,
                      isl_union_map *schedule, isl_union_map *empty,
                      isl_union_map **dep_raw, isl_union_map **dep_war,
                      isl_union_map **dep_waw, isl_union_map **dep_rar,
                      isl_union_map **trans_dep_war,
                      isl_union_map **trans_dep_waw, PlutoOptions *options);

void extract_accesses_for_pluto_stmt(Stmt *stmt, isl_union_map *reads,
                                     isl_union_map *writes,
                                     PlutoContext *context);
isl_stat isl_map_extract_access_func(__isl_take isl_map *map, void *user);

int read_codegen_context_from_file(PlutoConstraints *codegen_context);

bool is_tile_space_loop(Ploop *loop, const PlutoProg *prog);
unsigned get_num_invariant_accesses(Ploop *loop);
unsigned get_num_accesses(Ploop *loop);
unsigned get_num_unique_accesses_in_stmts(Stmt **stmts, unsigned nstmts,
                                          const PlutoProg *prog);
unsigned get_num_invariant_accesses_in_stmts(Stmt **stmts, unsigned nstmts,
                                             unsigned depth,
                                             const PlutoProg *prog);
#if defined(__cplusplus)
}
#endif

#endif // PROGRAM_H

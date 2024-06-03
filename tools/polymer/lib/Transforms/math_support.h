/*
 * Pluto: An automatic parallelier and locality optimizer
 *
 * Copyright (C) 2007-2008 Uday Bondhugula
 *
 * This software is available under the MIT license. Please see LICENSE
 * in the top-level directory for details.
 *
 * This file is part of libpluto.
 *
 */
#ifndef _MATH_SUPPORT_H
#define _MATH_SUPPORT_H

#include <gmp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct plutoContext PlutoContext;
typedef struct pluto_matrix PlutoMatrix;

#define PLMAX(a, b) ((a >= b) ? (a) : (b))
#define PLMIN(a, b) ((a <= b) ? (a) : (b))
#define PLABS(a) ((a >= 0) ? (a) : (-a))

#if defined(__cplusplus)
extern "C" {
#endif

void pluto_matrix_print(FILE *, const PlutoMatrix *);
void pluto_matrix_read(FILE *, const PlutoMatrix *);
PlutoMatrix *pluto_matrix_alloc(int nrows, int ncols, PlutoContext *context);
void pluto_matrix_free(PlutoMatrix *mat);
PlutoMatrix *pluto_matrix_dup(const PlutoMatrix *src);
PlutoMatrix *pluto_matrix_identity(int size, PlutoContext *context);
void pluto_matrix_set(PlutoMatrix *mat, int val);
PlutoMatrix *pluto_matrix_input(FILE *fp, PlutoContext *context);

PlutoMatrix *pluto_matrix_inverse(PlutoMatrix *mat);
PlutoMatrix *pluto_matrix_product(const PlutoMatrix *mat1,
                                  const PlutoMatrix *mat2);
unsigned pluto_matrix_get_rank(const PlutoMatrix *mat);

void pluto_matrix_add_row(PlutoMatrix *mat, int pos);
void pluto_matrix_add_col(PlutoMatrix *mat, int pos);
void pluto_matrix_remove_row(PlutoMatrix *mat, int pos);
void pluto_matrix_remove_col(PlutoMatrix *, int);
void pluto_matrix_zero_row(PlutoMatrix *mat, int pos);
void pluto_matrix_zero_col(PlutoMatrix *mat, int pos);

void pluto_matrix_move_col(PlutoMatrix *mat, int r1, int r2);
void pluto_matrix_interchange_cols(PlutoMatrix *mat, int c1, int c2);
void pluto_matrix_interchange_rows(PlutoMatrix *mat, int r1, int r2);

void pluto_matrix_normalize_row(PlutoMatrix *mat, int pos);
void pluto_matrix_negate_row(PlutoMatrix *mat, int pos);
void pluto_matrix_add(PlutoMatrix *mat1, const PlutoMatrix *mat2);
void gaussian_eliminate(PlutoMatrix *mat, int start, int end);

int64_t lcm(int64_t a, int64_t b);
int64_t gcd(int64_t a, int64_t b);
int64_t *min_lexical(int64_t *a, int64_t *b, int64_t num);

char *concat(const char *prefix, const char *suffix);
void pluto_affine_function_print(FILE *fp, int64_t *func, int ndims,
                                 const char **vars);
char *pluto_affine_function_sprint(int64_t *func, int ndims, const char **vars);

void pluto_matrix_reverse_rows(PlutoMatrix *mat);
void pluto_matrix_negate(PlutoMatrix *mat);
bool are_pluto_matrices_equal(PlutoMatrix *mat1, PlutoMatrix *mat2);

int pluto_vector_is_parallel(PlutoMatrix *mat1, int r1, PlutoMatrix *mat2,
                             int r2);
int pluto_vector_is_normal(PlutoMatrix *mat1, int r1, PlutoMatrix *mat2,
                           int r2);

void mpz_set_sll(mpz_t n, long long sll);

#if defined(__cplusplus)
}
#endif

#endif

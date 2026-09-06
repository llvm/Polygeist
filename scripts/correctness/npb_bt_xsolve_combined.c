/*
 * Translation-only aggregation for NPB BT.
 *
 * Keeping the original solve helpers and x_solve in one translation unit lets
 * MLIR's inliner expose matvec_sub/matmul_sub inside the line-solve loops.
 * The source computations remain the unmodified upstream NPB routines; this
 * file defines no replacement or project-authored computational kernel.
 */
#include "solve_subs.c"
#include "x_solve.c"

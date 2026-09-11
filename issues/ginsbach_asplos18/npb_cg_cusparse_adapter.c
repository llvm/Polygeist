/*
 * Link adapter for the original NPB CG application.
 *
 * The benchmark keeps conj_grad file-local.  The validation build removes
 * that one definition from LLVM bitcode and resolves its existing call sites
 * here.  This adapter contains no computation: it only supplies the known NPB
 * extents to the source-faithful Polygeist-lifted routine.
 */
#include "npbparams.h"

#define NZ (NA * (NONZER + 1) * (NONZER + 1))

extern void npb_cg_conj_grad_core(
    int n, int nnz, int *colidx, int *rowstr, double *x, double *z,
    double *a, double *p, double *q, double *r, double *rnorm);

__attribute__((visibility("hidden")))
void conj_grad(int *colidx, int *rowstr, double *x, double *z, double *a,
               double *p, double *q, double *r, double *rnorm) {
  npb_cg_conj_grad_core(NA, NZ, colidx, rowstr, x, z, a, p, q, r, rnorm);
}

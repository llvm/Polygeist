/* mtop/mtop_solvers.cpp:376-410: 2D -dfem elasticity operator. */
#include "stage_kernels.h"

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
// polygeist-arg-extents mfem_app_mtop_iso_elasticity_dfem_2d: B=20, G=20, x=32*MFEM_BENCH_NE, lambda=25*MFEM_BENCH_NE, mu=25*MFEM_BENCH_NE, J=100*MFEM_BENCH_NE, weights=25, y=32*MFEM_BENCH_NE
void mfem_app_mtop_iso_elasticity_dfem_2d(
    const double *B, const double *G, const double *x,
    const double *lambda, const double *mu, const double *J,
    const double *weights, double *y) {
  double grad[100 * MFEM_BENCH_NE];
  double stress[100 * MFEM_BENCH_NE];
  mfem_interp_grad_2d_stage_sliced(x, B, G, grad);
  mfem_interp_grad_2d_stage_sliced(
      x + 16 * MFEM_BENCH_NE, B, G, grad + 50 * MFEM_BENCH_NE);
  mfem_elasticity_qpoint_2d_scalarized(
      lambda, mu, J, weights, grad, stress);
  mfem_integrate_grad_2d_stage_sliced(stress, B, G, y);
  mfem_integrate_grad_2d_stage_sliced(
      stress + 50 * MFEM_BENCH_NE, B, G, y + 16 * MFEM_BENCH_NE);
}

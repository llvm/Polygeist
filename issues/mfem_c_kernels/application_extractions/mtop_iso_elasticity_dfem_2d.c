/* mtop/mtop_solvers.cpp:376-410: 2D -dfem elasticity operator. */
#include "stage_kernels.h"

// polygeist-arg-extents mfem_app_mtop_iso_elasticity_dfem_2d: B=20, G=20, x=64, lambda=50, mu=50, J=200, weights=25, y=64
void mfem_app_mtop_iso_elasticity_dfem_2d(
    const double *B, const double *G, const double *x,
    const double *lambda, const double *mu, const double *J,
    const double *weights, double *y) {
  double grad[200];
  double stress[200];
  mfem_interp_grad_2d_stage_sliced(x, B, G, grad);
  mfem_interp_grad_2d_stage_sliced(x + 32, B, G, grad + 100);
  mfem_elasticity_qpoint_2d_scalarized(
      lambda, mu, J, weights, grad, stress);
  mfem_integrate_grad_2d_stage_sliced(stress, B, G, y);
  mfem_integrate_grad_2d_stage_sliced(stress + 100, B, G, y + 32);
}

/*
 * miniapps/dfem/dfem-minimal-surface.cpp:308-356.
 * The scalar field is represented by component zero of the existing padded
 * two-component extraction; component one is ignored.
 */
#include "stage_kernels.h"

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
// polygeist-arg-extents mfem_app_dfem_minimal_surface_2d: B=20, G=20, field_padded=16*MFEM_BENCH_NE, jacobian=50*MFEM_BENCH_NE, weights=25, y_padded=16*MFEM_BENCH_NE
void mfem_app_dfem_minimal_surface_2d(
    const double *B, const double *G, const double *field_padded,
    const double *jacobian, const double *weights, double *y_padded) {
  double grad[50 * MFEM_BENCH_NE];
  double flux[50 * MFEM_BENCH_NE];
  mfem_interp_grad_2d_stage_sliced(field_padded, B, G, grad);
  for (int qx = 0; qx < 5; ++qx) {
    for (int qy = 0; qy < 5; ++qy) {
      int p = qy + 5 * qx;
      double g0 = grad[p];
      double g1 = grad[p + 25];
      double j00 = jacobian[4 * p];
      double j01 = jacobian[4 * p + 1];
      double j10 = jacobian[4 * p + 2];
      double j11 = jacobian[4 * p + 3];
      double det = j00 * j11 - j01 * j10;
      double i00 = j11 / det;
      double i01 = -j01 / det;
      double i10 = -j10 / det;
      double i11 = j00 / det;
      double x0 = g0 * i00 + g1 * i10;
      double x1 = g0 * i01 + g1 * i11;
      double coeff = 1.0 / __builtin_sqrt(1.0 + x0 * x0 + x1 * x1);
      double scale = coeff * det * weights[p];
      flux[p] = scale * (x0 * i00 + x1 * i01);
      flux[p + 25] = scale * (x0 * i10 + x1 * i11);
      flux[p + 50] = 0.0;
      flux[p + 75] = 0.0;
    }
  }
  mfem_integrate_grad_2d_stage_sliced(flux, B, G, y_padded);
}

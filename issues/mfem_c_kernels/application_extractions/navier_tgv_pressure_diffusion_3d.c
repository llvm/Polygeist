/*
 * miniapps/fluids/navier/navier_solver.cpp:153-161: pressure Poisson operator.
 * Other Navier operators are vector/nonlinear families not yet extracted.
 */
#include "stage_kernels.h"

void mfem_app_navier_tgv_pressure_diffusion_3d(
    const double *B, const double *G, const double *Bt, const double *Gt,
    const double *op, const double *x, double *y) {
  mfem_pa_diffusion_apply_3d_stage_sliced(B, G, Bt, Gt, op, x, y);
}

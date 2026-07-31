/* miniapps/diag-smoothers/abs-l1-jacobi.cpp:279-309. */
#include "stage_kernels.h"

void mfem_app_abs_l1_mass_3d(const double *B, const double *Bt,
    const double *op, const double *x, double *y) {
  mfem_pa_mass_apply_3d_stage_sliced(B, Bt, op, x, y);
}

void mfem_app_abs_l1_diffusion_3d(
    const double *B, const double *G, const double *Bt, const double *Gt,
    const double *op, const double *x, double *y) {
  mfem_pa_diffusion_apply_3d_stage_sliced(B, G, Bt, Gt, op, x, y);
}

/* CurlCurl part only; the accompanying VectorFEMass is not yet extracted. */
void mfem_app_abs_l1_curlcurl_3d_partial(
    const double *Bo, const double *Bc, const double *Bot, const double *Bct,
    const double *G, const double *Gt, const double *op,
    const double *x, double *y) {
  mfem_pa_curlcurl_apply_3d_stage_sliced(
      Bo, Bc, Bot, Bct, G, Gt, op, x, y);
}

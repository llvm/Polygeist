/* examples/ex35p.cpp:369-388: the three partial-assembly problem branches. */
#include "stage_kernels.h"

void mfem_app_ex35p_h1_3d(
    const double *B, const double *G, const double *Bt, const double *Gt,
    const double *diff_op, const double *mass_op, const double *x, double *y) {
  mfem_pa_diffusion_apply_3d_stage_sliced(B, G, Bt, Gt, diff_op, x, y);
  mfem_pa_mass_apply_3d_stage_sliced(B, Bt, mass_op, x, y);
}

/* VectorFEMass is not yet present in the extraction corpus. */
void mfem_app_ex35p_hcurl_3d_partial(
    const double *Bo, const double *Bc, const double *Bot, const double *Bct,
    const double *G, const double *Gt, const double *curl_op,
    const double *x, double *y) {
  mfem_pa_curlcurl_apply_3d_stage_sliced(
      Bo, Bc, Bot, Bct, G, Gt, curl_op, x, y);
}

/* VectorFEMass is not yet present in the extraction corpus. */
void mfem_app_ex35p_hdiv_3d_partial(
    const double *Bo, const double *Bot, const double *G, const double *Gt,
    const double *div_op, const double *x, double *y) {
  mfem_pa_divdiv_apply_3d_stage_sliced(Bo, Bot, G, Gt, div_op, x, y);
}

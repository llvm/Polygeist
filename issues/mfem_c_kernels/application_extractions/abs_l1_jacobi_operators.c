/* miniapps/diag-smoothers/abs-l1-jacobi.cpp:279-309. */
#include "stage_kernels.h"

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
// polygeist-arg-extents mfem_app_abs_l1_mass_3d: B=20, Bt=20, op=125*MFEM_BENCH_NE, x=64*MFEM_BENCH_NE, y=64*MFEM_BENCH_NE
void mfem_app_abs_l1_mass_3d(const double *B, const double *Bt,
    const double *op, const double *x, double *y) {
  mfem_pa_mass_apply_3d_stage_sliced(B, Bt, op, x, y);
}

// polygeist-arg-extents mfem_app_abs_l1_diffusion_3d: B=20, G=20, Bt=20, Gt=20, op=750*MFEM_BENCH_NE, x=64*MFEM_BENCH_NE, y=64*MFEM_BENCH_NE
void mfem_app_abs_l1_diffusion_3d(
    const double *B, const double *G, const double *Bt, const double *Gt,
    const double *op, const double *x, double *y) {
  mfem_pa_diffusion_apply_3d_stage_sliced(B, G, Bt, Gt, op, x, y);
}

// polygeist-arg-extents mfem_app_abs_l1_curlcurl_3d: Bo=15, Bc=20, Bot=15, Bct=20, G=20, Gt=20, curl_op=750*MFEM_BENCH_NE, mass_op=750*MFEM_BENCH_NE, x=144*MFEM_BENCH_NE, y=144*MFEM_BENCH_NE
void mfem_app_abs_l1_curlcurl_3d(
    const double *Bo, const double *Bc, const double *Bot, const double *Bct,
    const double *G, const double *Gt, const double *curl_op,
    const double *mass_op,
    const double *x, double *y) {
  mfem_pa_curlcurl_apply_3d_stage_sliced(
      Bo, Bc, Bot, Bct, G, Gt, curl_op, x, y);
  mfem_pa_hcurl_mass_apply_3d_direct(Bo, Bc, Bot, Bct, mass_op, x, y);
}

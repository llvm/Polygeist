/* miniapps/hdiv-linear-solver/grad_div.cpp:203-206. */
#include "stage_kernels.h"

/* This is the DivDiv part. VectorFEMass remains to be extracted. */
void mfem_app_grad_div_3d_partial(
    const double *Bo, const double *Bot, const double *G, const double *Gt,
    const double *div_op, const double *x, double *y) {
  mfem_pa_divdiv_apply_3d_stage_sliced(Bo, Bot, G, Gt, div_op, x, y);
}

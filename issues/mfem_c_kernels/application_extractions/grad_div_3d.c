/* miniapps/hdiv-linear-solver/grad_div.cpp:203-206. */
#include "stage_kernels.h"

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
// polygeist-arg-extents mfem_app_grad_div_3d: Bo=15, Bc=20, Bot=15, Bct=20, G=20, Gt=20, div_op=125*MFEM_BENCH_NE, mass_op=750*MFEM_BENCH_NE, x=108*MFEM_BENCH_NE, y=108*MFEM_BENCH_NE
void mfem_app_grad_div_3d(
    const double *Bo, const double *Bc, const double *Bot, const double *Bct,
    const double *G, const double *Gt, const double *div_op,
    const double *mass_op, const double *x, double *y) {
  mfem_pa_divdiv_apply_3d_stage_sliced(Bo, Bot, G, Gt, div_op, x, y);
  mfem_pa_hdiv_mass_apply_3d_direct(Bo, Bc, Bot, Bct, mass_op, x, y);
}

#include "stage_kernels.h"

// Small correctness-isolation entry points for the two operators composed by
// mfem_app_ex35p_hdiv_3d.  They intentionally keep the application ABI so the
// same deterministic harness data can be used for each half independently.
// polygeist-arg-extents mfem_debug_hdiv_divdiv_only: Bo=15, Bc=20, Bot=15, Bct=20, G=20, Gt=20, div_op=250, mass_op=1500, x=216, y=216
void mfem_debug_hdiv_divdiv_only(
    const double *Bo, const double *Bc, const double *Bot, const double *Bct,
    const double *G, const double *Gt, const double *div_op,
    const double *mass_op, const double *x, double *y) {
  (void)Bc;
  (void)Bct;
  (void)mass_op;
  mfem_pa_divdiv_apply_3d_stage_sliced(Bo, Bot, G, Gt, div_op, x, y);
}

// polygeist-arg-extents mfem_debug_hdiv_mass_only: Bo=15, Bc=20, Bot=15, Bct=20, G=20, Gt=20, div_op=250, mass_op=1500, x=216, y=216
void mfem_debug_hdiv_mass_only(
    const double *Bo, const double *Bc, const double *Bot, const double *Bct,
    const double *G, const double *Gt, const double *div_op,
    const double *mass_op, const double *x, double *y) {
  (void)G;
  (void)Gt;
  (void)div_op;
  mfem_pa_hdiv_mass_apply_3d_direct(Bo, Bc, Bot, Bct, mass_op, x, y);
}

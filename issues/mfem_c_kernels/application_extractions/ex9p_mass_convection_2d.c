/*
 * Hot-path C extraction for MFEM examples/ex9p.cpp.
 *
 * Upstream application:
 *   examples/ex9p.cpp:388-404 (partial-assembly mass and convection forms)
 *   examples/ex9p.cpp:704-709 (FE_Evolution::Mult)
 *
 * This represents one element-batch evaluation of the two PA forms.  The
 * global true/local-DOF maps and iterative mass solve remain application
 * orchestration and are intentionally outside this numerical extraction.
 */

#include "stage_kernels.h"

// polygeist-arg-extents mfem_app_ex9p_mass_convection_2d: B=20, G=20, Bt=20, mass_op=50, convection_op=100, x=32, mass_y=32, convection_y=32
void mfem_app_ex9p_mass_convection_2d(
    const double *B, const double *G, const double *Bt,
    const double *mass_op, const double *convection_op, const double *x,
    double *mass_y, double *convection_y) {
  mfem_pa_mass_apply_2d_stage_sliced(B, Bt, mass_op, x, mass_y);
  mfem_pa_convection_apply_2d_stage_sliced(
      B, G, Bt, convection_op, x, convection_y);
}

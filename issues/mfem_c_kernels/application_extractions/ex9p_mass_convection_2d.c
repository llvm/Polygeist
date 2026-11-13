/*
 * Hot-path C extraction for MFEM examples/ex9p.cpp.
 *
 * Upstream application:
 *   examples/ex9p.cpp:388-404 (partial-assembly mass and convection forms)
 *   examples/ex9p.cpp:704-709 (FE_Evolution::Mult)
 *
 * This represents one element-batch evaluation of the two PA forms.  The
 * global true/local-DOF maps and solver convergence control remain application
 * orchestration.  One complete preconditioned-CG algebra iteration is included.
 */

#include "stage_kernels.h"

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
// polygeist-arg-extents mfem_app_ex9p_mass_convection_2d: B=20, G=20, Bt=20, mass_op=25*MFEM_BENCH_NE, convection_op=50*MFEM_BENCH_NE, x=16*MFEM_BENCH_NE, inv_diag=16*MFEM_BENCH_NE, mass_y=16*MFEM_BENCH_NE, convection_y=16*MFEM_BENCH_NE, residual=16*MFEM_BENCH_NE, preconditioned=16*MFEM_BENCH_NE, direction=16*MFEM_BENCH_NE, r_dot_z=1
void mfem_app_ex9p_mass_convection_2d(
    const double *B, const double *G, const double *Bt,
    const double *mass_op, const double *convection_op, const double *x,
    const double *inv_diag, double alpha, double beta,
    double *mass_y, double *convection_y, double *residual,
    double *preconditioned, double *direction, double *r_dot_z) {
  mfem_pa_mass_apply_2d_stage_sliced(B, Bt, mass_op, x, mass_y);
  mfem_pa_convection_apply_2d_stage_sliced(
      B, G, Bt, convection_op, x, convection_y);
  mfem_mass_pcg_step_2d(mass_y, inv_diag, alpha, beta, convection_y,
                        residual, preconditioned, direction, r_dot_z);
}

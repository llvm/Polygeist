/*
 * miniapps/fluids/navier/navier_solver.cpp:128-205: partial-assembly operators
 * used by the TGV time-step hot path.
 */
#include "stage_kernels.h"

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
// polygeist-arg-extents mfem_app_navier_tgv_pa_operators_3d: B=20, G=20, Bt=20, Gt=20, pressure_diff_op=750*MFEM_BENCH_NE, vector_mass_op=125*MFEM_BENCH_NE, vector_diff_op=2250*MFEM_BENCH_NE, convection_op=1125*MFEM_BENCH_NE, mixed_op=1125*MFEM_BENCH_NE, velocity_x=192*MFEM_BENCH_NE, pressure_x=64*MFEM_BENCH_NE, velocity_y=192*MFEM_BENCH_NE, pressure_y=64*MFEM_BENCH_NE
void mfem_app_navier_tgv_pa_operators_3d(
    const double *B, const double *G, const double *Bt, const double *Gt,
    const double *pressure_diff_op, const double *vector_mass_op,
    const double *vector_diff_op, const double *convection_op,
    const double *mixed_op, const double *velocity_x, const double *pressure_x,
    double *velocity_y, double *pressure_y) {
  mfem_pa_vector_mass_apply_3d_sliced(B, vector_mass_op, velocity_x, velocity_y);
  mfem_pa_vector_diffusion_apply_3d_sliced(
      B, G, vector_diff_op, velocity_x, velocity_y);
  mfem_pa_vector_convection_nl_apply_3d_sliced(
      B, G, convection_op, velocity_x, velocity_y);
  mfem_pa_diffusion_apply_3d_stage_sliced(
      B, G, Bt, Gt, pressure_diff_op, pressure_x, pressure_y);
  mfem_pa_discrete_gradient_apply_3d_sliced(
      B, G, mixed_op, pressure_x, velocity_y);
  mfem_pa_discrete_divergence_apply_3d_direct(
      B, G, mixed_op, velocity_x, pressure_y);
}

/* Independent baseline retaining the original direct loop algorithms.  The
 * correctness harness links this implementation separately and compares it
 * with the scratch-sliced, fully raisable entry above. */
void mfem_app_navier_tgv_pa_operators_3d_direct_reference(
    const double *B, const double *G, const double *Bt, const double *Gt,
    const double *pressure_diff_op, const double *vector_mass_op,
    const double *vector_diff_op, const double *convection_op,
    const double *mixed_op, const double *velocity_x, const double *pressure_x,
    double *velocity_y, double *pressure_y) {
  mfem_pa_vector_mass_apply_3d_direct(
      B, vector_mass_op, velocity_x, velocity_y);
  mfem_pa_vector_diffusion_apply_3d_direct(
      B, G, vector_diff_op, velocity_x, velocity_y);
  mfem_pa_vector_convection_nl_apply_3d_direct(
      B, G, convection_op, velocity_x, velocity_y);
  mfem_pa_diffusion_apply_3d_stage_sliced(
      B, G, Bt, Gt, pressure_diff_op, pressure_x, pressure_y);
  mfem_pa_discrete_gradient_apply_3d_direct(
      B, G, mixed_op, pressure_x, velocity_y);
  mfem_pa_discrete_divergence_apply_3d_direct(
      B, G, mixed_op, velocity_x, pressure_y);
}

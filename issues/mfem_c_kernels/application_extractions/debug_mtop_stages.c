#include "stage_kernels.h"

// polygeist-arg-extents mfem_debug_mtop_interp: B=20, G=20, x=64, unused3=50, unused4=50, unused5=200, unused6=25, out=200
void mfem_debug_mtop_interp(
    const double *B, const double *G, const double *x,
    const double *unused3, const double *unused4, const double *unused5,
    const double *unused6, double *out) {
  (void)unused3; (void)unused4; (void)unused5; (void)unused6;
  mfem_interp_grad_2d_stage_sliced(x, B, G, out);
  mfem_interp_grad_2d_stage_sliced(x + 32, B, G, out + 100);
}

// polygeist-arg-extents mfem_debug_mtop_qpoint: unused0=20, unused1=20, Q=200, lambda=50, mu=50, J=200, weights=25, out=200
void mfem_debug_mtop_qpoint(
    const double *unused0, const double *unused1, const double *Q,
    const double *lambda, const double *mu, const double *J,
    const double *weights, double *out) {
  (void)unused0; (void)unused1;
  mfem_elasticity_qpoint_2d_scalarized(lambda, mu, J, weights, Q, out);
}

// polygeist-arg-extents mfem_debug_mtop_integrate: B=20, G=20, unused2=64, unused3=50, unused4=50, stress=200, unused6=25, out=64
void mfem_debug_mtop_integrate(
    const double *B, const double *G, const double *unused2,
    const double *unused3, const double *unused4, const double *stress,
    const double *unused6, double *out) {
  (void)unused2; (void)unused3; (void)unused4; (void)unused6;
  mfem_integrate_grad_2d_stage_sliced(stress, B, G, out);
  mfem_integrate_grad_2d_stage_sliced(stress + 100, B, G, out + 32);
}

// polygeist-arg-extents mfem_debug_mtop_integrate_second: B=20, G=20, unused2=64, unused3=50, unused4=50, stress=200, unused6=25, out=32
void mfem_debug_mtop_integrate_second(
    const double *B, const double *G, const double *unused2,
    const double *unused3, const double *unused4, const double *stress,
    const double *unused6, double *out) {
  (void)unused2; (void)unused3; (void)unused4; (void)unused6;
  mfem_integrate_grad_2d_stage_sliced(stress + 100, B, G, out);
}

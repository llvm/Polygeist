/* Harness regression control; never used as a raised implementation. */
#include <math.h>
void mfem_app_ex9p_mass_convection_2d_reference(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double, double,
    double *, double *, double *, double *, double *, double *);
void mfem_app_ex9p_mass_convection_2d(
    const double *b, const double *g, const double *bt, const double *m,
    const double *c, const double *x, const double *d, double alpha, double beta,
    double *a0, double *a1, double *a2, double *a3, double *a4, double *a5) {
  mfem_app_ex9p_mass_convection_2d_reference(b, g, bt, m, c, x, d, alpha, beta,
                                          a0, a1, a2, a3, a4, a5);
#ifdef INJECT_NAN
  a0[0] = NAN;
#endif
#ifdef INJECT_INF
  a0[0] = INFINITY;
#endif
}

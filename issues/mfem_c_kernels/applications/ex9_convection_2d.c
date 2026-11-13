#include "bench_common.h"
#include "../original/convection_apply.c"

void mfem_pa_convection_apply_2d_stage_sliced(
    const double *, const double *, const double *, const double *,
    const double *, double *);

int main(void) {
  double b[20], g[20], bt[20], op[2 * 2 * 25], x[2 * 16];
  double y_ref[2 * 16], y_raised[2 * 16];
  bench_fill(b, 20); bench_fill(g, 20); bench_transpose(b, bt, 5, 4);
  bench_fill(op, 2 * 2 * 25); bench_fill(x, 2 * 16);
  RUN_AND_REPORT("ex9", "convection_2d",
    mfem_pa_convection_apply_2d(b, g, bt, op, x, y_ref),
    mfem_pa_convection_apply_2d_stage_sliced(b, g, bt, op, x, y_raised),
    2 * 16);
}

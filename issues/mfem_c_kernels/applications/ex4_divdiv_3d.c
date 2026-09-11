#include "bench_common.h"
#include "../original/de_rham_apply.c"

void mfem_pa_divdiv_apply_3d_stage_sliced(
    const double *, const double *, const double *, const double *,
    const double *, const double *, double *);

int main(void) {
  double bo[15], bot[15], g[20], gt[20], op[2 * 125];
  double x[2 * 3 * 3 * 3 * 4], y_ref[2 * 3 * 3 * 3 * 4];
  double y_raised[2 * 3 * 3 * 3 * 4];
  bench_fill(bo, 15); bench_fill(g, 20);
  bench_transpose(bo, bot, 5, 3); bench_transpose(g, gt, 5, 4);
  bench_fill(op, 2 * 125); bench_fill(x, 2 * 3 * 3 * 3 * 4);
  RUN_AND_REPORT("ex4", "divdiv_3d",
    mfem_pa_divdiv_apply_3d(bo, bot, g, gt, op, x, y_ref),
    mfem_pa_divdiv_apply_3d_stage_sliced(bo, bot, g, gt, op, x, y_raised),
    2 * 3 * 3 * 3 * 4);
}

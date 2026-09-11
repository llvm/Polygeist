#include "bench_common.h"
#include "../original/hcurl3_apply.c"

void mfem_pa_curlcurl_apply_3d_stage_sliced(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *, double *);

int main(void) {
  double bo[15], bc[20], bot[15], bct[20], g[20], gt[20];
  double op[2 * 6 * 125], x[2 * 3 * 3 * 4 * 4];
  double y_ref[2 * 3 * 3 * 4 * 4], y_raised[2 * 3 * 3 * 4 * 4];
  bench_fill(bo, 15); bench_fill(bc, 20); bench_fill(g, 20);
  bench_transpose(bo, bot, 5, 3); bench_transpose(bc, bct, 5, 4);
  bench_transpose(g, gt, 5, 4);
  bench_fill(op, 2 * 6 * 125); bench_fill(x, 2 * 3 * 3 * 4 * 4);
  RUN_AND_REPORT("ex3", "curlcurl_3d",
    mfem_pa_curlcurl_apply_3d(bo, bc, bot, bct, g, gt, op, x, y_ref),
    mfem_pa_curlcurl_apply_3d_stage_sliced(bo, bc, bot, bct, g, gt, op, x,
                                          y_raised),
    2 * 3 * 3 * 4 * 4);
}

#include "bench_common.h"
#include "../original/diffusion_apply.c"

void mfem_pa_diffusion_apply_3d_stage_sliced(
    const double *, const double *, const double *, const double *,
    const double *, const double *, double *);

int main(void) {
  double b[20], g[20], bt[20], gt[20], op[2 * 6 * 125], x[2 * 64];
  double y_ref[2 * 64], y_raised[2 * 64];
  bench_fill(b, 20); bench_fill(g, 20);
  bench_transpose(b, bt, 5, 4); bench_transpose(g, gt, 5, 4);
  bench_fill(op, 2 * 6 * 125); bench_fill(x, 2 * 64);
  RUN_AND_REPORT("ex1", "diffusion_3d",
    mfem_pa_diffusion_apply_3d(b, g, bt, gt, op, x, y_ref),
    mfem_pa_diffusion_apply_3d_stage_sliced(b, g, bt, gt, op, x, y_raised),
    2 * 64);
}

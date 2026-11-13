#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 1024
#endif

/* Compile the selected normalized source into this harness under a different
 * symbol.  The raised object supplies the original symbol, letting one process
 * compare every output element from identical input buffers. */
#if defined(BENCH_MASS_2D)
#define mfem_pa_mass_apply_2d_stage_sliced reference_kernel
#include "../normalized/mass_stage_sliced.c"
#undef mfem_pa_mass_apply_2d_stage_sliced
#define FUNCTION mfem_pa_mass_apply_2d_stage_sliced
#define TEST_NAME "mass_apply_2d"
#define OP_SIZE (25 * MFEM_BENCH_NE)
#define INPUT_SIZE (16 * MFEM_BENCH_NE)
#define OUTPUT_SIZE INPUT_SIZE
#define CALL(fn, out) fn(a[0], a[1], op, input, out)
#define TRANSPOSE_MASS
#elif defined(BENCH_MASS_3D)
#define mfem_pa_mass_apply_3d_stage_sliced reference_kernel
#include "../normalized/mass_stage_sliced.c"
#undef mfem_pa_mass_apply_3d_stage_sliced
#define FUNCTION mfem_pa_mass_apply_3d_stage_sliced
#define TEST_NAME "mass_apply_3d"
#define OP_SIZE (125 * MFEM_BENCH_NE)
#define INPUT_SIZE (64 * MFEM_BENCH_NE)
#define OUTPUT_SIZE INPUT_SIZE
#define CALL(fn, out) fn(a[0], a[1], op, input, out)
#define TRANSPOSE_MASS
#elif defined(BENCH_DIFFUSION_2D)
#define mfem_pa_diffusion_apply_2d_stage_sliced reference_kernel
#include "../normalized/diffusion_stage_sliced.c"
#undef mfem_pa_diffusion_apply_2d_stage_sliced
#define FUNCTION mfem_pa_diffusion_apply_2d_stage_sliced
#define TEST_NAME "diffusion_apply_2d"
#define OP_SIZE (75 * MFEM_BENCH_NE)
#define INPUT_SIZE (16 * MFEM_BENCH_NE)
#define OUTPUT_SIZE INPUT_SIZE
#define CALL(fn, out) fn(a[0], a[1], a[2], a[3], op, input, out)
#define TRANSPOSE_BG
#elif defined(BENCH_DIFFUSION_3D)
#define mfem_pa_diffusion_apply_3d_stage_sliced reference_kernel
#include "../normalized/diffusion_stage_sliced.c"
#undef mfem_pa_diffusion_apply_3d_stage_sliced
#define FUNCTION mfem_pa_diffusion_apply_3d_stage_sliced
#define TEST_NAME "diffusion_apply_3d"
#define OP_SIZE (750 * MFEM_BENCH_NE)
#define INPUT_SIZE (64 * MFEM_BENCH_NE)
#define OUTPUT_SIZE INPUT_SIZE
#define CALL(fn, out) fn(a[0], a[1], a[2], a[3], op, input, out)
#define TRANSPOSE_BG
#elif defined(BENCH_CONVECTION_2D)
#define mfem_pa_convection_apply_2d_stage_sliced reference_kernel
#include "../normalized/convection_stage_sliced.c"
#undef mfem_pa_convection_apply_2d_stage_sliced
#define FUNCTION mfem_pa_convection_apply_2d_stage_sliced
#define TEST_NAME "convection_apply_2d"
#define OP_SIZE (50 * MFEM_BENCH_NE)
#define INPUT_SIZE (16 * MFEM_BENCH_NE)
#define OUTPUT_SIZE INPUT_SIZE
#define CALL(fn, out) fn(a[0], a[1], a[2], op, input, out)
#define TRANSPOSE_BG
#elif defined(BENCH_CONVECTION_3D)
#define mfem_pa_convection_apply_3d_stage_sliced reference_kernel
#include "../normalized/convection_stage_sliced.c"
#undef mfem_pa_convection_apply_3d_stage_sliced
#define FUNCTION mfem_pa_convection_apply_3d_stage_sliced
#define TEST_NAME "convection_apply_3d"
#define OP_SIZE (375 * MFEM_BENCH_NE)
#define INPUT_SIZE (64 * MFEM_BENCH_NE)
#define OUTPUT_SIZE INPUT_SIZE
#define CALL(fn, out) fn(a[0], a[1], a[2], op, input, out)
#define TRANSPOSE_BG
#elif defined(BENCH_CURLCURL_2D)
#define mfem_pa_curlcurl_apply_2d_stage_sliced reference_kernel
#include "../normalized/de_rham2_stage_sliced.c"
#undef mfem_pa_curlcurl_apply_2d_stage_sliced
#define FUNCTION mfem_pa_curlcurl_apply_2d_stage_sliced
#define TEST_NAME "curlcurl_apply_2d"
#define OP_SIZE (25 * MFEM_BENCH_NE)
#define INPUT_SIZE (24 * MFEM_BENCH_NE)
#define OUTPUT_SIZE INPUT_SIZE
#define CALL(fn, out) fn(a[0], a[1], a[4], a[5], op, input, out)
#define TRANSPOSE_DERHAM2
#elif defined(BENCH_CURLCURL_3D)
#define mfem_pa_curlcurl_apply_3d_stage_sliced reference_kernel
#include "../normalized/curlcurl3_stage_sliced.c"
#undef mfem_pa_curlcurl_apply_3d_stage_sliced
#define FUNCTION mfem_pa_curlcurl_apply_3d_stage_sliced
#define TEST_NAME "curlcurl_apply_3d"
#define OP_SIZE (750 * MFEM_BENCH_NE)
#define INPUT_SIZE (144 * MFEM_BENCH_NE)
#define OUTPUT_SIZE INPUT_SIZE
#define CALL(fn, out) fn(a[0], a[2], a[1], a[3], a[4], a[5], op, input, out)
#define TRANSPOSE_DERHAM2
#elif defined(BENCH_DIVDIV_2D)
#define mfem_pa_divdiv_apply_2d_stage_sliced reference_kernel
#include "../normalized/de_rham2_stage_sliced.c"
#undef mfem_pa_divdiv_apply_2d_stage_sliced
#define FUNCTION mfem_pa_divdiv_apply_2d_stage_sliced
#define TEST_NAME "divdiv_apply_2d"
#define OP_SIZE (25 * MFEM_BENCH_NE)
#define INPUT_SIZE (24 * MFEM_BENCH_NE)
#define OUTPUT_SIZE INPUT_SIZE
#define CALL(fn, out) fn(a[0], a[1], a[2], a[3], op, input, out)
#define TRANSPOSE_DERHAM2
#elif defined(BENCH_DIVDIV_3D)
#define mfem_pa_divdiv_apply_3d_stage_sliced reference_kernel
#include "../normalized/divdiv3_stage_sliced.c"
#undef mfem_pa_divdiv_apply_3d_stage_sliced
#define FUNCTION mfem_pa_divdiv_apply_3d_stage_sliced
#define TEST_NAME "divdiv_apply_3d"
#define OP_SIZE (125 * MFEM_BENCH_NE)
#define INPUT_SIZE (108 * MFEM_BENCH_NE)
#define OUTPUT_SIZE INPUT_SIZE
#define CALL(fn, out) fn(a[0], a[1], a[2], a[3], op, input, out)
#define TRANSPOSE_DERHAM2
#elif defined(BENCH_INTERP_VALUE_2D)
#define mfem_interp_value_2d_scratch_sliced reference_kernel
#include "../normalized/value_scratch_sliced.c"
#undef mfem_interp_value_2d_scratch_sliced
#define FUNCTION mfem_interp_value_2d_scratch_sliced
#define TEST_NAME "interp_value_2d"
#define OP_SIZE 1
#define INPUT_SIZE (16 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (25 * MFEM_BENCH_NE)
#define CALL(fn, out) fn(input, a[0], out)
#elif defined(BENCH_INTERP_VALUE_3D)
#define mfem_interp_value_3d_scratch_sliced reference_kernel
#include "../normalized/value_scratch_sliced.c"
#undef mfem_interp_value_3d_scratch_sliced
#define FUNCTION mfem_interp_value_3d_scratch_sliced
#define TEST_NAME "interp_value_3d"
#define OP_SIZE 1
#define INPUT_SIZE (64 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (125 * MFEM_BENCH_NE)
#define CALL(fn, out) fn(input, a[0], out)
#elif defined(BENCH_INTEGRATE_VALUE_2D)
#define mfem_integrate_value_2d_scratch_sliced reference_kernel
#include "../normalized/value_scratch_sliced.c"
#undef mfem_integrate_value_2d_scratch_sliced
#define FUNCTION mfem_integrate_value_2d_scratch_sliced
#define TEST_NAME "integrate_value_2d"
#define OP_SIZE 1
#define INPUT_SIZE (25 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (16 * MFEM_BENCH_NE)
#define CALL(fn, out) fn(input, a[0], out)
#elif defined(BENCH_INTEGRATE_VALUE_3D)
#define mfem_integrate_value_3d_scratch_sliced reference_kernel
#include "../normalized/value_scratch_sliced.c"
#undef mfem_integrate_value_3d_scratch_sliced
#define FUNCTION mfem_integrate_value_3d_scratch_sliced
#define TEST_NAME "integrate_value_3d"
#define OP_SIZE 1
#define INPUT_SIZE (125 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (64 * MFEM_BENCH_NE)
#define CALL(fn, out) fn(input, a[0], out)
#elif defined(BENCH_INTERP_GRAD_2D)
#define mfem_interp_grad_2d_stage_sliced reference_kernel
#include "../normalized/gradient_stage_sliced.c"
#undef mfem_interp_grad_2d_stage_sliced
#define FUNCTION mfem_interp_grad_2d_stage_sliced
#define TEST_NAME "interp_grad_2d"
#define OP_SIZE 1
#define INPUT_SIZE (16 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (50 * MFEM_BENCH_NE)
#define CALL(fn, out) fn(input, a[0], a[1], out)
#elif defined(BENCH_INTERP_GRAD_3D)
#define mfem_interp_grad_3d_stage_sliced reference_kernel
#include "../normalized/gradient_stage_sliced.c"
#undef mfem_interp_grad_3d_stage_sliced
#define FUNCTION mfem_interp_grad_3d_stage_sliced
#define TEST_NAME "interp_grad_3d"
#define OP_SIZE 1
#define INPUT_SIZE (64 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (375 * MFEM_BENCH_NE)
#define CALL(fn, out) fn(input, a[0], a[1], out)
#elif defined(BENCH_INTEGRATE_GRAD_2D)
#define mfem_integrate_grad_2d_stage_sliced reference_kernel
#include "../normalized/gradient_stage_sliced.c"
#undef mfem_integrate_grad_2d_stage_sliced
#define FUNCTION mfem_integrate_grad_2d_stage_sliced
#define TEST_NAME "integrate_grad_2d"
#define OP_SIZE 1
#define INPUT_SIZE (50 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (16 * MFEM_BENCH_NE)
#define CALL(fn, out) fn(input, a[0], a[1], out)
#elif defined(BENCH_INTEGRATE_GRAD_3D)
#define mfem_integrate_grad_3d_stage_sliced reference_kernel
#include "../normalized/gradient_stage_sliced.c"
#undef mfem_integrate_grad_3d_stage_sliced
#define FUNCTION mfem_integrate_grad_3d_stage_sliced
#define TEST_NAME "integrate_grad_3d"
#define OP_SIZE 1
#define INPUT_SIZE (375 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (64 * MFEM_BENCH_NE)
#define CALL(fn, out) fn(input, a[0], a[1], out)
#else
#error "Select one supported BENCH_* operator"
#endif

extern void FUNCTION();

static double a[6][20];
static double op[OP_SIZE];
static double input[INPUT_SIZE];
static double actual[OUTPUT_SIZE], expected[OUTPUT_SIZE];

static double value(int i, int salt) {
  return (double)(((i * 17 + salt * 13 + 5) % 101) - 50) / 257.0;
}

static void transpose(double *dst, const double *src, int rows, int cols) {
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
      dst[col * rows + row] = src[row * cols + col];
}

int main(void) {
  for (int j = 0; j < 6; ++j)
    for (int i = 0; i < 20; ++i) a[j][i] = value(i, j + 1);
#if defined(TRANSPOSE_MASS)
  transpose(a[1], a[0], 5, 4);
#elif defined(TRANSPOSE_DERHAM2)
  transpose(a[1], a[0], 5, 3);
  transpose(a[3], a[2], 5, 4);
  transpose(a[5], a[4], 5, 4);
#elif defined(TRANSPOSE_BG)
  transpose(a[2], a[0], 5, 4);
  transpose(a[3], a[1], 5, 4);
#endif
  for (int i = 0; i < OP_SIZE; ++i) op[i] = value(i, 7);
  for (int i = 0; i < INPUT_SIZE; ++i) input[i] = value(i, 8);
  for (int i = 0; i < OUTPUT_SIZE; ++i)
    actual[i] = expected[i] = value(i, 9);

  CALL(reference_kernel, expected);
  CALL(FUNCTION, actual);

  double max_abs = 0.0, max_rel = 0.0;
  size_t failures = 0, worst = 0;
  for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
    const double abs_error = fabs(actual[i] - expected[i]);
    const double rel_error = abs_error / fmax(1.0, fabs(expected[i]));
    if (abs_error > max_abs) { max_abs = abs_error; worst = i; }
    if (rel_error > max_rel) max_rel = rel_error;
    if (!isfinite(actual[i]) ||
        abs_error > 1.0e-11 + 1.0e-10 * fabs(expected[i])) ++failures;
  }
  printf("kernel=%s ne=%d elements=%d correctness=%s failures=%zu "
         "max_abs=%.17g max_rel=%.17g worst=%zu actual=%.17g expected=%.17g\n",
         TEST_NAME, MFEM_BENCH_NE, OUTPUT_SIZE, failures ? "FAIL" : "PASS",
         failures, max_abs, max_rel, worst, actual[worst], expected[worst]);
  return failures ? 1 : 0;
}

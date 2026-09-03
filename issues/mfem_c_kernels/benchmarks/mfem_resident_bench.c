#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 1024
#endif
#ifndef BENCH_ITERS
#define BENCH_ITERS 20
#endif

#define MAX_OP (750 * MFEM_BENCH_NE)
#define MAX_X (375 * MFEM_BENCH_NE)

static double a[6][20];
static double op[MAX_OP];
static double x[MAX_X];
static double y[MAX_X];

static double value(int i, int salt) {
  return (double)(((i * 17 + salt * 13 + 5) % 101) - 50) / 257.0;
}

static void transpose(double *dst, const double *src, int rows, int cols) {
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
      dst[col * rows + row] = src[row * cols + col];
}

static double seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

#if defined(BENCH_MASS_2D)
#define BENCH_NAME "mass_apply_2d"
#define FUNCTION mfem_pa_mass_apply_2d_stage_sliced
#define OP_SIZE (25 * MFEM_BENCH_NE)
#define X_SIZE (16 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *,
                     const double *, double *);
static void run(void) { FUNCTION(a[0], a[1], op, x, y); }
#elif defined(BENCH_MASS_3D)
#define BENCH_NAME "mass_apply_3d"
#define FUNCTION mfem_pa_mass_apply_3d_stage_sliced
#define OP_SIZE (125 * MFEM_BENCH_NE)
#define X_SIZE (64 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *,
                     const double *, double *);
static void run(void) { FUNCTION(a[0], a[1], op, x, y); }
#elif defined(BENCH_DIFFUSION_2D)
#define BENCH_NAME "diffusion_apply_2d"
#define FUNCTION mfem_pa_diffusion_apply_2d_stage_sliced
#define OP_SIZE (75 * MFEM_BENCH_NE)
#define X_SIZE (16 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *,
                     const double *, const double *, const double *, double *);
static void run(void) { FUNCTION(a[0], a[1], a[2], a[3], op, x, y); }
#elif defined(BENCH_DIFFUSION_3D)
#define BENCH_NAME "diffusion_apply_3d"
#define FUNCTION mfem_pa_diffusion_apply_3d_stage_sliced
#define OP_SIZE (750 * MFEM_BENCH_NE)
#define X_SIZE (64 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *,
                     const double *, const double *, const double *, double *);
static void run(void) { FUNCTION(a[0], a[1], a[2], a[3], op, x, y); }
#elif defined(BENCH_CONVECTION_2D)
#define BENCH_NAME "convection_apply_2d"
#define FUNCTION mfem_pa_convection_apply_2d_stage_sliced
#define OP_SIZE (50 * MFEM_BENCH_NE)
#define X_SIZE (16 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *,
                     const double *, const double *, double *);
static void run(void) { FUNCTION(a[0], a[1], a[2], op, x, y); }
#elif defined(BENCH_CONVECTION_3D)
#define BENCH_NAME "convection_apply_3d"
#define FUNCTION mfem_pa_convection_apply_3d_stage_sliced
#define OP_SIZE (375 * MFEM_BENCH_NE)
#define X_SIZE (64 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *,
                     const double *, const double *, double *);
static void run(void) { FUNCTION(a[0], a[1], a[2], op, x, y); }
#elif defined(BENCH_CURLCURL_2D)
#define BENCH_NAME "curlcurl_apply_2d"
#define FUNCTION mfem_pa_curlcurl_apply_2d_stage_sliced
#define OP_SIZE (25 * MFEM_BENCH_NE)
#define X_SIZE (24 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *,
                     const double *, const double *, const double *, double *);
static void run(void) { FUNCTION(a[0], a[1], a[4], a[5], op, x, y); }
#elif defined(BENCH_CURLCURL_3D)
#define BENCH_NAME "curlcurl_apply_3d"
#define FUNCTION mfem_pa_curlcurl_apply_3d_stage_sliced
#define OP_SIZE (750 * MFEM_BENCH_NE)
#define X_SIZE (144 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *,
                     const double *, const double *, const double *,
                     const double *, const double *, double *);
static void run(void) {
  FUNCTION(a[0], a[2], a[1], a[3], a[4], a[5], op, x, y);
}
#elif defined(BENCH_DIVDIV_2D)
#define BENCH_NAME "divdiv_apply_2d"
#define FUNCTION mfem_pa_divdiv_apply_2d_stage_sliced
#define OP_SIZE (25 * MFEM_BENCH_NE)
#define X_SIZE (24 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *,
                     const double *, const double *, const double *, double *);
static void run(void) { FUNCTION(a[0], a[1], a[2], a[3], op, x, y); }
#elif defined(BENCH_DIVDIV_3D)
#define BENCH_NAME "divdiv_apply_3d"
#define FUNCTION mfem_pa_divdiv_apply_3d_stage_sliced
#define OP_SIZE (125 * MFEM_BENCH_NE)
#define X_SIZE (108 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *,
                     const double *, const double *, const double *, double *);
static void run(void) { FUNCTION(a[0], a[1], a[2], a[3], op, x, y); }
#elif defined(BENCH_INTERP_VALUE_2D)
#define BENCH_NAME "interp_value_2d"
#define FUNCTION mfem_interp_value_2d_scratch_sliced
#define INPUT_SIZE (16 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (25 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, double *);
static void run(void) { FUNCTION(x, a[0], y); }
#elif defined(BENCH_INTERP_VALUE_3D)
#define BENCH_NAME "interp_value_3d"
#define FUNCTION mfem_interp_value_3d_scratch_sliced
#define INPUT_SIZE (64 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (125 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, double *);
static void run(void) { FUNCTION(x, a[0], y); }
#elif defined(BENCH_INTEGRATE_VALUE_2D)
#define BENCH_NAME "integrate_value_2d"
#define FUNCTION mfem_integrate_value_2d_scratch_sliced
#define INPUT_SIZE (25 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (16 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, double *);
static void run(void) { FUNCTION(x, a[0], y); }
#elif defined(BENCH_INTEGRATE_VALUE_3D)
#define BENCH_NAME "integrate_value_3d"
#define FUNCTION mfem_integrate_value_3d_scratch_sliced
#define INPUT_SIZE (125 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (64 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, double *);
static void run(void) { FUNCTION(x, a[0], y); }
#elif defined(BENCH_INTERP_GRAD_2D)
#define BENCH_NAME "interp_grad_2d"
#define FUNCTION mfem_interp_grad_2d_stage_sliced
#define INPUT_SIZE (16 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (50 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *, double *);
static void run(void) { FUNCTION(x, a[0], a[1], y); }
#elif defined(BENCH_INTERP_GRAD_3D)
#define BENCH_NAME "interp_grad_3d"
#define FUNCTION mfem_interp_grad_3d_stage_sliced
#define INPUT_SIZE (64 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (375 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *, double *);
static void run(void) { FUNCTION(x, a[0], a[1], y); }
#elif defined(BENCH_INTEGRATE_GRAD_2D)
#define BENCH_NAME "integrate_grad_2d"
#define FUNCTION mfem_integrate_grad_2d_stage_sliced
#define INPUT_SIZE (50 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (16 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *, double *);
static void run(void) { FUNCTION(x, a[0], a[1], y); }
#elif defined(BENCH_INTEGRATE_GRAD_3D)
#define BENCH_NAME "integrate_grad_3d"
#define FUNCTION mfem_integrate_grad_3d_stage_sliced
#define INPUT_SIZE (375 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (64 * MFEM_BENCH_NE)
extern void FUNCTION(const double *, const double *, const double *, double *);
static void run(void) { FUNCTION(x, a[0], a[1], y); }
#else
#error "Select one BENCH_* operator"
#endif

#ifndef INPUT_SIZE
#define INPUT_SIZE X_SIZE
#endif
#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE X_SIZE
#endif
#ifndef OP_SIZE
#define OP_SIZE 0
#endif

int main(void) {
  for (int j = 0; j < 6; ++j)
    for (int i = 0; i < 20; ++i) a[j][i] = value(0, j);
#if defined(BENCH_CURLCURL_2D) || defined(BENCH_CURLCURL_3D)
  transpose(a[1], a[0], 5, 3);
  transpose(a[3], a[2], 5, 4);
  transpose(a[5], a[4], 5, 4);
#elif defined(BENCH_DIVDIV_2D) || defined(BENCH_DIVDIV_3D)
  transpose(a[1], a[0], 5, 3);
  transpose(a[3], a[2], 5, 4);
#elif defined(BENCH_MASS_2D) || defined(BENCH_MASS_3D)
  transpose(a[1], a[0], 5, 4);
#else
  transpose(a[2], a[0], 5, 4);
  transpose(a[3], a[1], 5, 4);
#endif
  for (int i = 0; i < OP_SIZE; ++i) op[i] = value(i, 6);
  for (int i = 0; i < INPUT_SIZE; ++i) x[i] = value(i, 7);

  memset(y, 0, OUTPUT_SIZE * sizeof(double));
  run();
  double checksum = 0.0, max_abs = 0.0;
  for (int i = 0; i < OUTPUT_SIZE; ++i) {
    checksum += y[i];
    max_abs = fmax(max_abs, fabs(y[i]));
  }

  /* Resident-style best-of-N: operands are zero-copy-registered by the runtime
     (cudaHostRegisterMapped) on first call, so there is no per-iteration H2D/D2H
     copy.  Warm up (plan cache + registration), then time N single applies each
     fenced by a full device sync, and report the BEST (jitter-free) execution
     time -- matching the ATen device-resident methodology as closely as the
     partial lowering (residual host linalg loops stay on CPU) allows. */
  extern int cudaDeviceSynchronize(void);
  memset(y, 0, OUTPUT_SIZE * sizeof(double));
  for (int w = 0; w < 3; ++w) run();
  cudaDeviceSynchronize();
  double best_us = 1.0e30;
  for (int i = 0; i < BENCH_ITERS; ++i) {
    const double t0 = seconds();
    run();
    cudaDeviceSynchronize();
    const double us = (seconds() - t0) * 1.0e6;
    if (us < best_us) best_us = us;
  }
  printf("implementation=polygeist_raised_resident kernel=%s ne=%d iterations=%d "
         "runtime_us=%.6f checksum=%.17g max_abs=%.17g "
         "timing_scope=resident_best_of_n\n",
         BENCH_NAME, MFEM_BENCH_NE, BENCH_ITERS, best_us, checksum, max_abs);
  return isfinite(checksum) ? 0 : 1;
}

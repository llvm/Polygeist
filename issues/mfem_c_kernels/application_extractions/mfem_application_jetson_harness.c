#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 20
#endif
#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif

#define MAX_EXTENT (2250 * MFEM_BENCH_NE)
#define MAX_OUTPUTS 6

static double args[13][MAX_EXTENT];
static double initial_state[MAX_OUTPUTS][MAX_EXTENT];
static double reference_state[MAX_OUTPUTS][MAX_EXTENT];
static double raised_state[MAX_OUTPUTS][MAX_EXTENT];

#if defined(MFEM_APP_MTOP)
#define APP_NAME "mtop_iso_elasticity_dfem_2d"
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {32 * MFEM_BENCH_NE};
extern void mfem_app_mtop_iso_elasticity_dfem_2d(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double *);
extern void mfem_app_mtop_iso_elasticity_dfem_2d_reference(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_mtop_iso_elasticity_dfem_2d(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_mtop_iso_elasticity_dfem_2d_reference(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], state[0]);
}
#elif defined(MFEM_APP_MINIMAL_SURFACE)
#define APP_NAME "dfem_minimal_surface_2d"
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {16 * MFEM_BENCH_NE};
extern void mfem_app_dfem_minimal_surface_2d(
    const double *, const double *, const double *, const double *,
    const double *, double *);
extern void mfem_app_dfem_minimal_surface_2d_reference(
    const double *, const double *, const double *, const double *,
    const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_dfem_minimal_surface_2d(
      args[0], args[1], args[2], args[3], args[4], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_dfem_minimal_surface_2d_reference(
      args[0], args[1], args[2], args[3], args[4], state[0]);
}
#elif defined(MFEM_APP_EX35P_H1)
#define APP_NAME "ex35p_h1_3d"
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {64 * MFEM_BENCH_NE};
extern void mfem_app_ex35p_h1_3d(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double *);
extern void mfem_app_ex35p_h1_3d_reference(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_ex35p_h1_3d(args[0], args[1], args[2], args[3], args[4], args[5],
                       args[6], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_ex35p_h1_3d_reference(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], state[0]);
}
#elif defined(MFEM_APP_EX35P_HCURL)
#define APP_NAME "ex35p_hcurl_3d"
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {144 * MFEM_BENCH_NE};
extern void mfem_app_ex35p_hcurl_3d(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *);
extern void mfem_app_ex35p_hcurl_3d_reference(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_ex35p_hcurl_3d(args[0], args[1], args[2], args[3], args[4], args[5],
                          args[6], args[7], args[8], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_ex35p_hcurl_3d_reference(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      args[8], state[0]);
}
#elif defined(MFEM_APP_EX35P_HDIV)
#define APP_NAME "ex35p_hdiv_3d"
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {108 * MFEM_BENCH_NE};
extern void mfem_app_ex35p_hdiv_3d(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *);
extern void mfem_app_ex35p_hdiv_3d_reference(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_ex35p_hdiv_3d(args[0], args[1], args[2], args[3], args[4], args[5],
                         args[6], args[7], args[8], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_ex35p_hdiv_3d_reference(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      args[8], state[0]);
}
#elif defined(MFEM_DEBUG_HDIV_DIVDIV) || defined(MFEM_DEBUG_HDIV_MASS)
#if defined(MFEM_DEBUG_HDIV_DIVDIV)
#define APP_NAME "debug_hdiv_divdiv_only"
#define DEBUG_HDIV_FUNCTION mfem_debug_hdiv_divdiv_only
#else
#define APP_NAME "debug_hdiv_mass_only"
#define DEBUG_HDIV_FUNCTION mfem_debug_hdiv_mass_only
#endif
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {108 * MFEM_BENCH_NE};
extern void DEBUG_HDIV_FUNCTION(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *);
#define DEBUG_HDIV_REFERENCE_IMPL(name) name##_reference
#define DEBUG_HDIV_REFERENCE(name) DEBUG_HDIV_REFERENCE_IMPL(name)
extern void DEBUG_HDIV_REFERENCE(DEBUG_HDIV_FUNCTION)(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  DEBUG_HDIV_FUNCTION(args[0], args[1], args[2], args[3], args[4], args[5],
                      args[6], args[7], args[8], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  DEBUG_HDIV_REFERENCE(DEBUG_HDIV_FUNCTION)(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      args[8], state[0]);
}
#elif defined(MFEM_DEBUG_MTOP_INTERP) || defined(MFEM_DEBUG_MTOP_QPOINT) || defined(MFEM_DEBUG_MTOP_INTEGRATE) || defined(MFEM_DEBUG_MTOP_INTEGRATE_SECOND)
#if defined(MFEM_DEBUG_MTOP_INTERP)
#define APP_NAME "debug_mtop_interp"
#define DEBUG_MTOP_FUNCTION mfem_debug_mtop_interp
#define DEBUG_MTOP_EXTENT 200
#elif defined(MFEM_DEBUG_MTOP_QPOINT)
#define APP_NAME "debug_mtop_qpoint"
#define DEBUG_MTOP_FUNCTION mfem_debug_mtop_qpoint
#define DEBUG_MTOP_EXTENT 200
#elif defined(MFEM_DEBUG_MTOP_INTEGRATE)
#define APP_NAME "debug_mtop_integrate"
#define DEBUG_MTOP_FUNCTION mfem_debug_mtop_integrate
#define DEBUG_MTOP_EXTENT 64
#else
#define APP_NAME "debug_mtop_integrate_second"
#define DEBUG_MTOP_FUNCTION mfem_debug_mtop_integrate_second
#define DEBUG_MTOP_EXTENT 32
#endif
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {DEBUG_MTOP_EXTENT};
extern void DEBUG_MTOP_FUNCTION(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double *);
#define DEBUG_MTOP_REFERENCE_IMPL(name) name##_reference
#define DEBUG_MTOP_REFERENCE(name) DEBUG_MTOP_REFERENCE_IMPL(name)
extern void DEBUG_MTOP_REFERENCE(DEBUG_MTOP_FUNCTION)(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  DEBUG_MTOP_FUNCTION(args[0], args[1], args[2], args[3], args[4], args[5],
                      args[6], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  DEBUG_MTOP_REFERENCE(DEBUG_MTOP_FUNCTION)(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], state[0]);
}
#elif defined(MFEM_APP_EX9P)
#define APP_NAME "ex9p_mass_convection_2d"
#define OUTPUT_COUNT 6
static const int64_t output_extents[] = {
    16 * MFEM_BENCH_NE, 16 * MFEM_BENCH_NE, 16 * MFEM_BENCH_NE,
    16 * MFEM_BENCH_NE, 16 * MFEM_BENCH_NE, 1};
extern void mfem_app_ex9p_mass_convection_2d(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double, double, double *,
    double *, double *, double *, double *, double *);
extern void mfem_app_ex9p_mass_convection_2d_reference(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double, double, double *,
    double *, double *, double *, double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_ex9p_mass_convection_2d(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], 0.125,
      0.25, state[0], state[1], state[2], state[3], state[4], state[5]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_ex9p_mass_convection_2d_reference(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], 0.125,
      0.25, state[0], state[1], state[2], state[3], state[4], state[5]);
}
#elif defined(MFEM_APP_GRAD_DIV)
#define APP_NAME "grad_div_3d"
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {108 * MFEM_BENCH_NE};
extern void mfem_app_grad_div_3d(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *);
extern void mfem_app_grad_div_3d_reference(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_grad_div_3d(args[0], args[1], args[2], args[3], args[4], args[5],
                       args[6], args[7], args[8], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_grad_div_3d_reference(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      args[8], state[0]);
}
#elif defined(MFEM_APP_ABS_MASS)
#define APP_NAME "abs_l1_mass_3d"
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {64 * MFEM_BENCH_NE};
extern void mfem_app_abs_l1_mass_3d(
    const double *, const double *, const double *, const double *, double *);
extern void mfem_app_abs_l1_mass_3d_reference(
    const double *, const double *, const double *, const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_abs_l1_mass_3d(args[0], args[1], args[2], args[3], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_abs_l1_mass_3d_reference(
      args[0], args[1], args[2], args[3], state[0]);
}
#elif defined(MFEM_APP_ABS_DIFFUSION)
#define APP_NAME "abs_l1_diffusion_3d"
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {64 * MFEM_BENCH_NE};
extern void mfem_app_abs_l1_diffusion_3d(
    const double *, const double *, const double *, const double *,
    const double *, const double *, double *);
extern void mfem_app_abs_l1_diffusion_3d_reference(
    const double *, const double *, const double *, const double *,
    const double *, const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_abs_l1_diffusion_3d(
      args[0], args[1], args[2], args[3], args[4], args[5], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_abs_l1_diffusion_3d_reference(
      args[0], args[1], args[2], args[3], args[4], args[5], state[0]);
}
#elif defined(MFEM_APP_ABS_CURLCURL)
#define APP_NAME "abs_l1_curlcurl_3d"
#define OUTPUT_COUNT 1
static const int64_t output_extents[] = {144 * MFEM_BENCH_NE};
extern void mfem_app_abs_l1_curlcurl_3d(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *);
extern void mfem_app_abs_l1_curlcurl_3d_reference(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_abs_l1_curlcurl_3d(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      args[8], state[0]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_abs_l1_curlcurl_3d_reference(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      args[8], state[0]);
}
#elif defined(MFEM_APP_NAVIER)
#define APP_NAME "navier_tgv_pa_operators_3d"
#define OUTPUT_COUNT 2
static const int64_t output_extents[] = {
    192 * MFEM_BENCH_NE, 64 * MFEM_BENCH_NE};
extern void mfem_app_navier_tgv_pa_operators_3d(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double *, double *);
extern void mfem_app_navier_tgv_pa_operators_3d_direct_reference(
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, double *, double *);
static void run_raised(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_navier_tgv_pa_operators_3d(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      args[8], args[9], args[10], state[0], state[1]);
}
static void run_reference(double state[MAX_OUTPUTS][MAX_EXTENT]) {
  mfem_app_navier_tgv_pa_operators_3d_direct_reference(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
      args[8], args[9], args[10], state[0], state[1]);
}
#else
#error "Select one MFEM_APP_* application"
#endif

static double seconds(void) {
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (double)value.tv_sec + 1.0e-9 * (double)value.tv_nsec;
}

static void initialize_data(void) {
  for (int a = 0; a < 13; ++a)
    for (int i = 0; i < MAX_EXTENT; ++i)
      args[a][i] = ((double)((i * 17 + a * 13 + 5) % 101) - 50.0) / 257.0;
  for (int i = 0; i < MAX_EXTENT; ++i) {
    args[3][i] += 0.75;
    args[4][i] += 0.5;
    args[6][i] += 1.0;
  }

#if defined(MFEM_APP_MTOP)
  for (int p = 0; p < 25 * MFEM_BENCH_NE; ++p) {
    args[5][4 * p] = 1.0 + 0.001 * p;
    args[5][4 * p + 1] = 0.03;
    args[5][4 * p + 2] = -0.02;
    args[5][4 * p + 3] = 1.1 + 0.001 * p;
  }
#elif defined(MFEM_APP_MINIMAL_SURFACE)
  for (int p = 0; p < 25 * MFEM_BENCH_NE; ++p) {
    args[3][4 * p] = 1.0 + 0.001 * p;
    args[3][4 * p + 1] = 0.03;
    args[3][4 * p + 2] = -0.02;
    args[3][4 * p + 3] = 1.1 + 0.001 * p;
  }
#endif

  for (int output = 0; output < MAX_OUTPUTS; ++output)
    for (int i = 0; i < MAX_EXTENT; ++i)
      initial_state[output][i] =
          ((double)((i * 11 + output * 7 + 3) % 37) - 18.0) / 509.0;
}

static void reset_states(void) {
  memcpy(reference_state, initial_state, sizeof(initial_state));
  memcpy(raised_state, initial_state, sizeof(initial_state));
}

int main(void) {
  /* Compile-time element sweeps intentionally use large automatic scratch
   * tensors in the extracted direct and normalized algorithms.  Raise the
   * soft stack limit to the process hard limit before either path executes. */
  struct rlimit stack_limit;
  if (getrlimit(RLIMIT_STACK, &stack_limit) == 0) {
    stack_limit.rlim_cur = stack_limit.rlim_max;
    (void)setrlimit(RLIMIT_STACK, &stack_limit);
  }
  initialize_data();
  reset_states();
  run_reference(reference_state);
  run_raised(raised_state);

  double max_abs = 0.0;
  double max_rel = 0.0;
  for (int output = 0; output < OUTPUT_COUNT; ++output) {
    for (int64_t i = 0; i < output_extents[output]; ++i) {
      double abs_error = fabs(reference_state[output][i] - raised_state[output][i]);
      double scale = fmax(1.0, fabs(reference_state[output][i]));
      double rel_error = abs_error / scale;
#ifdef MFEM_DEBUG_MISMATCHES
      if (abs_error > 1.0e-12)
        printf("mismatch output=%d index=%lld reference=%.17g raised=%.17g "
               "abs=%.17g\n",
               output, (long long)i, reference_state[output][i],
               raised_state[output][i], abs_error);
#endif
      if (abs_error > max_abs)
        max_abs = abs_error;
      if (rel_error > max_rel)
        max_rel = rel_error;
    }
  }
  int correct = isfinite(max_abs) && max_rel <= 1.0e-10;
  printf("application=%s correctness=%s max_abs=%.17g max_rel=%.17g\n",
         APP_NAME, correct ? "PASS" : "FAIL", max_abs, max_rel);
  if (!correct)
    return 2;

  reset_states();
  run_reference(reference_state);
  memcpy(reference_state, initial_state, sizeof(initial_state));
  double start = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i)
    run_reference(reference_state);
  double cpu_us = (seconds() - start) * 1.0e6 / BENCH_ITERS;

  reset_states();
  run_raised(raised_state);
  memcpy(raised_state, initial_state, sizeof(initial_state));
  start = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i)
    run_raised(raised_state);
  double raised_us = (seconds() - start) * 1.0e6 / BENCH_ITERS;

  printf("application=%s iterations=%d cpu_reference_us=%.6f raised_gpu_us=%.6f speedup=%.6f\n",
         APP_NAME, BENCH_ITERS, cpu_us, raised_us, cpu_us / raised_us);
  return 0;
}

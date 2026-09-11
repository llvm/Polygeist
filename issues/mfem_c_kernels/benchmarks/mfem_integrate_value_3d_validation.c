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

#define INPUT_SIZE (125 * MFEM_BENCH_NE)
#define OUTPUT_SIZE (64 * MFEM_BENCH_NE)

static double input[INPUT_SIZE];
static double basis[20];
static double raised_output[OUTPUT_SIZE];
static double reference_output[OUTPUT_SIZE];

extern void mfem_integrate_value_3d_scratch_sliced(
    const double *, const double *, double *);
extern void mfem_integrate_value_3d_scratch_sliced_reference(
    const double *, const double *, double *);

static double value(int i, int salt) {
  return (double)(((i * 17 + salt * 13 + 5) % 101) - 50) / 257.0;
}

static double seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

int main(void) {
  for (int i = 0; i < INPUT_SIZE; ++i)
    input[i] = value(i, 7);
  for (int i = 0; i < 20; ++i)
    basis[i] = value(i, 0);
  memset(raised_output, 0, sizeof(raised_output));
  memset(reference_output, 0, sizeof(reference_output));

  mfem_integrate_value_3d_scratch_sliced_reference(
      input, basis, reference_output);
  mfem_integrate_value_3d_scratch_sliced(input, basis, raised_output);
  double max_abs = 0.0, max_rel = 0.0, checksum = 0.0;
  for (int i = 0; i < OUTPUT_SIZE; ++i) {
    double error = fabs(reference_output[i] - raised_output[i]);
    double scale = fmax(1.0, fabs(reference_output[i]));
    max_abs = fmax(max_abs, error);
    max_rel = fmax(max_rel, error / scale);
    checksum += raised_output[i];
  }
  int correct = isfinite(max_abs) && max_rel <= 1.0e-10;
  printf("kernel=integrate_value_3d correctness=%s max_abs=%.17g "
         "max_rel=%.17g checksum=%.17g\n",
         correct ? "PASS" : "FAIL", max_abs, max_rel, checksum);
  if (!correct)
    return 2;

  /* Warm once, then measure repeated warm calls with stable addresses. */
  memset(raised_output, 0, sizeof(raised_output));
  mfem_integrate_value_3d_scratch_sliced(input, basis, raised_output);
  double start = seconds();
  for (int iteration = 0; iteration < BENCH_ITERS; ++iteration)
    mfem_integrate_value_3d_scratch_sliced(input, basis, raised_output);
  double runtime_us = (seconds() - start) * 1.0e6 / BENCH_ITERS;
  printf("implementation=polygeist_composed kernel=integrate_value_3d "
         "ne=%d iterations=%d runtime_us=%.6f\n",
         MFEM_BENCH_NE, BENCH_ITERS, runtime_us);
  return 0;
}

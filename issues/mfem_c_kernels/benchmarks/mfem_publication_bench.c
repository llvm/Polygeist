/* Publication timing driver for the normalized MFEM kernels.
 *
 * Reuse the operator selection, storage, and call signatures from the retained
 * resident harness, but replace its best-of-N main with a raw-sample driver.
 * Each fresh process performs five warmups and prints twenty individual
 * wall-clock measurements. GPU measurements end after cudaDeviceSynchronize.
 */
#ifndef BENCH_GPU
int cudaDeviceSynchronize(void) { return 0; }
#endif

#define main mfem_legacy_benchmark_main
#include "mfem_resident_bench.c"
#undef main

int main(void) {
  for (int j = 0; j < 6; ++j)
    for (int i = 0; i < 20; ++i) a[j][i] = value(i, j + 1);
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
  for (int i = 0; i < OP_SIZE; ++i) op[i] = value(i, 7);
  for (int i = 0; i < INPUT_SIZE; ++i) x[i] = value(i, 8);
  for (int i = 0; i < OUTPUT_SIZE; ++i) y[i] = value(i, 9);

  for (int warmup = 0; warmup < 5; ++warmup) run();
#ifdef BENCH_GPU
  cudaDeviceSynchronize();
#endif

  for (int sample = 0; sample < 20; ++sample) {
    const double start = seconds();
    run();
#ifdef BENCH_GPU
    cudaDeviceSynchronize();
#endif
    const double elapsed_us = (seconds() - start) * 1.0e6;
    printf("implementation=%s kernel=%s ne=%d sample=%d runtime_us=%.9f\n",
#ifdef BENCH_GPU
           "polygeist_raised_gpu",
#elif defined(BENCH_RAISED_CPU)
           "polygeist_raised_cpu",
#else
           "vanilla_c_cpu",
#endif
           BENCH_NAME, MFEM_BENCH_NE, sample, elapsed_us);
  }

  double checksum = 0.0;
  for (int i = 0; i < OUTPUT_SIZE; ++i) checksum += y[i];
  fprintf(stderr, "kernel=%s final_checksum=%.17g\n", BENCH_NAME, checksum);
  return isfinite(checksum) ? 0 : 1;
}

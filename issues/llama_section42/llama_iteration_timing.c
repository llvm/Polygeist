#define _POSIX_C_SOURCE 200809L
#include "polygeist_cublas_rt.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double start, wall[5], device[5];
static double ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec / 1.e6;
}
void polygeist_benchmark_iteration_begin(int iteration) {
  (void)iteration;
  if (cudaDeviceSynchronize() != cudaSuccess)
    abort();
  polygeist_cublas_time_begin();
  start = ms();
}
void polygeist_benchmark_iteration_end(int iteration) {
  double gpu = polygeist_cublas_time_end_ms();
  double host = ms() - start;
  if (iteration >= 5 && iteration < 10) {
    wall[iteration - 5] = host;
    device[iteration - 5] = gpu;
  }
  if (iteration == 9)
    for (int i = 0; i < 5; ++i)
      printf("RESIDENT_SAMPLE,%d,%.9f,%.9f\n", i, wall[i], device[i]);
}

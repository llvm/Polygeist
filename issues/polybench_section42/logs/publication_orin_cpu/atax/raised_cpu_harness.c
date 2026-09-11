#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 1900
#define N 2100

extern void kernel_atax(int m, int n, double a[M][N], double x[N],
                        double y[N], double tmp[M]);

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void initialize(double a[M][N], double x[N]) {
  const double fn = N;
  for (int i = 0; i < N; ++i)
    x[i] = 1.0 + i / fn;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      a[i][j] = (double)((i + j) % N) / (5 * M);
}

int main(void) {
  double(*a)[N] = malloc(sizeof(double[M][N]));
  double *x = malloc(sizeof(double[N]));
  double *y = malloc(sizeof(double[N]));
  double *tmp = malloc(sizeof(double[M]));
  if (!a || !x || !y || !tmp) return 2;
  initialize(a, x);
#ifdef POLYGEIST_CPU_TIMING
  for (int i = 0; i < 10; ++i) {
    const double start = now_ms();
    kernel_atax(M, N, a, x, y, tmp);
    const double elapsed = now_ms() - start;
    printf("CPU_SAMPLE phase=%s index=%d runtime_ms=%.6f\n",
           i < 5 ? "warmup" : "sample", i < 5 ? i : i - 5, elapsed);
    fflush(stdout);
  }
#else
  kernel_atax(M, N, a, x, y, tmp);
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: y");
  for (int i = 0; i < N; ++i) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", y[i]);
  }
  fprintf(stderr, "\nend   dump: y\n");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
#endif
  free(tmp);
  free(y);
  free(x);
  free(a);
  return 0;
}

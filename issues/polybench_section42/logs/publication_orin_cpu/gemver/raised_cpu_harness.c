#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2000

extern void kernel_gemver(int, double, double, double[N][N], double[N],
                          double[N], double[N], double[N], double[N],
                          double[N], double[N], double[N]);

static double wall_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static int run_once(int dump) {
  double(*A)[N] = malloc(sizeof(double) * N * N);
  double *u1 = malloc(sizeof(double) * N);
  double *v1 = malloc(sizeof(double) * N);
  double *u2 = malloc(sizeof(double) * N);
  double *v2 = malloc(sizeof(double) * N);
  double *w = malloc(sizeof(double) * N);
  double *x = malloc(sizeof(double) * N);
  double *y = malloc(sizeof(double) * N);
  double *z = malloc(sizeof(double) * N);
  if (!A || !u1 || !v1 || !u2 || !v2 || !w || !x || !y || !z)
    return 2;

  const double fn = N;
  for (int i = 0; i < N; ++i) {
    u1[i] = i;
    u2[i] = ((i + 1) / fn) / 2.0;
    v1[i] = ((i + 1) / fn) / 4.0;
    v2[i] = ((i + 1) / fn) / 6.0;
    y[i] = ((i + 1) / fn) / 8.0;
    z[i] = ((i + 1) / fn) / 9.0;
    x[i] = 0.0;
    w[i] = 0.0;
    for (int j = 0; j < N; ++j)
      A[i][j] = (double)(i * j % N) / N;
  }

  double begin = wall_ms();
  kernel_gemver(N, 1.5, 1.2, A, u1, v1, u2, v2, w, x, y, z);
  printf("POLYGEIST_RAISED_GPU_E2E_MS %.9f\n", wall_ms() - begin);

  if (dump) {
    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\nbegin dump: w\n");
    for (int i = 0; i < N; ++i) {
      if (i % 20 == 0)
        fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", w[i]);
    }
    fprintf(stderr, "\nend   dump: w\n==END   DUMP_ARRAYS==\n");
  }

  free(A); free(u1); free(v1); free(u2); free(v2);
  free(w); free(x); free(y); free(z);
  return 0;
}

int main(void) {
#ifdef CORRECTNESS_RUN
  return run_once(1);
#else
  for (int i = 0; i < 10; ++i) {
    printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",
           i < 5 ? "warmup" : "sample", i < 5 ? i : i - 5);
    fflush(stdout);
    int status = run_once(0);
    if (status)
      return status;
  }
  return 0;
#endif
}

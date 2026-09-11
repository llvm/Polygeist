#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NI 1000
#define NJ 1100
#define NK 1200

extern void kernel_gemm(int, int, int, double, double, double[NI][NJ],
                        double[NI][NK], double[NK][NJ]);

static double wall_ms(void) {
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec / 1.0e6;
}

static int run_once(int dump) {
  double(*C)[NJ] = malloc(sizeof(double) * NI * NJ);
  double(*A)[NK] = malloc(sizeof(double) * NI * NK);
  double(*B)[NJ] = malloc(sizeof(double) * NK * NJ);
  if (!C || !A || !B)
    return 2;

  for (int i = 0; i < NI; ++i)
    for (int j = 0; j < NJ; ++j)
      C[i][j] = (double)((i * j + 1) % NI) / NI;
  for (int i = 0; i < NI; ++i)
    for (int j = 0; j < NK; ++j)
      A[i][j] = (double)(i * (j + 1) % NK) / NK;
  for (int i = 0; i < NK; ++i)
    for (int j = 0; j < NJ; ++j)
      B[i][j] = (double)(i * (j + 2) % NJ) / NJ;

  double begin = wall_ms();
  kernel_gemm(NI, NJ, NK, 1.5, 1.2, C, A, B);
  printf("POLYGEIST_RAISED_GPU_E2E_MS %.9f\n", wall_ms() - begin);

  if (dump) {
    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\nbegin dump: C\n");
    for (int i = 0; i < NI; ++i)
      for (int j = 0; j < NJ; ++j) {
        if ((i * NI + j) % 20 == 0)
          fprintf(stderr, "\n");
        fprintf(stderr, "%0.2lf ", C[i][j]);
      }
    fprintf(stderr, "\nend   dump: C\n==END   DUMP_ARRAYS==\n");
  }

  free(C);
  free(A);
  free(B);
  return 0;
}

int main(void) {
#ifdef CORRECTNESS_RUN
  return run_once(1);
#else
  for (int iteration = 0; iteration < 10; ++iteration) {
    printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",
           iteration < 5 ? "warmup" : "sample",
           iteration < 5 ? iteration : iteration - 5);
    fflush(stdout);
    int status = run_once(0);
    if (status)
      return status;
  }
  return 0;
#endif
}

// File name: main.c

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct TwoDMemrefF32 {
  float *ptrToData;
  float *alignedPtrToData;
  long offset;
  long shape[2];
  long stride[2];
};

#define M 1024
#define N 1024
#define K 1024

extern void _mlir_ciface_gemm(float alpha, float beta, struct TwoDMemrefF32 *,
                              struct TwoDMemrefF32 *, struct TwoDMemrefF32 *);
extern void _mlir_ciface_gemm_new(struct TwoDMemrefF32 *,
                                  struct TwoDMemrefF32 *,
                                  struct TwoDMemrefF32 *, float alpha,
                                  float beta);

int main(int argc, char *argv[]) {
  clock_t start, end;
  double orig_time, opt_time;

  int i, j;

  float *A = (float *)malloc(sizeof(float) * M * K);
  float *B = (float *)malloc(sizeof(float) * N * K);
  float *C = (float *)malloc(sizeof(float) * M * N);
  float *D = (float *)malloc(sizeof(float) * M * N);

  for (i = 0; i < M; i++)
    for (j = 0; j < K; j++)
      A[i * K + j] = ((float)i + j) / (i + j + 1);
  for (i = 0; i < K; i++)
    for (j = 0; j < N; j++)
      B[i * N + j] = ((float)i + j) / (i + j + 1);
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) {
      C[i * N + j] = (float)1.0;
      D[i * N + j] = (float)1.0;
    }

  struct TwoDMemrefF32 A_mem = {A, A, 0, {M, K}, {K, 1}};
  struct TwoDMemrefF32 B_mem = {B, B, 0, {K, N}, {N, 1}};
  struct TwoDMemrefF32 C_mem = {C, C, 0, {M, N}, {N, 1}};
  struct TwoDMemrefF32 D_mem = {D, D, 0, {M, N}, {N, 1}};

  printf("Running original MLIR ...\n");
  start = clock();
  _mlir_ciface_gemm(1.0, 1.0, &C_mem, &A_mem, &B_mem);
  end = clock();
  orig_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time: %10.6f s\n", orig_time);

  printf("Running Pluto optimised MLIR ...\n");
  start = clock();
  _mlir_ciface_gemm_new(&A_mem, &B_mem, &D_mem, 1.0, 1.0);
  end = clock();
  opt_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time: %10.6f s\n", opt_time);

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      if (fabsf(C[i * N + j] - D[i * N + j]) > 1e-6) {
        fprintf(stderr,
                "Error detected: i = %d j = %d C[i][j] = %f D[i][j] = %f\n", i,
                j, C[i * N + j], D[i * N + j]);
        return 1;
      }

  printf("TEST passed!\n");

  free(A);
  free(B);
  free(C);
  free(D);

  return 0;
}

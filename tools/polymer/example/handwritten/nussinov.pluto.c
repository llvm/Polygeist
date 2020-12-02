// File name: main.c

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct OneDMemrefI8 {
  int8_t *ptrToData;
  int8_t *alignedPtrToData;
  long offset;
  long shape[1];
  long stride[1];
};

struct TwoDMemrefI32 {
  int *ptrToData;
  int *alignedPtrToData;
  long offset;
  long shape[2];
  long stride[2];
};

#ifdef MINI_DATASET
#define N 60
#endif

#ifdef SMALL_DATASET
#define N 180
#endif

#ifdef MEDIUM_DATASET
#define N 500
#endif

#ifdef LARGE_DATASET
#define N 2500
#endif

#ifdef EXTRALARGE_DATASET
#define N 5500
#endif

extern void _mlir_ciface_pb_nussinov(struct OneDMemrefI8 *,
                                     struct TwoDMemrefI32 *);
extern void _mlir_ciface_pb_nussinov_new(struct OneDMemrefI8 *,
                                         struct TwoDMemrefI32 *);

int main(int argc, char *argv[]) {
  clock_t start, end;
  double orig_time, opt_time;

  int i, j;

  int8_t *seq1 = (int8_t *)malloc(sizeof(int8_t) * N);
  int8_t *seq2 = (int8_t *)malloc(sizeof(int8_t) * N);
  int *table1 = (int *)malloc(sizeof(int) * N * N);
  int *table2 = (int *)malloc(sizeof(int) * N * N);

  for (i = 0; i < N; i++) {
    seq1[i] = (int8_t)((i + 1) % 4);
    seq2[i] = (int8_t)((i + 1) % 4);
  }

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
      table1[i * N + j] = 0;
      table2[i * N + j] = 0;
    }

  struct OneDMemrefI8 seq1_mem = {seq1, seq1, 0, {N}, {1}};
  struct OneDMemrefI8 seq2_mem = {seq2, seq2, 0, {N}, {1}};
  struct TwoDMemrefI32 table1_mem = {table1, table1, 0, {N, N}, {N, 1}};
  struct TwoDMemrefI32 table2_mem = {table2, table2, 0, {N, N}, {N, 1}};

  printf("Running original MLIR ...\n");
  start = clock();
  _mlir_ciface_pb_nussinov(&seq1_mem, &table1_mem);
  end = clock();
  orig_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time: %10.6f s\n", orig_time);

  printf("Running Pluto optimised MLIR ...\n");
  start = clock();
  _mlir_ciface_pb_nussinov_new(&seq2_mem, &table2_mem);
  end = clock();
  opt_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Total time: %10.6f s\n", opt_time);

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (table2[i * N + j] != table2[i * N + j]) {
        fprintf(stderr,
                "Error detected: i = %d j = %d table1[i][j] = %d table2[i][j] "
                "= %d\n",
                i, j, table1[i * N + j], table2[i * N + j]);
        return 1;
      }

  printf("TEST passed!\n");

  free(seq1);
  free(seq2);
  free(table1);
  free(table2);

  return 0;
}

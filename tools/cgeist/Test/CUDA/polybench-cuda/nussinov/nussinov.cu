// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10
// clang-format on
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define match(b1, b2) (((b1) + (b2)) == 3 ? 1 : 0)

static void init_array(int n, double *table, double *oldtable, double *seq) {
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      table[i * n + j] = ((double)i * j) / n;
      oldtable[i * n + j] = ((double)i * j) / n;
    }
  for (i = 0; i < n; i++)
    seq[i] = ((double)i) / n;
}

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

// Dynamic programming wavefront
__global__ void kernel_max_score(int n, double *seq, double *table,
                                 double *oldtable, int w) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = ((int)n - 1) + j - w;

  if (0 <= i && i < n && i + 1 <= j && j < n) {
    double maximum = table[i * n + j];

    if (j - 1 >= 0)
      maximum = max(maximum, table[i * n + (j - 1)]);
    if (i + 1 < n)
      maximum = max(maximum, table[(i + 1) * n + j]);

    if (j - 1 >= 0 && i + 1 < n) {
      auto upd = table[(i + 1) * n + (j - 1)];

      /* don't allow adjacent elements to bond */
      if (i < j - 1)
        upd += (seq[i] + seq[j] == 3) ? (double)1 : (double)0;

      maximum = max(maximum, upd);
    }

    for (int k = i + 1; k < j; k++)
      maximum = max(maximum, table[i * n + k] + table[(k + 1) * n + j]);

    //  AtomicMax<double>::set_if_larger(table[i * n + j], maximum);
    table[i * n + j] = maximum;
  }
}

static void kernel(int n, double *seq, double *table, double *oldtable) {
  const unsigned threadsPerBlock = 32;

  for (int w = n; w < 2 * n - 1; ++w) { // wavefronting
    kernel_max_score<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(
        n, seq, table, oldtable, w);
  }
}

static void print_array(int n, double *table) {
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, "%0.2lf ", table[i * n + j]);
      if ((i * n + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {

  int dump_code = atoi(argv[1]);
  int n = atoi(argv[2]);
  double *table = (double *)malloc(n * n * sizeof(double));
  double *oldtable = (double *)malloc(n * n * sizeof(double));
  double *seq = (double *)malloc(n * sizeof(double));

  init_array(n, table, oldtable, seq);

  double *dev_table;
  double *dev_oldtable;
  double *dev_seq;
  cudaMalloc(&dev_table, n * n * sizeof(double));
  cudaMalloc(&dev_oldtable, n * n * sizeof(double));
  cudaMalloc(&dev_seq, n * sizeof(double));
  cudaMemcpy(dev_table, table, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_oldtable, oldtable, n * n * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_seq, seq, n * sizeof(double), cudaMemcpyHostToDevice);

  kernel(n, dev_seq, dev_table, dev_oldtable);
  cudaMemcpy(table, dev_table, n * n * sizeof(double), cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(n, table);

  free((void *)table);
  free((void *)oldtable);
  free((void *)seq);
}

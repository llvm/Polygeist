#include <omp.h>
#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trmm.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trmm.h"

/* Array initialization. */
static void init_array(int m, int n, DATA_TYPE *alpha,
                       DATA_TYPE POLYBENCH_2D(A, M, M, m, m),
                       DATA_TYPE POLYBENCH_2D(B, M, N, m, n)) {
  int i, j;

  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (DATA_TYPE)((i + j) % m) / m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (DATA_TYPE)((n + (i - j)) % n) / n;
    }
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m, int n, DATA_TYPE POLYBENCH_2D(B, M, N, m, n)) {
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("B");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, B[i][j]);
    }
  POLYBENCH_DUMP_END("B");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_trmm(int m, int n, DATA_TYPE alpha,
                        DATA_TYPE POLYBENCH_2D(A, M, M, m, m),
                        DATA_TYPE POLYBENCH_2D(B, M, N, m, n),
                        DATA_TYPE POLYBENCH_1D(S, M, m)) {
  int i, j, k;

// BLAS parameters
// SIDE   = 'L'
// UPLO   = 'L'
// TRANSA = 'T'
// DIAG   = 'U'
// => Form  B := alpha*A**T*B.
// A is MxM
// B is MxN
  int t1, t2, t3, t4, t5, t6, t7;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if ((_PB_M >= 1) && (_PB_N >= 1)) {
  for (t2=0;t2<=floord(_PB_M-2,16);t2++) {
    lbp=ceild(t2,2);
    ubp=min(floord(_PB_M-1,32),t2);
#pragma omp parallel for private(lbv,ubv,t4,t5,t6,t7)
    for (t3=lbp;t3<=ubp;t3++) {
      for (t4=32*t2-32*t3;t4<=min(min(_PB_M-2,32*t3+30),32*t2-32*t3+31);t4++) {
        for (t5=max(32*t3,t4+1);t5<=min(_PB_M-1,32*t3+31);t5++) {
          for (t6=0;t6<=_PB_N-1;t6++) {
            S[t5] = A[t5][t4] * B[t5][t6];;
            B[t4][t6] += S[t5];;
          }
        }
      }
    }
  }
  lbp=0;
  ubp=floord(_PB_M-1,32);
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7)
  for (t2=lbp;t2<=ubp;t2++) {
    for (t3=0;t3<=floord(_PB_N-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_M-1,32*t2+31);t4++) {
        for (t5=32*t3;t5<=min(_PB_N-1,32*t3+31);t5++) {
          B[t4][t5] = alpha * B[t4][t5];;
        }
      }
    }
  }
}
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, M, m, m);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, M, N, m, n);
  POLYBENCH_2D_ARRAY_DECL(S, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array(m, n, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trmm(m, n, alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
              POLYBENCH_ARRAY(S));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(B)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

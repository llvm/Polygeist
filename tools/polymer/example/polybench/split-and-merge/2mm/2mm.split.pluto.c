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
/* 2mm.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "2mm.h"

/* Array initialization. */
static void init_array(int ni, int nj, int nk, int nl, DATA_TYPE *alpha,
                       DATA_TYPE *beta,
                       DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                       DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                       DATA_TYPE POLYBENCH_2D(C, NJ, NL, nj, nl),
                       DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl)) {
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE)((i * j + 1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE)(i * (j + 1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (DATA_TYPE)((i * (j + 3) + 1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE)(i * (j + 2) % nk) / nk;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nl,
                        DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl)) {
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_2mm(int ni, int nj, int nk, int nl, DATA_TYPE alpha,
                       DATA_TYPE beta,
                       DATA_TYPE POLYBENCH_2D(tmp, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                       DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                       DATA_TYPE POLYBENCH_2D(C, NJ, NL, nj, nl),
                       DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl),
                       DATA_TYPE POLYBENCH_1D(S, NK, nk)) {
  int i, j, k;

  int t1, t2, t3, t4, t5, t6, t7, t8;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if (_PB_NI >= 1) {
  if (_PB_NL >= 1) {
    lbp=0;
    ubp=floord(_PB_NI-1,32);
#pragma omp parallel for private(lbv,ubv,t4,t5,t6,t7,t8)
    for (t3=lbp;t3<=ubp;t3++) {
      for (t4=0;t4<=floord(_PB_NL-1,32);t4++) {
        for (t5=32*t3;t5<=min(_PB_NI-1,32*t3+31);t5++) {
          for (t6=32*t4;t6<=min(_PB_NL-1,32*t4+31);t6++) {
            D[t5][t6] *= beta;;
          }
        }
      }
    }
  }
  if (_PB_NJ >= 1) {
    lbp=0;
    ubp=floord(_PB_NI-1,32);
#pragma omp parallel for private(lbv,ubv,t4,t5,t6,t7,t8)
    for (t3=lbp;t3<=ubp;t3++) {
      for (t4=0;t4<=floord(_PB_NJ-1,32);t4++) {
        for (t5=32*t3;t5<=min(_PB_NI-1,32*t3+31);t5++) {
          for (t6=32*t4;t6<=min(_PB_NJ-1,32*t4+31);t6++) {
            tmp[t5][t6] = SCALAR_VAL(0.0);;
          }
        }
      }
    }
  }
  if ((_PB_NJ >= 1) && (_PB_NK >= 1)) {
    for (t3=0;t3<=floord(_PB_NI+_PB_NK-2,32);t3++) {
      lbp=max(0,ceild(32*t3-_PB_NK+1,32));
      ubp=min(floord(_PB_NI-1,32),t3);
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8)
      for (t4=lbp;t4<=ubp;t4++) {
        for (t5=32*t3-32*t4;t5<=min(_PB_NK-1,32*t3-32*t4+31);t5++) {
          for (t6=32*t4;t6<=min(_PB_NI-1,32*t4+31);t6++) {
            for (t7=0;t7<=_PB_NJ-1;t7++) {
              S[t5] = alpha * A[t6][t5] * B[t5][t7];;
              tmp[t6][t7] += S[t5];;
            }
          }
        }
      }
    }
  }
  if ((_PB_NJ >= 1) && (_PB_NL >= 1)) {
    for (t3=0;t3<=floord(_PB_NI+_PB_NJ-2,32);t3++) {
      lbp=max(0,ceild(32*t3-_PB_NJ+1,32));
      ubp=min(floord(_PB_NI-1,32),t3);
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8)
      for (t4=lbp;t4<=ubp;t4++) {
        for (t5=32*t3-32*t4;t5<=min(_PB_NJ-1,32*t3-32*t4+31);t5++) {
          for (t6=32*t4;t6<=min(_PB_NI-1,32*t4+31);t6++) {
            for (t7=0;t7<=_PB_NL-1;t7++) {
              S[t5] = tmp[t6][t5] * C[t5][t7];;
              D[t6][t7] += S[t5];;
            }
          }
        }
      }
    }
  }
}
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(tmp, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NI, NL, ni, nl);
  POLYBENCH_1D_ARRAY_DECL(S, DATA_TYPE, NK, nk);

  /* Initialize array(s). */
  init_array(ni, nj, nk, nl, &alpha, &beta, POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_2mm(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp),
             POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C),
             POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(S));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(S);

  return 0;
}

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
/* covariance.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"

/* Array initialization. */
static void init_array(int m, int n, DATA_TYPE *float_n,
                       DATA_TYPE POLYBENCH_2D(data, N, M, n, m)) {
  int i, j;

  *float_n = (DATA_TYPE)n;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE)i * j) / M;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m, DATA_TYPE POLYBENCH_2D(cov, M, M, m, m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, cov[i][j]);
    }
  POLYBENCH_DUMP_END("cov");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_covariance(int m, int n, DATA_TYPE float_n,
                              DATA_TYPE POLYBENCH_2D(data, N, M, n, m),
                              DATA_TYPE POLYBENCH_2D(cov, M, M, m, m),
                              DATA_TYPE POLYBENCH_1D(mean, M, m),
                              DATA_TYPE POLYBENCH_1D(scr_data, N, n)) {
  int i, j, k;
  int t1, t2, t3, t4, t5, t6, t7;
 register int lbv, ubv;
if (_PB_M >= 1) {
  for (t2=0;t2<=floord(_PB_M-1,32);t2++) {
    for (t3=t2;t3<=floord(_PB_M-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_M-1,32*t2+31);t4++) {
        for (t5=max(32*t3,t4);t5<=min(_PB_M-1,32*t3+31);t5++) {
          cov[t4][t5] = SCALAR_VAL(0.0);;
        }
      }
    }
  }
  for (t2=0;t2<=floord(_PB_M-1,32);t2++) {
    for (t3=32*t2;t3<=min(_PB_M-1,32*t2+31);t3++) {
      mean[t3] = SCALAR_VAL(0.0);;
    }
  }
  if (_PB_N >= 1) {
    for (t2=0;t2<=floord(_PB_M-1,32);t2++) {
      for (t3=0;t3<=floord(_PB_N-1,32);t3++) {
        for (t4=32*t3;t4<=min(_PB_N-1,32*t3+31);t4++) {
          for (t5=32*t2;t5<=min(_PB_M-1,32*t2+31);t5++) {
            mean[t5] += data[t4][t5];;
          }
        }
      }
    }
  }
  for (t2=0;t2<=floord(_PB_M-1,32);t2++) {
    for (t3=32*t2;t3<=min(_PB_M-1,32*t2+31);t3++) {
      mean[t3] /= float_n;;
    }
  }
  for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_M-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_N-1,32*t2+31);t4++) {
        for (t5=32*t3;t5<=min(_PB_M-1,32*t3+31);t5++) {
          data[t4][t5] -= mean[t5];;
        }
      }
    }
  }
#if 0
  for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_M-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_N-1,32*t2+31);t4++) {
        for (t5=32*t3;t5<=min(_PB_M-1,32*t3+31);t5++) {
          for (t6=t5;t6<=_PB_M-1;t6++) {
            scr_data[t4] = data[t4][t5] * data[t4][t6];;
            cov[t5][t6] += scr_data[t4];;
          }
        }
      }
    }
  }
#endif
  for (t4=0;t4<=floord(_PB_N-1,32);t4++) {
    for (t2=0;t2<=floord(_PB_M-1,32);t2++) {
      for (t3=t2;t3<=floord(_PB_M-1,32);t3++) {
        for (t6=32*t4;t6<=min(_PB_N-1,32*t4+31);t6++) {
          for (t5=32*t2;t5<=min(_PB_M-1,32*t2+31);t5++) {
            for (t7=max(32*t3,t5);t7<=min(_PB_M-1,32*t3+31);t7++) {
              scr_data[t6] = data[t6][t5] * data[t6][t7];;
              cov[t5][t7] += scr_data[t6];;
            }
          }
        }
      }
    }
  }

  for (t2=0;t2<=floord(_PB_M-1,32);t2++) {
    for (t3=t2;t3<=floord(_PB_M-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_M-1,32*t2+31);t4++) {
        for (t5=max(32*t3,t4);t5<=min(_PB_M-1,32*t3+31);t5++) {
          cov[t4][t5] /= (float_n - SCALAR_VAL(1.0));;
          cov[t5][t4] = cov[t4][t5];;
        }
      }
    }
  }
}
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data, DATA_TYPE, N, M, n, m);
  POLYBENCH_2D_ARRAY_DECL(cov, DATA_TYPE, M, M, m, m);
  POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(scr_data, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance(m, n, float_n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(cov),
                    POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(scr_data));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(cov)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(cov);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}

#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

// RUN: mlir-clang %s %stdinclude | FileCheck %s
// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: mlir-clang %s %polyverify %stdinclude -emit-llvm | clang -x ir - -O3 -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | clang -x ir - -O3 -o %s.execm && %s.execm > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2 %s.execm

/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  int t1, t2, t3, t4, t5, t6, t7;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if ((_PB_NI >= 1) && (_PB_NJ >= 1)) {
  for (t2=0;t2<=floord(_PB_NI-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_NJ-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
        for (t5=32*t3;t5<=min(_PB_NJ-1,32*t3+31);t5++) {
          C[t4][t5] *= beta;;
        }
      }
    }
  }
  if (_PB_NK >= 1) {
    for (t2=0;t2<=floord(_PB_NI-1,32);t2++) {
      for (t3=0;t3<=floord(_PB_NJ-1,32);t3++) {
        for (t4=0;t4<=floord(_PB_NK-1,32);t4++) {
          for (t5=32*t2;t5<=min(_PB_NI-1,32*t2+31);t5++) {
            for (t6=32*t4;t6<=min(_PB_NK-1,32*t4+31);t6++) {
              for (t7=32*t3;t7<=min(_PB_NJ-1,32*t3+31);t7++) {
                C[t5][t7] += alpha * A[t5][t6] * B[t6][t7];;
              }
            }
          }
        }
      }
    }
  }
}

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

// CHECK:   func private @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<1000x1100xf64>, %arg6: memref<1000x1200xf64>, %arg7: memref<1200x1100xf64>) {
// CHECK-NEXT:    %0 = index_cast %arg0 : i32 to index  
// CHECK-NEXT:    %1 = index_cast %arg1 : i32 to index
// CHECK-NEXT:    %2 = index_cast %arg2 : i32 to index
// CHECK-NEXT:    affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:      affine.for %arg9 = 0 to %1 {
// CHECK-NEXT:        %3 = affine.load %arg5[%arg8, %arg9] : memref<1000x1100xf64>
// CHECK-NEXT:        %4 = mulf %3, %arg4 : f64
// CHECK-NEXT:        affine.store %4, %arg5[%arg8, %arg9] : memref<1000x1100xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg9 = 0 to %2 {
// CHECK-NEXT:        affine.for %arg10 = 0 to %1 {
// CHECK-NEXT:          %3 = affine.load %arg6[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %4 = mulf %arg3, %3 : f64
// CHECK-NEXT:          %5 = affine.load %arg7[%arg9, %arg10] : memref<1200x1100xf64>
// CHECK-NEXT:          %6 = mulf %4, %5 : f64
// CHECK-NEXT:          %7 = affine.load %arg5[%arg8, %arg10] : memref<1000x1100xf64>
// CHECK-NEXT:          %8 = addf %7, %6 : f64
// CHECK-NEXT:          affine.store %8, %arg5[%arg8, %arg10] : memref<1000x1100xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:   }

// EXEC: {{[0-9]\.[0-9]+}}

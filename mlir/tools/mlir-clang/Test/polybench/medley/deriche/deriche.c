// RUN: mlir-clang %s %stdinclude | FileCheck %s

/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* deriche.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "deriche.h"


/* Array initialization. */
static
void init_array (int w, int h, DATA_TYPE* alpha,
		 DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))
{
  int i, j;

  *alpha=0.25; //parameter of the filter

  //input should be between 0 and 1 (grayscale image pixel)
  for (i = 0; i < w; i++)
     for (j = 0; j < h; j++)
	imgIn[i][j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int w, int h,
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("imgOut");
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, imgOut[i][j]);
    }
  POLYBENCH_DUMP_END("imgOut");
  POLYBENCH_DUMP_FINISH;
}



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
static
void kernel_deriche(int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h)) {
    int i,j;
    DATA_TYPE xm1, tm1, ym1, ym2;
    DATA_TYPE xp1, xp2;
    DATA_TYPE tp1, tp2;
    DATA_TYPE yp1, yp2;

    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1, c2;

#pragma scop
   k = (SCALAR_VAL(1.0)-EXP_FUN(-alpha))*(SCALAR_VAL(1.0)-EXP_FUN(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*EXP_FUN(-alpha)-EXP_FUN(SCALAR_VAL(2.0)*alpha));
   a1 = a5 = k;
   a2 = a6 = k*EXP_FUN(-alpha)*(alpha-SCALAR_VAL(1.0));
   a3 = a7 = k*EXP_FUN(-alpha)*(alpha+SCALAR_VAL(1.0));
   a4 = a8 = -k*EXP_FUN(SCALAR_VAL(-2.0)*alpha);
   b1 =  POW_FUN(SCALAR_VAL(2.0),-alpha);
   b2 = -EXP_FUN(SCALAR_VAL(-2.0)*alpha);
   c1 = c2 = 1;

   for (i=0; i<_PB_W; i++) {
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        xm1 = SCALAR_VAL(0.0);
        for (j=0; j<_PB_H; j++) {
            y1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = imgIn[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

    for (i=0; i<_PB_W; i++) {
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        xp1 = SCALAR_VAL(0.0);
        xp2 = SCALAR_VAL(0.0);
        for (j=_PB_H-1; j>=0; j--) {
            y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
            xp2 = xp1;
            xp1 = imgIn[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<_PB_W; i++)
        for (j=0; j<_PB_H; j++) {
            imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }

    for (j=0; j<_PB_H; j++) {
        tm1 = SCALAR_VAL(0.0);
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        for (i=0; i<_PB_W; i++) {
            y1[i][j] = a5*imgOut[i][j] + a6*tm1 + b1*ym1 + b2*ym2;
            tm1 = imgOut[i][j];
            ym2 = ym1;
            ym1 = y1 [i][j];
        }
    }


    for (j=0; j<_PB_H; j++) {
        tp1 = SCALAR_VAL(0.0);
        tp2 = SCALAR_VAL(0.0);
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        for (i=_PB_W-1; i>=0; i--) {
            y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
            tp2 = tp1;
            tp1 = imgOut[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<_PB_W; i++)
        for (j=0; j<_PB_H; j++)
            imgOut[i][j] = c2*(y1[i][j] + y2[i][j]);

#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int w = W;
  int h = H;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);


  /* Initialize array(s). */
  init_array (w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_deriche (w, h, alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut), POLYBENCH_ARRAY(y1), POLYBENCH_ARRAY(y2));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(w, h, POLYBENCH_ARRAY(imgOut)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(imgIn);
  POLYBENCH_FREE_ARRAY(imgOut);
  POLYBENCH_FREE_ARRAY(y1);
  POLYBENCH_FREE_ARRAY(y2);

  return 0;
}

// CHECK:   func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<4096x2160xf32>, %arg6: memref<4096x2160xf32>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_0 = constant 2.000000e+00 : f32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %cst_1 = constant 0.000000e+00 : f32
// CHECK-NEXT:     %c1 = constant 1 : index
// CHECK-NEXT:     %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = index_cast %arg0 : i32 to index
// CHECK-NEXT:     %2 = alloca() : memref<1xf32>
// CHECK-NEXT:     %3 = alloca() : memref<1xf32>
// CHECK-NEXT:     %4 = alloca() : memref<1xf32>
// CHECK-NEXT:     %5 = alloca() : memref<1xf32>
// CHECK-NEXT:     %6 = alloca() : memref<1xf32>
// CHECK-NEXT:     %7 = alloca() : memref<1xf32>
// CHECK-NEXT:     %8 = alloca() : memref<1xf32>
// CHECK-NEXT:     %9 = alloca() : memref<1xf32>
// CHECK-NEXT:     %10 = alloca() : memref<1xf32>
// CHECK-NEXT:     %11 = alloca() : memref<1xf32>
// CHECK-NEXT:     %12 = negf %arg2 : f32
// CHECK-NEXT:     %13 = exp %12 : f32
// CHECK-NEXT:     %14 = subf %cst, %13 : f32
// CHECK-NEXT:     %15 = mulf %14, %14 : f32
// CHECK-NEXT:     %16 = mulf %cst_0, %arg2 : f32
// CHECK-NEXT:     %17 = mulf %16, %13 : f32
// CHECK-NEXT:     %18 = addf %cst, %17 : f32
// CHECK-NEXT:     %19 = exp %16 : f32
// CHECK-NEXT:     %20 = subf %18, %19 : f32
// CHECK-NEXT:     %21 = divf %15, %20 : f32
// CHECK-NEXT:     %22 = mulf %21, %13 : f32
// CHECK-NEXT:     %23 = subf %arg2, %cst : f32
// CHECK-NEXT:     %24 = mulf %22, %23 : f32
// CHECK-NEXT:     %25 = addf %arg2, %cst : f32
// CHECK-NEXT:     %26 = mulf %22, %25 : f32
// CHECK-NEXT:     %27 = negf %21 : f32
// CHECK-NEXT:     %28 = negf %cst_0 : f32
// CHECK-NEXT:     %29 = mulf %28, %arg2 : f32
// CHECK-NEXT:     %30 = exp %29 : f32
// CHECK-NEXT:     %31 = mulf %27, %30 : f32
// CHECK-NEXT:     %32 = llvm.mlir.cast %cst_0 : f32 to !llvm.float
// CHECK-NEXT:     %33 = llvm.mlir.cast %12 : f32 to !llvm.float
// CHECK-NEXT:     %34 = "llvm.intr.pow"(%32, %33) : (!llvm.float, !llvm.float) -> !llvm.float
// CHECK-NEXT:     %35 = llvm.mlir.cast %34 : !llvm.float to f32
// CHECK-NEXT:     %36 = negf %30 : f32
// CHECK-NEXT:     %37 = sitofp %c1_i32 : i32 to f32
// CHECK-NEXT:     affine.store %cst_1, %4[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %cst_1, %5[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %cst_1, %2[%c0] : memref<1xf32>
// CHECK-NEXT:     %38 = affine.load %2[%c0] : memref<1xf32>
// CHECK-NEXT:     %39 = mulf %24, %38 : f32
// CHECK-NEXT:     %40 = affine.load %4[%c0] : memref<1xf32>
// CHECK-NEXT:     %41 = mulf %35, %40 : f32
// CHECK-NEXT:     %42 = affine.load %5[%c0] : memref<1xf32>
// CHECK-NEXT:     %43 = mulf %36, %42 : f32
// CHECK-NEXT:     %44 = affine.load %4[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %44, %5[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:       affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:         %84 = affine.load %arg3[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:         %85 = mulf %21, %84 : f32
// CHECK-NEXT:         %86 = addf %85, %39 : f32
// CHECK-NEXT:         %87 = addf %86, %41 : f32
// CHECK-NEXT:         %88 = addf %87, %43 : f32
// CHECK-NEXT:         affine.store %88, %arg5[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:         %89 = affine.load %arg3[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:         affine.store %89, %2[%c0] : memref<1xf32>
// CHECK-NEXT:         %90 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:         affine.store %90, %4[%c0] : memref<1xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %cst_1, %10[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %cst_1, %11[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %cst_1, %6[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %cst_1, %7[%c0] : memref<1xf32>
// CHECK-NEXT:     %45 = subi %0, %c1 : index
// CHECK-NEXT:     %46 = addi %45, %c1 : index
// CHECK-NEXT:     %47 = subi %46, %c1 : index
// CHECK-NEXT:     %48 = affine.load %6[%c0] : memref<1xf32>
// CHECK-NEXT:     %49 = mulf %26, %48 : f32
// CHECK-NEXT:     %50 = affine.load %7[%c0] : memref<1xf32>
// CHECK-NEXT:     %51 = mulf %31, %50 : f32
// CHECK-NEXT:     %52 = addf %49, %51 : f32
// CHECK-NEXT:     %53 = affine.load %10[%c0] : memref<1xf32>
// CHECK-NEXT:     %54 = mulf %35, %53 : f32
// CHECK-NEXT:     %55 = addf %52, %54 : f32
// CHECK-NEXT:     %56 = affine.load %11[%c0] : memref<1xf32>
// CHECK-NEXT:     %57 = mulf %36, %56 : f32
// CHECK-NEXT:     %58 = addf %55, %57 : f32
// CHECK-NEXT:     %59 = affine.load %6[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %59, %7[%c0] : memref<1xf32>
// CHECK-NEXT:     %60 = affine.load %10[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %60, %11[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:       affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:         %84 = affine.apply #map(%arg8)
// CHECK-NEXT:         affine.store %58, %arg6[%arg7, %84] : memref<4096x2160xf32>
// CHECK-NEXT:         %85 = affine.load %arg3[%arg7, %84] : memref<4096x2160xf32>
// CHECK-NEXT:         affine.store %85, %6[%c0] : memref<1xf32>
// CHECK-NEXT:         %86 = affine.load %arg6[%arg7, %84] : memref<4096x2160xf32>
// CHECK-NEXT:         affine.store %86, %10[%c0] : memref<1xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:       affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:         %84 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:         %85 = affine.load %arg6[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:         %86 = addf %84, %85 : f32
// CHECK-NEXT:         %87 = mulf %37, %86 : f32
// CHECK-NEXT:         affine.store %87, %arg4[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %cst_1, %3[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %cst_1, %4[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %cst_1, %5[%c0] : memref<1xf32>
// CHECK-NEXT:     %61 = affine.load %3[%c0] : memref<1xf32>
// CHECK-NEXT:     %62 = mulf %24, %61 : f32
// CHECK-NEXT:     %63 = affine.load %4[%c0] : memref<1xf32>
// CHECK-NEXT:     %64 = mulf %35, %63 : f32
// CHECK-NEXT:     %65 = affine.load %5[%c0] : memref<1xf32>
// CHECK-NEXT:     %66 = mulf %36, %65 : f32
// CHECK-NEXT:     %67 = affine.load %4[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %67, %5[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.for %arg7 = 0 to %0 {
// CHECK-NEXT:       affine.for %arg8 = 0 to %1 {
// CHECK-NEXT:         %84 = affine.load %arg4[%arg8, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:         %85 = mulf %21, %84 : f32
// CHECK-NEXT:         %86 = addf %85, %62 : f32
// CHECK-NEXT:         %87 = addf %86, %64 : f32
// CHECK-NEXT:         %88 = addf %87, %66 : f32
// CHECK-NEXT:         affine.store %88, %arg5[%arg8, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:         %89 = affine.load %arg4[%arg8, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:         affine.store %89, %3[%c0] : memref<1xf32>
// CHECK-NEXT:         %90 = affine.load %arg5[%arg8, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:         affine.store %90, %4[%c0] : memref<1xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %cst_1, %8[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %cst_1, %9[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %cst_1, %10[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %cst_1, %11[%c0] : memref<1xf32>
// CHECK-NEXT:     %68 = subi %1, %c1 : index
// CHECK-NEXT:     %69 = addi %68, %c1 : index
// CHECK-NEXT:     %70 = subi %69, %c1 : index
// CHECK-NEXT:     %71 = affine.load %8[%c0] : memref<1xf32>
// CHECK-NEXT:     %72 = mulf %26, %71 : f32
// CHECK-NEXT:     %73 = affine.load %9[%c0] : memref<1xf32>
// CHECK-NEXT:     %74 = mulf %31, %73 : f32
// CHECK-NEXT:     %75 = addf %72, %74 : f32
// CHECK-NEXT:     %76 = affine.load %10[%c0] : memref<1xf32>
// CHECK-NEXT:     %77 = mulf %35, %76 : f32
// CHECK-NEXT:     %78 = addf %75, %77 : f32
// CHECK-NEXT:     %79 = affine.load %11[%c0] : memref<1xf32>
// CHECK-NEXT:     %80 = mulf %36, %79 : f32
// CHECK-NEXT:     %81 = addf %78, %80 : f32
// CHECK-NEXT:     %82 = affine.load %8[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %82, %9[%c0] : memref<1xf32>
// CHECK-NEXT:     %83 = affine.load %10[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.store %83, %11[%c0] : memref<1xf32>
// CHECK-NEXT:     affine.for %arg7 = 0 to %0 {
// CHECK-NEXT:       affine.for %arg8 = 0 to %1 {
// CHECK-NEXT:         %84 = affine.apply #map(%arg8)
// CHECK-NEXT:         affine.store %81, %arg6[%84, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:         %85 = affine.load %arg4[%84, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:         affine.store %85, %8[%c0] : memref<1xf32>
// CHECK-NEXT:         %86 = affine.load %arg6[%84, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:         affine.store %86, %10[%c0] : memref<1xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:       affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:         %84 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:         %85 = affine.load %arg6[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:         %86 = addf %84, %85 : f32
// CHECK-NEXT:         %87 = mulf %37, %86 : f32
// CHECK-NEXT:         affine.store %87, %arg4[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
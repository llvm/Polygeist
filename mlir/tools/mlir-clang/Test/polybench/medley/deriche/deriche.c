// RUN: mlir-clang %s %stdinclude | FileCheck %s
// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: mlir-clang %s %polyverify %stdinclude -emit-llvm | opt -O3 -S | lli - &> %s.out2
// RUN: rm -f %s.exec1
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: mlir-clang %s %polyexec %stdinclude -emit-llvm | opt -O3 -S | lli - > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2
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


// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str6:.+]]("==END   DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str5:.+]]("\0Aend   dump: %s\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str4:.+]]("%0.2f \00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str3:.+]]("\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str2:.+]]("imgOut\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str1:.+]]("begin dump: %s\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @[[str0:.+]]("==BEGIN DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("\00")
// CHECK-NEXT:   llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 2.500000e-01 : f64
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c313_i32 = constant 313 : i32
// CHECK-NEXT:     %c991_i32 = constant 991 : i32
// CHECK-NEXT:     %c65536_i32 = constant 65536 : i32
// CHECK-NEXT:     %cst_0 = constant 6.553500e+04 : f32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = fptrunc %cst : f64 to f32
// CHECK-NEXT:     store %0, %arg2[%c0] : memref<?xf32>
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %2 = cmpi "slt", %1, %arg0 : i32
// CHECK-NEXT:     cond_br %2, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%3: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %4 = cmpi "slt", %3, %arg1 : i32
// CHECK-NEXT:     cond_br %4, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %5 = index_cast %1 : i32 to index
// CHECK-NEXT:     %6 = index_cast %3 : i32 to index
// CHECK-NEXT:     %7 = muli %1, %c313_i32 : i32
// CHECK-NEXT:     %8 = muli %3, %c991_i32 : i32
// CHECK-NEXT:     %9 = addi %7, %8 : i32
// CHECK-NEXT:     %10 = remi_signed %9, %c65536_i32 : i32
// CHECK-NEXT:     %11 = sitofp %10 : i32 to f32
// CHECK-NEXT:     %12 = divf %11, %cst_0 : f32
// CHECK-NEXT:     store %12, %arg3[%5, %6] : memref<4096x2160xf32>
// CHECK-NEXT:     %13 = addi %3, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%13 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %14 = addi %1, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%14 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<4096x2160xf32>, %arg6: memref<4096x2160xf32>) {
// CHECK-NEXT:     %cst = constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_0 = constant 2.000000e+00 : f32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %cst_1 = constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = negf %arg2 : f32
// CHECK-NEXT:     %1 = exp %0 : f32
// CHECK-NEXT:     %2 = subf %cst, %1 : f32
// CHECK-NEXT:     %3 = mulf %2, %2 : f32
// CHECK-NEXT:     %4 = mulf %cst_0, %arg2 : f32
// CHECK-NEXT:     %5 = mulf %4, %1 : f32
// CHECK-NEXT:     %6 = addf %cst, %5 : f32
// CHECK-NEXT:     %7 = exp %4 : f32
// CHECK-NEXT:     %8 = subf %6, %7 : f32
// CHECK-NEXT:     %9 = divf %3, %8 : f32
// CHECK-NEXT:     %10 = mulf %9, %1 : f32
// CHECK-NEXT:     %11 = subf %arg2, %cst : f32
// CHECK-NEXT:     %12 = mulf %10, %11 : f32
// CHECK-NEXT:     %13 = addf %arg2, %cst : f32
// CHECK-NEXT:     %14 = mulf %10, %13 : f32
// CHECK-NEXT:     %15 = negf %9 : f32
// CHECK-NEXT:     %16 = negf %cst_0 : f32
// CHECK-NEXT:     %17 = mulf %16, %arg2 : f32
// CHECK-NEXT:     %18 = exp %17 : f32
// CHECK-NEXT:     %19 = mulf %15, %18 : f32
// CHECK-NEXT:     %20 = llvm.mlir.cast %cst_0 : f32 to !llvm.float
// CHECK-NEXT:     %21 = llvm.mlir.cast %0 : f32 to !llvm.float
// CHECK-NEXT:     %22 = "llvm.intr.pow"(%20, %21) : (!llvm.float, !llvm.float) -> !llvm.float
// CHECK-NEXT:     %23 = llvm.mlir.cast %22 : !llvm.float to f32
// CHECK-NEXT:     %24 = negf %18 : f32
// CHECK-NEXT:     %25 = sitofp %c1_i32 : i32 to f32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%26: i32):  // 2 preds: ^bb0, ^bb4
// CHECK-NEXT:     %27 = cmpi "slt", %26, %arg0 : i32
// CHECK-NEXT:     cond_br %27, ^bb2(%c0_i32, %cst_1, %cst_1, %cst_1 : i32, f32, f32, f32), ^bb5(%c0_i32 : i32)
// CHECK-NEXT:   ^bb2(%28: i32, %29: f32, %30: f32, %31: f32):  // 2 preds: ^bb1, ^bb3
// CHECK-NEXT:     %32 = cmpi "slt", %28, %arg1 : i32
// CHECK-NEXT:     cond_br %32, ^bb3, ^bb4
// CHECK-NEXT:   ^bb3:  // pred: ^bb2
// CHECK-NEXT:     %33 = index_cast %26 : i32 to index
// CHECK-NEXT:     %34 = index_cast %28 : i32 to index
// CHECK-NEXT:     %35 = load %arg3[%33, %34] : memref<4096x2160xf32>
// CHECK-NEXT:     %36 = mulf %9, %35 : f32
// CHECK-NEXT:     %37 = mulf %12, %29 : f32
// CHECK-NEXT:     %38 = addf %36, %37 : f32
// CHECK-NEXT:     %39 = mulf %23, %30 : f32
// CHECK-NEXT:     %40 = addf %38, %39 : f32
// CHECK-NEXT:     %41 = mulf %24, %31 : f32
// CHECK-NEXT:     %42 = addf %40, %41 : f32
// CHECK-NEXT:     store %42, %arg5[%33, %34] : memref<4096x2160xf32>
// CHECK-NEXT:     %43 = load %arg3[%33, %34] : memref<4096x2160xf32>
// CHECK-NEXT:     %44 = load %arg5[%33, %34] : memref<4096x2160xf32>
// CHECK-NEXT:     %45 = addi %28, %c1_i32 : i32
// CHECK-NEXT:     br ^bb2(%45, %43, %44, %30 : i32, f32, f32, f32)
// CHECK-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-NEXT:     %46 = addi %26, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%46 : i32)
// CHECK-NEXT:   ^bb5(%47: i32):  // 2 preds: ^bb1, ^bb9
// CHECK-NEXT:     %48 = cmpi "slt", %47, %arg0 : i32
// CHECK-NEXT:     cond_br %48, ^bb6, ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %49 = subi %arg1, %c1_i32 : i32
// CHECK-NEXT:     br ^bb7(%49, %cst_1, %cst_1, %cst_1, %cst_1 : i32, f32, f32, f32, f32)
// CHECK-NEXT:   ^bb7(%50: i32, %51: f32, %52: f32, %53: f32, %54: f32):  // 2 preds: ^bb6, ^bb8
// CHECK-NEXT:     %55 = cmpi "sge", %50, %c0_i32 : i32
// CHECK-NEXT:     cond_br %55, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %56 = index_cast %47 : i32 to index
// CHECK-NEXT:     %57 = index_cast %50 : i32 to index
// CHECK-NEXT:     %58 = mulf %14, %51 : f32
// CHECK-NEXT:     %59 = mulf %19, %52 : f32
// CHECK-NEXT:     %60 = addf %58, %59 : f32
// CHECK-NEXT:     %61 = mulf %23, %53 : f32
// CHECK-NEXT:     %62 = addf %60, %61 : f32
// CHECK-NEXT:     %63 = mulf %24, %54 : f32
// CHECK-NEXT:     %64 = addf %62, %63 : f32
// CHECK-NEXT:     store %64, %arg6[%56, %57] : memref<4096x2160xf32>
// CHECK-NEXT:     %65 = load %arg3[%56, %57] : memref<4096x2160xf32>
// CHECK-NEXT:     %66 = load %arg6[%56, %57] : memref<4096x2160xf32>
// CHECK-NEXT:     %67 = subi %50, %c1_i32 : i32
// CHECK-NEXT:     br ^bb7(%67, %65, %51, %66, %53 : i32, f32, f32, f32, f32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %68 = addi %47, %c1_i32 : i32
// CHECK-NEXT:     br ^bb5(%68 : i32)
// CHECK-NEXT:   ^bb10(%69: i32):  // 2 preds: ^bb5, ^bb13
// CHECK-NEXT:     %70 = cmpi "slt", %69, %arg0 : i32
// CHECK-NEXT:     cond_br %70, ^bb11(%c0_i32 : i32), ^bb14(%c0_i32 : i32)
// CHECK-NEXT:   ^bb11(%71: i32):  // 2 preds: ^bb10, ^bb12
// CHECK-NEXT:     %72 = cmpi "slt", %71, %arg1 : i32
// CHECK-NEXT:     cond_br %72, ^bb12, ^bb13
// CHECK-NEXT:   ^bb12:  // pred: ^bb11
// CHECK-NEXT:     %73 = index_cast %69 : i32 to index
// CHECK-NEXT:     %74 = index_cast %71 : i32 to index
// CHECK-NEXT:     %75 = load %arg5[%73, %74] : memref<4096x2160xf32>
// CHECK-NEXT:     %76 = load %arg6[%73, %74] : memref<4096x2160xf32>
// CHECK-NEXT:     %77 = addf %75, %76 : f32
// CHECK-NEXT:     %78 = mulf %25, %77 : f32
// CHECK-NEXT:     store %78, %arg4[%73, %74] : memref<4096x2160xf32>
// CHECK-NEXT:     %79 = addi %71, %c1_i32 : i32
// CHECK-NEXT:     br ^bb11(%79 : i32)
// CHECK-NEXT:   ^bb13:  // pred: ^bb11
// CHECK-NEXT:     %80 = addi %69, %c1_i32 : i32
// CHECK-NEXT:     br ^bb10(%80 : i32)
// CHECK-NEXT:   ^bb14(%81: i32):  // 2 preds: ^bb10, ^bb17
// CHECK-NEXT:     %82 = cmpi "slt", %81, %arg1 : i32
// CHECK-NEXT:     cond_br %82, ^bb15(%c0_i32, %cst_1, %cst_1, %cst_1 : i32, f32, f32, f32), ^bb18(%c0_i32 : i32)
// CHECK-NEXT:   ^bb15(%83: i32, %84: f32, %85: f32, %86: f32):  // 2 preds: ^bb14, ^bb16
// CHECK-NEXT:     %87 = cmpi "slt", %83, %arg0 : i32
// CHECK-NEXT:     cond_br %87, ^bb16, ^bb17
// CHECK-NEXT:   ^bb16:  // pred: ^bb15
// CHECK-NEXT:     %88 = index_cast %83 : i32 to index
// CHECK-NEXT:     %89 = index_cast %81 : i32 to index
// CHECK-NEXT:     %90 = load %arg4[%88, %89] : memref<4096x2160xf32>
// CHECK-NEXT:     %91 = mulf %9, %90 : f32
// CHECK-NEXT:     %92 = mulf %12, %84 : f32
// CHECK-NEXT:     %93 = addf %91, %92 : f32
// CHECK-NEXT:     %94 = mulf %23, %85 : f32
// CHECK-NEXT:     %95 = addf %93, %94 : f32
// CHECK-NEXT:     %96 = mulf %24, %86 : f32
// CHECK-NEXT:     %97 = addf %95, %96 : f32
// CHECK-NEXT:     store %97, %arg5[%88, %89] : memref<4096x2160xf32>
// CHECK-NEXT:     %98 = load %arg4[%88, %89] : memref<4096x2160xf32>
// CHECK-NEXT:     %99 = load %arg5[%88, %89] : memref<4096x2160xf32>
// CHECK-NEXT:     %100 = addi %83, %c1_i32 : i32
// CHECK-NEXT:     br ^bb15(%100, %98, %99, %85 : i32, f32, f32, f32)
// CHECK-NEXT:   ^bb17:  // pred: ^bb15
// CHECK-NEXT:     %101 = addi %81, %c1_i32 : i32
// CHECK-NEXT:     br ^bb14(%101 : i32)
// CHECK-NEXT:   ^bb18(%102: i32):  // 2 preds: ^bb14, ^bb22
// CHECK-NEXT:     %103 = cmpi "slt", %102, %arg1 : i32
// CHECK-NEXT:     cond_br %103, ^bb19, ^bb23(%c0_i32 : i32)
// CHECK-NEXT:   ^bb19:  // pred: ^bb18
// CHECK-NEXT:     %104 = subi %arg0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb20(%104, %cst_1, %cst_1, %cst_1, %cst_1 : i32, f32, f32, f32, f32)
// CHECK-NEXT:   ^bb20(%105: i32, %106: f32, %107: f32, %108: f32, %109: f32):  // 2 preds: ^bb19, ^bb21
// CHECK-NEXT:     %110 = cmpi "sge", %105, %c0_i32 : i32
// CHECK-NEXT:     cond_br %110, ^bb21, ^bb22
// CHECK-NEXT:   ^bb21:  // pred: ^bb20
// CHECK-NEXT:     %111 = index_cast %105 : i32 to index
// CHECK-NEXT:     %112 = index_cast %102 : i32 to index
// CHECK-NEXT:     %113 = mulf %14, %106 : f32
// CHECK-NEXT:     %114 = mulf %19, %107 : f32
// CHECK-NEXT:     %115 = addf %113, %114 : f32
// CHECK-NEXT:     %116 = mulf %23, %108 : f32
// CHECK-NEXT:     %117 = addf %115, %116 : f32
// CHECK-NEXT:     %118 = mulf %24, %109 : f32
// CHECK-NEXT:     %119 = addf %117, %118 : f32
// CHECK-NEXT:     store %119, %arg6[%111, %112] : memref<4096x2160xf32>
// CHECK-NEXT:     %120 = load %arg4[%111, %112] : memref<4096x2160xf32>
// CHECK-NEXT:     %121 = load %arg6[%111, %112] : memref<4096x2160xf32>
// CHECK-NEXT:     %122 = subi %105, %c1_i32 : i32
// CHECK-NEXT:     br ^bb20(%122, %120, %106, %121, %108 : i32, f32, f32, f32, f32)
// CHECK-NEXT:   ^bb22:  // pred: ^bb20
// CHECK-NEXT:     %123 = addi %102, %c1_i32 : i32
// CHECK-NEXT:     br ^bb18(%123 : i32)
// CHECK-NEXT:   ^bb23(%124: i32):  // 2 preds: ^bb18, ^bb27
// CHECK-NEXT:     %125 = cmpi "slt", %124, %arg0 : i32
// CHECK-NEXT:     cond_br %125, ^bb25(%c0_i32 : i32), ^bb24
// CHECK-NEXT:   ^bb24:  // pred: ^bb23
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb25(%126: i32):  // 2 preds: ^bb23, ^bb26
// CHECK-NEXT:     %127 = cmpi "slt", %126, %arg1 : i32
// CHECK-NEXT:     cond_br %127, ^bb26, ^bb27
// CHECK-NEXT:   ^bb26:  // pred: ^bb25
// CHECK-NEXT:     %128 = index_cast %124 : i32 to index
// CHECK-NEXT:     %129 = index_cast %126 : i32 to index
// CHECK-NEXT:     %130 = load %arg5[%128, %129] : memref<4096x2160xf32>
// CHECK-NEXT:     %131 = load %arg6[%128, %129] : memref<4096x2160xf32>
// CHECK-NEXT:     %132 = addf %130, %131 : f32
// CHECK-NEXT:     %133 = mulf %25, %132 : f32
// CHECK-NEXT:     store %133, %arg4[%128, %129] : memref<4096x2160xf32>
// CHECK-NEXT:     %134 = addi %126, %c1_i32 : i32
// CHECK-NEXT:     br ^bb25(%134 : i32)
// CHECK-NEXT:   ^bb27:  // pred: ^bb25
// CHECK-NEXT:     %135 = addi %124, %c1_i32 : i32
// CHECK-NEXT:     br ^bb23(%135 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<4096x2160xf32>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %2 = llvm.mlir.addressof @[[str0]] : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %3 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:     %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %5 = llvm.call @fprintf(%1, %4) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %6 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %8 = llvm.mlir.addressof @[[str1]] : !llvm.ptr<array<15 x i8>>
// CHECK-NEXT:     %9 = llvm.getelementptr %8[%3, %3] : (!llvm.ptr<array<15 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %10 = llvm.mlir.addressof @[[str2]] : !llvm.ptr<array<7 x i8>>
// CHECK-NEXT:     %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %14 = cmpi "slt", %13, %arg0 : i32
// CHECK-NEXT:     cond_br %14, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %15 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %16 = llvm.load %15 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %17 = llvm.mlir.addressof @[[str5]] : !llvm.ptr<array<17 x i8>>
// CHECK-NEXT:     %18 = llvm.getelementptr %17[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %19 = llvm.mlir.addressof @[[str2]] : !llvm.ptr<array<7 x i8>>
// CHECK-NEXT:     %20 = llvm.getelementptr %19[%3, %3] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %21 = llvm.call @fprintf(%16, %18, %20) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %22 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %23 = llvm.load %22 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %24 = llvm.mlir.addressof @[[str6]] : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %25 = llvm.getelementptr %24[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %26 = llvm.call @fprintf(%23, %25) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%27: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %28 = cmpi "slt", %27, %arg1 : i32
// CHECK-NEXT:     cond_br %28, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %29 = muli %13, %arg1 : i32
// CHECK-NEXT:     %30 = addi %29, %27 : i32
// CHECK-NEXT:     %31 = remi_signed %30, %c20_i32 : i32
// CHECK-NEXT:     %32 = cmpi "eq", %31, %c0_i32 : i32
// CHECK-NEXT:     scf.if %32 {
// CHECK-NEXT:       %45 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %46 = llvm.load %45 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %47 = llvm.mlir.addressof @[[str3]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:       %48 = llvm.getelementptr %47[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %49 = llvm.call @fprintf(%46, %48) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %33 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %34 = llvm.load %33 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %35 = llvm.mlir.addressof @[[str4]] : !llvm.ptr<array<7 x i8>>
// CHECK-NEXT:     %36 = llvm.getelementptr %35[%3, %3] : (!llvm.ptr<array<7 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %37 = index_cast %13 : i32 to index
// CHECK-NEXT:     %38 = index_cast %27 : i32 to index
// CHECK-NEXT:     %39 = load %arg2[%37, %38] : memref<4096x2160xf32>
// CHECK-NEXT:     %40 = fpext %39 : f32 to f64
// CHECK-NEXT:     %41 = llvm.mlir.cast %40 : f64 to !llvm.double
// CHECK-NEXT:     %42 = llvm.call @fprintf(%34, %36, %41) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %43 = addi %27, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%43 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %44 = addi %13, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%44 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @free(memref<?xi8>)
// CHECK-NEXT: }

// EXEC: {{[0-9]\.[0-9]+}}
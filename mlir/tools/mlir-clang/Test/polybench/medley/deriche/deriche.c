// RUN: mlir-clang %s main %stdinclude | FileCheck %s
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
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("==END   DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("\0Aend   dump: %s\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("%0.2f ")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("\0A")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("imgOut")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("begin dump: %s")
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("==BEGIN DUMP_ARRAYS==\0A")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c4096_i32 = constant 4096 : i32
// CHECK-NEXT:     %c2160_i32 = constant 2160 : i32
// CHECK-NEXT:     %0 = alloca() : memref<1xf32>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %1 = addi %c4096_i32, %c0_i32 : i32
// CHECK-NEXT:     %2 = addi %c2160_i32, %c0_i32 : i32
// CHECK-NEXT:     %3 = muli %1, %2 : i32
// CHECK-NEXT:     %4 = zexti %3 : i32 to i64
// CHECK-NEXT:     %c4_i64 = constant 4 : i64
// CHECK-NEXT:     %5 = trunci %c4_i64 : i64 to i32
// CHECK-NEXT:     %6 = call @polybench_alloc_data(%4, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %7 = memref_cast %6 : memref<?xi8> to memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:     %8 = call @polybench_alloc_data(%4, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %9 = memref_cast %8 : memref<?xi8> to memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:     %10 = call @polybench_alloc_data(%4, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<?xi8> to memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:     %12 = call @polybench_alloc_data(%4, %5) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %13 = memref_cast %12 : memref<?xi8> to memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:     %14 = memref_cast %0 : memref<1xf32> to memref<?xf32>
// CHECK-NEXT:     %15 = load %7[%c0] : memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<?x2160xf32> to memref<4096x2160xf32>
// CHECK-NEXT:     %18 = load %9[%c0] : memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?x2160xf32> to memref<4096x2160xf32>
// CHECK-NEXT:     call @init_array(%c4096_i32, %c2160_i32, %14, %17, %20) : (i32, i32, memref<?xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
// CHECK-NEXT:     %21 = load %0[%c0] : memref<1xf32>
// CHECK-NEXT:     %22 = load %7[%c0] : memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:     %23 = memref_cast %22 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %24 = memref_cast %23 : memref<?x2160xf32> to memref<4096x2160xf32>
// CHECK-NEXT:     %25 = load %9[%c0] : memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:     %26 = memref_cast %25 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %27 = memref_cast %26 : memref<?x2160xf32> to memref<4096x2160xf32>
// CHECK-NEXT:     %28 = load %11[%c0] : memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %30 = memref_cast %29 : memref<?x2160xf32> to memref<4096x2160xf32>
// CHECK-NEXT:     %31 = load %13[%c0] : memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:     %32 = memref_cast %31 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %33 = memref_cast %32 : memref<?x2160xf32> to memref<4096x2160xf32>
// CHECK-NEXT:     call @kernel_deriche(%c4096_i32, %c2160_i32, %21, %24, %27, %30, %33) : (i32, i32, f32, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>, memref<4096x2160xf32>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %34 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %35 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %36 = addi %c0, %35 : index
// CHECK-NEXT:     %37 = load %arg1[%36] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %38 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %39 = call @strcmp(%37, %38) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %40 = trunci %39 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %41 = xor %40, %true : i1
// CHECK-NEXT:     %42 = and %34, %41 : i1
// CHECK-NEXT:     scf.if %42 {
// CHECK-NEXT:       %47 = load %9[%c0] : memref<?xmemref<4096x2160xf32>>
// CHECK-NEXT:       %48 = memref_cast %47 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:       %49 = memref_cast %48 : memref<?x2160xf32> to memref<4096x2160xf32>
// CHECK-NEXT:       call @print_array(%c4096_i32, %c2160_i32, %49) : (i32, i32, memref<4096x2160xf32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %43 = memref_cast %7 : memref<?xmemref<4096x2160xf32>> to memref<?xi8>
// CHECK-NEXT:     call @free(%43) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %44 = memref_cast %9 : memref<?xmemref<4096x2160xf32>> to memref<?xi8>
// CHECK-NEXT:     call @free(%44) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %45 = memref_cast %11 : memref<?xmemref<4096x2160xf32>> to memref<?xi8>
// CHECK-NEXT:     call @free(%45) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %46 = memref_cast %13 : memref<?xmemref<4096x2160xf32>> to memref<?xi8>
// CHECK-NEXT:     call @free(%46) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 2.500000e-01 : f64
// CHECK-NEXT:     %0 = fptrunc %cst : f64 to f32
// CHECK-NEXT:     store %0, %arg2[%c0] : memref<?xf32>
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %2 = cmpi "slt", %1, %arg0 : i32
// CHECK-NEXT:     cond_br %2, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%3: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %4 = cmpi "slt", %3, %arg1 : i32
// CHECK-NEXT:     cond_br %4, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %5 = index_cast %1 : i32 to index
// CHECK-NEXT:     %6 = addi %c0, %5 : index
// CHECK-NEXT:     %7 = memref_cast %arg3 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %8 = index_cast %3 : i32 to index
// CHECK-NEXT:     %9 = addi %c0, %8 : index
// CHECK-NEXT:     %c313_i32 = constant 313 : i32
// CHECK-NEXT:     %10 = muli %c313_i32, %1 : i32
// CHECK-NEXT:     %c991_i32 = constant 991 : i32
// CHECK-NEXT:     %11 = muli %c991_i32, %3 : i32
// CHECK-NEXT:     %12 = addi %10, %11 : i32
// CHECK-NEXT:     %c65536_i32 = constant 65536 : i32
// CHECK-NEXT:     %13 = remi_signed %12, %c65536_i32 : i32
// CHECK-NEXT:     %14 = sitofp %13 : i32 to f32
// CHECK-NEXT:     %cst_0 = constant 6.553500e+04 : f32
// CHECK-NEXT:     %15 = divf %14, %cst_0 : f32
// CHECK-NEXT:     store %15, %7[%6, %9] : memref<?x2160xf32>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %16 = addi %3, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%16 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %17 = addi %1, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb1(%17 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<4096x2160xf32>, %arg6: memref<4096x2160xf32>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.000000e+00 : f32
// CHECK-NEXT:     %0 = negf %arg2 : f32
// CHECK-NEXT:     %1 = exp %0 : f32
// CHECK-NEXT:     %2 = subf %cst, %1 : f32
// CHECK-NEXT:     %3 = mulf %2, %2 : f32
// CHECK-NEXT:     %cst_0 = constant 2.000000e+00 : f32
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
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %25 = sitofp %c1_i32 : i32 to f32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%26: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %27 = cmpi "slt", %26, %arg0 : i32
// CHECK-NEXT:     cond_br %27, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %cst_1 = constant 0.000000e+00 : f32
// CHECK-NEXT:     br ^bb4(%c0_i32, %cst_1, %cst_1, %cst_1 : i32, f32, f32, f32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%28: i32, %29: f32, %30: f32, %31: f32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %32 = cmpi "slt", %28, %arg1 : i32
// CHECK-NEXT:     cond_br %32, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %33 = index_cast %26 : i32 to index
// CHECK-NEXT:     %34 = addi %c0, %33 : index
// CHECK-NEXT:     %35 = memref_cast %arg5 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %36 = index_cast %28 : i32 to index
// CHECK-NEXT:     %37 = addi %c0, %36 : index
// CHECK-NEXT:     %38 = memref_cast %arg3 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %39 = load %38[%34, %37] : memref<?x2160xf32>
// CHECK-NEXT:     %40 = mulf %9, %39 : f32
// CHECK-NEXT:     %41 = mulf %12, %29 : f32
// CHECK-NEXT:     %42 = addf %40, %41 : f32
// CHECK-NEXT:     %43 = mulf %23, %30 : f32
// CHECK-NEXT:     %44 = addf %42, %43 : f32
// CHECK-NEXT:     %45 = mulf %24, %31 : f32
// CHECK-NEXT:     %46 = addf %44, %45 : f32
// CHECK-NEXT:     store %46, %35[%34, %37] : memref<?x2160xf32>
// CHECK-NEXT:     %47 = load %38[%34, %37] : memref<?x2160xf32>
// CHECK-NEXT:     %48 = load %35[%34, %37] : memref<?x2160xf32>
// CHECK-NEXT:     %49 = addi %28, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%49, %47, %48, %30 : i32, f32, f32, f32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %50 = addi %26, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%50 : i32)
// CHECK-NEXT:   ^bb7(%51: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %52 = cmpi "slt", %51, %arg0 : i32
// CHECK-NEXT:     cond_br %52, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %cst_2 = constant 0.000000e+00 : f32
// CHECK-NEXT:     %53 = subi %arg1, %c1_i32 : i32
// CHECK-NEXT:     br ^bb10(%53, %cst_2, %cst_2, %cst_2, %cst_2 : i32, f32, f32, f32, f32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%54: i32, %55: f32, %56: f32, %57: f32, %58: f32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %59 = cmpi "sge", %54, %c0_i32 : i32
// CHECK-NEXT:     cond_br %59, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %60 = index_cast %51 : i32 to index
// CHECK-NEXT:     %61 = addi %c0, %60 : index
// CHECK-NEXT:     %62 = memref_cast %arg6 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %63 = index_cast %54 : i32 to index
// CHECK-NEXT:     %64 = addi %c0, %63 : index
// CHECK-NEXT:     %65 = mulf %14, %55 : f32
// CHECK-NEXT:     %66 = mulf %19, %56 : f32
// CHECK-NEXT:     %67 = addf %65, %66 : f32
// CHECK-NEXT:     %68 = mulf %23, %57 : f32
// CHECK-NEXT:     %69 = addf %67, %68 : f32
// CHECK-NEXT:     %70 = mulf %24, %58 : f32
// CHECK-NEXT:     %71 = addf %69, %70 : f32
// CHECK-NEXT:     store %71, %62[%61, %64] : memref<?x2160xf32>
// CHECK-NEXT:     %72 = memref_cast %arg3 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %73 = load %72[%61, %64] : memref<?x2160xf32>
// CHECK-NEXT:     %74 = load %62[%61, %64] : memref<?x2160xf32>
// CHECK-NEXT:     %75 = subi %54, %c1_i32 : i32
// CHECK-NEXT:     br ^bb10(%75, %73, %55, %74, %57 : i32, f32, f32, f32, f32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %76 = addi %51, %c1_i32 : i32
// CHECK-NEXT:     br ^bb7(%76 : i32)
// CHECK-NEXT:   ^bb13(%77: i32):  // 2 preds: ^bb9, ^bb18
// CHECK-NEXT:     %78 = cmpi "slt", %77, %arg0 : i32
// CHECK-NEXT:     cond_br %78, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     br ^bb19(%c0_i32 : i32)
// CHECK-NEXT:   ^bb16(%79: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %80 = cmpi "slt", %79, %arg1 : i32
// CHECK-NEXT:     cond_br %80, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %81 = index_cast %77 : i32 to index
// CHECK-NEXT:     %82 = addi %c0, %81 : index
// CHECK-NEXT:     %83 = memref_cast %arg4 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %84 = index_cast %79 : i32 to index
// CHECK-NEXT:     %85 = addi %c0, %84 : index
// CHECK-NEXT:     %86 = memref_cast %arg5 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %87 = load %86[%82, %85] : memref<?x2160xf32>
// CHECK-NEXT:     %88 = memref_cast %arg6 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %89 = load %88[%82, %85] : memref<?x2160xf32>
// CHECK-NEXT:     %90 = addf %87, %89 : f32
// CHECK-NEXT:     %91 = mulf %25, %90 : f32
// CHECK-NEXT:     store %91, %83[%82, %85] : memref<?x2160xf32>
// CHECK-NEXT:     %92 = addi %79, %c1_i32 : i32
// CHECK-NEXT:     br ^bb16(%92 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %93 = addi %77, %c1_i32 : i32
// CHECK-NEXT:     br ^bb13(%93 : i32)
// CHECK-NEXT:   ^bb19(%94: i32):  // 2 preds: ^bb15, ^bb24
// CHECK-NEXT:     %95 = cmpi "slt", %94, %arg1 : i32
// CHECK-NEXT:     cond_br %95, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     %cst_3 = constant 0.000000e+00 : f32
// CHECK-NEXT:     br ^bb22(%c0_i32, %cst_3, %cst_3, %cst_3 : i32, f32, f32, f32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     br ^bb25(%c0_i32 : i32)
// CHECK-NEXT:   ^bb22(%96: i32, %97: f32, %98: f32, %99: f32):  // 2 preds: ^bb20, ^bb23
// CHECK-NEXT:     %100 = cmpi "slt", %96, %arg0 : i32
// CHECK-NEXT:     cond_br %100, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     %101 = index_cast %96 : i32 to index
// CHECK-NEXT:     %102 = addi %c0, %101 : index
// CHECK-NEXT:     %103 = memref_cast %arg5 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %104 = index_cast %94 : i32 to index
// CHECK-NEXT:     %105 = addi %c0, %104 : index
// CHECK-NEXT:     %106 = memref_cast %arg4 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %107 = load %106[%102, %105] : memref<?x2160xf32>
// CHECK-NEXT:     %108 = mulf %9, %107 : f32
// CHECK-NEXT:     %109 = mulf %12, %97 : f32
// CHECK-NEXT:     %110 = addf %108, %109 : f32
// CHECK-NEXT:     %111 = mulf %23, %98 : f32
// CHECK-NEXT:     %112 = addf %110, %111 : f32
// CHECK-NEXT:     %113 = mulf %24, %99 : f32
// CHECK-NEXT:     %114 = addf %112, %113 : f32
// CHECK-NEXT:     store %114, %103[%102, %105] : memref<?x2160xf32>
// CHECK-NEXT:     %115 = load %106[%102, %105] : memref<?x2160xf32>
// CHECK-NEXT:     %116 = load %103[%102, %105] : memref<?x2160xf32>
// CHECK-NEXT:     %117 = addi %96, %c1_i32 : i32
// CHECK-NEXT:     br ^bb22(%117, %115, %116, %98 : i32, f32, f32, f32)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     %118 = addi %94, %c1_i32 : i32
// CHECK-NEXT:     br ^bb19(%118 : i32)
// CHECK-NEXT:   ^bb25(%119: i32):  // 2 preds: ^bb21, ^bb30
// CHECK-NEXT:     %120 = cmpi "slt", %119, %arg1 : i32
// CHECK-NEXT:     cond_br %120, ^bb26, ^bb27
// CHECK-NEXT:   ^bb26:  // pred: ^bb25
// CHECK-NEXT:     %cst_4 = constant 0.000000e+00 : f32
// CHECK-NEXT:     %121 = subi %arg0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb28(%121, %cst_4, %cst_4, %cst_4, %cst_4 : i32, f32, f32, f32, f32)
// CHECK-NEXT:   ^bb27:  // pred: ^bb25
// CHECK-NEXT:     br ^bb31(%c0_i32 : i32)
// CHECK-NEXT:   ^bb28(%122: i32, %123: f32, %124: f32, %125: f32, %126: f32):  // 2 preds: ^bb26, ^bb29
// CHECK-NEXT:     %127 = cmpi "sge", %122, %c0_i32 : i32
// CHECK-NEXT:     cond_br %127, ^bb29, ^bb30
// CHECK-NEXT:   ^bb29:  // pred: ^bb28
// CHECK-NEXT:     %128 = index_cast %122 : i32 to index
// CHECK-NEXT:     %129 = addi %c0, %128 : index
// CHECK-NEXT:     %130 = memref_cast %arg6 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %131 = index_cast %119 : i32 to index
// CHECK-NEXT:     %132 = addi %c0, %131 : index
// CHECK-NEXT:     %133 = mulf %14, %123 : f32
// CHECK-NEXT:     %134 = mulf %19, %124 : f32
// CHECK-NEXT:     %135 = addf %133, %134 : f32
// CHECK-NEXT:     %136 = mulf %23, %125 : f32
// CHECK-NEXT:     %137 = addf %135, %136 : f32
// CHECK-NEXT:     %138 = mulf %24, %126 : f32
// CHECK-NEXT:     %139 = addf %137, %138 : f32
// CHECK-NEXT:     store %139, %130[%129, %132] : memref<?x2160xf32>
// CHECK-NEXT:     %140 = memref_cast %arg4 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %141 = load %140[%129, %132] : memref<?x2160xf32>
// CHECK-NEXT:     %142 = load %130[%129, %132] : memref<?x2160xf32>
// CHECK-NEXT:     %143 = subi %122, %c1_i32 : i32
// CHECK-NEXT:     br ^bb28(%143, %141, %123, %142, %125 : i32, f32, f32, f32, f32)
// CHECK-NEXT:   ^bb30:  // pred: ^bb28
// CHECK-NEXT:     %144 = addi %119, %c1_i32 : i32
// CHECK-NEXT:     br ^bb25(%144 : i32)
// CHECK-NEXT:   ^bb31(%145: i32):  // 2 preds: ^bb27, ^bb36
// CHECK-NEXT:     %146 = cmpi "slt", %145, %arg0 : i32
// CHECK-NEXT:     cond_br %146, ^bb32, ^bb33
// CHECK-NEXT:   ^bb32:  // pred: ^bb31
// CHECK-NEXT:     br ^bb34(%c0_i32 : i32)
// CHECK-NEXT:   ^bb33:  // pred: ^bb31
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb34(%147: i32):  // 2 preds: ^bb32, ^bb35
// CHECK-NEXT:     %148 = cmpi "slt", %147, %arg1 : i32
// CHECK-NEXT:     cond_br %148, ^bb35, ^bb36
// CHECK-NEXT:   ^bb35:  // pred: ^bb34
// CHECK-NEXT:     %149 = index_cast %145 : i32 to index
// CHECK-NEXT:     %150 = addi %c0, %149 : index
// CHECK-NEXT:     %151 = memref_cast %arg4 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %152 = index_cast %147 : i32 to index
// CHECK-NEXT:     %153 = addi %c0, %152 : index
// CHECK-NEXT:     %154 = memref_cast %arg5 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %155 = load %154[%150, %153] : memref<?x2160xf32>
// CHECK-NEXT:     %156 = memref_cast %arg6 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %157 = load %156[%150, %153] : memref<?x2160xf32>
// CHECK-NEXT:     %158 = addf %155, %157 : f32
// CHECK-NEXT:     %159 = mulf %25, %158 : f32
// CHECK-NEXT:     store %159, %151[%150, %153] : memref<?x2160xf32>
// CHECK-NEXT:     %160 = addi %147, %c1_i32 : i32
// CHECK-NEXT:     br ^bb34(%160 : i32)
// CHECK-NEXT:   ^bb36:  // pred: ^bb34
// CHECK-NEXT:     %161 = addi %145, %c1_i32 : i32
// CHECK-NEXT:     br ^bb31(%161 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<4096x2160xf32>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %0 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %1 = llvm.mlir.addressof @str0 : !llvm.ptr<array<22 x i8>>
// CHECK-NEXT:     %2 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:     %3 = llvm.getelementptr %1[%2, %2] : (!llvm.ptr<array<22 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %4 = llvm.call @fprintf(%0, %3) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %5 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %6 = llvm.mlir.addressof @str1 : !llvm.ptr<array<14 x i8>>
// CHECK-NEXT:     %7 = llvm.getelementptr %6[%2, %2] : (!llvm.ptr<array<14 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %8 = llvm.mlir.addressof @str2 : !llvm.ptr<array<6 x i8>>
// CHECK-NEXT:     %9 = llvm.getelementptr %8[%2, %2] : (!llvm.ptr<array<6 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %10 = llvm.call @fprintf(%5, %7, %9) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%11: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %12 = cmpi "slt", %11, %arg0 : i32
// CHECK-NEXT:     cond_br %12, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     %13 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %14 = llvm.mlir.addressof @str5 : !llvm.ptr<array<16 x i8>>
// CHECK-NEXT:     %15 = llvm.getelementptr %14[%2, %2] : (!llvm.ptr<array<16 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %16 = llvm.mlir.addressof @str2 : !llvm.ptr<array<6 x i8>>
// CHECK-NEXT:     %17 = llvm.getelementptr %16[%2, %2] : (!llvm.ptr<array<6 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %18 = llvm.call @fprintf(%13, %15, %17) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %19 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %20 = llvm.mlir.addressof @str6 : !llvm.ptr<array<22 x i8>>
// CHECK-NEXT:     %21 = llvm.getelementptr %20[%2, %2] : (!llvm.ptr<array<22 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %22 = llvm.call @fprintf(%19, %21) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%23: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %24 = cmpi "slt", %23, %arg1 : i32
// CHECK-NEXT:     cond_br %24, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %25 = muli %11, %arg1 : i32
// CHECK-NEXT:     %26 = addi %25, %23 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %27 = remi_signed %26, %c20_i32 : i32
// CHECK-NEXT:     %28 = cmpi "eq", %27, %c0_i32 : i32
// CHECK-NEXT:     scf.if %28 {
// CHECK-NEXT:       %43 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:       %44 = llvm.mlir.addressof @str3 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:       %45 = llvm.getelementptr %44[%2, %2] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %46 = llvm.call @fprintf(%43, %45) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %29 = llvm.mlir.addressof @stderr : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:     %30 = llvm.mlir.addressof @str4 : !llvm.ptr<array<6 x i8>>
// CHECK-NEXT:     %31 = llvm.getelementptr %30[%2, %2] : (!llvm.ptr<array<6 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %32 = index_cast %11 : i32 to index
// CHECK-NEXT:     %33 = addi %c0, %32 : index
// CHECK-NEXT:     %34 = memref_cast %arg2 : memref<4096x2160xf32> to memref<?x2160xf32>
// CHECK-NEXT:     %35 = index_cast %23 : i32 to index
// CHECK-NEXT:     %36 = addi %c0, %35 : index
// CHECK-NEXT:     %37 = load %34[%33, %36] : memref<?x2160xf32>
// CHECK-NEXT:     %38 = fpext %37 : f32 to f64
// CHECK-NEXT:     %39 = llvm.mlir.cast %38 : f64 to !llvm.double
// CHECK-NEXT:     %40 = llvm.call @fprintf(%29, %31, %39) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", (ptr<struct<"struct._IO_marker">>, ptr<struct<"struct._IO_FILE">>, i32, array<4 x i8>)>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %41 = addi %23, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%41 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %42 = addi %11, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%42 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }
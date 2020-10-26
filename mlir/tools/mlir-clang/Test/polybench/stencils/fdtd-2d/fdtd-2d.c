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
/* fdtd-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "fdtd-2d.h"


/* Array initialization. */
static
void init_array (int tmax,
		 int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int i, j;

  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
	ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
	ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
	hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
    }
  POLYBENCH_DUMP_END("hz");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_fdtd_2d(int tmax,
		    int nx,
		    int ny,
		    DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int t, i, j;

#pragma scop

  for(t = 0; t < _PB_TMAX; t++)
    {
      for (j = 0; j < _PB_NY; j++)
	ey[0][j] = _fict_[t];
      for (i = 1; i < _PB_NX; i++)
	for (j = 0; j < _PB_NY; j++)
	  ey[i][j] = ey[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < _PB_NX; i++)
	for (j = 1; j < _PB_NY; j++)
	  ex[i][j] = ex[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < _PB_NX - 1; i++)
	for (j = 0; j < _PB_NY - 1; j++)
	  hz[i][j] = hz[i][j] - SCALAR_VAL(0.7)*  (ex[i][j+1] - ex[i][j] +
				       ey[i+1][j] - ey[i][j]);
    }

#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);

  /* Initialize array(s). */
  init_array (tmax, nx, ny,
	      POLYBENCH_ARRAY(ex),
	      POLYBENCH_ARRAY(ey),
	      POLYBENCH_ARRAY(hz),
	      POLYBENCH_ARRAY(_fict_));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_fdtd_2d (tmax, nx, ny,
		  POLYBENCH_ARRAY(ex),
		  POLYBENCH_ARRAY(ey),
		  POLYBENCH_ARRAY(hz),
		  POLYBENCH_ARRAY(_fict_));


  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(ex),
				    POLYBENCH_ARRAY(ey),
				    POLYBENCH_ARRAY(hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c500_i32 = constant 500 : i32
// CHECK-NEXT:     %c1000_i32 = constant 1000 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c1000_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = addi %c1200_i32, %c0_i32 : i32
// CHECK-NEXT:     %2 = muli %0, %1 : i32
// CHECK-NEXT:     %3 = zexti %2 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %4 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %5 = call @polybench_alloc_data(%3, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %6 = memref_cast %5 : memref<?xi8> to memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %7 = call @polybench_alloc_data(%3, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %8 = memref_cast %7 : memref<?xi8> to memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %9 = call @polybench_alloc_data(%3, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %10 = memref_cast %9 : memref<?xi8> to memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %11 = addi %c500_i32, %c0_i32 : i32
// CHECK-NEXT:     %12 = zexti %11 : i32 to i64
// CHECK-NEXT:     %13 = call @polybench_alloc_data(%12, %4) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?xi8> to memref<?xmemref<500xf64>>
// CHECK-NEXT:     %15 = load %6[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %18 = load %8[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %21 = load %10[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %23 = memref_cast %22 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %24 = load %14[%c0] : memref<?xmemref<500xf64>>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<500xf64> to memref<?xf64>
// CHECK-NEXT:     %26 = memref_cast %25 : memref<?xf64> to memref<500xf64>
// CHECK-NEXT:     call @init_array(%c500_i32, %c1000_i32, %c1200_i32, %17, %20, %23, %26) : (i32, i32, i32, memref<1000x1200xf64>, memref<1000x1200xf64>, memref<1000x1200xf64>, memref<500xf64>) -> ()
// CHECK-NEXT:     %27 = load %6[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %28 = memref_cast %27 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %29 = memref_cast %28 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %30 = load %8[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %31 = memref_cast %30 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %32 = memref_cast %31 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %33 = load %10[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:     %34 = memref_cast %33 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %35 = memref_cast %34 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:     %36 = load %14[%c0] : memref<?xmemref<500xf64>>
// CHECK-NEXT:     %37 = memref_cast %36 : memref<500xf64> to memref<?xf64>
// CHECK-NEXT:     %38 = memref_cast %37 : memref<?xf64> to memref<500xf64>
// CHECK-NEXT:     call @kernel_fdtd_2d(%c500_i32, %c1000_i32, %c1200_i32, %29, %32, %35, %38) : (i32, i32, i32, memref<1000x1200xf64>, memref<1000x1200xf64>, memref<1000x1200xf64>, memref<500xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %39 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %40 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %41 = addi %c0, %40 : index
// CHECK-NEXT:     %42 = load %arg1[%41] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %43 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %44 = call @strcmp(%42, %43) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %45 = trunci %44 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %46 = xor %45, %true : i1
// CHECK-NEXT:     %47 = and %39, %46 : i1
// CHECK-NEXT:     scf.if %47 {
// CHECK-NEXT:       %52 = load %6[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:       %53 = memref_cast %52 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %54 = memref_cast %53 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:       %55 = load %8[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:       %56 = memref_cast %55 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %57 = memref_cast %56 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:       %58 = load %10[%c0] : memref<?xmemref<1000x1200xf64>>
// CHECK-NEXT:       %59 = memref_cast %58 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:       %60 = memref_cast %59 : memref<?x1200xf64> to memref<1000x1200xf64>
// CHECK-NEXT:       call @print_array(%c1000_i32, %c1200_i32, %54, %57, %60) : (i32, i32, memref<1000x1200xf64>, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %48 = memref_cast %6 : memref<?xmemref<1000x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%48) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %49 = memref_cast %8 : memref<?xmemref<1000x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%49) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %50 = memref_cast %10 : memref<?xmemref<1000x1200xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%50) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %51 = memref_cast %14 : memref<?xmemref<500xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%51) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1200xf64>, %arg6: memref<500xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %2 = index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = addi %c0, %2 : index
// CHECK-NEXT:     %4 = sitofp %0 : i32 to f64
// CHECK-NEXT:     store %4, %arg6[%3] : memref<500xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %5 = addi %0, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%5 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%6: i32):  // 2 preds: ^bb3, ^bb9
// CHECK-NEXT:     %7 = cmpi "slt", %6, %arg1 : i32
// CHECK-NEXT:     cond_br %7, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb7(%8: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %9 = cmpi "slt", %8, %arg2 : i32
// CHECK-NEXT:     cond_br %9, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %10 = index_cast %6 : i32 to index
// CHECK-NEXT:     %11 = addi %c0, %10 : index
// CHECK-NEXT:     %12 = memref_cast %arg3 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %13 = index_cast %8 : i32 to index
// CHECK-NEXT:     %14 = addi %c0, %13 : index
// CHECK-NEXT:     %15 = sitofp %6 : i32 to f64
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %16 = addi %8, %c1_i32_0 : i32
// CHECK-NEXT:     %17 = sitofp %16 : i32 to f64
// CHECK-NEXT:     %18 = mulf %15, %17 : f64
// CHECK-NEXT:     %19 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %20 = divf %18, %19 : f64
// CHECK-NEXT:     store %20, %12[%11, %14] : memref<?x1200xf64>
// CHECK-NEXT:     %21 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %22 = addi %8, %c2_i32 : i32
// CHECK-NEXT:     %23 = sitofp %22 : i32 to f64
// CHECK-NEXT:     %24 = mulf %15, %23 : f64
// CHECK-NEXT:     %25 = sitofp %arg2 : i32 to f64
// CHECK-NEXT:     %26 = divf %24, %25 : f64
// CHECK-NEXT:     store %26, %21[%11, %14] : memref<?x1200xf64>
// CHECK-NEXT:     %27 = memref_cast %arg5 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %c3_i32 = constant 3 : i32
// CHECK-NEXT:     %28 = addi %8, %c3_i32 : i32
// CHECK-NEXT:     %29 = sitofp %28 : i32 to f64
// CHECK-NEXT:     %30 = mulf %15, %29 : f64
// CHECK-NEXT:     %31 = divf %30, %19 : f64
// CHECK-NEXT:     store %31, %27[%11, %14] : memref<?x1200xf64>
// CHECK-NEXT:     br ^bb7(%16 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %c1_i32_1 = constant 1 : i32
// CHECK-NEXT:     %32 = addi %6, %c1_i32_1 : i32
// CHECK-NEXT:     br ^bb4(%32 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1200xf64>, %arg6: memref<500xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb21
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg2 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = index_cast %0 : i32 to index
// CHECK-NEXT:     %10 = addi %c0, %9 : index
// CHECK-NEXT:     %11 = load %arg6[%10] : memref<500xf64>
// CHECK-NEXT:     store %11, %6[%5, %8] : memref<?x1200xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %12 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%12 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     br ^bb7(%c1_i32_0, %2 : i32, i32)
// CHECK-NEXT:   ^bb7(%13: i32, %14: i32):  // 2 preds: ^bb6, ^bb12
// CHECK-NEXT:     %15 = cmpi "slt", %13, %arg1 : i32
// CHECK-NEXT:     cond_br %15, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%16: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %17 = cmpi "slt", %16, %arg2 : i32
// CHECK-NEXT:     cond_br %17, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %18 = index_cast %13 : i32 to index
// CHECK-NEXT:     %19 = addi %c0, %18 : index
// CHECK-NEXT:     %20 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %21 = index_cast %16 : i32 to index
// CHECK-NEXT:     %22 = addi %c0, %21 : index
// CHECK-NEXT:     %23 = load %20[%19, %22] : memref<?x1200xf64>
// CHECK-NEXT:     %cst = constant 5.000000e-01 : f64
// CHECK-NEXT:     %24 = memref_cast %arg5 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %25 = load %24[%19, %22] : memref<?x1200xf64>
// CHECK-NEXT:     %26 = subi %13, %c1_i32_0 : i32
// CHECK-NEXT:     %27 = index_cast %26 : i32 to index
// CHECK-NEXT:     %28 = addi %c0, %27 : index
// CHECK-NEXT:     %29 = load %24[%28, %22] : memref<?x1200xf64>
// CHECK-NEXT:     %30 = subf %25, %29 : f64
// CHECK-NEXT:     %31 = mulf %cst, %30 : f64
// CHECK-NEXT:     %32 = subf %23, %31 : f64
// CHECK-NEXT:     store %32, %20[%19, %22] : memref<?x1200xf64>
// CHECK-NEXT:     %33 = addi %16, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb10(%33 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %34 = addi %13, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb7(%34, %16 : i32, i32)
// CHECK-NEXT:   ^bb13(%35: i32):  // 2 preds: ^bb9, ^bb18
// CHECK-NEXT:     %36 = cmpi "slt", %35, %arg1 : i32
// CHECK-NEXT:     cond_br %36, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%c1_i32_0 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     br ^bb19(%c0_i32 : i32)
// CHECK-NEXT:   ^bb16(%37: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %38 = cmpi "slt", %37, %arg2 : i32
// CHECK-NEXT:     cond_br %38, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %39 = index_cast %35 : i32 to index
// CHECK-NEXT:     %40 = addi %c0, %39 : index
// CHECK-NEXT:     %41 = memref_cast %arg3 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %42 = index_cast %37 : i32 to index
// CHECK-NEXT:     %43 = addi %c0, %42 : index
// CHECK-NEXT:     %44 = load %41[%40, %43] : memref<?x1200xf64>
// CHECK-NEXT:     %cst_1 = constant 5.000000e-01 : f64
// CHECK-NEXT:     %45 = memref_cast %arg5 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %46 = load %45[%40, %43] : memref<?x1200xf64>
// CHECK-NEXT:     %47 = subi %37, %c1_i32_0 : i32
// CHECK-NEXT:     %48 = index_cast %47 : i32 to index
// CHECK-NEXT:     %49 = addi %c0, %48 : index
// CHECK-NEXT:     %50 = load %45[%40, %49] : memref<?x1200xf64>
// CHECK-NEXT:     %51 = subf %46, %50 : f64
// CHECK-NEXT:     %52 = mulf %cst_1, %51 : f64
// CHECK-NEXT:     %53 = subf %44, %52 : f64
// CHECK-NEXT:     store %53, %41[%40, %43] : memref<?x1200xf64>
// CHECK-NEXT:     %54 = addi %37, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb16(%54 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %55 = addi %35, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb13(%55 : i32)
// CHECK-NEXT:   ^bb19(%56: i32):  // 2 preds: ^bb15, ^bb24
// CHECK-NEXT:     %57 = subi %arg1, %c1_i32_0 : i32
// CHECK-NEXT:     %58 = cmpi "slt", %56, %57 : i32
// CHECK-NEXT:     cond_br %58, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     br ^bb22(%c0_i32 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     %59 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%59 : i32)
// CHECK-NEXT:   ^bb22(%60: i32):  // 2 preds: ^bb20, ^bb23
// CHECK-NEXT:     %61 = subi %arg2, %c1_i32_0 : i32
// CHECK-NEXT:     %62 = cmpi "slt", %60, %61 : i32
// CHECK-NEXT:     cond_br %62, ^bb23, ^bb24
// CHECK-NEXT:   ^bb23:  // pred: ^bb22
// CHECK-NEXT:     %63 = index_cast %56 : i32 to index
// CHECK-NEXT:     %64 = addi %c0, %63 : index
// CHECK-NEXT:     %65 = memref_cast %arg5 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %66 = index_cast %60 : i32 to index
// CHECK-NEXT:     %67 = addi %c0, %66 : index
// CHECK-NEXT:     %68 = load %65[%64, %67] : memref<?x1200xf64>
// CHECK-NEXT:     %cst_2 = constant 0.69999999999999996 : f64
// CHECK-NEXT:     %69 = memref_cast %arg3 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %70 = addi %60, %c1_i32_0 : i32
// CHECK-NEXT:     %71 = index_cast %70 : i32 to index
// CHECK-NEXT:     %72 = addi %c0, %71 : index
// CHECK-NEXT:     %73 = load %69[%64, %72] : memref<?x1200xf64>
// CHECK-NEXT:     %74 = load %69[%64, %67] : memref<?x1200xf64>
// CHECK-NEXT:     %75 = subf %73, %74 : f64
// CHECK-NEXT:     %76 = addi %56, %c1_i32_0 : i32
// CHECK-NEXT:     %77 = index_cast %76 : i32 to index
// CHECK-NEXT:     %78 = addi %c0, %77 : index
// CHECK-NEXT:     %79 = memref_cast %arg4 : memref<1000x1200xf64> to memref<?x1200xf64>
// CHECK-NEXT:     %80 = load %79[%78, %67] : memref<?x1200xf64>
// CHECK-NEXT:     %81 = addf %75, %80 : f64
// CHECK-NEXT:     %82 = load %79[%64, %67] : memref<?x1200xf64>
// CHECK-NEXT:     %83 = subf %81, %82 : f64
// CHECK-NEXT:     %84 = mulf %cst_2, %83 : f64
// CHECK-NEXT:     %85 = subf %68, %84 : f64
// CHECK-NEXT:     store %85, %65[%64, %67] : memref<?x1200xf64>
// CHECK-NEXT:     br ^bb22(%70 : i32)
// CHECK-NEXT:   ^bb24:  // pred: ^bb22
// CHECK-NEXT:     %86 = addi %56, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb19(%86 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: i32, %arg2: memref<1000x1200xf64>, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     br ^bb7(%c0_i32 : i32)
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg1 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = muli %0, %arg0 : i32
// CHECK-NEXT:     %5 = addi %4, %2 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %6 = remi_signed %5, %c20_i32 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %7 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%7 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %8 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%8 : i32)
// CHECK-NEXT:   ^bb7(%9: i32):  // 2 preds: ^bb3, ^bb12
// CHECK-NEXT:     %10 = cmpi "slt", %9, %arg0 : i32
// CHECK-NEXT:     cond_br %10, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     br ^bb10(%c0_i32 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     br ^bb13(%c0_i32 : i32)
// CHECK-NEXT:   ^bb10(%11: i32):  // 2 preds: ^bb8, ^bb11
// CHECK-NEXT:     %12 = cmpi "slt", %11, %arg1 : i32
// CHECK-NEXT:     cond_br %12, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %13 = muli %9, %arg0 : i32
// CHECK-NEXT:     %14 = addi %13, %11 : i32
// CHECK-NEXT:     %c20_i32_1 = constant 20 : i32
// CHECK-NEXT:     %15 = remi_signed %14, %c20_i32_1 : i32
// CHECK-NEXT:     %c1_i32_2 = constant 1 : i32
// CHECK-NEXT:     %16 = addi %11, %c1_i32_2 : i32
// CHECK-NEXT:     br ^bb10(%16 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %c1_i32_3 = constant 1 : i32
// CHECK-NEXT:     %17 = addi %9, %c1_i32_3 : i32
// CHECK-NEXT:     br ^bb7(%17 : i32)
// CHECK-NEXT:   ^bb13(%18: i32):  // 2 preds: ^bb9, ^bb18
// CHECK-NEXT:     %19 = cmpi "slt", %18, %arg0 : i32
// CHECK-NEXT:     cond_br %19, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     br ^bb16(%c0_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb16(%20: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %21 = cmpi "slt", %20, %arg1 : i32
// CHECK-NEXT:     cond_br %21, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %22 = muli %18, %arg0 : i32
// CHECK-NEXT:     %23 = addi %22, %20 : i32
// CHECK-NEXT:     %c20_i32_4 = constant 20 : i32
// CHECK-NEXT:     %24 = remi_signed %23, %c20_i32_4 : i32
// CHECK-NEXT:     %c1_i32_5 = constant 1 : i32
// CHECK-NEXT:     %25 = addi %20, %c1_i32_5 : i32
// CHECK-NEXT:     br ^bb16(%25 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %c1_i32_6 = constant 1 : i32
// CHECK-NEXT:     %26 = addi %18, %c1_i32_6 : i32
// CHECK-NEXT:     br ^bb13(%26 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }
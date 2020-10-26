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
/* adi.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	u[i][j] =  (DATA_TYPE)(i + n-j) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("u");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
    }
  POLYBENCH_DUMP_END("u");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel Computers"
 * by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
static
void kernel_adi(int tsteps, int n,
		DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
  int t, i, j;
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;

#pragma scop

  DX = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DY = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DT = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0)+mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0)+mul2;
  f = d;

 for (t=1; t<=_PB_TSTEPS; t++) {
    //Column Sweep
    for (i=1; i<_PB_N-1; i++) {
      v[0][i] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = v[0][i];
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -c / (a*p[i][j-1]+b);
        q[i][j] = (-d*u[j][i-1]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b);
      }

      v[_PB_N-1][i] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
      }
    }
    //Row Sweep
    for (i=1; i<_PB_N-1; i++) {
      u[i][0] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = u[i][0];
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -f / (d*p[i][j-1]+e);
        q[i][j] = (-a*v[i-1][j]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e);
      }
      u[i][_PB_N-1] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
      }
    }
  }
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(u));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_adi (tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(u);
  POLYBENCH_FREE_ARRAY(v);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(q);

  return 0;
}

// CHECK: module {
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1000_i32 = constant 1000 : i32
// CHECK-NEXT:     %c500_i32 = constant 500 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = addi %c1000_i32, %c0_i32 : i32
// CHECK-NEXT:     %1 = muli %0, %0 : i32
// CHECK-NEXT:     %2 = zexti %1 : i32 to i64
// CHECK-NEXT:     %c8_i64 = constant 8 : i64
// CHECK-NEXT:     %3 = trunci %c8_i64 : i64 to i32
// CHECK-NEXT:     %4 = call @polybench_alloc_data(%2, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %5 = memref_cast %4 : memref<?xi8> to memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %6 = call @polybench_alloc_data(%2, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %7 = memref_cast %6 : memref<?xi8> to memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %8 = call @polybench_alloc_data(%2, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %9 = memref_cast %8 : memref<?xi8> to memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %10 = call @polybench_alloc_data(%2, %3) : (i64, i32) -> memref<?xi8>
// CHECK-NEXT:     %11 = memref_cast %10 : memref<?xi8> to memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %12 = load %5[%c0] : memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %13 = memref_cast %12 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %14 = memref_cast %13 : memref<?x1000xf64> to memref<1000x1000xf64>
// CHECK-NEXT:     call @init_array(%c1000_i32, %14) : (i32, memref<1000x1000xf64>) -> ()
// CHECK-NEXT:     %15 = load %5[%c0] : memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %16 = memref_cast %15 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %17 = memref_cast %16 : memref<?x1000xf64> to memref<1000x1000xf64>
// CHECK-NEXT:     %18 = load %7[%c0] : memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %19 = memref_cast %18 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %20 = memref_cast %19 : memref<?x1000xf64> to memref<1000x1000xf64>
// CHECK-NEXT:     %21 = load %9[%c0] : memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %22 = memref_cast %21 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %23 = memref_cast %22 : memref<?x1000xf64> to memref<1000x1000xf64>
// CHECK-NEXT:     %24 = load %11[%c0] : memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:     %25 = memref_cast %24 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %26 = memref_cast %25 : memref<?x1000xf64> to memref<1000x1000xf64>
// CHECK-NEXT:     call @kernel_adi(%c500_i32, %c1000_i32, %17, %20, %23, %26) : (i32, i32, memref<1000x1000xf64>, memref<1000x1000xf64>, memref<1000x1000xf64>, memref<1000x1000xf64>) -> ()
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %27 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %28 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %29 = addi %c0, %28 : index
// CHECK-NEXT:     %30 = load %arg1[%29] : memref<?xmemref<?xi8>>
// CHECK-NEXT:     %cst = constant ""
// CHECK-NEXT:     %31 = memref_cast %cst : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:     %32 = call @strcmp(%30, %31) : (memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:     %33 = trunci %32 : i32 to i1
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %34 = xor %33, %true : i1
// CHECK-NEXT:     %35 = and %27, %34 : i1
// CHECK-NEXT:     scf.if %35 {
// CHECK-NEXT:       %40 = load %5[%c0] : memref<?xmemref<1000x1000xf64>>
// CHECK-NEXT:       %41 = memref_cast %40 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:       %42 = memref_cast %41 : memref<?x1000xf64> to memref<1000x1000xf64>
// CHECK-NEXT:       call @print_array(%c1000_i32, %42) : (i32, memref<1000x1000xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %36 = memref_cast %5 : memref<?xmemref<1000x1000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%36) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %37 = memref_cast %7 : memref<?xmemref<1000x1000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%37) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %38 = memref_cast %9 : memref<?xmemref<1000x1000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%38) : (memref<?xi8>) -> ()
// CHECK-NEXT:     %39 = memref_cast %11 : memref<?xmemref<1000x1000xf64>> to memref<?xi8>
// CHECK-NEXT:     call @free(%39) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func @polybench_alloc_data(i64, i32) -> memref<?xi8>
// CHECK-NEXT:   func @init_array(%arg0: i32, %arg1: memref<1000x1000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
// CHECK-NEXT:     cond_br %3, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %4 = index_cast %0 : i32 to index
// CHECK-NEXT:     %5 = addi %c0, %4 : index
// CHECK-NEXT:     %6 = memref_cast %arg1 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %7 = index_cast %2 : i32 to index
// CHECK-NEXT:     %8 = addi %c0, %7 : index
// CHECK-NEXT:     %9 = addi %0, %arg0 : i32
// CHECK-NEXT:     %10 = subi %9, %2 : i32
// CHECK-NEXT:     %11 = sitofp %10 : i32 to f64
// CHECK-NEXT:     %12 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %13 = divf %11, %12 : f64
// CHECK-NEXT:     store %13, %6[%5, %8] : memref<?x1000xf64>
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %14 = addi %2, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%14 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     %c1_i32_0 = constant 1 : i32
// CHECK-NEXT:     %15 = addi %0, %c1_i32_0 : i32
// CHECK-NEXT:     br ^bb1(%15 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1000x1000xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 1.000000e+00 : f64
// CHECK-NEXT:     %0 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %1 = divf %cst, %0 : f64
// CHECK-NEXT:     %2 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %3 = divf %cst, %2 : f64
// CHECK-NEXT:     %cst_0 = constant 2.000000e+00 : f64
// CHECK-NEXT:     %4 = mulf %cst_0, %3 : f64
// CHECK-NEXT:     %5 = mulf %1, %1 : f64
// CHECK-NEXT:     %6 = divf %4, %5 : f64
// CHECK-NEXT:     %7 = mulf %cst, %3 : f64
// CHECK-NEXT:     %8 = divf %7, %5 : f64
// CHECK-NEXT:     %9 = negf %6 : f64
// CHECK-NEXT:     %10 = divf %9, %cst_0 : f64
// CHECK-NEXT:     %11 = addf %cst, %6 : f64
// CHECK-NEXT:     %12 = negf %8 : f64
// CHECK-NEXT:     %13 = divf %12, %cst_0 : f64
// CHECK-NEXT:     %14 = addf %cst, %8 : f64
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     br ^bb1(%c1_i32 : i32)
// CHECK-NEXT:   ^bb1(%15: i32):  // 2 preds: ^bb0, ^bb15
// CHECK-NEXT:     %16 = cmpi "sle", %15, %arg0 : i32
// CHECK-NEXT:     cond_br %16, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c1_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%17: i32):  // 2 preds: ^bb2, ^bb12
// CHECK-NEXT:     %18 = subi %arg1, %c1_i32 : i32
// CHECK-NEXT:     %19 = cmpi "slt", %17, %18 : i32
// CHECK-NEXT:     cond_br %19, ^bb5, ^bb6
// CHECK-NEXT:   ^bb5:  // pred: ^bb4
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %20 = index_cast %c0_i32 : i32 to index
// CHECK-NEXT:     %21 = addi %c0, %20 : index
// CHECK-NEXT:     %22 = memref_cast %arg3 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %23 = index_cast %17 : i32 to index
// CHECK-NEXT:     %24 = addi %c0, %23 : index
// CHECK-NEXT:     store %cst, %22[%21, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %25 = memref_cast %arg4 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %cst_1 = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst_1, %25[%24, %21] : memref<?x1000xf64>
// CHECK-NEXT:     %26 = memref_cast %arg5 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %27 = load %22[%21, %24] : memref<?x1000xf64>
// CHECK-NEXT:     store %27, %26[%24, %21] : memref<?x1000xf64>
// CHECK-NEXT:     br ^bb7(%c1_i32 : i32)
// CHECK-NEXT:   ^bb6:  // pred: ^bb4
// CHECK-NEXT:     br ^bb13(%c1_i32 : i32)
// CHECK-NEXT:   ^bb7(%28: i32):  // 2 preds: ^bb5, ^bb8
// CHECK-NEXT:     %29 = cmpi "slt", %28, %18 : i32
// CHECK-NEXT:     cond_br %29, ^bb8, ^bb9
// CHECK-NEXT:   ^bb8:  // pred: ^bb7
// CHECK-NEXT:     %30 = index_cast %28 : i32 to index
// CHECK-NEXT:     %31 = addi %c0, %30 : index
// CHECK-NEXT:     %32 = negf %10 : f64
// CHECK-NEXT:     %33 = subi %28, %c1_i32 : i32
// CHECK-NEXT:     %34 = index_cast %33 : i32 to index
// CHECK-NEXT:     %35 = addi %c0, %34 : index
// CHECK-NEXT:     %36 = load %25[%24, %35] : memref<?x1000xf64>
// CHECK-NEXT:     %37 = mulf %10, %36 : f64
// CHECK-NEXT:     %38 = addf %37, %11 : f64
// CHECK-NEXT:     %39 = divf %32, %38 : f64
// CHECK-NEXT:     store %39, %25[%24, %31] : memref<?x1000xf64>
// CHECK-NEXT:     %40 = negf %13 : f64
// CHECK-NEXT:     %41 = memref_cast %arg2 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %42 = subi %17, %c1_i32 : i32
// CHECK-NEXT:     %43 = index_cast %42 : i32 to index
// CHECK-NEXT:     %44 = addi %c0, %43 : index
// CHECK-NEXT:     %45 = load %41[%31, %44] : memref<?x1000xf64>
// CHECK-NEXT:     %46 = mulf %40, %45 : f64
// CHECK-NEXT:     %47 = mulf %cst_0, %13 : f64
// CHECK-NEXT:     %48 = addf %cst, %47 : f64
// CHECK-NEXT:     %49 = load %41[%31, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %50 = mulf %48, %49 : f64
// CHECK-NEXT:     %51 = addf %46, %50 : f64
// CHECK-NEXT:     %52 = addi %17, %c1_i32 : i32
// CHECK-NEXT:     %53 = index_cast %52 : i32 to index
// CHECK-NEXT:     %54 = addi %c0, %53 : index
// CHECK-NEXT:     %55 = load %41[%31, %54] : memref<?x1000xf64>
// CHECK-NEXT:     %56 = mulf %13, %55 : f64
// CHECK-NEXT:     %57 = subf %51, %56 : f64
// CHECK-NEXT:     %58 = load %26[%24, %35] : memref<?x1000xf64>
// CHECK-NEXT:     %59 = mulf %10, %58 : f64
// CHECK-NEXT:     %60 = subf %57, %59 : f64
// CHECK-NEXT:     %61 = load %25[%24, %35] : memref<?x1000xf64>
// CHECK-NEXT:     %62 = mulf %10, %61 : f64
// CHECK-NEXT:     %63 = addf %62, %11 : f64
// CHECK-NEXT:     %64 = divf %60, %63 : f64
// CHECK-NEXT:     store %64, %26[%24, %31] : memref<?x1000xf64>
// CHECK-NEXT:     %65 = addi %28, %c1_i32 : i32
// CHECK-NEXT:     br ^bb7(%65 : i32)
// CHECK-NEXT:   ^bb9:  // pred: ^bb7
// CHECK-NEXT:     %66 = index_cast %18 : i32 to index
// CHECK-NEXT:     %67 = addi %c0, %66 : index
// CHECK-NEXT:     store %cst, %22[%67, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %68 = subi %arg1, %c2_i32 : i32
// CHECK-NEXT:     br ^bb10(%68 : i32)
// CHECK-NEXT:   ^bb10(%69: i32):  // 2 preds: ^bb9, ^bb11
// CHECK-NEXT:     %70 = cmpi "sge", %69, %c1_i32 : i32
// CHECK-NEXT:     cond_br %70, ^bb11, ^bb12
// CHECK-NEXT:   ^bb11:  // pred: ^bb10
// CHECK-NEXT:     %71 = index_cast %69 : i32 to index
// CHECK-NEXT:     %72 = addi %c0, %71 : index
// CHECK-NEXT:     %73 = load %25[%24, %72] : memref<?x1000xf64>
// CHECK-NEXT:     %74 = addi %69, %c1_i32 : i32
// CHECK-NEXT:     %75 = index_cast %74 : i32 to index
// CHECK-NEXT:     %76 = addi %c0, %75 : index
// CHECK-NEXT:     %77 = load %22[%76, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %78 = mulf %73, %77 : f64
// CHECK-NEXT:     %79 = load %26[%24, %72] : memref<?x1000xf64>
// CHECK-NEXT:     %80 = addf %78, %79 : f64
// CHECK-NEXT:     store %80, %22[%72, %24] : memref<?x1000xf64>
// CHECK-NEXT:     %81 = subi %69, %c1_i32 : i32
// CHECK-NEXT:     br ^bb10(%81 : i32)
// CHECK-NEXT:   ^bb12:  // pred: ^bb10
// CHECK-NEXT:     %82 = addi %17, %c1_i32 : i32
// CHECK-NEXT:     br ^bb4(%82 : i32)
// CHECK-NEXT:   ^bb13(%83: i32):  // 2 preds: ^bb6, ^bb21
// CHECK-NEXT:     %84 = cmpi "slt", %83, %18 : i32
// CHECK-NEXT:     cond_br %84, ^bb14, ^bb15
// CHECK-NEXT:   ^bb14:  // pred: ^bb13
// CHECK-NEXT:     %85 = index_cast %83 : i32 to index
// CHECK-NEXT:     %86 = addi %c0, %85 : index
// CHECK-NEXT:     %87 = memref_cast %arg2 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %c0_i32_2 = constant 0 : i32
// CHECK-NEXT:     %88 = index_cast %c0_i32_2 : i32 to index
// CHECK-NEXT:     %89 = addi %c0, %88 : index
// CHECK-NEXT:     store %cst, %87[%86, %89] : memref<?x1000xf64>
// CHECK-NEXT:     %90 = memref_cast %arg4 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %cst_3 = constant 0.000000e+00 : f64
// CHECK-NEXT:     store %cst_3, %90[%86, %89] : memref<?x1000xf64>
// CHECK-NEXT:     %91 = memref_cast %arg5 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %92 = load %87[%86, %89] : memref<?x1000xf64>
// CHECK-NEXT:     store %92, %91[%86, %89] : memref<?x1000xf64>
// CHECK-NEXT:     br ^bb16(%c1_i32 : i32)
// CHECK-NEXT:   ^bb15:  // pred: ^bb13
// CHECK-NEXT:     %93 = addi %15, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%93 : i32)
// CHECK-NEXT:   ^bb16(%94: i32):  // 2 preds: ^bb14, ^bb17
// CHECK-NEXT:     %95 = cmpi "slt", %94, %18 : i32
// CHECK-NEXT:     cond_br %95, ^bb17, ^bb18
// CHECK-NEXT:   ^bb17:  // pred: ^bb16
// CHECK-NEXT:     %96 = index_cast %94 : i32 to index
// CHECK-NEXT:     %97 = addi %c0, %96 : index
// CHECK-NEXT:     %98 = negf %13 : f64
// CHECK-NEXT:     %99 = subi %94, %c1_i32 : i32
// CHECK-NEXT:     %100 = index_cast %99 : i32 to index
// CHECK-NEXT:     %101 = addi %c0, %100 : index
// CHECK-NEXT:     %102 = load %90[%86, %101] : memref<?x1000xf64>
// CHECK-NEXT:     %103 = mulf %13, %102 : f64
// CHECK-NEXT:     %104 = addf %103, %14 : f64
// CHECK-NEXT:     %105 = divf %98, %104 : f64
// CHECK-NEXT:     store %105, %90[%86, %97] : memref<?x1000xf64>
// CHECK-NEXT:     %106 = negf %10 : f64
// CHECK-NEXT:     %107 = subi %83, %c1_i32 : i32
// CHECK-NEXT:     %108 = index_cast %107 : i32 to index
// CHECK-NEXT:     %109 = addi %c0, %108 : index
// CHECK-NEXT:     %110 = memref_cast %arg3 : memref<1000x1000xf64> to memref<?x1000xf64>
// CHECK-NEXT:     %111 = load %110[%109, %97] : memref<?x1000xf64>
// CHECK-NEXT:     %112 = mulf %106, %111 : f64
// CHECK-NEXT:     %113 = mulf %cst_0, %10 : f64
// CHECK-NEXT:     %114 = addf %cst, %113 : f64
// CHECK-NEXT:     %115 = load %110[%86, %97] : memref<?x1000xf64>
// CHECK-NEXT:     %116 = mulf %114, %115 : f64
// CHECK-NEXT:     %117 = addf %112, %116 : f64
// CHECK-NEXT:     %118 = addi %83, %c1_i32 : i32
// CHECK-NEXT:     %119 = index_cast %118 : i32 to index
// CHECK-NEXT:     %120 = addi %c0, %119 : index
// CHECK-NEXT:     %121 = load %110[%120, %97] : memref<?x1000xf64>
// CHECK-NEXT:     %122 = mulf %10, %121 : f64
// CHECK-NEXT:     %123 = subf %117, %122 : f64
// CHECK-NEXT:     %124 = load %91[%86, %101] : memref<?x1000xf64>
// CHECK-NEXT:     %125 = mulf %13, %124 : f64
// CHECK-NEXT:     %126 = subf %123, %125 : f64
// CHECK-NEXT:     %127 = load %90[%86, %101] : memref<?x1000xf64>
// CHECK-NEXT:     %128 = mulf %13, %127 : f64
// CHECK-NEXT:     %129 = addf %128, %14 : f64
// CHECK-NEXT:     %130 = divf %126, %129 : f64
// CHECK-NEXT:     store %130, %91[%86, %97] : memref<?x1000xf64>
// CHECK-NEXT:     %131 = addi %94, %c1_i32 : i32
// CHECK-NEXT:     br ^bb16(%131 : i32)
// CHECK-NEXT:   ^bb18:  // pred: ^bb16
// CHECK-NEXT:     %132 = index_cast %18 : i32 to index
// CHECK-NEXT:     %133 = addi %c0, %132 : index
// CHECK-NEXT:     store %cst, %87[%86, %133] : memref<?x1000xf64>
// CHECK-NEXT:     %c2_i32_4 = constant 2 : i32
// CHECK-NEXT:     %134 = subi %arg1, %c2_i32_4 : i32
// CHECK-NEXT:     br ^bb19(%134 : i32)
// CHECK-NEXT:   ^bb19(%135: i32):  // 2 preds: ^bb18, ^bb20
// CHECK-NEXT:     %136 = cmpi "sge", %135, %c1_i32 : i32
// CHECK-NEXT:     cond_br %136, ^bb20, ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     %137 = index_cast %135 : i32 to index
// CHECK-NEXT:     %138 = addi %c0, %137 : index
// CHECK-NEXT:     %139 = load %90[%86, %138] : memref<?x1000xf64>
// CHECK-NEXT:     %140 = addi %135, %c1_i32 : i32
// CHECK-NEXT:     %141 = index_cast %140 : i32 to index
// CHECK-NEXT:     %142 = addi %c0, %141 : index
// CHECK-NEXT:     %143 = load %87[%86, %142] : memref<?x1000xf64>
// CHECK-NEXT:     %144 = mulf %139, %143 : f64
// CHECK-NEXT:     %145 = load %91[%86, %138] : memref<?x1000xf64>
// CHECK-NEXT:     %146 = addf %144, %145 : f64
// CHECK-NEXT:     store %146, %87[%86, %138] : memref<?x1000xf64>
// CHECK-NEXT:     %147 = subi %135, %c1_i32 : i32
// CHECK-NEXT:     br ^bb19(%147 : i32)
// CHECK-NEXT:   ^bb21:  // pred: ^bb19
// CHECK-NEXT:     %148 = addi %83, %c1_i32 : i32
// CHECK-NEXT:     br ^bb13(%148 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func @strcmp(memref<?xi8>, memref<?xi8>) -> i32
// CHECK-NEXT:   func @print_array(%arg0: i32, %arg1: memref<1000x1000xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %1 = cmpi "slt", %0, %arg0 : i32
// CHECK-NEXT:     cond_br %1, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     br ^bb4(%c0_i32 : i32)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb4(%2: i32):  // 2 preds: ^bb2, ^bb5
// CHECK-NEXT:     %3 = cmpi "slt", %2, %arg0 : i32
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
// CHECK-NEXT:   }
// CHECK-NEXT:   func @free(memref<?xi8>)
// CHECK-NEXT: }
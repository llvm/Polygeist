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

// CHECK: #map0 = affine_map<()[s0] -> (s0 + 1)>
// CHECK: #map1 = affine_map<()[s0] -> (s0 - 1)>
// CHECK:   func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1000x1000xf64>) {
// CHECK-NEXT:  %c0 = constant 0 : index
// CHECK-NEXT:  %cst = constant 1.000000e+00 : f64
// CHECK-NEXT:  %cst_0 = constant 2.000000e+00 : f64
// CHECK-NEXT:  %cst_1 = constant 0.000000e+00 : f64
// CHECK-NEXT:  %c2 = constant 2 : index
// CHECK-NEXT:  %c1 = constant 1 : index
// CHECK-NEXT:  %0 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:  %1 = divf %cst, %0 : f64
// CHECK-NEXT:  %2 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:  %3 = divf %cst, %2 : f64
// CHECK-NEXT:  %4 = mulf %cst_0, %3 : f64
// CHECK-NEXT:  %5 = mulf %1, %1 : f64
// CHECK-NEXT:  %6 = divf %4, %5 : f64
// CHECK-NEXT:  %7 = mulf %cst, %3 : f64
// CHECK-NEXT:  %8 = divf %7, %5 : f64
// CHECK-NEXT:  %9 = negf %6 : f64
// CHECK-NEXT:  %10 = divf %9, %cst_0 : f64
// CHECK-NEXT:  %11 = addf %cst, %6 : f64
// CHECK-NEXT:  %12 = negf %8 : f64
// CHECK-NEXT:  %13 = divf %12, %cst_0 : f64
// CHECK-NEXT:  %14 = addf %cst, %8 : f64
// CHECK-NEXT:  %15 = index_cast %arg0 : i32 to index
// CHECK-NEXT:  %16 = index_cast %arg1 : i32 to index
// CHECK-NEXT:  %17 = subi %16, %c1 : index
// CHECK-NEXT:  %18 = negf %10 : f64
// CHECK-NEXT:  %19 = negf %13 : f64
// CHECK-NEXT:  %20 = mulf %cst_0, %13 : f64
// CHECK-NEXT:  %21 = addf %cst, %20 : f64
// CHECK-NEXT:  %22 = index_cast %arg1 : i32 to index
// CHECK-NEXT:  %23 = subi %22, %c2 : index
// CHECK-NEXT:  %24 = addi %23, %c1 : index
// CHECK-NEXT:  %25 = subi %24, %c1 : index
// CHECK-NEXT:  %26 = negf %13 : f64
// CHECK-NEXT:  %27 = negf %10 : f64
// CHECK-NEXT:  %28 = mulf %cst_0, %10 : f64
// CHECK-NEXT:  %29 = addf %cst, %28 : f64
// CHECK-NEXT:  %30 = index_cast %arg1 : i32 to index
// CHECK-NEXT:  %31 = subi %30, %c2 : index
// CHECK-NEXT:  %32 = addi %31, %c1 : index
// CHECK-NEXT:  %33 = subi %32, %c1 : index
// CHECK-NEXT:  affine.for %arg6 = 1 to #map0()[%15] {
// CHECK-NEXT:    affine.for %arg7 = 1 to #map1()[%16] {
// CHECK-NEXT:      affine.store %cst, %arg3[%c0, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:      affine.store %cst_1, %arg4[%arg7, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:      %34 = affine.load %arg3[%c0, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:      affine.store %34, %arg5[%arg7, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:      %35 = subi %arg7, %c1 : index
// CHECK-NEXT:      %36 = addi %arg7, %c1 : index
// CHECK-NEXT:      affine.for %arg8 = 1 to #map1()[%16] {
// CHECK-NEXT:        %37 = subi %arg8, %c1 : index
// CHECK-NEXT:        %38 = load %arg4[%arg7, %37] : memref<1000x1000xf64>
// CHECK-NEXT:        %39 = mulf %10, %38 : f64
// CHECK-NEXT:        %40 = addf %39, %11 : f64
// CHECK-NEXT:        %41 = divf %18, %40 : f64
// CHECK-NEXT:        affine.store %41, %arg4[%arg7, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:        %42 = load %arg2[%arg8, %35] : memref<1000x1000xf64>
// CHECK-NEXT:        %43 = mulf %19, %42 : f64
// CHECK-NEXT:        %44 = affine.load %arg2[%arg8, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:        %45 = mulf %21, %44 : f64
// CHECK-NEXT:        %46 = addf %43, %45 : f64
// CHECK-NEXT:        %47 = load %arg2[%arg8, %36] : memref<1000x1000xf64>
// CHECK-NEXT:        %48 = mulf %13, %47 : f64
// CHECK-NEXT:        %49 = subf %46, %48 : f64
// CHECK-NEXT:        %50 = load %arg5[%arg7, %37] : memref<1000x1000xf64>
// CHECK-NEXT:        %51 = mulf %10, %50 : f64
// CHECK-NEXT:        %52 = subf %49, %51 : f64
// CHECK-NEXT:        %53 = load %arg4[%arg7, %37] : memref<1000x1000xf64>
// CHECK-NEXT:        %54 = mulf %10, %53 : f64
// CHECK-NEXT:        %55 = addf %54, %11 : f64
// CHECK-NEXT:        %56 = divf %52, %55 : f64
// CHECK-NEXT:        affine.store %56, %arg5[%arg7, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      store %cst, %arg3[%17, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:      affine.for %arg8 = 1 to #map1()[%22] {
// CHECK-NEXT:        %37 = subi %arg8, %c1 : index
// CHECK-NEXT:        %38 = subi %25, %37 : index
// CHECK-NEXT:        %39 = load %arg4[%arg7, %38] : memref<1000x1000xf64>
// CHECK-NEXT:        %40 = addi %38, %c1 : index
// CHECK-NEXT:        %41 = load %arg3[%40, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:        %42 = mulf %39, %41 : f64
// CHECK-NEXT:        %43 = load %arg5[%arg7, %38] : memref<1000x1000xf64>
// CHECK-NEXT:        %44 = addf %42, %43 : f64
// CHECK-NEXT:        store %44, %arg3[%38, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.for %arg7 = 1 to #map1()[%16] {
// CHECK-NEXT:      affine.store %cst, %arg2[%arg7, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:      affine.store %cst_1, %arg4[%arg7, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:      %34 = affine.load %arg2[%arg7, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:      affine.store %34, %arg5[%arg7, %c0] : memref<1000x1000xf64>
// CHECK-NEXT:      %35 = subi %arg7, %c1 : index
// CHECK-NEXT:      %36 = addi %arg7, %c1 : index
// CHECK-NEXT:      affine.for %arg8 = 1 to #map1()[%16] {
// CHECK-NEXT:        %37 = subi %arg8, %c1 : index
// CHECK-NEXT:        %38 = load %arg4[%arg7, %37] : memref<1000x1000xf64>
// CHECK-NEXT:        %39 = mulf %13, %38 : f64
// CHECK-NEXT:        %40 = addf %39, %14 : f64
// CHECK-NEXT:        %41 = divf %26, %40 : f64
// CHECK-NEXT:        affine.store %41, %arg4[%arg7, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:        %42 = load %arg3[%35, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:        %43 = mulf %27, %42 : f64
// CHECK-NEXT:        %44 = affine.load %arg3[%arg7, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:        %45 = mulf %29, %44 : f64
// CHECK-NEXT:        %46 = addf %43, %45 : f64
// CHECK-NEXT:        %47 = load %arg3[%36, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:        %48 = mulf %10, %47 : f64
// CHECK-NEXT:        %49 = subf %46, %48 : f64
// CHECK-NEXT:        %50 = load %arg5[%arg7, %37] : memref<1000x1000xf64>
// CHECK-NEXT:        %51 = mulf %13, %50 : f64
// CHECK-NEXT:        %52 = subf %49, %51 : f64
// CHECK-NEXT:        %53 = load %arg4[%arg7, %37] : memref<1000x1000xf64>
// CHECK-NEXT:        %54 = mulf %13, %53 : f64
// CHECK-NEXT:        %55 = addf %54, %14 : f64
// CHECK-NEXT:        %56 = divf %52, %55 : f64
// CHECK-NEXT:        affine.store %56, %arg5[%arg7, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      store %cst, %arg2[%arg7, %17] : memref<1000x1000xf64>
// CHECK-NEXT:      affine.for %arg8 = 1 to #map1()[%30] {
// CHECK-NEXT:        %37 = subi %arg8, %c1 : index
// CHECK-NEXT:        %38 = subi %33, %37 : index
// CHECK-NEXT:        %39 = load %arg4[%arg7, %38] : memref<1000x1000xf64>
// CHECK-NEXT:        %40 = addi %38, %c1 : index
// CHECK-NEXT:        %41 = load %arg2[%arg7, %40] : memref<1000x1000xf64>
// CHECK-NEXT:        %42 = mulf %39, %41 : f64
// CHECK-NEXT:        %43 = load %arg5[%arg7, %38] : memref<1000x1000xf64>
// CHECK-NEXT:        %44 = addf %42, %43 : f64
// CHECK-NEXT:        store %44, %arg2[%arg7, %38] : memref<1000x1000xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT: }




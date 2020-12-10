#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

// TODO: mlir-clang %s %stdinclude | FileCheck %s
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

  int t1, t2, t3, t4, t5, t6;
 register int lbv, ubv;
if ((_PB_NY >= 1) && (_PB_TMAX >= 1)) {
  for (t1=ceild(-_PB_NY-30,32);t1<=floord(_PB_TMAX-1,32);t1++) {
    for (t2=max(t1,-t1-1);t2<=min(min(floord(-16*t1+_PB_TMAX-1,16),floord(16*t1+_PB_NY+14,16)),floord(_PB_TMAX+_PB_NY-2,32));t2++) {
      if (_PB_NX <= -1) {
        for (t3=max(0,ceild(t1+t2-1,2));t3<=min(floord(t1+t2+1,2),floord(_PB_TMAX-1,32));t3++) {
          for (t4=max(max(32*t3,16*t1+16*t2),32*t2-_PB_NY+1);t4<=min(min(min(_PB_TMAX-1,32*t3+31),16*t1+16*t2+31),32*t1+_PB_NY+30);t4++) {
            for (t6=max(max(32*t2,t4),-32*t1+2*t4-31);t6<=min(min(-32*t1+2*t4,32*t2+31),t4+_PB_NY-1);t6++) {
              ey[0][(-t4+t6)] = _fict_[t4];;
            }
          }
        }
      }
      if (_PB_NX >= 0) {
        for (t3=max(0,ceild(t1+t2-1,2));t3<=min(min(min(floord(_PB_TMAX+_PB_NX-1,32),floord(16*t1+16*t2+_PB_NX+30,32)),floord(16*t1+16*t2+14*_PB_TMAX+15*_PB_NX+2,480)),floord(464*t1+16*t2+14*_PB_NY+15*_PB_NX+436,480));t3++) {
          if ((_PB_NX >= 2) && (_PB_NY >= 2) && (t1 == t2) && (t1 == t3)) {
            ey[0][0] = _fict_[32*t1];;
            for (t5=32*t1+1;t5<=min(32*t1+31,32*t1+_PB_NX-1);t5++) {
              ey[(-32*t1+t5)][0] = ey[(-32*t1+t5)][0] - SCALAR_VAL(0.5)*(hz[(-32*t1+t5)][0]-hz[(-32*t1+t5)-1][0]);;
              hz[(-32*t1+t5-1)][0] = hz[(-32*t1+t5-1)][0] - SCALAR_VAL(0.7)* (ex[(-32*t1+t5-1)][0 +1] - ex[(-32*t1+t5-1)][0] + ey[(-32*t1+t5-1)+1][0] - ey[(-32*t1+t5-1)][0]);;
            }
          }
          if ((_PB_NX >= 2) && (_PB_NY == 1) && (t1 == t2) && (t1 == t3)) {
            for (t4=32*t1;t4<=min(_PB_TMAX-1,32*t1+30);t4++) {
              ey[0][0] = _fict_[t4];;
              for (t5=t4+1;t5<=min(32*t1+31,t4+_PB_NX-1);t5++) {
                ey[(-t4+t5)][0] = ey[(-t4+t5)][0] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][0]-hz[(-t4+t5)-1][0]);;
              }
            }
          }
          if ((_PB_NX <= 1) && (_PB_NY == 1) && (t1 == t2) && (t1 == t3)) {
            for (t4=32*t1;t4<=min(_PB_TMAX-1,32*t1+31);t4++) {
              ey[0][0] = _fict_[t4];;
            }
          }
          if ((_PB_NX == 0) && (_PB_NY >= 2)) {
            for (t4=max(max(32*t3,16*t1+16*t2),32*t2-_PB_NY+1);t4<=min(min(min(_PB_TMAX-1,32*t3+31),16*t1+16*t2+31),32*t1+_PB_NY+30);t4++) {
              for (t6=max(max(32*t2,t4),-32*t1+2*t4-31);t6<=min(min(-32*t1+2*t4,32*t2+31),t4+_PB_NY-1);t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
              }
            }
          }
          if ((_PB_NX == 1) && (_PB_NY >= 2) && (t1 == t2) && (t1 == t3)) {
            ey[0][0] = _fict_[32*t1];;
          }
          if ((_PB_NY == 1) && (t1 == t2)) {
            for (t4=max(32*t1,32*t3-_PB_NX+1);t4<=min(min(_PB_TMAX-1,32*t1+31),32*t3-1);t4++) {
              for (t5=32*t3;t5<=min(32*t3+31,t4+_PB_NX-1);t5++) {
                ey[(-t4+t5)][0] = ey[(-t4+t5)][0] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][0]-hz[(-t4+t5)-1][0]);;
              }
            }
          }
          if ((_PB_NY >= 2) && (t1 == t2) && (t1 >= ceild(32*t3-_PB_NX+1,32)) && (t1 <= t3-1)) {
            for (t5=32*t3;t5<=min(32*t3+31,32*t1+_PB_NX-1);t5++) {
              ey[(-32*t1+t5)][0] = ey[(-32*t1+t5)][0] - SCALAR_VAL(0.5)*(hz[(-32*t1+t5)][0]-hz[(-32*t1+t5)-1][0]);;
              hz[(-32*t1+t5-1)][0] = hz[(-32*t1+t5-1)][0] - SCALAR_VAL(0.7)* (ex[(-32*t1+t5-1)][0 +1] - ex[(-32*t1+t5-1)][0] + ey[(-32*t1+t5-1)+1][0] - ey[(-32*t1+t5-1)][0]);;
            }
          }
          if ((_PB_NX == 1) && (_PB_NY >= 2) && (t1 == t2) && (t1 == t3)) {
            for (t4=32*t1+1;t4<=min(_PB_TMAX-1,32*t1+30);t4++) {
              ey[0][0] = _fict_[t4];;
              for (t6=t4+1;t6<=min(min(-32*t1+2*t4,32*t1+31),t4+_PB_NY-1);t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
                ex[0][(-t4+t6)] = ex[0][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[0][(-t4+t6)]-hz[0][(-t4+t6)-1]);;
              }
            }
          }
          if ((_PB_NX == 1) && (t1 <= t2-1)) {
            for (t4=max(max(32*t3,16*t1+16*t2),32*t2-_PB_NY+1);t4<=min(min(min(_PB_TMAX-1,32*t3+31),16*t1+16*t2+31),32*t1+_PB_NY+30);t4++) {
              for (t6=max(32*t2,-32*t1+2*t4-31);t6<=min(min(-32*t1+2*t4,32*t2+31),t4+_PB_NY-1);t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
                ex[0][(-t4+t6)] = ex[0][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[0][(-t4+t6)]-hz[0][(-t4+t6)-1]);;
              }
            }
          }
          if ((_PB_NX == 1) && (_PB_NY >= 2) && (t1 == t2) && (t1 == t3) && (t1 <= floord(_PB_TMAX-32,32))) {
            ey[0][0] = _fict_[(32*t1+31)];;
          }
          if (t1 == t2) {
            for (t4=max(32*t1+1,32*t3-_PB_NX+1);t4<=min(min(min(_PB_TMAX-1,32*t1+15),32*t3-1),32*t1+_PB_NY-2);t4++) {
              for (t5=32*t3;t5<=min(32*t3+31,t4+_PB_NX-1);t5++) {
                ey[(-t4+t5)][0] = ey[(-t4+t5)][0] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][0]-hz[(-t4+t5)-1][0]);;
                for (t6=t4+1;t6<=-32*t1+2*t4;t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
                hz[(-t4+t5-1)][(-32*t1+t4)] = hz[(-t4+t5-1)][(-32*t1+t4)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-32*t1+t4)+1] - ex[(-t4+t5-1)][(-32*t1+t4)] + ey[(-t4+t5-1)+1][(-32*t1+t4)] - ey[(-t4+t5-1)][(-32*t1+t4)]);;
              }
            }
          }
          if ((_PB_NY >= 2) && (t1 == t2)) {
            for (t4=max(32*t1+_PB_NY-1,32*t3-_PB_NX+1);t4<=min(min(_PB_TMAX-1,32*t1+15),32*t3-1);t4++) {
              for (t5=32*t3;t5<=min(32*t3+31,t4+_PB_NX-1);t5++) {
                ey[(-t4+t5)][0] = ey[(-t4+t5)][0] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][0]-hz[(-t4+t5)-1][0]);;
                for (t6=t4+1;t6<=t4+_PB_NY-1;t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
              }
            }
          }
          if (t1 <= t2-1) {
            for (t4=max(max(0,16*t1+16*t2),32*t3-_PB_NX+1);t4<=min(min(min(_PB_TMAX-1,32*t3-1),16*t1+16*t2+15),32*t1+_PB_NY-2);t4++) {
              for (t5=32*t3;t5<=min(32*t3+31,t4+_PB_NX-1);t5++) {
                for (t6=32*t2;t6<=-32*t1+2*t4;t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
                hz[(-t4+t5-1)][(-32*t1+t4)] = hz[(-t4+t5-1)][(-32*t1+t4)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-32*t1+t4)+1] - ex[(-t4+t5-1)][(-32*t1+t4)] + ey[(-t4+t5-1)+1][(-32*t1+t4)] - ey[(-t4+t5-1)][(-32*t1+t4)]);;
              }
            }
          }
          if (t1 <= t2-1) {
            for (t4=max(max(max(0,32*t1+_PB_NY-1),32*t2-_PB_NY+1),32*t3-_PB_NX+1);t4<=min(min(_PB_TMAX-1,32*t3-1),16*t1+16*t2+15);t4++) {
              for (t5=32*t3;t5<=min(32*t3+31,t4+_PB_NX-1);t5++) {
                for (t6=32*t2;t6<=t4+_PB_NY-1;t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
              }
            }
          }
          for (t4=max(max(32*t1+32,16*t1+16*t2+16),32*t3-_PB_NX+1);t4<=min(min(min(_PB_TMAX-1,32*t3-1),16*t1+16*t2+30),32*t1+_PB_NY+29);t4++) {
            for (t5=32*t3;t5<=min(32*t3+31,t4+_PB_NX-1);t5++) {
              ey[(-t4+t5)][(-32*t1+t4-31)] = ey[(-t4+t5)][(-32*t1+t4-31)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-32*t1+t4-31)]-hz[(-t4+t5)-1][(-32*t1+t4-31)]);;
              ex[(-t4+t5)][(-32*t1+t4-31)] = ex[(-t4+t5)][(-32*t1+t4-31)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-32*t1+t4-31)]-hz[(-t4+t5)][(-32*t1+t4-31)-1]);;
              for (t6=-32*t1+2*t4-30;t6<=min(32*t2+31,t4+_PB_NY-1);t6++) {
                ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
              }
            }
          }
          if ((_PB_NY >= 2) && (t1 == t2)) {
            for (t4=max(32*t1+16,32*t3-_PB_NX+1);t4<=min(min(_PB_TMAX-1,32*t1+30),32*t3-1);t4++) {
              for (t5=32*t3;t5<=min(32*t3+31,t4+_PB_NX-1);t5++) {
                ey[(-t4+t5)][0] = ey[(-t4+t5)][0] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][0]-hz[(-t4+t5)-1][0]);;
                for (t6=t4+1;t6<=min(32*t1+31,t4+_PB_NY-1);t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
              }
            }
          }
          if ((t1 <= min(min(floord(-16*t2+_PB_TMAX-32,16),t2-1),-t2+2*t3-2)) && (t1 >= ceild(16*t2-_PB_NY+1,16))) {
            for (t5=32*t3;t5<=min(32*t3+31,16*t1+16*t2+_PB_NX+30);t5++) {
              ey[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)] = ey[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)] - SCALAR_VAL(0.5)*(hz[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)]-hz[(-16*t1-16*t2+t5-31)-1][(-16*t1+16*t2)]);;
              ex[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)] = ex[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)] - SCALAR_VAL(0.5)*(hz[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)]-hz[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)-1]);;
            }
          }
          if ((t1 <= min(min(floord(16*t2-_PB_NY,16),floord(_PB_TMAX-_PB_NY-31,32)),floord(32*t3-_PB_NY-31,32))) && (t1 >= ceild(32*t3-_PB_NY-_PB_NX-29,32))) {
            for (t5=32*t3;t5<=min(32*t3+31,32*t1+_PB_NY+_PB_NX+29);t5++) {
              ey[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)] = ey[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)] - SCALAR_VAL(0.5)*(hz[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)]-hz[(-32*t1+t5-_PB_NY-30)-1][(_PB_NY-1)]);;
              ex[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)] = ex[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)] - SCALAR_VAL(0.5)*(hz[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)]-hz[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)-1]);;
            }
          }
          if ((_PB_NY >= 2) && (t1 == t2) && (t1 <= min(floord(_PB_TMAX-32,32),t3-1))) {
            for (t5=32*t3;t5<=min(32*t3+31,32*t1+_PB_NX+30);t5++) {
              ey[(-32*t1+t5-31)][0] = ey[(-32*t1+t5-31)][0] - SCALAR_VAL(0.5)*(hz[(-32*t1+t5-31)][0]-hz[(-32*t1+t5-31)-1][0]);;
            }
          }
          if ((_PB_NX >= 2) && (t1 == t2) && (t1 == t3)) {
            for (t4=32*t1+1;t4<=min(min(_PB_TMAX-1,32*t1+15),32*t1+_PB_NY-2);t4++) {
              ey[0][0] = _fict_[t4];;
              for (t6=t4+1;t6<=-32*t1+2*t4;t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
                ex[0][(-t4+t6)] = ex[0][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[0][(-t4+t6)]-hz[0][(-t4+t6)-1]);;
              }
              for (t5=t4+1;t5<=min(32*t1+31,t4+_PB_NX-1);t5++) {
                ey[(-t4+t5)][0] = ey[(-t4+t5)][0] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][0]-hz[(-t4+t5)-1][0]);;
                for (t6=t4+1;t6<=-32*t1+2*t4;t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
                hz[(-t4+t5-1)][(-32*t1+t4)] = hz[(-t4+t5-1)][(-32*t1+t4)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-32*t1+t4)+1] - ex[(-t4+t5-1)][(-32*t1+t4)] + ey[(-t4+t5-1)+1][(-32*t1+t4)] - ey[(-t4+t5-1)][(-32*t1+t4)]);;
              }
            }
          }
          if ((_PB_NX >= 2) && (_PB_NY >= 2) && (t1 == t2) && (t1 == t3)) {
            for (t4=32*t1+_PB_NY-1;t4<=min(_PB_TMAX-1,32*t1+15);t4++) {
              ey[0][0] = _fict_[t4];;
              for (t6=t4+1;t6<=t4+_PB_NY-1;t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
                ex[0][(-t4+t6)] = ex[0][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[0][(-t4+t6)]-hz[0][(-t4+t6)-1]);;
              }
              for (t5=t4+1;t5<=min(32*t1+31,t4+_PB_NX-1);t5++) {
                ey[(-t4+t5)][0] = ey[(-t4+t5)][0] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][0]-hz[(-t4+t5)-1][0]);;
                for (t6=t4+1;t6<=t4+_PB_NY-1;t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
              }
            }
          }
          if (_PB_NX >= 2) {
            for (t4=max(max(32*t3,16*t1+16*t2),32*t1+32);t4<=min(min(min(_PB_TMAX-1,32*t3+30),16*t1+16*t2+15),32*t1+_PB_NY-2);t4++) {
              for (t6=32*t2;t6<=-32*t1+2*t4;t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
                ex[0][(-t4+t6)] = ex[0][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[0][(-t4+t6)]-hz[0][(-t4+t6)-1]);;
              }
              for (t5=t4+1;t5<=min(32*t3+31,t4+_PB_NX-1);t5++) {
                for (t6=32*t2;t6<=-32*t1+2*t4;t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
                hz[(-t4+t5-1)][(-32*t1+t4)] = hz[(-t4+t5-1)][(-32*t1+t4)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-32*t1+t4)+1] - ex[(-t4+t5-1)][(-32*t1+t4)] + ey[(-t4+t5-1)+1][(-32*t1+t4)] - ey[(-t4+t5-1)][(-32*t1+t4)]);;
              }
            }
          }
          if ((_PB_NX >= 2) && (t1 == t3) && (t1 == t2-1)) {
            for (t4=32*t1+16;t4<=min(min(_PB_TMAX-1,32*t1+30),32*t1+_PB_NY-2);t4++) {
              for (t6=32*t1+32;t6<=-32*t1+2*t4;t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
                ex[0][(-t4+t6)] = ex[0][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[0][(-t4+t6)]-hz[0][(-t4+t6)-1]);;
              }
              for (t5=t4+1;t5<=min(32*t1+31,t4+_PB_NX-1);t5++) {
                for (t6=32*t1+32;t6<=-32*t1+2*t4;t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
                hz[(-t4+t5-1)][(-32*t1+t4)] = hz[(-t4+t5-1)][(-32*t1+t4)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-32*t1+t4)+1] - ex[(-t4+t5-1)][(-32*t1+t4)] + ey[(-t4+t5-1)+1][(-32*t1+t4)] - ey[(-t4+t5-1)][(-32*t1+t4)]);;
              }
            }
          }
          if ((_PB_NX >= 2) && (t1 == t3) && (t1 == t2-1)) {
            for (t4=max(32*t1-_PB_NY+33,32*t1+_PB_NY-1);t4<=min(_PB_TMAX-1,32*t1+30);t4++) {
              for (t6=32*t1+32;t6<=t4+_PB_NY-1;t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
                ex[0][(-t4+t6)] = ex[0][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[0][(-t4+t6)]-hz[0][(-t4+t6)-1]);;
              }
              for (t5=t4+1;t5<=min(32*t1+31,t4+_PB_NX-1);t5++) {
                for (t6=32*t1+32;t6<=t4+_PB_NY-1;t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
              }
            }
          }
          if (_PB_NX >= 2) {
            for (t4=max(max(max(32*t3,32*t1+32),32*t1+_PB_NY-1),32*t2-_PB_NY+1);t4<=min(min(_PB_TMAX-1,32*t3+30),16*t1+16*t2+15);t4++) {
              for (t6=32*t2;t6<=t4+_PB_NY-1;t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
                ex[0][(-t4+t6)] = ex[0][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[0][(-t4+t6)]-hz[0][(-t4+t6)-1]);;
              }
              for (t5=t4+1;t5<=min(32*t3+31,t4+_PB_NX-1);t5++) {
                for (t6=32*t2;t6<=t4+_PB_NY-1;t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
              }
            }
          }
          if ((_PB_NX >= 2) && (_PB_NY >= 2) && (t1 == t2) && (t1 == t3)) {
            for (t4=32*t1+16;t4<=min(_PB_TMAX-1,32*t1+30);t4++) {
              ey[0][0] = _fict_[t4];;
              for (t6=t4+1;t6<=min(32*t1+31,t4+_PB_NY-1);t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
                ex[0][(-t4+t6)] = ex[0][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[0][(-t4+t6)]-hz[0][(-t4+t6)-1]);;
              }
              for (t5=t4+1;t5<=min(32*t1+31,t4+_PB_NX-1);t5++) {
                ey[(-t4+t5)][0] = ey[(-t4+t5)][0] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][0]-hz[(-t4+t5)-1][0]);;
                for (t6=t4+1;t6<=min(32*t1+31,t4+_PB_NY-1);t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
              }
            }
          }
          if ((_PB_NX >= 2) && (t1 <= t2-1)) {
            for (t4=max(32*t3,16*t1+16*t2+16);t4<=min(min(min(_PB_TMAX-1,32*t3+30),16*t1+16*t2+30),32*t1+_PB_NY+29);t4++) {
              for (t6=-32*t1+2*t4-31;t6<=min(32*t2+31,t4+_PB_NY-1);t6++) {
                ey[0][(-t4+t6)] = _fict_[t4];;
                ex[0][(-t4+t6)] = ex[0][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[0][(-t4+t6)]-hz[0][(-t4+t6)-1]);;
              }
              for (t5=t4+1;t5<=min(32*t3+31,t4+_PB_NX-1);t5++) {
                ey[(-t4+t5)][(-32*t1+t4-31)] = ey[(-t4+t5)][(-32*t1+t4-31)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-32*t1+t4-31)]-hz[(-t4+t5)-1][(-32*t1+t4-31)]);;
                ex[(-t4+t5)][(-32*t1+t4-31)] = ex[(-t4+t5)][(-32*t1+t4-31)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-32*t1+t4-31)]-hz[(-t4+t5)][(-32*t1+t4-31)-1]);;
                for (t6=-32*t1+2*t4-30;t6<=min(32*t2+31,t4+_PB_NY-1);t6++) {
                  ey[(-t4+t5)][(-t4+t6)] = ey[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)-1][(-t4+t6)]);;
                  ex[(-t4+t5)][(-t4+t6)] = ex[(-t4+t5)][(-t4+t6)] - SCALAR_VAL(0.5)*(hz[(-t4+t5)][(-t4+t6)]-hz[(-t4+t5)][(-t4+t6)-1]);;
                  hz[(-t4+t5-1)][(-t4+t6-1)] = hz[(-t4+t5-1)][(-t4+t6-1)] - SCALAR_VAL(0.7)* (ex[(-t4+t5-1)][(-t4+t6-1)+1] - ex[(-t4+t5-1)][(-t4+t6-1)] + ey[(-t4+t5-1)+1][(-t4+t6-1)] - ey[(-t4+t5-1)][(-t4+t6-1)]);;
                }
              }
            }
          }
          if ((_PB_NX >= 2) && (t1 <= min(min(floord(16*t2-_PB_NY,16),floord(32*t3-_PB_NY,32)),floord(_PB_TMAX-_PB_NY-31,32))) && (t1 >= ceild(32*t3-_PB_NY-30,32))) {
            ex[0][(_PB_NY-1)] = ex[0][(_PB_NY-1)] - SCALAR_VAL(0.5)*(hz[0][(_PB_NY-1)]-hz[0][(_PB_NY-1)-1]);;
            ey[0][(_PB_NY-1)] = _fict_[(32*t1+_PB_NY+30)];;
            for (t5=32*t1+_PB_NY+31;t5<=min(32*t3+31,32*t1+_PB_NY+_PB_NX+29);t5++) {
              ey[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)] = ey[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)] - SCALAR_VAL(0.5)*(hz[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)]-hz[(-32*t1+t5-_PB_NY-30)-1][(_PB_NY-1)]);;
              ex[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)] = ex[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)] - SCALAR_VAL(0.5)*(hz[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)]-hz[(-32*t1+t5-_PB_NY-30)][(_PB_NY-1)-1]);;
            }
          }
          if ((_PB_NX >= 2) && (t1 == -t2+2*t3-1) && (t1 <= floord(-16*t2+_PB_TMAX-32,16)) && (t1 >= ceild(16*t2-_PB_NY+1,16))) {
            if ((t1+t2+1)%2 == 0) {
              ex[0][(-16*t1+16*t2)] = ex[0][(-16*t1+16*t2)] - SCALAR_VAL(0.5)*(hz[0][(-16*t1+16*t2)]-hz[0][(-16*t1+16*t2)-1]);;
            }
            if ((t1+t2+1)%2 == 0) {
              ey[0][(-16*t1+16*t2)] = _fict_[(16*t1+16*t2+31)];;
            }
            for (t5=16*t1+16*t2+32;t5<=min(16*t1+16*t2+47,16*t1+16*t2+_PB_NX+30);t5++) {
              if ((t1+t2+1)%2 == 0) {
                ey[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)] = ey[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)] - SCALAR_VAL(0.5)*(hz[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)]-hz[(-16*t1-16*t2+t5-31)-1][(-16*t1+16*t2)]);;
              }
              if ((t1+t2+1)%2 == 0) {
                ex[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)] = ex[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)] - SCALAR_VAL(0.5)*(hz[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)]-hz[(-16*t1-16*t2+t5-31)][(-16*t1+16*t2)-1]);;
              }
            }
          }
          if ((_PB_NX >= 2) && (t1 >= max(ceild(32*t3-_PB_NY+1,32),-t2+2*t3)) && (t2 >= t3+1) && (t3 <= floord(_PB_TMAX-32,32))) {
            for (t6=max(32*t2,-32*t1+64*t3+31);t6<=min(min(32*t2+31,-32*t1+64*t3+62),32*t3+_PB_NY+30);t6++) {
              ey[0][(-32*t3+t6-31)] = _fict_[(32*t3+31)];;
              ex[0][(-32*t3+t6-31)] = ex[0][(-32*t3+t6-31)] - SCALAR_VAL(0.5)*(hz[0][(-32*t3+t6-31)]-hz[0][(-32*t3+t6-31)-1]);;
            }
          }
          if ((_PB_NX >= 2) && (t1 == t2) && (t1 == t3) && (t1 <= floord(_PB_TMAX-32,32))) {
            ey[0][0] = _fict_[(32*t1+31)];;
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

// CHECK: #map = affine_map<()[s0] -> (s0 - 1)>

// CHECK:    func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1200xf64>, %arg6: memref<500xf64>) {
// CHECK-NEXT:    %cst = constant 5.000000e-01 : f64
// CHECK-NEXT:    %cst_0 = constant 0.69999999999999996 : f64
// CHECK-NEXT:    %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:    %1 = index_cast %arg2 : i32 to index
// CHECK-NEXT:    %2 = index_cast %arg0 : i32 to index
// CHECK-NEXT:    affine.for %arg7 = 0 to %2 {
// CHECK-NEXT:      %3 = affine.load %arg6[%arg7] : memref<500xf64>
// CHECK-NEXT:      affine.for %arg8 = 0 to %1 {
// CHECK-NEXT:        affine.store %3, %arg4[0, %arg8] : memref<1000x1200xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg8 = 1 to %0 {
// CHECK-NEXT:        affine.for %arg9 = 0 to %1 {
// CHECK-NEXT:          %4 = affine.load %arg4[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %5 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %6 = affine.load %arg5[%arg8 - 1, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %7 = subf %5, %6 : f64
// CHECK-NEXT:          %8 = mulf %cst, %7 : f64
// CHECK-NEXT:          %9 = subf %4, %8 : f64
// CHECK-NEXT:          affine.store %9, %arg4[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:        affine.for %arg9 = 1 to %1 {
// CHECK-NEXT:          %4 = affine.load %arg3[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %5 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %6 = affine.load %arg5[%arg8, %arg9 - 1] : memref<1000x1200xf64>
// CHECK-NEXT:          %7 = subf %5, %6 : f64
// CHECK-NEXT:          %8 = mulf %cst, %7 : f64
// CHECK-NEXT:          %9 = subf %4, %8 : f64
// CHECK-NEXT:          affine.store %9, %arg3[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg8 = 0 to #map()[%0] {
// CHECK-NEXT:        affine.for %arg9 = 0 to #map()[%1] {
// CHECK-NEXT:          %4 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %5 = affine.load %arg3[%arg8, %arg9 + 1] : memref<1000x1200xf64>
// CHECK-NEXT:          %6 = affine.load %arg3[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %7 = subf %5, %6 : f64
// CHECK-NEXT:          %8 = affine.load %arg4[%arg8 + 1, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %9 = addf %7, %8 : f64
// CHECK-NEXT:          %10 = affine.load %arg4[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %11 = subf %9, %10 : f64
// CHECK-NEXT:          %12 = mulf %cst_0, %11 : f64
// CHECK-NEXT:          %13 = subf %4, %12 : f64
// CHECK-NEXT:          affine.store %13, %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


// EXEC: {{[0-9]\.[0-9]+}}

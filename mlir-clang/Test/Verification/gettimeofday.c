// RUN: mlir-clang %s %stdinclude --function=alloc -S | FileCheck %s

#include <time.h>
#include <sys/time.h>
double alloc() {
  struct timeval Tp;
  gettimeofday(&Tp, NULL);
  return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

// CHECK:   func @alloc() -> f64
// CHECK-NEXT:     %cst = constant 9.9999999999999995E-7 : f64
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c1_i64 = constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr<struct<(i64, i64)>>
// CHECK-NEXT:     %1 = llvm.mlir.null : !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT:     %2 = llvm.call @gettimeofday(%0, %1) : (!llvm.ptr<struct<(i64, i64)>>, !llvm.ptr<struct<(i32, i32)>>) -> i32
// CHECK-NEXT:     %3 = llvm.getelementptr %0[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:     %4 = llvm.load %3 : !llvm.ptr<i64>
// CHECK-NEXT:     %5 = llvm.getelementptr %0[%c0_i32, %c1_i32] : (!llvm.ptr<struct<(i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:     %6 = llvm.load %5 : !llvm.ptr<i64>
// CHECK-NEXT:     %7 = sitofp %4 : i64 to f64
// CHECK-NEXT:     %8 = sitofp %6 : i64 to f64
// CHECK-NEXT:     %9 = mulf %8, %cst : f64
// CHECK-NEXT:     %10 = addf %7, %9 : f64
// CHECK-NEXT:     return %10 : f64
// CHECK-NEXT:   }

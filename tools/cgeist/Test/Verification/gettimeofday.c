// RUN: cgeist %s %stdinclude --function=alloc -S | FileCheck %s

#include <time.h>
#include <sys/time.h>
double alloc() {
  struct timeval Tp;
  gettimeofday(&Tp, NULL);
  return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

// CHECK:   func @alloc() -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %cst = arith.constant 9.9999999999999995E-7 : f64
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x2xi64>
// CHECK-NEXT:     %1 = "polygeist.memref2pointer"(%0) : (memref<1x2xi64>) -> !llvm.ptr<struct<(i64, i64)>>
// CHECK-NEXT:     %2 = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT:     %3 = llvm.call @gettimeofday(%1, %2) : (!llvm.ptr<struct<(i64, i64)>>, !llvm.ptr<i8>) -> i32
// CHECK-NEXT:     %4 = affine.load %0[0, 0] : memref<1x2xi64>
// CHECK-NEXT:     %5 = arith.sitofp %4 : i64 to f64
// CHECK-NEXT:     %6 = affine.load %0[0, 1] : memref<1x2xi64>
// CHECK-NEXT:     %7 = arith.sitofp %6 : i64 to f64
// CHECK-NEXT:     %8 = arith.mulf %7, %cst : f64
// CHECK-NEXT:     %9 = arith.addf %5, %8 : f64
// CHECK-NEXT:     return %9 : f64
// CHECK-NEXT:   }

// RUN: cgeist %s %stdinclude --function=alloc -S | FileCheck %s

#include <time.h>
#include <sys/time.h>
double alloc() {
  struct timeval Tp;
  gettimeofday(&Tp, NULL);
  return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

// CHECK:   func @alloc() -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[cst:.+]] = arith.constant 9.9999999999999995E-7 : f64
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x2xi64>
// CHECK-NEXT:     %[[V1:.+]] = memref.cast %[[V0]] : memref<1x2xi64> to memref<?x2xi64>
// CHECK-NEXT:     %[[V2:.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT:     %[[V3:.+]] = "polygeist.pointer2memref"(%[[V2]]) : (!llvm.ptr<i8>) -> memref
// CHECK-NEXT:     %[[V4:.+]] = call @gettimeofday(%[[V1]], %[[V3]]) : (memref<?x2xi64>, memref
// CHECK-NEXT:     %[[V5:.+]] = affine.load %[[V0]][0, 0] : memref<1x2xi64>
// CHECK-NEXT:     %[[V6:.+]] = arith.sitofp %[[V5]] : i64 to f64
// CHECK-NEXT:     %[[V7:.+]] = affine.load %[[V0]][0, 1] : memref<1x2xi64>
// CHECK-NEXT:     %[[V8:.+]] = arith.sitofp %[[V7]] : i64 to f64
// CHECK-NEXT:     %[[V9:.+]] = arith.mulf %[[V8]], %[[cst]] : f64
// CHECK-NEXT:     %[[V10:.+]] = arith.addf %[[V6]], %[[V9]] : f64
// CHECK-NEXT:     return %[[V10]] : f64
// CHECK-NEXT:   }

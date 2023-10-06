// RUN: cgeist %s %stdinclude --function=alloc -S | FileCheck %s

#include <time.h>
#include <sys/time.h>
double alloc() {
  struct timeval Tp;
  gettimeofday(&Tp, NULL);
  return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}



// CHECK-LABEL:   func.func @alloc() -> f64  
// CHECK-DAG:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 9.9999999999999995E-7 : f64
// CHECK-DAG:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x2xi64>
// CHECK-DAG:           %[[VAL_2:[A-Za-z0-9_]*]] = memref.cast %[[VAL_1]] : memref<1x2xi64> to memref<?x2xi64>
// CHECK-DAG:           %[[VAL_3:[A-Za-z0-9_]*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<[[MEMREF_TY:.*]]>
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = call @gettimeofday(%[[VAL_2]], %[[VAL_4]]) : (memref<?x2xi64>, memref<[[MEMREF_TY:.*]]>) -> i32
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = affine.load %[[VAL_1]][0, 0] : memref<1x2xi64>
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = arith.sitofp %[[VAL_6]] : i64 to f64
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = affine.load %[[VAL_1]][0, 1] : memref<1x2xi64>
// CHECK:           %[[VAL_9:[A-Za-z0-9_]*]] = arith.sitofp %[[VAL_8]] : i64 to f64
// CHECK:           %[[VAL_10:[A-Za-z0-9_]*]] = arith.mulf %[[VAL_9]], %[[VAL_0]] : f64
// CHECK:           %[[VAL_11:[A-Za-z0-9_]*]] = arith.addf %[[VAL_7]], %[[VAL_10]] : f64
// CHECK:           return %[[VAL_11]] : f64
// CHECK:         }

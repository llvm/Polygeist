// RUN: cgeist %s --function=* -S | FileCheck %s

typedef struct {
  int a, b;
} pair;

pair byval0(pair a, int x);
pair byval(pair a, int x) {
  a.b = x;
  return a;
}

int create() {
  pair p;
  p.a = 0;
  p.b = 1;
  pair p2 = byval0(p, 2);
  return p2.a;
}

// CHECK:   func @byval(%[[arg0:.+]]: memref<?x2xi32>, %[[arg1:.+]]: i32, %[[arg2:.+]]: memref<?x2xi32>)
// CHECK-NEXT:     affine.store %[[arg1]], %[[arg0]][0, 1] : memref<?x2xi32>
// CHECK-NEXT:     %[[V0:.+]] = affine.load %[[arg0]][0, 0] : memref<?x2xi32>
// CHECK-NEXT:     affine.store %[[V0]], %[[arg2]][0, 0] : memref<?x2xi32>
// CHECK-NEXT:     %[[V1:.+]] = affine.load %[[arg0]][0, 1] : memref<?x2xi32>
// CHECK-NEXT:     affine.store %[[V1]], %[[arg2]][0, 1] : memref<?x2xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @create() -> i32
// CHECK-DAG:     %[[c2_i32:.+]] = arith.constant 2 : i32
// CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x2xi32>
// CHECK-NEXT:     %[[V1:.+]] = memref.alloca() : memref<1x2xi32>
// CHECK-NEXT:     affine.store %[[c0_i32]], %[[V1]][0, 0] : memref<1x2xi32>
// CHECK-NEXT:     affine.store %[[c1_i32]], %[[V1]][0, 1] : memref<1x2xi32>
// CHECK-NEXT:     %[[V2:.+]] = memref.cast %[[V1]] : memref<1x2xi32> to memref<?x2xi32>
// CHECK-NEXT:     %[[V3:.+]] = memref.cast %[[V0]] : memref<1x2xi32> to memref<?x2xi32>
// CHECK-NEXT:     call @byval0(%[[V2]], %[[c2_i32]], %[[V3]]) : (memref<?x2xi32>, i32, memref<?x2xi32>) -> ()
// CHECK-NEXT:     %[[V4:.+]] = affine.load %[[V0]][0, 0] : memref<1x2xi32>
// CHECK-NEXT:     return %[[V4]] : i32
// CHECK-NEXT:   }

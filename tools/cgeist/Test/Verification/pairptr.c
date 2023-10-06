// RUN: cgeist %s --function=* -S | FileCheck %s

typedef struct {
  int a, b;
} pair;

pair byval0(pair* a, int x);
pair byval(pair* a, int x) {
  return *a;
}

int create() {
  pair p;
  p.a = 0;
  p.b = 1;
  pair p2 = byval0(&p, 2);
  return p2.a;
}



// CHECK-LABEL:   func.func @byval(
// CHECK-SAME:                     %[[VAL_0:[a-zA-Z0-9]*]]: memref<?x2xi32>,
// CHECK-SAME:                     %[[VAL_1:[A-Za-z0-9_]*]]: i32,
// CHECK-SAME:                     %[[VAL_2:[A-Za-z0-9_]*]]: memref<?x2xi32>)
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = affine.load %[[VAL_0]][0, 0] : memref<?x2xi32>
// CHECK:           affine.store %[[VAL_3]], %[[VAL_2]][0, 0] : memref<?x2xi32>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = affine.load %[[VAL_0]][0, 1] : memref<?x2xi32>
// CHECK:           affine.store %[[VAL_4]], %[[VAL_2]][0, 1] : memref<?x2xi32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @create() -> i32
// CHECK-DAG:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 2 : i32
// CHECK-DAG:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 1 : i32
// CHECK-DAG:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 0 : i32
// CHECK-DAG:           %[[VAL_3:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x2xi32>
// CHECK-DAG:           %[[VAL_4:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x2xi32>
// CHECK-DAG:           %[[VAL_5:[A-Za-z0-9_]*]] = memref.cast %[[VAL_4]] : memref<1x2xi32> to memref<?x2xi32>
// CHECK-DAG:           affine.store %[[VAL_2]], %[[VAL_4]][0, 0] : memref<1x2xi32>
// CHECK-DAG:           affine.store %[[VAL_1]], %[[VAL_4]][0, 1] : memref<1x2xi32>
// CHECK-DAG:           %[[VAL_6:[A-Za-z0-9_]*]] = memref.cast %[[VAL_3]] : memref<1x2xi32> to memref<?x2xi32>
// CHECK:           call @byval0(%[[VAL_5]], %[[VAL_0]], %[[VAL_6]]) : (memref<?x2xi32>, i32, memref<?x2xi32>) -> ()
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = affine.load %[[VAL_3]][0, 0] : memref<1x2xi32>
// CHECK:           return %[[VAL_7]] : i32
// CHECK:         }


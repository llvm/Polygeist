// RUN: mlir-clang %s --function=struct_pass_all_same | FileCheck %s

typedef struct {
  int a, b, c;
} threeInt;

int struct_pass_all_same(threeInt* a) {
  return a->b;
}

// CHECK:  func @struct_pass_all_same(%arg0: memref<?x3xi32>) -> i32 {
// CHECK-NEXT:    %c1 = constant 1 : index
// CHECK-NEXT:    %c0 = constant 0 : index
// CHECK-NEXT:    %0 = memref.load %arg0[%c1, %c0] : memref<?x3xi32>
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
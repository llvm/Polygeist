// RUN: cgeist %s --function=* -S | FileCheck %s

#include <utility>

float moo(float &&x) {
  return std::move(x);
}

// CHECK:  func.func @_Z3mooOf(%[[arg0:.+]]: memref<?xf32>)
// CHECK-NEXT:    %[[V0:.+]] = affine.load %[[arg0]][0] : memref<?xf32>
// CHECK-NEXT:    return %[[V0]] : f32
// CHECK-NEXT:  }

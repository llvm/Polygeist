// RUN: polygeist-opt --lower-polygeist-ops --split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @main(
// CHECK-SAME:               %[[VAL_0:.*]]: index) -> memref<30xi32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<30x30xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_0]], %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_3]]], sizes: [30], strides: [1] : memref<30x30xi32> to memref<30xi32>
// CHECK:           return %[[VAL_4]] : memref<30xi32>
// CHECK:         }
module {
  func @main(%arg0 : index) -> memref<30xi32> {
    %0 = memref.alloca() : memref<30x30xi32>
    %1 = "polygeist.subindex"(%0, %arg0) : (memref<30x30xi32>, index) -> memref<30xi32>
    return %1 : memref<30xi32>
  }
}

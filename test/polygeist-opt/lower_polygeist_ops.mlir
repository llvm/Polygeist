// RUN: polygeist-opt --lower-polygeist-ops --split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @main(
// CHECK-SAME:               %[[VAL_0:.*]]: index) -> memref<30xi32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<30x30xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_0]], %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_3]]], sizes: [30], strides: [1] : memref<30x30xi32> to memref<30xi32>
// CHECK:           return %[[VAL_4]] : memref<30xi32>
// CHECK:         }
func @main(%arg0 : index) -> memref<30xi32> {
  %0 = memref.alloca() : memref<30x30xi32>
  %1 = "polygeist.subindex"(%0, %arg0) : (memref<30x30xi32>, index) -> memref<30xi32>
  return %1 : memref<30xi32>
}

// -----

// CHECK-LABEL:   func @main(
// CHECK-SAME:               %[[VAL_0:.*]]: index) -> memref<42x43x44x45xi32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<41x42x43x44x45xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 3575880 : index
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_0]], %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_3]]], sizes: [42, 43, 44, 45], strides: [85140, 1980, 45, 1] : memref<41x42x43x44x45xi32> to memref<42x43x44x45xi32>
// CHECK:           return %[[VAL_4]] : memref<42x43x44x45xi32>
// CHECK:         }
func @main(%arg0 : index) -> memref<42x43x44x45xi32> {
  %0 = memref.alloca() : memref<41x42x43x44x45xi32>
  %1 = "polygeist.subindex"(%0, %arg0) : (memref<41x42x43x44x45xi32>, index) -> memref<42x43x44x45xi32>
  return %1 : memref<42x43x44x45xi32>
}

// -----

// CHECK-LABEL:   func @main(
// CHECK-SAME:               %[[VAL_0:.*]]: index) -> memref<29x30xi32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<30x30xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 870 : index
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_0]], %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_3]]], sizes: [29, 30], strides: [30, 1] : memref<30x30xi32> to memref<29x30xi32>
// CHECK:           return %[[VAL_4]] : memref<29x30xi32>
// CHECK:         }

func @main(%arg0 : index) -> memref<29x30xi32> {
  %0 = memref.alloca() : memref<30x30xi32>
  %1 = "polygeist.subindex"(%0, %arg0) : (memref<30x30xi32>, index) -> memref<29x30xi32>
  return %1 : memref<29x30xi32>
}

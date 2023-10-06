// RUN: polygeist-opt --cpuify="method=distribute" --split-input-file %s | FileCheck %s

module {
  func.func private @capture(%a : memref<i32>) 
  func.func private @use(%a : memref<?xi32>, %b : f32, %d : i32, %e : f32)
  func.func @main() {
    %c0 = arith.constant 0 : index
    %cc1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    scf.parallel (%arg2) = (%c0) to (%c5) step (%cc1) {
      %a1 = memref.alloca() : memref<2xi32>
      %a2 = memref.cast %a1 : memref<2xi32> to memref<?xi32>
      %b1 = memref.alloca() : memref<f32>
      %c1 = memref.alloca() : memref<i32>
      %d1 = memref.alloca() : memref<1xi32>
      %b2 = memref.load %b1[] : memref<f32>
      func.call @capture(%c1) : (memref<i32>) -> ()
      %d2 = memref.cast %d1 : memref<1xi32> to memref<?xi32>

      %e1 = memref.alloca() : memref<1xf32>
      %e2 = memref.cast %e1 : memref<1xf32> to memref<?xf32>
      %e3 = memref.load %e2[%c0] : memref<?xf32>

      "polygeist.barrier"(%arg2) : (index) -> ()

      %d3 = memref.load %d2[%c0] : memref<?xi32>
      func.call @use(%a2, %b2, %d3, %e3) : (memref<?xi32>, f32, i32, f32) -> ()
      scf.yield
    }
    return
  }
}

// CHECK-LABEL:   func.func @main() {
// CHECK:           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = arith.constant 5 : index
// CHECK:           memref.alloca_scope  {
// CHECK:             %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.alloca(%[[VAL_2]]) : memref<?xf32>
// CHECK:             %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.alloca(%[[VAL_2]]) : memref<?xf32>
// CHECK:             %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.alloca(%[[VAL_2]]) : memref<?xmemref<?xi32>>
// CHECK:             %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.alloca(%[[VAL_2]]) : memref<?x2xi32>
// CHECK:             %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.alloca(%[[VAL_2]]) : memref<?xi32>
// CHECK:             %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.alloca(%[[VAL_2]]) : memref<?x1xi32>
// CHECK:             scf.parallel (%[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]) = (%[[VAL_0]]) to (%[[VAL_2]]) step (%[[VAL_1]]) {
// CHECK:               %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = "polygeist.subindex"(%[[VAL_6]], %[[VAL_9]]) : (memref<?x2xi32>, index) -> memref<2xi32>
// CHECK:               %[[VAL_11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.cast %[[VAL_10]] : memref<2xi32> to memref<?xi32>
// CHECK:               memref.store %[[VAL_11]], %[[VAL_5]]{{\[}}%[[VAL_9]]] : memref<?xmemref<?xi32>>
// CHECK:               %[[VAL_12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.alloca() : memref<f32>
// CHECK:               %[[VAL_13:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.load %[[VAL_12]][] : memref<f32>
// CHECK:               memref.store %[[VAL_13]], %[[VAL_4]]{{\[}}%[[VAL_9]]] : memref<?xf32>
// CHECK:               %[[VAL_14:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = "polygeist.subindex"(%[[VAL_7]], %[[VAL_9]]) : (memref<?xi32>, index) -> memref<i32>
// CHECK:               func.call @capture(%[[VAL_14]]) : (memref<i32>) -> ()
// CHECK:               %[[VAL_15:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.alloca() : memref<1xf32>
// CHECK:               %[[VAL_16:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_0]]] : memref<1xf32>
// CHECK:               memref.store %[[VAL_16]], %[[VAL_3]]{{\[}}%[[VAL_9]]] : memref<?xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.parallel (%[[VAL_17:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]) = (%[[VAL_0]]) to (%[[VAL_2]]) step (%[[VAL_1]]) {
// CHECK:               %[[VAL_18:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_17]]] : memref<?xmemref<?xi32>>
// CHECK:               %[[VAL_19:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_17]]] : memref<?xf32>
// CHECK:               %[[VAL_20:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_17]]] : memref<?xf32>
// CHECK:               %[[VAL_21:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = "polygeist.subindex"(%[[VAL_8]], %[[VAL_17]]) : (memref<?x1xi32>, index) -> memref<1xi32>
// CHECK:               %[[VAL_22:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_0]]] : memref<1xi32>
// CHECK:               func.call @use(%[[VAL_18]], %[[VAL_19]], %[[VAL_22]], %[[VAL_20]]) : (memref<?xi32>, f32, i32, f32) -> ()
// CHECK:               scf.yield
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }


// RUN: polygeist-opt --canonicalize-polygeist --split-input-file %s --allow-unregistered-dialect | FileCheck %s

#set = affine_set<(d0, d1, d2, d3)[s0, s1] : (d0 + d1 * 16 - s1 + (d2 + d3 * 16) * s0 >= 0)>
#set1 = affine_set<(d0, d1, d2, d3)[s0, s1] : (d1 - s1 + d3 * 16 + (d0 + d2 * 16) * s0 - 1 >= 0)>
module {
  func.func @main(%arg0: i32, %arg1: index, %arg2: index, %arg3: memref<?xi32>, %arg4: memref<?xi32>) {
    %c0_i32 = arith.constant 0 : i32
    affine.parallel (%arg5, %arg6) = (0, 0) to (10, 10) {
      %alloca = memref.alloca() : memref<16x16xi32>
      %alloca_0 = memref.alloca() : memref<16x16xi32>
      affine.parallel (%arg7, %arg8) = (0, 0) to (16, 16) {
        affine.for %arg9 = 0 to 10 {
          %0 = affine.if #set(%arg7, %arg9, %arg8, %arg6)[%arg1, %arg2] -> i32 {
            affine.yield %c0_i32 : i32
          } else {
            %5 = affine.load %arg3[%arg7 + %arg9 * 16 + (%arg8 + %arg6 * 16) * symbol(%arg1)] : memref<?xi32>
            affine.yield %5 : i32
          }
          affine.store %0, %alloca[%arg8, %arg7] : memref<16x16xi32>
          %1 = affine.if #set1(%arg8, %arg7, %arg9, %arg5)[%arg1, %arg2] -> i32 {
            affine.yield %c0_i32 : i32
          } else {
            %5 = affine.load %arg4[%arg7 + %arg5 * 16 + (%arg8 + %arg9 * 16) * symbol(%arg1)] : memref<?xi32>
            affine.yield %5 : i32
          }
          affine.store %1, %alloca_0[%arg8, %arg7] : memref<16x16xi32>
          "polygeist.barrier"(%arg7, %arg8) : (index, index) -> ()
          %2 = affine.load %alloca[%arg8, 0] : memref<16x16xi32>
          %3 = affine.load %alloca_0[0, %arg7] : memref<16x16xi32>
          %4 = arith.muli %2, %3 : i32
          "test.use"(%4) : (i32) -> ()
        }
      }
    }
    return
  }
}
// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0, d1, d2)[s0, s1] : (d0 * 16 - s1 + (d1 * 16 + d2) * s0 >= 0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1, d2)[s0, s1] : ((d0 * s0) * 16 + d1 * 16 - s1 + d2 - 1 >= 0)>
// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                    %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index,
// CHECK-SAME:                    %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index,
// CHECK-SAME:                    %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xi32>,
// CHECK-SAME:                    %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xi32>) {
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = arith.constant 0 : i32
// CHECK:           affine.parallel (%[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]], %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]], %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]], %[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]) = (0, 0, 0, 0) to (10, 10, 16, 16) {
// CHECK:             affine.for %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = 0 to 10 {
// CHECK:               "polygeist.barrier"(%[[VAL_8]], %[[VAL_9]]) : (index, index) -> ()
// CHECK:               %[[VAL_11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = affine.if #[[$ATTR_0]](%[[VAL_10]], %[[VAL_7]], %[[VAL_9]]){{\[}}%[[VAL_1]], %[[VAL_2]]] -> i32 {
// CHECK:                 affine.yield %[[VAL_5]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_10]] * 16 + (%[[VAL_7]] * 16 + %[[VAL_9]]) * symbol(%[[VAL_1]])] : memref<?xi32>
// CHECK:                 affine.yield %[[VAL_12]] : i32
// CHECK:               }
// CHECK:               %[[VAL_13:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = affine.if #[[$ATTR_1]](%[[VAL_10]], %[[VAL_6]], %[[VAL_8]]){{\[}}%[[VAL_1]], %[[VAL_2]]] -> i32 {
// CHECK:                 affine.yield %[[VAL_5]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_14:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_8]] + %[[VAL_6]] * 16 + (%[[VAL_10]] * symbol(%[[VAL_1]])) * 16] : memref<?xi32>
// CHECK:                 affine.yield %[[VAL_14]] : i32
// CHECK:               }
// CHECK:               %[[VAL_15:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = arith.muli %[[VAL_11]], %[[VAL_13]] : i32
// CHECK:               "test.use"(%[[VAL_15]]) : (i32) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }


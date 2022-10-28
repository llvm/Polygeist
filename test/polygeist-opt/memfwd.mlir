// RUN: polygeist-opt --canonicalize --split-input-file %s --allow-unregistered-dialect | FileCheck %s

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

// CHECK:   func.func @main(%arg0: i32, %arg1: index, %arg2: index, %arg3: memref<?xi32>, %arg4: memref<?xi32>) {
// CHECK-NEXT:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:     affine.parallel (%arg5, %arg6, %[[arg7:.+]], %[[arg8:.+]]) = (0, 0, 0, 0) to (10, 10, 16, 16) {
// CHECK-NEXT:       affine.for %[[arg9:.+]] = 0 to 10 {
// CHECK-NEXT:         "polygeist.barrier"(%[[arg7]], %[[arg8]]) : (index, index) -> ()
// CHECK-NEXT:         %[[i0:.+]] = affine.if #set(%[[arg9]], %arg6, %[[arg8]])[%arg1, %arg2] -> i32 {
// CHECK-NEXT:           affine.yield %[[c0_i32]] : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %[[i3:.+]] = affine.load %arg3[%[[arg9]] * 16 + (%arg6 * 16 + %arg8) * symbol(%arg1)] : memref<?xi32>
// CHECK-NEXT:           affine.yield %[[i3]] : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[i1:.+]] = affine.if #set1(%[[arg9]], %arg5, %[[arg7]])[%arg1, %arg2] -> i32 {
// CHECK-NEXT:           affine.yield %[[c0_i32]] : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %[[i3:.+]] = affine.load %arg4[%arg5 * 16 + (%[[arg9]] * symbol(%arg1)) * 16 + %[[arg7]]] : memref<?xi32>
// CHECK-NEXT:           affine.yield %[[i3]] : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[i2:.+]] = arith.muli %[[i0]], %[[i1]] : i32
// CHECK-NEXT:         "test.use"(%[[i2]]) : (i32) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



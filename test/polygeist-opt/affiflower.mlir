// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

#set0 = affine_set<(d0, d1) : (d0 + d1 * 512 == 0)>

module {
  func.func @f(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32

    affine.parallel (%arg7) = (0) to (symbol(%636)) {
      %706 = arith.index_cast %arg7 : index to i32
      %707 = arith.muli %706, %c512_i32 : i32
      affine.parallel (%arg8) = (0) to (512) {
        %708 = arith.index_cast %arg8 : index to i32
        %709 = arith.addi %707, %708 : i32
        affine.if #set0(%arg8, %arg7) {
          %712 = arith.sitofp %709 : i32 to f64
          affine.store %712, %603[0] : memref<?xf64>
        }
      }
    }
    return
  }

}

// CHECK: #set = affine_set<()[s0] : (s0 - 1 >= 0)>
// CHECK:   func.func @f(%arg0: index, %arg1: memref<?xf64>) {
// CHECK-NEXT:     %[[cst:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     affine.if #set()[%arg0] {
// CHECK-NEXT:       affine.store %cst, %arg1[0] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

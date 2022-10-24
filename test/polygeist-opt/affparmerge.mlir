// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

module {
  func.func @f(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32 
        affine.parallel (%arg7, %arg8) = (0, 0) to (symbol(%636), 512) {
          %706 = arith.index_cast %arg7 : index to i32
          %707 = arith.muli %706, %c512_i32 : i32
            %708 = arith.index_cast %arg8 : index to i32
            %709 = arith.addi %707, %708 : i32
              %712 = arith.sitofp %709 : i32 to f64
              affine.store %712, %603[%arg8 + %arg7 * 512] : memref<?xf64>
        }
    return
  }

}

// CHECK:   func.func @f(%[[arg0:.+]]: index, %[[arg1:.+]]: memref<?xf64>) {
// CHECK-NEXT:     affine.parallel (%[[arg2:.+]]) = (0) to (symbol(%[[arg0]]) * 512) {
// CHECK-NEXT:       %[[V0:.+]] = arith.index_cast %[[arg2]] : index to i32
// CHECK-NEXT:       %[[V1:.+]] = arith.sitofp %[[V0]] : i32 to f64
// CHECK-NEXT:       affine.store %[[V1]], %arg1[%[[arg2]]] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

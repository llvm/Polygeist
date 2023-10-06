// RUN: polygeist-opt --cpuify="method=distribute" --allow-unregistered-dialect --canonicalize-polygeist --split-input-file %s | FileCheck %s

module {
  func.func private @print()
  func.func @main() {
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%arg2) = (%c0) to (%c5) step (%c1) {
      %0 = llvm.alloca %c1_i64 x i8 : (i64) -> !llvm.ptr
      scf.parallel (%arg3) = (%c0) to (%c2) step (%c1) {
        %4 = scf.while (%arg4 = %c1_i8) : (i8) -> i8 {
          %6 = arith.cmpi ne, %arg4, %c0_i8 : i8
          scf.condition(%6) %arg4 : i8
        } do {
        ^bb0(%arg4: i8):  // no predecessors
          llvm.store %c0_i8, %0 : i8, !llvm.ptr
          "polygeist.barrier"(%arg3) : (index) -> ()
          scf.yield %c0_i8 : i8
        }
        %5 = arith.cmpi ne, %4, %c0_i8 : i8
        scf.if %5 {
          func.call @print() : () -> ()
        }
        scf.yield
      }
      "test.use"(%0) : (!llvm.ptr) -> ()
      scf.yield
    }
    return
  }
  func.func @_Z17compute_tran_tempPfPS_iiiiiiii(%arg0: memref<?xf32>, %len : index, %f : f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
        affine.parallel (%arg15, %arg16) = (0, 0) to (16, 16) {
            scf.for %arg17 = %c0 to %len step %c1 {
              affine.store %f, %arg0[%arg15] : memref<?xf32>
              "polygeist.barrier"(%arg15, %arg16, %c0) : (index, index, index) -> ()
            }
        }
    return
  }
}


// CHECK-LABEL:   func.func @main() {
// CHECK-NOT: polygeist.barrier

// CHECK-LABEL:   func.func @_Z17compute_tran_tempPfPS_iiiiiiii(
// CHECK-SAME:                                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                                                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index,
// CHECK-SAME:                                                  %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = %[[VAL_3]] to %[[VAL_1]] step %[[VAL_4]] {
// CHECK:             affine.parallel (%[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]], %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]) = (0, 0) to (16, 16) {
// CHECK:               affine.store %[[VAL_2]], %[[VAL_0]]{{\[}}%[[VAL_6]]] : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

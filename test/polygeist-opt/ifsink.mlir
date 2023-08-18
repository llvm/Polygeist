// RUN: polygeist-opt --canonicalize --allow-unregistered-dialect --split-input-file %s | FileCheck %s

#set0 = affine_set<(d0) : (-d0 == 0)>
#set1 = affine_set<(d0) : (d0 == 0)>
module {
  func.func @bpnn_train_cuda() {
      affine.parallel (%arg7) = (0) to (16) {
        "test.pre"() : () -> ()
        affine.if #set0(%arg7) {
          %a = "test.create"() : () -> i32
          "test.use"(%a) : (i32) -> ()
        }
    }
    return
  }
  func.func @bpnn_train_cuda1() {
      affine.parallel (%arg7) = (0) to (16) {
        "test.pre"() : () -> ()
        affine.if #set1(%arg7) {
          %a = "test.create"() : () -> i32
          "test.use"(%a) : (i32) -> ()
        }
    }
    return
  }
  func.func @bpnn_train_cuda2() {
      affine.parallel (%arg7) = (0) to (16) {
        %a = "test.create"() : () -> i32
        affine.if #set1(%arg7) {
          "test.use"(%a) : (i32) -> ()
        }
    }
    return
  }
}

// CHECK:   func.func @bpnn_train_cuda() {
// CHECK-NEXT:     affine.parallel (%[[arg0:.+]]) = (0) to (16) {
// CHECK-NEXT:       "test.pre"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V0:.+]] = "test.create"() : () -> i32
// CHECK-NEXT:     "test.use"(%[[V0]]) : (i32) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @bpnn_train_cuda1() {
// CHECK-NEXT:     affine.parallel (%[[arg0:.+]]) = (0) to (16) {
// CHECK-NEXT:       "test.pre"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V0:.+]] = "test.create"() : () -> i32
// CHECK-NEXT:     "test.use"(%[[V0]]) : (i32) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @bpnn_train_cuda2() {
// CHECK-NEXT:     affine.parallel (%[[arg0:.+]]) = (0) to (16) {
// CHECK-NEXT:       %[[V0:.+]] = "test.create"() : () -> i32
// CHECK-NEXT:       affine.if #set(%[[arg0]]) {
// CHECK-NEXT:         "test.use"(%[[V0]]) : (i32) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#set = affine_set<(d0, d1) : (d0 == 0, d1 == 0)>
module {
  func.func @_Z9test_caseIiEiPili(%30: memref<1024xi32>, %14: memref<i32>, %0: i32, %arg0: memref<?xi32>, %arg1: i64, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c256_i64 = arith.constant 256 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1024_i32 = arith.constant 1024 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %alloca = memref.alloca() : memref<1xi32>
    affine.parallel (%arg3) = (0) to (256) {
      affine.parallel (%arg4) = (0) to (256) {
        %alloca_1 = memref.alloca() : memref<i32>
        affine.store %0, %alloca_1[] : memref<i32>
        affine.store %c0_i32, %alloca_1[] : memref<i32>
        affine.for %arg5 = 0 to 4 {
          affine.for %arg6 = 0 to 1024 step 4 {
            %31 = affine.load %30[%arg6 + %arg5] : memref<1024xi32>
            %32 = affine.load %alloca_1[] : memref<i32>
            %33 = arith.addi %32, %31 : i32
            affine.store %33, %alloca_1[] : memref<i32>
          }
        }
        affine.if #set(%arg4, %arg3) {
          %31 = affine.load %alloca_1[] : memref<i32>
          affine.store %31, %14[] : memref<i32>
        }
      }
    }
    return
  }
}

// CHECK: TODO: fix bug in AffineIfSinking
// Output is this:
// module {
//   func.func @_Z9test_caseIiEiPili(%arg0: memref<1024xi32>, %arg1: memref<i32>, %arg2: i32, %arg3: memref<?xi32>, %arg4: i64, %arg5: i32) {
//     %c0_i32 = arith.constant 0 : i32
//     affine.parallel (%arg6, %arg7) = (0, 0) to (256, 256) {
//       %alloca_0 = memref.alloca() : memref<i32>
//       affine.store %arg2, %alloca_0[] : memref<i32>
//       affine.store %c0_i32, %alloca_0[] : memref<i32>
//       affine.for %arg8 = 0 to 4 {
//         affine.for %arg9 = 0 to 1024 step 4 {
//           %1 = affine.load %arg0[%arg9 + %arg8] : memref<1024xi32>
//           %2 = affine.load %alloca_0[] : memref<i32>
//           %3 = arith.addi %2, %1 : i32
//           affine.store %3, %alloca_0[] : memref<i32>
//         }
//       }
//     }
//     %alloca = memref.alloca() : memref<i32>
//     %0 = affine.load %alloca[] : memref<i32>
//     affine.store %0, %arg1[] : memref<i32>
//     return
//   }
// }


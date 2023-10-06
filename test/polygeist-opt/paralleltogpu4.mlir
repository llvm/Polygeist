// RUN: polygeist-opt --lower-affine --canonicalize-polygeist --convert-parallel-to-gpu1 --canonicalize-polygeist %s | FileCheck %s

module {
  func.func private @use(%arg0: index)
  func.func private @use_memref(%arg0: memref<16x16xf32, 5>)
  // Test case for when we originally had one block dim (%bd) and the rest are
  // grid dims (merged in one parallel op) and we want to preserve that
  func.func @f1(%gd : index, %bd : index) {
    %mc0 = arith.constant 0 : index
    %mc1 = arith.constant 1 : index
    %mc4 = arith.constant 4 : index
    %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"() ({
      affine.parallel (%a1, %a2, %a3) = (0, 0, 0) to (%gd, %mc4, %bd) {
        "polygeist.noop"(%a3, %mc0, %mc0) {polygeist.noop_type="gpu_kernel.thread_only"} : (index, index, index) -> ()
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        func.call @use(%a3) : (index) -> ()
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> index
    return
  }
}

// CHECK-LABEL:   func.func @f1(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_4:.*]] = "polygeist.gpu_error"() ({
// CHECK:               gpu.launch blocks(%[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]]) in (%[[VAL_8:.*]] = %[[VAL_0]], %[[VAL_9:.*]] = %[[VAL_3]], %[[VAL_10:.*]] = %[[VAL_2]]) threads(%[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]]) in (%[[VAL_14:.*]] = %[[VAL_1]], %[[VAL_15:.*]] = %[[VAL_2]], %[[VAL_16:.*]] = %[[VAL_2]]) {
// CHECK:                 %[[VAL_17:.*]] = gpu.block_id  x
// CHECK:                 %[[VAL_18:.*]] = gpu.block_id  y
// CHECK:                 %[[VAL_19:.*]] = gpu.thread_id  x
// CHECK:                 func.call @use(%[[VAL_17]]) : (index) -> ()
// CHECK:                 func.call @use(%[[VAL_18]]) : (index) -> ()
// CHECK:                 func.call @use(%[[VAL_19]]) : (index) -> ()
// CHECK:                 gpu.terminator
// CHECK:               }
// CHECK:               "polygeist.polygeist_yield"() : () -> ()
// CHECK:             }) : () -> index
// CHECK:           return
// CHECK:         }


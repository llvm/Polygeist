// RUN: polygeist-opt --lower-affine --canonicalize-polygeist --convert-parallel-to-gpu1 --canonicalize-polygeist %s | FileCheck %s

module {
  func.func private @use(%arg0: index)
  func.func private @use_memref(%arg0: memref<16x16xf32, 5>)
  func.func @f1() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1) = (0) to (10000000) {
        "polygeist.gpu_block"(%a1, %mc1, %mc1) ({
          affine.parallel (%a2) = (0) to (1024) {
            "polygeist.gpu_thread"(%a2, %mc1, %mc1) ({
              func.call @use(%a1) : (index) -> ()
              func.call @use(%a2) : (index) -> ()
              "polygeist.polygeist_yield"() : () -> ()
            }) : (index, index, index) -> ()
            affine.yield
          }
          "polygeist.polygeist_yield"() : () -> ()
        }) : (index, index, index) -> ()
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
  func.func @f2() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1) = (0) to (1) {
        "polygeist.gpu_block"(%mc1, %mc1, %mc1) ({
          affine.parallel (%a2) = (0) to (1024) {
            "polygeist.gpu_thread"(%a2, %mc1, %mc1) ({
              func.call @use(%a1) : (index) -> ()
              func.call @use(%a2) : (index) -> ()
              "polygeist.polygeist_yield"() : () -> ()
            }) : (index, index, index) -> ()
            affine.yield
          }
          "polygeist.polygeist_yield"() : () -> ()
        }) : (index, index, index) -> ()
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
  func.func @f3() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      %alloca_8 = memref.alloca() : memref<16x16xf32, 5>
      "polygeist.gpu_block"(%mc1, %mc1, %mc1) ({
          affine.parallel (%a2) = (0) to (1024) {
          "polygeist.gpu_thread"(%a2, %mc1, %mc1) ({
              func.call @use(%a2) : (index) -> ()
              func.call @use_memref(%alloca_8) : (memref<16x16xf32, 5>) -> ()
              "polygeist.polygeist_yield"() : () -> ()
          }) : (index, index, index) -> ()
          }
          "polygeist.polygeist_yield"() : () -> ()
      }) : (index, index, index) -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
}

// CHECK-LABEL:   func.func @f1() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 10000000 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_3:.*]] = "polygeist.gpu_error"() ({
// CHECK:             gpu.launch blocks(%[[VAL_4:.*]], %[[VAL_5:.*]], %[[VAL_6:.*]]) in (%[[VAL_7:.*]] = %[[VAL_1]], %[[VAL_8:.*]] = %[[VAL_0]], %[[VAL_9:.*]] = %[[VAL_0]]) threads(%[[VAL_10:.*]], %[[VAL_11:.*]], %[[VAL_12:.*]]) in (%[[VAL_13:.*]] = %[[VAL_2]], %[[VAL_14:.*]] = %[[VAL_0]], %[[VAL_15:.*]] = %[[VAL_0]]) {
// CHECK:               %[[VAL_16:.*]] = gpu.block_id  x
// CHECK:               %[[VAL_17:.*]] = gpu.thread_id  x
// CHECK:               func.call @use(%[[VAL_16]]) : (index) -> ()
// CHECK:               func.call @use(%[[VAL_17]]) : (index) -> ()
// CHECK:               gpu.terminator
// CHECK:             }
// CHECK:             "polygeist.polygeist_yield"() : () -> ()
// CHECK:           }) : () -> index
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @f2() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_3:.*]] = "polygeist.gpu_error"() ({
// CHECK:             gpu.launch blocks(%[[VAL_4:.*]], %[[VAL_5:.*]], %[[VAL_6:.*]]) in (%[[VAL_7:.*]] = %[[VAL_0]], %[[VAL_8:.*]] = %[[VAL_0]], %[[VAL_9:.*]] = %[[VAL_0]]) threads(%[[VAL_10:.*]], %[[VAL_11:.*]], %[[VAL_12:.*]]) in (%[[VAL_13:.*]] = %[[VAL_2]], %[[VAL_14:.*]] = %[[VAL_0]], %[[VAL_15:.*]] = %[[VAL_0]]) {
// CHECK:               %[[VAL_16:.*]] = gpu.thread_id  x
// CHECK:               func.call @use(%[[VAL_1]]) : (index) -> ()
// CHECK:               func.call @use(%[[VAL_16]]) : (index) -> ()
// CHECK:               gpu.terminator
// CHECK:             }
// CHECK:             "polygeist.polygeist_yield"() : () -> ()
// CHECK:           }) : () -> index
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @f3() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_2:.*]] = "polygeist.gpu_error"() ({
// CHECK:             gpu.launch blocks(%[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]]) in (%[[VAL_6:.*]] = %[[VAL_0]], %[[VAL_7:.*]] = %[[VAL_0]], %[[VAL_8:.*]] = %[[VAL_0]]) threads(%[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]]) in (%[[VAL_12:.*]] = %[[VAL_1]], %[[VAL_13:.*]] = %[[VAL_0]], %[[VAL_14:.*]] = %[[VAL_0]]) {
// CHECK-DAG:               %[[VAL_15:.*]] = gpu.thread_id  x
// TODO converting this to shared memory should probably happen in this parallel-to-gpu1 and not 2 because having an alloca doesnt really make sense here
// CHECK-DAG:               %alloca = memref.alloca() : memref<16x16xf32, 5>
// CHECK:               func.call @use(%1) : (index) -> ()
// CHECK:               func.call @use_memref(%alloca) : (memref<16x16xf32, 5>) -> ()
// CHECK:               gpu.terminator
// CHECK:             }
// CHECK:             "polygeist.polygeist_yield"() : () -> ()
// CHECK:           }) : () -> index
// CHECK:           return
// CHECK:         }


// RUN: polygeist-opt --lower-affine --canonicalize --convert-parallel-to-gpu1 --canonicalize %s | FileCheck %s

module {
  func.func private @use(%arg0: index)
  func.func private @wow()
  func.func @f1() {
    "polygeist.parallel_wrapper"() ({
      func.call @wow() : () -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> ()
    return
  }
// CHECK-LABEL:   func.func @f1() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           gpu.launch blocks(%[[VAL_1:.*]], %[[VAL_2:.*]], %[[VAL_3:.*]]) in (%[[VAL_4:.*]] = %[[VAL_0]], %[[VAL_5:.*]] = %[[VAL_0]], %[[VAL_6:.*]] = %[[VAL_0]]) threads(%[[VAL_7:.*]], %[[VAL_8:.*]], %[[VAL_9:.*]]) in (%[[VAL_10:.*]] = %[[VAL_0]], %[[VAL_11:.*]] = %[[VAL_0]], %[[VAL_12:.*]] = %[[VAL_0]]) {
// CHECK:             func.call @wow() : () -> ()
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

  func.func @f2() {
    "polygeist.parallel_wrapper"() ({
      affine.parallel (%a1) = (0) to (10000000) {
        func.call @use(%a1) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> ()
    return
  }
// CHECK-LABEL:   func.func @f2() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 9766 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10000000 : index
// CHECK:           gpu.launch blocks(%[[VAL_4:.*]], %[[VAL_5:.*]], %[[VAL_6:.*]]) in (%[[VAL_7:.*]] = %[[VAL_2]], %[[VAL_8:.*]] = %[[VAL_0]], %[[VAL_9:.*]] = %[[VAL_0]]) threads(%[[VAL_10:.*]], %[[VAL_11:.*]], %[[VAL_12:.*]]) in (%[[VAL_13:.*]] = %[[VAL_1]], %[[VAL_14:.*]] = %[[VAL_0]], %[[VAL_15:.*]] = %[[VAL_0]]) {
// CHECK:             %[[VAL_16:.*]] = gpu.block_id  x
// CHECK:             %[[VAL_17:.*]] = gpu.block_dim  x
// CHECK:             %[[VAL_18:.*]] = gpu.thread_id  x
// CHECK:             %[[VAL_19:.*]] = arith.muli %[[VAL_16]], %[[VAL_17]] : index
// CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_18]] : index
// CHECK:             %[[VAL_21:.*]] = arith.cmpi ult, %[[VAL_20]], %[[VAL_3]] : index
// CHECK:             scf.if %[[VAL_21]] {
// CHECK:               func.call @use(%[[VAL_20]]) : (index) -> ()
// CHECK:             }
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

  func.func @f3() {
    "polygeist.parallel_wrapper"() ({
      affine.parallel (%a1, %a2) = (0, 0) to (16, 16) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> ()
    return
  }
// CHECK-LABEL:   func.func @f3() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           gpu.launch blocks(%[[VAL_2:.*]], %[[VAL_3:.*]], %[[VAL_4:.*]]) in (%[[VAL_5:.*]] = %[[VAL_0]], %[[VAL_6:.*]] = %[[VAL_0]], %[[VAL_7:.*]] = %[[VAL_0]]) threads(%[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]]) in (%[[VAL_11:.*]] = %[[VAL_1]], %[[VAL_12:.*]] = %[[VAL_1]], %[[VAL_13:.*]] = %[[VAL_0]]) {
// CHECK:             %[[VAL_14:.*]] = gpu.thread_id  x
// CHECK:             %[[VAL_15:.*]] = gpu.thread_id  y
// CHECK:             func.call @use(%[[VAL_14]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_15]]) : (index) -> ()
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

  func.func @f4(%b1: index, %b2: index) {
    "polygeist.parallel_wrapper"() ({
      affine.parallel (%a1, %a2, %a3, %a4) = (0, 0, 0, 0) to (%b1, 17, %b2, 32) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        func.call @use(%a3) : (index) -> ()
        func.call @use(%a4) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> ()
    return
  }
// CHECK-LABEL:   func.func @f4(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 17 : index
// CHECK:           gpu.launch blocks(%[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]]) in (%[[VAL_8:.*]] = %[[VAL_0]], %[[VAL_9:.*]] = %[[VAL_1]], %[[VAL_10:.*]] = %[[VAL_2]]) threads(%[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]]) in (%[[VAL_14:.*]] = %[[VAL_4]], %[[VAL_15:.*]] = %[[VAL_3]], %[[VAL_16:.*]] = %[[VAL_2]]) {
// CHECK:             %[[VAL_17:.*]] = gpu.block_id  x
// CHECK:             %[[VAL_18:.*]] = gpu.block_id  y
// CHECK:             %[[VAL_19:.*]] = gpu.thread_id  x
// CHECK:             %[[VAL_20:.*]] = gpu.thread_id  y
// CHECK:             func.call @use(%[[VAL_17]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_19]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_18]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_20]]) : (index) -> ()
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

  func.func @f5(%b1: index, %b2: index) {
    "polygeist.parallel_wrapper"() ({
      affine.parallel (%a1, %a2, %a3, %a4) = (0, 0, 0, 0) to (%b1, 5, %b2, 3) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        func.call @use(%a3) : (index) -> ()
        func.call @use(%a4) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> ()
    return
  }
// CHECK-LABEL:   func.func @f5(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_6:.*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           %[[VAL_7:.*]] = arith.divui %[[VAL_6]], %[[VAL_3]] : index
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_2]] : index
// CHECK:           gpu.launch blocks(%[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]]) in (%[[VAL_12:.*]] = %[[VAL_0]], %[[VAL_13:.*]] = %[[VAL_8]], %[[VAL_14:.*]] = %[[VAL_2]]) threads(%[[VAL_15:.*]], %[[VAL_16:.*]], %[[VAL_17:.*]]) in (%[[VAL_18:.*]] = %[[VAL_3]], %[[VAL_19:.*]] = %[[VAL_5]], %[[VAL_20:.*]] = %[[VAL_4]]) {
// CHECK:             %[[VAL_21:.*]] = gpu.block_id  x
// CHECK:             %[[VAL_22:.*]] = gpu.block_id  y
// CHECK:             %[[VAL_23:.*]] = gpu.block_dim  y
// CHECK:             %[[VAL_24:.*]] = gpu.thread_id  x
// CHECK:             %[[VAL_25:.*]] = gpu.thread_id  y
// CHECK:             %[[VAL_26:.*]] = gpu.thread_id  z
// CHECK:             %[[VAL_27:.*]] = arith.muli %[[VAL_22]], %[[VAL_23]] : index
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_27]], %[[VAL_24]] : index
// CHECK:             %[[VAL_29:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_1]] : index
// CHECK:             scf.if %[[VAL_29]] {
// CHECK:               func.call @use(%[[VAL_21]]) : (index) -> ()
// CHECK:               func.call @use(%[[VAL_25]]) : (index) -> ()
// CHECK:               func.call @use(%[[VAL_28]]) : (index) -> ()
// CHECK:               func.call @use(%[[VAL_26]]) : (index) -> ()
// CHECK:             }
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

  func.func @f6(%b1: index, %b2: index) {
    "polygeist.parallel_wrapper"() ({
      affine.parallel (%a1, %a2, %a3, %a4) = (0, 0, 0, 0) to (%b1, 1023, %b2, 1025) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        func.call @use(%a3) : (index) -> ()
        func.call @use(%a4) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : () -> ()
    return
  }
// CHECK-LABEL:   func.func @f6(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1025 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1023 : index
// CHECK:           gpu.launch blocks(%[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]]) in (%[[VAL_8:.*]] = %[[VAL_0]], %[[VAL_9:.*]] = %[[VAL_1]], %[[VAL_10:.*]] = %[[VAL_3]]) threads(%[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]]) in (%[[VAL_14:.*]] = %[[VAL_4]], %[[VAL_15:.*]] = %[[VAL_2]], %[[VAL_16:.*]] = %[[VAL_2]]) {
// CHECK:             %[[VAL_17:.*]] = gpu.block_id  x
// CHECK:             %[[VAL_18:.*]] = gpu.block_id  y
// CHECK:             %[[VAL_19:.*]] = gpu.block_id  z
// CHECK:             %[[VAL_20:.*]] = gpu.thread_id  x
// CHECK:             func.call @use(%[[VAL_17]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_20]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_18]]) : (index) -> ()
// CHECK:             func.call @use(%[[VAL_19]]) : (index) -> ()
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

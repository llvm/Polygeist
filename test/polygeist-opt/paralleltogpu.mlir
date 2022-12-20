// RUN: polygeist-opt --lower-affine --canonicalize --convert-parallel-to-gpu1 --canonicalize %s | FileCheck %s

module {
  func.func private @use(%arg0: index)
  func.func private @wow()
  func.func @f1() {
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      func.call @wow() : () -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
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
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1) = (0) to (10000000) {
        func.call @use(%a1) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f2() {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1024 : index
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 9766 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 10000000 : index
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
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1, %a2) = (0, 0) to (16, 16) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f3() {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 16 : index
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
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1, %a2, %a3, %a4) = (0, 0, 0, 0) to (%b1, 17, %b2, 32) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        func.call @use(%a3) : (index) -> ()
        func.call @use(%a4) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f4(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 17 : index
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
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1, %a2, %a3, %a4) = (0, 0, 0, 0) to (%b1, 5, %b2, 3) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        func.call @use(%a3) : (index) -> ()
        func.call @use(%a4) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// %a3 gets split into blockIdx.y and threadIdx.x, so use(%a3) becomes use(blockIdx.y x blockDim.x + threadIdx.x)
//
// CHECK-LABEL:   func.func @f5(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 701 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           %[[VAL_9:.*]] = arith.divui %[[VAL_8]], %[[VAL_4]] : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_2]] : index
// CHECK:           %[[VAL_11:.*]] = "polygeist.gpu_error"() ({
// CHECK:             gpu.launch blocks(%[[VAL_12:.*]], %[[VAL_13:.*]], %[[VAL_14:.*]]) in (%[[VAL_15:.*]] = %[[VAL_0]], %[[VAL_16:.*]] = %[[VAL_10]], %[[VAL_17:.*]] = %[[VAL_2]]) threads(%[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]]) in (%[[VAL_21:.*]] = %[[VAL_4]], %[[VAL_22:.*]] = %[[VAL_7]], %[[VAL_23:.*]] = %[[VAL_6]]) {
// CHECK:               %[[VAL_24:.*]] = gpu.block_id  x
// CHECK:               %[[VAL_25:.*]] = gpu.block_id  y
// CHECK:               %[[VAL_26:.*]] = gpu.block_dim  x
// CHECK:               %[[VAL_27:.*]] = gpu.thread_id  x
// CHECK:               %[[VAL_28:.*]] = gpu.thread_id  y
// CHECK:               %[[VAL_29:.*]] = gpu.thread_id  z
// CHECK:               %[[VAL_30:.*]] = arith.muli %[[VAL_25]], %[[VAL_26]] : index
// CHECK:               %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_27]] : index
// CHECK:               %[[VAL_32:.*]] = arith.cmpi ult, %[[VAL_31]], %[[VAL_1]] : index
// CHECK:               scf.if %[[VAL_32]] {
// CHECK:                 func.call @use(%[[VAL_24]]) : (index) -> ()
// CHECK:                 func.call @use(%[[VAL_28]]) : (index) -> ()
// CHECK:                 func.call @use(%[[VAL_31]]) : (index) -> ()
// CHECK:                 func.call @use(%[[VAL_29]]) : (index) -> ()
// CHECK:               }
// CHECK:               gpu.terminator
// CHECK:             }
// CHECK:             "polygeist.polygeist_yield"() : () -> ()
// CHECK:           }) : () -> index
// CHECK:           %[[VAL_33:.*]] = arith.cmpi eq, %[[VAL_34:.*]], %[[VAL_3]] : index
// CHECK:           scf.if %[[VAL_33]] {
// CHECK:             %[[VAL_35:.*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : index
// CHECK:             %[[VAL_36:.*]] = arith.divui %[[VAL_35]], %[[VAL_5]] : index
// CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_36]], %[[VAL_2]] : index
// CHECK:             %[[VAL_38:.*]] = "polygeist.gpu_error"() ({
// CHECK:               gpu.launch blocks(%[[VAL_39:.*]], %[[VAL_40:.*]], %[[VAL_41:.*]]) in (%[[VAL_42:.*]] = %[[VAL_0]], %[[VAL_43:.*]] = %[[VAL_37]], %[[VAL_44:.*]] = %[[VAL_2]]) threads(%[[VAL_45:.*]], %[[VAL_46:.*]], %[[VAL_47:.*]]) in (%[[VAL_48:.*]] = %[[VAL_5]], %[[VAL_49:.*]] = %[[VAL_7]], %[[VAL_50:.*]] = %[[VAL_6]]) {
// CHECK:                 %[[VAL_51:.*]] = gpu.block_id  x
// CHECK:                 %[[VAL_52:.*]] = gpu.block_id  y
// CHECK:                 %[[VAL_53:.*]] = gpu.block_dim  x
// CHECK:                 %[[VAL_54:.*]] = gpu.thread_id  x
// CHECK:                 %[[VAL_55:.*]] = gpu.thread_id  y
// CHECK:                 %[[VAL_56:.*]] = gpu.thread_id  z
// CHECK:                 %[[VAL_57:.*]] = arith.muli %[[VAL_52]], %[[VAL_53]] : index
// CHECK:                 %[[VAL_58:.*]] = arith.addi %[[VAL_57]], %[[VAL_54]] : index
// CHECK:                 %[[VAL_59:.*]] = arith.cmpi ult, %[[VAL_58]], %[[VAL_1]] : index
// CHECK:                 scf.if %[[VAL_59]] {
// CHECK:                   func.call @use(%[[VAL_51]]) : (index) -> ()
// CHECK:                   func.call @use(%[[VAL_55]]) : (index) -> ()
// CHECK:                   func.call @use(%[[VAL_58]]) : (index) -> ()
// CHECK:                   func.call @use(%[[VAL_56]]) : (index) -> ()
// CHECK:                 }
// CHECK:                 gpu.terminator
// CHECK:               }
// CHECK:               "polygeist.polygeist_yield"() : () -> ()
// CHECK:             }) : () -> index
// CHECK:           }
// CHECK:           return

  func.func @f6(%b1: index, %b2: index) {
     %mc1 = arith.constant 1 : index
     %mc1024 = arith.constant 1024 : index
    %err = "polygeist.gpu_wrapper"(%mc1024, %mc1, %mc1) ({
      affine.parallel (%a1, %a2, %a3, %a4) = (0, 0, 0, 0) to (%b1, 1023, %b2, 1025) {
        func.call @use(%a1) : (index) -> ()
        func.call @use(%a2) : (index) -> ()
        func.call @use(%a3) : (index) -> ()
        func.call @use(%a4) : (index) -> ()
        affine.yield
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f6(
// CHECK-SAME:                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                  %[[VAL_1:.*]]: index) {
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1025 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 1023 : index
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

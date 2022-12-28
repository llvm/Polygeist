// RUN: polygeist-opt --lower-affine --canonicalize --convert-parallel-to-gpu1 --canonicalize %s | FileCheck %s

// TODO we need versions that need gpu cache to split wrapper (from particlefilter), lud or sradv1 had an alloca in wrapper case

module {
  func.func private @use(%arg0: index)
  func.func private @wow()
  func.func @foo(%i : i1, %129 : index) -> () {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %err = "polygeist.gpu_wrapper"(%c512, %c1, %c1) ({
      scf.if %i {
        %c0_111 = arith.constant 0 : index
        %c1_112 = arith.constant 1 : index
        scf.parallel (%arg7) = (%c0_111) to (%129) step (%c1_112) {
          func.call @use(%arg7) : (index) -> ()
          scf.yield
        }
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
}

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: i1,
// CHECK-SAME:                   %[[VAL_1:.*]]: index) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 701 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 512 : index
// CHECK:           scf.if %[[VAL_0]] {
// CHECK:             %[[VAL_5:.*]] = arith.subi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:             %[[VAL_6:.*]] = arith.divui %[[VAL_5]], %[[VAL_4]] : index
// CHECK:             %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_3]] : index
// CHECK:             %[[VAL_8:.*]] = "polygeist.gpu_error"() ({
// CHECK:               gpu.launch blocks(%[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]]) in (%[[VAL_12:.*]] = %[[VAL_7]], %[[VAL_13:.*]] = %[[VAL_3]], %[[VAL_14:.*]] = %[[VAL_3]]) threads(%[[VAL_15:.*]], %[[VAL_16:.*]], %[[VAL_17:.*]]) in (%[[VAL_18:.*]] = %[[VAL_4]], %[[VAL_19:.*]] = %[[VAL_3]], %[[VAL_20:.*]] = %[[VAL_3]]) {
// CHECK:                 %[[VAL_21:.*]] = gpu.block_id  x
// CHECK:                 %[[VAL_22:.*]] = gpu.block_dim  x
// CHECK:                 %[[VAL_23:.*]] = gpu.thread_id  x
// CHECK:                 %[[VAL_24:.*]] = arith.muli %[[VAL_21]], %[[VAL_22]] : index
// CHECK:                 %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_23]] : index
// CHECK:                 %[[VAL_26:.*]] = arith.cmpi ult, %[[VAL_25]], %[[VAL_1]] : index
// CHECK:                 scf.if %[[VAL_26]] {
// CHECK:                   func.call @use(%[[VAL_25]]) : (index) -> ()
// CHECK:                 }
// CHECK:                 gpu.terminator
// CHECK:               }
// CHECK:               "polygeist.polygeist_yield"() : () -> ()
// CHECK:             }) : () -> index
// CHECK:             %[[VAL_27:.*]] = arith.cmpi eq, %[[VAL_28:.*]], %[[VAL_2]] : index
// CHECK:             scf.if %[[VAL_27]] {
// CHECK:               %[[VAL_29:.*]] = arith.subi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:               %[[VAL_30:.*]] = arith.divui %[[VAL_29]], %[[VAL_4]] : index
// CHECK:               %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_3]] : index
// CHECK:               %[[VAL_32:.*]] = "polygeist.gpu_error"() ({
// CHECK:                 gpu.launch blocks(%[[VAL_33:.*]], %[[VAL_34:.*]], %[[VAL_35:.*]]) in (%[[VAL_36:.*]] = %[[VAL_31]], %[[VAL_37:.*]] = %[[VAL_3]], %[[VAL_38:.*]] = %[[VAL_3]]) threads(%[[VAL_39:.*]], %[[VAL_40:.*]], %[[VAL_41:.*]]) in (%[[VAL_42:.*]] = %[[VAL_4]], %[[VAL_43:.*]] = %[[VAL_3]], %[[VAL_44:.*]] = %[[VAL_3]]) {
// CHECK:                   %[[VAL_45:.*]] = gpu.block_id  x
// CHECK:                   %[[VAL_46:.*]] = gpu.block_dim  x
// CHECK:                   %[[VAL_47:.*]] = gpu.thread_id  x
// CHECK:                   %[[VAL_48:.*]] = arith.muli %[[VAL_45]], %[[VAL_46]] : index
// CHECK:                   %[[VAL_49:.*]] = arith.addi %[[VAL_48]], %[[VAL_47]] : index
// CHECK:                   %[[VAL_50:.*]] = arith.cmpi ult, %[[VAL_49]], %[[VAL_1]] : index
// CHECK:                   scf.if %[[VAL_50]] {
// CHECK:                     func.call @use(%[[VAL_49]]) : (index) -> ()
// CHECK:                   }
// CHECK:                   gpu.terminator
// CHECK:                 }
// CHECK:                 "polygeist.polygeist_yield"() : () -> ()
// CHECK:               }) : () -> index
// CHECK:             }
// CHECK:           }
// CHECK:           return

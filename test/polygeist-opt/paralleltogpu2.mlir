// RUN: polygeist-opt --lower-affine --canonicalize-polygeist --convert-parallel-to-gpu1 --canonicalize-polygeist %s | FileCheck %s

module {
  func.func @f7(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %aindex: index) {
     %mc1 = arith.constant 1 : index
     %mc512 = arith.constant 512 : index
     %cst3 = arith.constant 3.0 : f64
     %cst5 = arith.constant 5.0 : f64
    %err = "polygeist.gpu_wrapper"(%mc512, %mc1, %mc1) ({
      %c0_3 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg6) = (%c0_3) to (%aindex) step (%c1) {
        memref.store %cst3, %arg0[%arg6] : memref<?xf64>
        scf.yield
      }
      %c0_4 = arith.constant 0 : index
      memref.store %cst5, %arg1[%c0_4] : memref<?xf64>
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f7(
// CHECK-SAME:                  %[[VAL_0:[a-z0-9]+]]: memref<?xf64>,
// CHECK-SAME:                  %[[VAL_1:[a-z0-9]+]]: memref<?xf64>,
// CHECK-SAME:                  %[[VAL_2:.*]]: index) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 512 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 3.000000e+00 : f64
// CHECK:           %[[VAL_7:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_2]], %[[VAL_3]] : index
// CHECK:           %[[VAL_9:.*]] = arith.divui %[[VAL_8]], %[[VAL_5]] : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_3]] : index
// CHECK:           gpu.launch blocks(%[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]]) in (%[[VAL_14:.*]] = %[[VAL_10]], %[[VAL_15:.*]] = %[[VAL_3]], %[[VAL_16:.*]] = %[[VAL_3]]) threads(%[[VAL_17:.*]], %[[VAL_18:.*]], %[[VAL_19:.*]]) in (%[[VAL_20:.*]] = %[[VAL_5]], %[[VAL_21:.*]] = %[[VAL_3]], %[[VAL_22:.*]] = %[[VAL_3]]) {
// CHECK:             %[[VAL_23:.*]] = gpu.block_id  x
// CHECK:             %[[VAL_25:.*]] = gpu.thread_id  x
// CHECK:             %[[VAL_26:.*]] = arith.muli %[[VAL_23]], %[[VAL_5]] : index
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_25]] : index
// CHECK:             %[[VAL_28:.*]] = arith.cmpi ult, %[[VAL_27]], %[[VAL_2]] : index
// CHECK:             scf.if %[[VAL_28]] {
// CHECK:               memref.store %[[VAL_6]], %[[VAL_0]]{{\[}}%[[VAL_27]]] : memref<?xf64>
// CHECK:             }
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           gpu.launch blocks(%[[VAL_29:.*]], %[[VAL_30:.*]], %[[VAL_31:.*]]) in (%[[VAL_32:.*]] = %[[VAL_3]], %[[VAL_33:.*]] = %[[VAL_3]], %[[VAL_34:.*]] = %[[VAL_3]]) threads(%[[VAL_35:.*]], %[[VAL_36:.*]], %[[VAL_37:.*]]) in (%[[VAL_38:.*]] = %[[VAL_3]], %[[VAL_39:.*]] = %[[VAL_3]], %[[VAL_40:.*]] = %[[VAL_3]]) {
// CHECK:             memref.store %[[VAL_7]], %[[VAL_1]]{{\[}}%[[VAL_4]]] : memref<?xf64>
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return

  func.func @f8(%arg0: memref<?x100xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %aindex: index) {
     %mc1 = arith.constant 1 : index
     %mc512 = arith.constant 512 : index
     %cst3 = arith.constant 3.0 : f64
     %cst4 = arith.constant 4.0 : f64
     %cst5 = arith.constant 5.0 : f64
    %err = "polygeist.gpu_wrapper"(%mc512, %mc1, %mc1) ({
      %c0_3 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg6) = (%c0_3) to (%aindex) step (%c1) {
        memref.store %cst3, %arg2[%arg6] : memref<?xf64>
        scf.parallel (%arg7) = (%c0_3) to (%aindex) step (%c1) {
          memref.store %cst4, %arg0[%arg6, %arg7] : memref<?x100xf64>
          scf.yield
        }
      }
      %c0_4 = arith.constant 0 : index
      memref.store %cst5, %arg1[%c0_4] : memref<?xf64>
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> index
    return
  }
// CHECK-LABEL:   func.func @f8(
// CHECK-SAME:                  %[[VAL_0:[a-z0-9]+]]: memref<?x100xf64>,
// CHECK-SAME:                  %[[VAL_1:[a-z0-9]+]]: memref<?xf64>,
// CHECK-SAME:                  %[[VAL_2:[a-z0-9]+]]: memref<?xf64>,
// CHECK-SAME:                  %[[VAL_3:.*]]: index) {
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 3.000000e+00 : f64
// CHECK:           %[[VAL_7:.*]] = arith.constant 4.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:           gpu.launch blocks(%[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]]) in (%[[VAL_12:.*]] = %[[VAL_3]], %[[VAL_13:.*]] = %[[VAL_4]], %[[VAL_14:.*]] = %[[VAL_4]]) threads(%[[VAL_15:.*]], %[[VAL_16:.*]], %[[VAL_17:.*]]) in (%[[VAL_18:.*]] = %[[VAL_3]], %[[VAL_19:.*]] = %[[VAL_4]], %[[VAL_20:.*]] = %[[VAL_4]]) {
// CHECK:             %[[VAL_21:.*]] = gpu.block_id  x
// CHECK:             %[[VAL_22:.*]] = gpu.thread_id  x
// CHECK:             %[[VAL_23:.*]] = arith.cmpi eq, %[[VAL_22]], %[[VAL_5]] : index
// CHECK:             scf.if %[[VAL_23]] {
// CHECK:               memref.store %[[VAL_6]], %[[VAL_2]]{{\[}}%[[VAL_21]]] : memref<?xf64>
// CHECK:             }
// CHECK:             gpu.barrier
// CHECK:             memref.store %[[VAL_7]], %[[VAL_0]]{{\[}}%[[VAL_21]], %[[VAL_22]]] : memref<?x100xf64>
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           gpu.launch blocks(%[[VAL_24:.*]], %[[VAL_25:.*]], %[[VAL_26:.*]]) in (%[[VAL_27:.*]] = %[[VAL_4]], %[[VAL_28:.*]] = %[[VAL_4]], %[[VAL_29:.*]] = %[[VAL_4]]) threads(%[[VAL_30:.*]], %[[VAL_31:.*]], %[[VAL_32:.*]]) in (%[[VAL_33:.*]] = %[[VAL_4]], %[[VAL_34:.*]] = %[[VAL_4]], %[[VAL_35:.*]] = %[[VAL_4]]) {
// CHECK:             memref.store %[[VAL_8]], %[[VAL_1]]{{\[}}%[[VAL_5]]] : memref<?xf64>
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return

}

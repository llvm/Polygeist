// RUN: polygeist-opt --lower-affine --canonicalize --scf-parallel-loop-unroll="unrollFactor=3" --cse %s | FileCheck %s

module {
  func.func private @use0(%arg0: index)
  func.func private @use1(%arg0: index)
  func.func private @use2(%arg0: index)
  func.func private @wow()
  func.func @f1() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    affine.parallel (%a0, %a1) = (0, 0) to (300, 30) {
      func.call @use0(%a0) : (index) -> ()
      %b = arith.addi %a0,  %mc1 : index
      func.call @use1(%b) : (index) -> ()
      func.call @use2(%a1) : (index) -> ()
      affine.yield
    }
    return
  }
}
// CHECK-LABEL:   func.func @f1() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:           scf.parallel (%[[VAL_5:.*]], %[[VAL_6:.*]]) = (%[[VAL_1]], %[[VAL_1]]) to (%[[VAL_4]], %[[VAL_2]]) step (%[[VAL_0]], %[[VAL_0]]) {
// CHECK:             %[[VAL_7:.*]] = arith.muli %[[VAL_5]], %[[VAL_3]] : index
// CHECK:             %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_1]] : index
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_0]] : index
// CHECK:             %[[VAL_10:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_7]], %[[VAL_10]] : index
// CHECK:             func.call @use0(%[[VAL_8]]) : (index) -> ()
// CHECK:             func.call @use0(%[[VAL_9]]) : (index) -> ()
// CHECK:             func.call @use0(%[[VAL_11]]) : (index) -> ()
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_8]], %[[VAL_0]] : index
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_9]], %[[VAL_0]] : index
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_11]], %[[VAL_0]] : index
// CHECK:             func.call @use1(%[[VAL_12]]) : (index) -> ()
// CHECK:             func.call @use1(%[[VAL_13]]) : (index) -> ()
// CHECK:             func.call @use1(%[[VAL_14]]) : (index) -> ()
// CHECK:             func.call @use2(%[[VAL_6]]) : (index) -> ()
// CHECK:             func.call @use2(%[[VAL_6]]) : (index) -> ()
// CHECK:             func.call @use2(%[[VAL_6]]) : (index) -> ()
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }


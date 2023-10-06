// RUN: polygeist-opt --lower-affine --canonicalize-polygeist --scf-parallel-loop-unroll="unrollFactor=3" --cse %s | FileCheck %s

module {
  func.func private @use0(%arg0: index)
  func.func private @use1(%arg0: index)
  func.func private @use2(%arg0: index)
  func.func private @get() -> i1
  func.func private @wow()
  func.func @f00(%upperBound: index) {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    affine.parallel (%a0, %a1) = (0, 0) to (%upperBound, 30) {
      func.call @use0(%a0) : (index) -> ()
      affine.yield
    }
    return
  }
  func.func @f0() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    affine.parallel (%a0, %a1) = (0, 0) to (299, 30) {
      func.call @use0(%a0) : (index) -> ()
      affine.yield
    }
    return
  }
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
  func.func @f2(%cond: i1) {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    affine.parallel (%a0, %a1) = (0, 0) to (300, 30) {
      affine.for %a2 = 0 to 30 {
        func.call @use0(%a0) : (index) -> ()
        "polygeist.barrier"(%a0) : (index) -> ()
        func.call @use1(%a0) : (index) -> ()
      }
      "polygeist.barrier"(%a0) : (index) -> ()
      scf.if %cond {
        func.call @use0(%a0) : (index) -> ()
        "polygeist.barrier"(%a0) : (index) -> ()
        func.call @use0(%a0) : (index) -> ()
      }
      "polygeist.barrier"(%a0) : (index) -> ()
      scf.if %cond {
        func.call @use0(%a0) : (index) -> ()
        "polygeist.barrier"(%a0) : (index) -> ()
        func.call @use1(%a0) : (index) -> ()
      } else {
        func.call @use2(%a0) : (index) -> ()
      }
      "polygeist.barrier"(%a0) : (index) -> ()
      affine.parallel (%a2, %a3) = (0, 0) to (300, 30) {
        func.call @use0(%a2) : (index) -> ()
        affine.yield
      }
      affine.yield
    }
    return
  }
  func.func @f3() {
    %mc0 = arith.constant 0 : index
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    affine.parallel (%a0, %a1) = (0, 0) to (300, 30) {
      scf.for %a2 = %mc0 to %a0 step %mc1 {
        func.call @use0(%a0) : (index) -> ()
      }
      affine.yield
    }
    return
  }
  func.func @f4() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    affine.parallel (%a0, %a1) = (0, 0) to (300, 30) {
      %cond = func.call @get() : () -> i1
      scf.if %cond {
        func.call @use0(%a0) : (index) -> ()
        scf.yield
      }
      affine.yield
    }
    return
  }
  func.func @f5() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    affine.parallel (%a0, %a1) = (0, 0) to (300, 30) {
      func.call @use0(%a0) : (index) -> ()
      affine.parallel (%a2, %a3) = (0, 0) to (300, 30) {
        func.call @use0(%a2) : (index) -> ()
        affine.yield
      }
      affine.yield
    }
    return
  }
  func.func @f6() {
    %mc1 = arith.constant 1 : index
    %mc1024 = arith.constant 1024 : index
    affine.parallel (%a0, %a1) = (0, 0) to (300, 30) {
      %cond = func.call @get() : () -> i1
      scf.if %cond {
        func.call @use0(%a0) : (index) -> ()
        "polygeist.barrier"(%a0) : (index) -> ()
        func.call @use0(%a0) : (index) -> ()
        scf.yield
      }
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

// CHECK-LABEL:   func.func @f2(
// CHECK-SAME:                  %[[VAL_0:.*]]: i1) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 300 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 100 : index
// CHECK:           scf.parallel (%[[VAL_7:.*]], %[[VAL_8:.*]]) = (%[[VAL_2]], %[[VAL_2]]) to (%[[VAL_6]], %[[VAL_4]]) step (%[[VAL_1]], %[[VAL_1]]) {
// CHECK:             %[[VAL_9:.*]] = arith.muli %[[VAL_7]], %[[VAL_5]] : index
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_2]] : index
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : index
// CHECK:             %[[VAL_12:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_9]], %[[VAL_12]] : index
// CHECK:             scf.for %[[VAL_14:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_1]] {
// CHECK:               func.call @use0(%[[VAL_10]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_11]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_13]]) : (index) -> ()
// CHECK:               "polygeist.barrier"(%[[VAL_7]]) : (index) -> ()
// CHECK:               func.call @use1(%[[VAL_10]]) : (index) -> ()
// CHECK:               func.call @use1(%[[VAL_11]]) : (index) -> ()
// CHECK:               func.call @use1(%[[VAL_13]]) : (index) -> ()
// CHECK:             }
// CHECK:             "polygeist.barrier"(%[[VAL_7]]) : (index) -> ()
// CHECK:             scf.if %[[VAL_0]] {
// CHECK:               func.call @use0(%[[VAL_10]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_11]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_13]]) : (index) -> ()
// CHECK:               "polygeist.barrier"(%[[VAL_7]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_10]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_11]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_13]]) : (index) -> ()
// CHECK:             }
// CHECK:             "polygeist.barrier"(%[[VAL_7]]) : (index) -> ()
// CHECK:             scf.if %[[VAL_0]] {
// CHECK:               func.call @use0(%[[VAL_10]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_11]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_13]]) : (index) -> ()
// CHECK:               "polygeist.barrier"(%[[VAL_7]]) : (index) -> ()
// CHECK:               func.call @use1(%[[VAL_10]]) : (index) -> ()
// CHECK:               func.call @use1(%[[VAL_11]]) : (index) -> ()
// CHECK:               func.call @use1(%[[VAL_13]]) : (index) -> ()
// CHECK:             } else {
// CHECK:               func.call @use2(%[[VAL_10]]) : (index) -> ()
// CHECK:               func.call @use2(%[[VAL_11]]) : (index) -> ()
// CHECK:               func.call @use2(%[[VAL_13]]) : (index) -> ()
// CHECK:             }
// CHECK:             "polygeist.barrier"(%[[VAL_7]]) : (index) -> ()
// CHECK:             scf.parallel (%[[VAL_15:.*]], %[[VAL_16:.*]]) = (%[[VAL_2]], %[[VAL_2]]) to (%[[VAL_3]], %[[VAL_4]]) step (%[[VAL_1]], %[[VAL_1]]) {
// CHECK:               func.call @use0(%[[VAL_15]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_15]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_15]]) : (index) -> ()
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }

// CHECK-LABEL:   func.func @f3() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:           scf.parallel (%[[VAL_5:.*]], %[[VAL_6:.*]]) = (%[[VAL_0]], %[[VAL_0]]) to (%[[VAL_4]], %[[VAL_2]]) step (%[[VAL_1]], %[[VAL_1]]) {
// CHECK:             %[[VAL_7:.*]] = arith.muli %[[VAL_5]], %[[VAL_3]] : index
// CHECK:             %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_0]] : index
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_1]] : index
// CHECK:             %[[VAL_10:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_7]], %[[VAL_10]] : index
// CHECK:             scf.for %[[VAL_12:.*]] = %[[VAL_0]] to %[[VAL_8]] step %[[VAL_1]] {
// CHECK:               func.call @use0(%[[VAL_8]]) : (index) -> ()
// CHECK:             }
// CHECK:             scf.for %[[VAL_13:.*]] = %[[VAL_0]] to %[[VAL_9]] step %[[VAL_1]] {
// CHECK:               func.call @use0(%[[VAL_9]]) : (index) -> ()
// CHECK:             }
// CHECK:             scf.for %[[VAL_14:.*]] = %[[VAL_0]] to %[[VAL_11]] step %[[VAL_1]] {
// CHECK:               func.call @use0(%[[VAL_11]]) : (index) -> ()
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }

// CHECK-LABEL:   func.func @f4() {
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
// CHECK:             %[[VAL_12:.*]] = func.call @get() : () -> i1
// CHECK:             %[[VAL_13:.*]] = func.call @get() : () -> i1
// CHECK:             %[[VAL_14:.*]] = func.call @get() : () -> i1
// CHECK:             scf.if %[[VAL_12]] {
// CHECK:               func.call @use0(%[[VAL_8]]) : (index) -> ()
// CHECK:             }
// CHECK:             scf.if %[[VAL_13]] {
// CHECK:               func.call @use0(%[[VAL_9]]) : (index) -> ()
// CHECK:             }
// CHECK:             scf.if %[[VAL_14]] {
// CHECK:               func.call @use0(%[[VAL_11]]) : (index) -> ()
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }

// CHECK-LABEL:   func.func @f5() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 300 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 100 : index
// CHECK:           scf.parallel (%[[VAL_6:.*]], %[[VAL_7:.*]]) = (%[[VAL_1]], %[[VAL_1]]) to (%[[VAL_5]], %[[VAL_3]]) step (%[[VAL_0]], %[[VAL_0]]) {
// CHECK:             %[[VAL_8:.*]] = arith.muli %[[VAL_6]], %[[VAL_4]] : index
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_1]] : index
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_0]] : index
// CHECK:             %[[VAL_11:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_8]], %[[VAL_11]] : index
// CHECK:             func.call @use0(%[[VAL_9]]) : (index) -> ()
// CHECK:             func.call @use0(%[[VAL_10]]) : (index) -> ()
// CHECK:             func.call @use0(%[[VAL_12]]) : (index) -> ()
// CHECK:             scf.parallel (%[[VAL_13:.*]], %[[VAL_14:.*]]) = (%[[VAL_1]], %[[VAL_1]]) to (%[[VAL_2]], %[[VAL_3]]) step (%[[VAL_0]], %[[VAL_0]]) {
// CHECK:               func.call @use0(%[[VAL_13]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_13]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_13]]) : (index) -> ()
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }

// CHECK-LABEL:   func.func @f6() {
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
// CHECK:             %[[VAL_12:.*]] = func.call @get() : () -> i1
// CHECK:             %[[VAL_13:.*]] = func.call @get() : () -> i1
// CHECK:             %[[VAL_14:.*]] = func.call @get() : () -> i1
// CHECK:             scf.if %[[VAL_12]] {
// CHECK:               func.call @use0(%[[VAL_8]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_9]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_11]]) : (index) -> ()
// CHECK:               "polygeist.barrier"(%[[VAL_5]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_8]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_9]]) : (index) -> ()
// CHECK:               func.call @use0(%[[VAL_11]]) : (index) -> ()
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }

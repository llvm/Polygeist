// RUN: polygeist-opt --canonicalize-polygeist %s | FileCheck %s

#set0 = affine_set<(d0, d1) : (d0 + d1 * 512 == 0)>

module {
  func.func private @use(index)
  func.func @k(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32
    affine.parallel (%arg7) = (0) to (symbol(%636)) {
      %706 = arith.index_cast %arg7 : index to i32
      %707 = arith.muli %706, %c512_i32 : i32
      affine.parallel (%arg8) = (0) to (512) {
        %708 = arith.index_cast %arg8 : index to i32
        %709 = arith.addi %707, %708 : i32
        %ifres = affine.if #set0(%arg8, %arg7) -> f64 {
          %712 = arith.sitofp %709 : i32 to f64
          func.call @use(%arg7) : (index) -> ()
          affine.yield %712 : f64
        } else {
          %712 = arith.sitofp %708 : i32 to f64
          func.call @use(%arg8) : (index) -> ()
          affine.yield %712 : f64
        }
        affine.if #set0(%arg8, %arg7) {
          func.call @use(%arg7) : (index) -> ()
        } else {
          func.call @use(%arg8) : (index) -> ()
        }
        affine.store %ifres, %603[0] : memref<?xf64>
      }
    }
    return
  }
// CHECK-LABEL:   func.func @k(
// CHECK-SAME:                 %[[VAL_0:.*]]: index,
// CHECK-SAME:                 %[[VAL_1:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 512 : i32
// CHECK:           affine.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (0, 0) to (symbol(%[[VAL_0]]), 512) {
// CHECK:             %[[VAL_5:.*]] = arith.index_cast %[[VAL_3]] : index to i32
// CHECK:             %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_7:.*]] = arith.index_cast %[[VAL_4]] : index to i32
// CHECK:             %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_9:.*]] = affine.if #set(%[[VAL_4]], %[[VAL_3]]) -> f64 {
// CHECK:               %[[VAL_10:.*]] = arith.sitofp %[[VAL_8]] : i32 to f64
// CHECK:               func.call @use(%[[VAL_3]]) : (index) -> ()
// CHECK:               func.call @use(%[[VAL_3]]) : (index) -> ()
// CHECK:               affine.yield %[[VAL_10]] : f64
// CHECK:             } else {
// CHECK:               %[[VAL_11:.*]] = arith.sitofp %[[VAL_7]] : i32 to f64
// CHECK:               func.call @use(%[[VAL_4]]) : (index) -> ()
// CHECK:               func.call @use(%[[VAL_4]]) : (index) -> ()
// CHECK:               affine.yield %[[VAL_11]] : f64
// CHECK:             }
// CHECK:             affine.store %[[VAL_12:.*]], %[[VAL_1]][0] : memref<?xf64>
// CHECK:           }
// CHECK:           return

  func.func @h(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32
    affine.parallel (%arg7) = (0) to (symbol(%636)) {
      %706 = arith.index_cast %arg7 : index to i32
      %707 = arith.muli %706, %c512_i32 : i32
      affine.parallel (%arg8) = (0) to (512) {
        %708 = arith.index_cast %arg8 : index to i32
        %709 = arith.addi %707, %708 : i32
        %ifres = affine.if #set0(%arg8, %arg7) -> f64 {
          %712 = arith.sitofp %709 : i32 to f64
          func.call @use(%arg7) : (index) -> ()
          affine.yield %712 : f64
        } else {
          %712 = arith.sitofp %708 : i32 to f64
          func.call @use(%arg8) : (index) -> ()
          affine.yield %712 : f64
        }
        affine.if #set0(%arg8, %arg7) {
          func.call @use(%arg7) : (index) -> ()
        }
        affine.store %ifres, %603[0] : memref<?xf64>
      }
    }
    return
  }
// CHECK-LABEL:   func.func @h(
// CHECK-SAME:                 %[[VAL_0:.*]]: index,
// CHECK-SAME:                 %[[VAL_1:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 512 : i32
// CHECK:           affine.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (0, 0) to (symbol(%[[VAL_0]]), 512) {
// CHECK:             %[[VAL_5:.*]] = arith.index_cast %[[VAL_3]] : index to i32
// CHECK:             %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_7:.*]] = arith.index_cast %[[VAL_4]] : index to i32
// CHECK:             %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_9:.*]] = affine.if #set(%[[VAL_4]], %[[VAL_3]]) -> f64 {
// CHECK:               %[[VAL_10:.*]] = arith.sitofp %[[VAL_8]] : i32 to f64
// CHECK:               func.call @use(%[[VAL_3]]) : (index) -> ()
// CHECK:               func.call @use(%[[VAL_3]]) : (index) -> ()
// CHECK:               affine.yield %[[VAL_10]] : f64
// CHECK:             } else {
// CHECK:               %[[VAL_11:.*]] = arith.sitofp %[[VAL_7]] : i32 to f64
// CHECK:               func.call @use(%[[VAL_4]]) : (index) -> ()
// CHECK:               affine.yield %[[VAL_11]] : f64
// CHECK:             }
// CHECK:             affine.store %[[VAL_12:.*]], %[[VAL_1]][0] : memref<?xf64>
// CHECK:           }
// CHECK:           return

  func.func @g(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32

    affine.parallel (%arg7) = (0) to (symbol(%636)) {
      %706 = arith.index_cast %arg7 : index to i32
      %707 = arith.muli %706, %c512_i32 : i32
      affine.parallel (%arg8) = (0) to (512) {
        %708 = arith.index_cast %arg8 : index to i32
        %709 = arith.addi %707, %708 : i32
        affine.if #set0(%arg8, %arg7) {
          func.call @use(%arg7) : (index) -> ()
        }
        %ifres = affine.if #set0(%arg8, %arg7) -> f64 {
          %712 = arith.sitofp %709 : i32 to f64
          func.call @use(%arg7) : (index) -> ()
          affine.yield %712 : f64
        } else {
          %712 = arith.sitofp %708 : i32 to f64
          func.call @use(%arg8) : (index) -> ()
          affine.yield %712 : f64
        }
        affine.store %ifres, %603[0] : memref<?xf64>
      }
    }
    return
  }
// CHECK-LABEL:   func.func @g(
// CHECK-SAME:                 %[[VAL_0:.*]]: index,
// CHECK-SAME:                 %[[VAL_1:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 512 : i32
// CHECK:           affine.if #set1(){{\[}}%[[VAL_0]]] {
// CHECK:             func.call @use(%[[VAL_2]]) : (index) -> ()
// CHECK:           }
// CHECK:           affine.parallel (%[[VAL_4:.*]], %[[VAL_5:.*]]) = (0, 0) to (symbol(%[[VAL_0]]), 512) {
// CHECK:             %[[VAL_6:.*]] = arith.index_cast %[[VAL_4]] : index to i32
// CHECK:             %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_8:.*]] = arith.index_cast %[[VAL_5]] : index to i32
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:             %[[VAL_10:.*]] = affine.if #set(%[[VAL_5]], %[[VAL_4]]) -> f64 {
// CHECK:               %[[VAL_11:.*]] = arith.sitofp %[[VAL_9]] : i32 to f64
// CHECK:               func.call @use(%[[VAL_4]]) : (index) -> ()
// CHECK:               affine.yield %[[VAL_11]] : f64
// CHECK:             } else {
// CHECK:               %[[VAL_12:.*]] = arith.sitofp %[[VAL_8]] : i32 to f64
// CHECK:               func.call @use(%[[VAL_5]]) : (index) -> ()
// CHECK:               affine.yield %[[VAL_12]] : f64
// CHECK:             }
// CHECK:             affine.store %[[VAL_13:.*]], %[[VAL_1]][0] : memref<?xf64>
// CHECK:           }
// CHECK:           return

  func.func @f(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32

    affine.parallel (%arg7) = (0) to (symbol(%636)) {
      %706 = arith.index_cast %arg7 : index to i32
      %707 = arith.muli %706, %c512_i32 : i32
      affine.parallel (%arg8) = (0) to (512) {
        %708 = arith.index_cast %arg8 : index to i32
        %709 = arith.addi %707, %708 : i32
        affine.if #set0(%arg8, %arg7) {
          func.call @use(%arg7) : (index) -> ()
        } else {
          func.call @use(%arg8) : (index) -> ()
        }
        %ifres = affine.if #set0(%arg8, %arg7) -> f64 {
          %712 = arith.sitofp %709 : i32 to f64
          func.call @use(%arg7) : (index) -> ()
          affine.yield %712 : f64
        } else {
          %712 = arith.sitofp %708 : i32 to f64
          func.call @use(%arg8) : (index) -> ()
          affine.yield %712 : f64
        }
        affine.store %ifres, %603[0] : memref<?xf64>
      }
    }
    return
  }
// CHECK-LABEL:   func.func @f(
// CHECK-SAME:                 %[[VAL_0:.*]]: index,
// CHECK-SAME:                 %[[VAL_1:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 512 : i32
// CHECK:           affine.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (0, 0) to (symbol(%[[VAL_0]]), 512) {
// CHECK:             %[[VAL_5:.*]] = arith.index_cast %[[VAL_3]] : index to i32
// CHECK:             %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_7:.*]] = arith.index_cast %[[VAL_4]] : index to i32
// CHECK:             %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_9:.*]] = affine.if #set(%[[VAL_4]], %[[VAL_3]]) -> f64 {
// CHECK:               func.call @use(%[[VAL_3]]) : (index) -> ()
// CHECK:               %[[VAL_10:.*]] = arith.sitofp %[[VAL_8]] : i32 to f64
// CHECK:               func.call @use(%[[VAL_3]]) : (index) -> ()
// CHECK:               affine.yield %[[VAL_10]] : f64
// CHECK:             } else {
// CHECK:               func.call @use(%[[VAL_4]]) : (index) -> ()
// CHECK:               %[[VAL_11:.*]] = arith.sitofp %[[VAL_7]] : i32 to f64
// CHECK:               func.call @use(%[[VAL_4]]) : (index) -> ()
// CHECK:               affine.yield %[[VAL_11]] : f64
// CHECK:             }
// CHECK:             affine.store %[[VAL_12:.*]], %[[VAL_1]][0] : memref<?xf64>
// CHECK:           }
// CHECK:           return


}

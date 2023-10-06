// RUN: polygeist-opt --canonicalize-polygeist --split-input-file %s | FileCheck %s
module {
  func.func @foo(%arg0: i1) -> i32 {
    %c512_i32 = arith.constant 512 : i32
    %c16_i32 = arith.constant 16 : i32
    %1 = arith.select %arg0, %c512_i32, %c16_i32 : i32
    return %1 : i32
  }
  func.func @foo1(%arg0: i1, %c1_i32: i32, %c2_i32: i32) -> i64 {
    %c1_i64 = arith.extui %c1_i32 : i32 to i64
    %c2_i64 = arith.extui %c2_i32 : i32 to i64
    %1 = arith.select %arg0, %c1_i64, %c2_i64 : i64
    return %1 : i64
  }
  func.func @foo2(%arg0: i1, %c1_i64: i64, %c2_i32: i32) -> i64 {
    %c2_i64 = arith.extui %c2_i32 : i32 to i64
    %1 = arith.select %arg0, %c1_i64, %c2_i64 : i64
    return %1 : i64
  }
  func.func @foo3(%arg0: i1, %c1_i64: i1, %i8: i8) -> i32 {
    %c255 = arith.constant 255 : i32
    %i32 = arith.extui %i8 : i8 to i32
    %1 = arith.select %arg0, %c255, %i32 : i32
    return %1 : i32
  }
  func.func @foo3.1() -> i8 {
    %c255 = arith.constant 255 : i32
    %i8 = arith.trunci %c255 : i32 to i8
    return %i8 : i8
  }
  func.func @foo4(%arg0: i1, %c1_i64: i1, %i8: i8) -> i32 {
    %c256 = arith.constant 256 : i32
    %i32 = arith.extui %i8 : i8 to i32
    %1 = arith.select %arg0, %c256, %i32 : i32
    return %1 : i32
  }

}

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: i1) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 512 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 16 : i32
// CHECK:           %[[VAL_3:.*]] = arith.select %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @foo1(
// CHECK-SAME:                    %[[VAL_0:.*]]: i1,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32,
// CHECK-SAME:                    %[[VAL_2:.*]]: i32) -> i64 {
// CHECK:           %[[VAL_3:.*]] = arith.select %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_4:.*]] = arith.extui %[[VAL_3]] : i32 to i64
// CHECK:           return %[[VAL_4]] : i64
// CHECK:         }

// CHECK-LABEL:   func.func @foo2(
// CHECK-SAME:                    %[[VAL_0:.*]]: i1,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64,
// CHECK-SAME:                    %[[VAL_2:.*]]: i32) -> i64 {
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_4:.*]] = arith.select %[[VAL_0]], %[[VAL_1]], %[[VAL_3]] : i64
// CHECK:           return %[[VAL_4]] : i64
// CHECK:         }


// NOTE (i8) 255 = (i8) -1

// CHECK-LABEL:   func.func @foo3(
// CHECK-SAME:                    %[[VAL_0:[a-zA-Z0-9]*]]: i1,
// CHECK-SAME:                    %[[VAL_1:[a-zA-Z0-9]*]]: i1,
// CHECK-SAME:                    %[[VAL_2:.*]]: i8) -> i32 {
// CHECK:           %[[VAL_3:.*]] = arith.constant -1 : i8
// CHECK:           %[[VAL_4:.*]] = arith.select %[[VAL_0]], %[[VAL_3]], %[[VAL_2]] : i8
// CHECK:           %[[VAL_5:.*]] = arith.extui %[[VAL_4]] : i8 to i32
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @foo3.1() -> i8 {
// CHECK:           %[[VAL_0:.*]] = arith.constant -1 : i8
// CHECK:           return %[[VAL_0]] : i8
// CHECK:         }

// CHECK-LABEL:   func.func @foo4(
// CHECK-SAME:                    %[[VAL_0:[a-zA-Z0-9]*]]: i1,
// CHECK-SAME:                    %[[VAL_1:[a-zA-Z0-9]*]]: i1,
// CHECK-SAME:                    %[[VAL_2:.*]]: i8) -> i32 {
// CHECK:           %[[VAL_3:.*]] = arith.constant 256 : i32
// CHECK:           %[[VAL_4:.*]] = arith.extui %[[VAL_2]] : i8 to i32
// CHECK:           %[[VAL_5:.*]] = arith.select %[[VAL_0]], %[[VAL_3]], %[[VAL_4]] : i32
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }


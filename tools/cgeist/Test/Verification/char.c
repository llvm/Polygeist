// RUN: cgeist %s --function=* -S | FileCheck %s

char add_one(char x) {
  char one = 1;
  return x + one;
}

// CHECK-LABEL:   func.func @add_one(
// CHECK-SAME:     %[[x:.*]]: i8) -> i8
// CHECK-NEXT:       %[[c1:.*]] = arith.constant 1
// CHECK-NEXT:       %[[res:.*]] = arith.addi %[[x]], %[[c1]] : i8
// CHECK-NEXT:       return %[[res]] : i8
// CHECK:         }

// RUN: cgeist %s --function=* -S | FileCheck %s

typedef unsigned int ui32;
typedef unsigned long long ui64;

// CHECK:      func.func @shift(%[[Varg0:.*]]: i64, %[[Varg1:.*]]: i64) -> i64
// CHECK-NEXT:   %[[V0:.*]] = arith.shrui %[[Varg0]], %[[Varg1]] : i64
// CHECK-NEXT:   return %[[V0]] : i64
// CHECK-NEXT: }

ui64 shift(ui64 input, ui64 shift) {
  return input >> shift;
}

// CHECK:      func.func @ge(%[[Varg0:.*]]: i64, %[[Varg1:.*]]: i64) -> i32
// CHECK-NEXT:   %[[V0:.*]] = arith.cmpi ugt, %[[Varg0]], %[[Varg1]] : i64
// CHECK-NEXT:   %[[V1:.*]] = arith.extui %[[V0]] : i1 to i32
// CHECK-NEXT:   return %[[V1]] : i32
// CHECK-NEXT: }

int ge(ui64 input, ui64 v) {
  return input > v;
}

// CHECK-NEXT:   func.func @ret(%[[Varg0:.*]]: i32) -> i64
// CHECK-NEXT:     %[[V0:.*]] = arith.extui %[[Varg0]] : i32 to i64
// CHECK-NEXT:     return %[[V0]] : i64
// CHECK-NEXT:   }

ui64 ret(ui32 input) {
  return input;
}

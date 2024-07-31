// RUN: cgeist %s %stdinclude --function=* -S | FileCheck %s

// CHECK:      func.func @do_ctzs(%[[ARG:[A-Za-z0-9_]*]]: i16) -> i32
// CHECK-NEXT:   %[[VAL_0:[A-Za-z0-9_]*]] = math.cttz %[[ARG]] : i16
// CHECK-NEXT:   %[[VAL_1:[A-Za-z0-9_]*]] = arith.extui %[[VAL_0]] : i16 to i32
// CHECK-NEXT:   return %[[VAL_1]] : i32
// CHECK-NEXT: }

int do_ctzs(short int i) {
  return __builtin_ctzs(i);
}

// CHECK:      func.func @do_ctz(%[[ARG:[A-Za-z0-9_]*]]: i32) -> i32
// CHECK-NEXT:   %[[VAL:[A-Za-z0-9_]*]] = math.cttz %[[ARG]] : i32
// CHECK-NEXT:   return %[[VAL]] : i32
// CHECK-NEXT: }

int do_ctz(int i) {
  return __builtin_ctz(i);
}

// CHECK:      func.func @do_ctzl(%[[ARG:[A-Za-z0-9_]*]]: i64) -> i32
// CHECK-NEXT:   %[[VAL_0:[A-Za-z0-9_]*]] = math.cttz %[[ARG]] : i64
// CHECK-NEXT:   %[[VAL_1:[A-Za-z0-9_]*]] = arith.trunci %[[VAL_0]] : i64 to i32
// CHECK-NEXT:   return %[[VAL_1]] : i32
// CHECK-NEXT: }

int do_ctzl(unsigned long i) {
  return __builtin_ctzl(i);
}

// CHECK:      func.func @do_ctzll(%[[ARG:[A-Za-z0-9_]*]]: i64) -> i32
// CHECK-NEXT:   %[[VAL_0:[A-Za-z0-9_]*]] = math.cttz %[[ARG]] : i64
// CHECK-NEXT:   %[[VAL_1:[A-Za-z0-9_]*]] = arith.trunci %[[VAL_0]] : i64 to i32
// CHECK-NEXT:   return %[[VAL_1]] : i32
// CHECK-NEXT: }

int do_ctzll(unsigned long long i) {
  return __builtin_ctzl(i);
}

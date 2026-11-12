// RUN: cgeist %s --function=* -S | FileCheck %s
typedef unsigned char byte_t;
typedef signed char signed_byte_t;
int widen_unsigned(byte_t x) { return x; }
int widen_signed(signed_byte_t x) { return x; }
int unsigned_index(const int *p, byte_t x) { return p[x]; }
// CHECK-LABEL: func.func @widen_unsigned
// CHECK: arith.extui
// CHECK-LABEL: func.func @widen_signed
// CHECK: arith.extsi
// CHECK-LABEL: func.func @unsigned_index
// CHECK: arith.index_castui

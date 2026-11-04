// __builtin_bit_cast between scalars of one width is an arith.bitcast (the
// bit-exact ap_fixed shim decodes a double's IEEE representation with it);
// on a constant it folds. Enabled by --scalar-wrapper-abi.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @_Z4bitsd(%arg0: f64) -> i64
// CHECK: arith.bitcast %arg0 : f64 to i64
unsigned long long bits(double d) { return __builtin_bit_cast(unsigned long long, d); }

// CHECK-LABEL: func.func @_Z4backy(%arg0: i64) -> f64
// CHECK: arith.bitcast %arg0 : i64 to f64
double back(unsigned long long b) { return __builtin_bit_cast(double, b); }

// CHECK-LABEL: func.func @_Z8exponentv() -> i32
// CHECK: return %c1023_i32 : i32
int exponent() {
  unsigned long long b = __builtin_bit_cast(unsigned long long, 1.0);
  return (int)((b >> 52) & 0x7ff);
}

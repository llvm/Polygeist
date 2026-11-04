// sizeof of a scalar wrapper is the size of its scalar (an i28 occupies 4
// bytes, like clang's sizeof(_BitInt(28))), and arrays of wrappers copy
// element by element.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s

struct W28 {
  _BitInt(28) v;
};
struct W16 {
  short v;
};

// CHECK-LABEL: func.func @_Z4sz28v() -> i64
// CHECK: {{(polygeist.typeSize.*i28)|(return %c4_i64)}}
unsigned long sz28() { return sizeof(W28); }

// CHECK-LABEL: func.func @_Z4sz16v() -> i64
// CHECK: {{(polygeist.typeSize.*i16)|(return %c2_i64)}}
unsigned long sz16() { return sizeof(W16); }

// CHECK-LABEL: func.func @_Z4copyPA4_3W28S1_(%arg0: memref<?x4xi28>, %arg1: memref<?x4xi28>)
// CHECK-NOT: llvm.struct
void copy(W28 a[][4], W28 b[][4]) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      a[i][j] = b[i][j];
}

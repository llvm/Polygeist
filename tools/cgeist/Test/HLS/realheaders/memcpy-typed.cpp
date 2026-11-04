// memcpy between two arrays of one element type is an element-wise copy in
// that type, through the void * views clang makes of the arguments (OPEN_ISSUES
// O84 (a), kernel w9r8p_mc_cpy: Vitis partitions `loc`; an i8 copy through
// pointers names no array, so hls-est could not bank it). A copy between
// arrays of different element types stays a byte copy.
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <string.h>

// CHECK-LABEL: func.func @_Z5typedPA128_sS0_
// CHECK-NOT: llvm.getelementptr
// CHECK: affine.for %[[I:.*]] = 0 to 64 {
// CHECK:   affine.for %[[J:.*]] = 0 to 128 {
// CHECK:     %[[V:.*]] = affine.load %arg1[%[[I]], %[[J]]] : memref<64x128xi16>
// CHECK:     affine.store %[[V]], %arg0[%[[I]], %[[J]]] : memref<64x128xi16>
void typed(short a[64][128], short b[64][128]) {
  memcpy(a, b, sizeof(short) * 64 * 128);
}

// CHECK-LABEL: func.func @_Z5localPsS_
// CHECK: %[[L:.*]] = memref.alloca() {{.*}} : memref<32xi16>
// CHECK: affine.load %arg0[%{{.*}}] : memref<?xi16>
// CHECK: affine.store %{{.*}}, %[[L]][%{{.*}}] : memref<32xi16>
void local(short *in, short *out) {
  short loc[32];
  memcpy(loc, in, sizeof(loc));
  for (int i = 0; i < 32; i++)
    out[i] = loc[31 - i];
}

// CHECK-LABEL: func.func @_Z5mixedPiPf
// CHECK: llvm.getelementptr
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> i8
void mixed(int *a, float *b) { memcpy(a, b, 64); }

// The element count is the length over the element's size in bytes: an
// unsigned _BitInt(12) occupies 2 bytes, so 16 bytes are 8 elements (W9 round
// 8 review defect 1: the bit width over 8, 1, copied 16 elements, past the
// end of both arrays; probe mc12p.cpp with ap_uint<12>).
// CHECK-LABEL: func.func @_Z4bi12PDU12_S0_
// CHECK-NOT: llvm.getelementptr
// CHECK: affine.for %[[K:.*]] = 0 to 8 {
// CHECK:   %[[W:.*]] = affine.load %arg1[%[[K]]] : memref<8xi12>
// CHECK:   affine.store %[[W]], %arg0[%[[K]]] : memref<8xi12>
void bi12(unsigned _BitInt(12) a[8], unsigned _BitInt(12) b[8]) {
  memcpy(a, b, sizeof(a[0]) * 8);
}

// A width whose byte count is not a power of two has a size that depends on
// the source type (ap_uint<24> 3 bytes under the Vitis headers, _BitInt(24)
// 4): the copy stays a byte copy.
// CHECK-LABEL: func.func @_Z4bi24PDU24_S0_
// CHECK: llvm.getelementptr
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> i8
void bi24(unsigned _BitInt(24) a[6], unsigned _BitInt(24) b[6]) {
  memcpy(a, b, sizeof(a[0]) * 6);
}

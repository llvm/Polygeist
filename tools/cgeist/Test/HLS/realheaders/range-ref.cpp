// ap_range_ref / ap_bit_ref / ap_concat_ref / af_range_ref (structs holding a
// reference, etc/ap_int_ref.h, ap_fixed_ref.h) with runtime operands: with
// --xlx-headers their member functions are inlined and the proxies
// scalarised, so each kernel is plain shift / mask arithmetic on the iN and
// hls-est can read it (no memref of an LLVM struct, no memref2pointer, no
// call, no LLVM intrinsic).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s --check-prefix=CLEAN
#include <ap_fixed.h>
#include <ap_int.h>

// CLEAN-NOT: llvm.struct
// CLEAN-NOT: memref2pointer
// CLEAN-NOT: call
// CLEAN-NOT: llvm.intr
// CLEAN-NOT: llvm.fpga

extern "C" {
// CHECK-LABEL: func.func @k_range(%arg0: memref<?xi32>, %arg1: i8, %arg2: memref<?xi8>, %arg3: i4, %arg4: memref<?xi8>, %arg5: memref<?xi8>)
void k_range(ap_uint<32> &a, ap_uint<8> b, ap_uint<8> &c, ap_uint<4> d,
             ap_uint<8> &r, bool &x) {
  a.range(15, 8) = b;
  c = a.range(7, 0);
  a(3, 0) = d;
  r = a.range(0, 7);
  x = a[5];
  a[6] = 1;
}

// CHECK-LABEL: func.func @k_fixed(%arg0: memref<?xi16>, %arg1: i8, %arg2: memref<?xi8>)
void k_fixed(ap_fixed<16, 5> &f, ap_uint<8> v, ap_uint<8> &o) {
  f.range(7, 0) = v;
  o = f.range(15, 8);
}

// CHECK-LABEL: func.func @k_wide(%arg0: memref<4xi256>, %arg1: memref<4xi256>)
// CHECK: arith.shli {{.*}} : i256
void k_wide(ap_uint<256> in[4], ap_uint<256> out[4]) {
  for (int i = 0; i < 4; i++) {
    ap_uint<256> t = in[i];
    t.range(255, 128) = in[i].range(127, 0);
    out[i] = t;
  }
}

// Runtime indices: the forward and the reversed field are both built and
// selected on lo > hi (Vitis' part.select supports both directions).
// CHECK-LABEL: func.func @k_loop(%arg0: memref<16xi64>, %arg1: memref<16x8xi8>)
// CHECK: affine.for
// CHECK: arith.cmpi ugt
// CHECK: arith.select
void k_loop(ap_uint<64> in[16], ap_uint<8> out[16][8]) {
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 8; j++)
      out[i][j] = in[i].range(8 * j + 7, 8 * j);
}

// The comma concatenation (ap_concat_ref of two references).
// CHECK-LABEL: func.func @k_concat(%arg0: i5, %arg1: i7) -> i12
ap_uint<12> k_concat(ap_uint<5> a, ap_uint<7> b) { return (a, b); }
}

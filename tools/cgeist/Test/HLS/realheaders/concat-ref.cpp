// A concatenation whose operands are range or bit references (an ap_concat_ref
// holding references to ap_range_ref / ap_bit_ref proxies, etc/ap_int_ref.h),
// read and assigned. The inner proxies' addresses are stored into the outer
// one, so they are scalarised only in a second round of the proxy pass
// (Lib/XlxProxyScalarize.cc). Before that fix cgeist exited 0 and printed a
// memref of an LLVM struct holding a memref, which neither mlir-opt nor hls-est
// parses (W11 review D1, kernels c1-c4). The values are checked at runtime by
// the W11 differential kernels (k27, k31).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: FileCheck %s --check-prefix=CLEAN < %t.mlir
#include <ap_int.h>

// CLEAN-NOT: llvm.struct
// CLEAN-NOT: memref2pointer
// CLEAN-NOT: call
// CLEAN-NOT: llvm.fpga

extern "C" {
// Read: (x.range(15,8), y.range(3,0)).
// CHECK-LABEL: func.func @c1(%arg0: i32, %arg1: i32) -> i12
ap_uint<12> c1(ap_uint<32> x, ap_uint<32> y) {
  ap_uint<12> r = (x.range(15, 8), y.range(3, 0));
  return r;
}

// Assigned: (x.range(3,0), y.range(7,4)) = v.
// CHECK-LABEL: func.func @c2(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i8)
void c2(ap_uint<32> &x, ap_uint<32> &y, ap_uint<8> v) {
  (x.range(3, 0), y.range(7, 4)) = v;
}

// A bit reference and a range reference.
// CHECK-LABEL: func.func @c3(%arg0: i32, %arg1: i32) -> i9
ap_uint<9> c3(ap_uint<32> x, ap_uint<32> y) {
  ap_uint<9> q = (x[5], y.range(7, 0));
  return q;
}

// A range reference and a plain value.
// CHECK-LABEL: func.func @c4(%arg0: i32, %arg1: i4) -> i12
ap_uint<12> c4(ap_uint<32> x, ap_uint<4> y) {
  ap_uint<12> r = (x.range(15, 8), y);
  return r;
}
}

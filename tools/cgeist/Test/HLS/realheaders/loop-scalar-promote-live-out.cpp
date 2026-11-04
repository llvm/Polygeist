// XlxLoopScalarPromote on scalars a loop body writes before it reads them,
// whose value is still read after the loop (CIRCT W9 round 15 review,
// defect 1): the loop must carry the value out, and a path through the body
// that does not write it (a break, a zero-trip inner loop, a conditional
// write) must keep the previous value. The first version started such
// bodies from llvm.mlir.undef, so `last` and `v` below left their loops as
// undef. Checked bit-exact against g++ by check_semantics --probe
// (exp-src/w9r15fx/sem in the hls-est work root).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_int.h>

void wfirst(int in[64], int n, int out[2]) {
  ap_uint<10> last = 77;
  out[0] = 0;
  for (ap_uint<8> i = 0; i < (n & 7); i++) {
    last = in[i];
    out[0] = out[0] + last + 1;
  }
  out[1] = last;
}

// CHECK-LABEL: func.func @_Z6wfirstPiiS_
// CHECK-NOT: llvm.mlir.undef
// CHECK: %[[R:.*]] = affine.for %{{.*}} = 0 to %{{.*}} iter_args(%{{.*}} = %c77_i10) -> (i10)
// CHECK: affine.yield %[[V:.*]] : i10
// CHECK: arith.extui %[[R]] : i10 to i32
// CHECK: return

void breakfirst(int in[64], int n, int out[2]) {
  ap_int<14> v = -3;
  for (ap_uint<7> i = 0; i < 64; i++) {
    if (in[i] == (n & 15))
      break;
    v = in[i] + i;
  }
  out[0] = v;
  out[1] = 1;
}

// CHECK-LABEL: func.func @_Z10breakfirstPiiS_
// CHECK-NOT: llvm.mlir.undef
// CHECK: scf.while (%{{.*}} = %true, %{{.*}} = %c0_i7, %{{.*}} = %c-3_i14)
// CHECK: return

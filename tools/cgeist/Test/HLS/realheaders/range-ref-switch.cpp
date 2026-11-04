// A range proxy assigned in the cases of a switch (Vitis_Libraries vision
// xf_threshold.hpp:84-104): the canonicalizer merges the identical cases into
// one block that receives each case's ap_range_ref as a block argument; the
// proxy scalarisation copies the block per incoming edge and resolves both
// (CIRCT OPEN_ISSUES O241; before: "a Vitis HLS proxy object ... could not
// be scalarised").
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK-LABEL: func.func @_Z3top
// CHECK-NOT: llvm.struct<(memref
// CHECK: cf.switch
// CHECK: arith.extui
// CHECK: arith.andi
// CHECK: affine.store
#include <ap_int.h>
void top(ap_uint<16> *out, int type, ap_uint<8> a, ap_uint<8> b) {
  ap_uint<16> d = 0;
  switch (type) {
    case 0: d.range(7, 0) = a; break;
    default: d.range(7, 0) = b;
  }
  *out = d;
}

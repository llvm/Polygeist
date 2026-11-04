// --xlx-headers: a store that the next memory access in the same block
// overwrites is deleted (Lib/XlxDeadStore.cc), as Vitis' LLVM pipeline does
// with the guard store of the headers' ap_fixed conversion
// (etc/ap_fixed_base.h:691 `if (!op) Base::V = 0;` before the unconditional
// `Base::V = ...`; the Vitis run of forgebench/mult_op_module_dot has no
// conditional store left in db/a.o.1.ll). Without the option nothing changes.
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine --xlx-headers -o - | FileCheck %s
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine -o - | FileCheck %s --check-prefix=OFF

extern "C" {
// CHECK-LABEL: func.func @guarded(
// CHECK: affine.for
// CHECK-NOT: scf.if
// CHECK: affine.store
// CHECK-NOT: scf.if
// CHECK: return
// OFF-LABEL: func.func @guarded(
// OFF: affine.for
// OFF: scf.if
void guarded(short out[1], const short a[64], const short b[64]) {
  out[0] = 0;
  for (int i = 0; i < 64; i++) {
    int t = out[0] + a[i] * b[i];
    if (!t)
      out[0] = 0;
    out[0] = (short)(t >> 2);
  }
}

// The guarded store is read before it is overwritten: kept.
// CHECK-LABEL: func.func @read_between(
// CHECK: scf.if
// CHECK: affine.load
// CHECK: affine.store
void read_between(short out[1], short x) {
  if (!x)
    out[0] = 0;
  short y = out[0];
  out[0] = y + 1;
}

// Two stores to the same element: the first one goes.
// CHECK-LABEL: func.func @twice(
// CHECK-NEXT: arith.constant 2
// CHECK-NEXT: affine.store
// CHECK-NEXT: return
void twice(int out[4]) {
  out[1] = 1;
  out[1] = 2;
}

// Different elements: both stay.
// CHECK-LABEL: func.func @other_element(
// CHECK-COUNT-2: affine.store
void other_element(int out[4]) {
  out[1] = 1;
  out[2] = 2;
}
}

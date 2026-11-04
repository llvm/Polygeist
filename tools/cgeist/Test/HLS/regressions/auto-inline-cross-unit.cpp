// Vitis's automatic inlining applies to a call of a function defined in
// another translation unit: Vitis links the units before it inlines (kernel
// w9r8i_tu2, td=off: big() of tu_b.cpp is a module, grp_big_fu_*, with its own
// `loc`). Before, such a call was not counted (the callee is a declaration in
// the caller's unit) and the inliner inlined it whatever its size.
// RUN: cgeist %s %S/Inputs/auto-inline-cross-unit-callee.cpp %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %S/Inputs/auto-inline-cross-unit-callee.cpp %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s %S/Inputs/auto-inline-cross-unit-callee.cpp %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-report -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// REPORT: hls-auto-inline: caller 'tu_top' callee 'big' size {{[0-9]+}} calls 3: kept (size x calls > threshold)
// CHECK-LABEL: func.func @_Z6tu_topPiS_S_S_
// CHECK-COUNT-3: call @_Z3bigPiS_({{.*}}hls.inline_auto_off
void big(int a[256], int b[256]);
void tu_top(int a[256], int b[256], int c[256], int d[256]) {
  big(a, b);
  big(c, d);
  big(b, c);
}

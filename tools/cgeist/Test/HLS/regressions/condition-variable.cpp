// A C++ condition variable is initialised before each test of its loop
// (`while (int v = f())`, `for (; int v = f();)`) and before the condition of
// a switch; an if or switch init statement (`if (int v = ..; v > 3)`) runs
// first. cgeist never initialised the for / switch condition variable and the
// if / switch init statement (the test read an undefined value), and built
// the while condition variable under the loop's keepRunning flag, which is
// unset before the first test and false after a continue (garbage, or a loop
// that never ends; wrong code found by the W8b review). The declaration is now
// built inside the condition's no-break guard, with keepRunning reset first.
// RUN: cgeist %s %resourcedir --function='*' -S -o - | FileCheck %s

int next(int *a, int &i) { return i < 63 ? a[i++] & 31 : 0; }

// CHECK-LABEL: func.func @_Z2wvPi(
// CHECK:         scf.while (%[[S:.*]] = %{{.*}}) : (i32) -> (i32, i32) {
// CHECK-NEXT:      %[[V:.*]] = func.call @_Z4nextPiRi(
// CHECK-NEXT:      %[[C:.*]] = arith.cmpi ne, %[[V]], %{{.*}} : i32
// CHECK-NEXT:      scf.condition(%[[C]]) %[[S]], %[[V]] : i32, i32
// CHECK:         ^bb0(%[[S2:.*]]: i32, %[[V2:.*]]: i32):
// CHECK-NEXT:      arith.addi %[[S2]], %[[V2]] : i32
int wv(int *a) {
  int i = 0, s = 0;
  while (int v = next(a, i))
    s += v;
  return s;
}

// CHECK-LABEL: func.func @_Z2fvPi(
// CHECK:         scf.while (%[[S:.*]] = %{{.*}}) : (i32) -> (i32, i32) {
// CHECK-NEXT:      %[[V:.*]] = func.call @_Z4nextPiRi(
// CHECK-NEXT:      %[[C:.*]] = arith.cmpi ne, %[[V]], %{{.*}} : i32
// CHECK-NEXT:      scf.condition(%[[C]]) %[[S]], %[[V]] : i32, i32
// CHECK:         ^bb0(%[[S2:.*]]: i32, %[[V2:.*]]: i32):
// CHECK-NEXT:      arith.addi %[[S2]], %[[V2]] : i32
int fv(int *a) {
  int i = 0, s = 0;
  for (; int v = next(a, i);)
    s += v;
  return s;
}

// CHECK-LABEL: func.func @_Z2svPi(
// CHECK:         %[[L:.*]] = affine.load %arg0[0]
// CHECK:         %[[V:.*]] = arith.andi %[[L]], %{{.*}} : i32
// CHECK:         cf.switch %[[V]] : i32, [
// CHECK:           1: ^bb1(%[[V]] : i32)
int sv(int *a) {
  switch (int v = a[0] & 3) {
  case 1:
    return v;
  default:
    return -1;
  }
}

// CHECK-LABEL: func.func @_Z2ivPi(
// CHECK:         %[[V:.*]] = affine.load %arg0[0]
// CHECK:         %[[GT:.*]] = arith.cmpi sgt, %[[V]], %{{.*}} : i32
// CHECK:         arith.select %[[GT]], %[[V]], %{{.*}} : i32
int iv(int *a) {
  if (int v = a[0]; v > 3)
    return v;
  return 0;
}

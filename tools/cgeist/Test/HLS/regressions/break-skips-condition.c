// After a `break` a loop's condition is not evaluated (C 6.8.6.3): its side
// effects must not happen. cgeist evaluated the condition and only then
// combined it with the no-break flag, so `do { if (..) break; }
// while (++i < 100)` returned i + 1 on the break (machsuite aes_tableless
// gf_log, wrong code found by check_semantics.py, W8b). The condition is now
// built inside an scf.if on the no-break flag. Polygeist's IfAndLazy pattern
// also gave the inner if's else branch the outer if's other-branch values,
// which dropped the increment of `while (i++ < 100)` on loop exit.
// RUN: cgeist %s %resourcedir --function='*' -S -o - | FileCheck %s

// CHECK-LABEL: func.func @dw(
// CHECK:         %[[W:.*]]:3 = scf.while (%[[I:.*]] = %{{.*}}, %[[NB:.*]] = %true)
// CHECK:           %[[NE:.*]] = arith.cmpi ne, %[[I]], %arg0 : i32
// CHECK:           %[[GO:.*]] = arith.andi %[[NB]], %[[NE]] : i1
// CHECK:           scf.if %[[GO]] -> (i1, i32) {
// CHECK:             %[[INC:.*]] = arith.addi %[[I]], %{{.*}} : i32
// CHECK:             scf.yield %{{.*}}, %[[INC]] : i1, i32
// CHECK:           } else {
// CHECK:             scf.yield %false, %[[I]] : i1, i32
// CHECK:         return %[[W]]#0 : i32
int dw(int x) {
  int i = 0;
  do {
    if (i == x)
      break;
  } while (++i < 100);
  return i;
}

// The value leaving the loop is i + 1 unless a break was taken, whatever the
// comparison gives.
// CHECK-LABEL: func.func @wh(
// CHECK:         scf.while (%[[I:.*]] = %{{.*}}, %[[NB:.*]] = %true)
// CHECK:           %[[INC:.*]] = arith.addi %[[I]], %{{.*}} : i32
// CHECK:           arith.select %[[NB]], %[[INC]], %[[I]] : i32
int wh(int *a, int x) {
  int i = 0;
  while (i++ < 100) {
    if (a[i] == x)
      break;
  }
  return i;
}

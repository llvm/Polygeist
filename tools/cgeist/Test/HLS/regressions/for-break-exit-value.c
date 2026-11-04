// The counter of a for loop left by a `break` keeps the value it had when the
// break was taken (C 6.8.5.3: the increment does not run). Polygeist's
// ForBreakAddUpgrade replaced the "if no break: i + 1, else i" value by i + 1
// for every user, although only the copy of the counter whose loop result is
// unused may take the continue value, so cgeist returned i + 1 after the break
// (`for (i = 0; i < n; i++) if (a == x) { r = i; break; }`, the aes gf_log
// shape; wrong code found by the W8b review). Only that copy's yield slot is
// rewritten now, and only when the counter is read nowhere but under the
// loop's no-break guard.
// RUN: cgeist %s %resourcedir --function='*' -S -o - | FileCheck %s

// CHECK-LABEL: func.func @fb(
// CHECK:         scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(
// CHECK:           %[[I:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK:           %[[INC:.*]] = arith.addi %[[I]], %{{.*}} : i32
// CHECK:           %[[SEL:.*]] = arith.select %{{.*}}, %[[INC]], %[[I]] : i32
// CHECK:           scf.yield %[[SEL]],
void fb(int O[2], int x) {
  int r = 0, i;
  for (i = 0; i < 64; i++) {
    if (i == (x & 31)) {
      r = 1;
      break;
    }
  }
  O[0] = i;
  O[1] = r;
}

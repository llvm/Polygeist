// Store-to-load forwarding (affine scalar replacement) must not forward a
// value into a load when a write to the location can execute in between: a
// write later in the body of a loop that holds the load (but not the store)
// reaches the load in the next iteration (O23). cgeist runs the forwarding
// while its loops are still scf.while / scf.for, which the affine dependence
// analysis does not see as loops.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s %resourcedir --function='*' -S -o - | FileCheck %s --check-prefix=SCF

// O23: the load of out[0] in the loop stays and feeds the accumulation.
// CHECK-LABEL: func.func @dot_i(
// CHECK: affine.store %{{.*}}, %[[OUT:.*]][0]
// CHECK: affine.for
// CHECK: %[[ACC:.*]] = affine.load %[[OUT]][0]
// CHECK: %[[SUM:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK: affine.store %[[SUM]], %[[OUT]][0]
// SCF-LABEL: func.func @dot_i(
// SCF: affine.store %{{.*}}, %[[OUT:.*]][0]
// SCF: scf.for
// SCF: %[[ACC:.*]] = affine.load %[[OUT]][0]
// SCF: %[[SUM:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// SCF: affine.store %[[SUM]], %[[OUT]][0]
void dot_i(int a[64], int b[64], int out[1]) {
  out[0] = 0;
  for (int i = 0; i < 64; i++)
    out[0] += a[i] * b[i];
}

// CHECK-LABEL: func.func @dot_f(
// CHECK: affine.store %{{.*}}, %[[OUT:.*]][0]
// CHECK: affine.for
// CHECK: %[[ACC:.*]] = affine.load %[[OUT]][0]
// CHECK: %[[SUM:.*]] = arith.addf %[[ACC]], %{{.*}} : f32
// CHECK: affine.store %[[SUM]], %[[OUT]][0]
// SCF-LABEL: func.func @dot_f(
// SCF: scf.for
// SCF: %[[ACC:.*]] = affine.load %[[OUT:.*]][0]
// SCF: arith.addf %[[ACC]], %{{.*}} : f32
void dot_f(float a[64], float b[64], float out[1]) {
  out[0] = 0;
  for (int i = 0; i < 64; i++)
    out[0] += a[i] * b[i];
}

// The same with an HLS pragma on the loop.
// CHECK-LABEL: func.func @dot_pipelined(
// CHECK: affine.for
// CHECK: %[[ACC:.*]] = affine.load %[[OUT:.*]][0]
// CHECK: arith.addi %[[ACC]], %{{.*}} : i32
void dot_pipelined(int a[64], int b[64], int out[1]) {
  out[0] = 0;
  for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II = 1
    out[0] += a[i] * b[i];
  }
}

// A loop that can exit early: it is still an scf.while when the forwarding
// first runs (raising turns it into an affine.for with iter_args later), and
// the wrong forwarding used to crash cgeist. Within one iteration the second
// load of out[0] is still replaced by the first one.
// CHECK-LABEL: func.func @acc_break(
// CHECK: affine.store %{{.*}}, %[[OUT:.*]][0]
// CHECK: affine.for
// CHECK: %[[ACC:.*]] = affine.load %[[OUT]][0]
// CHECK-NOT: affine.load %[[OUT]][0]
// CHECK: arith.addi %[[ACC]], %{{.*}} : i32
void acc_break(int *a, int *out) {
  out[0] = 0;
  for (int i = 0; i < 64; i++) {
    if (out[0] > 1000000)
      break;
    out[0] += a[i];
  }
}

// 2-D: the row accumulator is initialised in the outer loop and accumulated
// in an inner loop that stays an scf.for (its bound is loaded from memory).
// CHECK-LABEL: func.func @row_acc_var(
// CHECK: affine.for %[[I:.*]] = 0 to 8 {
// CHECK: affine.store %{{.*}}, %[[OUT:.*]][%[[I]] * 2]
// CHECK: scf.for
// CHECK: %[[ACC:.*]] = affine.load %[[OUT]][%[[I]] * 2]
// CHECK: %[[SUM:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK: affine.store %[[SUM]], %[[OUT]][%[[I]] * 2]
void row_acc_var(int *a, int *out, int *len) {
  for (int i = 0; i < 8; i++) {
    out[2 * i] = 0;
    int m = len[i] & 15;
    for (int j = 0; j < m; j++)
      out[2 * i] += a[i * 16 + j];
  }
}

// 2-D with both loops raised.
// CHECK-LABEL: func.func @row_acc(
// CHECK: affine.for %[[I:.*]] = 0 to 8 {
// CHECK: affine.store %{{.*}}, %[[OUT:.*]][%[[I]], 0]
// CHECK: affine.for
// CHECK: %[[ACC:.*]] = affine.load %[[OUT]][%[[I]], 0]
// CHECK: arith.addi %[[ACC]], %{{.*}} : i32
void row_acc(int a[8][8], int out[8][2]) {
  for (int i = 0; i < 8; i++) {
    out[i][0] = 0;
    for (int j = 0; j < 8; j++)
      out[i][0] += a[i][j];
  }
}

// May alias: q[0] is out[k & 1]. The load of out[0] at the top of the body
// stays (q[0] may have overwritten it in the previous iteration); the load
// after the store to out[0] in the same iteration is forwarded.
// CHECK-LABEL: func.func @alias_inloop(
// CHECK: affine.for
// CHECK: %[[ACC:.*]] = affine.load %[[OUT:.*]][0]
// CHECK: %[[SUM:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK: affine.store %[[SUM]], %[[OUT]][0]
// CHECK-NOT: affine.load
// CHECK: arith.muli %[[SUM]]
// CHECK: affine.store %{{.*}}, %[[OUT]][symbol(%{{.*}})]
void alias_inloop(int *out, int *a, int k) {
  int *q = out + (k & 1);
  out[0] = 0;
  for (int i = 0; i < 16; i++) {
    out[0] += a[i];
    q[0] = out[0] * 2;
  }
}

// May alias through two pointer arguments: p[0] is reloaded after the loop.
// CHECK-LABEL: func.func @two_ptr(
// CHECK-SAME: %[[P:[^:]*]]: memref<?xi32>, %[[Q:[^:]*]]: memref<?xi32>
// CHECK: affine.store %{{.*}}, %[[P]][0]
// CHECK: affine.for
// CHECK: affine.store %{{.*}}, %[[Q]][0]
// CHECK: }
// CHECK: %[[R:.*]] = affine.load %[[P]][0]
// CHECK: affine.store %[[R]], %[[Q]][1]
void two_ptr(int *p, int *q, int *a) {
  p[0] = 0;
  for (int i = 0; i < 16; i++)
    q[0] += a[i];
  q[1] = p[0];
}

// p[0] in the loop is reloaded every iteration: q[0] may be p[0].
// CHECK-LABEL: func.func @two_ptr_acc(
// CHECK-SAME: %[[P:[^:]*]]: memref<?xi32>, %[[Q:[^:]*]]: memref<?xi32>
// CHECK: affine.for
// CHECK: %[[ACC:.*]] = affine.load %[[P]][0]
// CHECK: arith.addi %[[ACC]]
void two_ptr_acc(int *p, int *q, int *a) {
  p[0] = 0;
  for (int i = 0; i < 16; i++) {
    p[0] += a[i];
    q[0] = p[0] * 2;
  }
}

// The loop writes the element it reads, through an index computed separately
// (o[k & 3] twice) or as another expression (o[r + 1] with r = s - 1): the
// load stays.
// CHECK-LABEL: func.func @while_sym_same(
// CHECK: affine.store %{{.*}}, %[[O:.*]][symbol(%[[S:.*]])]
// CHECK: scf.while
// CHECK: } do {
// CHECK: %[[L:.*]] = affine.load %[[O]][symbol(%[[S]])]
// CHECK: arith.addi %{{.*}}, %[[L]] : i32
// CHECK: affine.store %{{.*}}, %[[O]][symbol(%[[S]])]
void while_sym_same(int *o, int *a, int k) {
  o[k & 3] = k * 5;
  int t = 0;
  int i = 0;
  while (a[i] != 0 && i < 12) {
    t += o[k & 3];
    o[k & 3] = (t + a[i]) & 255;
    i++;
  }
  o[10] = t;
}

// CHECK-LABEL: func.func @while_sym_alias(
// CHECK: affine.store %{{.*}}, %[[O:.*]][symbol(%[[S:.*]])]
// CHECK: scf.while
// CHECK: } do {
// CHECK: %[[L:.*]] = affine.load %[[O]][symbol(%[[S]])]
// CHECK: arith.addi %{{.*}}, %[[L]] : i32
void while_sym_alias(int *o, int *a, int k) {
  int s = k & 3;
  int r = s - 1;
  o[s] = k * 5;
  int t = 0;
  int i = 0;
  while (a[i] != 0 && i < 12) {
    t += o[s];
    o[r + 1] = (t + a[i]) & 255;
    i++;
  }
  o[10] = t;
}

// Controls: forwarding is legal and still happens.

// Store then load, nothing in between.
// CHECK-LABEL: func.func @fwd_simple(
// CHECK-SAME: %[[A:[^:]*]]: memref<?xi32>, %[[O:[^:]*]]: memref<?xi32>
// CHECK: %[[V:.*]] = arith.addi
// CHECK: affine.store %[[V]], %[[O]][0]
// CHECK-NOT: affine.load %[[O]]
// CHECK: arith.muli %[[V]]
void fwd_simple(int *a, int *o) {
  o[0] = a[0] + 1;
  o[1] = o[0] * 2;
}

// A loop that writes other elements of the array only.
// CHECK-LABEL: func.func @fwd_loop_nowrite(
// CHECK: %[[V:.*]] = affine.load %{{.*}}[0]
// CHECK: affine.store %[[V]], %[[O:.*]][0]
// CHECK: affine.for
// CHECK-NOT: affine.load %[[O]]
// CHECK: arith.addi %[[V]]
void fwd_loop_nowrite(int *a, int *o) {
  o[0] = a[0];
  for (int i = 0; i < 16; i++)
    o[i + 1] = o[0] + a[i];
}

// A while loop (never raised) that writes a different constant element.
// CHECK-LABEL: func.func @fwd_while_other(
// CHECK: %[[V:.*]] = affine.load %{{.*}}[0]
// CHECK: affine.store %[[V]], %[[O:.*]][0]
// CHECK: scf.while
// CHECK-NOT: affine.load %[[O]]
// CHECK: arith.addi %[[V]]
// CHECK: affine.store %{{.*}}, %[[O]][1]
void fwd_while_other(int *a, int *o) {
  o[0] = a[0];
  int i = 0;
  while (a[i + 1] > 0 && i < 30) {
    o[1] = o[0] + a[i];
    i++;
  }
}

// A while loop (never raised) that writes o[s + 1] while it reads o[s]: the
// two never overlap, so the value stored to o[s] before the loop is forwarded.
// Each access has its own arith.index_cast of s when the forwarding runs.
// CHECK-LABEL: func.func @fwd_while_sym(
// CHECK: %[[V:.*]] = arith.muli
// CHECK: affine.store %[[V]], %[[O:.*]][symbol(%{{.*}})]
// CHECK: scf.while
// CHECK: } do {
// CHECK-NOT: affine.load %[[O]][symbol(%{{[0-9]+}})] :
// CHECK: arith.addi %{{.*}}, %[[V]] : i32
// CHECK: affine.store %{{.*}}, %[[O]][symbol(%{{.*}}) + 1]
void fwd_while_sym(int *o, int *a, int k) {
  int s = k & 3;
  o[s] = k * 5;
  int t = 0;
  int i = 0;
  while (a[i] != 0 && i < 12) {
    t += o[s];
    o[s + 1] = (o[s + 1] + a[i]) & 255;
    i++;
  }
  o[10] = t;
}

// Store and load in the same iteration.
// CHECK-LABEL: func.func @fwd_inloop(
// CHECK: affine.for
// CHECK: %[[V:.*]] = affine.load
// CHECK: affine.store %[[V]]
// CHECK-NOT: affine.load
// CHECK: arith.muli %[[V]]
void fwd_inloop(int *a, int *o) {
  for (int i = 0; i < 16; i++) {
    o[2 * i] = a[i];
    o[2 * i + 1] = o[2 * i] * 3;
  }
}

// A load after a loop that does not write the element.
// CHECK-LABEL: func.func @fwd_after_loop(
// CHECK: affine.for
// CHECK: }
// CHECK-NOT: affine.load
// CHECK: affine.store %{{.*}}, %{{.*}}[20]
void fwd_after_loop(int *a, int *o) {
  o[0] = 5;
  for (int i = 0; i < 16; i++)
    o[i + 1] = a[i];
  o[20] = o[0];
}

// RUN: cgeist %s --function='*' -S --raise-scf-to-affine | FileCheck %s

// Counted loops that rotate their variables (`B = A; A = t;`, the SHA rounds
// of chstone/sha) raise to affine.for: the do-while form cgeist builds
// forwards the old value of a rotated variable to the condition next to the
// if results, so the exit value of a variable is not at its block argument's
// position, and a variable the next loop reads only after the exit has an
// undef then value (MoveWhileDown2 in CanonicalizeFor.cpp).

int W[80];
int out[5];

// CHECK-LABEL: func.func @rot2(
// CHECK-NOT: scf.while
// CHECK: %[[R:.*]]:2 = affine.for %[[I:.*]] = 0 to 20 iter_args(%[[A:.*]] = %{{.*}}, %[[B:.*]] = %{{.*}}) -> (i32, i32) {
// CHECK: %[[W:.*]] = affine.load %{{.*}}[%[[I]]] : memref<80xi32>
// CHECK: %[[T:.*]] = arith.addi %[[A]], %[[W]] : i32
// CHECK: affine.yield %[[T]], %[[A]] : i32, i32
// CHECK: affine.store %[[R]]#0, %{{.*}}[0]
// CHECK: affine.store %[[R]]#1, %{{.*}}[1]
void rot2(void) {
  int i;
  int A = out[0], B = out[1];
  for (i = 0; i < 20; ++i) {
    int t = A + W[i];
    B = A;
    A = t;
  }
  out[0] = A;
  out[1] = B;
}

// Five rotated variables over two loops: the second loop starts from the
// exit values of the first.
// CHECK-LABEL: func.func @rot5(
// CHECK-NOT: scf.while
// CHECK: %[[L1:.*]]:5 = affine.for %{{.*}} = 0 to 20 iter_args(
// CHECK: affine.yield
// CHECK: %[[L2:.*]]:5 = affine.for %{{.*}} = 20 to 40 iter_args(%{{.*}} = %[[L1]]#0, %{{.*}} = %[[L1]]#1, %{{.*}} = %[[L1]]#2, %{{.*}} = %[[L1]]#3, %{{.*}} = %[[L1]]#4)
// CHECK: affine.yield
// CHECK-NOT: scf.while
// CHECK: return
#define ROT(x, n) ((x << n) | (x >> (32 - n)))
void rot5(void) {
  int i;
  unsigned A = out[0], B = out[1], C = out[2], D = out[3], E = out[4], t;
  for (i = 0; i < 20; ++i) {
    t = ROT(A, 5) + (B ^ C ^ D) + E + W[i];
    E = D; D = C; C = ROT(B, 30); B = A; A = t;
  }
  for (i = 20; i < 40; ++i) {
    t = ROT(A, 5) + ((B & C) | (~B & D)) + E + W[i];
    E = D; D = C; C = ROT(B, 30); B = A; A = t;
  }
  out[0] = A; out[1] = B; out[2] = C; out[3] = D; out[4] = E;
}

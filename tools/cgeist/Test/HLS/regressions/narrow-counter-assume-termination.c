// A loop over a counter narrower than int that its runtime bound does not
// provably keep in range (`for (uint8_t i = 0; i < n; i++)` with int n never
// ends for n > 255) is raised by assuming that it terminates (C11 6.8.5p6,
// C++ [intro.progress]: no I/O, volatile or atomic work in it): counting by
// +-2^k with a test that reads the counter with its own signedness, a wrap
// would return the counter to values the test has accepted, so a loop that
// terminates never wraps. A bound of a wider type than the counter marks the
// loop hls.uncounted (Vitis does not collapse such a loop with its perfect
// inner nest; hls-est-validate VITIS_NOTES.md N15).
// -canonicalize-for-assume-termination=false keeps such loops as while loops.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s --implicit-check-not=polygeist.counter_nowrap --implicit-check-not=polygeist.nuw --implicit-check-not=polygeist.nsw --implicit-check-not=polygeist.volatile_access
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -canonicalize-for-assume-termination=false -o - | FileCheck %s --check-prefix=OFF
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -keep-frontend-assumptions -o - | FileCheck %s --check-prefix=MARK

#include <stdint.h>

void ext(int);

// CHECK-LABEL: func.func @u8_int_bound(
// CHECK: affine.for %[[I:.*]] = 0 to %{{.*}} {
// CHECK: affine.load %{{.*}}[%[[I]]]
// CHECK: affine.store %{{.*}}, %{{.*}}[%[[I]]]
// CHECK: } {hls.uncounted}
// OFF-LABEL: func.func @u8_int_bound(
// OFF-NOT: affine.for
// OFF: scf.while
// MARK-LABEL: func.func @u8_int_bound(
// MARK: } {hls.uncounted}
void u8_int_bound(int *A, int *O, int n) {
  for (uint8_t i = 0; i < n; i++)
    O[i] = A[i];
}

// The counter read sign-extended (an int8_t compared with an int).
// CHECK-LABEL: func.func @s8_int_bound(
// CHECK: affine.for %[[I:.*]] = -100 to %{{.*}} {
// CHECK: affine.load %{{.*}}[%[[I]] + 128]
// CHECK: } {hls.uncounted}
void s8_int_bound(int *A, int *O, int n) {
  for (int8_t i = -100; i < n; i++)
    O[i + 128] = A[i + 128];
}

// CHECK-LABEL: func.func @u16_int_bound_step4(
// CHECK: affine.for %[[I:.*]] = 0 to %{{.*}} step 4 {
// CHECK: affine.load %{{.*}}[%[[I]]]
// CHECK: } {hls.uncounted}
void u16_int_bound_step4(int *A, int *O, int n) {
  for (uint16_t i = 0; i < n; i += 4)
    O[i] = A[i];
}

// Counting down to a runtime int bound.
// CHECK-LABEL: func.func @u8_int_bound_down(
// CHECK: affine.for
// CHECK: } {hls.uncounted}
void u8_int_bound_down(int *A, int *O, int n) {
  for (uint8_t i = 200; i > n; i--)
    O[i] = A[i];
}

// A bound of the counter's own width (254 + 2 wraps for m = 255, where the
// loop never ends): raised under the same assumption, not uncounted.
// CHECK-LABEL: func.func @u8_u8_bound_step2(
// CHECK: affine.for %[[I:.*]] = 0 to %{{.*}} step 2 {
// CHECK: affine.load %{{.*}}[%[[I]]]
// CHECK-NOT: hls.uncounted
// CHECK: return
// OFF-LABEL: func.func @u8_u8_bound_step2(
// OFF: scf.while
void u8_u8_bound_step2(int *A, int *O, uint8_t m) {
  for (uint8_t i = 0; i < m; i += 2)
    O[i] = A[i];
}

// A step that is not a power of two can wrap into a residue that ends the
// loop (uint8_t, step 3, n = 254: .., 253, 0, 3, .., 255): not raised.
// CHECK-LABEL: func.func @u8_int_bound_step3(
// CHECK-NOT: affine.for
// CHECK: scf.while
void u8_int_bound_step3(int *A, int *O, int n) {
  for (uint8_t i = 0; i < n; i += 3)
    O[i] = A[i];
}

// A call to a function cgeist cannot see into may do I/O, which makes
// running forever observable: not raised.
// CHECK-LABEL: func.func @u8_int_bound_call(
// CHECK-NOT: affine.for
// CHECK: scf.while
void u8_int_bound_call(int *A, int n) {
  for (uint8_t i = 0; i < n; i++)
    ext(A[i]);
}

// A sign-extended counter compared unsigned: a wrap to a negative value is a
// huge value there, which ends the loop. Not raised.
// CHECK-LABEL: func.func @s8_unsigned_bound(
// CHECK-NOT: affine.for
// CHECK: scf.while
void s8_unsigned_bound(int *A, int *O, unsigned n) {
  for (int8_t i = 0; i < n; i++)
    O[i] = A[i];
}

// The counter's value after the loop is its exit value, which fits it.
// CHECK-LABEL: func.func @u8_exit_value(
// CHECK: affine.for
// CHECK: } {hls.uncounted}
// CHECK: return
int u8_exit_value(int *A, int n) {
  uint8_t i;
  for (i = 0; i < n; i++)
    A[i] = 1;
  return i;
}

// A volatile access makes running forever observable (C11 6.8.5p6): for
// n > 255 the loop below writes *p forever, which is defined behaviour. cgeist
// does not represent volatile on memref accesses; the frontend marks every
// function whose body accesses a volatile-qualified object
// (polygeist.volatile_access, removed from the output unless
// -keep-frontend-assumptions), and no loop in such a function, or calling such
// a function, is raised by the assumption.
// CHECK-LABEL: func.func @u8_volatile_store(
// CHECK-NOT: affine.for
// CHECK: scf.while
// MARK-LABEL: func.func @u8_volatile_store(
// MARK-SAME: polygeist.volatile_access
void u8_volatile_store(volatile int *p, int n) {
  for (uint8_t k = 0; k < n; k++)
    *p = k;
}

// CHECK-LABEL: func.func @u8_volatile_array(
// CHECK-NOT: affine.for
// CHECK: scf.while
void u8_volatile_array(volatile int A[64], int *O, int n) {
  for (uint8_t k = 0; k < n; k++)
    O[k] = A[k & 63];
}

// The volatile access is in a callee (kept as a call or inlined).
static void store_volatile(volatile int *p, int v) { *p = v; }

// CHECK-LABEL: func.func @u8_volatile_callee(
// CHECK-NOT: affine.for
// CHECK: scf.while
void u8_volatile_callee(volatile int *p, int n) {
  for (uint8_t k = 0; k < n; k++)
    store_volatile(p, k);
}

// A volatile object the body never accesses (the parameter q itself) does not
// count, nor does a pointer to non-volatile int.
// CHECK-LABEL: func.func @u8_volatile_pointer_object(
// CHECK: affine.for
// CHECK: } {hls.uncounted}
void u8_volatile_pointer_object(int *volatile q, int *O, int n) {
  int *r = O;
  for (uint8_t k = 0; k < n; k++)
    r[k] = k;
}

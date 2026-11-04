// AffineBufferElimination forwards a stored value to the loads of a buffer.
// It used to compute the stored value again at the load when the value was
// defined inside the loop, re-reading memory that was written in between:
// `r = f(&s); out[j] = W(r).v` (W's constructor reads r through a const
// reference) wrote f(f(s)) instead of f(s) (OPEN_ISSUES O37, forgebench
// gpt_transformer_p1's lcg_rand / dropout). A value that dominates the load is
// now used as is, and a value computed again is checked against the writes
// since its first computation (W8b).
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

unsigned f(unsigned *s) { *s = *s * 3u + 1u; return *s; }
struct W { int v; W(const unsigned &x) { v = (int)x; } };

// CHECK-LABEL: func.func @_Z3topPij(
// CHECK:         affine.for %[[J:.*]] = 0 to 8 {
// CHECK:           %[[S:.*]] = affine.load %[[SEED:.*]][] : memref<i32>
// CHECK:           %[[M:.*]] = arith.muli %[[S]], %{{.*}} : i32
// CHECK:           %[[R:.*]] = arith.addi %[[M]], %{{.*}} : i32
// CHECK:           affine.store %[[R]], %[[SEED]][] : memref<i32>
// CHECK-NOT:       arith.muli
// CHECK:           affine.store %[[R]], %{{.*}}[%[[J]]] : memref<8xi32>
void top(int out[8], unsigned seed) {
  for (int j = 0; j < 8; j++) {
    unsigned r = f(&seed);
    out[j] = W(r).v;
  }
}

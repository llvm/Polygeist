// C ignores the first dimension of an array parameter, and a `T p[]`
// parameter takes any array: an argument's static shape may differ from the
// parameter's. cgeist used to pass the caller's memref unchanged, which does
// not verify ("'func.call' op operand type mismatch: expected operand type
// 'memref<?xi16>', but provided 'memref<160xi16>'", chstone gsm, machsuite
// aes / backprop, W8b). It now casts it towards a dynamic shape, or, when two
// static shapes differ, reinterprets it with the parameter's shape (a chain
// of two casts through a dynamic shape is folded back into an invalid cast:
// "'memref.cast' op operand type 'memref<3x8xi32>' and result type
// 'memref<16x8xi32>' are cast incompatible", YOLOv4-Tiny, DGNN_Booster). A
// kept call whose argument does not fit the parameter shape is still inlined
// (inline-off-kept-call.c), and once inlined, the accesses use the argument
// itself when only the first dimension differs (same strides).
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK-LABEL: func.func @top(
// CHECK-SAME:    %[[X:[^:]*]]: memref<4096xi16>, %[[Y:[^:]*]]: memref<4096xi16>, %[[Z:[^:]*]]: memref<16xi16>, %[[W:[^:]*]]: memref<16xi16>, %[[V:[^:]*]]: memref<3xf64>, %[[T:[^:]*]]: memref<3x8xi32>, %[[U:[^:]*]]: memref<16x8xi32>
// CHECK:         call @f(%[[X]], %[[Y]]) : (memref<4096xi16>, memref<4096xi16>) -> ()
// CHECK:         %[[XC:.*]] = memref.cast %[[X]] : memref<4096xi16> to memref<?xi16>
// CHECK:         %[[YC:.*]] = memref.cast %[[Y]] : memref<4096xi16> to memref<?xi16>
// CHECK:         call @g(%[[XC]], %[[YC]]) : (memref<?xi16>, memref<?xi16>) -> ()
// CHECK:         affine.for %[[I:.*]] = 0 to 16 {
// CHECK:           affine.load %[[Z]][%[[I]]] : memref<16xi16>
// CHECK:           affine.store %{{.*}}, %[[W]][%[[I]]] : memref<16xi16>
// CHECK:         affine.for %[[J:.*]] = 0 to 3 {
// CHECK:           affine.load %[[V]][%[[J]]] : memref<3xf64>
// CHECK-NOT:     call @h
// CHECK-NOT:     memref.reinterpret_cast
// CHECK-NOT:     call @k
// CHECK:         affine.for %[[R:.*]] = 0 to 16 {
// CHECK:           affine.for %[[C:.*]] = 0 to 8 {
// CHECK:             affine.load %[[T]][%[[R]], %[[C]]] : memref<3x8xi32>
// CHECK:             affine.store %{{.*}}, %[[U]][%[[R]], %[[C]]] : memref<16x8xi32>
void f(short a[4096], short b[4096]) {
#pragma HLS inline off
  for (int i = 0; i < 16; i++) b[i] = a[i] + 1;
}
void g(short a[], short b[]) {
#pragma HLS inline off
  for (int i = 0; i < 16; i++) b[i] = a[i] * 2;
}
void h(double a[64]) {
#pragma HLS inline off
  for (int i = 0; i < 3; i++) a[i] = a[i] + 1.0;
}
void k(int a[16][8], int o[16][8]) {
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 8; j++)
      o[i][j] = a[i][j] + 1;
}
void top(short x[4096], short y[4096], short z[16], short w[16], double v[3],
         int t[3][8], int u[16][8]) {
  f(x, y);
  g(x, y);
  f(z, w);
  h(v);
  k(t, u);
}

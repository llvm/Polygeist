// Braced zero initialisers of aggregates. A struct whose member is an array
// (`window w = {};`) left the member an ImplicitValueInitExpr that cgeist
// could not store (assertion `toStore.isReference`, spector_hls
// template_matching); a 3-D array of a class type (`fm_t buf[2][2][2] =
// {0};`) is wrapped in an ExprWithCleanups that hid the init list
// (assertion on a rank-3 store, YOLOv4-Tiny, DSC_Acceleration). An all-zero
// initialiser of an array of scalars is now one zero-filling loop nest; a
// global gets a zero initial value (W8b).
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

struct window { unsigned char win[4][4]; };
// The element class has the exactclass shim's ap_fixed constructors (a
// defaulted default constructor, a constexpr converting one), so clang's
// constant evaluation sees the all-zero value. (With user-provided,
// non-constexpr constructors the initialiser is still emitted element by
// element, which is slow for large arrays.)
struct fx { short v; fx() = default; constexpr fx(int x) : v(x) {} };

// The global is printed first.
// CHECK: memref.global @zeros : memref<8x8xi32> = dense<0>
// CHECK-LABEL: func.func @_Z12local_structPhS_(
// CHECK:         llvm.store %{{.*}}, %{{.*}} : !llvm.array<4 x array<4 x i8>>
void local_struct(unsigned char in[16], unsigned char out[16]) {
  window w = {};
  for (int i = 0; i < 16; i++) w.win[i / 4][i % 4] = in[i];
  for (int i = 0; i < 16; i++) out[i] = w.win[i % 4][i / 4];
}

// The zero-filled buffer folds away: every element read is 0.
// CHECK-LABEL: func.func @_Z10class_3d_a
// CHECK-NOT:     memref.alloca
// CHECK:         %[[Z:.*]] = arith.constant 0 : i16
// CHECK:         affine.for %[[I:.*]] = 0 to 64 {
// CHECK-NEXT:      affine.for %[[J:.*]] = 0 to 112 {
// CHECK-NEXT:        affine.store %[[Z]], %{{.*}}[%[[I]], %[[J]], 0] : memref<64x112x1xi16>
void class_3d_a(fx out[64][112], fx sel[1]) {
  fx buf[64][112][112] = {0};
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 112; j++)
      out[i][j] = buf[i][j][sel[0].v & 63];
}

int zeros[8][8] = {0};
// CHECK-LABEL: func.func @_Z9read_zeroi(
// CHECK:         memref.get_global @zeros
int read_zero(int i) { return zeros[i & 7][i & 7]; }

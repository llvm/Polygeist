// An array_partition on a single hls::vector (a class whose only data
// member is, through std::array, an array; not an array of vectors): Vitis
// ignores it ([HLS 214-253] "Ignored unsupported array_partition pragma on
// variable", kernel w9r7fix_vp_local with this source; class_ece8893
// nikhil_p lab3 `col_ptr_B_local`), so cgeist drops it with a warning. An
// array of vectors keeps its pragma. Before W9r7 it was rejected ("requires
// an array or pointer"), then (W9r7-impl) applied to the lanes.
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - 2>%t.err | FileCheck %s
// RUN: FileCheck %s --check-prefix=WARN < %t.err
#include <hls_vector.h>

// WARN-DAG: warning: '#pragma HLS array_partition' on 'v' of class type 'hls::vector<int, 65>' is ignored
// WARN-DAG: warning: '#pragma HLS array_partition' on 'w' of class type 'hls::vector<int, 8>' is ignored
// CHECK-DAG: memref.alloca() {{.*}}hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "cyclic", variable = "a"}]{{.*}} : memref<4x!llvm.struct<(struct<(array<8 x i32>)>)>>
// CHECK-NOT: variable = "v"
// CHECK-NOT: variable = "w"
void c_top(const int in[65], int out[65]) {
  hls::vector<int, 65> v;
#pragma HLS array_partition variable=v complete
  hls::vector<int, 8> w;
#pragma HLS array_partition variable=w cyclic factor=4 dim=1
  hls::vector<int, 8> a[4];
#pragma HLS array_partition variable=a cyclic factor=2 dim=1
  for (int i = 0; i < 65; i++) v[i] = in[i];
  for (int i = 0; i < 8; i++) w[i] = in[i];
  for (int i = 0; i < 32; i++) a[i >> 3][i & 7] = in[i];
  for (int i = 0; i < 65; i++) out[i] = v[64 - i] + w[i & 7] + a[i & 3][i & 7];
}

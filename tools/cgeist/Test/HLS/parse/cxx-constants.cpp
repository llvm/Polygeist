// Integer options given by C++ constants: constexpr locals and globals,
// namespace constants, enumerators, and template parameters of the
// instantiation being emitted.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s

// CHECK-NOT: warning
// CHECK-NOT: error

constexpr int kGlobalFactor = 2;
namespace cfg {
constexpr int kII = 3;
enum Dims { kDim = 1 };
} // namespace cfg

// CHECK-LABEL: func.func @_Z9constantsPi(
// CHECK: hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "cyclic", variable = "buf"}]
// CHECK: } {hls.pipeline = [{II = 3 : i32, rewind = false, style = "stp"}]}
// CHECK: } {hls.unroll = [{factor = 4 : i32, skip_exit_check = false}]}
int constants(int *a) {
  using namespace cfg;
  constexpr int kLocal = 4;
  int buf[16];
#pragma HLS array_partition variable=buf cyclic factor=kGlobalFactor dim=cfg::kDim
  for (int i = 0; i < 16; ++i) {
#pragma HLS pipeline II=kII
    buf[i] = a[i];
  }
  int s = 0;
  for (int i = 0; i < 16; ++i) {
#pragma HLS unroll factor=kLocal
    s += buf[i];
  }
  return s;
}

// Each instantiation gets its own attributes.
template <int F, int II> int templated(int *a) {
  int buf[16];
#pragma HLS array_partition variable=buf cyclic factor=F
  for (int i = 0; i < 16; ++i) {
#pragma HLS pipeline II=II
    buf[i] = a[i];
  }
  int s = 0;
  for (int i = 0; i < 16; ++i)
    s += buf[15 - i] * i;
  return s;
}

int use_templates(int *a) { return templated<2, 1>(a) + templated<8, 4>(a); }

// CHECK-DAG: hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "cyclic", variable = "buf"}]
// CHECK-DAG: hls.array_partition = [{dim = 1 : i32, factor = 8 : i32, kind = "cyclic", variable = "buf"}]
// CHECK-DAG: hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]
// CHECK-DAG: hls.pipeline = [{II = 4 : i32, rewind = false, style = "stp"}]

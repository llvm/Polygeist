// _Pragma, pragmas produced by macro expansion, and option values given by
// macros or constant expressions.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s

// CHECK-NOT: warning
// CHECK-NOT: error

#define N 16
#define FACTOR 4
#define PIPE_II 2
#define STRINGIFY(x) #x
#define DO_PRAGMA(x) _Pragma(STRINGIFY(x))
#define HLS_PIPELINE(ii) DO_PRAGMA(HLS pipeline II = ii)
#define UNROLL_ALL _Pragma("HLS unroll")
#define PART_OPTS type=block factor=FACTOR dim=1

enum { ENUM_FACTOR = 8 };

// CHECK-LABEL: func.func @pragma_operator(
// CHECK: } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
void pragma_operator(int *a) {
  for (int i = 0; i < N; ++i) {
    _Pragma("HLS pipeline II=2")
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @macro_argument(
// CHECK: } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
void macro_argument(int *a) {
  for (int i = 0; i < N; ++i) {
    HLS_PIPELINE(PIPE_II)
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @macro_whole_pragma(
// CHECK: } {hls.unroll = [{factor = 0 : i32, skip_exit_check = false}]}
void macro_whole_pragma(int *a) {
  for (int i = 0; i < N; ++i) {
    UNROLL_ALL
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @macro_values(
// CHECK: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "block", variable = "buf"}]
int macro_values(int *a) {
  int buf[N];
#pragma HLS array_partition variable=buf PART_OPTS
  for (int i = 0; i < N; ++i)
    buf[i] = a[i];
  int s = 0;
  for (int i = 0; i < N; ++i)
    s += buf[i];
  return s;
}

// Constant expressions, hex literals, enumerators and const variables.
// CHECK-LABEL: func.func @const_exprs(
// CHECK: hls.array_partition = [{dim = 1 : i32, factor = 8 : i32, kind = "cyclic", variable = "buf"}, {dim = 2 : i32, factor = 2 : i32, kind = "cyclic", variable = "buf"}]
// CHECK: } {hls.pipeline = [{II = 4 : i32, rewind = false, style = "stp"}]}
// CHECK: } {hls.unroll = [{factor = 3 : i32, skip_exit_check = false}]}
int const_exprs(int *a) {
  const int two = 2;
  int buf[N][N];
#pragma HLS array_partition variable=buf cyclic factor=ENUM_FACTOR dim=1
#pragma HLS array_partition variable=buf cyclic factor=N / 8 dim=two
  int s = 0;
  for (int i = 0; i < N; ++i) {
#pragma HLS pipeline II=(0x1 + 1) * two
    for (int j = 0; j < N; ++j)
      buf[i][j] = a[i + j];
  }
  for (int i = 0; i < N; ++i) {
#pragma HLS unroll factor=FACTOR-1
    s += buf[i][i];
  }
  return s;
}

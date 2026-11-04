// Variable resolution for array_partition: local arrays (1-D, multi-D, in
// inner scopes), array and pointer parameters (argument attributes),
// globals and static locals (memref.global), shadowing, pragmas placed
// before the declaration, and several pragmas on one array.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s --check-prefix=FULLRANK

// CHECK-NOT: warning
// CHECK-NOT: error

int gbuf[32][4];
int gunused[8];

// Globals are annotated on their memref.global.
// CHECK-DAG: memref.global @gbuf : memref<32x4xi32> = uninitialized {hls.array_partition = [{dim = 2 : i32, kind = "complete", variable = "gbuf"}, {dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "gbuf"}]}
// CHECK-DAG: memref.global "private" @"statics@static@st" : memref<8xi32> = uninitialized {hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "block", variable = "st"}]}

// CHECK-LABEL: func.func @multi_dim(
// CHECK: memref.alloca() {hls.array_partition = [{dim = 3 : i32, kind = "complete", variable = "buf"}, {dim = 1 : i32, factor = 2 : i32, kind = "cyclic", variable = "buf"}], hls.preserve, polygeist.varname = "buf"} : memref<4x4x2xi32>
void multi_dim(int *a) {
  int buf[4][4][2];
#pragma HLS array_partition variable=buf complete dim=3
#pragma HLS array_partition variable=buf cyclic factor=2 dim=1
  for (int i = 0; i < 32; ++i)
    buf[i / 8][(i / 2) % 4][i % 2] = a[i];
  for (int i = 0; i < 32; ++i)
    a[i] = buf[i % 4][(i / 4) % 4][i / 16];
}

// Array and pointer parameters: argument attributes.
// CHECK-LABEL: func.func @params(
// CHECK-SAME: %arg0: memref<?x4xi32> {hls.array_partition = [{dim = 2 : i32, kind = "complete", variable = "m"}]}
// CHECK-SAME: %arg1: memref<?xi32> {hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "cyclic", variable = "p"}]}
// CHECK-SAME: %arg2: memref<?xi32> {hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "block", variable = "v"}]}
// CHECK-SAME: %arg3: i32)
// FULLRANK-LABEL: func.func @params(
// FULLRANK-SAME: {hls.array_partition = [{dim = 2 : i32, kind = "complete", variable = "m"}]}
void params(int m[8][4], int *p, int v[16], int n) {
#pragma HLS array_partition variable=m complete dim=2
#pragma HLS array_partition variable=p cyclic factor=2
#pragma HLS array_partition variable=v block factor=4
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j)
      m[i][j] = p[i + j] + v[i] + n;
}

// Globals and static locals, named from inside the function.
// CHECK-LABEL: func.func @statics(
void statics(int *a) {
  static int st[8];
#pragma HLS array_partition variable=st block factor=2
#pragma HLS array_partition variable=gbuf complete dim=2
#pragma HLS array_partition variable=gbuf cyclic factor=4 dim=1
  for (int i = 0; i < 8; ++i) {
    st[i] += a[i];
    gbuf[i][i % 4] += st[i];
  }
}

// Pragma before the declaration (accepted, as Vitis does).
// CHECK-LABEL: func.func @before_declaration(
// CHECK: {hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "cyclic", variable = "late"}], hls.preserve, polygeist.varname = "late"}
void before_declaration(int *a) {
#pragma HLS array_partition variable=late cyclic factor=2
  int late[8];
  for (int i = 0; i < 8; ++i)
    late[i] = a[i];
  for (int i = 0; i < 8; ++i)
    a[7 - i] = late[i];
}

// Shadowing: the pragma names the variable visible at its location.
// CHECK-LABEL: func.func @shadowing(
// CHECK-DAG: {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "buf"}], hls.preserve, polygeist.varname = "buf"} : memref<4xi32>
// CHECK-DAG: {hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "cyclic", variable = "buf"}], hls.preserve, polygeist.varname = "buf"} : memref<16xi32>
void shadowing(int *a) {
  int buf[16];
#pragma HLS array_partition variable=buf cyclic factor=2
  for (int i = 0; i < 16; ++i)
    buf[i] = a[i];
  {
    int buf[4];
#pragma HLS array_partition variable=buf complete
    for (int i = 0; i < 4; ++i)
      buf[i] = a[i];
    for (int i = 0; i < 4; ++i)
      a[i] = buf[3 - i];
  }
  for (int i = 0; i < 16; ++i)
    a[i] += buf[15 - i];
}

// A local array declared inside a loop body: the alloca is hoisted out of
// the loop and keeps the attribute; the pragma inside the loop names it.
// CHECK-LABEL: func.func @in_loop_body(
// CHECK: memref.alloca() {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "row"}], hls.preserve, polygeist.varname = "row"} : memref<4xi32>
void in_loop_body(int *a) {
  for (int i = 0; i < 8; ++i) {
    int row[4];
#pragma HLS array_partition variable=row complete
    for (int j = 0; j < 4; ++j)
      row[j] = a[i * 4 + j];
    for (int j = 0; j < 4; ++j)
      a[i * 4 + j] = row[3 - j];
  }
}

// Struct members: the pragma annotates the struct's storage and records
// the member path.
struct Pair {
  int x[8];
  int y[8];
};
// CHECK-LABEL: func.func @members(
// A member path records the member's position in the struct (member_path),
// for hls-est's struct splitting (CIRCT OPEN_ISSUES O78).
// CHECK: {hls.array_partition = [{dim = 1 : i32, kind = "complete", member_path = [0 : i32], variable = "s.x"}, {dim = 1 : i32, factor = 2 : i32, kind = "cyclic", member_path = [1 : i32], variable = "s.y"}]
int members(int *a) {
  struct Pair s;
#pragma HLS array_partition variable=s.x complete
#pragma HLS array_partition variable=s.y cyclic factor=2
  for (int i = 0; i < 8; ++i) {
    s.x[i] = a[i];
    s.y[i] = a[i + 8];
  }
  int r = 0;
  for (int i = 0; i < 8; ++i)
    r += s.x[i] * s.y[7 - i];
  return r;
}

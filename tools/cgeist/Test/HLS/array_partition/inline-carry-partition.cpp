// A callee parameter's array_partition reaches the root array of the
// argument when cgeist inlines the call (also a misfit `inline off` call,
// whose argument's first dimension differs from the parameter's), unless the
// parameter's first dimension is smaller than the root's: then Vitis applies
// it one dimension deeper in the root (parameter dimension d to root
// dimension d + 1, dim=0 to dimensions 2..rank) and ignores what falls past
// the root's rank (HLS 214-253). The parameters in between do not decide; a caller's own
// parameter carries the entry with the first dimension of the parameter that
// had the pragma (`rows`) until a root is reached. VITIS_NOTES N123 in the
// hls-est tree, kernels w9r16p_pm_mis (2 BRAM), pm_big (8), pm_two2 (8),
// w9r16i_pm_kept (2); W9r16 review fixes: w9r16rv_pm_dim1 (dim=1 cyclic 2:
// Vitis "Cyclic partitioning with factor 2 on dimension 2", 4 BRAM),
// w9r16fx_pm_d2cyc (dim=2 cyclic 2: ignored, 2) and w9r16fx_pm_d0 (dim=0
// complete: "Complete partitioning on dimension 2" and 214-253, 8).
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - 2>/dev/null | FileCheck %s
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-inline-carry-partition=false -o - 2>/dev/null | FileCheck %s --check-prefix=OFF
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-keep-misfit-inline-off -o - 2>/dev/null | FileCheck %s --check-prefix=KEEP

// CHECK-DAG: memref.global @g_big : memref<100x8xi32> = uninitialized {hls.array_partition = [{dim = 2 : i32, kind = "complete", variable = "a"}]}
// CHECK-DAG: memref.global @g_kept : memref<100x8xi32> = uninitialized{{ *$}}
// CHECK-DAG: memref.global @g_mis : memref<100x8xi32> = uninitialized{{ *$}}
// CHECK-DAG: memref.global @g_dim1 : memref<100x8xi32> = uninitialized {hls.array_partition = [{dim = 2 : i32, factor = 2 : i32, kind = "cyclic", variable = "a"}]}
// CHECK-DAG: memref.global @g_d2c : memref<100x8xi32> = uninitialized{{ *$}}
// CHECK-DAG: memref.global @g_d0 : memref<100x8xi32> = uninitialized {hls.array_partition = [{dim = 2 : i32, kind = "complete", variable = "a"}]}
// CHECK-DAG: memref.global @g_two : memref<100x8xi32> = uninitialized {hls.array_partition = [{dim = 2 : i32, kind = "complete", variable = "a"}]}
// CHECK-LABEL: func.func @_Z6w_keptPA8_iPii(
// CHECK-SAME: hls.array_partition = [{dim = 2 : i32, kind = "complete", rows = 50 : i64, variable = "a"}]
// WARN-DAG: warning: array_partition on parameter 'a' of 'f_mis' ignored past dimension 2: its first dimension (50) is smaller than that of the array it reaches (100)
// WARN-DAG: warning: array_partition on parameter 'a' of 'f_d2c' ignored past dimension 2
// WARN-DAG: warning: array_partition on parameter 'a' of 'f_d0' ignored past dimension 2
// WARN-NOT: array_partition on parameter 'a' of 'f_dim1' 
// OFF-NOT: memref.global {{.*}}hls.array_partition
// OFF-NOT: rows =
// --hls-keep-misfit-inline-off (off by default) keeps a misfit call whose
// argument is reinterpreted to the parameter type, but not when a partition
// is involved (the callee's parameter here: hls-est cannot yet bank through
// the reinterpreted view, OPEN_ISSUES O167 (a) in the hls-est tree), so f_two
// is still inlined and its partition carried; the calls without a partition
// are kept (keep-misfit-global.cpp):
// KEEP-LABEL: func.func @_Z5w_twoPA8_iPii(
// KEEP-NOT: call @_Z5f_twoPA8_iPii(
// KEEP: return

int g_mis[100][8], g_big[100][8], g_two[100][8], g_kept[100][8];
int g_dim1[100][8], g_d2c[100][8], g_d0[100][8];

void f_mis(int a[50][8], int out[8], int n) {
#pragma HLS inline off
#pragma HLS array_partition variable=a dim=2 complete
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 8; j++) out[j] += a[i][j];
}
void top_mis(int in[800], int out[8], int n) {
  for (int i = 0; i < 800; i++) g_mis[i / 8][i % 8] = in[i];
  f_mis(g_mis, out, n);
}

void f_big(int a[200][8], int out[8], int n) {
#pragma HLS inline off
#pragma HLS array_partition variable=a dim=2 complete
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 8; j++) out[j] += a[i][j];
}
void top_big(int in[800], int out[8], int n) {
  for (int i = 0; i < 800; i++) g_big[i / 8][i % 8] = in[i];
  f_big(g_big, out, n);
}

void f_two(int a[100][8], int out[8], int n) {
#pragma HLS inline off
#pragma HLS array_partition variable=a dim=2 complete
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 8; j++) out[j] += a[i][j];
}
void w_two(int a[200][8], int out[8], int n) {
#pragma HLS inline off
  f_two(a, out, n);
}
void top_two(int in[800], int out[8], int n) {
  for (int i = 0; i < 800; i++) g_two[i / 8][i % 8] = in[i];
  w_two(g_two, out, n);
}

void f_kept(int a[50][8], int out[8], int n) {
#pragma HLS inline off
#pragma HLS array_partition variable=a dim=2 complete
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 8; j++) out[j] += a[i][j];
}
void w_kept(int a[100][8], int out[8], int n) {
#pragma HLS inline off
  f_kept(a, out, n);
}
void top_kept(int in[800], int out[8], int n) {
  for (int i = 0; i < 800; i++) g_kept[i / 8][i % 8] = in[i];
  w_kept(g_kept, out, n);
}

void f_dim1(int a[50][8], int out[8], int n) {
#pragma HLS inline off
#pragma HLS array_partition variable=a dim=1 cyclic factor=2
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 8; j++) out[j] += a[2 * i][j] + a[2 * i + 1][j];
}
void top_dim1(int in[800], int out[8], int n) {
  for (int i = 0; i < 800; i++) g_dim1[i / 8][i % 8] = in[i];
  f_dim1(g_dim1, out, n);
}

void f_d2c(int a[50][8], int out[8], int n) {
#pragma HLS inline off
#pragma HLS array_partition variable=a dim=2 cyclic factor=2
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 8; j++) out[j] += a[i][j];
}
void top_d2c(int in[800], int out[8], int n) {
  for (int i = 0; i < 800; i++) g_d2c[i / 8][i % 8] = in[i];
  f_d2c(g_d2c, out, n);
}

void f_d0(int a[50][8], int out[8], int n) {
#pragma HLS inline off
#pragma HLS array_partition variable=a dim=0 complete
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 8; j++) out[j] += a[i][j];
}
void top_d0(int in[800], int out[8], int n) {
  for (int i = 0; i < 800; i++) g_d0[i / 8][i % 8] = in[i];
  f_d0(g_d0, out, n);
}

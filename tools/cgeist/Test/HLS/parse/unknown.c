// Unknown HLS pragmas are preserved verbatim on the nearest loop or function
// with a warning; unknown options are dropped with a warning; pragmas of
// other namespaces keep clang's behaviour.
// RUN: cgeist %s --function='*' -S --hls-annotate -o %t.mlir 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: FileCheck %s < %t.mlir
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @in_function(
// CHECK-SAME: hls.unknown_pragmas = ["frobnicate variable=a core=RAM_1P"]
void in_function(int *a) {
// WARN: unknown.c:[[@LINE+1]]:13: warning: unknown HLS pragma 'frobnicate'; it is preserved in 'hls.unknown_pragmas'
#pragma HLS frobnicate variable=a core=RAM_1P
  a[0] = 1;
}

// CHECK-LABEL: func.func @in_loop(
// CHECK: } {hls.unknown_pragmas = ["loop_name L1", "Fancy_Thing x = 1, 2"]}
void in_loop(int *a) {
  for (int i = 0; i < 8; ++i) {
// WARN: unknown.c:[[@LINE+1]]:13: warning: unknown HLS pragma 'loop_name'
#pragma HLS loop_name L1
// WARN: unknown.c:[[@LINE+1]]:13: warning: unknown HLS pragma 'Fancy_Thing'
#pragma HLS Fancy_Thing x = 1, 2
    a[i] = i;
  }
}

// A pragma with an option Vitis does not know is dropped as a whole, as
// Vitis 2025.1 drops it ([HLS 207-5570], VITIS_NOTES N108; every pragma
// goes through the open front end's XlxPragmaArgParser).
// CHECK-LABEL: func.func @unknown_option(
// CHECK-NOT: hls.pipeline
// CHECK: return
void unknown_option(int *a) {
  for (int i = 0; i < 8; ++i) {
// WARN: unknown.c:[[@LINE+1]]:27: warning: unknown option 'enable_turbo' in '#pragma HLS pipeline'; the pragma is ignored, as Vitis ignores it
#pragma HLS pipeline II=1 enable_turbo=yes
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @partition_unknown_option(
// CHECK-NOT: hls.array_partition
// CHECK: return
int partition_unknown_option(int i) {
  int b[16];
// WARN: unknown.c:[[@LINE+1]]:{{[0-9]+}}: warning: unknown option 'turbo' in '#pragma HLS array_partition'; the pragma is ignored, as Vitis ignores it
#pragma HLS array_partition variable=b type=cyclic factor=4 turbo
  for (int k = 0; k < 16; ++k)
    b[k] = k * i;
  return b[i & 15];
}

// The bare compact mode of AGGREGATE (`bit`, `byte`, `none`, `auto`) is an
// option Vitis knows (the open front end's presentId): no warning.
struct pair_t { int x; char y; };
// CHECK-LABEL: func.func @aggregate_bare_mode(
void aggregate_bare_mode(struct pair_t *p) {
#pragma HLS aggregate variable=p bit
  p[0].x = 1;
}

// An INTERFACE pragma with an option Vitis does not know is dropped as a
// whole, as Vitis 2025.1 drops it ([HLS 207-5570], VITIS_NOTES N108): the
// port is not m_axi.
// CHECK-LABEL: func.func @interface_unknown_option(
// CHECK-NOT: m_axi
// CHECK: return
void interface_unknown_option(int *a) {
// WARN: unknown.c:[[@LINE+1]]:72: warning: unknown option 'bus_bitwidth' in '#pragma HLS interface'; the pragma is ignored, as Vitis ignores it
#pragma HLS interface m_axi port=a bundle=g0 max_read_burst_length=512 bus_bitwidth=512
  a[0] = 1;
}

// CHECK-LABEL: func.func @interface_known_options(
// CHECK-SAME: m_axi
void interface_known_options(int *a) {
#pragma HLS interface m_axi port=a bundle=g0 max_read_burst_length=256
  a[0] = 1;
}

// Pragmas of other namespaces are not ours (clang ignores them silently by
// default, and so do we).
// CHECK-LABEL: func.func @other_namespace(
// CHECK-NOT: hls.unknown_pragmas
// CHECK-NOT: hls.unroll
// CHECK: return
void other_namespace(int *a) {
#pragma foo bar
#pragma GCC unroll 4
  for (int i = 0; i < 8; ++i)
    a[i] = i;
}

// WARN-NOT: warning

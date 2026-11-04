// #pragma HLS stream: hls.stream = [{variable, depth, type}] on the storage of
// the variable (arrays between dataflow processes, parameters, globals).
// depth defaults to 2 (marked depth_defaulted: Vitis sizes an array FIFO by
// the array then) and type to "fifo"; `off` gives hls.stream_off.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,KEEP
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --hls-auto-inline=false 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

int gchan[16];
// CHECK-DAG: memref.global @gchan : memref<16xi32> = uninitialized {hls.stream = [{depth = 16 : i32, type = "shared", variable = "gchan"}]}

static void stage1(int *in, int *mid) {
  for (int i = 0; i < 16; ++i)
    mid[i] = in[i] + 1;
}
static void stage2(int *mid, int *out) {
  for (int i = 0; i < 16; ++i)
    out[i] = mid[i] * 2;
}

// CHECK-LABEL: func.func @pipeline_stages(
// CHECK-DAG: {hls.preserve, hls.stream = [{depth = 2 : i32, depth_defaulted, type = "fifo", variable = "a"}], polygeist.varname = "a"} : memref<16xi32>
// CHECK-DAG: {hls.preserve, hls.stream = [{depth = 8 : i32, type = "pipo", variable = "b"}], polygeist.varname = "b"} : memref<16xi32>
// CHECK-DAG: {hls.preserve, hls.stream_off = [{variable = "c"}], polygeist.varname = "c"} : memref<16xi32>
void pipeline_stages(int *in, int *out) {
#pragma HLS dataflow
  int a[16], b[16], c[16];
#pragma HLS stream variable=a
#pragma HLS STREAM variable=b depth=8 type=pipo
#pragma HLS stream variable=c off
  stage1(in, a);
  stage2(a, b);
  stage1(b, c);
  stage2(c, out);
}

// CHECK-LABEL: func.func @port(
// CHECK-SAME: %arg0: memref<{{.*}}> {hls.stream = [{depth = 4 : i32, type = "unsync", variable = "p"}]}
void port(int *p, int *q) {
#pragma HLS stream variable=p depth=4 type=unsync
#pragma HLS stream variable=gchan depth=16 type=shared
  for (int i = 0; i < 16; ++i) {
    gchan[i] = p[i];
    q[i] = gchan[i];
  }
}

// CHECK-LABEL: func.func @top(
// NOINLINE: call @pipeline_stages
// A function that holds a dataflow region is never inlined by Vitis (VITIS_NOTES
// N18, --hls-auto-inline): the call stays; --hls-auto-inline=false inlines it.
// KEEP: call @pipeline_stages({{.*}}hls.inline_auto_off
// INLINE: hls.stream = [{depth = 8 : i32, type = "pipo", variable = "b"}]
void top(int *in, int *out) { pipeline_stages(in, out); }

#ifdef ERRORS
void errors(int *p) {
  int a[4];
// ERR-DAG: stream.c:[[@LINE+1]]:37: error: option 'depth' in '#pragma HLS stream' must be >= 1, got 0
#pragma HLS stream variable=a depth=0
// ERR-DAG: stream.c:[[@LINE+1]]:36: error: invalid value 'lifo' for option 'type' in '#pragma HLS stream': expected one of: fifo, pipo, shared, unsync
#pragma HLS stream variable=a type=lifo
#pragma HLS stream variable=a depth=4
// ERR-DAG: stream.c:[[@LINE+1]]:1: error: '#pragma HLS stream' conflicts with an earlier pragma on the same variable 'a'
#pragma HLS stream variable=a depth=8
  a[0] = p[0];
  p[1] = a[0];
}
#endif

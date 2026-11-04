// An AGGREGATE pragma reaches the variable a call passes to the annotated
// parameter (Vitis 2025.1: [HLS 214-241] "Aggregating bram variable" names
// the caller's variable, kernels w9s_b and w9v_s_inl of the hls-est notes,
// VITIS_NOTES N37 S2 (b)), also through a chain of calls and when the callee
// is inlined. `variable=this` inside a member function names the object it is
// called on (Vitis's hls_vector.h runs `SYN_PRAGMA(HLS AGGREGATE
// variable=this)` in its members, N37 S2 (c)). The variable gets
// hls.aggregate with `from` = the callee the actual was passed to.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --memref-fullrank -DINLINED 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --memref-fullrank --hls-propagate-aggregate=false 2>&1 | FileCheck %s --check-prefix=OFF

// CHECK-NOT: error

template <typename T, int N> struct vec {
  T d[N];
  T &operator[](int i) {
#pragma HLS aggregate variable=this
    return d[i];
  }
  void clear() {
    for (int i = 0; i < N; ++i)
      d[i] = 0;
  }
};
struct plain {
  short d[8];
};

#ifdef INLINED
#define KEEP
#else
#define KEEP _Pragma("HLS inline off")
#endif

static void fill(plain p[64], const short *in) {
  KEEP
#pragma HLS aggregate variable=p
  for (int i = 0; i < 64 * 8; i++)
    p[i >> 3].d[i & 7] = in[i];
}
static void relay(plain q[64], const short *in) {
  KEEP
  fill(q, in);
}

// CHECK: memref.global @gv {{.*}}hls.aggregate = [{compact = "auto", from = "{{.*}}vec{{.*}}", variable = "this"}]
// OFF: memref.global @gv : memref<32x!llvm.struct<(array<8 x i16>)>> = uninitialized{{$}}
vec<short, 8> gv[32];

// CHECK-LABEL: func.func @top(
// CHECK-DAG: hls.aggregate = [{compact = "auto", from = "{{.*}}relay{{.*}}", variable = "p"}], hls.preserve, polygeist.varname = "pa"
// CHECK-DAG: hls.aggregate = [{compact = "auto", from = "{{.*}}vec{{.*}}", variable = "this"}], hls.preserve, polygeist.varname = "va"
// CHECK-DAG: memref.alloca() {hls.preserve, polygeist.varname = "vc"}
// CHECK-DAG: memref.alloca() {hls.preserve, polygeist.varname = "pb"}
// OFF-LABEL: func.func @top(
// OFF-NOT: hls.aggregate
// OFF: return
extern "C" void top(const short in[512], short out[4], int k) {
  vec<short, 8> va[64]; // operator[] applied: aggregated through `this`
  vec<short, 8> vc[64]; // only clear(): not aggregated
  plain pa[64];         // reaches fill's parameter through relay
  plain pb[64];         // no pragma
  for (int i = 0; i < 64 * 8; i++)
    va[i >> 3][i & 7] = in[i];
  for (int i = 0; i < 64; i++)
    vc[i].clear();
  for (int i = 0; i < 32 * 8; i++)
    gv[i >> 3][i & 7] = in[i];
  relay(pa, in);
  for (int i = 0; i < 64 * 8; i++)
    pb[i >> 3].d[i & 7] = in[i];
  int s = 0;
  for (int i = 0; i < 512; i++) {
    int j = (i * k) & 511;
    s += va[j >> 3][j & 7] + vc[j >> 3].d[j & 7] + pa[j >> 3].d[j & 7] +
         pb[j >> 3].d[j & 7] + gv[(j >> 3) & 31][j & 7];
  }
  out[0] = s;
}

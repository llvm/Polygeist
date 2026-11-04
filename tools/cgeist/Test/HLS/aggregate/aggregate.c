// #pragma HLS aggregate / disaggregate: hls.aggregate = [{variable, compact}]
// (compact defaults to "auto") and hls.disaggregate = [{variable}] on the
// storage of a struct variable, parameter or global. The two conflict on
// the same variable.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

struct Pixel {
  unsigned char r, g, b, a;
};
struct Pair {
  int x[4];
  int y[4];
};

// CHECK-LABEL: func.func @params(
// CHECK-SAME: {hls.aggregate = [{compact = "bit", variable = "in"}]}
// CHECK-SAME: {hls.aggregate = [{compact = "auto", variable = "out"}]}
void params(struct Pixel in[16], struct Pixel out[16]) {
#pragma HLS aggregate variable=in compact=bit
#pragma HLS AGGREGATE variable=out
  // (Field-wise: cgeist cannot yet copy whole struct array elements.)
  for (int i = 0; i < 16; ++i) {
    out[i].r = in[i].r;
    out[i].g = in[i].g;
    out[i].b = in[i].b;
    out[i].a = in[i].a;
  }
}

// CHECK-LABEL: func.func @locals(
// CHECK-DAG: hls.aggregate = [{compact = "byte", variable = "p"}]
// CHECK-DAG: hls.disaggregate = [{variable = "q"}]
int locals(int *a) {
  struct Pair p, q;
#pragma HLS aggregate variable=p compact=byte
#pragma HLS disaggregate variable=q
  for (int i = 0; i < 4; ++i) {
    p.x[i] = a[i];
    p.y[i] = a[i + 4];
    q.x[i] = a[i] * 2;
    q.y[i] = a[i] * 3;
  }
  int s = 0;
  for (int i = 0; i < 4; ++i)
    s += p.x[3 - i] * q.y[i] + p.y[i] * q.x[3 - i];
  return s;
}

// cgeist's inliner does not inline functions with struct-typed locals, so
// `locals` stays a call at every -O level; its attributes stay on its own
// allocas (checked above).
// CHECK-LABEL: func.func @top(
// CHECK: call @locals
int top(int *a) { return locals(a); }

#ifdef ERRORS
void errors(struct Pixel *px) {
#pragma HLS aggregate variable=px
// ERR-DAG: aggregate.c:[[@LINE+1]]:1: error: '#pragma HLS disaggregate' conflicts with an earlier pragma on the same variable 'px'
#pragma HLS disaggregate variable=px
// ERR-DAG: aggregate.c:[[@LINE+1]]:43: error: invalid value 'nibble' for option 'compact' in '#pragma HLS aggregate'
#pragma HLS aggregate variable=px compact=nibble
  px[0].r = 1;
}
#endif

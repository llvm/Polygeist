// #pragma HLS inline: hls.inline = [{recursive}] on the enclosing function;
// `off` gives hls.inline_off. Placement anywhere in the body (also inside a
// loop) applies to the function. cgeist's inliner (-O1 and up) does not
// inline a function marked `inline off` (regressions/inline-off-kept-call.cpp);
// these RUN lines use -O0, where nothing is inlined.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -O0 -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @_Z5plainPi(
// CHECK-SAME: hls.inline = [{recursive = false}]
void plain(int *a) {
#pragma HLS inline
  a[0] += 1;
}

// CHECK-LABEL: func.func @_Z9recursivePi(
// CHECK-SAME: hls.inline = [{recursive = true}]
void recursive(int *a) {
#pragma HLS INLINE recursive
  plain(a);
}

// CHECK-LABEL: func.func @_Z3offPi(
// CHECK-SAME: hls.inline_off
// CHECK-NOT: hls.inline =
void off(int *a) {
#pragma HLS inline off
  a[0] += 2;
}

// Inside a loop: still the function (inline is function-only).
// CHECK-LABEL: func.func @_Z11inside_loopPi(
// CHECK-SAME: hls.inline_off
void inside_loop(int *a) {
  for (int i = 0; i < 4; ++i) {
#pragma HLS inline off
    a[i] = i;
  }
}

// Methods and templates.
struct Acc {
  int v;
  void add(int x) {
#pragma HLS inline
    v += x;
  }
};
template <typename T> T twice(T x) {
#pragma HLS inline off
  return x + x;
}
// CHECK-LABEL: func.func @_Z3useP3Accf(
int use(Acc *acc, float f) {
  acc->add(3);
  return twice(4) + (int)twice(f);
}
// CHECK-DAG: func.func @_ZN3Acc3addEi({{.*}} attributes {{{.*}}hls.inline = [{recursive = false}]
// CHECK-DAG: func.func @_Z5twiceIiET_S0_({{.*}} attributes {{{.*}}hls.inline_off
// CHECK-DAG: func.func @_Z5twiceIfET_S0_({{.*}} attributes {{{.*}}hls.inline_off

#ifdef ERRORS
void errors(int *a) {
// ERR-DAG: inline.cpp:[[@LINE+1]]:1: error: 'recursive' conflicts with 'off' in '#pragma HLS inline'
#pragma HLS inline recursive off
  a[0] = 0;
}
void errors2(int *a) {
#pragma HLS inline
// ERR-DAG: inline.cpp:[[@LINE+1]]:1: error: '#pragma HLS inline' conflicts with an earlier pragma on the same function
#pragma HLS inline off
  a[0] = 0;
}
#endif

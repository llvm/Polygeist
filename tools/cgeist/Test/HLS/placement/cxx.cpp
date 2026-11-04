// Placement in C++ constructs: range-based for, lambdas (a pragma belongs to
// the innermost function body that contains it), methods, templates that
// are only emitted when instantiated.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --strict-whitespace
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 --raise-scf-to-affine 2>&1 | FileCheck %s --strict-whitespace

// CHECK-NOT: warning
// CHECK-NOT: error

// CHECK-LABEL: func.func @_Z9range_forRA8_i(
// CHECK: } attributes {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void range_for(int (&a)[8]) {
  for (int &x : a) {
#pragma HLS pipeline II=1
    x += 1;
  }
}

struct Filter {
  int taps[4];
  void apply(int *out) {
    for (int i = 0; i < 4; ++i) {
#pragma HLS unroll
      out[i] = taps[i] * 2;
    }
  }
};
void use_filter(Filter *f, int *o) { f->apply(o); }
// CHECK-LABEL: func.func @_ZN6Filter5applyEPi(
// CHECK: } {hls.unroll = [{factor = 0 : i32, skip_exit_check = false}]}

// Never instantiated: its pragmas are neither emitted nor diagnosed
// (even when they would be invalid for some instantiation).
template <typename T> void unused_template(T *a) {
#pragma HLS array_partition variable=a cyclic factor=4
#pragma HLS pipeline II=1
  a[0] = 0;
}

// A pragma inside a lambda applies to the lambda's loop, not to the loop
// of the enclosing function.
// CHECK-LABEL: func.func @_Z11with_lambdaPi(
// CHECK: {{^}}    } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
void with_lambda(int *a) {
  auto body = [](int *p, int i) {
    for (int k = 0; k < 2; ++k) {
#pragma HLS unroll
      p[i + k] = k;
    }
  };
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=2
    body(a, 2 * i);
  }
}
// CHECK-LABEL: func.func private @_ZZ11with_lambdaPiENK3$_0clES_i(
// CHECK: } {hls.unroll = [{factor = 0 : i32, skip_exit_check = false}]}

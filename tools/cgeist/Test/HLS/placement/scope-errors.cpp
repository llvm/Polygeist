// Pragmas outside any function body are errors: at file scope, in namespace
// scope, and in class bodies (outside member function bodies).
// RUN: not cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s

// CHECK: scope-errors.cpp:[[@LINE+1]]:1: error: '#pragma HLS pipeline' must be placed inside a function body
#pragma HLS pipeline II=1

namespace ns {
// CHECK: scope-errors.cpp:[[@LINE+1]]:1: error: '#pragma HLS array_partition' must be placed inside a function body
#pragma HLS array_partition variable=x complete
int x[4];
} // namespace ns

struct S {
// CHECK: scope-errors.cpp:[[@LINE+1]]:1: error: '#pragma HLS inline' must be placed inside a function body
#pragma HLS inline off
  int v;
  void f(int *a) {
#pragma HLS inline off
    a[0] = v;
  }
};

// Between functions.
void g(int *a) { a[0] = 1; }
// CHECK: scope-errors.cpp:[[@LINE+1]]:1: error: '#pragma HLS dataflow' must be placed inside a function body
#pragma HLS dataflow
void h(int *a) { a[0] = 2; }

void use(S *s, int *a) { s->f(a); }

// CHECK-NOT: error

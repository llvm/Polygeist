// Regression: with --hls-annotate, emitting a function whose name is not a
// plain identifier (a lambda's operator(), an overloaded operator) asserted
// in NamedDecl::getName() while setting hls.orig_name.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s

// CHECK-NOT: Assertion
// CHECK-DAG: hls.orig_name = "operator()"
// CHECK-DAG: hls.orig_name = "operator+"
struct V {
  int x;
};
V operator+(V a, V b) { return V{a.x + b.x}; }

int f(int *a) {
  auto add = [](int x, int y) { return x + y; };
  V v = V{a[0]} + V{a[1]};
  return add(v.x, a[2]);
}

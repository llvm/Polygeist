// A source with an error stops cgeist with clang's diagnostic. The size
// consumer of --hls-auto-inline (HLSAutoInline.cc, InlineSizeConsumer) must
// not run clang's LLVM codegen over the invalid AST: it crashed in "LLVM IR
// generation of compound statement" and turned the error into a crash (soda
// designs, `no viable overloaded '='`).
// RUN: not cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate --hls-auto-inline=false -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: no viable overloaded '='
// CHECK-NOT: Stack dump

struct A { int v; };
struct B { int w; };
int g(int x) { return x + 1; }
void top(int *a) {
  A x;
  B y;
  y.w = a[1];
  x = y;
  a[0] = g(x.v);
}

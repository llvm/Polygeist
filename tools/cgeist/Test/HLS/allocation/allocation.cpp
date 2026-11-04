// #pragma HLS allocation: hls.allocation = [{instances, limit, type}] on the
// enclosing function (or loop); instances is a list of names; type is
// "function" (default) or "operation", also accepted positionally.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

template <typename T> T mac(T a, T b, T c) { return a * b + c; }
int sub(int a, int b) { return a - b; }

// Several allocation pragmas accumulate in source order.
// CHECK-LABEL: func.func @_Z3topPiS_(
// CHECK-SAME: hls.allocation = [{instances = ["mac<int>"], limit = 1 : i32, type = "function"}, {instances = ["mul"], limit = 2 : i32, type = "operation"}, {instances = ["sub", "mac<int>"], limit = 3 : i32, type = "function"}, {instances = ["add"], limit = 4 : i32, type = "operation"}]
void top(int *a, int *b) {
#pragma HLS ALLOCATION function instances=mac<int> limit=1
#pragma HLS allocation operation instances=mul limit=2
#pragma HLS allocation instances=sub, mac<int> limit=3 type=function
#pragma HLS allocation type=operation instances=add limit=4
  for (int i = 0; i < 8; ++i)
    b[i] = mac(a[i], a[i], sub(b[i], 1));
}

// Loop region.
// CHECK-LABEL: func.func @_Z4loopPi(
// CHECK: } {hls.allocation = [{instances = ["mul"], limit = 1 : i32, type = "operation"}]}
void loop(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS allocation operation instances=mul limit=1
    a[i] = a[i] * a[i] * 3;
  }
}

#ifdef ERRORS
void errors(int *a) {
// ERR-DAG: allocation.cpp:[[@LINE+1]]:1: error: '#pragma HLS allocation' requires option 'limit'
#pragma HLS allocation function instances=sub
// ERR-DAG: allocation.cpp:[[@LINE+1]]:1: error: '#pragma HLS allocation' requires option 'instances'
#pragma HLS allocation function limit=1
// ERR-DAG: allocation.cpp:[[@LINE+1]]:53: error: option 'limit' in '#pragma HLS allocation' must be >= 1, got 0
#pragma HLS allocation function instances=sub limit=0
#pragma HLS allocation function instances=sub limit=2
// ERR-DAG: allocation.cpp:[[@LINE+1]]:1: warning: duplicate '#pragma HLS allocation' on the same function is ignored
#pragma HLS allocation function instances=sub limit=2
  a[0] = sub(a[0], 1);
}
#endif

// RUN: mlir-clang -S -O0 %s | FileCheck %s

void foo(int A[10]) {
#pragma scop
  for (int i = 0; i < 10; ++i)
    A[i] = A[i] * 2;
#pragma endscop
}

// CHECK-LABEL: func @main()
// CHECK: call @foo
int main() {
  int A[10];
  foo(A);
  return 0;
}

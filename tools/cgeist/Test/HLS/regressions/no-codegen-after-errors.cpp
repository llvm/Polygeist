// A translation unit with errors gets no code, as with clang's own CodeGen:
// cgeist reports the errors and exits with a failure instead of emitting the
// invalid declarations. It used to assert in ValueCategory::getValue when the
// function using an unknown type was selected by its source name
// (class_ece8893_2023_spring FocusedACO LocalSearchOptimized, `uint32_t`
// without <stdint.h>, W8b).
// RUN: not cgeist %s %resourcedir --function=top -S -o - 2>&1 | FileCheck %s
// RUN: not cgeist %s %resourcedir --function='*' -S -o - 2>&1 | FileCheck %s

// CHECK: error: unknown type name 'uint32_t'
// CHECK-NOT: Assertion
// CHECK-NOT: func.func
int top(uint32_t a, int b[4]) {
  uint32_t s = a;
  for (int i = 0; i < 4; i++)
    s += b[i];
  return s;
}

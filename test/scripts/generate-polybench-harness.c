// RUN: %python %S/../../scripts/correctness/generate_polybench_harness.py %s --function kernel_fixture -o %t.c
// RUN: FileCheck %s --check-prefix=HARNESS < %t.c
// RUN: %clang -c %t.c -o %t.o
// RUN: nm %t.o | FileCheck %s --check-prefix=SYMBOL

static void helper(void) {}

static
void kernel_fixture(int value)
{
  // A brace in a comment must not terminate the function: }
  const char *text = "{";
  (void)value;
  (void)text;
}

int main(void) {
  helper();
  kernel_fixture(7);
  return 0;
}

// HARNESS: void kernel_fixture(int value);
// HARNESS-NOT: A brace in a comment
// HARNESS: kernel_fixture(7);
// SYMBOL: U kernel_fixture

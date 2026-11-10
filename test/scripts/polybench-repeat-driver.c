// RUN: clang -DPOLYGEIST_WARMUP_RUNS=2 -DPOLYGEIST_TIMED_RUNS=3 \
// RUN:   %S/../../runtime/polybench_repeat_driver.c %s -o %t
// RUN: %t 2>&1 | FileCheck %s

#include <stdio.h>

int polygeist_polybench_iteration(int argc, char **argv) {
  (void)argc;
  (void)argv;
  puts("ENTRY");
  return 0;
}

// CHECK: POLYGEIST_BENCHMARK_ITERATION phase=warmup index=0 total=5
// CHECK-NEXT: ENTRY
// CHECK-NEXT: POLYGEIST_BENCHMARK_ITERATION phase=warmup index=1 total=5
// CHECK-NEXT: ENTRY
// CHECK-NEXT: POLYGEIST_BENCHMARK_ITERATION phase=sample index=0 total=5
// CHECK-NEXT: ENTRY
// CHECK-NEXT: POLYGEIST_BENCHMARK_ITERATION phase=sample index=1 total=5
// CHECK-NEXT: ENTRY
// CHECK-NEXT: POLYGEIST_BENCHMARK_ITERATION phase=sample index=2 total=5
// CHECK-NEXT: ENTRY

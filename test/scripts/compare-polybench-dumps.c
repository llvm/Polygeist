// RUN: %S/../../scripts/correctness/compare_polybench_dumps.py %S/Inputs/polybench-dump-reference.txt %S/Inputs/polybench-dump-close.txt | FileCheck %s --check-prefix=PASS
// RUN: not %S/../../scripts/correctness/compare_polybench_dumps.py %S/Inputs/polybench-dump-reference.txt %S/Inputs/polybench-dump-far.txt | FileCheck %s --check-prefix=FAIL
// RUN: not %S/../../scripts/correctness/compare_polybench_dumps.py %S/Inputs/polybench-dump-reference.txt %S/Inputs/polybench-dump-nan.txt | FileCheck %s --check-prefix=NAN
// RUN: sed '1iPOLYBENCH_NATIVE_GPU_TIMING device_ms=12.5' %S/Inputs/polybench-dump-close.txt > %t.timed
// RUN: %S/../../scripts/correctness/compare_polybench_dumps.py %S/Inputs/polybench-dump-reference.txt %t.timed | FileCheck %s --check-prefix=PASS

// PASS: PASS values=3 failures=0
// FAIL: FAIL values=3 failures=1
// NAN: FAIL values=3 failures=1
// NAN: first_failure index=1 reference=0.0 candidate=nan

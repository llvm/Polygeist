// N18's size proxy in loop-rotated form (VITIS_NOTES N131 in the hls-est tree,
// OPEN_ISSUES O175): Vitis' step-2 count keeps the guard / exit block LLVM 7's loop
// rotation leaves per loop, so `merge` (three loops with variable bounds) counts 59
// with its casts in Vitis (machsuite sort_merge, td=off) and 52 without them:
// 52 x 2 calls = 104 > 100, kept. The plain proxy gives 48 (96 <= 100, inlined).
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-rotated-size --hls-auto-inline-report -o /dev/null 2>&1 | FileCheck %s --check-prefix=ROT
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-report -o /dev/null 2>&1 | FileCheck %s --check-prefix=OLD
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-rotated-size -o - | FileCheck %s

// ROT: caller 'ms_mergesort' callee 'merge' size 52 calls 2: kept (size x calls > threshold)
// OLD: caller 'ms_mergesort' callee 'merge' size 48 calls 2: inlined (size x calls <= threshold)
// CHECK-COUNT-2: call @merge(

#define SIZE 2048
#define TYPE int

void merge(TYPE a[SIZE], int start, int m, int stop) {
  TYPE temp[SIZE];
  int i, j, k;
  for (i = start; i <= m; i++)
    temp[i] = a[i];
  for (j = m + 1; j <= stop; j++)
    temp[m + 1 + stop - j] = a[j];
  i = start;
  j = stop;
  for (k = start; k <= stop; k++) {
    TYPE tmp_j = temp[j];
    TYPE tmp_i = temp[i];
    if (tmp_j < tmp_i) {
      a[k] = tmp_j;
      j--;
    } else {
      a[k] = tmp_i;
      i++;
    }
  }
}

void ms_mergesort(TYPE a[SIZE]) {
  int start = 0, stop = SIZE;
  int i, m, from, mid, to;
  for (m = 1; m < stop - start; m += m) {
    for (i = start; i < stop; i += m + m) {
      from = i;
      mid = i + m - 1;
      to = i + m + m - 1;
      if (to < stop)
        merge(a, from, mid, to);
      else
        merge(a, from, mid, stop);
    }
  }
}

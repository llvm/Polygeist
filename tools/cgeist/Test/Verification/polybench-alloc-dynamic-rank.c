// RUN: cgeist %s --function=allocate %stdinclude -S | FileCheck %s

#include <stddef.h>

void *polybench_alloc_data(unsigned long long count, int element_size);
void consume(int n, int m, double data[n][m]);

void allocate(int n, int m) {
  double(*data)[n][m] = (double(*)[n][m])polybench_alloc_data(
      (unsigned long long)n * (unsigned long long)m, sizeof(double));
  consume(n, m, *data);
}

// CHECK-LABEL: func.func @allocate(
// CHECK-DAG: %[[N:.*]] = arith.index_cast %arg0 : i32 to index
// CHECK-DAG: %[[M:.*]] = arith.index_cast %arg1 : i32 to index
// CHECK: %[[DATA:.*]] = memref.alloc(%[[N]], %[[M]]) : memref<?x?xf64>
// CHECK: call @consume(%arg0, %arg1, %{{.*}})

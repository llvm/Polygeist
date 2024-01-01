// RUN: cgeist %s --function=* -fopenmp -S | FileCheck %s

int omp_get_thread_num();

void test_parallel_num_threads(double* x, int sinc) {
    // CHECK: %[[c32:.+]] = arith.constant 32 : i32
    // CHECK: omp.parallel   num_threads(%[[c32]] : i32) {
    #pragma omp parallel num_threads(32)
    {
        int tid = omp_get_thread_num();
        x[tid] = 1;
    }
}

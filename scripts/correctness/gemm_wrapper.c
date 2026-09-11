/* C wrapper: bridges the PolyBench-style call to the MLIR-lowered kernel
 * which uses MLIR's bare memref descriptor calling convention.
 *
 * The lowered function `kernel_gemm_impl` expects, for each 2D dynamic
 * memref operand, 7 arguments: (ptr base, ptr aligned, i64 offset,
 * i64 size0, i64 size1, i64 stride0, i64 stride1).
 */
#include <stdint.h>

extern void kernel_gemm_impl(
    int ni, int nj, int nk, double alpha, double beta,
    /* C: memref<?x?xf64> */
    double *C_base, double *C_aligned, int64_t C_offset,
    int64_t C_size0, int64_t C_size1, int64_t C_stride0, int64_t C_stride1,
    /* A: memref<?x?xf64> */
    double *A_base, double *A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    /* B: memref<?x?xf64> */
    double *B_base, double *B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1);

/* PolyBench-style entry. The arrays are passed as VLAs (or pointers in the
 * heap-allocated PolyBench version). For PolyBench's POLYBENCH_USE_C99_PROTO
 * mode the function signature uses VLA syntax; otherwise it's flat double*.
 * We accept double* and use the explicit ni/nj/nk to compute strides. */
void kernel_gemm(int ni, int nj, int nk, double alpha, double beta,
                 double *C, double *A, double *B) {
  kernel_gemm_impl(ni, nj, nk, alpha, beta,
                   C, C, 0, ni, nj, nj, 1,
                   A, A, 0, ni, nk, nk, 1,
                   B, B, 0, nk, nj, nj, 1);
}

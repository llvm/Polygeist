// RUN: FileCheck %s --check-prefix=DIRECT < %S/../../runtime/polygeist_cublas_rt_cuda.c

// Keep the ordinary FP64 GEMM wrapper as one real GEMM call.  A previous
// device-specific workaround expanded it into M row-wise GEMV calls, hiding
// both the intended library selection and most of GEMM's performance.

// DIRECT-LABEL: void polygeist_cublas_dgemm(
// DIRECT: timing_gpu_begin();
// DIRECT-NEXT: CUBLAS_CHECK(cublasDgemm(g_handle,
// DIRECT-NOT: cublasDgemv
// DIRECT: timing_gpu_end("cublasDgemm"

/* Mechanical matrix layout adaptation; arithmetic remains in the vendor API.
 * Included by the CPU and CUDA v2 runtime templates outside generated code. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static void pv2_matrix_fail(const char *reason) {
  fprintf(stderr, "polygeist_v2 matrix layout: %s\n", reason);
  abort();
}

static int64_t pv2_matrix_mul(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || (b && a > INT64_MAX / b))
    pv2_matrix_fail("extent overflow");
  return a * b;
}

static int64_t pv2_matrix_validate(const void *data, int64_t available,
                                  int64_t rows, int64_t cols,
                                  int64_t s0, int64_t s1, int32_t element) {
  if (rows < 0 || cols < 0 || s0 < 0 || s1 < 0 || available < 0 ||
      (element != 1 && element != 2 && element != 4 && element != 8))
    pv2_matrix_fail("invalid descriptor");
  int64_t dense = pv2_matrix_mul(pv2_matrix_mul(rows, cols), element);
  if (!rows || !cols)
    return dense;
  int64_t a = pv2_matrix_mul(rows - 1, s0);
  int64_t b = pv2_matrix_mul(cols - 1, s1);
  if (a > INT64_MAX - b - 1)
    pv2_matrix_fail("footprint overflow");
  int64_t required = pv2_matrix_mul(a + b + 1, element);
  if (!data || required > available || (uint64_t)dense > SIZE_MAX)
    pv2_matrix_fail("insufficient buffer capacity");
  return dense;
}

static int pv2_matrix_device(const void *data) {
#ifdef POLYGEIST_V2_MATRIX_CUDA
  struct cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, data);
  if (status == cudaSuccess)
    return attributes.type == cudaMemoryTypeDevice ||
           attributes.type == cudaMemoryTypeManaged;
  if (status != cudaErrorInvalidValue)
    pv2_matrix_fail(cudaGetErrorString(status));
  (void)cudaGetLastError();
#else
  (void)data;
#endif
  return 0;
}

/* Reject original overlapping operands before separate packed buffers could
 * hide the alias from a vendor wrapper. A single formal inout array is packed
 * once and retains its identity. Contiguous calls keep their original policy. */
void polygeist_v2_matrix_require_disjoint(const void *a, int64_t an,
                                         const void *b, int64_t bn,
                                         int32_t packing) {
  if (!packing || !an || !bn) return;
  if (an < 0 || bn < 0) pv2_matrix_fail("negative buffer extent");
  uintptr_t x=(uintptr_t)a, y=(uintptr_t)b;
  if (x <= y ? y-x < (uint64_t)an : x-y < (uint64_t)bn)
    pv2_matrix_fail("packing distinct overlapping operands is unsupported");
}

static void pv2_matrix_synchronize(int device) {
#ifdef POLYGEIST_V2_MATRIX_CUDA
  if (device) {
    /* The producer/vendor may use a nonblocking pipeline stream. Physical
     * layout adaptation is synchronous and does not promise graph capture. */
    cudaError_t status=cudaDeviceSynchronize();
    if (status != cudaSuccess) pv2_matrix_fail(cudaGetErrorString(status));
  }
#else
  (void)device;
#endif
}

static void pv2_matrix_copy(void *dense, void *view, int64_t rows, int64_t cols,
                            int64_t s0, int64_t s1, int32_t element,
                            int device, int copy_back) {
  unsigned char *d = (unsigned char *)dense;
  unsigned char *v = (unsigned char *)view;
  for (int64_t r = 0; r < rows; ++r) {
    unsigned char *dr = d + r * cols * element;
    unsigned char *vr = v + r * s0 * element;
#ifdef POLYGEIST_V2_MATRIX_CUDA
    if (device && s1 > 0) {
      cudaError_t status = copy_back
          ? cudaMemcpy2D(vr, (size_t)s1 * element, dr, element, element,
                         (size_t)cols, cudaMemcpyDeviceToDevice)
          : cudaMemcpy2D(dr, element, vr, (size_t)s1 * element, element,
                         (size_t)cols, cudaMemcpyDeviceToDevice);
      if (status != cudaSuccess)
        pv2_matrix_fail(cudaGetErrorString(status));
      continue;
    }
#else
    (void)device;
#endif
    for (int64_t c = 0; c < cols; ++c) {
      void *dst = copy_back ? vr + c * s1 * element : dr + c * element;
      const void *src = copy_back ? dr + c * element : vr + c * s1 * element;
#ifdef POLYGEIST_V2_MATRIX_CUDA
      if (device) {
        cudaError_t status = cudaMemcpy(dst, src, element, cudaMemcpyDeviceToDevice);
        if (status != cudaSuccess)
          pv2_matrix_fail(cudaGetErrorString(status));
      } else
#endif
        memcpy(dst, src, element);
    }
  }
}

void *polygeist_v2_matrix_pack(void *data, int64_t available, int64_t rows,
                               int64_t cols, int64_t s0, int64_t s1,
                               int32_t element) {
  int64_t dense = pv2_matrix_validate(data, available, rows, cols, s0, s1, element);
  if (!dense || cols <= 1 || s1 == 1)
    return data;
  int device = pv2_matrix_device(data);
  pv2_matrix_synchronize(device);
  void *packed = NULL;
#ifdef POLYGEIST_V2_MATRIX_CUDA
  if (device) {
    cudaError_t status = cudaMalloc(&packed, (size_t)dense);
    if (status != cudaSuccess)
      pv2_matrix_fail(cudaGetErrorString(status));
  } else
#endif
    packed = malloc((size_t)dense);
  if (!packed)
    pv2_matrix_fail("allocation failed");
  pv2_matrix_copy(packed, data, rows, cols, s0, s1, element, device, 0);
  pv2_matrix_synchronize(device);
  return packed;
}

void polygeist_v2_matrix_release(void *packed, void *data, int64_t available,
                                 int64_t rows, int64_t cols, int64_t s0,
                                 int64_t s1, int32_t element, int32_t written) {
  (void)pv2_matrix_validate(data, available, rows, cols, s0, s1, element);
  if (packed == data)
    return;
  if (!packed)
    pv2_matrix_fail("missing packed buffer");
  if (written && rows > 1 && cols > 1) {
    int64_t a = s0, b = s1;
    while (b) { int64_t t = a % b; a = b; b = t; }
    if (!s0 || !s1 || (s1 / a < rows && s0 / a < cols))
      pv2_matrix_fail("overlapping output coordinates");
  }
  int device = pv2_matrix_device(packed);
  pv2_matrix_synchronize(device);
  if (written) {
    pv2_matrix_copy(packed, data, rows, cols, s0, s1, element, device, 1);
    pv2_matrix_synchronize(device);
  }
#ifdef POLYGEIST_V2_MATRIX_CUDA
  if (device) {
    cudaError_t status = cudaFree(packed);
    if (status != cudaSuccess)
      pv2_matrix_fail(cudaGetErrorString(status));
    return;
  }
#endif
  free(packed);
}

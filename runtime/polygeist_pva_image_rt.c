/* Typed adapters for PVA Solutions image operators.
 *
 * This file only marshals host buffers into NVCV/PVA tensors and calls the
 * vendor pva*Create/pva*Submit entry points.  It contains no replacement
 * implementation of the image operations.  Gated SDK headers are resolved
 * from PVASOL_ROOT at build time and are never vendored into this repository.
 */
#include "polygeist_cublas_rt.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cupva_host.h>
#include <nvcv/Tensor.h>
#include <nvcv/TensorData.h>
#include <nvcv/alloc/Allocator.h>

#include <OpBilateralFilter.h>
#include <OpBoxFilter.h>
#include <OpGaussianFilter.h>
#include <OpHistogramEqualization.h>
#include <OpImageHistogram.h>
#include <OpMorphology.h>
#include <PvaAllocator.h>

#define NVCV_CHECK(call) do {                                                \
  NVCVStatus status_ = (call);                                               \
  if (status_ != NVCV_SUCCESS) {                                             \
    fprintf(stderr, "polygeist PVA: %s failed with NVCV status %d\n",       \
            #call, (int)status_);                                            \
    abort();                                                                 \
  }                                                                          \
} while (0)

#define CUDA_CHECK(call) do {                                                \
  cudaError_t status_ = (call);                                              \
  if (status_ != cudaSuccess) {                                              \
    fprintf(stderr, "polygeist PVA: %s failed: %s\n", #call,                \
            cudaGetErrorString(status_));                                    \
    abort();                                                                 \
  }                                                                          \
} while (0)

static int g_initialized;
static cudaStream_t g_stream;
static NVCVAllocatorHandle g_allocator;

static void initialize(void) {
  if (g_initialized)
    return;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  NVCV_CHECK(nvcvAllocatorConstructPva(&g_allocator));
  g_initialized = 1;
}

static int dtype_bytes(NVCVDataType dtype) {
  if (dtype == NVCV_DATA_TYPE_U8 || dtype == NVCV_DATA_TYPE_S8)
    return 1;
  if (dtype == NVCV_DATA_TYPE_U16 || dtype == NVCV_DATA_TYPE_S16)
    return 2;
  if (dtype == NVCV_DATA_TYPE_U32 || dtype == NVCV_DATA_TYPE_S32)
    return 4;
  fprintf(stderr, "polygeist PVA: unsupported NVCV dtype\n");
  abort();
}

static void make_hwc_tensor(int32_t height, int32_t width,
                            NVCVDataType dtype,
                            NVCVTensorRequirements *requirements,
                            NVCVTensorHandle *tensor) {
  NVCVTensorLayout layout;
  int64_t shape[] = {height, width, 1};
  NVCV_CHECK(nvcvTensorLayoutMake("HWC", &layout));
  NVCV_CHECK(nvcvTensorCalcRequirementsPva(
      3, shape, dtype, layout, 0, 0, requirements));
  NVCV_CHECK(nvcvTensorConstruct(requirements, g_allocator, tensor));
}

static void *tensor_host_pointer(NVCVTensorHandle tensor,
                                 NVCVTensorData *data) {
  void *pointer = NULL;
  NVCV_CHECK(nvcvTensorExportData(tensor, data));
  if (data->bufferType != NVCV_TENSOR_BUFFER_STRIDED_CUDA) {
    fprintf(stderr, "polygeist PVA: tensor is not strided CUDA storage\n");
    abort();
  }
  cupvaError_t status = CupvaMemGetHostPointer(
      &pointer, (void *)data->buffer.strided.basePtr);
  if (status != CUPVA_ERROR_NONE || pointer == NULL) {
    fprintf(stderr, "polygeist PVA: CupvaMemGetHostPointer failed: %d\n",
            (int)status);
    abort();
  }
  return pointer;
}

static void copy_host_to_hwc(NVCVTensorHandle tensor, const void *source,
                             int32_t height, int32_t width,
                             int element_bytes) {
  NVCVTensorData data;
  char *destination = (char *)tensor_host_pointer(tensor, &data);
  for (int32_t row = 0; row < height; ++row)
    memcpy(destination + row * data.buffer.strided.strides[0],
           (const char *)source + (size_t)row * width * element_bytes,
           (size_t)width * element_bytes);
}

static void copy_hwc_to_host(NVCVTensorHandle tensor, void *destination,
                             int32_t height, int32_t width,
                             int element_bytes, int interior_only) {
  NVCVTensorData data;
  const char *source = (const char *)tensor_host_pointer(tensor, &data);
  int32_t first = interior_only ? 1 : 0;
  int32_t last_row = interior_only ? height - 1 : height;
  int32_t columns = interior_only ? width - 2 : width;
  for (int32_t row = first; row < last_row; ++row)
    memcpy((char *)destination +
               ((size_t)row * width + first) * element_bytes,
           source + row * data.buffer.strided.strides[0] +
               (size_t)first * element_bytes,
           (size_t)columns * element_bytes);
}

typedef enum {
  IMAGE_BOX,
  IMAGE_GAUSSIAN,
  IMAGE_MORPH_DILATE,
  IMAGE_BILATERAL
} ImageOperation;

static void run_image_operation(ImageOperation operation, NVCVDataType dtype,
                                int32_t height, int32_t width,
                                float parameter0, float parameter1,
                                const void *input, void *output) {
  initialize();
  NVCVTensorRequirements requirements;
  NVCVTensorHandle input_tensor, output_tensor;
  NVCVOperatorHandle handle = NULL;
  make_hwc_tensor(height, width, dtype, &requirements, &input_tensor);
  NVCV_CHECK(nvcvTensorConstruct(&requirements, g_allocator, &output_tensor));
  copy_host_to_hwc(input_tensor, input, height, width, dtype_bytes(dtype));

  switch (operation) {
  case IMAGE_BOX:
    NVCV_CHECK(pvaBoxFilterCreate(&handle, &requirements, 3,
                                  NVCV_BORDER_REPLICATE, 0));
    NVCV_CHECK(pvaBoxFilterSubmit(handle, g_stream, input_tensor,
                                  output_tensor));
    break;
  case IMAGE_GAUSSIAN:
    NVCV_CHECK(pvaGaussianFilterCreate(&handle, &requirements, parameter0,
                                       parameter1, 3,
                                       NVCV_BORDER_REPLICATE, 0));
    NVCV_CHECK(pvaGaussianFilterSubmit(handle, g_stream, input_tensor,
                                       output_tensor));
    break;
  case IMAGE_MORPH_DILATE: {
    PvaMorphologyMaskParams mask = {3, 3, RECTANGLE_MASK};
    NVCV_CHECK(pvaMorphologyCreate(&handle, &requirements, PVA_DILATE, &mask,
                                   NVCV_BORDER_REPLICATE, 0));
    NVCV_CHECK(pvaMorphologySubmit(handle, g_stream, input_tensor,
                                   output_tensor));
    break;
  }
  case IMAGE_BILATERAL:
    NVCV_CHECK(pvaBilateralFilterCreate(&handle, &requirements, 3,
                                        NVCV_BORDER_REPLICATE, 0));
    NVCV_CHECK(pvaBilateralFilterSubmit(handle, g_stream, input_tensor,
                                        parameter0, parameter1,
                                        output_tensor));
    break;
  }
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  /* The source fixtures update only pixels whose complete 3x3 neighborhood
   * is in bounds.  Retain the caller's border and copy the identical PVA
   * interior coordinates back. */
  copy_hwc_to_host(output_tensor, output, height, width, dtype_bytes(dtype), 1);
  nvcvTensorDecRef(input_tensor, NULL);
  nvcvTensorDecRef(output_tensor, NULL);
  nvcvOperatorDestroy(handle);
}

#define DEFINE_FILTER_WRAPPERS(suffix, ctype, nvcv_dtype)                    \
  void polygeist_pva_boxfilter_3x3_##suffix(                                \
      int32_t h, int32_t w, const ctype *in, ctype *out) {                   \
    run_image_operation(IMAGE_BOX, nvcv_dtype, h, w, 0.0f, 0.0f, in, out);  \
  }                                                                          \
  void polygeist_pva_gaussian_3x3_##suffix(                                 \
      int32_t h, int32_t w, float sigma_x, float sigma_y,                    \
      const ctype *in, ctype *out) {                                         \
    run_image_operation(IMAGE_GAUSSIAN, nvcv_dtype, h, w, sigma_x, sigma_y,  \
                        in, out);                                            \
  }                                                                          \
  void polygeist_pva_morphology_dilate_3x3_##suffix(                        \
      int32_t h, int32_t w, const ctype *in, ctype *out) {                   \
    run_image_operation(IMAGE_MORPH_DILATE, nvcv_dtype, h, w, 0.0f, 0.0f,  \
                        in, out);                                            \
  }

DEFINE_FILTER_WRAPPERS(u8, uint8_t, NVCV_DATA_TYPE_U8)
DEFINE_FILTER_WRAPPERS(s8, int8_t, NVCV_DATA_TYPE_S8)
DEFINE_FILTER_WRAPPERS(u16, uint16_t, NVCV_DATA_TYPE_U16)
DEFINE_FILTER_WRAPPERS(s16, int16_t, NVCV_DATA_TYPE_S16)

void polygeist_pva_bilateral_3x3_u8(int32_t h, int32_t w,
                                     float sigma_range, float sigma_space,
                                     const uint8_t *in, uint8_t *out) {
  run_image_operation(IMAGE_BILATERAL, NVCV_DATA_TYPE_U8, h, w,
                      sigma_range, sigma_space, in, out);
}

static void run_histogram(NVCVDataType input_dtype, NVCVDataType output_dtype,
                          int32_t height, int32_t width, float range_end,
                          int32_t bins, const void *input, void *output) {
  initialize();
  NVCVTensorRequirements input_requirements, output_requirements;
  NVCVTensorHandle input_tensor, output_tensor;
  make_hwc_tensor(height, width, input_dtype, &input_requirements,
                  &input_tensor);
  make_hwc_tensor(1, bins, output_dtype, &output_requirements, &output_tensor);
  copy_host_to_hwc(input_tensor, input, height, width,
                   dtype_bytes(input_dtype));
  PvaImageHistogramParams parameters = {0.0f, range_end, bins};
  NVCVOperatorHandle handle = NULL;
  NVCV_CHECK(pvaImageHistogramCreate(&handle, &input_requirements,
                                     &output_requirements, &parameters));
  NVCV_CHECK(pvaImageHistogramSubmit(handle, g_stream, input_tensor,
                                     output_tensor));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  copy_hwc_to_host(output_tensor, output, 1, bins,
                   dtype_bytes(output_dtype), 0);
  nvcvTensorDecRef(input_tensor, NULL);
  nvcvTensorDecRef(output_tensor, NULL);
  nvcvOperatorDestroy(handle);
}

#define DEFINE_HISTOGRAM_WRAPPERS(input_suffix, input_type, input_dtype,     \
                                  range_end)                                 \
  void polygeist_pva_histogram_256_##input_suffix##_u32(                    \
      int32_t h, int32_t w, const input_type *in, uint32_t *out) {           \
    run_histogram(input_dtype, NVCV_DATA_TYPE_U32, h, w, range_end, 256,     \
                  in, out);                                                  \
  }                                                                          \
  void polygeist_pva_histogram_256_##input_suffix##_s32(                    \
      int32_t h, int32_t w, const input_type *in, int32_t *out) {            \
    run_histogram(input_dtype, NVCV_DATA_TYPE_S32, h, w, range_end, 256,     \
                  in, out);                                                  \
  }

DEFINE_HISTOGRAM_WRAPPERS(u8, uint8_t, NVCV_DATA_TYPE_U8, 256.0f)
DEFINE_HISTOGRAM_WRAPPERS(u16, uint16_t, NVCV_DATA_TYPE_U16, 65536.0f)

void polygeist_pva_histeq_u8(int32_t height, int32_t width,
                              const uint8_t *input, uint8_t *output) {
  initialize();
  NVCVTensorRequirements requirements;
  NVCVTensorHandle input_tensor, output_tensor;
  make_hwc_tensor(height, width, NVCV_DATA_TYPE_U8, &requirements,
                  &input_tensor);
  NVCV_CHECK(nvcvTensorConstruct(&requirements, g_allocator, &output_tensor));
  copy_host_to_hwc(input_tensor, input, height, width, 1);
  NVCVOperatorHandle handle = NULL;
  NVCV_CHECK(pvaHistogramEqualizationCreate(&handle, &requirements));
  NVCV_CHECK(pvaHistogramEqualizationSubmit(handle, g_stream, input_tensor,
                                             output_tensor));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  copy_hwc_to_host(output_tensor, output, height, width, 1, 0);
  nvcvTensorDecRef(input_tensor, NULL);
  nvcvTensorDecRef(output_tensor, NULL);
  nvcvOperatorDestroy(handle);
}

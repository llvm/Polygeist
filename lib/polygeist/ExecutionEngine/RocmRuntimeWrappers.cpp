//===- PolygeistRocmRuntimeWrappers.cpp - MLIR ROCM API wrapper library ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the ROCM library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>

#include "hip/hip_runtime.h"

#include "PGORuntime.h"

#ifdef _WIN32
#define MLIR_HIP_WRAPPERS_EXPORT __declspec(dllexport) __attribute__((weak))
#else
#define MLIR_HIP_WRAPPERS_EXPORT __attribute__((weak))
#endif // _WIN32

#define HIP_REPORT_IF_ERROR(expr)                                              \
  [](hipError_t result) {                                                      \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = hipGetErrorName(result);                                \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

#define ERR_HIP_REPORT_IF_ERROR(expr)                                          \
  [](hipError_t result) -> hipError_t {                                        \
    if (!result)                                                               \
      return result;                                                           \
    const char *name = hipGetErrorName(result);                                \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
    return result;                                                             \
  }(expr)

extern "C" MLIR_HIP_WRAPPERS_EXPORT int32_t
mgpurtMemAllocErr(void **mem, uint64_t sizeBytes) {
  return ERR_HIP_REPORT_IF_ERROR(hipMalloc(mem, sizeBytes));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void *
mgpurtMemAlloc(uint64_t sizeBytes, hipStream_t /*stream*/) {
  void *ptr;
  HIP_REPORT_IF_ERROR(hipMalloc(&ptr, sizeBytes));
  return reinterpret_cast<void *>(ptr);
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void mgpuMemFree(void *ptr,
                                                     hipStream_t /*stream*/) {
  HIP_REPORT_IF_ERROR(hipFree(ptr));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT int32_t
mgpurtMemcpyErr(void *dst, void *src, intptr_t sizeBytes) {
  return ERR_HIP_REPORT_IF_ERROR(
      hipMemcpy(dst, src, sizeBytes, hipMemcpyDefault));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT int32_t mgpurtMemcpyAsyncErr(
    void *dst, void *src, intptr_t sizeBytes, hipStream_t stream) {
  return ERR_HIP_REPORT_IF_ERROR(
      hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT int32_t mgpurtDeviceSynchronizeErr(void) {
  return ERR_HIP_REPORT_IF_ERROR(hipDeviceSynchronize());
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT int32_t mgpurtLaunchKernelErr(
    void *function, intptr_t gridX, intptr_t gridY, intptr_t gridZ,
    intptr_t blockX, intptr_t blockY, intptr_t blockZ, int32_t smem,
    hipStream_t stream, void **params) {
  return ERR_HIP_REPORT_IF_ERROR(
      hipLaunchKernel(function, dim3(gridX, gridY, gridZ),
                      dim3(blockX, blockY, blockZ), params, smem, stream));
}

extern "C" void __hipRegisterFunction(void **fatCubinHandle, void *hostFun,
                                      void *deviceFun, void *deviceName,
                                      int32_t thread_limit, void *tid,
                                      void *bid, void *bDim, void *gDim,
                                      void *wSize);
extern "C" void **__hipRegisterFatBinary(void *fatCubin);
extern "C" void __hipRegisterFatBinaryEnd(void **fatCubinHandle);
extern "C" void __hipUnregisterFatBinary(void **fatCubinHandle);

extern "C" MLIR_HIP_WRAPPERS_EXPORT void
__mgpurtRegisterFunction(void **fatCubinHandle, void *hostFun, void *deviceFun,
                         void *deviceName, int32_t thread_limit, void *tid,
                         void *bid, void *bDim, void *gDim, void *wSize) {
  __hipRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                        thread_limit, tid, bid, bDim, gDim, wSize);
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void **
__mgpurtRegisterFatBinary(void *fatCubin) {
  return __hipRegisterFatBinary(fatCubin);
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void
__mgpurtRegisterFatBinaryEnd(void **fatCubinHandle) {
  return __hipRegisterFatBinaryEnd(fatCubinHandle);
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void
__mgpurtUnregisterFatBinary(void **fatCubinHandle) {
  return __hipUnregisterFatBinary(fatCubinHandle);
}

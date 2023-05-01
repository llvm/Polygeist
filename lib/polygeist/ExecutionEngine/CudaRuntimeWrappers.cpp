//===- PolygeistCudaRuntimeWrappers.cpp - MLIR CUDA API wrapper library ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the CUDA library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"

#ifdef _WIN32
#define MLIR_CUDA_WRAPPERS_EXPORT __declspec(dllexport) __attribute__((weak))
#else
#define MLIR_CUDA_WRAPPERS_EXPORT __attribute__((weak))
#endif // _WIN32

#define CUDART_REPORT_IF_ERROR(expr)                                           \
  [](auto result) {                                                            \
    if (!result)                                                               \
      return result;                                                           \
    const char *name = cudaGetErrorString(result);                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
    return result;                                                             \
  }(expr)

#define CUDA_REPORT_IF_ERROR(expr)                                             \
  [](CUresult result) {                                                        \
    if (!result)                                                               \
      return result;                                                           \
    const char *name = nullptr;                                                \
    cuGetErrorName(result, &name);                                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
    return result;                                                             \
  }(expr)

thread_local static int32_t defaultDevice = 0;

// Make the primary context of the current default device current for the
// duration
//  of the instance and restore the previous context on destruction.
class ScopedContext {
public:
  ScopedContext() {
    // Static reference to CUDA primary context for device ordinal
    // defaultDevice.
    static CUcontext context = [] {
      CUDA_REPORT_IF_ERROR(cuInit(/*flags=*/0));
      CUdevice device;
      CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
      CUcontext ctx;
      // Note: this does not affect the current context.
      CUDA_REPORT_IF_ERROR(cuDevicePrimaryCtxRetain(&ctx, device));
      return ctx;
    }();

    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(context));
  }
};

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpurtLaunchKernel(void *function, intptr_t gridX, intptr_t gridY,
                   intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                   intptr_t blockZ, int32_t smem, CUstream stream,
                   void **params) {
  CUDART_REPORT_IF_ERROR(cudaLaunchKernel(function, dim3(gridX, gridY, gridZ),
                                          dim3(blockX, blockY, blockZ), params,
                                          smem, stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *mgpurtMemAlloc(uint64_t sizeBytes,
                                                          CUstream /*stream*/) {
  void *ptr;
  CUDART_REPORT_IF_ERROR(cudaMalloc(&ptr, sizeBytes));
  return reinterpret_cast<void *>(ptr);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT int32_t mgpuLaunchKernelErr(
    CUfunction function, intptr_t gridX, intptr_t gridY, intptr_t gridZ,
    intptr_t blockX, intptr_t blockY, intptr_t blockZ, int32_t smem,
    CUstream stream, void **params, void **extra) {
  return CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ,
                                             blockX, blockY, blockZ, smem,
                                             stream, params, extra));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule mgpuModuleLoad(void *data) {
  ScopedContext scopedContext;
  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
  return module;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuModuleUnload(CUmodule module) {
  CUDA_REPORT_IF_ERROR(cuModuleUnload(module));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUfunction
mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}

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

#include "cuda.h"
#include "cuda_runtime.h"

#include "PGORuntime.h"

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

  ~ScopedContext() { CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)); }
};

//========= CUDA RUNTIME API =========//

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpurtLaunchKernel(void *function, intptr_t gridX, intptr_t gridY,
                   intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                   intptr_t blockZ, int32_t smem, cudaStream_t stream,
                   void **params) {
  CUDART_REPORT_IF_ERROR(cudaLaunchKernel(function, dim3(gridX, gridY, gridZ),
                                          dim3(blockX, blockY, blockZ), params,
                                          smem, stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT int32_t mgpurtLaunchKernelErr(
    void *function, intptr_t gridX, intptr_t gridY, intptr_t gridZ,
    intptr_t blockX, intptr_t blockY, intptr_t blockZ, int32_t smem,
    cudaStream_t stream, void **params) {
  return CUDART_REPORT_IF_ERROR(
      cudaLaunchKernel(function, dim3(gridX, gridY, gridZ),
                       dim3(blockX, blockY, blockZ), params, smem, stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpurtMemAlloc(uint64_t sizeBytes, cudaStream_t /*stream*/) {
  void *ptr;
  CUDART_REPORT_IF_ERROR(cudaMalloc(&ptr, sizeBytes));
  return reinterpret_cast<void *>(ptr);
}

extern "C" void mgpurtMemcpyErr(void *dst, void *src, size_t sizeBytes) {
  CUDART_REPORT_IF_ERROR(cudaMemcpy(dst, src, sizeBytes, cudaMemcpyDefault));
}

extern "C" void mgpurtMemcpyAsyncErr(void *dst, void *src, size_t sizeBytes,
                                     cudaStream_t stream) {
  CUDART_REPORT_IF_ERROR(
      cudaMemcpyAsync(dst, src, sizeBytes, cudaMemcpyDefault, stream));
}

//========= CUDA DRIVER API =========//

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, CUstream stream, void **params,
                 void **extra) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                      blockY, blockZ, smem, stream, params,
                                      extra));
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT int32_t mgpuLaunchKernelErr(
    CUfunction function, intptr_t gridX, intptr_t gridY, intptr_t gridZ,
    intptr_t blockX, intptr_t blockY, intptr_t blockZ, int32_t smem,
    CUstream stream, void **params, void **extra) {
  ScopedContext scopedContext;
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

extern "C" MLIR_CUDA_WRAPPERS_EXPORT int32_t mgpurtDeviceSynchronizeErr(void) {
  return CUDART_REPORT_IF_ERROR(cudaDeviceSynchronize());
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle, void *hostFun,
                                       void *deviceFun, void *deviceName,
                                       int32_t thread_limit, void *tid,
                                       void *bid, void *bDim, void *gDim,
                                       void *wSize);
extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global);
extern "C" void **__cudaRegisterFatBinary(void *fatCubin);
extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle);

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
__mgpurtRegisterFunction(void **fatCubinHandle, void *hostFun, void *deviceFun,
                         void *deviceName, int32_t thread_limit, void *tid,
                         void *bid, void *bDim, void *gDim, void *wSize) {
  __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                         thread_limit, tid, bid, bDim, gDim, wSize);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
__mgpurtRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                    const char *deviceName, int ext, size_t size, int constant,
                    int global) {
  __cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext,
                    size, constant, global);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void **
__mgpurtRegisterFatBinary(void *fatCubin) {
  return __cudaRegisterFatBinary(fatCubin);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
__mgpurtRegisterFatBinaryEnd(void **fatCubinHandle) {
  __cudaRegisterFatBinaryEnd(fatCubinHandle);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
__mgpurtUnregisterFatBinary(void **fatCubinHandle) {
  __cudaUnregisterFatBinary(fatCubinHandle);
}

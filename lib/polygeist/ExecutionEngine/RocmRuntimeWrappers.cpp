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
extern "C" void __hipRegisterVar(void **fatCubinHandle, char *hostVar,
                                 char *deviceAddress, const char *deviceName,
                                 int ext, size_t size, int constant,
                                 int global);
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
extern "C" MLIR_HIP_WRAPPERS_EXPORT void
__mgpurtRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                    const char *deviceName, int ext, size_t size, int constant,
                    int global) {
  __hipRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext,
                   size, constant, global);
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

#if POLYGEIST_ENABLE_CUDA

#pragma push_macro("__forceinline__")
#define __VECTOR_TYPES_H__
#include <cuda_runtime_api.h>
#undef __VECTOR_TYPES_H__
#pragma pop_macro("__forceinline__")

extern "C" MLIR_HIP_WRAPPERS_EXPORT int32_t
mgpurtCudaGetDeviceProperties(struct cudaDeviceProp *cudaProp, int device) {
  struct hipDeviceProp_t hipProp;
  int err = ERR_HIP_REPORT_IF_ERROR(hipGetDeviceProperties(&hipProp, device));

  // Reassign all corresponding fields to the hip props, the commented ones dont
  // exist in hip one-for-one
#define __polygeist_assign_field(f)                                            \
  memcpy(&(cudaProp->f), &(hipProp.f), sizeof(cudaProp->f))
  __polygeist_assign_field(name);
  // __polygeist_assign_field(uuid);
  __polygeist_assign_field(totalGlobalMem);
  __polygeist_assign_field(sharedMemPerBlock);
  __polygeist_assign_field(regsPerBlock);
  __polygeist_assign_field(warpSize);
  __polygeist_assign_field(memPitch);
  __polygeist_assign_field(maxThreadsPerBlock);
  __polygeist_assign_field(maxThreadsDim);
  __polygeist_assign_field(maxGridSize);
  __polygeist_assign_field(clockRate);
  __polygeist_assign_field(totalConstMem);
  __polygeist_assign_field(major);
  __polygeist_assign_field(minor);
  __polygeist_assign_field(textureAlignment);
  __polygeist_assign_field(texturePitchAlignment);
  // __polygeist_assign_field(deviceOverlap);
  __polygeist_assign_field(multiProcessorCount);
  __polygeist_assign_field(kernelExecTimeoutEnabled);
  __polygeist_assign_field(integrated);
  __polygeist_assign_field(canMapHostMemory);
  __polygeist_assign_field(computeMode);
  __polygeist_assign_field(maxTexture1D);
  // __polygeist_assign_field(maxTexture1DMipmap);
  __polygeist_assign_field(maxTexture1DLinear);
  __polygeist_assign_field(maxTexture2D);
  // __polygeist_assign_field(maxTexture2DMipmap);
  // __polygeist_assign_field(maxTexture2DLinear);
  // __polygeist_assign_field(maxTexture2DGather);
  __polygeist_assign_field(maxTexture3D);
  // __polygeist_assign_field(maxTexture3DAlt);
  // __polygeist_assign_field(maxTextureCubemap);
  // __polygeist_assign_field(maxTexture1DLayered);
  // __polygeist_assign_field(maxTexture2DLayered);
  // __polygeist_assign_field(maxTextureCubemapLayered);
  // __polygeist_assign_field(maxSurface1D);
  // __polygeist_assign_field(maxSurface2D);
  // __polygeist_assign_field(maxSurface3D);
  // __polygeist_assign_field(maxSurface1DLayered);
  // __polygeist_assign_field(maxSurface2DLayered);
  // __polygeist_assign_field(maxSurfaceCubemap);
  // __polygeist_assign_field(maxSurfaceCubemapLayered);
  // __polygeist_assign_field(surfaceAlignment);
  __polygeist_assign_field(concurrentKernels);
  __polygeist_assign_field(ECCEnabled);
  __polygeist_assign_field(pciBusID);
  __polygeist_assign_field(pciDeviceID);
  __polygeist_assign_field(pciDomainID);
  __polygeist_assign_field(tccDriver);
  // __polygeist_assign_field(asyncEngineCount);
  // __polygeist_assign_field(unifiedAddressing);
  __polygeist_assign_field(memoryClockRate);
  __polygeist_assign_field(memoryBusWidth);
  __polygeist_assign_field(l2CacheSize);
  // __polygeist_assign_field(persistingL2CacheMaxSize);
  __polygeist_assign_field(maxThreadsPerMultiProcessor);
  // __polygeist_assign_field(streamPrioritiesSupported);
  // __polygeist_assign_field(globalL1CacheSupported);
  // __polygeist_assign_field(localL1CacheSupported);
  // __polygeist_assign_field(sharedMemPerMultiprocessor);
  // __polygeist_assign_field(regsPerMultiprocessor);
  __polygeist_assign_field(managedMemory);
  __polygeist_assign_field(isMultiGpuBoard);
  // __polygeist_assign_field(multiGpuBoardGroupID);
  // __polygeist_assign_field(singleToDoublePrecisionPerfRatio);
  __polygeist_assign_field(pageableMemoryAccess);
  __polygeist_assign_field(concurrentManagedAccess);
  // __polygeist_assign_field(computePreemptionSupported);
  // __polygeist_assign_field(canUseHostPointerForRegisteredMem);
  __polygeist_assign_field(cooperativeLaunch);
  __polygeist_assign_field(cooperativeMultiDeviceLaunch);
  __polygeist_assign_field(pageableMemoryAccessUsesHostPageTables);
  __polygeist_assign_field(directManagedMemAccessFromHost);
  // __polygeist_assign_field(accessPolicyMaxWindowSize);
#undef __polygeist_assign_field

  return err;
}

#endif

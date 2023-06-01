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

// Cuda definitions - cannot include cuda_runtime.h because it conflicts with
// the hip includes
struct CUuuid_st {
  char bytes[16];
};
typedef struct CUuuid_st CUuuid;
typedef struct CUuuid_st cudaUUID_t;
struct cudaDeviceProp {
  char name[256];
  cudaUUID_t uuid;
  char luid[8];
  unsigned int luidDeviceNodeMask;
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture1DMipmap;
  int maxTexture1DLinear;
  int maxTexture2D[2];
  int maxTexture2DMipmap[2];
  int maxTexture2DLinear[3];
  int maxTexture2DGather[2];
  int maxTexture3D[3];
  int maxTexture3DAlt[3];
  int maxTextureCubemap;
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  int maxTextureCubemapLayered[2];
  int maxSurface1D;
  int maxSurface2D[2];
  int maxSurface3D[3];
  int maxSurface1DLayered[2];
  int maxSurface2DLayered[3];
  int maxSurfaceCubemap;
  int maxSurfaceCubemapLayered[2];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int persistingL2CacheMaxSize;
  int maxThreadsPerMultiProcessor;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemory;
  int isMultiGpuBoard;
  int multiGpuBoardGroupID;
  int hostNativeAtomicSupported;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  size_t sharedMemPerBlockOptin;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
  int maxBlocksPerMultiProcessor;
  int accessPolicyMaxWindowSize;
  size_t reservedSharedMemPerBlock;
};

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

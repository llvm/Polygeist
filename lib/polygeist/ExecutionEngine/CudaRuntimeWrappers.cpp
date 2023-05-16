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
#include <map>
#include <mutex>
#include <time.h>

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

  ~ScopedContext() { CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)); }
};

// TODO Remove the syncs and move PGO related stuff in a separate wrapper file.
// The syncs should instead be emitted by the code that emits the calls to the
// PGO functions which should know whether the code in the alternatives op is
// GPU code - we can add an attrib to the alternatives op for that
class PGOState {
public:
  enum Type { Start, End };
  struct State {
    struct timespec start_clock;
  };

  inline static int alternative;
  inline static std::string dirname;
  inline thread_local static std::mutex mutex;
  inline thread_local static std::map<std::string, State *> states;

  std::string kernelId;
  int totalAlternatives;

  PGOState(const char *kernelId_c, int totalAlternatives)
      : totalAlternatives(totalAlternatives) {
    kernelId = kernelId_c;
    for (char &c : kernelId)
      if (c == '/')
        c = '+';
  }
  void end() {
    struct timespec end_clock;
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end_clock);

    std::unique_lock<std::mutex> lock(mutex);
    if (states.count(kernelId) == 0) {
      std::cerr << "No kernel with id " << kernelId << "running" << std::endl;
      exit(1);
    }
    State *state = states[kernelId];
    struct timespec tmp_clock {
      end_clock.tv_sec - state->start_clock.tv_sec,
          end_clock.tv_nsec - state->start_clock.tv_nsec
    };
    double elapsed =
        (tmp_clock.tv_sec + ((double)tmp_clock.tv_nsec) * .000000001);

    // Only write to file if we are profiling a valid alternative
    if (0 <= alternative && alternative < totalAlternatives) {
      // TODO error handling
      std::ofstream ofile;
      ofile.open(std::string(dirname) + "/" + kernelId,
                 std::ios::out | std::ios::app);
      ofile << alternative << " " << elapsed << std::endl;
      ofile.close();
    }

    delete state;
    states.erase(states.find(kernelId));
  }

  void start() {
    std::unique_lock<std::mutex> lock(mutex);
    State *state = new State();
    if (states.count(kernelId) == 1) {
      std::cerr << "Two kernels with id " << kernelId
                << "running at the same time" << std::endl;
      exit(1);
    }
    states[kernelId] = state;
    // Start timing
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &state->start_clock);
  }

  int getAlternative() {
    static int init = [&] {
      if (char *i = getenv(POLYGEIST_PGO_ALTERNATIVE_ENV_VAR)) {
        this->alternative = atoi(i);
      } else {
        std::cerr << POLYGEIST_PGO_ALTERNATIVE_ENV_VAR << " not defined"
                  << std::endl;
        exit(1);
      }
      if (char *d = getenv(POLYGEIST_PGO_DATA_DIR_ENV_VAR)) {
        this->dirname = d;
      } else {
        this->dirname = POLYGEIST_PGO_DEFAULT_DATA_DIR;
      }
      std::filesystem::create_directories(dirname);
      return 0;
    }();
    if (0 <= alternative && alternative < totalAlternatives)
      return alternative;
    else
      return 0;
  }

  ~PGOState() {}
};

extern "C" MLIR_CUDA_WRAPPERS_EXPORT int32_t
mgpurtPGOGetAlternative(const char *kernelID, int totalAlternatives) {
  PGOState pgoState(kernelID, totalAlternatives);
  return pgoState.getAlternative();
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpurtPGOStart(const char *kernelID, int totalAlternatives) {
  PGOState pgoState(kernelID, totalAlternatives);
  pgoState.start();
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpurtPGOEnd(const char *kernelID,
                                                       int totalAlternatives) {
  PGOState pgoState(kernelID, totalAlternatives);
  pgoState.end();
}

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

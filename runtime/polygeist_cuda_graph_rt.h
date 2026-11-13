// Minimal ABI shared by compiler-generated CUDA wrappers and the Polygeist
// library runtime. Keep this header independent of CUDA and tensor datatypes;
// generated .cu files only need a stream and graph scope, not every library
// shim declaration in polygeist_cublas_rt.h.
#ifndef POLYGEIST_CUDA_GRAPH_RT_H
#define POLYGEIST_CUDA_GRAPH_RT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t polygeist_cuda_graph_begin(int64_t graph_id);
void polygeist_cuda_graph_end(int64_t graph_id);

// Returns the runtime-owned CUDA stream as an opaque pointer. CUDA wrappers
// cast it to cudaStream_t. Returns NULL in the CPU runtime. Callers must not
// destroy or synchronize the stream.
void *polygeist_cuda_graph_stream(void);

#ifdef __cplusplus
}
#endif

#endif // POLYGEIST_CUDA_GRAPH_RT_H

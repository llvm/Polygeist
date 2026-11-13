// Exercise compiler-planned copies across partially registered host ranges.
// This occurs when an allocator reuses pages registered by a prior call.
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern void *mgpuMemAlloc(uint64_t bytes, void *stream, uint8_t host_shared);
extern void mgpuMemFree(void *pointer, void *stream);
extern void mgpuMemcpy(void *destination, void *source, size_t bytes,
                       void *stream);

static void check(cudaError_t error) {
  if (error != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(error));
    exit(1);
  }
}

int main(void) {
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  size_t bytes = page * 3;
  unsigned char *host = NULL;
  if (posix_memalign((void **)&host, page, bytes))
    return 1;
  void *device = mgpuMemAlloc(bytes, NULL, 0);
  for (int registered = 0; registered < 2; ++registered) {
    if (registered)
      check(cudaHostRegister(host + page, page, cudaHostRegisterMapped));
    for (int iteration = 0; iteration < 10; ++iteration) {
      for (size_t i = 0; i < bytes; ++i)
        host[i] = (unsigned char)(i + iteration);
      mgpuMemcpy(device, host, bytes, NULL);
      check(cudaDeviceSynchronize());
      for (size_t i = 0; i < bytes; ++i)
        host[i] = 0;
      mgpuMemcpy(host, device, bytes, NULL);
      check(cudaDeviceSynchronize());
      for (size_t i = 0; i < bytes; ++i) {
        if (host[i] != (unsigned char)(i + iteration)) {
          fprintf(stderr, "copy mismatch: registered=%d iteration=%d byte=%zu\n",
                  registered, iteration, i);
          return 1;
        }
      }
    }
    if (registered)
      check(cudaHostUnregister(host + page));
  }
  mgpuMemFree(device, NULL);
  free(host);
  puts("PASS: pageable and partially registered H2D/D2H, 10 iterations each");
  return 0;
}

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

namespace {
cudaEvent_t start_event;
cudaEvent_t stop_event;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(3);
  }
}
} // namespace

extern "C" void bench_cuda_event_init(void) {
  check(cudaEventCreate(&start_event), "cudaEventCreate(start)");
  check(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");
}

extern "C" void bench_cuda_event_start(void) {
  check(cudaEventRecord(start_event), "cudaEventRecord(start)");
}

extern "C" double bench_cuda_event_stop_ms(void) {
  check(cudaEventRecord(stop_event), "cudaEventRecord(stop)");
  check(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");
  float elapsed_ms = 0.0f;
  check(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event),
        "cudaEventElapsedTime");
  return elapsed_ms;
}

extern "C" void bench_cuda_event_destroy(void) {
  check(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");
  check(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
}

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" int polygeist_cub_segmented_prefix_sum_f32_cuda(
    int32_t, int32_t, const float *, const int32_t *, float *, cudaStream_t);
extern "C" int polygeist_cub_segmented_prefix_logical_and_i32_cuda(
    int32_t, int32_t, const int32_t *, const int32_t *, int32_t *,
    cudaStream_t);

int main() {
  constexpr int rows = 65536, cols = 128;
  std::vector<float> f((size_t)rows * cols), f_out(rows), f_ref(rows);
  std::vector<int32_t> x((size_t)rows * cols), lengths(rows), out(rows), ref(rows);
  for (int r = 0; r < rows; ++r) {
    lengths[r] = (r * 37 + 11) % (cols + 1);
    float sum = 0.0f; int all = 1;
    for (int c = 0; c < cols; ++c) {
      f[(size_t)r * cols + c] = (float)((r + c) % 5 - 2);
      x[(size_t)r * cols + c] = ((r * 13 + c * 7) % 19) != 0;
      if (c < lengths[r]) { sum += f[(size_t)r * cols + c]; all &= x[(size_t)r * cols + c] != 0; }
    }
    f_ref[r] = sum; ref[r] = all;
  }
  cudaStream_t stream; cudaStreamCreate(&stream);
  for (int warm = 0; warm < 3; ++warm) {
    if (polygeist_cub_segmented_prefix_sum_f32_cuda(
            rows, cols, f.data(), lengths.data(), f_out.data(), stream)) return 2;
    if (polygeist_cub_segmented_prefix_logical_and_i32_cuda(
            rows, cols, x.data(), lengths.data(), out.data(), stream)) return 3;
  }
  cudaEvent_t begin, end; cudaEventCreate(&begin); cudaEventCreate(&end);
  auto time = [&](auto call) { float total=0; for(int t=0;t<5;++t){
    cudaEventRecord(begin,stream); if(call()) return -1.0f;
    cudaEventRecord(end,stream); cudaEventSynchronize(end); float ms=0;
    cudaEventElapsedTime(&ms,begin,end); total+=ms;} return total*200.0f; };
  float sum_us = time([&]{return polygeist_cub_segmented_prefix_sum_f32_cuda(
      rows,cols,f.data(),lengths.data(),f_out.data(),stream);});
  float all_us = time([&]{return polygeist_cub_segmented_prefix_logical_and_i32_cuda(
      rows,cols,x.data(),lengths.data(),out.data(),stream);});
  int sum_errors=0, all_errors=0;
  for(int r=0;r<rows;++r){sum_errors += f_out[r] != f_ref[r]; all_errors += out[r] != ref[r];}
  std::printf("rows=%d cols=%d sum_us=%.6f all_us=%.6f sum_errors=%d all_errors=%d\n",
              rows,cols,sum_us,all_us,sum_errors,all_errors);
  cudaEventDestroy(end);cudaEventDestroy(begin);cudaStreamDestroy(stream);
  return sum_errors || all_errors;
}

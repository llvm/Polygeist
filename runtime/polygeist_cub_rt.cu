// CUB-backed operations that require CUDA C++ templates.  Keep this separate
// from polygeist_cublas_rt_cuda.c, which deliberately remains compilable by
// an ordinary aarch64 C cross compiler.
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

struct ArithmeticRightShiftI32 {
  int32_t amount;
  __host__ __device__ int32_t operator()(int32_t value) const {
    return value >> amount;
  }
};

static bool cub_device_pointer(const void *pointer, void **device_pointer);

struct ClampRowIndexI32 {
  int32_t rows;
  __host__ __device__ int32_t operator()(int32_t value) const {
    return value < -1 ? -1 : (value > rows ? rows : value);
  }
};

// For arbitrary input order, count(prefix_max(input) < r) is exactly the
// stopping position of a scan that advances while input[p] < r. Clamping to
// [-1, rows] preserves every queried comparison, including negative values.
extern "C" int polygeist_cub_prefix_max_lower_bounds_i32_cuda(
    int32_t rows, int32_t count, const int32_t *input, int32_t *output,
    cudaStream_t stream) {
  if (rows < 0 || rows > INT_MAX-3 || count < 0 || !output ||
      (count && !input)) return cudaErrorInvalidValue;
  if (!count)
    return cudaMemsetAsync(output, 0, (size_t)(rows+1)*sizeof(int32_t), stream);
  struct Cache {
    int device;
    size_t input_capacity, bin_capacity, scratch_capacity;
    int32_t *prefix, *bins;
    void *scratch;
    Cache *next;
  };
  static thread_local Cache *cache_list = nullptr;
  int device;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  Cache *cache = cache_list;
  for (; cache && cache->device != device; cache = cache->next) {}
  if (!cache) {
    cache = new Cache{};
    cache->device = device; cache->next = cache_list; cache_list = cache;
  }
  auto reserve = [&](void **pointer, size_t &capacity, size_t required) {
    if (capacity >= required) return cudaSuccess;
    cudaError_t result = cudaStreamSynchronize(stream);
    if (result == cudaSuccess) result = cudaFree(*pointer);
    *pointer = nullptr; capacity = 0;
    if (result == cudaSuccess) result = cudaMalloc(pointer, required);
    if (result == cudaSuccess) capacity = required;
    return result;
  };
  status = reserve((void **)&cache->prefix, cache->input_capacity,
                   (size_t)count*sizeof(int32_t));
  if (status == cudaSuccess)
    status = reserve((void **)&cache->bins, cache->bin_capacity,
                     (size_t)(rows+2)*sizeof(int32_t));
  if (status != cudaSuccess) return status;
  using Input = cub::TransformInputIterator<int32_t, ClampRowIndexI32,
                                            const int32_t *>;
  Input values(input, ClampRowIndexI32{rows});
  size_t prefix_bytes = 0, histogram_bytes = 0, sum_bytes = 0;
  status = cub::DeviceScan::InclusiveScan(nullptr, prefix_bytes, values,
      cache->prefix, cub::Max(), count, stream);
  if (status == cudaSuccess)
    status = cub::DeviceHistogram::HistogramEven(nullptr, histogram_bytes,
        cache->prefix, cache->bins, rows+3, -1, rows+1, count, stream);
  if (status == cudaSuccess)
    status = cub::DeviceScan::InclusiveSum(nullptr, sum_bytes, cache->bins,
        output, rows+1, stream);
  size_t required = prefix_bytes > histogram_bytes ? prefix_bytes : histogram_bytes;
  if (sum_bytes > required) required = sum_bytes;
  if (status == cudaSuccess)
    status = reserve(&cache->scratch, cache->scratch_capacity, required);
  if (status == cudaSuccess)
    status = cub::DeviceScan::InclusiveScan(cache->scratch, prefix_bytes,
        values, cache->prefix, cub::Max(), count, stream);
  if (status == cudaSuccess)
    status = cub::DeviceHistogram::HistogramEven(cache->scratch, histogram_bytes,
        cache->prefix, cache->bins, rows+3, -1, rows+1, count, stream);
  if (status == cudaSuccess)
    status = cub::DeviceScan::InclusiveSum(cache->scratch, sum_bytes,
        cache->bins, output, rows+1, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
  return status;
}

extern "C" int polygeist_cub_histogram_even_i32_shift_zero_cuda(
    int32_t count, int32_t num_bins, const int32_t *host_samples,
    int32_t *host_histogram, int32_t right_shift, cudaStream_t stream) {
  if (count < 0 || num_bins <= 0 || !host_samples || !host_histogram ||
      right_shift < 0 || right_shift >= 31)
    return -1;
  int32_t *samples = nullptr;
  int32_t *histogram = nullptr;
  void *temporary = nullptr;
  void *resident = nullptr;
  size_t temporary_bytes = 0;
  bool owns_samples = !cub_device_pointer(host_samples, &resident);
  bool owns_histogram = false;
  cudaError_t status = cudaSuccess;
  if (owns_samples) {
    status = cudaMalloc(&samples, (size_t)count * sizeof(int32_t));
    if (status != cudaSuccess) return static_cast<int>(status);
    status = cudaMemcpyAsync(samples, host_samples,
                             (size_t)count * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) goto cleanup;
  } else {
    samples = static_cast<int32_t *>(resident);
  }
  owns_histogram = !cub_device_pointer(host_histogram, &resident);
  if (owns_histogram) {
    status = cudaMalloc(&histogram, (size_t)num_bins * sizeof(int32_t));
    if (status != cudaSuccess) goto cleanup;
  } else {
    histogram = static_cast<int32_t *>(resident);
  }
  {
    cub::TransformInputIterator<int32_t, ArithmeticRightShiftI32,
                                const int32_t *> bins(
        samples, ArithmeticRightShiftI32{right_shift});
    status = cub::DeviceHistogram::HistogramEven(
        temporary, temporary_bytes, bins, histogram,
        num_bins + 1, 0, num_bins, count, stream);
  }
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&temporary, temporary_bytes);
  if (status != cudaSuccess) goto cleanup;
  {
    cub::TransformInputIterator<int32_t, ArithmeticRightShiftI32,
                                const int32_t *> bins(
        samples, ArithmeticRightShiftI32{right_shift});
    status = cub::DeviceHistogram::HistogramEven(
        temporary, temporary_bytes, bins, histogram,
        num_bins + 1, 0, num_bins, count, stream);
  }
  if (status == cudaSuccess && owns_histogram)
    status = cudaMemcpyAsync(host_histogram, histogram,
                             (size_t)num_bins * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cleanup:
  cudaFree(temporary);
  if (owns_histogram) cudaFree(histogram);
  if (owns_samples) cudaFree(samples);
  return static_cast<int>(status);
}




struct LinearToColI32 {
  int32_t cols;
  __host__ __device__ int32_t operator()(int32_t linear) const {
    return linear % cols;
  }
};

struct SegmentOffsetI32 {
  int32_t width;
  __host__ __device__ int32_t operator()(int32_t row) const {
    return row*width;
  }
};



extern "C" int polygeist_cub_segmented_sort_descending_f32_i32_cuda(
    int32_t rows,int32_t cols,int32_t top,const float *host_input,
    float *host_values,int32_t *host_indices,cudaStream_t stream){
  int64_t n64=(int64_t)rows*cols;
  if(rows<=0||cols<=0||top<=0||top>cols||n64>INT_MAX||!host_input||!host_values||!host_indices)return-1;
  int32_t n=(int32_t)n64;float *input=nullptr,*sorted=nullptr;
  int32_t *input_indices=nullptr,*sorted_indices=nullptr;
  int32_t *host_input_indices=(int32_t*)malloc((size_t)n*sizeof(int32_t));
  void *temporary=nullptr;size_t temporary_bytes=0;
  using Counting = cub::CountingInputIterator<int32_t>;
  using Offsets = cub::TransformInputIterator<int32_t,SegmentOffsetI32,Counting>;
  Counting counting(0);Offsets offsets(counting,SegmentOffsetI32{cols});
  if(!host_input_indices)return cudaErrorMemoryAllocation;
  for(int32_t i=0;i<n;++i)host_input_indices[i]=i%cols;
  cudaError_t status=cudaMalloc(&input,(size_t)n*sizeof(float));
  if(status!=cudaSuccess){free(host_input_indices);return status;}
#define SORT_ALLOC(p,z) status=cudaMalloc(&(p),(z));if(status!=cudaSuccess)goto done_segmented_sort
  SORT_ALLOC(sorted,(size_t)n*sizeof(float));SORT_ALLOC(input_indices,(size_t)n*sizeof(int32_t));SORT_ALLOC(sorted_indices,(size_t)n*sizeof(int32_t));
#undef SORT_ALLOC
  status=cudaMemcpyAsync(input,host_input,(size_t)n*sizeof(float),cudaMemcpyHostToDevice,stream);
  if(status==cudaSuccess)status=cudaMemcpyAsync(input_indices,host_input_indices,(size_t)n*sizeof(int32_t),cudaMemcpyHostToDevice,stream);
  if(status==cudaSuccess)status=cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temporary,temporary_bytes,input,sorted,input_indices,sorted_indices,n,rows,offsets,offsets+1,0,8*sizeof(float),stream);
  if(status==cudaSuccess){status=cudaMalloc(&temporary,temporary_bytes);}
  if(status==cudaSuccess)status=cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temporary,temporary_bytes,input,sorted,input_indices,sorted_indices,n,rows,offsets,offsets+1,0,8*sizeof(float),stream);
  if(status==cudaSuccess)status=cudaMemcpy2DAsync(host_values,(size_t)top*sizeof(float),sorted,(size_t)cols*sizeof(float),(size_t)top*sizeof(float),rows,cudaMemcpyDeviceToHost,stream);
  if(status==cudaSuccess)status=cudaMemcpy2DAsync(host_indices,(size_t)top*sizeof(int32_t),sorted_indices,(size_t)cols*sizeof(int32_t),(size_t)top*sizeof(int32_t),rows,cudaMemcpyDeviceToHost,stream);
  if(status==cudaSuccess)status=cudaStreamSynchronize(stream);
done_segmented_sort:cudaFree(temporary);cudaFree(sorted_indices);cudaFree(input_indices);cudaFree(sorted);cudaFree(input);free(host_input_indices);return status;
}

extern "C" int polygeist_cub_segment_reduce_lengths_f32_cuda(
    int32_t n,int32_t segments,int32_t op,const float *host_input,
    const int32_t *host_lengths,float *host_output,cudaStream_t stream){
  if(n<0||segments<=0||op<0||op>3||!host_input||!host_lengths||!host_output)return-1;
  int32_t *host_offsets=(int32_t*)malloc((size_t)(segments+1)*sizeof(int32_t));if(!host_offsets)return-1;host_offsets[0]=0;
  for(int32_t s=0;s<segments;++s){if(host_lengths[s]<0||host_offsets[s]>n-host_lengths[s]){free(host_offsets);return-1;}host_offsets[s+1]=host_offsets[s]+host_lengths[s];}
  if(host_offsets[segments]>n){free(host_offsets);return-1;}
  float *input=nullptr,*output=nullptr;int32_t *offsets=nullptr;void *temporary=nullptr;size_t temporary_bytes=0;
  cudaError_t status=cudaMalloc(&input,(size_t)n*sizeof(float));if(status!=cudaSuccess){free(host_offsets);return status;}
#define SR_ALLOC(p,z) status=cudaMalloc(&(p),(z));if(status!=cudaSuccess)goto done_segment_lengths
  SR_ALLOC(output,(size_t)segments*sizeof(float));SR_ALLOC(offsets,(size_t)(segments+1)*sizeof(int32_t));
#undef SR_ALLOC
  status=cudaMemcpyAsync(input,host_input,(size_t)n*sizeof(float),cudaMemcpyHostToDevice,stream);
  if(status==cudaSuccess)status=cudaMemcpyAsync(offsets,host_offsets,(size_t)(segments+1)*sizeof(int32_t),cudaMemcpyHostToDevice,stream);
  if(status==cudaSuccess){if(op<=1)status=cub::DeviceSegmentedReduce::Reduce(temporary,temporary_bytes,input,output,segments,offsets,offsets+1,cub::Sum{},0.0f,stream);
    else if(op==2)status=cub::DeviceSegmentedReduce::Reduce(temporary,temporary_bytes,input,output,segments,offsets,offsets+1,cub::Max{},-3.402823466e38f,stream);
    else status=cub::DeviceSegmentedReduce::Reduce(temporary,temporary_bytes,input,output,segments,offsets,offsets+1,cub::Min{},3.402823466e38f,stream);}
  if(status==cudaSuccess)status=cudaMalloc(&temporary,temporary_bytes);
  if(status==cudaSuccess){if(op<=1)status=cub::DeviceSegmentedReduce::Reduce(temporary,temporary_bytes,input,output,segments,offsets,offsets+1,cub::Sum{},0.0f,stream);
    else if(op==2)status=cub::DeviceSegmentedReduce::Reduce(temporary,temporary_bytes,input,output,segments,offsets,offsets+1,cub::Max{},-3.402823466e38f,stream);
    else status=cub::DeviceSegmentedReduce::Reduce(temporary,temporary_bytes,input,output,segments,offsets,offsets+1,cub::Min{},3.402823466e38f,stream);}
  if(status==cudaSuccess)status=cudaMemcpyAsync(host_output,output,(size_t)segments*sizeof(float),cudaMemcpyDeviceToHost,stream);
  if(status==cudaSuccess)status=cudaStreamSynchronize(stream);
  if(status==cudaSuccess&&op==1)for(int32_t s=0;s<segments;++s)if(host_lengths[s]>0)host_output[s]/=host_lengths[s];
done_segment_lengths:cudaFree(temporary);cudaFree(offsets);cudaFree(output);cudaFree(input);free(host_offsets);return status;
}
struct ProductSegmentOffset {
  int32_t width;
  __host__ __device__ int32_t operator()(int32_t row) const {
    return row * width;
  }
};

struct ProductSegmentKey {
  int32_t width;
  __host__ __device__ int32_t operator()(int64_t index) const {
    return (int32_t)(index / width);
  }
};

struct ProductF32 {
  __host__ __device__ float operator()(float lhs, float rhs) const {
    return lhs * rhs;
  }
};


struct NonzeroF32 {
  __host__ __device__ int32_t operator()(float value) const {
    return value != 0.0f ? 1 : 0;
  }
};

struct EqualIndexF32 {
  const float *lhs;
  const float *rhs;
  __host__ __device__ int32_t operator()(int32_t index) const {
    return lhs[index] == rhs[index] ? 1 : 0;
  }
};

extern "C" int polygeist_cub_inclusive_sum1d_f32_cuda(
    int32_t n, const float *input, float *final_value, float *output,
    cudaStream_t stream) {
  if (n < 0) return -1;
  if (n == 0) {
    if (final_value) *final_value = 0.0f;
    return 0;
  }
  float *d_input = nullptr, *d_output = nullptr;
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
  cudaError_t status = cudaMalloc(&d_input, (size_t)n * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&d_output, (size_t)n * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(d_input, input, (size_t)n * sizeof(float),
                             cudaMemcpyHostToDevice, stream);
  if (status == cudaSuccess)
    status = cub::DeviceScan::InclusiveSum(
        temporary, temporary_bytes, d_input, d_output, n, stream);
  if (status == cudaSuccess) status = cudaMalloc(&temporary, temporary_bytes);
  if (status == cudaSuccess)
    status = cub::DeviceScan::InclusiveSum(
        temporary, temporary_bytes, d_input, d_output, n, stream);
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(output, d_output, (size_t)n * sizeof(float),
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess && final_value)
    status = cudaMemcpyAsync(final_value, d_output + n - 1, sizeof(float),
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
  cudaFree(temporary);
  cudaFree(d_output);
  cudaFree(d_input);
  return status;
}

extern "C" int polygeist_cub_count_nonzero1d_f32_cuda(
    int32_t n, const float *host_input, int32_t *host_out,
    cudaStream_t stream) {
  if (n < 0) return -1;
  if (n == 0) { *host_out = 0; return 0; }
  size_t input_bytes = static_cast<size_t>(n) * sizeof(float);
  float *device_input = nullptr;
  int32_t *device_out = nullptr;
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
  cudaError_t status = cudaMalloc(&device_input, input_bytes);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaMalloc(&device_out, sizeof(int32_t));
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpyAsync(device_input, host_input, input_bytes,
                           cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) goto cleanup;
  {
    cub::TransformInputIterator<int32_t, NonzeroF32, const float *> input(
        device_input, NonzeroF32{});
    status = cub::DeviceReduce::Sum(
        temporary, temporary_bytes, input, device_out, n, stream);
    if (status != cudaSuccess) goto cleanup;
    status = cudaMalloc(&temporary, temporary_bytes);
    if (status != cudaSuccess) goto cleanup;
    status = cub::DeviceReduce::Sum(
        temporary, temporary_bytes, input, device_out, n, stream);
  }
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(host_out, device_out, sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cleanup:
  cudaFree(temporary);
  cudaFree(device_out);
  cudaFree(device_input);
  return static_cast<int>(status);
}

extern "C" int polygeist_cub_segmented_count_nonzero2d_f32_cuda(
    int32_t rows, int32_t cols, const float *host_input, int32_t *host_out,
    cudaStream_t stream) {
  if (rows < 0 || cols < 0) return -1;
  if (rows == 0) return 0;
  int64_t count = static_cast<int64_t>(rows) * cols;
  size_t input_bytes = static_cast<size_t>(count) * sizeof(float);
  size_t output_bytes = static_cast<size_t>(rows) * sizeof(int32_t);
  float *device_input = nullptr;
  int32_t *device_out = nullptr;
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
  cudaError_t status = cudaMalloc(&device_input, input_bytes);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaMalloc(&device_out, output_bytes);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpyAsync(device_input, host_input, input_bytes,
                           cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) goto cleanup;
  {
    using Counting = cub::CountingInputIterator<int32_t>;
    using Offsets = cub::TransformInputIterator<
        int32_t, ProductSegmentOffset, Counting>;
    Counting counting(0);
    Offsets offsets(counting, ProductSegmentOffset{cols});
    cub::TransformInputIterator<int32_t, NonzeroF32, const float *> input(
        device_input, NonzeroF32{});
    status = cub::DeviceSegmentedReduce::Sum(
        temporary, temporary_bytes, input, device_out, rows,
        offsets, offsets + 1, stream);
    if (status != cudaSuccess) goto cleanup;
    status = cudaMalloc(&temporary, temporary_bytes);
    if (status != cudaSuccess) goto cleanup;
    status = cub::DeviceSegmentedReduce::Sum(
        temporary, temporary_bytes, input, device_out, rows,
        offsets, offsets + 1, stream);
  }
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(host_out, device_out, output_bytes,
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cleanup:
  cudaFree(temporary);
  cudaFree(device_out);
  cudaFree(device_input);
  return static_cast<int>(status);
}

extern "C" int polygeist_cub_equal_all1d_f32_cuda(
    int32_t n, const float *host_lhs, const float *host_rhs,
    int32_t *host_out, cudaStream_t stream) {
  if (n < 0) return -1;
  if (n == 0) { *host_out = 1; return 0; }
  size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float *device_lhs = nullptr, *device_rhs = nullptr;
  int32_t *device_out = nullptr;
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
  cudaError_t status = cudaMalloc(&device_lhs, bytes);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaMalloc(&device_rhs, bytes);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&device_out, sizeof(int32_t));
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpyAsync(device_lhs, host_lhs, bytes,
                           cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpyAsync(device_rhs, host_rhs, bytes,
                           cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) goto cleanup;
  {
    using Counting = cub::CountingInputIterator<int32_t>;
    Counting first(0);
    cub::TransformInputIterator<int32_t, EqualIndexF32, Counting> input(
        first, EqualIndexF32{device_lhs, device_rhs});
    status = cub::DeviceReduce::Min(
        temporary, temporary_bytes, input, device_out, n, stream);
    if (status != cudaSuccess) goto cleanup;
    status = cudaMalloc(&temporary, temporary_bytes);
    if (status != cudaSuccess) goto cleanup;
    status = cub::DeviceReduce::Min(
        temporary, temporary_bytes, input, device_out, n, stream);
  }
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(host_out, device_out, sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cleanup:
  cudaFree(temporary);
  cudaFree(device_out);
  cudaFree(device_rhs);
  cudaFree(device_lhs);
  return static_cast<int>(status);
}

extern "C" int polygeist_cub_exclusive_sum1d_i32_cuda(
    int32_t n, const int32_t *input, int32_t *output, cudaStream_t stream) {
  if (n < 0) return -1;
  if (n == 0) { output[0] = 0; return 0; }
  int32_t *d_input = nullptr, *d_output = nullptr;
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
  cudaError_t status = cudaMalloc(&d_input, (size_t)n * sizeof(int32_t));
  if (status == cudaSuccess)
    status = cudaMalloc(&d_output, (size_t)(n + 1) * sizeof(int32_t));
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(d_input, input, (size_t)n * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream);
  if (status == cudaSuccess)
    status = cub::DeviceScan::ExclusiveSum(
        temporary, temporary_bytes, d_input, d_output, n, stream);
  if (status == cudaSuccess) status = cudaMalloc(&temporary, temporary_bytes);
  if (status == cudaSuccess)
    status = cub::DeviceScan::ExclusiveSum(
        temporary, temporary_bytes, d_input, d_output, n, stream);
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(output, d_output, (size_t)n * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream);
  int32_t last_prefix = 0, last_input = 0;
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(&last_prefix, d_output + n - 1, sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(&last_input, d_input + n - 1, sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
  if (status == cudaSuccess) output[n] = last_prefix + last_input;
  cudaFree(temporary); cudaFree(d_output); cudaFree(d_input);
  return status;
}

extern "C" int polygeist_cub_segmented_inclusive_product2d_f32_cuda(
    int32_t rows, int32_t cols, const float *input, float *final_values,
    float *output, cudaStream_t stream) {
  if (rows < 0 || cols < 0) return -1;
  int64_t count = (int64_t)rows * cols;
  if (count == 0) return 0;
  if (!input || !output) return cudaErrorInvalidValue;
  float *d_input = nullptr, *d_output = nullptr;
  void *resident = nullptr;
  bool owns_input = !cub_device_pointer(input, &resident);
  if (!owns_input) d_input = static_cast<float *>(resident);
  bool owns_output = !cub_device_pointer(output, &resident);
  if (!owns_output) d_output = static_cast<float *>(resident);
  // The CUB temporary buffer is shape-dependent scratch, not a result.
  // Retain it across resident calls, with separate storage per host thread
  // and CUDA device. No project-authored scan kernel is involved.
  struct Workspace {
    void *pointer = nullptr;
    size_t capacity = 0;
    int device = -1;
  };
  static thread_local Workspace workspace;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (workspace.device != device) {
    if (workspace.pointer) {
      cudaSetDevice(workspace.device);
      cudaFree(workspace.pointer);
      cudaSetDevice(device);
    }
    workspace = Workspace{};
    workspace.device = device;
  }
  size_t temporary_bytes = 0;
  if (owns_input)
    status = cudaMalloc(&d_input, (size_t)count * sizeof(float));
  if (status == cudaSuccess && owns_output)
    status = cudaMalloc(&d_output, (size_t)count * sizeof(float));
  if (status == cudaSuccess && owns_input)
    status = cudaMemcpyAsync(d_input, input, (size_t)count * sizeof(float),
                             cudaMemcpyHostToDevice, stream);
  using Counting = cub::CountingInputIterator<int64_t>;
  using Keys = cub::TransformInputIterator<int32_t, ProductSegmentKey, Counting>;
  Counting counting(0);
  Keys keys(counting, ProductSegmentKey{cols});
  if (status == cudaSuccess)
    status = cub::DeviceScan::InclusiveScanByKey(
        nullptr, temporary_bytes, keys, d_input, d_output,
        ProductF32{}, count, cub::Equality{}, stream);
  if (status == cudaSuccess && temporary_bytes > workspace.capacity) {
    if (workspace.pointer) cudaFree(workspace.pointer);
    workspace.pointer = nullptr;
    workspace.capacity = 0;
    status = cudaMalloc(&workspace.pointer, temporary_bytes);
    if (status == cudaSuccess) workspace.capacity = temporary_bytes;
  }
  if (status == cudaSuccess)
    status = cub::DeviceScan::InclusiveScanByKey(
        workspace.pointer, temporary_bytes, keys, d_input, d_output,
        ProductF32{}, count, cub::Equality{}, stream);
  if (status == cudaSuccess && owns_output)
    status = cudaMemcpyAsync(output, d_output, (size_t)count * sizeof(float),
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess && final_values)
    status = cudaMemcpy2DAsync(final_values, sizeof(float),
                               d_output + cols - 1,
                               (size_t)cols * sizeof(float), sizeof(float), rows,
                               cudaMemcpyDefault, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
  if (owns_output) cudaFree(d_output);
  if (owns_input) cudaFree(d_input);
  return status;
}

struct SegmentOffset {
  int32_t width;
  __host__ __device__ int32_t operator()(int32_t row) const {
    return row * width;
  }
};

struct LogicalAndI32 {
  __host__ __device__ int32_t operator()(int32_t a, int32_t b) const {
    return (a != 0 && b != 0) ? 1 : 0;
  }
};

struct LogicalOrI32 {
  __host__ __device__ int32_t operator()(int32_t a, int32_t b) const {
    return (a != 0 || b != 0) ? 1 : 0;
  }
};

struct BitXorI32 {
  __host__ __device__ int32_t operator()(int32_t a, int32_t b) const {
    return a ^ b;
  }
};

struct PrefixBeginOffset {
  int32_t width;
  __host__ __device__ int32_t operator()(int32_t row) const {
    return row * width;
  }
};

struct PrefixEndOffset {
  int32_t width;
  const int32_t *lengths;
  __host__ __device__ int32_t operator()(int32_t row) const {
    int32_t length = lengths[row];
    if (length < 0) length = 0;
    if (length > width) length = width;
    return row * width + length;
  }
};

static bool cub_device_pointer(const void *pointer, void **device_pointer);

template <typename Op>
static cudaError_t segmented_reduce(
    int32_t rows, int32_t cols, const int32_t *host_x, int32_t *host_out,
    int32_t identity, Op op, cudaStream_t stream) {
  if (rows <= 0 || cols < 0) return cudaSuccess;
  size_t input_bytes = static_cast<size_t>(rows) * cols * sizeof(int32_t);
  size_t output_bytes = static_cast<size_t>(rows) * sizeof(int32_t);
  int32_t *device_x = nullptr, *device_out = nullptr;
  void *resident = nullptr;
  bool owns_x = !cub_device_pointer(host_x, &resident);
  if (!owns_x)
    device_x = static_cast<int32_t *>(resident);
  bool owns_out = !cub_device_pointer(host_out, &resident);
  if (!owns_out)
    device_out = static_cast<int32_t *>(resident);
  using Counting = cub::CountingInputIterator<int32_t>;
  using Offsets = cub::TransformInputIterator<int32_t, SegmentOffset, Counting>;
  Counting counting(0);
  Offsets offsets(counting, SegmentOffset{cols});
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
  cudaError_t status = cudaSuccess;
  if (owns_x) {
    status = cudaMalloc(&device_x, input_bytes);
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(device_x, host_x, input_bytes,
                             cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) goto cleanup;
  }
  if (owns_out) {
    status = cudaMalloc(&device_out, output_bytes);
    if (status != cudaSuccess) goto cleanup;
  }

  status = cub::DeviceSegmentedReduce::Reduce(
      temporary, temporary_bytes, device_x, device_out, rows,
      offsets, offsets + 1, op, identity, stream);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&temporary, temporary_bytes);
  if (status != cudaSuccess) goto cleanup;
  status = cub::DeviceSegmentedReduce::Reduce(
      temporary, temporary_bytes, device_x, device_out, rows,
      offsets, offsets + 1, op, identity, stream);
  if (status == cudaSuccess && owns_out)
    status = cudaMemcpyAsync(host_out, device_out, output_bytes,
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
  cudaFree(temporary);
cleanup:
  if (owns_out) cudaFree(device_out);
  if (owns_x) cudaFree(device_x);
  return status;
}

extern "C" int polygeist_cub_segmented_reduce_i32_cuda(
    int32_t op, int32_t rows, int32_t cols, const int32_t *x, int32_t *out,
    cudaStream_t stream) {
  cudaError_t status;
  if (op == 0)
    status = segmented_reduce(rows, cols, x, out, 1, LogicalAndI32{}, stream);
  else if (op == 1)
    status = segmented_reduce(rows, cols, x, out, 0, LogicalOrI32{}, stream);
  else if (op == 2)
    status = segmented_reduce(rows, cols, x, out, 0, BitXorI32{}, stream);
  else
    return -1;
  return static_cast<int>(status);
}

static bool cub_device_pointer(const void *pointer, void **device_pointer);

template <typename T, typename Op>
static cudaError_t segmented_reduce_numeric(
    int32_t rows, int32_t cols, const T *host_x, T *host_out,
    T identity, Op op, cudaStream_t stream) {
  if (rows <= 0) return cudaSuccess;
  if (cols <= 0) return cudaErrorInvalidValue;
  size_t input_bytes = (size_t)rows * cols * sizeof(T);
  size_t output_bytes = (size_t)rows * sizeof(T);
  T *device_x = nullptr, *device_out = nullptr;
  bool owns_x = !cub_device_pointer(host_x, (void **)&device_x);
  bool owns_out = !cub_device_pointer(host_out, (void **)&device_out);
  using Counting = cub::CountingInputIterator<int32_t>;
  using Offsets = cub::TransformInputIterator<int32_t, SegmentOffset, Counting>;
  Counting counting(0); Offsets offsets(counting, SegmentOffset{cols});
  void *temporary = nullptr; size_t temporary_bytes = 0;
  cudaError_t status = cudaSuccess;
  if (owns_x) status = cudaMalloc(&device_x, input_bytes);
  if (status != cudaSuccess) return status;
  if (owns_out) status = cudaMalloc(&device_out, output_bytes);
  if (status != cudaSuccess) goto cleanup_f32_reduce;
  if (owns_x)
    status = cudaMemcpyAsync(device_x, host_x, input_bytes,
                             cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) goto cleanup_f32_reduce;
  status = cub::DeviceSegmentedReduce::Reduce(
      temporary, temporary_bytes, device_x, device_out, rows,
      offsets, offsets + 1, op, identity, stream);
  if (status != cudaSuccess) goto cleanup_f32_reduce;
  status = cudaMalloc(&temporary, temporary_bytes);
  if (status != cudaSuccess) goto cleanup_f32_reduce;
  status = cub::DeviceSegmentedReduce::Reduce(
      temporary, temporary_bytes, device_x, device_out, rows,
      offsets, offsets + 1, op, identity, stream);
  if (status == cudaSuccess && owns_out)
    status = cudaMemcpyAsync(host_out, device_out, output_bytes,
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cleanup_f32_reduce:
  cudaFree(temporary);
  if (owns_out) cudaFree(device_out);
  if (owns_x) cudaFree(device_x);
  return status;
}

struct NanToZeroF32 {
  __host__ __device__ float operator()(float value) const {
    return isnan(value) ? 0.0f : value;
  }
};

static cudaError_t segmented_nansum_f32(
    int32_t rows, int32_t cols, const float *host_x, float *host_out,
    cudaStream_t stream) {
  if (rows <= 0) return cudaSuccess;
  if (cols <= 0) return cudaErrorInvalidValue;
  size_t input_bytes = (size_t)rows * cols * sizeof(float);
  size_t output_bytes = (size_t)rows * sizeof(float);
  float *device_x = nullptr, *device_out = nullptr;
  bool owns_x = !cub_device_pointer(host_x, (void **)&device_x);
  bool owns_out = !cub_device_pointer(host_out, (void **)&device_out);
  using Counting = cub::CountingInputIterator<int32_t>;
  using Offsets = cub::TransformInputIterator<int32_t, SegmentOffset, Counting>;
  Counting counting(0);
  Offsets offsets(counting, SegmentOffset{cols});
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
  cudaError_t status = cudaSuccess;
  if (owns_x) status = cudaMalloc(&device_x, input_bytes);
  if (status != cudaSuccess) return status;
  if (owns_out) status = cudaMalloc(&device_out, output_bytes);
  if (status != cudaSuccess) goto cleanup_nansum;
  if (owns_x)
    status = cudaMemcpyAsync(device_x, host_x, input_bytes,
                             cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) goto cleanup_nansum;
  {
    using Input = cub::TransformInputIterator<
        float, NanToZeroF32, const float *>;
    Input input(device_x, NanToZeroF32{});
    status = cub::DeviceSegmentedReduce::Reduce(
        temporary, temporary_bytes, input, device_out, rows,
        offsets, offsets + 1, cub::Sum{}, 0.0f, stream);
    if (status != cudaSuccess) goto cleanup_nansum;
    status = cudaMalloc(&temporary, temporary_bytes);
    if (status != cudaSuccess) goto cleanup_nansum;
    status = cub::DeviceSegmentedReduce::Reduce(
        temporary, temporary_bytes, input, device_out, rows,
        offsets, offsets + 1, cub::Sum{}, 0.0f, stream);
  }
  if (status == cudaSuccess && owns_out)
    status = cudaMemcpyAsync(host_out, device_out, output_bytes,
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cleanup_nansum:
  cudaFree(temporary);
  if (owns_out) cudaFree(device_out);
  if (owns_x) cudaFree(device_x);
  return status;
}

extern "C" int polygeist_cub_segmented_reduce_f32_cuda(
    int32_t op, int32_t rows, int32_t cols, const float *x, float *out,
    cudaStream_t stream) {
  cudaError_t status;
  if (op == 0)
    status = segmented_reduce_numeric(rows, cols, x, out, 0.0f, cub::Sum{}, stream);
  else if (op == 1)
    status = segmented_reduce_numeric(rows, cols, x, out, INFINITY, cub::Min{}, stream);
  else if (op == 2)
    status = segmented_reduce_numeric(rows, cols, x, out, -INFINITY, cub::Max{}, stream);
  else if (op == 3)
    status = segmented_nansum_f32(rows, cols, x, out, stream);
  else return -1;
  return (int)status;
}

extern "C" int polygeist_cub_segmented_reduce_f64_cuda(
    int32_t op, int32_t rows, int32_t cols, const double *x, double *out,
    cudaStream_t stream) {
  if (op != 0) return -1;
  return (int)segmented_reduce_numeric(
      rows, cols, x, out, 0.0, cub::Sum{}, stream);
}

static bool cub_device_pointer(const void *pointer, void **device_pointer) {
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
#if CUDART_VERSION >= 10000
  if (attributes.type != cudaMemoryTypeDevice &&
      attributes.type != cudaMemoryTypeManaged)
    return false;
#else
  if (attributes.memoryType != cudaMemoryTypeDevice)
    return false;
#endif
  *device_pointer = attributes.devicePointer
                        ? attributes.devicePointer
                        : const_cast<void *>(pointer);
  return true;
}

struct QuantColOffsetValueI8I32 {
  const int8_t *weights;
  int32_t rows;
  int32_t cols;
  __host__ __device__ int32_t operator()(int32_t linear) const {
    int32_t col = linear / rows;
    int32_t row = linear - col * rows;
    return (int32_t)weights[(int64_t)row * cols + col];
  }
};

extern "C" int polygeist_cub_quant_col_offsets_i8_i32_cuda(
    int32_t rows, int32_t cols, int32_t offset,
    const int8_t *weights, int32_t *out, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !weights || !out)
    return -1;
  if ((int64_t)rows * cols > INT32_MAX)
    return -1;

  const size_t input_bytes = (size_t)rows * cols * sizeof(int8_t);
  const size_t output_bytes = (size_t)cols * sizeof(int32_t);
  int8_t *device_weights = nullptr;
  int32_t *device_out = nullptr;
  void *resident = nullptr;
  bool owns_weights = !cub_device_pointer(weights, &resident);
  if (owns_weights) {
    cudaError_t status = cudaMalloc(&device_weights, input_bytes);
    if (status != cudaSuccess) return (int)status;
    status = cudaMemcpyAsync(device_weights, weights, input_bytes,
                             cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
      cudaFree(device_weights);
      return (int)status;
    }
  } else {
    device_weights = static_cast<int8_t *>(resident);
  }
  bool owns_out = !cub_device_pointer(out, &resident);
  if (owns_out) {
    cudaError_t status = cudaMalloc(&device_out, output_bytes);
    if (status != cudaSuccess) {
      if (owns_weights) cudaFree(device_weights);
      return (int)status;
    }
  } else {
    device_out = static_cast<int32_t *>(resident);
  }

  using Counting = cub::CountingInputIterator<int32_t>;
  using Values = cub::TransformInputIterator<
      int32_t, QuantColOffsetValueI8I32, Counting>;
  using Offsets = cub::TransformInputIterator<
      int32_t, SegmentOffset, Counting>;
  Counting counting(0);
  Values values(counting, QuantColOffsetValueI8I32{
      device_weights, rows, cols});
  Offsets offsets(counting, SegmentOffset{rows});
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
  cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
      temporary, temporary_bytes, values, device_out, cols,
      offsets, offsets + 1, cub::Sum{}, -offset, stream);
  if (status == cudaSuccess)
    status = cudaMalloc(&temporary, temporary_bytes);
  if (status == cudaSuccess)
    status = cub::DeviceSegmentedReduce::Reduce(
        temporary, temporary_bytes, values, device_out, cols,
        offsets, offsets + 1, cub::Sum{}, -offset, stream);
  if (status == cudaSuccess && owns_out)
    status = cudaMemcpyAsync(out, device_out, output_bytes,
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess)
    status = cudaStreamSynchronize(stream);
  cudaFree(temporary);
  if (owns_out) cudaFree(device_out);
  if (owns_weights) cudaFree(device_weights);
  return (int)status;
}

extern "C" int polygeist_cub_adjacent_difference_f32_cuda(
    int32_t count, const float *input, float *out, cudaStream_t stream) {
  if (count < 2 || !input || !out)
    return -1;

  const size_t input_bytes = (size_t)count * sizeof(float);
  const size_t output_bytes = (size_t)(count - 1) * sizeof(float);
  const uintptr_t input_begin = reinterpret_cast<uintptr_t>(input);
  const uintptr_t output_begin = reinterpret_cast<uintptr_t>(out);
  if (input_begin < output_begin + output_bytes &&
      output_begin < input_begin + input_bytes)
    return -1;
  float *device_input = nullptr;
  float *device_result = nullptr;
  void *resident = nullptr;
  bool owns_input = !cub_device_pointer(input, &resident);
  cudaError_t status = cudaSuccess;
  if (owns_input) {
    status = cudaMalloc(&device_input, input_bytes);
    if (status != cudaSuccess) return (int)status;
    status = cudaMemcpyAsync(device_input, input, input_bytes,
                             cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) goto cleanup_adjacent_difference;
  } else {
    device_input = static_cast<float *>(resident);
  }
  status = cudaMalloc(&device_result, input_bytes);
  if (status != cudaSuccess) goto cleanup_adjacent_difference;

  {
    void *temporary = nullptr;
    size_t temporary_bytes = 0;
    status = cub::DeviceAdjacentDifference::SubtractLeftCopy(
        temporary, temporary_bytes, device_input, device_result,
        count, cub::Difference{}, stream);
    if (status == cudaSuccess)
      status = cudaMalloc(&temporary, temporary_bytes);
    if (status == cudaSuccess)
      status = cub::DeviceAdjacentDifference::SubtractLeftCopy(
          temporary, temporary_bytes, device_input, device_result,
          count, cub::Difference{}, stream);
    if (status == cudaSuccess) {
      void *device_out = nullptr;
      if (cub_device_pointer(out, &device_out))
        status = cudaMemcpyAsync(device_out, device_result + 1, output_bytes,
                                 cudaMemcpyDeviceToDevice, stream);
      else
        status = cudaMemcpyAsync(out, device_result + 1, output_bytes,
                                 cudaMemcpyDeviceToHost, stream);
    }
    if (status == cudaSuccess)
      status = cudaStreamSynchronize(stream);
    cudaFree(temporary);
  }

cleanup_adjacent_difference:
  cudaFree(device_result);
  if (owns_input) cudaFree(device_input);
  return (int)status;
}

struct IndexedValueF32 {
  int32_t index;
  float value;
};

struct MakeIndexedValueF32 {
  const float *values;
  int32_t cols;
  __host__ __device__ IndexedValueF32 operator()(int64_t linear) const {
    return {(int32_t)(linear % cols), values[linear]};
  }
};

struct ArgReduceF32 {
  int32_t op;
  __host__ __device__ IndexedValueF32 operator()(
      IndexedValueF32 a, IndexedValueF32 b) const {
    if (a.index == INT32_MAX) return b;
    if (b.index == INT32_MAX) return a;
    bool a_nan = isnan(a.value);
    bool b_nan = isnan(b.value);
    if (a_nan || b_nan) {
      // The source seeds from column zero and uses an ordered comparison:
      // NaN at column zero remains the answer, while later NaNs never win.
      int32_t a_rank = a_nan ? (a.index == 0 ? 2 : 0) : 1;
      int32_t b_rank = b_nan ? (b.index == 0 ? 2 : 0) : 1;
      if (a_rank != b_rank) return b_rank > a_rank ? b : a;
      return b.index < a.index ? b : a;
    }
    bool b_better = op == 0 ? b.value > a.value : b.value < a.value;
    bool tie = b.value == a.value;
    return (b_better || (tie && b.index < a.index)) ? b : a;
  }
};

extern "C" int polygeist_cub_segmented_argreduce_f32_cuda(
    int32_t op, int32_t rows, int32_t cols, const float *host_x,
    int32_t *host_out, cudaStream_t stream) {
  if ((op != 0 && op != 1) || rows <= 0 || cols <= 0) return -1;
  size_t input_bytes = (size_t)rows * cols * sizeof(float);
  size_t pair_bytes = (size_t)rows * sizeof(IndexedValueF32);
  float *device_x = nullptr;
  IndexedValueF32 *device_pairs = nullptr;
  int32_t *device_out = nullptr;
  void *resident = nullptr;
  bool owns_x = !cub_device_pointer(host_x, &resident);
  if (!owns_x) device_x = static_cast<float *>(resident);
  bool owns_out = !cub_device_pointer(host_out, &resident);
  if (!owns_out) device_out = static_cast<int32_t *>(resident);
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
  cudaError_t status = cudaSuccess;
  if (owns_x) {
    status = cudaMalloc(&device_x, input_bytes);
    if (status != cudaSuccess) return (int)status;
  }
  status = cudaMalloc(&device_pairs, pair_bytes);
  if (status != cudaSuccess) goto cleanup;
  if (owns_x) {
    status = cudaMemcpyAsync(device_x, host_x, input_bytes,
                             cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) goto cleanup;
  }
  {
    using Counting = cub::CountingInputIterator<int64_t>;
    using Values = cub::TransformInputIterator<
        IndexedValueF32, MakeIndexedValueF32, Counting>;
    using OffsetCounting = cub::CountingInputIterator<int32_t>;
    using Offsets = cub::TransformInputIterator<
        int32_t, SegmentOffset, OffsetCounting>;
    Counting counting(0);
    Values values(counting, MakeIndexedValueF32{device_x, cols});
    OffsetCounting offset_counting(0);
    Offsets offsets(offset_counting, SegmentOffset{cols});
    IndexedValueF32 identity = {
        INT32_MAX, op == 0 ? -INFINITY : INFINITY};
    status = cub::DeviceSegmentedReduce::Reduce(
        temporary, temporary_bytes, values, device_pairs, rows,
        offsets, offsets + 1, ArgReduceF32{op}, identity, stream);
    if (status != cudaSuccess) goto cleanup;
    status = cudaMalloc(&temporary, temporary_bytes);
    if (status != cudaSuccess) goto cleanup;
    status = cub::DeviceSegmentedReduce::Reduce(
        temporary, temporary_bytes, values, device_pairs, rows,
        offsets, offsets + 1, ArgReduceF32{op}, identity, stream);
    if (status != cudaSuccess) goto cleanup;
  }
  if (status == cudaSuccess) {
    void *destination = owns_out ? static_cast<void *>(host_out)
                                 : static_cast<void *>(device_out);
    status = cudaMemcpy2DAsync(destination, sizeof(int32_t), device_pairs,
                               sizeof(IndexedValueF32), sizeof(int32_t), rows,
                               owns_out ? cudaMemcpyDeviceToHost
                                        : cudaMemcpyDeviceToDevice,
                               stream);
  }
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cleanup:
  if (temporary) cudaFree(temporary);
  if (device_pairs) cudaFree(device_pairs);
  if (owns_x && device_x) cudaFree(device_x);
  return (int)status;
}

template <typename T, typename Op>
static cudaError_t segmented_prefix_reduce(
    int32_t rows, int32_t cols, const T *host_x, const int32_t *host_lengths,
    T *host_out, T identity, Op op, cudaStream_t stream) {
  if (rows <= 0 || cols < 0) return cudaSuccess;
  size_t input_bytes = static_cast<size_t>(rows) * cols * sizeof(T);
  size_t lengths_bytes = static_cast<size_t>(rows) * sizeof(int32_t);
  size_t output_bytes = static_cast<size_t>(rows) * sizeof(T);
  T *device_x = nullptr, *device_out = nullptr;
  int32_t *device_lengths = nullptr;
  void *temporary = nullptr;
  size_t temporary_bytes = 0;
  cudaError_t status = cudaMalloc(&device_x, input_bytes);
  if (status != cudaSuccess) return status;
  status = cudaMalloc(&device_out, output_bytes);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&device_lengths, lengths_bytes);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpyAsync(device_x, host_x, input_bytes,
                           cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpyAsync(device_lengths, host_lengths, lengths_bytes,
                           cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) goto cleanup;
  {
    using Counting = cub::CountingInputIterator<int32_t>;
    using Begins = cub::TransformInputIterator<
        int32_t, PrefixBeginOffset, Counting>;
    using Ends = cub::TransformInputIterator<
        int32_t, PrefixEndOffset, Counting>;
    Counting counting(0);
    Begins begins(counting, PrefixBeginOffset{cols});
    Ends ends(counting, PrefixEndOffset{cols, device_lengths});
    status = cub::DeviceSegmentedReduce::Reduce(
        temporary, temporary_bytes, device_x, device_out, rows,
        begins, ends, op, identity, stream);
    if (status != cudaSuccess) goto cleanup;
    status = cudaMalloc(&temporary, temporary_bytes);
    if (status != cudaSuccess) goto cleanup;
    status = cub::DeviceSegmentedReduce::Reduce(
        temporary, temporary_bytes, device_x, device_out, rows,
        begins, ends, op, identity, stream);
  }
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(host_out, device_out, output_bytes,
                             cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cleanup:
  cudaFree(temporary);
  cudaFree(device_lengths);
  cudaFree(device_out);
  cudaFree(device_x);
  return status;
}

extern "C" int polygeist_cub_segmented_prefix_sum_f32_cuda(
    int32_t rows, int32_t cols, const float *x, const int32_t *lengths,
    float *out, cudaStream_t stream) {
  return static_cast<int>(segmented_prefix_reduce(
      rows, cols, x, lengths, out, 0.0f, cub::Sum{}, stream));
}

extern "C" int polygeist_cub_segmented_prefix_logical_and_i32_cuda(
    int32_t rows, int32_t cols, const int32_t *x, const int32_t *lengths,
    int32_t *out, cudaStream_t stream) {
  return static_cast<int>(segmented_prefix_reduce(
      rows, cols, x, lengths, out, int32_t{1}, LogicalAndI32{}, stream));
}

// Absolute maximum with the source reduction's zero seed and ordered
// comparison: NaNs are ignored, and an all-NaN input reduces to zero.
struct OrderedAbsF32 {
  __host__ __device__ float operator()(float value) const {
    return value == value ? fabsf(value) : 0.0f;
  }
};

extern "C" int polygeist_cub_maxabs_f32_cuda(
    int32_t count, const float *device_input, float *device_output,
    cudaStream_t stream) {
  if (count < 0 || !device_output || (count && !device_input))
    return cudaErrorInvalidValue;
  if (!count)
    return cudaMemsetAsync(device_output, 0, sizeof(float), stream);
  using Values = cub::TransformInputIterator<float, OrderedAbsF32, const float *>;
  Values values(device_input, OrderedAbsF32{});
  struct Workspace {
    void *pointer = nullptr;
    size_t capacity = 0;
    int device = -1;
  };
  static thread_local Workspace workspace;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (workspace.device != device) {
    if (workspace.pointer) {
      cudaSetDevice(workspace.device);
      cudaFree(workspace.pointer);
      cudaSetDevice(device);
    }
    workspace = Workspace{};
    workspace.device = device;
  }
  size_t bytes = 0;
  status = cub::DeviceReduce::Max(nullptr, bytes, values, device_output,
                                 count, stream);
  if (status == cudaSuccess && bytes > workspace.capacity) {
    if (workspace.pointer) cudaFree(workspace.pointer);
    workspace.pointer = nullptr;
    workspace.capacity = 0;
    status = cudaMalloc(&workspace.pointer, bytes);
    if (status == cudaSuccess) workspace.capacity = bytes;
  }
  if (status == cudaSuccess)
    status = cub::DeviceReduce::Max(workspace.pointer, bytes, values,
                                   device_output, count, stream);
  // The thread-local scratch may be reused by a later call on another stream.
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
  return status;
}

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" int polygeist_cub_segmented_reduce_f32_cuda(
    int32_t, int32_t, int32_t, const float *, float *, cudaStream_t);
extern "C" int polygeist_cub_segmented_reduce_i32_cuda(
    int32_t, int32_t, int32_t, const int32_t *, int32_t *, cudaStream_t);

int main() {
  constexpr int rows=65536, cols=128;
  std::vector<float> f((size_t)rows*cols), sum(rows), minv(rows), maxv(rows),
      sum_ref(rows), min_ref(rows), max_ref(rows);
  std::vector<int32_t> x((size_t)rows*cols), xorv(rows), xor_ref(rows);
  for(int r=0;r<rows;++r){float s=0,mn=0,mx=0;int32_t xv=0;
    for(int c=0;c<cols;++c){float v=(float)(((int64_t)r*17+c*29)%101-50);
      int32_t q=(r*131+c*17)^0x55aa; f[(size_t)r*cols+c]=v;x[(size_t)r*cols+c]=q;
      s+=v;xv^=q;if(c==0||v<mn)mn=v;if(c==0||v>mx)mx=v;}
    sum_ref[r]=s;min_ref[r]=mn;max_ref[r]=mx;xor_ref[r]=xv;}
  cudaStream_t stream;cudaStreamCreate(&stream);
  for(int w=0;w<3;++w){
    if(polygeist_cub_segmented_reduce_f32_cuda(0,rows,cols,f.data(),sum.data(),stream))return 2;
    if(polygeist_cub_segmented_reduce_f32_cuda(1,rows,cols,f.data(),minv.data(),stream))return 3;
    if(polygeist_cub_segmented_reduce_f32_cuda(2,rows,cols,f.data(),maxv.data(),stream))return 4;
    if(polygeist_cub_segmented_reduce_i32_cuda(2,rows,cols,x.data(),xorv.data(),stream))return 5;}
  cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
  auto time=[&](auto call){float total=0;for(int t=0;t<5;++t){cudaEventRecord(begin,stream);
    if(call())return -1.0f;cudaEventRecord(end,stream);cudaEventSynchronize(end);float ms=0;
    cudaEventElapsedTime(&ms,begin,end);total+=ms;}return total*200.0f;};
  float su=time([&]{return polygeist_cub_segmented_reduce_f32_cuda(0,rows,cols,f.data(),sum.data(),stream);});
  float mi=time([&]{return polygeist_cub_segmented_reduce_f32_cuda(1,rows,cols,f.data(),minv.data(),stream);});
  float ma=time([&]{return polygeist_cub_segmented_reduce_f32_cuda(2,rows,cols,f.data(),maxv.data(),stream);});
  float xo=time([&]{return polygeist_cub_segmented_reduce_i32_cuda(2,rows,cols,x.data(),xorv.data(),stream);});
  int se=0,mie=0,mae=0,xe=0;for(int r=0;r<rows;++r){se+=sum[r]!=sum_ref[r];
    mie+=minv[r]!=min_ref[r];mae+=maxv[r]!=max_ref[r];xe+=xorv[r]!=xor_ref[r];}
  std::printf("rows=%d cols=%d sum_us=%.6f min_us=%.6f max_us=%.6f xor_us=%.6f errors=%d/%d/%d/%d\n",
      rows,cols,su,mi,ma,xo,se,mie,mae,xe);
  cudaEventDestroy(end);cudaEventDestroy(begin);cudaStreamDestroy(stream);
  return se||mie||mae||xe;
}

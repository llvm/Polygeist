#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>
extern "C" int polygeist_cub_segmented_sort_descending_f32_i32_cuda(
    int32_t,int32_t,int32_t,const float*,float*,int32_t*,cudaStream_t);
static int64_t check(int rows,int cols,int top,const std::vector<float>&input,
                     const std::vector<float>&values,const std::vector<int32_t>&indices){
  int64_t errors=0;std::vector<int32_t>order(cols);std::iota(order.begin(),order.end(),0);
  for(int r=0;r<rows;++r){std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](int a,int b){return input[(int64_t)r*cols+a]>input[(int64_t)r*cols+b];});
    for(int c=0;c<top;++c){int64_t o=(int64_t)r*top+c;errors+=indices[o]!=order[c]||values[o]!=input[(int64_t)r*cols+order[c]];}}
  return errors;}
static float run(int rows,int cols,int top,const std::vector<float>&input,
                 std::vector<float>&values,std::vector<int32_t>&indices,cudaStream_t st){
  for(int w=0;w<3;++w)if(polygeist_cub_segmented_sort_descending_f32_i32_cuda(rows,cols,top,input.data(),values.data(),indices.data(),st))return -1;
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);float total=0;for(int t=0;t<5;++t){cudaEventRecord(a,st);if(polygeist_cub_segmented_sort_descending_f32_i32_cuda(rows,cols,top,input.data(),values.data(),indices.data(),st))return -1;cudaEventRecord(b,st);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);total+=ms;}return total*200;}
int main(){const int rows=32768,cols=256,top=16;int64_t n=(int64_t)rows*cols;std::vector<float>input(n),sorted(n),top_values((int64_t)rows*top);std::vector<int32_t>indices(n),top_indices((int64_t)rows*top);
  for(int64_t i=0;i<n;++i)input[i]=(float)((i*17+3)%251-125)*.01f;
  cudaStream_t st;cudaStreamCreate(&st);float sort_us=run(rows,cols,cols,input,sorted,indices,st);float top_us=run(rows,cols,top,input,top_values,top_indices,st);
  int64_t errors=check(rows,cols,cols,input,sorted,indices)+check(rows,cols,top,input,top_values,top_indices);
  printf("rows=%d cols=%d top=%d sort_warm_us=%.6f topk_warm_us=%.6f errors=%lld\n",rows,cols,top,sort_us,top_us,(long long)errors);return errors!=0;}

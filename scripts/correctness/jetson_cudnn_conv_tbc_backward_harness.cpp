#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void polygeist_cudnn_conv_tbc_backward_f32(
    int32_t,int32_t,int32_t,int32_t,int32_t,const float*,const float*,float*);

static float reference(int t_out,int b,int i,int T,int B,int I,int O,int K,
                       const float *grad,const float *filter){float acc=0;
  for(int k=0;k<K;++k){int t=t_out-k;if(t<0||t>=T)continue;
    for(int o=0;o<O;++o)acc+=grad[((size_t)t*B+b)*O+o]*filter[((size_t)k*I+i)*O+o];}
  return acc;}

int main(){
  {constexpr int T=30,B=8,I=16,O=24,K=3,TO=T+K-1;
   std::vector<float> grad((size_t)T*B*O),filter((size_t)K*I*O),output((size_t)TO*B*I);
   for(size_t x=0;x<grad.size();++x)grad[x]=(float)((x*17+3)%3-1);
   for(size_t x=0;x<filter.size();++x)filter[x]=(float)((x*13+5)%3-1);
   polygeist_cudnn_conv_tbc_backward_f32(T,B,I,O,K,grad.data(),filter.data(),output.data());
   int errors=0;for(int t=0;t<TO;++t)for(int b=0;b<B;++b)for(int i=0;i<I;++i)
     errors+=output[((size_t)t*B+b)*I+i]!=reference(t,b,i,T,B,I,O,K,grad.data(),filter.data());
   if(errors){std::printf("small_errors=%d\n",errors);return 2;}}
  constexpr int T=4096,B=64,I=64,O=96,K=5,TO=T+K-1;
  std::vector<float> grad((size_t)T*B*O),filter((size_t)K*I*O),output((size_t)TO*B*I);
  for(size_t x=0;x<grad.size();++x)grad[x]=(float)((x*19+7)%3-1);
  for(size_t x=0;x<filter.size();++x)filter[x]=(float)((x*23+11)%3-1);
  for(int w=0;w<3;++w)polygeist_cudnn_conv_tbc_backward_f32(T,B,I,O,K,grad.data(),filter.data(),output.data());
  auto begin=std::chrono::steady_clock::now();for(int n=0;n<5;++n)
    polygeist_cudnn_conv_tbc_backward_f32(T,B,I,O,K,grad.data(),filter.data(),output.data());
  auto end=std::chrono::steady_clock::now();double warm_us=std::chrono::duration<double,std::micro>(end-begin).count()/5;
  int errors=0;float max_error=0;for(int sample=0;sample<8192;++sample){size_t linear=((size_t)sample*104729+17)%output.size();
    int i=linear%I;linear/=I;int b=linear%B;int t=linear/B;float expected=reference(t,b,i,T,B,I,O,K,grad.data(),filter.data());
    float error=fabsf(output[((size_t)t*B+b)*I+i]-expected);if(error>max_error)max_error=error;errors+=error>1e-4f;}
  std::printf("T=%d B=%d I=%d O=%d K=%d warm_us=%.6f sampled=8192 errors=%d max_error=%g\n",
      T,B,I,O,K,warm_us,errors,max_error);return errors!=0;}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void polygeist_cudnn_conv_transpose3d_f32(
    int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,
    const float*,const float*,float*);
extern "C" void polygeist_cudnn_conv_backward_filter3d_f32(
    int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,
    int32_t,int32_t,int32_t,const float*,const float*,float*);

int main(){constexpr int IC=8,OC=12,D=32,H=32,W=32,KD=3,KH=3,KW=3;
  constexpr int OD=D+KD-1,OH=H+KH-1,OW=W+KW-1;
  std::vector<float> small((size_t)IC*D*H*W),filter((size_t)IC*OC*KD*KH*KW),
      transpose((size_t)OC*OD*OH*OW),transpose_ref(transpose.size(),0);
  for(size_t i=0;i<small.size();++i)small[i]=(float)((i*17+3)%3-1);
  for(size_t i=0;i<filter.size();++i)filter[i]=(float)((i*13+5)%3-1);
  for(int ic=0;ic<IC;++ic)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int x=0;x<W;++x)
    for(int oc=0;oc<OC;++oc)for(int kz=0;kz<KD;++kz)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)
      transpose_ref[(((size_t)oc*OD+z+kz)*OH+y+ky)*OW+x+kx]+=
        small[((size_t)ic*D+z)*H*W+(size_t)y*W+x]*
        filter[(((size_t)ic*OC+oc)*KD+kz)*KH*KW+(size_t)ky*KW+kx];
  for(int w=0;w<3;++w)polygeist_cudnn_conv_transpose3d_f32(IC,OC,D,H,W,KD,KH,KW,small.data(),filter.data(),transpose.data());
  auto t0=std::chrono::steady_clock::now();for(int t=0;t<5;++t)
    polygeist_cudnn_conv_transpose3d_f32(IC,OC,D,H,W,KD,KH,KW,small.data(),filter.data(),transpose.data());
  auto t1=std::chrono::steady_clock::now();double transpose_us=std::chrono::duration<double,std::micro>(t1-t0).count()/5;
  int transpose_errors=0;float transpose_max=0;for(size_t i=0;i<transpose.size();++i){float e=fabsf(transpose[i]-transpose_ref[i]);
    transpose_max=std::max(transpose_max,e);transpose_errors+=e>1e-4f;}

  std::vector<float> large((size_t)IC*OD*OH*OW),grad((size_t)OC*D*H*W),
      grad_filter((size_t)OC*IC*KD*KH*KW),grad_filter_ref(grad_filter.size());
  for(size_t i=0;i<large.size();++i)large[i]=(float)((i*19+7)%3-1);
  for(size_t i=0;i<grad.size();++i)grad[i]=(float)((i*23+11)%3-1);
  for(int oc=0;oc<OC;++oc)for(int ic=0;ic<IC;++ic)for(int kz=0;kz<KD;++kz)
    for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx){float acc=0;
      for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int x=0;x<W;++x)
        acc+=large[((size_t)ic*OD+z+kz)*OH*OW+(size_t)(y+ky)*OW+x+kx]*
             grad[((size_t)oc*D+z)*H*W+(size_t)y*W+x];
      grad_filter_ref[(((size_t)oc*IC+ic)*KD+kz)*KH*KW+(size_t)ky*KW+kx]=acc;}
  for(int w=0;w<3;++w)polygeist_cudnn_conv_backward_filter3d_f32(
      IC,OC,OD,OH,OW,D,H,W,KD,KH,KW,large.data(),grad.data(),grad_filter.data());
  t0=std::chrono::steady_clock::now();for(int t=0;t<5;++t)
    polygeist_cudnn_conv_backward_filter3d_f32(IC,OC,OD,OH,OW,D,H,W,KD,KH,KW,
        large.data(),grad.data(),grad_filter.data());
  t1=std::chrono::steady_clock::now();double filter_us=std::chrono::duration<double,std::micro>(t1-t0).count()/5;
  int filter_errors=0;float filter_max=0;for(size_t i=0;i<grad_filter.size();++i){float e=fabsf(grad_filter[i]-grad_filter_ref[i]);
    filter_max=std::max(filter_max,e);filter_errors+=e>1e-4f;}
  std::printf("IC=%d OC=%d D=%d H=%d W=%d K=%d transpose_us=%.6f backward_filter_us=%.6f transpose_errors=%d transpose_max=%g filter_errors=%d filter_max=%g\n",
      IC,OC,D,H,W,KD,transpose_us,filter_us,transpose_errors,transpose_max,filter_errors,filter_max);
  return transpose_errors||filter_errors;}

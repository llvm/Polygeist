#define N 256
void aten_grid_sampler_2d_quantized_cpu(unsigned char input[N],float scale,int zero,unsigned char out[N]){for(int i=0;i<N;++i){float v=((int)input[i]-zero)*scale;int q=(int)(v/scale)+zero;if(q<0)q=0;if(q>255)q=255;out[i]=(unsigned char)q;}}

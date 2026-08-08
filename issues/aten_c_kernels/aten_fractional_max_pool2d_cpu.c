#define B 2
#define C 3
#define IH 9
#define IW 10
#define OH 4
#define OW 5
#define KH 3
#define KW 3
void aten_fractional_max_pool2d_cpu(float x[B][C][IH][IW],float sample[B][C][2],float out[B][C][OH][OW],int index[B][C][OH][OW]){for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox){int sy=(int)((oy+sample[b][c][0])*(IH-KH)/(float)(OH-1));int sx=(int)((ox+sample[b][c][1])*(IW-KW)/(float)(OW-1));if(sy>IH-KH)sy=IH-KH;if(sx>IW-KW)sx=IW-KW;float v=-3.402823466e38f;int best=0;for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)if(x[b][c][sy+ky][sx+kx]>v){v=x[b][c][sy+ky][sx+kx];best=(sy+ky)*IW+sx+kx;}out[b][c][oy][ox]=v;index[b][c][oy][ox]=best;}}

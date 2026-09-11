#define B 1
#define C 2
#define ID 8
#define IH 9
#define IW 10
#define OD 3
#define OH 4
#define OW 5
#define KD 2
#define KH 3
#define KW 3
void aten_fractional_max_pool3d_cpu(float x[B][C][ID][IH][IW],float sample[B][C][3],float out[B][C][OD][OH][OW],int index[B][C][OD][OH][OW]){for(int b=0;b<B;++b)for(int c=0;c<C;++c)for(int oz=0;oz<OD;++oz)for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox){int sz=(int)((oz+sample[b][c][0])*(ID-KD)/(float)(OD-1));int sy=(int)((oy+sample[b][c][1])*(IH-KH)/(float)(OH-1));int sx=(int)((ox+sample[b][c][2])*(IW-KW)/(float)(OW-1));if(sz>ID-KD)sz=ID-KD;if(sy>IH-KH)sy=IH-KH;if(sx>IW-KW)sx=IW-KW;float v=-3.402823466e38f;int best=0;for(int kz=0;kz<KD;++kz)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)if(x[b][c][sz+kz][sy+ky][sx+kx]>v){v=x[b][c][sz+kz][sy+ky][sx+kx];best=((sz+kz)*IH+sy+ky)*IW+sx+kx;}out[b][c][oz][oy][ox]=v;index[b][c][oz][oy][ox]=best;}}

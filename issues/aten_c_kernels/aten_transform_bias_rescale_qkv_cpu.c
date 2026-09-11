#define B 2
#define S 16
#define H 4
#define D 8
void aten_transform_bias_rescale_qkv_cpu(float qkv[B][S][3][H][D],float bias[3][H][D],float scale,float q[B][H][S][D],float k[B][H][S][D],float v[B][H][S][D]){for(int b=0;b<B;++b)for(int s=0;s<S;++s)for(int h=0;h<H;++h)for(int d=0;d<D;++d){q[b][h][s][d]=(qkv[b][s][0][h][d]+bias[0][h][d])*scale;k[b][h][s][d]=qkv[b][s][1][h][d]+bias[1][h][d];v[b][h][s][d]=qkv[b][s][2][h][d]+bias[2][h][d];}}

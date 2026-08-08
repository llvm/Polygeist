#define N 8
#define C 16
#define H 16
#define W 16
void aten_batch_norm_transform_cpu(float x[N][C][H][W],float mean[C],float invstd[C],float weight[C],float bias[C],float out[N][C][H][W]){for(int n=0;n<N;++n)for(int c=0;c<C;++c)for(int y=0;y<H;++y)for(int z=0;z<W;++z)out[n][c][y][z]=(x[n][c][y][z]-mean[c])*invstd[c]*weight[c]+bias[c];}

#define N 8
#define C 16
#define H 16
#define W 16
void aten_batch_norm_backward_template_cpu(float grad[N][C][H][W],float x[N][C][H][W],float mean[C],float invstd[C],float out[N][C][H][W]){for(int c=0;c<C;++c){float sg=0,sgx=0;for(int n=0;n<N;++n)for(int y=0;y<H;++y)for(int z=0;z<W;++z){sg+=grad[n][c][y][z];sgx+=grad[n][c][y][z]*(x[n][c][y][z]-mean[c]);}for(int n=0;n<N;++n)for(int y=0;y<H;++y)for(int z=0;z<W;++z)out[n][c][y][z]=invstd[c]*(grad[n][c][y][z]-sg/(N*H*W)-(x[n][c][y][z]-mean[c])*invstd[c]*invstd[c]*sgx/(N*H*W));}}

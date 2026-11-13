#define N 8
#define C 16
#define H 16
#define W 16
void aten_batch_norm_stats_cpu(float x[N][C][H][W],float mean[C],float var[C]){for(int c=0;c<C;++c){float m=0;for(int n=0;n<N;++n)for(int y=0;y<H;++y)for(int z=0;z<W;++z)m+=x[n][c][y][z];m/=N*H*W;float v=0;for(int n=0;n<N;++n)for(int y=0;y<H;++y)for(int z=0;z<W;++z){float d=x[n][c][y][z]-m;v+=d*d;}mean[c]=m;var[c]=v/(N*H*W);}}

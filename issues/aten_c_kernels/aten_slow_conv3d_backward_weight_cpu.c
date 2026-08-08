#define C 2
#define O 3
#define D 6
#define H 7
#define W 8
#define K 3
void aten_slow_conv3d_backward_weight_cpu(float x[C][D+2][H+2][W+2],float g[O][D][H][W],float out[O][C][K][K][K]){for(int p=0;p<O*C*K*K*K;++p)((float*)out)[p]=0;for(int o=0;o<O;++o)for(int c=0;c<C;++c)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)out[o][c][kz][ky][kx]+=x[c][z+kz][y+ky][q+kx]*g[o][z][y][q];}

#define C 2
#define O 3
#define D 6
#define H 7
#define W 8
#define K 3
void aten_slow_conv3d_backward_input_cpu(float g[O][D][H][W],float w[O][C][K][K][K],float out[C][D+2][H+2][W+2]){for(int p=0;p<C*(D+2)*(H+2)*(W+2);++p)((float*)out)[p]=0;for(int o=0;o<O;++o)for(int c=0;c<C;++c)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)out[c][z+kz][y+ky][x+kx]+=g[o][z][y][x]*w[o][c][kz][ky][kx];}

#define C 2
#define O 3
#define D 6
#define H 7
#define W 8
#define K 3
void aten_conv_transpose3d_cpu(float x[C][D][H][W],float w[C][O][K][K][K],float out[O][D+2][H+2][W+2]){for(int p=0;p<O*(D+2)*(H+2)*(W+2);++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int o=0;o<O;++o)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)out[o][z+kz][y+ky][q+kx]+=x[c][z][y][q]*w[c][o][kz][ky][kx];}

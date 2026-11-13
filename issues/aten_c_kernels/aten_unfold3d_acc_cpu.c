#define C 2
#define D 6
#define H 7
#define W 8
#define K 3
void aten_unfold3d_acc_cpu(float x[C][K][K][K][D][H][W],float out[C][D+2][H+2][W+2]){for(int p=0;p<C*(D+2)*(H+2)*(W+2);++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)out[c][z+kz][y+ky][q+kx]+=x[c][kz][ky][kx][z][y][q];}

#define C 2
#define D 8
#define H 9
#define W 10
#define K 3
void aten_unfold3d_zero_copy_cpu(float x[C][D][H][W],float out[C][K][K][K][D][H][W]){for(int c=0;c<C;++c)for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int q=0;q<W;++q)for(int kz=0;kz<K;++kz)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx){int iz=z+kz-1,iy=y+ky-1,ix=q+kx-1;out[c][kz][ky][kx][z][y][q]=(iz>=0&&iz<D&&iy>=0&&iy<H&&ix>=0&&ix<W)?x[c][iz][iy][ix]:0;}}

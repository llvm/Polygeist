#define C 3
#define H 16
#define W 16
#define K 3
void aten_conv2d_columns_cpu(float x[C][H][W],float col[C][K][K][H-2][W-2]){for(int c=0;c<C;++c)for(int ky=0;ky<K;++ky)for(int kx=0;kx<K;++kx)for(int y=0;y<H-2;++y)for(int z=0;z<W-2;++z)col[c][ky][kx][y][z]=x[c][y+ky][z+kx];}
